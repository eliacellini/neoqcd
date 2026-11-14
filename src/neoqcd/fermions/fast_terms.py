"""Opt-in fast path for staggered fermions and for Hasenbusch mass preconditioning (Wilson).

Built on `fast.py` (site tables, `colour_mul`, `lean_cg`, `mixed_precision_cg`,
`HalfLatticeWilson`, `FastTwoFlavourWilson`) by import and subclassing; the reference classes
(`staggered.StaggeredEvenOdd`, `staggered_pseudofermion.StaggeredPseudofermion`,
`hasenbusch.HeavyWilsonTerm`, `hasenbusch.RatioWilsonTerm`, `hasenbusch.HasenbuschHamiltonian`)
stay the default and the ground truth; test/test_fermions_fast_terms.py checks every class here
against them. Conventions are those of `staggered.py` and `hasenbusch.py`.

Staggered half-lattice operator
-------------------------------
A staggered field of one parity is stored as (B, V/2, N): the sites of that parity in the
lexicographic order of `fast.site_tables` (the order of `StaggeredEvenOdd` and of
`staggered_dense_matrix`), then colour. `to_half_staggered` / `to_full_staggered` convert from /
to (B, lattice, N).

The hop K from parity 1 - t to parity t is, at a target site x,
    (K chi)(x) = sum_{k < D}  c_k(x) L_mu(x) chi(x + mu) + sum_{k >= D} c_k(x) L_mu(x - mu)^dagger chi(x - mu),
    c_k = +eta_mu(x) / 2 (forward, mu = k), -eta_mu(x) / 2 (backward, mu = k - D)
(eta_mu(x - mu) = eta_mu(x)). The 2D link matrices c_k L (boundary phase, eta and the 1/2
absorbed, daggers materialised) are gathered once per gauge configuration and target parity,
stored transposed as (B, V/2, 2D, N, N) contiguous (memory 2 x the gauge field), so one hop is
    1. ONE index_select of the source field on the site axis with the precomputed (V/2 * 2D,)
       neighbour table -> (B, V/2, 2D, 1, N);
    2. ONE colour product (B, V/2, 2D, 1, N) x (B, V/2, 2D, N, N) in the mode of
       `fast.colour_mul` ("matmul", "broadcast", "broadcast_sum"; "auto" = broadcast_sum on CUDA,
       matmul on the CPU);
    3. a sum over the 2D terms.
M_ee x = m^2 x - K_eo K_oe x costs two half-lattice hops (the reference: two full-lattice hops
on masked fields). The operator works in the dtype of its links (complex128, or complex64 via
`dtype=` / `to(dtype)`); the mass is a float64 (B,) tensor cast to the matching real dtype; a
field of another dtype is rejected (TypeError), so nothing is promoted silently.

Staggered force from half-lattice fields
----------------------------------------
`staggered_hopping_derivative(op, a, b)` (the reference) is P_ht(i Q) per link with
Q = (eta/2) (L O1 + O2 L^dagger), O1 = b(x + mu) a(x)^dagger, O2 = b(x) a(x + mu)^dagger. The
colour products can be done on vectors first:
    (eta/2) L O1        = u_b(x) a(x)^dagger,       u_b(x) = (eta/2) L_mu(x) b(x + mu),
    (eta/2) O2 L^dagger = b(x) u_a(x)^dagger,       u_a(x) = (eta/2) L_mu(x) a(x + mu),
and u_b, u_a are exactly the forward terms k < D of the hop to the parity of x. For a link at x
of parity t, a(x), b(x) are the parity-t halves and b(x + mu), a(x + mu) the parity-(1 - t) ones:
one gather of the stacked (b_{1-t}, a_{1-t}) and one colour product with the forward links per
parity, then two outer products (N^2, no N^3 product). For the pseudofermion force
F = 2 G(X, W), X = M_ee^-1 phi (even), W = K_oe X (odd), only u_b on even links and u_a on odd
links are nonzero (`hopping_derivative` skips missing halves given as None).

Hasenbusch on the half lattice
------------------------------
`FastRatioWilsonTerm` keeps phi_2 and X as half-lattice fields and uses two `HalfLatticeWilson`
operators, light (kappa) and heavy (kappa'). The links do not depend on kappa: the heavy operator
is a shallow copy of the light one with its kappa coefficients reset (`HalfLatticeWilson._set_dtype`
recomputes them and keeps the gathered links and tables: no second gather, no extra memory). The
formulas are those of `hasenbusch.py` with the reference full-lattice pieces replaced by half fields:
    chi = D_hat'^dagger phi_2,      X = M^-1 chi (lean_cg on light.normal),
    Y = D_hat X,   c = kappa^2 Y - kappa'^2 phi_2,   Z c = (H^dagger)_oe c,   W X = H_oe X,
    F = 2 G(c + Z c, X + W X)       (one `HalfLatticeWilson.hopping_derivative` call),
refresh phi_2 = D_hat' (D_hat'^dagger D_hat')^-1 D_hat^dagger eta with one heavy solve at
`action_tol` and the reference refresh guard (RuntimeError above 10 max(action_tol, 1e3 eps), for
any `on_unconverged` policy, as the reference).
"""

import copy

import torch

from neoqcd.fermions.algebra import project_hermitian_traceless
from neoqcd.fermions.dirac import _kappa_tensor
from neoqcd.fermions.fast import (
    COLOUR_PRODUCTS,
    FastTwoFlavourWilson,
    colour_mul,
    lean_cg,
    mixed_precision_cg,
    resolve_colour_product,
    site_tables,
    to_half,
)
from neoqcd.fermions.hasenbusch import HasenbuschHamiltonian
from neoqcd.fermions.hmc import Hamiltonian, WilsonGaugeAction
from neoqcd.fermions.lattice import Geometry, fermion_boundary_links
from neoqcd.fermions.pseudofermion import ON_UNCONVERGED, TwoFlavourWilson, convergence_limit
from neoqcd.fermions.staggered import _mass_tensor, staggered_phases
from neoqcd.fermions.staggered_pseudofermion import StaggeredPseudofermion

__all__ = [
    "to_half_staggered",
    "to_full_staggered",
    "HalfLatticeStaggered",
    "FastStaggeredPseudofermion",
    "FastStaggeredHamiltonian",
    "FastHeavyWilsonTerm",
    "FastRatioWilsonTerm",
    "FastHasenbuschHamiltonian",
]

_REAL = {torch.complex128: torch.float64, torch.complex64: torch.float32}


# ---- staggered half-lattice fields --------------------------------------------------------------
def to_half_staggered(chi, geometry, parity=0):
    """Full-lattice staggered field (B, lattice, N) -> its sites of `parity`, (B, V/2, N)."""
    tab = site_tables(geometry, chi.device)
    return chi.reshape(chi.shape[0], geometry.volume, geometry.N).index_select(1, tab["sites"][parity])


def to_full_staggered(phi, geometry, parity=0, other=None):
    """Half-lattice staggered field (B, V/2, N) of `parity` -> (B, lattice, N); the other parity is
    zero, or `other` (a half field of the other parity) if given."""
    tab = site_tables(geometry, phi.device)
    B = phi.shape[0]
    out = phi.new_zeros((B, geometry.volume, geometry.N))
    out.index_copy_(1, tab["sites"][parity], phi)
    if other is not None:
        out.index_copy_(1, tab["sites"][1 - parity], other)
    return out.reshape(geometry.staggered_shape(B))


_STAG_CACHE = {}


def _staggered_tables(geometry, device):
    """Neighbour tables of the staggered hop (cached): per target parity t,
    "gather" [t] (V/2 * 2D,): position (in the parity 1 - t half) of x_i + mu (k < D) or x_i - mu
    (k >= D), entry i * 2D + k; "forward" [t] (V/2 * D,): x_i + mu, entry i * D + mu."""
    device = torch.device(device)
    key = (geometry, str(device))
    if key in _STAG_CACHE:
        return _STAG_CACHE[key]
    tab = site_tables(geometry, device)
    D = geometry.D
    V2 = geometry.volume // 2
    gather, forward = [], []
    for t in (0, 1):
        pos = tab["hop_index"][t].view(V2, 2 * D) // (2 * D)
        gather.append(pos.reshape(-1).contiguous())
        forward.append(pos[:, :D].reshape(-1).contiguous())
    out = {"gather": gather, "forward": forward}
    _STAG_CACHE[key] = out
    return out


class HalfLatticeStaggered:
    """Staggered even-site normal operator M_ee = m^2 - K_eo K_oe on half-lattice fields (B, V/2, N).

    Args:
        U: links (B, D, lattice, N, N), complex128 or complex64.
        mass: float or (B,) tensor.
        geometry: `Geometry` (inferred from U if omitted); even extents.
        time_sign: -1 antiperiodic (default) or +1 periodic in time.
        dtype: complex dtype of the operator (default U.dtype); links are gathered in U.dtype and
            then cast (`to(dtype)`; an upcast re-gathers from U).
        colour_product: "auto" (default), "matmul", "broadcast" or "broadcast_sum" (`fast.colour_mul`).

    `apply` = M_ee (hermitian positive definite for m > 0: CG runs on it), `apply_dagger` =
    `apply`, `normal` = M_ee^2, `hop(chi, t)` = K from parity 1 - t to t, `prepare_rhs_half`,
    `reconstruct_odd`, `hopping_derivative` (module docstring). Fields with batch B' = B, or any B'
    if the links have B = 1.
    """

    def __init__(self, U, mass, geometry=None, time_sign=-1.0, dtype=None, colour_product="auto"):
        if geometry is None:
            D = U.shape[1]
            geometry = Geometry(D, U.shape[2], U.shape[3] if D > 1 else 1, U.shape[-1])
        if tuple(U.shape[1:]) != geometry.link_shape(1)[1:]:
            raise ValueError(f"links of shape {tuple(U.shape)} do not match geometry {geometry}")
        dtype = U.dtype if dtype is None else dtype
        if dtype not in _REAL:
            raise TypeError(f"HalfLatticeStaggered supports complex128 and complex64, got {dtype}")
        self.colour_product = resolve_colour_product(colour_product, U.device)
        self.geometry = geometry
        self.U = U
        self.time_sign = float(time_sign)
        self.mass = _mass_tensor(mass, U.shape[0], U.device)  # float64, (B,) or (1,)
        self._tables = site_tables(geometry, U.device)
        self._stag = _staggered_tables(geometry, U.device)
        self._links = self._gather_links(U)
        self._set_dtype(dtype)

    def _gather_links(self, U):
        """Per target parity t: (B, V/2, 2D, N, N) contiguous; term k < D is (eta/2 L_mu(x))^T,
        term k >= D is (-eta/2 L_mu(x - mu)^dagger)^T = -eta/2 conj(L_mu(x - mu))."""
        geo = self.geometry
        D, V, N = geo.D, geo.volume, geo.N
        B = U.shape[0]
        flat = fermion_boundary_links(U, self.time_sign).reshape(B, D, V, N, N)
        eta = staggered_phases(geo, device=U.device, dtype=U.real.dtype).reshape(D, V)
        out = []
        for t in (0, 1):
            T = self._tables["sites"][t]
            c = 0.5 * eta[:, T]  # (D, V/2)
            fwd = [flat[:, mu].index_select(1, T).transpose(-1, -2) * c[mu].view(1, -1, 1, 1) for mu in range(D)]
            bwd = [flat[:, mu].index_select(1, self._tables["bwd"][mu][T]).conj() * (-c[mu]).view(1, -1, 1, 1)
                   for mu in range(D)]
            out.append(torch.stack(fwd + bwd, dim=2).contiguous())
        return out

    def _set_dtype(self, dtype):
        if self._links[0].dtype == torch.complex64 and dtype == torch.complex128 and self.U.dtype == torch.complex128:
            self._links = self._gather_links(self.U)  # upcast: never from the rounded copy
        self._dtype = dtype
        self._links = [l.to(dtype).contiguous() for l in self._links]
        self._fwd_links = [l[:, :, :self.geometry.D] for l in self._links]
        real = _REAL[dtype]
        self._m = self.mass.to(real).view(-1, 1, 1)
        self._m2 = (self.mass ** 2).to(real).view(-1, 1, 1)

    def to(self, dtype):
        """The same operator in another complex dtype (tables shared, links cast)."""
        if dtype == self._dtype:
            return self
        new = copy.copy(self)
        new._set_dtype(dtype)
        return new

    # ---- protocol attributes -----------------------------------------------------------------
    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self.U.device

    @property
    def batch_size(self):
        return self.U.shape[0]

    @property
    def field_shape(self):
        return (self.batch_size, self.geometry.volume // 2, self.geometry.N)

    def _check(self, chi):
        if chi.dtype != self._dtype:
            raise TypeError(f"field dtype {chi.dtype} differs from the operator dtype {self._dtype} (use op.to(dtype))")

    # ---- kernels -----------------------------------------------------------------------------
    def hop(self, chi, target_parity):
        """K from the parity 1 - t field chi (B, V/2, N) to parity t: K_eo (t = 0) or K_oe (t = 1)."""
        self._check(chi)
        D, N = self.geometry.D, self.geometry.N
        B, V2 = chi.shape[:2]
        h = chi.index_select(1, self._stag["gather"][target_parity]).view(B, V2, 2 * D, 1, N)
        return colour_mul(h, self._links[target_parity], self.colour_product).sum(dim=2).view(B, V2, N)

    def hop_eo(self, chi_odd):
        return self.hop(chi_odd, 0)

    def hop_oe(self, chi_even):
        return self.hop(chi_even, 1)

    # ---- operator ----------------------------------------------------------------------------
    def apply(self, x):
        """M_ee x = m^2 x - K_eo K_oe x."""
        y = self.hop(self.hop(x, 1), 0)
        return torch.addcmul(y.neg_(), self._m2, x)

    def apply_dagger(self, x):
        return self.apply(x)

    def normal(self, x):
        """M_ee^2 x (protocol completeness; CG runs on `apply`)."""
        return self.apply(self.apply(x))

    # ---- full-lattice helpers ----------------------------------------------------------------
    def to_half(self, chi, parity=0):
        return to_half_staggered(chi, self.geometry, parity)

    def to_full(self, x_even, x_odd=None):
        return to_full_staggered(x_even, self.geometry, 0, other=x_odd)

    def prepare_rhs_half(self, b_e, b_o):
        """m b_e - K_eo b_o = (D^dagger b)_e."""
        return torch.addcmul(self.hop(b_o, 0).neg_(), self._m, b_e)

    def prepare_rhs(self, b):
        return self.prepare_rhs_half(self.to_half(b, 0), self.to_half(b, 1))

    def reconstruct_odd(self, x_e, b_o):
        """x_o = (b_o - K_oe x_e) / m."""
        return (b_o - self.hop(x_e, 1)) / self._m

    def reconstruct(self, x_even, b):
        return self.to_full(x_even, self.reconstruct_odd(x_even, self.to_half(b, 1)))

    # ---- force -------------------------------------------------------------------------------
    def hopping_derivative(self, a_e, a_o, b_e, b_o):
        """Algebra gradient of Re[a^dagger K b] (as `staggered_pseudofermion.staggered_hopping_derivative`)
        from half fields a = (a_e, a_o), b = (b_e, b_o); a missing half is None (zero). Shape
        (B, D, lattice, N, N). Formula: module docstring."""
        geo = self.geometry
        D, N, V = geo.D, geo.N, geo.volume
        a, b = (a_e, a_o), (b_e, b_o)
        ref = next(f for f in (a_e, a_o, b_e, b_o) if f is not None)
        B, V2 = ref.shape[0], V // 2
        out = ref.new_zeros((B, V, D, N, N))
        mode = self.colour_product
        for t in (0, 1):
            n = 1 - t
            first = b[n] is not None and a[t] is not None  # u_b(x) a(x)^dagger
            second = b[t] is not None and a[n] is not None  # b(x) u_a(x)^dagger
            if not (first or second):
                continue
            srcs = ([b[n]] if first else []) + ([a[n]] if second else [])
            src = torch.stack(srcs, dim=2)  # (B, V/2, s, N)
            S = src.shape[2]
            nb = src.index_select(1, self._stag["forward"][t]).view(B, V2, D, S, N)
            u = colour_mul(nb, self._fwd_links[t], mode)  # rows: (eta/2 L nb)^T
            Q = None
            if first:
                Q = u[:, :, :, 0, :, None] * a[t].conj()[:, :, None, None, :]
            if second:
                q = b[t][:, :, None, :, None] * u[:, :, :, S - 1, None, :].conj()
                Q = q if Q is None else Q + q
            out.index_copy_(1, self._tables["sites"][t], project_hermitian_traceless(1j * Q))
        return out.permute(0, 2, 1, 3, 4).reshape((B, D) + geo.lattice + (N, N))


# ---- shared helpers of the fast terms ----------------------------------------------------------
def _value_key(v):
    """A hashable-comparable snapshot of a float or tensor parameter (by value)."""
    if torch.is_tensor(v):
        return ("tensor", tuple(v.shape), tuple(v.detach().to("cpu", torch.float64).reshape(-1).tolist()))
    return ("float", float(v))


def _run_cg(term, apply, low_apply, rhs, x0, tol):
    """lean_cg (precision "double") or mixed_precision_cg ("mixed") with the stats of the fast terms."""
    info = {}
    if term.precision == "double":
        res = lean_cg(apply, rhs, x0=x0, tol=tol, max_iter=term.max_iter, check_every=term.check_every, info=info)
    else:
        res = mixed_precision_cg(apply, low_apply(), rhs, x0=x0, tol=tol, inner_tol=term.inner_tol,
                                 max_outer=term.max_outer, max_iter=term.max_iter, check_every=term.check_every,
                                 info=info)
        term.stats["outer_iterations"] += info["outer"]
        term.stats["double_applications"] += info["double_applications"]
    term.stats["cg_loop_iterations"] += info["loop_iterations"]
    return res


# ---- staggered pseudofermion -------------------------------------------------------------------
class FastStaggeredPseudofermion(StaggeredPseudofermion):
    """`StaggeredPseudofermion` through `HalfLatticeStaggered` and `lean_cg` / `mixed_precision_cg`.

    Same heat bath (the same full-lattice draw of chi as the reference, phi = m chi_e - K_eo chi_o),
    action, `action_difference` (S_new solved first, S at the refresh links solved once and cached
    under the identity + `_version` guard) and force F = 2 G(X, K_oe X) as the reference, to the
    solver tolerance. Internally phi and X are half-lattice fields (`_phi_h`, `_X_h`); `phi` and
    `X` read/write full-lattice tensors (odd sites zero).

    Extra args (as `fast.FastTwoFlavourWilson`): precision ("double" | "mixed"), check_every,
    inner_tol, max_outer, colour_product, on_unconverged ("raise" default | "warn" | "ignore": the
    convergence policy of `pseudofermion`, on the true residual of every solve; the reference
    staggered term has none). Extra stats: "cg_loop_iterations", "outer_iterations",
    "double_applications", "unconverged_solves".

    The operator (and its complex64 copy) is cached per links tensor (identity and `_version`) and
    per value of mass, time_sign, colour_product and geometry. Writes that do not bump
    `U._version` (U.data, numpy views) are not detected.
    """

    def __init__(self, mass, geometry, time_sign=-1.0, solver_tol=1e-10, action_tol=None, max_iter=2000,
                 reuse_guess=True, precision="double", check_every=10, inner_tol=1e-6, max_outer=30,
                 colour_product="auto", on_unconverged="raise"):
        if precision not in ("double", "mixed"):
            raise ValueError(f"precision must be 'double' or 'mixed', got {precision!r}")
        if colour_product not in COLOUR_PRODUCTS:
            raise ValueError(f"colour_product must be one of {COLOUR_PRODUCTS}, got {colour_product!r}")
        if on_unconverged not in ON_UNCONVERGED:
            raise ValueError(f"on_unconverged must be one of {ON_UNCONVERGED}, got {on_unconverged!r}")
        self._phi_h = None
        self._X_h = None
        self._op_cache = None
        self._warned = set()
        self._solve_kind = None
        self.precision = precision
        self.check_every = int(check_every)
        self.inner_tol = float(inner_tol)
        self.max_outer = int(max_outer)
        self.colour_product = colour_product
        self.on_unconverged = on_unconverged
        super().__init__(mass, geometry, time_sign=time_sign, solver_tol=solver_tol, action_tol=action_tol,
                         max_iter=max_iter, reuse_guess=reuse_guess)

    def reset_stats(self):
        super().reset_stats()
        self.stats.update(cg_loop_iterations=0, outer_iterations=0, double_applications=0, unconverged_solves=0)

    # convergence policy of the Wilson terms (pseudofermion module doc), reused as plain functions
    _check_converged = TwoFlavourWilson._check_converged
    _solve_kind_of = TwoFlavourWilson._solve_kind_of
    _solve_as = TwoFlavourWilson._solve_as

    def _count(self, result, tol=None, kind=None):
        super()._count(result)
        self._check_converged(result, tol, kind)

    # ---- full-lattice views ------------------------------------------------------------------
    @property
    def phi(self):
        return None if self._phi_h is None else to_full_staggered(self._phi_h, self.geometry)

    @phi.setter
    def phi(self, value):
        self._phi_h = None if value is None else to_half_staggered(value, self.geometry)

    @property
    def X(self):
        return None if self._X_h is None else to_full_staggered(self._X_h, self.geometry)

    @X.setter
    def X(self, value):
        self._X_h = None if value is None else to_half_staggered(value, self.geometry)

    # ---- operators ---------------------------------------------------------------------------
    def _operator_key(self):
        return (_value_key(self.mass), float(self.time_sign), self.colour_product, self.geometry)

    def operator(self, U):
        """HalfLatticeStaggered at links U, cached while U is the same tensor with the same
        `_version` and mass (by value), time_sign, colour_product and geometry are unchanged."""
        c = self._op_cache
        key = self._operator_key()
        if c is not None and c[0] is U and c[1] == U._version and c[2] == key:
            return c[3]
        op = HalfLatticeStaggered(U, self.mass, self.geometry, self.time_sign, colour_product=self.colour_product)
        op._low = None
        self._op_cache = (U, U._version, key, op)
        return op

    @staticmethod
    def _low_apply(op):
        def get():
            if op._low is None:
                op._low = op.to(torch.complex64)
            return op._low.apply
        return get

    def _solve(self, op, tol):
        if self._phi_h is None:
            raise RuntimeError("the pseudofermion field is not set: call refresh(U, generator) first")
        x0 = self._X_h if (self.reuse_guess and self._X_h is not None) else None
        res = _run_cg(self, op.apply, self._low_apply(op), self._phi_h, x0, tol)
        self._count(res, tol, self._solve_kind_of(tol))
        self._X_h = res.x
        return res.x

    # ---- ActionTerm interface ----------------------------------------------------------------
    def refresh(self, U, generator=None):
        """phi = m chi_e - K_eo chi_o with the full-lattice draw of chi of the reference."""
        op = self.operator(U)
        chi = torch.randn(self.geometry.staggered_shape(U.shape[0]), generator=generator,
                          dtype=U.dtype, device=U.device)
        self._phi_h = op.prepare_rhs(chi)
        self._X_h = None
        self._refresh_links = U
        self._refresh_version = U._version
        self._refresh_action = None
        self.stats["refresh_calls"] += 1

    def action(self, U):
        self.stats["action_calls"] += 1
        X = self._solve_as("action", self.operator(U), self.action_tol)
        return (self._phi_h.conj() * X).real.reshape(X.shape[0], -1).sum(1)

    def force(self, U):
        """F = 2 G(X, K_oe X) from half fields (module docstring)."""
        self.stats["force_calls"] += 1
        op = self.operator(U)
        X = self._solve_as("force", op, self.solver_tol)
        return self.force_from_solution(op, X)

    @staticmethod
    def force_from_solution(op, X):
        """The force from a half-lattice solution X = M_ee^-1 phi and its operator (no solve)."""
        return 2.0 * op.hopping_derivative(X, None, None, op.hop(X, 1))


class FastStaggeredHamiltonian(Hamiltonian):
    """`StaggeredHamiltonian` with `FastStaggeredPseudofermion` (fermion outside, gauge inside)."""

    def __init__(self, beta, mass, geometry, n_gauge_substeps=1, plaquette_weights=None, **fermion_kwargs):
        self.fermion = FastStaggeredPseudofermion(mass, geometry, **fermion_kwargs)
        self.gauge = WilsonGaugeAction(beta, geometry, plaquette_weights)
        super().__init__([([self.fermion], 1), ([self.gauge], n_gauge_substeps)])


# ---- Hasenbusch (Wilson) -----------------------------------------------------------------------
def _with_kappa(op, kappa):
    """A `HalfLatticeWilson` sharing the tables and gathered links of `op` at another kappa (the
    links do not depend on kappa; `_set_dtype` recomputes only the kappa coefficients)."""
    new = copy.copy(op)
    new.kappa = _kappa_tensor(kappa, op.U.shape[0], op.U.device)
    new._set_dtype(op.dtype)
    new._low = None
    return new


class FastHeavyWilsonTerm(FastTwoFlavourWilson):
    """`hasenbusch.HeavyWilsonTerm` on the fast path: `FastTwoFlavourWilson` at kappa_heavy.

    Inside `FastHasenbuschHamiltonian` its operator is the ratio term's heavy operator (the links
    gathered once per configuration for both terms) whenever kappa, time_sign, colour_product
    and geometry agree (checked at every call by value; otherwise it builds its own).
    """

    def __init__(self, kappa_heavy, geometry, **kwargs):
        kwargs.setdefault("even_odd", True)
        self._share_with = None
        super().__init__(kappa_heavy, geometry, **kwargs)

    def operator(self, U):
        src = self._share_with
        if (src is not None and _value_key(self.kappa) == _value_key(src.kappa_heavy)
                and float(self.time_sign) == float(src.time_sign) and self.colour_product == src.colour_product
                and self.geometry == src.geometry):
            return src.heavy_operator(U)
        return super().operator(U)


class FastRatioWilsonTerm(FastTwoFlavourWilson):
    """`hasenbusch.RatioWilsonTerm` on half-lattice fields (module docstring, Hasenbusch).

    Same arguments, heat bath (same Gaussian draws), action, force, refresh cache, refresh
    convergence guard and stats keys (including "refresh_solves", "refresh_cg_iterations") as
    the reference, plus the fast options of `FastTwoFlavourWilson` (precision, check_every,
    inner_tol, max_outer, colour_product, on_unconverged). `phi` and `X` are full-lattice views
    of the half fields `_phi_h` (phi_2) and `_X_h` (last solution of M X = chi). `_solve(op, tol)`
    takes the light operator and solves M X = chi with chi = D_hat'^dagger phi_2 (kept in `_chi_h`).
    """

    def __init__(self, kappa, kappa_heavy, geometry, **kwargs):
        if kwargs.pop("even_odd", True) is not True:
            raise ValueError("FastRatioWilsonTerm is implemented for the even-odd operator only")
        self.kappa_heavy = kappa_heavy
        self._chi_h = None
        super().__init__(kappa, geometry, even_odd=True, **kwargs)

    def reset_stats(self):
        super().reset_stats()
        self.stats["refresh_solves"] = 0
        self.stats["refresh_cg_iterations"] = 0

    # ---- operators ---------------------------------------------------------------------------
    def heavy_operator(self, U):
        """D_hat' at links U: the light operator's links and tables at kappa' (cached on it)."""
        light = self.operator(U)
        key = _value_key(self.kappa_heavy)
        cached = getattr(light, "_heavy", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        heavy = _with_kappa(light, self.kappa_heavy)
        light._heavy = (key, heavy)
        return heavy

    def _low_of(self, op):
        """complex64 copy of `op` (for the heavy one: derived from the light copy, no re-cast of links)."""
        def get():
            if op._low is None:
                light = self.operator(op.U)
                if op is not light:
                    op._low = _with_kappa(self._low_precision(light), op.kappa)
                else:
                    self._low_precision(op)
            return op._low.normal
        return get

    def _require_phi(self):
        if self._phi_h is None:
            raise RuntimeError("the pseudofermion field is not set: call refresh(U, generator) first")
        return self._phi_h

    def _solve(self, op, tol):
        """Light solve M X = chi, chi = D_hat'^dagger phi_2 (op: the light operator)."""
        phi = self._require_phi()
        self._chi_h = self.heavy_operator(op.U).apply_dagger(phi)
        x0 = self._X_h if (self.reuse_guess and self._X_h is not None) else None
        res = _run_cg(self, op.normal, self._low_of(op), self._chi_h, x0, tol)
        self._count(res, tol, self._solve_kind_of(tol))
        self._X_h = res.x
        return res.x

    # ---- ActionTerm interface ----------------------------------------------------------------
    def refresh(self, U, generator=None):
        """phi_2 = D_hat' (D_hat'^dagger D_hat')^-1 D_hat^dagger eta, one heavy solve at action_tol."""
        light = self.operator(U)
        heavy = self.heavy_operator(U)
        eta = torch.randn(self.geometry.wilson_shape(U.shape[0]), generator=generator, dtype=U.dtype, device=U.device)
        eta = to_half(eta, self.geometry)
        res = _run_cg(self, heavy.normal, self._low_of(heavy), light.apply_dagger(eta), None, self.action_tol)
        self._count(res)  # as the reference: book-keeping and the policy check at max(solver_tol, action_tol)
        self.stats["refresh_solves"] += 1
        self.stats["refresh_cg_iterations"] += int(res.iterations.max().item())
        limit = convergence_limit(self.action_tol, _REAL[U.dtype])
        worst = res.residual.max().item()
        if not worst <= limit:
            raise RuntimeError(f"FastRatioWilsonTerm.refresh: heavy CG not converged, relative residual {worst:.3e} "
                               f"> {limit:.1e} (action_tol {self.action_tol:.1e}, max_iter {self.max_iter})")
        self._phi_h = heavy.apply(res.x)
        self._X_h = None
        self._refresh_links = U
        self._refresh_version = U._version
        self._refresh_action = (eta.conj() * eta).real.reshape(eta.shape[0], -1).sum(1)
        self.stats["refresh_calls"] += 1

    def action(self, U):
        """S_2 = chi^dagger M^-1 chi, one light solve at action_tol."""
        self.stats["action_calls"] += 1
        X = self._solve_as("action", self.operator(U), self.action_tol)
        return (self._chi_h.conj() * X).real.reshape(X.shape[0], -1).sum(1)

    def force(self, U):
        """2 G(c + Z c, X + W X), c = kappa^2 Y - kappa'^2 phi_2 (module docstring)."""
        self.stats["force_calls"] += 1
        light = self.operator(U)
        heavy = self.heavy_operator(U)
        X = self._solve_as("force", light, self.solver_tol)
        W = light.hop(X, 1)  # H_oe X
        Y = light.hop(W, 0).mul_(-light._k2).add_(X)  # D_hat X
        c = Y.mul_(light._k2).sub_(heavy._k2 * self._phi_h)
        Zc = light.hop(c, 1, dagger=True)  # (H^dagger)_oe c
        return 2.0 * light.hopping_derivative(c, Zc, X, W)


_FAST_OPTIONS = ("precision", "check_every", "inner_tol", "max_outer", "colour_product")


class FastHasenbuschHamiltonian(HasenbuschHamiltonian):
    """`HasenbuschHamiltonian` with `FastRatioWilsonTerm` and `FastHeavyWilsonTerm`.

    Same arguments, layouts and argument validation (the reference constructor runs first and is
    then re-pointed to the fast terms with the same arguments); the fast options (precision,
    check_every, inner_tol, max_outer, colour_product) may be given in `fermion_kwargs` (both
    terms) and in `heavy_kwargs` (heavy term only). The heavy term reuses the ratio term's
    heavy operator (one link gather per configuration for both fermion terms).
    """

    def __init__(self, beta, kappa, kappa_heavy, geometry, n_ratio_substeps=None, n_heavy_substeps=None,
                 n_gauge_substeps=None, layout="nested", plaquette_weights=None, heavy_kwargs=None,
                 **fermion_kwargs):
        fast = {k: fermion_kwargs.pop(k) for k in _FAST_OPTIONS if k in fermion_kwargs}
        hk = dict(heavy_kwargs or {})
        fast_heavy = {k: hk.pop(k) for k in _FAST_OPTIONS if k in hk}
        super().__init__(beta, kappa, kappa_heavy, geometry, n_ratio_substeps=n_ratio_substeps,
                         n_heavy_substeps=n_heavy_substeps, n_gauge_substeps=n_gauge_substeps, layout=layout,
                         plaquette_weights=plaquette_weights, heavy_kwargs=hk, **fermion_kwargs)
        self.ratio = FastRatioWilsonTerm(kappa, kappa_heavy, geometry, **fermion_kwargs, **fast)
        heavy_args = dict(fermion_kwargs)
        heavy_args.update(fast)
        heavy_args.update(hk)
        heavy_args.update(fast_heavy)
        self.heavy = FastHeavyWilsonTerm(kappa_heavy, geometry, **heavy_args)
        self.heavy._share_with = self.ratio
        terms = {"ratio": self.ratio, "heavy": self.heavy, "gauge": self.gauge}
        Hamiltonian.__init__(self, [([terms[n] for n in names], s) for names, s in self.layout])
