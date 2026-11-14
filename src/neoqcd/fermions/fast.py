"""Opt-in fast path for N_f = 2 Wilson fermions: half-lattice even-odd operator, lean CG,
mixed-precision CG and a pseudofermion term built on them.

The reference classes (`dirac.WilsonDirac`, `dirac.EvenOddWilsonDirac`, `solver.cg`,
`pseudofermion.TwoFlavourWilson`) stay the default and the ground truth;
test/test_fermions_fast.py checks every kernel here against them. Conventions are those of
`dirac.py` and doc/fermions.md (shift(f, mu, +1)[n] = f[n - e_mu], D = 1 - kappa H, links with
the time boundary phase on the t = 0 time links, openQCD chiral gammas).

Half-lattice fields
-------------------
A field of one parity is stored as (B, V/2, S, N): sites of that parity in lexicographic order
of (t, x_1, ...) (the order of `EvenOddWilsonDirac` and `dense_matrix`), then spin, then colour.
`to_half(psi, geometry, parity)` and `to_full(phi, geometry, parity)` convert from / to the
full-lattice layout (B, lattice, S, N) (zero on the other parity).

Gamma matrices and half spinors
-------------------------------
Every gamma matrix of `gamma_matrices` (D = 4 openQCD chiral, D = 2 Pauli) is block
off-diagonal in (upper, lower) halves of size H = S/2,
    gamma_mu = [[0, g_mu], [g_mu^dagger, 0]],
    D = 4:  g_0 = -1,  g_1 = -i sigma_1,  g_2 = -i sigma_2,  g_3 = -i sigma_3   (2 x 2),
    D = 2:  g_0 = 1,   g_1 = -i                                                   (1 x 1),
and each g_mu is monomial (one entry +-1 or +-i per row). For a sign s = +-1 write
psi = (u, d) (upper, lower half). Then (1 + s gamma_mu) = R R^dagger with
R = [1; s g_mu^dagger] (S x H), R^dagger R = 2, and
    projection      h = R^dagger psi = u + s g_mu d                     (H components),
    reconstruction  (1 + s gamma_mu) psi = R h = (h, s g_mu^dagger h).
Written out for D = 4 (h_0, h_1; psi_0..psi_3):
    mu = 0:  h = (psi_0 - s psi_2,       psi_1 - s psi_3),        lower = -s h
    mu = 1:  h = (psi_0 - i s psi_3,     psi_1 - i s psi_2),      lower = (i s h_1, i s h_0)
    mu = 2:  h = (psi_0 - s psi_3,       psi_1 + s psi_2),        lower = (s h_1, -s h_0)
    mu = 3:  h = (psi_0 - i s psi_2,     psi_1 + i s psi_3),      lower = (i s h_0, -i s h_1)
The forward hop uses s = -1 (projector 1 - gamma_mu), the backward hop s = +1. The colour
product acts on h, so a hop multiplies H = S/2 spin components by the link instead of S.
H^dagger = gamma_5 H gamma_5 is the same hop with s -> -s (gamma_5 anticommutes with gamma_mu),
so D_hat^dagger needs no gamma_5 multiplications.

Layout and contraction (the choice for speed)
---------------------------------------------
For a target parity t the 2D hop terms k = 0..D-1 (forward along mu = k) and k = D..2D-1
(backward along mu = k - D) are stacked on an axis of size 2D placed after the site axis:
    1. project the source field (parity 1 - t) for all 2D terms at the source sites:
       (B, V/2, S, N) -> (B, V/2, 2D, H, N): one index_select on the spin axis (the monomial
       permutation) and one addcmul with the (2D, H, 1) coefficients s g_mu;
    2. gather the neighbours with one index_select on the flattened (site, term) axis with a
       precomputed table (projecting before the gather moves 2D H N instead of 2D S N numbers);
    3. one batched torch.matmul (B, V/2, 2D, H, N) @ (B, V/2, 2D, N, N) -> (B, V/2, 2D, H, N):
       per site and term a (H x N) @ (N x N) product, i.e. (L h^T)^T with the links stored
       transposed (L^T forward, conj(L) = (L^dagger)^T backward), so a single bmm does all 8
       (4D) or 4 (2D) link products of the hopping term;
    4. reconstruct: the lower halves s g_mu^dagger chi (index_select + mul on the (2D H) axis),
       concatenated with chi on the spin axis, summed over the term axis -> (B, V/2, S, N).
Colour product (`colour_product`): "matmul" is the torch.matmul above (one bmm); "broadcast"
writes the same (.., M, K) x (.., K, P) product elementwise, accumulating over the K = 3 (or
H) values of the contracted index with one mul and K - 1 in-place addcmul on strided slices
(no temporary larger than the output); "broadcast_sum" is the one-shot form of
`sun_utils.SUN_mul` on CUDA, (a[..., :, :, None] * b[..., None, :, :]).sum(-2), with a K x
temporary. "auto" is "broadcast_sum" on CUDA and "matmul" on the CPU. Measured on an A100
(torch 2.0, complex128) per application of the half-lattice normal operator: 8^4 batch 4,
reference 5.55 ms, matmul 2.51, broadcast 0.625, broadcast_sum 0.592; 16^4 batch 16, reference
336 ms, matmul 176, broadcast 55.0, broadcast_sum 53.7. broadcast_sum is the fastest in every
configuration measured (a few per cent ahead of broadcast, 3-4x ahead of matmul: tiny batched
complex GEMMs are slow on the GPU). The same mode is used for the force outer products and link products.
The links are gathered once per gauge configuration and parity, with the boundary phase
absorbed and the dagger resolved (materialised), shape (B, V/2, 2D, N, N) contiguous: memory
2 x the gauge field. No einsum, no masks: D_hat = 1 - kappa^2 H_eo H_oe costs two half-lattice
hops (the reference costs two full-lattice hops on masked fields).

dtype: the operator works in the dtype of its links (complex128 or complex64, `to(dtype)`);
kappa is a float64 (B,) tensor cast to the matching real dtype, the spin coefficients are in the
field dtype, so nothing is promoted. A field of another dtype is rejected (TypeError).

Solvers
-------
`lean_cg`: CG with the convergence test on the host only every `check_every` iterations; no
per-sample masking (every sample iterates until the slowest has converged; the divisions are
guarded on the device with torch.where). Per-sample iteration counts are those of `solver.cg`
(iterations performed while above the target); the loop count is returned through `info`.
`mixed_precision_cg`: defect correction with complex64 inner solves and complex128 residuals.

Pseudofermion term
------------------
`FastTwoFlavourWilson` has the interface and the physics of `TwoFlavourWilson` (even-odd only):
same heat bath draws, same action and force to the solver tolerance; internally phi and X are
half-lattice fields (`phi` and `X` are exposed as full-lattice tensors for compatibility).
The force bilinear Re[a^dagger dH b] is computed from half spinors (`hopping_derivative`), using
sum_s b_s [(1 + s' gamma) a]_s^* = sum_j (R^dagger b)_j (R^dagger a)_j^*.
"""

import copy

import torch

from neoqcd.fermions.algebra import project_hermitian_traceless
from neoqcd.fermions.dirac import _kappa_tensor, gamma_matrices
from neoqcd.fermions.hmc import Hamiltonian, WilsonGaugeAction
from neoqcd.fermions.lattice import Geometry, fermion_boundary_links, parity
from neoqcd.fermions.pseudofermion import TwoFlavourWilson
from neoqcd.fermions.solver import SolverResult, _norm2, _target

__all__ = [
    "spin_projection_tables",
    "COLOUR_PRODUCTS",
    "resolve_colour_product",
    "colour_mul",
    "site_tables",
    "to_half",
    "to_full",
    "HalfLatticeWilson",
    "lean_cg",
    "mixed_precision_cg",
    "FastTwoFlavourWilson",
    "FastTwoFlavourHamiltonian",
]

_REAL = {torch.complex128: torch.float64, torch.complex64: torch.float32}
COLOUR_PRODUCTS = ("auto", "matmul", "broadcast", "broadcast_sum")


def resolve_colour_product(mode, device):
    """'auto' -> 'broadcast_sum' on CUDA, 'matmul' elsewhere; other modes are returned checked."""
    if mode not in COLOUR_PRODUCTS:
        raise ValueError(f"colour_product must be one of {COLOUR_PRODUCTS}, got {mode!r}")
    if mode == "auto":
        return "broadcast_sum" if torch.device(device).type == "cuda" else "matmul"
    return mode


def colour_mul(a, b, mode="matmul"):
    """Batched matrix product a (.., M, K) @ b (.., K, P) in the given mode (module docstring)."""
    if mode == "matmul":
        return torch.matmul(a, b)
    if mode == "broadcast_sum":
        return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)
    if mode != "broadcast":
        raise ValueError(f"unknown colour product mode {mode!r}")
    K = a.shape[-1]
    out = a[..., :, 0:1] * b[..., 0:1, :]
    for k in range(1, K):
        out.addcmul_(a[..., :, k:k + 1], b[..., k:k + 1, :])
    return out


# ---- tables ------------------------------------------------------------------------------------
def _monomial(M, what):
    """(perm, coef) with M[j, perm[j]] = coef[j], the only nonzero entry of row j."""
    nz = M.abs() > 1e-12
    if not bool((nz.sum(dim=1) == 1).all()):
        raise ValueError(f"{what} is not monomial: {M}")
    perm = nz.to(torch.int64).argmax(dim=1)
    coef = M[torch.arange(M.shape[0]), perm]
    return perm, coef


def spin_projection_tables(D):
    """Half-spinor tables of the 2D hop terms (module docstring), complex128 on the CPU.

    Term k < D is the forward hop along mu = k (s = -1), term k >= D the backward hop along
    mu = k - D (s = +1). Returns a dict with
        "proj_perm" (2D H,) long, "proj_coef" (2D, H) complex: h_{k,j} = u_j + proj_coef[k, j] d[proj_perm[k H + j]],
        "rec_perm" (2D H,) long into the flattened (k, j) axis, "rec_coef" (2D H,) complex:
            lower_{k,j} = rec_coef[k H + j] chi[rec_perm[k H + j]],
        "sign" (2D,) the s of each term, and "g" (D, H, H) the blocks g_mu.
    The dagger tables (H^dagger) are the same with s -> -s, i.e. both coefficients negated.
    """
    gammas, _ = gamma_matrices(D)
    S = gammas.shape[-1]
    H = S // 2
    perms, coefs, gs = [], [], []
    for mu in range(D):
        gm = gammas[mu]
        if gm[:H, :H].abs().max() > 0 or gm[H:, H:].abs().max() > 0:
            raise ValueError(f"gamma_{mu} is not block off-diagonal")
        g = gm[:H, H:]
        if (gm[H:, :H] - g.conj().transpose(0, 1)).abs().max() > 1e-14:
            raise ValueError(f"gamma_{mu} lower block is not g^dagger")
        perms.append((_monomial(g, f"g_{mu}"), _monomial(g.conj().transpose(0, 1), f"g_{mu}^dagger")))
        gs.append(g)
    proj_perm, proj_coef, rec_perm, rec_coef, sign = [], [], [], [], []
    for k in range(2 * D):
        mu, s = k % D, (-1.0 if k < D else 1.0)
        (pp, pc), (rp, rc) = perms[mu]
        proj_perm.append(pp)
        proj_coef.append(s * pc)
        rec_perm.append(k * H + rp)
        rec_coef.append(s * rc)
        sign.append(s)
    return {
        "H": H,
        "proj_perm": torch.cat(proj_perm),
        "proj_coef": torch.stack(proj_coef),
        "rec_perm": torch.cat(rec_perm),
        "rec_coef": torch.cat(rec_coef),
        "sign": torch.tensor(sign, dtype=torch.float64),
        "g": torch.stack(gs),
    }


_SITE_CACHE = {}


def site_tables(geometry, device=None):
    """Site and neighbour index tables of the half-lattice layout (cached per geometry and device).

    Returns a dict with
        "sites" [t]: (V/2,) flat site indices of parity t in lexicographic order;
        "fwd" [mu], "bwd" [mu]: (V,) flat index of x + mu (array n - 1) and x - mu (array n + 1);
        "hop_index" [t]: (V/2 * 2D,) index into the flattened (source site, term) axis of the
            projected source field: entry i * 2D + k is pos(x_i + mu) * 2D + k for k < D and
            pos(x_i - mu) * 2D + k for k >= D (mu = k mod D, x_i the i-th site of parity t);
        "force_index" [t]: as hop_index but the forward neighbour x_i + mu for every k.
    """
    device = torch.device("cpu") if device is None else torch.device(device)
    key = (geometry, str(device))
    if key in _SITE_CACHE:
        return _SITE_CACHE[key]
    if not geometry.has_even_extents:
        raise ValueError(f"even-odd decomposition needs even extents, got lattice {geometry.lattice}")
    D, V = geometry.D, geometry.volume
    sites = torch.arange(V).reshape(geometry.lattice)
    par = parity(geometry).reshape(-1)
    by_parity = [(par == p).nonzero().squeeze(-1) for p in (0, 1)]
    pos = torch.empty(V, dtype=torch.long)
    for p in (0, 1):
        pos[by_parity[p]] = torch.arange(V // 2)
    fwd = [torch.roll(sites, 1, dims=mu).reshape(-1) for mu in range(D)]
    bwd = [torch.roll(sites, -1, dims=mu).reshape(-1) for mu in range(D)]
    terms = torch.arange(2 * D)
    hop_index, force_index = [], []
    for t in (0, 1):
        T = by_parity[t]
        nbr = torch.stack([pos[fwd[mu][T]] for mu in range(D)] + [pos[bwd[mu][T]] for mu in range(D)], dim=1)
        hop_index.append((nbr * 2 * D + terms).reshape(-1))
        nbr_f = torch.stack([pos[fwd[mu][T]] for mu in range(D)] * 2, dim=1)
        force_index.append((nbr_f * 2 * D + terms).reshape(-1))
    out = {
        "sites": [s.to(device) for s in by_parity],
        "fwd": [f.to(device) for f in fwd],
        "bwd": [b.to(device) for b in bwd],
        "hop_index": [h.to(device) for h in hop_index],
        "force_index": [f.to(device) for f in force_index],
    }
    _SITE_CACHE[key] = out
    return out


def to_half(psi, geometry, parity=0):
    """Full-lattice field (B, lattice, S, N) -> its sites of `parity`, (B, V/2, S, N)."""
    tab = site_tables(geometry, psi.device)
    B = psi.shape[0]
    flat = psi.reshape(B, geometry.volume, geometry.spin, geometry.N)
    return flat.index_select(1, tab["sites"][parity])


def to_full(phi, geometry, parity=0, odd=None):
    """Half-lattice field (B, V/2, S, N) of `parity` -> full lattice (B, lattice, S, N).

    The other parity is zero, or `odd` (a half field of the other parity) if given.
    """
    tab = site_tables(geometry, phi.device)
    B = phi.shape[0]
    out = phi.new_zeros((B, geometry.volume, geometry.spin, geometry.N))
    out.index_copy_(1, tab["sites"][parity], phi)
    if odd is not None:
        out.index_copy_(1, tab["sites"][1 - parity], odd)
    return out.reshape(geometry.wilson_shape(B))


# ---- operator ----------------------------------------------------------------------------------
class HalfLatticeWilson:
    """Even-odd preconditioned Wilson operator D_hat = 1 - kappa^2 H_eo H_oe on half-lattice fields.

    Args:
        U: links (B, D, lattice, N, N) (complex128 or complex64).
        kappa: float or (B,) tensor.
        geometry: `Geometry` (inferred from U if omitted); even extents.
        time_sign: -1 antiperiodic (default) or +1 periodic in time.
        colour_product: "auto" (default), "matmul", "broadcast" or "broadcast_sum" (module
            docstring); the resolved mode is the attribute `colour_product`.
        dtype: complex dtype of the operator (default U.dtype); links are gathered in U.dtype
            and then cast.

    Fields are (B', V/2, S, N) on even sites (`apply`, `apply_dagger`, `normal`) with B' = B, or
    any B' if the links have B = 1. Layout and contraction: module docstring.
    """

    def __init__(self, U, kappa, geometry=None, time_sign=-1.0, dtype=None, colour_product="auto"):
        if geometry is None:
            D = U.shape[1]
            geometry = Geometry(D, U.shape[2], U.shape[3] if D > 1 else 1, U.shape[-1])
        if tuple(U.shape[1:]) != geometry.link_shape(1)[1:]:
            raise ValueError(f"links of shape {tuple(U.shape)} do not match geometry {geometry}")
        dtype = U.dtype if dtype is None else dtype
        self.colour_product = resolve_colour_product(colour_product, U.device)
        if dtype not in _REAL:
            raise TypeError(f"HalfLatticeWilson supports complex128 and complex64, got {dtype}")
        self.geometry = geometry
        self.U = U
        self.time_sign = float(time_sign)
        self.kappa = _kappa_tensor(kappa, U.shape[0], U.device)  # float64, (B,) or (1,)
        self._tables = site_tables(geometry, U.device)
        self._spin = spin_projection_tables(geometry.D)
        self._links = self._gather_links(U)
        self._set_dtype(dtype)

    def _gather_links(self, U):
        """Per target parity t: (B, V/2, 2D, N, N) contiguous; term k < D is L_mu(x)^T, term
        k >= D is conj(L_mu(x - mu)) = (L_mu(x - mu)^dagger)^T, boundary phase included."""
        geo = self.geometry
        D, V, N = geo.D, geo.volume, geo.N
        B = U.shape[0]
        flat = fermion_boundary_links(U, self.time_sign).reshape(B, D, V, N, N)
        out = []
        for t in (0, 1):
            T = self._tables["sites"][t]
            fwd = [flat[:, mu].index_select(1, T).transpose(-1, -2) for mu in range(D)]
            bwd = [flat[:, mu].index_select(1, self._tables["bwd"][mu][T]).conj().resolve_conj()
                   for mu in range(D)]
            out.append(torch.stack(fwd + bwd, dim=2).contiguous())
        return out

    def _set_dtype(self, dtype):
        dev = self.U.device
        self._dtype = dtype
        real = _REAL[dtype]
        self._links = [l.to(dtype).contiguous() for l in self._links]
        sp = self._spin
        D, H = self.geometry.D, sp["H"]
        self._pperm = (sp["proj_perm"] + H).to(dev)  # index into the full spin axis (lower half)
        pc = sp["proj_coef"].to(device=dev, dtype=dtype).reshape(2 * D, H, 1)
        self._pcoef = {False: pc, True: -pc}
        self._rperm = sp["rec_perm"].to(dev)
        rc = sp["rec_coef"].to(device=dev, dtype=dtype).reshape(2 * D * H, 1)
        self._rcoef = {False: rc, True: -rc}
        g5 = torch.ones(self.geometry.spin, 1, dtype=real, device=dev)
        g5[H:] = -1.0
        self._g5 = g5
        k = self.kappa.to(real)
        self._k = k.view(-1, 1, 1, 1)
        self._k2 = (self.kappa ** 2).to(real).view(-1, 1, 1, 1)

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
        g = self.geometry
        return (self.batch_size, g.volume // 2, g.spin, g.N)

    def _check(self, psi):
        if psi.dtype != self._dtype:
            raise TypeError(f"field dtype {psi.dtype} differs from the operator dtype {self._dtype} "
                            "(use op.to(dtype))")

    # ---- kernels -----------------------------------------------------------------------------
    def project(self, psi, dagger=False):
        """Half spinors of all 2D hop terms at the sites of psi: (B, V/2, S, N) -> (B, V/2, 2D, H, N)."""
        D, H, N = self.geometry.D, self._spin["H"], self.geometry.N
        B, V2 = psi.shape[:2]
        lower = psi.index_select(2, self._pperm).view(B, V2, 2 * D, H, N)
        return torch.addcmul(psi[:, :, :H].unsqueeze(2), lower, self._pcoef[dagger])

    def hop(self, psi, target_parity, dagger=False):
        """H (dagger: H^dagger) from the parity 1 - t field psi to parity t: H_eo (t = 0) or H_oe (t = 1)."""
        self._check(psi)
        D, H, N = self.geometry.D, self._spin["H"], self.geometry.N
        B, V2 = psi.shape[:2]
        h = self.project(psi, dagger).view(B, V2 * 2 * D, H, N)
        h = h.index_select(1, self._tables["hop_index"][target_parity]).view(B, V2, 2 * D, H, N)
        chi = colour_mul(h, self._links[target_parity], self.colour_product)
        lower = chi.view(B, V2, 2 * D * H, N).index_select(2, self._rperm).mul_(self._rcoef[dagger])
        return torch.cat((chi, lower.view(B, V2, 2 * D, H, N)), dim=3).sum(dim=2)

    def hop_eo(self, psi_odd, dagger=False):
        return self.hop(psi_odd, 0, dagger)

    def hop_oe(self, psi_even, dagger=False):
        return self.hop(psi_even, 1, dagger)

    def gamma5(self, psi):
        return psi * self._g5

    # ---- operator ----------------------------------------------------------------------------
    def apply(self, x, dagger=False):
        """D_hat x = x - kappa^2 H_eo H_oe x (dagger: D_hat^dagger = 1 - kappa^2 H^dag_eo H^dag_oe)."""
        y = self.hop(self.hop(x, 1, dagger), 0, dagger)
        return y.mul_(-self._k2).add_(x)

    def apply_dagger(self, x):
        return self.apply(x, dagger=True)

    def normal(self, x):
        """D_hat^dagger D_hat x."""
        return self.apply(self.apply(x), dagger=True)

    # ---- full-lattice helpers ----------------------------------------------------------------
    def to_half(self, psi, parity=0):
        return to_half(psi, self.geometry, parity)

    def to_full(self, phi_even, phi_odd=None):
        return to_full(phi_even, self.geometry, 0, odd=phi_odd)

    def prepare_rhs(self, b):
        """Half-lattice right-hand side b_e + kappa H_eo b_o of the Schur system for D x = b (b full)."""
        b_e, b_o = self.to_half(b, 0), self.to_half(b, 1)
        return self.prepare_rhs_half(b_e, b_o)

    def prepare_rhs_half(self, b_e, b_o):
        return self.hop(b_o, 0).mul_(self._k).add_(b_e)

    def reconstruct_odd(self, x_e, b_o):
        """x_o = b_o + kappa H_oe x_e."""
        return self.hop(x_e, 1).mul_(self._k).add_(b_o)

    def reconstruct(self, x_even, b):
        """Full solution (B, lattice, S, N) of D x = b from the even half x_even (b full)."""
        return self.to_full(x_even, self.reconstruct_odd(x_even, self.to_half(b, 1)))

    # ---- force -------------------------------------------------------------------------------
    def hopping_derivative(self, a_e, a_o, b_e, b_o):
        """Algebra gradient of Re[a^dagger H b] (as `pseudofermion.hopping_derivative`) from
        half-lattice fields a = (a_e, a_o), b = (b_e, b_o); shape (B, D, lattice, N, N).

        With O1 = sum_s b(x + mu)_s [(1 - gamma_mu) a(x)]_s^dagger and
        O2 = sum_s b(x)_s [(1 + gamma_mu) a(x + mu)]_s^dagger, written with the half spinors
        h = R^dagger psi of the respective projectors (module docstring), G = P_ht(i (L O1 - O2 L^dagger)).
        """
        geo = self.geometry
        D, H, N, V = geo.D, self._spin["H"], geo.N, geo.volume
        B, V2 = a_e.shape[:2]
        pa = [self.project(a_e), self.project(a_o)]
        pb = [self.project(b_e), self.project(b_o)]
        out = a_e.new_empty((B, V, D, N, N))
        for t in (0, 1):
            n = 1 - t
            local = torch.cat((pa[t][:, :, :D], pb[t][:, :, D:]), dim=2)
            src = torch.cat((pb[n][:, :, :D], pa[n][:, :, D:]), dim=2).reshape(B, V2 * 2 * D, H, N)
            nb = src.index_select(1, self._tables["force_index"][t]).view(B, V2, 2 * D, H, N)
            first = torch.cat((nb[:, :, :D], local[:, :, D:]), dim=2)
            second = torch.cat((local[:, :, :D], nb[:, :, D:]), dim=2)
            mode = self.colour_product
            O = colour_mul(first.transpose(-1, -2), second.conj(), mode)  # (B, V/2, 2D, N, N)
            Lt = self._links[t][:, :, :D]  # L^T at the link site x (parity t)
            Q = colour_mul(Lt.transpose(-1, -2), O[:, :, :D], mode) - colour_mul(O[:, :, D:], Lt.conj(), mode)
            out.index_copy_(1, self._tables["sites"][t], project_hermitian_traceless(1j * Q))
        return out.permute(0, 2, 1, 3, 4).reshape((B, D) + geo.lattice + (N, N))


# ---- solvers -----------------------------------------------------------------------------------
def _rdot(a, b):
    """Per-sample Re <a, b>, shape (B,), without forming the complex product."""
    return (torch.view_as_real(a) * torch.view_as_real(b)).reshape(a.shape[0], -1).sum(dim=1)


def _bview(c, field):
    return c.view((-1,) + (1,) * (field.dim() - 1))


def lean_cg(apply_normal, b, x0=None, tol=1e-10, max_iter=1000, check_every=10, info=None):
    """CG for a batched hermitian positive-definite system with few host synchronisations.

    The convergence test (a host sync) runs every `check_every` iterations (first at iteration
    0) and the loop stops when every sample has |r| <= max(tol |b|, eps |r_0|) (the target of
    `solver.cg`) or a non-finite residual. There is no per-sample masking: every sample iterates
    until the check stops the loop; for samples already converged the step sizes are guarded on
    the device (alpha = 0 if p^dagger A p <= tiny, beta = 0 if |r|^2 <= tiny), so they never divide
    by zero and their x keeps improving or stays put. A sample's result therefore depends on how
    long the batch runs, at the level of its tolerance.

    Returns SolverResult(x, iterations, residual): `iterations` (B,) counts, as in `solver.cg`,
    the iterations performed while the sample was above its target (sticky once reached);
    `tol` is a float or a (B,) tensor of per-sample tolerances.
    `residual` is the true |b - A x| / |b| (one extra application of A). If `info` is a dict it
    receives "loop_iterations" (applications of A in the loop: the cost) and "checks".
    """
    check_every = max(1, int(check_every))
    B = b.shape[0]
    b = b.resolve_conj()  # a lazily conjugated rhs would break view_as_real in _rdot
    x0 = None if x0 is None else x0.resolve_conj()
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - apply_normal(x)
    p = r.clone()
    rr = _norm2(r)
    if torch.is_tensor(tol):
        tol = tol.to(device=b.device, dtype=rr.dtype)
    target = _target(b, r, tol)
    tiny = torch.finfo(rr.dtype).tiny
    converged = rr <= target
    iterations = torch.zeros(B, dtype=torch.long, device=b.device)
    zero = torch.zeros_like(rr)
    one = torch.ones_like(rr)
    n, checks = 0, 0
    for n in range(int(max_iter) + 1):
        if n % check_every == 0 or n == int(max_iter):
            checks += 1
            if bool((converged | ~torch.isfinite(rr)).all()) or n == int(max_iter):
                break
        Ap = apply_normal(p)
        pAp = _rdot(p, Ap)
        ok = pAp > tiny
        alpha = torch.where(ok, rr / torch.where(ok, pAp, one), zero)
        a = _bview(alpha.to(b.dtype), b)
        x.addcmul_(p, a)
        r.addcmul_(Ap, a, value=-1)
        rr_new = _norm2(r)
        ok = rr > tiny
        beta = torch.where(ok, rr_new / torch.where(ok, rr, one), zero)
        p.mul_(_bview(beta.to(b.dtype), p)).add_(r)
        iterations += (~converged).long()
        rr = rr_new
        converged = converged | (rr <= target)
    if info is not None:
        info["loop_iterations"] = n
        info["checks"] = checks
    r_true = _norm2(b - apply_normal(x))
    b2 = _norm2(b)
    residual = torch.sqrt(r_true / torch.where(b2 > 0, b2, torch.ones_like(b2)))
    return SolverResult(x, iterations, residual)


def _normal_of(op):
    return op.normal if hasattr(op, "normal") else op


def mixed_precision_cg(op128, op64, b, x0=None, tol=1e-10, inner_tol=1e-6, max_outer=30, max_iter=1000,
                       check_every=10, info=None):
    """Defect correction (iterative refinement) for A x = b, A hermitian positive definite.

    `op128` and `op64` are the operator in complex128 and complex64 (objects with `normal`, e.g.
    HalfLatticeWilson and its `.to(torch.complex64)`, or callables). Outer loop, in complex128:
    r = b - A x (true residual); stop when |r| <= max(tol |b|, eps |r_0|) per sample; else solve
    A e = r / |r| in complex64 with `lean_cg` to the per-sample relative tolerance
    max(inner_tol, 0.5 target / |r|) (no tighter than needed to reach the outer target in this
    step) and update x += |r| e. Converged samples get a zero inner right-hand side (they exit the
    inner CG at once).

    Returns SolverResult(x complex128, iterations (B,) = inner CG iterations summed over the outer
    steps, residual = true complex128 relative residual). `info` (dict) receives "outer" (outer
    steps taken), "inner_iterations" (list of per-outer max inner counts) and "loop_iterations"
    (total complex64 operator applications in the inner loops).
    """
    A128, A64 = _normal_of(op128), _normal_of(op64)
    low = torch.complex64 if b.dtype == torch.complex128 else b.dtype
    B = b.shape[0]
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A128(x)
    target = _target(b, r, tol)
    iterations = torch.zeros(B, dtype=torch.long, device=b.device)
    inner_max, loops, outer = [], 0, 0
    for outer in range(int(max_outer) + 1):
        rr = _norm2(r)
        done = (rr <= target) | ~torch.isfinite(rr)
        if bool(done.all()) or outer == int(max_outer):
            break
        scale = torch.sqrt(torch.where(done, torch.ones_like(rr), rr))
        rhs = torch.where(_bview(done, r), torch.zeros_like(r), r / _bview(scale, r).to(r.dtype)).to(low)
        sub = {}
        inner = torch.clamp(0.5 * torch.sqrt(target) / scale, min=float(inner_tol), max=0.5)
        e = lean_cg(A64, rhs, tol=inner, max_iter=max_iter, check_every=check_every, info=sub)
        loops += sub["loop_iterations"]
        inner_max.append(int(e.iterations.max().item()))
        iterations += e.iterations
        x.addcmul_(e.x.to(b.dtype), _bview(scale.to(b.dtype), x))
        r = b - A128(x)
    if info is not None:
        info.update(outer=outer, inner_iterations=inner_max, loop_iterations=loops)
    b2 = _norm2(b)
    residual = torch.sqrt(_norm2(r) / torch.where(b2 > 0, b2, torch.ones_like(b2)))
    return SolverResult(x, iterations, residual)


# ---- pseudofermion term ------------------------------------------------------------------------
class FastTwoFlavourWilson(TwoFlavourWilson):
    """`TwoFlavourWilson` (even-odd) through `HalfLatticeWilson` and `lean_cg` / `mixed_precision_cg`.

    Same physics, heat bath (the same Gaussian draws: eta is drawn on the full lattice exactly as
    in the reference and restricted to the even sites), action and force as `TwoFlavourWilson`
    to the solver tolerance; same `stats` keys plus "cg_loop_iterations" (operator applications
    actually performed by the batch, the cost of `lean_cg`) and "outer_iterations" (mixed).
    The eta^dagger eta cache of `action_difference` keeps the identity + `_version` guard.

    Extra args:
        precision: "double" (lean_cg in complex128) or "mixed" (complex64 inner solves,
            complex128 residuals, converging to the same tolerances).
        check_every: host convergence check interval of lean_cg.
        inner_tol, max_outer: of `mixed_precision_cg`.
        colour_product: of `HalfLatticeWilson` ("auto": broadcast_sum on CUDA, matmul on CPU).

    Internally `phi` and `X` are half-lattice fields (`_phi_h`, `_X_h`); the attributes `phi`
    and `X` read/write full-lattice tensors (odd sites zero) for compatibility. `_solve(op, tol)`
    takes a `HalfLatticeWilson` and returns the half-lattice X, so code that mixes `_solve` with
    full-lattice phi (snf_layers.PseudofermionAction) must keep using the reference class.
    The operator (and its complex64 copy) is cached per links tensor (identity and `_version`).
    """

    def __init__(self, kappa, geometry, time_sign=-1.0, even_odd=True, solver_tol=1e-10, action_tol=None,
                 max_iter=2000, reuse_guess=True, precision="double", check_every=10, inner_tol=1e-6,
                 max_outer=30, colour_product="auto"):
        if not even_odd:
            raise ValueError("FastTwoFlavourWilson is even-odd only")
        if precision not in ("double", "mixed"):
            raise ValueError(f"precision must be 'double' or 'mixed', got {precision!r}")
        self._phi_h = None
        self._X_h = None
        self._op_cache = None
        self.precision = precision
        self.check_every = int(check_every)
        self.inner_tol = float(inner_tol)
        self.max_outer = int(max_outer)
        if colour_product not in COLOUR_PRODUCTS:
            raise ValueError(f"colour_product must be one of {COLOUR_PRODUCTS}, got {colour_product!r}")
        self.colour_product = colour_product
        super().__init__(kappa, geometry, time_sign=time_sign, even_odd=True, solver_tol=solver_tol,
                         action_tol=action_tol, max_iter=max_iter, reuse_guess=reuse_guess)

    def reset_stats(self):
        super().reset_stats()
        self.stats["cg_loop_iterations"] = 0
        self.stats["outer_iterations"] = 0

    # ---- full-lattice views of the internal half fields --------------------------------------
    @property
    def phi(self):
        return None if self._phi_h is None else to_full(self._phi_h, self.geometry)

    @phi.setter
    def phi(self, value):
        self._phi_h = None if value is None else to_half(value, self.geometry)

    @property
    def X(self):
        return None if self._X_h is None else to_full(self._X_h, self.geometry)

    @X.setter
    def X(self, value):
        self._X_h = None if value is None else to_half(value, self.geometry)

    # ---- operators ---------------------------------------------------------------------------
    def operator(self, U):
        """HalfLatticeWilson at links U, cached while U is the same tensor with the same _version."""
        c = self._op_cache
        if c is not None and c[0] is U and c[1] == U._version:
            return c[2]
        op = HalfLatticeWilson(U, self.kappa, self.geometry, self.time_sign, colour_product=self.colour_product)
        op._low = None
        self._op_cache = (U, U._version, op)
        return op

    @staticmethod
    def _low_precision(op):
        low = getattr(op, "_low", None)
        if low is None:
            low = op.to(torch.complex64)
            op._low = low
        return low

    def _solve(self, op, tol):
        if self._phi_h is None:
            raise RuntimeError("the pseudofermion field is not set: call refresh(U, generator) first")
        x0 = self._X_h if (self.reuse_guess and self._X_h is not None) else None
        info = {}
        if self.precision == "double":
            res = lean_cg(op.normal, self._phi_h, x0=x0, tol=tol, max_iter=self.max_iter,
                          check_every=self.check_every, info=info)
        else:
            res = mixed_precision_cg(op, self._low_precision(op), self._phi_h, x0=x0, tol=tol,
                                     inner_tol=self.inner_tol, max_outer=self.max_outer,
                                     max_iter=self.max_iter, check_every=self.check_every, info=info)
            self.stats["outer_iterations"] += info["outer"]
        self.stats["cg_loop_iterations"] += info["loop_iterations"]
        self._count(res)
        self._X_h = res.x
        return res.x

    # ---- ActionTerm interface ----------------------------------------------------------------
    def refresh(self, U, generator=None):
        """phi = D_hat^dagger eta_e with the same full-lattice draw of eta as the reference."""
        op = self.operator(U)
        eta = torch.randn(self.geometry.wilson_shape(U.shape[0]), generator=generator,
                          dtype=U.dtype, device=U.device)
        eta = to_half(eta, self.geometry)
        self._phi_h = op.apply_dagger(eta)
        self._X_h = None
        self._refresh_links = U
        self._refresh_version = U._version
        self._refresh_action = (eta.conj() * eta).real.reshape(eta.shape[0], -1).sum(1)
        self.stats["refresh_calls"] += 1

    def action(self, U):
        self.stats["action_calls"] += 1
        X = self._solve(self.operator(U), self.action_tol)
        return (self._phi_h.conj() * X).real.reshape(X.shape[0], -1).sum(1)

    def force(self, U):
        """Force of the reference: 2 kappa^2 grad Re[(Y + Z)^dagger H (X + W)], from half fields
        X, W = H_oe X, Y = D_hat X = X - kappa^2 H_eo W, Z = (H^dagger)_oe Y."""
        self.stats["force_calls"] += 1
        op = self.operator(U)
        X = self._solve(op, self.solver_tol)
        W = op.hop(X, 1)
        Y = op.hop(W, 0).mul_(-op._k2).add_(X)
        Z = op.hop(Y, 1, dagger=True)
        G = op.hopping_derivative(Y, Z, X, W)
        return (2.0 * op.kappa ** 2).view((-1,) + (1,) * (G.dim() - 1)).to(G.dtype) * G


class FastTwoFlavourHamiltonian(Hamiltonian):
    """`TwoFlavourHamiltonian` with `FastTwoFlavourWilson` (same levels: fermion outside, gauge inside)."""

    def __init__(self, beta, kappa, geometry, n_gauge_substeps=1, plaquette_weights=None, **fermion_kwargs):
        self.fermion = FastTwoFlavourWilson(kappa, geometry, **fermion_kwargs)
        self.gauge = WilsonGaugeAction(beta, geometry, plaquette_weights)
        super().__init__([([self.fermion], 1), ([self.gauge], n_gauge_substeps)])
