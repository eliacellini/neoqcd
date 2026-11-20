"""Fast path of the DD-HMC (`ddhmc.py`, Luescher hep-lat/0409106): the pseudofermion terms through
the half-lattice operator of `fast.py` (and the fused hop of `fused.py`).

The reference classes of `ddhmc.py` stay the ground truth; test/test_fermions_ddhmc_fast.py checks
every class here against them (and against dense matrices) to the solver tolerance. Conventions
and formulas are those of `ddhmc.py` (module doc) and `fast.py`.

Block term
----------
S_bl = phi^dag (D_bl_hat^dag D_bl_hat)^-1 phi is `FastTwoFlavourWilson` with the operator built on
the block links: `HalfLatticeWilson(U * inblock_mask)` (the cut links set to zero, so both hops
across block faces vanish and D_bl = D_Omega (+) D_Omega*). The global parity is the parity of the
blocks' own even-odd decomposition because the block extents are even. One global CG covers all
blocks (no block-local solver). The operator is cached on the UNMASKED links (identity and
`_version`, as `FastTwoFlavourWilson`), so the masked copy is made once per configuration; the force
is the parent's half-lattice force (`HalfLatticeWilson.hopping_derivative` with the masked links,
zero on the cut links) times the active-link mask. Same heat-bath draws as `BlockTwoFlavour`, same
stats, policy (`on_unconverged`), precisions ("double": lean CG; "mixed": complex64 inner CG) and,
in `FusedBlockTwoFlavour`, the fused hop kernels ("cuda", "cpu", "loop", "auto"): the kernel reads
full links, so the zeroed cut links need no kernel change.

Schur term
----------
`FastSchurTwoFlavour` is `SchurTwoFlavour` (S_R = |R^-1 phi_R|^2, phi_R on the code's dOmega*) with
every field kept as a pair of half-lattice fields (even, odd sites; `fast.to_half` order) and
every solve D^-1 b (full operator) or D_bl^-1 b (block operator) done as in `solver.solve` but on
the half lattice:
    rhs = b_e + kappa H_eo b_o                       (`HalfLatticeWilson.prepare_rhs_half`),
    D_hat^dag D_hat x_e = D_hat^dag rhs               (`lean_cg`, or `mixed_precision_cg`),
    x_o = b_o + kappa H_oe x_e                        (`reconstruct_odd`),
from zero (no chronological guess, as the reference). The residual checked by the convergence
policy is the reference's, the true |b - D x| / |b| of the full-lattice system: its even part is
rhs - D_hat x_e and its odd part vanishes up to rounding, so it is |rhs - D_hat x_e| / |b| (one
extra application of D_hat per solve).
The hops between the colours use the FULL operator and site masks only:
    D_dOmega psi  = -kappa P_Omega  H P_Omega* psi,     D_dOmegaS psi = -kappa P_Omega* H P_Omega psi,
which is exactly the reference's hop through the cut links: the in-block hops join sites of one
colour, so P_Omega H_in P_Omega* = 0 term by term (the masked sources contribute exact zeros), and
no third operator (cut links) is gathered. Refresh: phi_R = P(v - D_bl^-1 D_dOmegaS D_bl^-1
D_dOmega v), v = P eta (two block solves, the block operator built for the refresh only);
action: chi = P(phi_R - D^-1 D_dOmega phi_R); force: zeta = gamma_5 D^-1 gamma_5 chi and
F_R = -2 kappa `HalfLatticeWilson.hopping_derivative`(zeta, x) masked to the active links (two full
solves). The full operator (and its complex64 copy with mixed precision) is cached per links tensor
(identity and `_version`) and operator parameters, as in `FastTwoFlavourWilson`. Same draws
(eta on the full lattice, then masked and split), stats (plus cg_loop_iterations,
outer_iterations, double_applications) and `on_unconverged` policy as the reference. With
`kernel` != "off" the operators are `FusedHalfLatticeWilson` (kernel, link_layout, strict).

`FastDDHMCHamiltonian` is `DDHMCHamiltonian` with these terms.
"""

import torch

from neoqcd.fermions.ddhmc import DDHMCHamiltonian, SchurTwoFlavour
from neoqcd.fermions.fast import (
    COLOUR_PRODUCTS,
    FastTwoFlavourWilson,
    HalfLatticeWilson,
    lean_cg,
    mixed_precision_cg,
    site_tables,
    to_full,
    to_half,
)
from neoqcd.fermions.fused import (
    DEFAULT_LINK_LAYOUT,
    KERNELS,
    LINK_LAYOUTS,
    FusedHalfLatticeWilson,
    FusedTwoFlavourWilson,
)
from neoqcd.fermions.solver import SolverResult, _norm2

__all__ = ["FastBlockTwoFlavour", "FusedBlockTwoFlavour", "FastSchurTwoFlavour", "FastDDHMCHamiltonian",
           "FAST_OPTIONS"]

# options of the fast terms (FastTwoFlavourWilson) shared by both DD terms
FAST_OPTIONS = ("precision", "check_every", "inner_tol", "max_outer", "colour_product")


class _BlockOperator:
    """`operator(U)`: the half-lattice operator of the block links U * inblock_mask, cached on the
    unmasked U (identity, `_version`) and the operator key; `force` masked to the active links."""

    def operator(self, U):
        c = self._op_cache
        key = self._operator_key()
        if c is not None and c[0] is U and c[1] == U._version and self._same_key(c[2], key):
            return c[3]
        op = self._half_lattice(U * self.blocks.link_mask(self.blocks.inblock_links, U))
        op._low = None
        self._op_cache = (U, U._version, key, op)
        return op

    def force(self, U):
        return self.blocks.link_mask(self.blocks.active_links, U) * super().force(U)


class FastBlockTwoFlavour(_BlockOperator, FastTwoFlavourWilson):
    """`ddhmc.BlockTwoFlavour` through `HalfLatticeWilson` on the block links (module doc). Extra
    keyword arguments are those of `FastTwoFlavourWilson` (precision, check_every, inner_tol,
    max_outer, colour_product, solver_tol, action_tol, max_iter, reuse_guess, on_unconverged)."""

    def __init__(self, kappa, blocks, time_sign=-1.0, **kwargs):
        self.blocks = blocks
        super().__init__(kappa, blocks.geometry, time_sign=time_sign, **kwargs)

    def _half_lattice(self, links):
        return HalfLatticeWilson(links, self.kappa, self.geometry, self.time_sign, colour_product=self.colour_product)


class FusedBlockTwoFlavour(_BlockOperator, FusedTwoFlavourWilson):
    """`FastBlockTwoFlavour` with `FusedHalfLatticeWilson` (kernel, link_layout, strict as
    `FusedTwoFlavourWilson`)."""

    def __init__(self, kappa, blocks, time_sign=-1.0, **kwargs):
        self.blocks = blocks
        super().__init__(kappa, blocks.geometry, time_sign=time_sign, **kwargs)

    def _half_lattice(self, links):
        return FusedHalfLatticeWilson(links, self.kappa, self.geometry, self.time_sign,
                                      colour_product=self.colour_product, kernel=self.kernel,
                                      link_layout=self.link_layout, strict=self.strict)


class FastSchurTwoFlavour(SchurTwoFlavour):
    """`ddhmc.SchurTwoFlavour` on the half lattice (module doc, "Schur term").

    Extra args (as `FastTwoFlavourWilson`): precision ("double": `lean_cg`; "mixed": complex64
    inner CG with complex128 residuals), check_every, inner_tol, max_outer (mixed: max_iter is
    the total inner budget per solve), colour_product; kernel ("off": `HalfLatticeWilson`, else a
    kernel of `fused.KERNELS` for `FusedHalfLatticeWilson`), link_layout, strict.

    Internally phi_R is the half-lattice pair `_phi_h` = (even, odd); `phi` reads and writes it as a
    full-lattice tensor. `_solve(op, b, tol)` takes a half-lattice operator and a pair b = (b_e, b_o)
    and returns the pair x with D x = b (all solves go through it, so a wrapper of the instance
    attribute, as main_hmc's SolveTimer, sees them). `operators(U, tol)` (inherited) still gives
    the reference `DDOperators`.
    """

    def __init__(self, kappa, blocks, time_sign=-1.0, solver_tol=1e-10, action_tol=None, max_iter=5000,
                 on_unconverged="raise", precision="double", check_every=10, inner_tol=1e-6, max_outer=30,
                 colour_product="auto", kernel="off", link_layout=DEFAULT_LINK_LAYOUT, strict=False):
        if precision not in ("double", "mixed"):
            raise ValueError(f"precision must be 'double' or 'mixed', got {precision!r}")
        if colour_product not in COLOUR_PRODUCTS:
            raise ValueError(f"colour_product must be one of {COLOUR_PRODUCTS}, got {colour_product!r}")
        if kernel != "off" and kernel not in KERNELS:
            raise ValueError(f"kernel must be 'off' or one of {KERNELS}, got {kernel!r}")
        if link_layout not in LINK_LAYOUTS:
            raise ValueError(f"link_layout must be one of {LINK_LAYOUTS}, got {link_layout!r}")
        self._phi_h = None
        self._op_cache = None
        self._half_masks = {}
        self.precision = precision
        self.check_every = int(check_every)
        self.inner_tol = float(inner_tol)
        self.max_outer = int(max_outer)
        self.colour_product = colour_product
        self.kernel = kernel
        self.link_layout = link_layout
        self.strict = bool(strict)
        super().__init__(kappa, blocks, time_sign=time_sign, solver_tol=solver_tol, action_tol=action_tol,
                         max_iter=max_iter, on_unconverged=on_unconverged)

    def reset_stats(self):
        super().reset_stats()
        self.stats.update(cg_loop_iterations=0, outer_iterations=0, double_applications=0)

    # ---- full-lattice view of phi_R --------------------------------------------------------------
    @property
    def phi(self):
        if self._phi_h is None:
            return None
        return to_full(self._phi_h[0], self.geometry, 0, odd=self._phi_h[1])

    @phi.setter
    def phi(self, value):
        self._phi_h = None if value is None else self._split(value)

    def _split(self, psi):
        return to_half(psi, self.geometry, 0), to_half(psi, self.geometry, 1)

    # ---- masks and operators ---------------------------------------------------------------------
    def _mask(self, name, like):
        """(even, odd) halves of the site mask `blocks.<name>` as real (1, V/2, 1, 1) tensors on
        like's device and real dtype (cached)."""
        key = (name, str(like.device), like.real.dtype)
        out = self._half_masks.get(key)
        if out is None:
            tab = site_tables(self.geometry, like.device)
            flat = getattr(self.blocks, name).reshape(-1).to(like.device)
            out = tuple(flat.index_select(0, tab["sites"][p]).to(like.real.dtype).view(1, -1, 1, 1) for p in (0, 1))
            self._half_masks[key] = out
        return out

    def _masked(self, name, psi):
        m = self._mask(name, psi[0])
        return psi[0] * m[0], psi[1] * m[1]

    def _operator_key(self):
        k = self.kappa
        kappa = ("tensor", k.detach().to("cpu", torch.float64).clone()) if torch.is_tensor(k) else ("float", float(k))
        return (kappa, float(self.time_sign), self.colour_product, self.geometry, self.kernel, self.link_layout)

    def _half_lattice(self, links):
        if self.kernel == "off":
            op = HalfLatticeWilson(links, self.kappa, self.geometry, self.time_sign, colour_product=self.colour_product)
        else:
            op = FusedHalfLatticeWilson(links, self.kappa, self.geometry, self.time_sign,
                                        colour_product=self.colour_product, kernel=self.kernel,
                                        link_layout=self.link_layout, strict=self.strict)
        op._low = None
        return op

    def full_operator(self, U):
        """Half-lattice D_hat at links U, cached on U (identity, _version) and the operator key."""
        c = self._op_cache
        key = self._operator_key()
        if c is not None and c[0] is U and c[1] == U._version and FastTwoFlavourWilson._same_key(c[2], key):
            return c[3]
        op = self._half_lattice(U)
        self._op_cache = (U, U._version, key, op)
        return op

    def block_operator(self, U):
        """Half-lattice D_bl_hat on the block links U * inblock_mask (not cached: the refresh only)."""
        return self._half_lattice(U * self.blocks.link_mask(self.blocks.inblock_links, U))

    # ---- solves and hops -------------------------------------------------------------------------
    def _solve(self, op, b, tol):
        """x = (x_e, x_o) with D x = b for the full or block half-lattice operator `op` (module doc);
        bookkeeping and the convergence check on the true residual |b - D x| / |b|."""
        b_e, b_o = b
        rhs = op.prepare_rhs_half(b_e, b_o)
        nrhs = op.apply_dagger(rhs)
        info = {}
        if self.precision == "double":
            res = lean_cg(op.normal, nrhs, tol=tol, max_iter=self.max_iter, check_every=self.check_every, info=info)
        else:
            res = mixed_precision_cg(op, FastTwoFlavourWilson._low_precision(op), nrhs, tol=tol,
                                     inner_tol=self.inner_tol, max_outer=self.max_outer, max_iter=self.max_iter,
                                     check_every=self.check_every, info=info)
            self.stats["outer_iterations"] += info["outer"]
            self.stats["double_applications"] += info["double_applications"]
        self.stats["cg_loop_iterations"] += info["loop_iterations"]
        x_e = res.x
        x_o = op.reconstruct_odd(x_e, b_o)
        b2 = _norm2(b_e) + _norm2(b_o)
        residual = torch.sqrt(_norm2(rhs - op.apply(x_e)) / torch.where(b2 > 0, b2, torch.ones_like(b2)))
        self._count(SolverResult(None, res.iterations, residual), tol, self._solve_kind or "solve")
        return x_e, x_o

    def _hop(self, op, psi, source, target):
        """-kappa P_target H P_source psi with the full hopping H (module doc: = the cut-link hop)."""
        s_e, s_o = self._masked(source, psi)
        t_e, t_o = self._mask(target, psi[0])
        k = -op._k
        return op.hop(s_o, 0).mul_(k).mul_(t_e), op.hop(s_e, 1).mul_(k).mul_(t_o)

    def D_dOmega(self, op, psi):
        """P_Omega (-kappa H_cut) P_Omega* psi (white -> black), psi a half-lattice pair."""
        return self._hop(op, psi, "white", "black")

    def D_dOmegaS(self, op, psi):
        """P_Omega* (-kappa H_cut) P_Omega psi (black -> white), psi a half-lattice pair."""
        return self._hop(op, psi, "black", "white")

    @staticmethod
    def _sub(a, b):
        return a[0] - b[0], a[1] - b[1]

    def R(self, U, eta):
        """R eta = P (v - D_bl^-1 D_dOmegaS D_bl^-1 D_dOmega v), v = P eta (two block solves at
        action_tol)."""
        full, block = self.full_operator(U), self.block_operator(U)
        v = self._masked("exterior_boundary", eta)
        y = self._solve(block, self.D_dOmega(full, v), self.action_tol)
        z = self._solve(block, self.D_dOmegaS(full, y), self.action_tol)
        return self._masked("exterior_boundary", self._sub(v, z))

    def chi_half(self, U, tol):
        """(x, chi) with x = D^-1 D_dOmega phi_R and chi = R^-1 phi_R = P(phi_R - x) (one full solve)."""
        if self._phi_h is None:
            raise RuntimeError("phi_R is not set: call refresh(U, generator) first")
        op = self.full_operator(U)
        x = self._solve(op, self.D_dOmega(op, self._phi_h), tol)
        return op, x, self._masked("exterior_boundary", self._sub(self._phi_h, x))

    # ---- ActionTerm interface --------------------------------------------------------------------
    def refresh(self, U, generator=None):
        """phi_R = R eta with the reference draw of eta (full lattice, masked to dOmega*)."""
        eta = self._draw_eta(U, generator)
        self._phi_h = self._as("refresh", self.R, U, self._split(eta))
        self._set_refresh(U, eta)

    def chi(self, U, tol):
        """(full operator, chi = R^-1 phi_R as a full-lattice tensor)."""
        op, _, chi = self.chi_half(U, tol)
        return op, to_full(chi[0], self.geometry, 0, odd=chi[1])

    def action(self, U):
        self.stats["action_calls"] += 1
        _, _, chi = self._as("action", self.chi_half, U, self.action_tol)
        return sum((c.conj() * c).real.reshape(c.shape[0], -1).sum(1) for c in chi)

    def force(self, U):
        """F_R = -2 kappa grad Re[zeta^dag H x] on the active links, zeta = gamma_5 D^-1 gamma_5 chi."""
        self.stats["force_calls"] += 1
        op, x, chi = self._as("force", self.chi_half, U, self.solver_tol)
        g5 = (op.gamma5(chi[0]), op.gamma5(chi[1]))
        z = self._as("force", self._solve, op, g5, self.solver_tol)
        zeta = (op.gamma5(z[0]), op.gamma5(z[1]))
        G = op.hopping_derivative(zeta[0], zeta[1], x[0], x[1])
        k = op.kappa.view((-1,) + (1,) * (G.dim() - 1)).to(G.dtype)
        return self.blocks.link_mask(self.blocks.active_links, U) * (-2.0 * k * G)


class FastDDHMCHamiltonian(DDHMCHamiltonian):
    """`DDHMCHamiltonian` with the fast terms (levels, masks and translations unchanged).

    `fast_kwargs` (keys in FAST_OPTIONS) go to both fermion terms; `kernel` ("off" = the --fast
    operator, else a kernel of `fused.KERNELS`), `link_layout` and `strict` select the fused hop
    for both. `block_kwargs` / `schur_kwargs` as in `DDHMCHamiltonian`. `fast_schur=False` keeps the
    reference Schur term (the fast block term alone).
    """

    def __init__(self, beta, kappa, blocks, n_block_substeps=1, n_gauge_substeps=1, time_sign=-1.0,
                 block_kwargs=None, schur_kwargs=None, fast_kwargs=None, kernel="off",
                 link_layout=DEFAULT_LINK_LAYOUT, strict=False, fast_schur=True):
        fast_kwargs = dict(fast_kwargs or {})
        unknown = set(fast_kwargs) - set(FAST_OPTIONS)
        if unknown:
            raise ValueError(f"unknown fast options {sorted(unknown)}; expected a subset of {FAST_OPTIONS}")
        self.fast_kwargs = fast_kwargs
        self.fast_schur = bool(fast_schur)
        self.kernel = kernel
        self.kernel_kwargs = {} if kernel == "off" else {"kernel": kernel, "link_layout": link_layout,
                                                         "strict": bool(strict)}
        super().__init__(beta, kappa, blocks, n_block_substeps=n_block_substeps, n_gauge_substeps=n_gauge_substeps,
                         time_sign=time_sign, block_kwargs=block_kwargs, schur_kwargs=schur_kwargs)

    def _make_block(self, kappa, blocks, time_sign, kwargs):
        kwargs = dict(self.fast_kwargs, **kwargs)
        if self.kernel == "off":
            return FastBlockTwoFlavour(kappa, blocks, time_sign=time_sign, **kwargs)
        return FusedBlockTwoFlavour(kappa, blocks, time_sign=time_sign, **self.kernel_kwargs, **kwargs)

    def _make_schur(self, kappa, blocks, time_sign, kwargs):
        if not self.fast_schur:
            return SchurTwoFlavour(kappa, blocks, time_sign=time_sign, **kwargs)
        return FastSchurTwoFlavour(kappa, blocks, time_sign=time_sign, **self.kernel_kwargs,
                                   **dict(self.fast_kwargs, **kwargs))
