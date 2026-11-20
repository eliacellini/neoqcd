"""Domain-decomposition HMC (DD-HMC) for N_f = 2 Wilson fermions, first stage.

Reference: M. Luescher, Comput. Phys. Commun. 165 (2005) 199, hep-lat/0409106. Conventions of
neoqcd.fermions.dirac / lattice / pseudofermion / hmc (D = 1 - kappa H, "x + mu" = array index
n - 1, force tr(F T) = d/dt S(e^{itT} U)|_0, hermitian traceless).

Blocks. The lattice is divided into non-overlapping rectangular blocks Lambda of size
b = (b_0, ..., b_{D-1}) (even b_mu, extent n_mu a multiple of b_mu with n_mu / b_mu even, so that
the chessboard colouring of the blocks is consistent across the periodic wrap). The block of the
site with array index n has coordinates n_mu // b_mu; its colour is sum_mu (n_mu // b_mu) mod 2,
0 = black (Omega), 1 = white (Omega*). Local coordinates are y_mu = n_mu mod b_mu.

Cut links. U_mu at array index n connects n and n + mu = array index n - 1 along mu. It joins two
different blocks iff y_mu = 0 (the block's first slice: its partner n - 1 is the last slice of the
previous block, and at n_mu = 0 the wrap-around partner n_mu - 1 = n_mu - 1 + extent). These are
the "cut" links; all others are "in-block" links. In particular the time links at t = 0, which
carry the antiperiodic fermion sign, are always cut.

Operators. With hopping H = H_in + H_cut (H_cut = the hopping built from the cut links only, both
the forward term U_mu(x) psi(x + mu) and the backward term U_mu(x - mu)^dag psi(x - mu)):
    D = D_bl - kappa H_cut,   D_bl = 1 - kappa H_in = D_Omega (+) D_Omega*,
D_bl is the direct sum of the block operators D_Lambda with Dirichlet boundary conditions; it is
`BlockWilsonDirac` = `WilsonDirac` with the cut links set to zero. In the (Omega, Omega*) block
order
    D = [[D_Omega, D_dOmega], [D_dOmegaS, D_Omega*]],
    D_dOmega  = P_Omega  (-kappa H_cut) P_Omega*   (hops Omega* -> Omega),
    D_dOmegaS = P_Omega* (-kappa H_cut) P_Omega    (hops Omega -> Omega*).
Every face neighbour of a block has the opposite colour, so H_cut only connects different colours.

Naming: COLOUR-MIRRORED with respect to the paper. In hep-lat/0409106 dOmega is the exterior
boundary of Omega (it lies in Omega*), dOmega* the black sites next to white ones, and R (R_2 on
dOmega*) comes from the Schur complement on Omega (black). This module takes the Schur complement
on Omega* (white) and calls "dOmega*" (attribute `exterior_boundary`) the WHITE sites with a black
neighbour, i.e. the paper's dOmega; `D_dOmega` / `D_dOmegaS` below are the hops white -> black and
black -> white. Exchanging the colours relabels Omega <-> Omega*, so det D and the algorithm are
the same; only the names differ. Below, "dOmega*" always means the code's set (paper's dOmega).

Exact deviations from the paper (both leave the target distribution unchanged):
  - phi_R lives on all spin-colour components of the boundary sites ("dOmega*" x spin x colour),
    not on the reduced space of eq. (B.4) (the half of the spinor components that the projectors
    (1 -+ gamma_mu) actually touch); R then contains trivial directions and costs a bit more.
  - the block heat bath is phi = D_bl_hat^dag eta (the repo convention of TwoFlavourWilson,
    S = phi^dag (D_hat^dag D_hat)^-1 phi), the same Gaussian as the paper's up to the convention.

Factorisation (eq. 3.5 with the colours mirrored: white Schur complement). With the Schur complement on Omega*,
    det D = det D_Omega det(D_Omega* - D_dOmegaS D_Omega^-1 D_dOmega)
          = det D_Omega det D_Omega* det(1 - K),   K = D_Omega*^-1 D_dOmegaS D_Omega^-1 D_dOmega.
D_dOmega reads only white sites with a black neighbour, i.e. the exterior boundary of the black
blocks, dOmega* (= white sites on the interior boundary of their block), so K = K P with
P = P_dOmega*; in the basis (dOmega*, rest of Omega*) 1 - K is block lower triangular, hence
    det(1 - K) = det R,   R = P (1 - D_Omega*^-1 D_dOmegaS D_Omega^-1 D_dOmega) P   on dOmega*,
and 1 - K is the identity on the white sites outside dOmega* (checked densely in the tests).
R^-1 with ONE full solve: (1 - K)^-1 = S^-1 D_Omega* with S = D_Omega* - D_dOmegaS D_Omega^-1
D_dOmega = [(D^-1)_{Omega* Omega*}]^-1; for psi on dOmega*, D psi = D_Omega* psi + D_dOmega psi, so
D^-1 D_Omega* psi = psi - D^-1 D_dOmega psi and
    R^-1 psi = P (psi - D^-1 D_dOmega psi)      (P_Omega* D^-1 ... , then restricted to dOmega*).
R psi uses two block solves (D_Omega^-1, D_Omega*^-1, both = D_bl^-1 on the right support).

N_f = 2 pseudofermion actions (det(D^dag D) = |det D_bl|^2 |det R|^2):
    S_bl = phi^dag (D_bl_hat^dag D_bl_hat)^-1 phi       (even-odd inside the blocks: the global
           parity works because D_bl only hops between neighbours; det D_bl = det D_bl_hat),
    S_R  = |R^-1 phi_R|^2 = phi_R^dag (R R^dag)^-1 phi_R,   phi_R on dOmega*,
with integral d phi_R exp(-S_R) ~ det(R R^dag) = |det R|^2.
Heat bath: phi_bl = D_bl_hat^dag eta (TwoFlavourWilson with the block operator), phi_R = R eta with
eta complex Gaussian on dOmega* (<|eta_i|^2> = 1); then S_R = eta^dag eta exactly.

Forces (only on ACTIVE links, see below; all active links are in-block, so dH_cut = 0 and
d D_dOmega = 0 there):
    S_bl: the TwoFlavourWilson force of D_bl, masked;
    S_R:  chi = R^-1 phi_R = P(phi_R - x), x = D^-1 D_dOmega phi_R. On active links
          d chi = P D^-1 dD x, so dS_R = 2 Re[chi^dag d chi] = 2 Re[zeta^dag dD x] with
          zeta = D^-dag chi = gamma_5 D^-1 gamma_5 chi; dD = -kappa dH gives
              F_R = -2 kappa hopping_derivative(D, zeta, x)   (masked to the active links).
          Two full solves (x and zeta).
    gauge: the Wilson gauge force, masked.

Active links (Luescher, sect. 3.3 and fig. 2): links with both endpoints in the same block and at
most one endpoint on the block's interior boundary (sites with y_nu in {0, b_nu - 1} for some nu).
In direction mu the in-block link at local y (y_mu >= 1) joins y and y - e_mu; both are boundary
points iff some transverse y_nu (nu != mu) is in {0, b_nu - 1}, or b_mu = 2. Count per block and
direction: a_mu = (b_mu - 1) [b_mu > 2] prod_{nu != mu} (b_nu - 2), fraction
sum_mu a_mu / (D prod b). For b = 6^4 it is 0.247 and for 12^4 0.531, the paper's "a minority of
25%" and "greater than 50%" (sect. 3.5); for 8^4 it is 7 * 6^3 / 8^4 = 0.369. A block size 2 in
any direction leaves no active link at all (`DDHMC` refuses such a decomposition).

Translations. Only active links are evolved in a trajectory (momenta zero elsewhere, forces
masked), so the gauge field is translated once per update cycle (eq. 3.9),
    U(x, mu) -> U(x + v, mu),   i.e. `translate(U, v)`: U'(n) = U(n - v) in array indices,
since "x + mu" is array index n - 1. The paper allows it "at the beginning or the end of every
update cycle" (sect. 3.3) and does it at the end (sect. 3.4 (c)); `DDHMC.cycle` does it at the
beginning, the same Markov chain up to where it starts. The blocks (and their colours) stay
fixed; the pseudofermions are drawn anew in every trajectory, so nothing else is translated. It is
an exact symmetry of the action: the antiperiodic twist on the time links at t = 0 is fixed, and
moving it with the field is the change of variables psi(n) -> -psi(n) on the time slices that
wrapped around (tests). The vectors v (`DDHMC(translation=...)`, appendix C):
  "random" (default): v uniform on the lattice in every cycle. Valid (eq. 3.9), but appendix C
      warns that some links are then visited far below average even after hundreds of cycles.
  "luscher": appendix C, "a purely random choice of v is followed, in the next few update cycles,
      by translations along a diagonal lattice direction in regular steps": one uniform random v,
      then n - 1 cycles translated by v = (b_0, ..., b_{D-1}) / n with
      n = min{10, b_0, ..., b_{D-1}} / 2 (eq. C.2), then a new random v. For b = 8^4: n = 4, three
      steps v = (2, 2, 2, 2) (eq. C.1). Our choices where the paper leaves one: when b_mu / n is not
      an integer the paper allows "close-by integer vectors"; we take the nearest integer, halves
      rounded up (`luscher_translation_step`), the same step in every regular cycle; the paper
      writes D = 4, and other D use the same formula over all D directions.
  "odometer": ours, not in the paper: one random v, then the prod(b) - 1 block-relative offsets
      (`block_offset`), so that over a period every link is active exactly a_mu / prod(b) of
      the cycles. The paper's scheme is balanced only on average: in one period of n cycles 4.7%
      of the links are never active for 8^4 blocks (33% for 6^4, none for cubic blocks 10^4 to
      20^4), although the visit counts per period vary less than with n random vectors (variance
      0.42 against 0.93 for 8^4; tests).
The translation is a symmetry ONLY WITHOUT a boundary-condition defect: `DDHMCHamiltonian` takes
no plaquette weights, and `DDHMC.cycle` raises if the gauge term carries some
(`check_translation_invariant`); translation=None is the only option with a defect, and then the
fixed grid alone is not ergodic.
"""

import math

import torch

from neoqcd.fermions._gauge import mean_plaquette, plaquette_fields
from neoqcd.fermions.algebra import unitarize
from neoqcd.fermions.dirac import WilsonDirac
from neoqcd.fermions.hmc import HMC, ActionTerm, Hamiltonian, WilsonGaugeAction, random_momenta
from neoqcd.fermions.pseudofermion import ON_UNCONVERGED, TwoFlavourWilson, convergence_limit, hopping_derivative
from neoqcd.fermions.solver import solve

__all__ = ["BlockDecomposition", "BlockWilsonDirac", "DDOperators", "BlockTwoFlavour", "SchurTwoFlavour",
           "MaskedActionTerm", "DDHMCHamiltonian", "DDHMC", "ddhmc_step", "random_translation", "translate",
           "random_shifts", "block_offset", "luscher_translation_step", "check_translation_invariant",
           "TRANSLATIONS"]

TRANSLATIONS = ("random", "luscher", "odometer", None)


class BlockDecomposition:
    """Blocks, colours, boundaries and link masks (module doc). Masks are B-independent.

    Attributes (shapes: `lattice` for site masks, (D,) + lattice for link masks, bool):
        block_size, n_blocks: tuples per direction; block_coords (D,) + lattice; block_index
        (lattice, flat block number); colour (0 black / Omega, 1 white / Omega*); black, white;
        interior_boundary; exterior_boundary (the code's "dOmega*": white sites with a black
        neighbour = the paper's dOmega, colour-mirrored naming, module doc);
        inblock_links, cut_links, active_links.

    `device`: where the bool masks are built (default CPU). `field_mask` / `link_mask` of one of
    these attributes return a broadcastable real mask cached per (attribute, device, real dtype),
    so the trajectory does no host-to-device copy; the cached tensors are shared: never modify
    them in place. Any other mask tensor is converted on every call (no cache).
    """

    SITE_MASKS = ("black", "white", "interior_boundary", "exterior_boundary")
    LINK_MASKS = ("inblock_links", "cut_links", "active_links")

    def __init__(self, geometry, block_size, device=None):
        self._mask_cache = {}
        D = geometry.D
        if isinstance(block_size, int):
            block_size = (block_size,) * D
        block_size = tuple(int(b) for b in block_size)
        if len(block_size) != D:
            raise ValueError(f"block_size needs {D} entries, got {block_size}")
        for mu, (n, b) in enumerate(zip(geometry.lattice, block_size)):
            if b < 2 or b % 2:
                raise ValueError(f"block size must be even and >= 2, got {b} in direction {mu}")
            if n % b:
                raise ValueError(f"extent {n} is not a multiple of the block size {b} in direction {mu}")
            if (n // b) % 2:
                raise ValueError(f"the number of blocks {n // b} in direction {mu} must be even (chessboard)")
        self.geometry = geometry
        self.block_size = block_size
        self.n_blocks = tuple(n // b for n, b in zip(geometry.lattice, block_size))
        grids = torch.meshgrid(*[torch.arange(n, device=device) for n in geometry.lattice], indexing="ij")
        self.block_coords = torch.stack([g // b for g, b in zip(grids, block_size)])
        local = torch.stack([g % b for g, b in zip(grids, block_size)])
        index = torch.zeros(geometry.lattice, dtype=torch.long, device=device)
        for mu in range(D):
            index = index * self.n_blocks[mu] + self.block_coords[mu]
        self.block_index = index
        self.colour = self.block_coords.sum(0) % 2
        self.black = self.colour == 0
        self.white = self.colour == 1
        on_face = [(local[mu] == 0) | (local[mu] == block_size[mu] - 1) for mu in range(D)]
        self.interior_boundary = torch.stack(on_face).any(0)
        # exterior boundary of the black blocks: white sites with a black nearest neighbour
        nb_black = torch.zeros_like(self.black)
        for mu in range(D):
            nb_black = nb_black | torch.roll(self.black, 1, mu) | torch.roll(self.black, -1, mu)
        self.exterior_boundary = self.white & nb_black
        # U_mu at n joins n and n - 1 (array index) along mu: cut iff local y_mu = 0
        self.cut_links = torch.stack([local[mu] == 0 for mu in range(D)])
        self.inblock_links = ~self.cut_links
        active = []
        for mu in range(D):
            both_boundary = self.interior_boundary & torch.roll(self.interior_boundary, 1, mu)
            active.append(self.inblock_links[mu] & ~both_boundary)
        self.active_links = torch.stack(active)

    # ---- counts ------------------------------------------------------------------------------
    @property
    def active_fraction(self):
        return self.active_links.double().mean().item()

    def expected_active_fraction(self):
        """sum_mu (b_mu - 1)[b_mu > 2] prod_{nu != mu}(b_nu - 2) / (D prod b) (module doc)."""
        b = self.block_size
        D = len(b)
        total = 0
        for mu in range(D):
            a = (b[mu] - 1) if b[mu] > 2 else 0
            for nu in range(D):
                if nu != mu:
                    a *= b[nu] - 2
            total += a
        return total / (D * math.prod(b))

    # ---- broadcastable masks -----------------------------------------------------------------
    @property
    def device(self):
        return self.black.device

    def _broadcast(self, mask, like, names):
        """mask (lattice or (D,) + lattice) as a real (1, mask shape, 1, 1) tensor on like's device;
        cached when `mask` IS one of the attributes `names` (identity, not equality: a temporary
        such as ~active_links is never cached)."""
        dtype = like.real.dtype
        for name in names:
            if mask is getattr(self, name, None):
                key = (name, str(like.device), dtype)
                out = self._mask_cache.get(key)
                if out is None:
                    out = mask.reshape((1,) + mask.shape + (1, 1)).to(device=like.device, dtype=dtype)
                    self._mask_cache[key] = out
                return out
        return mask.reshape((1,) + mask.shape + (1, 1)).to(device=like.device, dtype=dtype)

    def field_mask(self, mask, like):
        """Site mask (lattice) as a real (1, lattice, 1, 1) tensor for Wilson fields like `like`
        (cached for the attributes in SITE_MASKS)."""
        return self._broadcast(mask, like, self.SITE_MASKS)

    def link_mask(self, mask, like):
        """Link mask (D, lattice) as a real (1, D, lattice, 1, 1) tensor for links like `like`
        (cached for the attributes in LINK_MASKS)."""
        return self._broadcast(mask, like, self.LINK_MASKS)


class BlockWilsonDirac(WilsonDirac):
    """D_bl = D_Omega (+) D_Omega*: the Wilson operator with the cut links set to zero, i.e. the
    block operators with Dirichlet boundary conditions (both hops across block faces dropped)."""

    def __init__(self, U, kappa, blocks, time_sign=-1.0):
        self.blocks = blocks
        self.U_full = U
        super().__init__(U * blocks.link_mask(blocks.inblock_links, U), kappa, blocks.geometry, time_sign)


class DDOperators:
    """D, D_bl, D_dOmega, D_dOmegaS, R and R^-1 at links U (module doc for the formulas).

    `tol` is the relative tolerance of every solve (CGNE, even-odd). Fields: (B, lattice, S, N).
    Without `solver` a solve whose true residual is above `convergence_limit(tol)` raises and
    `solves` / `iterations` count them. With `solver` (a callable (op, b, tol) -> x, the term's
    `SchurTwoFlavour._solve`, which does the bookkeeping and applies its `on_unconverged` policy)
    every solve goes through it.
    """

    def __init__(self, U, kappa, blocks, time_sign=-1.0, tol=1e-12, max_iter=5000, solver=None):
        self.blocks = blocks
        self.geometry = blocks.geometry
        self.kappa = kappa
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.solver = solver
        self.full = WilsonDirac(U, kappa, blocks.geometry, time_sign)
        self.block = BlockWilsonDirac(U, kappa, blocks, time_sign)
        self.cut = WilsonDirac(U * blocks.link_mask(blocks.cut_links, U), kappa, blocks.geometry, time_sign)
        self.iterations = 0
        self.solves = 0

    def _masks(self, psi):
        b = self.blocks
        return b.field_mask(b.black, psi), b.field_mask(b.white, psi), b.field_mask(b.exterior_boundary, psi)

    def cut_hopping(self, psi):
        """H_cut psi: the hopping across block faces (both directions)."""
        return self.cut.hopping(psi)

    def D_dOmega(self, psi):
        """P_Omega (-kappa H_cut) P_Omega* psi: hops from the white to the black blocks."""
        black, white, _ = self._masks(psi)
        return black * (-self.cut._kappa_view(psi) * self.cut_hopping(white * psi))

    def D_dOmegaS(self, psi):
        """P_Omega* (-kappa H_cut) P_Omega psi: hops from the black to the white blocks."""
        black, white, _ = self._masks(psi)
        return white * (-self.cut._kappa_view(psi) * self.cut_hopping(black * psi))

    def _solve(self, op, b):
        if self.solver is not None:
            return self.solver(op, b, self.tol)
        res = solve(op.even_odd(), b, method="cgne", tol=self.tol, max_iter=self.max_iter)
        self.solves += 1
        self.iterations += int(res.iterations.max().item())
        limit = convergence_limit(self.tol, res.residual.dtype)
        if not bool((res.residual <= limit).all()):
            raise RuntimeError(f"DD solve not converged: residual {res.residual.max().item():.3e} > {limit:.1e}")
        return res.x

    def block_solve(self, b):
        """D_bl^-1 b: all blocks at once (they decouple); D_Omega^-1 on black, D_Omega*^-1 on white."""
        return self._solve(self.block, b)

    def full_solve(self, b):
        """D^-1 b."""
        return self._solve(self.full, b)

    def full_solve_dagger(self, b):
        """D^-dag b = gamma_5 D^-1 gamma_5 b."""
        g5 = self.full.gamma5
        return g5(self._solve(self.full, g5(b)))

    def one_minus_K(self, psi):
        """(1 - D_Omega*^-1 D_dOmegaS D_Omega^-1 D_dOmega) P_Omega* psi, on the whole of Omega*."""
        _, white, _ = self._masks(psi)
        v = white * psi
        return v - self.block_solve(self.D_dOmegaS(self.block_solve(self.D_dOmega(v))))

    def R(self, psi):
        """R psi = P (1 - K) P psi on dOmega* (two block solves)."""
        _, _, P = self._masks(psi)
        return P * self.one_minus_K(P * psi)

    def R_inverse(self, psi):
        """R^-1 psi = P (psi - D^-1 D_dOmega psi) on dOmega* (one full solve)."""
        _, _, P = self._masks(psi)
        v = P * psi
        return P * (v - self.full_solve(self.D_dOmega(v)))


class BlockTwoFlavour(TwoFlavourWilson):
    """S_bl = phi^dag (D_bl_hat^dag D_bl_hat)^-1 phi: TwoFlavourWilson with the block operator
    (one global solve covers all blocks), force restricted to the active links."""

    def __init__(self, kappa, blocks, time_sign=-1.0, **kwargs):
        self.blocks = blocks
        super().__init__(kappa, blocks.geometry, time_sign=time_sign, **kwargs)

    def wilson(self, U):
        return BlockWilsonDirac(U, self.kappa, self.blocks, self.time_sign)

    def force(self, U):
        return self.blocks.link_mask(self.blocks.active_links, U) * super().force(U)


class SchurTwoFlavour(ActionTerm):
    """S_R = |R^-1 phi_R|^2 with phi_R on dOmega* (module doc): refresh phi_R = R eta (two block
    solves at action_tol), action with one full solve (action_tol), force on the active links with
    two full solves (solver_tol). No chronological guess: every solve starts from zero.

    Bookkeeping and convergence policy as `TwoFlavourWilson` (the same methods): `stats` holds
    refresh_calls, action_calls, force_calls, solves, cg_iterations (sum over solves of the largest
    per-sample count), cg_iterations_per_sample ((B,) totals), last_iterations, last_residual
    (the TRUE relative residual |b - D x| / |b| of the full or block system, per sample) and
    unconverged_solves; `reset_stats()` zeroes them. After every solve the residual is compared
    with `convergence_limit(tol)` = 10 max(tol, 1e3 eps): `on_unconverged` "raise" (default) raises
    a RuntimeError naming the kind ("refresh": the block solves of R eta; "action"; "force"),
    "warn" warns once per kind, "ignore" only counts. Every solve goes through `_solve(op, b, tol)`
    (looked up on the instance at call time, so a wrapper such as main_hmc's SolveTimer sees it).
    """

    _count = TwoFlavourWilson._count
    _check_converged = TwoFlavourWilson._check_converged

    def __init__(self, kappa, blocks, time_sign=-1.0, solver_tol=1e-10, action_tol=None, max_iter=5000,
                 on_unconverged="raise"):
        if on_unconverged not in ON_UNCONVERGED:
            raise ValueError(f"on_unconverged must be one of {ON_UNCONVERGED}, got {on_unconverged!r}")
        self.on_unconverged = on_unconverged
        self._warned = set()
        self._solve_kind = None
        self.kappa = kappa
        self.blocks = blocks
        self.geometry = blocks.geometry
        self.time_sign = float(time_sign)
        self.solver_tol = float(solver_tol)
        self.action_tol = min(self.solver_tol, 1e-12) if action_tol is None else float(action_tol)
        self.max_iter = int(max_iter)
        self.phi = None
        self._refresh_links = None
        self._refresh_action = None
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"refresh_calls": 0, "action_calls": 0, "force_calls": 0, "solves": 0,
                      "cg_iterations": 0, "cg_iterations_per_sample": None, "last_iterations": None,
                      "last_residual": None, "unconverged_solves": 0}

    def _solve(self, op, b, tol):
        """D^-1 b for a `WilsonDirac` op (the full or the block operator): even-odd CGNE from zero,
        then the bookkeeping and the convergence check on the true residual of D x = b."""
        res = solve(op.even_odd(), b, method="cgne", tol=tol, max_iter=self.max_iter)
        self._count(res, tol, self._solve_kind or "solve")
        return res.x

    def operators(self, U, tol):
        return DDOperators(U, self.kappa, self.blocks, self.time_sign, tol=tol, max_iter=self.max_iter,
                           solver=lambda op, b, t: self._solve(op, b, t))

    def _as(self, kind, fn, *args):
        self._solve_kind = kind
        try:
            return fn(*args)
        finally:
            self._solve_kind = None

    @property
    def n_components(self):
        g = self.geometry
        return int(self.blocks.exterior_boundary.sum().item()) * g.spin * g.N

    def _draw_eta(self, U, generator):
        eta = torch.randn(self.geometry.wilson_shape(U.shape[0]), generator=generator, dtype=U.dtype,
                          device=U.device)
        return self.blocks.field_mask(self.blocks.exterior_boundary, eta) * eta

    def _set_refresh(self, U, eta):
        self._refresh_links = U
        self._refresh_version = U._version
        self._refresh_action = (eta.conj() * eta).real.reshape(eta.shape[0], -1).sum(1)
        self.stats["refresh_calls"] += 1

    def refresh(self, U, generator=None):
        """phi_R = R eta, eta complex Gaussian on dOmega*; S_R(U) = eta^dag eta."""
        ops = self.operators(U, self.action_tol)
        eta = self._draw_eta(U, generator)
        self.phi = self._as("refresh", ops.R, eta)
        self._set_refresh(U, eta)

    def chi(self, U, tol):
        if self.phi is None:
            raise RuntimeError("phi_R is not set: call refresh(U, generator) first")
        ops = self.operators(U, tol)
        return ops, ops.R_inverse(self.phi)

    def action(self, U):
        self.stats["action_calls"] += 1
        _, chi = self._as("action", self.chi, U, self.action_tol)
        return (chi.conj() * chi).real.reshape(chi.shape[0], -1).sum(1)

    def action_difference(self, U_old, U_new):
        if (U_old is self._refresh_links and self._refresh_action is not None
                and U_old._version == self._refresh_version):
            S_old = self._refresh_action
        else:
            S_old = self.action(U_old)
        return self.action(U_new) - S_old

    def force(self, U):
        """F_R = -2 kappa hopping_derivative(D, zeta, x) on the active links (module doc)."""
        if self.phi is None:
            raise RuntimeError("phi_R is not set: call refresh(U, generator) first")
        self.stats["force_calls"] += 1
        ops = self.operators(U, self.solver_tol)
        P = self.blocks.field_mask(self.blocks.exterior_boundary, self.phi)
        x = self._as("force", ops.full_solve, ops.D_dOmega(self.phi))
        chi = P * (self.phi - x)
        zeta = self._as("force", ops.full_solve_dagger, chi)
        G = hopping_derivative(ops.full, zeta, x)
        k = ops.full.kappa.view((-1,) + (1,) * (G.dim() - 1)).to(G.dtype)
        return self.blocks.link_mask(self.blocks.active_links, U) * (-2.0 * k * G)


class MaskedActionTerm(ActionTerm):
    """An action term whose force is restricted to the links of `link_mask` ((D,) + lattice bool)."""

    def __init__(self, term, blocks, link_mask=None):
        self.term = term
        self.blocks = blocks
        self.mask = blocks.active_links if link_mask is None else link_mask

    def action(self, U):
        return self.term.action(U)

    def action_difference(self, U_old, U_new):
        return self.term.action_difference(U_old, U_new)

    def refresh(self, U, generator=None):
        return self.term.refresh(U, generator)

    def force(self, U):
        return self.blocks.link_mask(self.mask, U) * self.term.force(U)


class DDHMCHamiltonian(Hamiltonian):
    """Levels: [Schur term] (outermost), [block term] x n_block_substeps, [masked gauge] x
    n_gauge_substeps. All forces vanish off the active links. `block_kwargs` go to
    `BlockTwoFlavour` (i.e. `TwoFlavourWilson`: solver_tol, action_tol, max_iter, reuse_guess,
    even_odd, on_unconverged, ...), `schur_kwargs` to `SchurTwoFlavour` (solver_tol, action_tol,
    max_iter, on_unconverged). The hierarchy is the paper's: the Schur (global) force is the
    smallest and the most expensive, so it has the largest step (the HMC n_steps), the block force
    the middle one, the gauge force the smallest. No
    `plaquette_weights`: the gauge term has no defect, as required by the translations of
    `DDHMC.cycle` (`check_translation_invariant`)."""

    def __init__(self, beta, kappa, blocks, n_block_substeps=1, n_gauge_substeps=1, time_sign=-1.0,
                 block_kwargs=None, schur_kwargs=None):
        self.blocks = blocks
        self.schur = SchurTwoFlavour(kappa, blocks, time_sign=time_sign, **(schur_kwargs or {}))
        self.block = BlockTwoFlavour(kappa, blocks, time_sign=time_sign, **(block_kwargs or {}))
        self.gauge = MaskedActionTerm(WilsonGaugeAction(beta, blocks.geometry), blocks)
        super().__init__([([self.schur], 1), ([self.block], n_block_substeps), ([self.gauge], n_gauge_substeps)])


def translate(U, shifts):
    """Roll chain b of U by shifts[b] ((B, D) integers): U'(n) = U(n - shifts[b]) (torch.roll)."""
    B, D = U.shape[0], U.shape[1]
    out = []
    for b in range(B):
        s = tuple(int(v) for v in shifts[b].tolist())
        out.append(torch.roll(U[b:b + 1], shifts=s, dims=tuple(range(2, 2 + D))))
    return torch.cat(out, dim=0)


def random_shifts(B, extents, generator=None, device=None):
    """(B, D) long: each component uniform in [0, extent)."""
    return torch.stack([torch.randint(0, int(n), (B,), generator=generator, device=device)
                        for n in extents], dim=1)


def random_translation(U, generator=None):
    """Translate every chain by its own random vector, each component uniform in [0, extent).

    Returns (U_translated, shifts (B, D) long). Exact symmetry of the action WITHOUT a boundary
    defect (module doc); `DDHMC.cycle` refuses gauge terms with plaquette weights.
    """
    D = U.shape[1]
    shifts = random_shifts(U.shape[0], U.shape[2:2 + D], generator, U.device)
    return translate(U, shifts), shifts


def block_offset(k, block_size):
    """The k-th block-relative offset in mixed-radix (odometer) order, direction 0 fastest:
    k = 0, ..., prod(b) - 1 enumerates every o with 0 <= o_mu < b_mu exactly once
    (`translation="odometer"`)."""
    o = []
    for b in block_size:
        o.append(k % b)
        k //= b
    return tuple(o)


def luscher_translation_step(block_size, n=None):
    """(n, v) of appendix C of hep-lat/0409106: after a random translation, n - 1 cycles are
    translated by the regular diagonal step v = (b_0, ..., b_{D-1}) / n (eq. C.2), with
    n = min{10, b_0, ..., b_{D-1}} / 2 unless `n` is given; b = 8^4 gives n = 4, v = (2, 2, 2, 2)
    (eq. C.1). When b_mu / n is not an integer v_mu is the nearest integer, halves rounded up (the
    paper: "or close-by integer vectors"). The block sizes are even (`BlockDecomposition`), so the
    default n is an integer >= 1; n = 1 (a block size 2) means no regular steps."""
    b = tuple(int(x) for x in block_size)
    if not b or any(x < 2 or x % 2 for x in b):
        raise ValueError(f"block sizes must be even and >= 2, got {b}")
    n = min(10, *b) // 2 if n is None else int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    v = tuple((2 * x + n) // (2 * n) for x in b)  # floor(b_mu / n + 1/2)
    return n, v


def check_translation_invariant(hamiltonian):
    """Raise if the gauge term carries plaquette weights (a defect): then a translation is not a
    symmetry of the target and `DDHMC.cycle` would sample the wrong distribution."""
    gauge = getattr(hamiltonian, "gauge", None)
    term = getattr(gauge, "term", gauge)
    if getattr(term, "plaquette_weights", None) is not None:
        raise ValueError("DD-HMC translations are a symmetry only without a boundary defect: the gauge "
                         "term has plaquette_weights; use DDHMC(..., translation=None) and accept that "
                         "only the active links of the fixed block grid are updated")


class DDHMC(HMC):
    """HMC on the active links of a `DDHMCHamiltonian`: momenta drawn only on the active links,
    inactive links restored exactly after the reunitarisation. `cycle` = translation + trajectory
    (Luescher's update cycle, eq. 3.9).

    translation (module doc, "Translations"; U'(n) = U(n - v)):
        "random" (default): every cycle translates each chain by an independent vector uniform on
            the lattice.
        "luscher": appendix C of hep-lat/0409106. One uniform random vector, then `n_regular`
            cycles translated by the regular diagonal step v = b / n, n = n_regular + 1
            (`luscher_translation_step`), then a new random vector. Default n_regular = n - 1 with
            n = min{10, b_0, ..., b_{D-1}} / 2 (eq. C.2); for 8^4 blocks a random vector and then
            three steps (2, 2, 2, 2) (eq. C.1). Every chain takes its own random vector and the
            same regular steps.
        "odometer": ours, not in the paper. One uniform random vector, then `n_regular` cycles in
            which the total offset from it runs through the block-relative offsets
            o(k) = `block_offset(k, block_size)`, k = 1, ..., n_regular; then a new random vector.
            With n_regular = prod(b) - 1 (its default) a period visits every block-relative
            offset once, so every link is active in exactly a_mu / prod(b) of the cycles of a
            period (tests); the paper's scheme is balanced only on average. A fixed diagonal step
            (1, ..., 1) would not do this (for 4x4 blocks in 2D it activates some links twice as
            often as others), and no single step generates all offsets when the b_mu share a
            factor 2.
        None: no translation (cycle = trajectory); the only option allowed with a defect, and
            then only the active links of the fixed grid move (not ergodic on its own).
    n_regular: the regular cycles after each random vector ("luscher" and "odometer"; ignored
        otherwise). Raises ValueError if the blocks have no active link (a block size 2 in some
        direction, module doc).
    State of the "luscher"/"odometer" sequence: `self._k` (position in the period), `self._offset`
    (B, D) (total offset from the last random vector).
    """

    def __init__(self, geometry, hamiltonian, tau, n_steps, translation="random", n_regular=None, **kwargs):
        super().__init__(geometry, hamiltonian, tau, n_steps, **kwargs)
        if translation not in TRANSLATIONS:
            raise ValueError(f"translation must be one of {TRANSLATIONS}, got {translation!r}")
        self.translation = translation
        blocks = hamiltonian.blocks
        if not bool(blocks.active_links.any()):
            raise ValueError(f"block size {blocks.block_size} has no active links (a block size 2 in some "
                             "direction freezes every link, fig. 2 of hep-lat/0409106): use blocks >= 4 "
                             "in every direction")
        bs = blocks.block_size
        if translation == "odometer":
            self.n_regular = math.prod(bs) - 1 if n_regular is None else int(n_regular)
        else:
            self.n_regular = luscher_translation_step(bs)[0] - 1 if n_regular is None else int(n_regular)
        if self.n_regular < 0:
            raise ValueError("n_regular must be >= 0")
        self.luscher_step = luscher_translation_step(bs, self.n_regular + 1)[1]
        self._k = 0
        self._offset = None

    def regular_offset(self, k):
        """Total offset from the period's random vector after k regular cycles (k >= 1)."""
        if self.translation == "odometer":
            return block_offset(k, self.hamiltonian.blocks.block_size)
        return tuple(k * v for v in self.luscher_step)

    def _active(self, U):
        return self.hamiltonian.blocks.link_mask(self.hamiltonian.blocks.active_links, U)

    def momenta(self, U, generator=None):
        return self._active(U) * random_momenta(U.shape, generator=generator, dtype=U.dtype, device=U.device)

    @torch.no_grad()
    def trajectory(self, U, generator=None):
        B = U.shape[0]
        self.hamiltonian.refresh(U, generator)
        P = self.momenta(U, generator)
        U1, P1, force_evals = self.integrate(U, P)
        active = self._active(U) > 0
        U1 = torch.where(active, unitarize(U1), U)
        dH = self.hamiltonian.energy_difference(U, P, U1, P1)
        r = torch.rand(B, generator=generator, dtype=dH.dtype, device=dH.device)
        accepted = torch.log(r) < -dH
        U_new = torch.where(accepted.reshape((B,) + (1,) * (U.dim() - 1)), U1, U)
        stats = {"exp_minus_dH": torch.exp(-dH), "acceptance": accepted.to(dH.dtype).mean().item(),
                 "force_evals": force_evals,
                 "plaquette": mean_plaquette(plaquette_fields(U_new, self.geometry.D), self.geometry.N)}
        return U_new, accepted, dH, stats

    def next_shifts(self, B, generator=None, device=None):
        """(B, D) shifts to apply in the next cycle (advances the "luscher"/"odometer" sequence;
        a new batch size B restarts it with a random vector)."""
        extents = self.geometry.lattice
        if self.translation == "random":
            return random_shifts(B, extents, generator, device)
        if self.translation is None:
            return torch.zeros(B, len(extents), dtype=torch.long, device=device)
        if self._k == 0 or self._offset is None or self._offset.shape[0] != B:
            self._k = 0
            shifts = random_shifts(B, extents, generator, device)
            self._offset = torch.zeros(B, len(extents), dtype=torch.long, device=device)
        else:
            new = torch.tensor(self.regular_offset(self._k), dtype=torch.long, device=device).expand(B, -1)
            shifts = new - self._offset
            self._offset = new.clone()
        self._k = (self._k + 1) % (self.n_regular + 1)
        return shifts

    def cycle(self, U, generator=None):
        """Translation of the gauge field (`translation`), then one trajectory on the active links.
        stats["shifts"] holds the (B, D) shifts applied (U'(n) = U(n - shift))."""
        if self.translation is not None:
            check_translation_invariant(self.hamiltonian)
        shifts = self.next_shifts(U.shape[0], generator, U.device)
        if self.translation is not None:
            U = translate(U, shifts)
        U_new, accepted, dH, stats = self.trajectory(U, generator)
        stats["shifts"] = shifts
        return U_new, accepted, dH, stats


def ddhmc_step(hmc, U, generator=None):
    """One DD-HMC update cycle (eq. 3.9): `hmc.cycle(U, generator)`."""
    return hmc.cycle(U, generator)
