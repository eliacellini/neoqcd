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

Factorisation (eq. 3.5, white Schur complement). With the Schur complement on Omega*,
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

Active links (Luescher, fig. 2): links with both endpoints in the same block and at most one
endpoint on the block's interior boundary (sites with y_nu in {0, b_nu - 1} for some nu). In
direction mu the in-block link at local y (y_mu >= 1) joins y and y - e_mu; both are boundary
points iff some transverse y_nu (nu != mu) is in {0, b_nu - 1}, or b_mu = 2. Count per block and
direction: a_mu = (b_mu - 1) [b_mu > 2] prod_{nu != mu} (b_nu - 2), fraction
sum_mu a_mu / (D prod b). For b = 8^4: 7 * 6^3 / 8^4 = 0.369; for 2^D blocks there are none.
Only active links are evolved in a trajectory (momenta zero elsewhere, forces masked); the gauge
field is translated by a random vector before every trajectory (eq. 3.9), an exact symmetry of
the action (the antiperiodic twist can be moved by psi -> -psi on a slab), so that all links are
updated on average.
"""

import math

import torch

from neoqcd.fermions.algebra import unitarize
from neoqcd.fermions.dirac import WilsonDirac
from neoqcd.fermions.hmc import HMC, ActionTerm, Hamiltonian, WilsonGaugeAction, random_momenta
from neoqcd.fermions.pseudofermion import TwoFlavourWilson, convergence_limit, hopping_derivative
from neoqcd.fermions.solver import solve

__all__ = ["BlockDecomposition", "BlockWilsonDirac", "DDOperators", "BlockTwoFlavour", "SchurTwoFlavour",
           "MaskedActionTerm", "DDHMCHamiltonian", "DDHMC", "random_translation"]


class BlockDecomposition:
    """Blocks, colours, boundaries and link masks (module doc). Masks are B-independent.

    Attributes (shapes: `lattice` for site masks, (D,) + lattice for link masks, bool):
        block_size, n_blocks: tuples per direction; block_coords (D,) + lattice; block_index
        (lattice, flat block number); colour (0 black / Omega, 1 white / Omega*); black, white;
        interior_boundary; exterior_boundary (dOmega*: white sites with a black neighbour);
        inblock_links, cut_links, active_links.
    """

    def __init__(self, geometry, block_size, device=None):
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
    def field_mask(self, mask, like):
        """Site mask (lattice) as a real (1, lattice, 1, 1) tensor for Wilson fields like `like`."""
        return mask.reshape((1,) + mask.shape + (1, 1)).to(device=like.device, dtype=like.real.dtype)

    def link_mask(self, mask, like):
        """Link mask (D, lattice) as a real (1, D, lattice, 1, 1) tensor for links like `like`."""
        return mask.reshape((1,) + mask.shape + (1, 1)).to(device=like.device, dtype=like.real.dtype)


class BlockWilsonDirac(WilsonDirac):
    """D_bl = D_Omega (+) D_Omega*: the Wilson operator with the cut links set to zero, i.e. the
    block operators with Dirichlet boundary conditions (both hops across block faces dropped)."""

    def __init__(self, U, kappa, blocks, time_sign=-1.0):
        self.blocks = blocks
        self.U_full = U
        super().__init__(U * blocks.link_mask(blocks.inblock_links, U), kappa, blocks.geometry, time_sign)


class DDOperators:
    """D, D_bl, D_dOmega, D_dOmegaS, R and R^-1 at links U (module doc for the formulas).

    `tol` is the relative tolerance of every solve (CGNE, even-odd); a solve whose true residual
    is above `convergence_limit(tol)` raises. Fields: (B, lattice, S, N).
    """

    def __init__(self, U, kappa, blocks, time_sign=-1.0, tol=1e-12, max_iter=5000):
        self.blocks = blocks
        self.geometry = blocks.geometry
        self.kappa = kappa
        self.tol = float(tol)
        self.max_iter = int(max_iter)
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
    """S_R = |R^-1 phi_R|^2 with phi_R on dOmega* (module doc): refresh phi_R = R eta, action with
    one full solve, force on the active links with two full solves."""

    def __init__(self, kappa, blocks, time_sign=-1.0, solver_tol=1e-10, action_tol=None, max_iter=5000):
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
        self.stats = {"solves": 0, "iterations": 0}

    def operators(self, U, tol):
        return DDOperators(U, self.kappa, self.blocks, self.time_sign, tol=tol, max_iter=self.max_iter)

    def _book(self, ops):
        self.stats["solves"] += ops.solves
        self.stats["iterations"] += ops.iterations

    @property
    def n_components(self):
        g = self.geometry
        return int(self.blocks.exterior_boundary.sum().item()) * g.spin * g.N

    def refresh(self, U, generator=None):
        """phi_R = R eta, eta complex Gaussian on dOmega*; S_R(U) = eta^dag eta."""
        ops = self.operators(U, self.action_tol)
        eta = torch.randn(self.geometry.wilson_shape(U.shape[0]), generator=generator, dtype=U.dtype,
                          device=U.device)
        eta = self.blocks.field_mask(self.blocks.exterior_boundary, eta) * eta
        self.phi = ops.R(eta)
        self._book(ops)
        self._refresh_links = U
        self._refresh_version = U._version
        self._refresh_action = (eta.conj() * eta).real.reshape(eta.shape[0], -1).sum(1)

    def chi(self, U, tol):
        if self.phi is None:
            raise RuntimeError("phi_R is not set: call refresh(U, generator) first")
        ops = self.operators(U, tol)
        return ops, ops.R_inverse(self.phi)

    def action(self, U):
        ops, chi = self.chi(U, self.action_tol)
        self._book(ops)
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
        ops = self.operators(U, self.solver_tol)
        P = self.blocks.field_mask(self.blocks.exterior_boundary, self.phi)
        x = ops.full_solve(ops.D_dOmega(self.phi))
        chi = P * (self.phi - x)
        zeta = ops.full_solve_dagger(chi)
        self._book(ops)
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
    n_gauge_substeps. All forces vanish off the active links. Extra kwargs go to both fermion terms
    (solver_tol, action_tol, max_iter)."""

    def __init__(self, beta, kappa, blocks, n_block_substeps=1, n_gauge_substeps=1, time_sign=-1.0,
                 block_kwargs=None, schur_kwargs=None):
        self.blocks = blocks
        self.schur = SchurTwoFlavour(kappa, blocks, time_sign=time_sign, **(schur_kwargs or {}))
        self.block = BlockTwoFlavour(kappa, blocks, time_sign=time_sign, **(block_kwargs or {}))
        self.gauge = MaskedActionTerm(WilsonGaugeAction(beta, blocks.geometry), blocks)
        super().__init__([([self.schur], 1), ([self.block], n_block_substeps), ([self.gauge], n_gauge_substeps)])


def random_translation(U, generator=None):
    """Translate every chain by its own random vector, each component uniform in [0, extent).

    Returns (U_translated, shifts (B, D) long). Exact symmetry of the full action (module doc).
    """
    B, D = U.shape[0], U.shape[1]
    extents = U.shape[2:2 + D]
    shifts = torch.stack([torch.randint(0, int(n), (B,), generator=generator, device=U.device)
                          for n in extents], dim=1)
    out = []
    for b in range(B):
        s = tuple(int(v) for v in shifts[b].tolist())
        out.append(torch.roll(U[b:b + 1], shifts=s, dims=tuple(range(2, 2 + D))))
    return torch.cat(out, dim=0), shifts


class DDHMC(HMC):
    """HMC on the active links of a `DDHMCHamiltonian`: momenta drawn only on the active links,
    inactive links restored exactly after the reunitarisation. `cycle` = random translation +
    trajectory (Luescher's update cycle)."""

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
                 "force_evals": force_evals}
        return U_new, accepted, dH, stats

    def cycle(self, U, generator=None):
        """Random translation of the gauge field, then one trajectory on the active links."""
        U, shifts = random_translation(U, generator)
        U_new, accepted, dH, stats = self.trajectory(U, generator)
        stats["shifts"] = shifts
        return U_new, accepted, dH, stats


def ddhmc_step(hmc, U, generator=None):
    """One DD-HMC update cycle (eq. 3.9): `hmc.cycle(U, generator)`."""
    return hmc.cycle(U, generator)
