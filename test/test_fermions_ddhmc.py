"""Tests of the domain-decomposition HMC (neoqcd.fermions.ddhmc), Luescher hep-lat/0409106.

CPU, complex128, fixed seeds. Stage 1: block geometry. Stage 2: block operators and the exact
factorisation det D = det D_Omega det D_Omega* det R with dense matrices. Stage 3: the
pseudofermion terms (stochastic determinant ratios, forces vs finite differences). Stage 4: the
DD-HMC update. Prints behind NEOQCD_TEST_VERBOSE, slow comparisons behind NEOQCD_SLOW=1.
"""

import math
import os
import unittest

import torch

from neoqcd.fermions import hmc as H
from neoqcd.fermions._gauge import SUN_haar, SUN_identity, SUN_mul, mean_plaquette, plaquette_fields
from neoqcd.fermions.algebra import expi, random_algebra, trace_product
from neoqcd.fermions.ddhmc import (
    BlockDecomposition,
    BlockTwoFlavour,
    DDHMC,
    DDHMCHamiltonian,
    DDOperators,
    SchurTwoFlavour,
    random_translation,
)
from neoqcd.fermions.dirac import dense_matrix
from neoqcd.fermions.exact import ExactMetropolis, assemble_hopping, log_det_two_flavour
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.pseudofermion import TwoFlavourHamiltonian

VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))
SLOW = os.environ.get("NEOQCD_SLOW") == "1"

G4 = Geometry(4, 4, 4, 2)  # 4^4 SU(2), 2^4 blocks (SU(2) keeps the dense 4D checks fast)
G2_8 = Geometry(2, 8, 8, 3)  # 2D 8x8 SU(3), 4x4 blocks (the smallest 2D case with active links)
G2_4 = Geometry(2, 4, 4, 3)  # 2D 4x4 SU(3), 2x2 blocks


def haar_links(geometry, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=gen), gen


def say(msg):
    if VERBOSE:
        print("\n" + msg)


class _Wrapped:
    """`apply` = mask * fn(mask * psi), with `support` for dense_matrix (batch 1 links)."""

    def __init__(self, fn, geometry, support=None, dtype=torch.cdouble):
        self.fn = fn
        self.geometry = geometry
        self.support = support
        self.dtype = dtype
        self.device = torch.device("cpu")

    def apply(self, psi):
        if self.support is None:
            return self.fn(psi)
        m = self.support.reshape((1,) + self.support.shape + (1, 1)).to(psi.real.dtype)
        return m * self.fn(m * psi)


# ---------------------------------------------------------------------------------------------
# Stage 1: geometry
# ---------------------------------------------------------------------------------------------
class GeometryTest(unittest.TestCase):

    def check(self, geometry, block_size):
        bd = BlockDecomposition(geometry, block_size)
        D = geometry.D
        # Omega and Omega* partition the lattice, blocks have the right number of sites
        self.assertTrue(bool((bd.black ^ bd.white).all()))
        n_blocks = math.prod(bd.n_blocks)
        counts = torch.bincount(bd.block_index.reshape(-1), minlength=n_blocks)
        self.assertTrue(bool((counts == math.prod(bd.block_size)).all()))
        self.assertEqual(int(bd.black.sum()), geometry.volume // 2)
        # every face neighbour of a site in another block has the other colour (chessboard)
        for mu in range(D):
            for s in (1, -1):
                other = torch.roll(bd.block_index, s, mu) != bd.block_index
                self.assertTrue(bool((torch.roll(bd.colour, s, mu)[other] != bd.colour[other]).all()))
        # the cut links are exactly those joining two blocks (U_mu at n joins n and n - 1)
        for mu in range(D):
            joins = torch.roll(bd.block_index, 1, mu) != bd.block_index
            self.assertTrue(torch.equal(joins, bd.cut_links[mu]))
        # dOmega* = white sites on the interior boundary of their block
        self.assertTrue(torch.equal(bd.exterior_boundary, bd.white & bd.interior_boundary))
        # interior boundary = sites of the block with a neighbour in another block
        has_out = torch.zeros_like(bd.black)
        for mu in range(D):
            for s in (1, -1):
                has_out |= torch.roll(bd.block_index, s, mu) != bd.block_index
        self.assertTrue(torch.equal(has_out, bd.interior_boundary))
        # active links: both ends in one block and at most one end on the interior boundary
        for mu in range(D):
            same = torch.roll(bd.block_index, 1, mu) == bd.block_index
            nb = bd.interior_boundary.int() + torch.roll(bd.interior_boundary, 1, mu).int()
            self.assertTrue(torch.equal(bd.active_links[mu], same & (nb <= 1)))
        self.assertAlmostEqual(bd.active_fraction, bd.expected_active_fraction(), places=14)
        return bd

    def test_cases(self):
        for g, b in ((G4, 2), (G2_8, 4), (G2_4, 2), (Geometry(4, 8, 8, 2), 4), (Geometry(4, 16, 8, 2), (8, 4, 4, 4)),
                     (Geometry(2, 12, 12, 3), 6)):
            bd = self.check(g, b)
            say(f"{g.lattice} blocks {bd.block_size}: active fraction {bd.active_fraction:.4f}, "
                f"|dOmega*| = {int(bd.exterior_boundary.sum())} of {g.volume} sites")
        self.assertEqual(BlockDecomposition(G4, 2).active_fraction, 0.0)
        self.assertAlmostEqual(BlockDecomposition(G2_8, 4).active_fraction, 6 / 16)
        self.assertAlmostEqual(BlockDecomposition(Geometry(4, 16, 16, 2), 8).expected_active_fraction(),
                               7 * 6 ** 3 / 8 ** 4)

    def test_validation(self):
        with self.assertRaises(ValueError):
            BlockDecomposition(G2_8, 3)  # odd
        with self.assertRaises(ValueError):
            BlockDecomposition(Geometry(2, 6, 6, 3), 4)  # not a multiple
        with self.assertRaises(ValueError):
            BlockDecomposition(Geometry(2, 12, 12, 3), 4)  # 3 blocks: no chessboard
        with self.assertRaises(ValueError):
            BlockDecomposition(G2_8, (4, 4, 4))


# ---------------------------------------------------------------------------------------------
# Stage 2: operators and factorisation
# ---------------------------------------------------------------------------------------------
class FactorisationTest(unittest.TestCase):
    CASES = ((G2_4, 2, 0.2, 11), (G2_8, 4, 0.2, 12), (G4, 2, 0.12, 13))

    def run_case(self, geometry, b, kappa, seed):
        bd = BlockDecomposition(geometry, b)
        U, _ = haar_links(geometry, 1, seed)
        ops = DDOperators(U, kappa, bd, tol=1e-14)
        M = dense_matrix(ops.full)
        Mb = dense_matrix(ops.block)
        Mdo = dense_matrix(_Wrapped(ops.D_dOmega, geometry))
        Mdos = dense_matrix(_Wrapped(ops.D_dOmegaS, geometry))
        # D = D_bl + D_dOmega + D_dOmegaS, and D_bl - kappa H_cut
        self.assertLess((M - (Mb + Mdo + Mdos)).abs().max().item(), 1e-14)
        Mcut = dense_matrix(_Wrapped(ops.cut_hopping, geometry))
        self.assertLess((M - (Mb - kappa * Mcut)).abs().max().item(), 1e-14)
        # D_bl does not couple different blocks
        site = bd.block_index.reshape(-1).repeat_interleave(geometry.spin * geometry.N)
        self.assertEqual(Mb[site[:, None] != site[None, :]].abs().max().item(), 0.0)
        ld = torch.linalg.slogdet(M)[1].item()
        ld_black = torch.linalg.slogdet(dense_matrix(_Wrapped(ops.block.apply, geometry, bd.black)))[1].item()
        ld_white = torch.linalg.slogdet(dense_matrix(_Wrapped(ops.block.apply, geometry, bd.white)))[1].item()
        MR = dense_matrix(_Wrapped(ops.R, geometry, bd.exterior_boundary))
        ld_R = torch.linalg.slogdet(MR)[1].item()
        # 1 - K on the whole of Omega*: identity on the white columns outside dOmega*, same det
        M1K = dense_matrix(_Wrapped(ops.one_minus_K, geometry, bd.white))
        white_sites = bd.white.reshape(-1).nonzero().squeeze(-1)
        off = (~bd.exterior_boundary.reshape(-1)[white_sites]).repeat_interleave(geometry.spin * geometry.N)
        eye = torch.eye(M1K.shape[0], dtype=M1K.dtype)
        if bool(off.any()):  # with 2^D blocks every white site is on dOmega*
            self.assertLess((M1K[:, off] - eye[:, off]).abs().max().item(), 1e-13)
        ld_1K = torch.linalg.slogdet(M1K)[1].item()
        # R^-1 with one full solve against the dense inverse of R
        MRinv = dense_matrix(_Wrapped(ops.R_inverse, geometry, bd.exterior_boundary))
        inv_err = (MRinv - torch.linalg.inv(MR)).abs().max().item() / MRinv.abs().max().item()
        err = abs(ld - (ld_black + ld_white + ld_R))
        say(f"{geometry.lattice} blocks {b}^{geometry.D} kappa {kappa}: log|det D| {ld:.12f}, "
            f"black {ld_black:.6f} + white {ld_white:.6f} + R {ld_R:.6f} (dim {MR.shape[0]}), "
            f"mismatch {err:.2e}; log|det(1-K)| - log|det R| {abs(ld_1K - ld_R):.1e}; R^-1 rel. err {inv_err:.1e}")
        self.assertLess(err, 1e-10)
        self.assertLess(abs(ld_1K - ld_R), 1e-10)
        self.assertLess(inv_err, 1e-11)
        # the complex phase too: det D = det D_Omega det D_Omega* det R
        sd, sb = torch.linalg.slogdet(M)[0], torch.linalg.slogdet(Mb)[0]
        sR = torch.linalg.slogdet(MR)[0]
        self.assertLess(abs((sd - sb * sR).item()), 1e-10)

    def test_2d_4x4_blocks_2(self):
        self.run_case(*self.CASES[0])

    def test_2d_8x8_blocks_4(self):
        self.run_case(*self.CASES[1])

    @unittest.skipUnless(SLOW, "4D dense checks take ~80 s: NEOQCD_SLOW=1")
    def test_4d_4x4x4x4_blocks_2(self):
        self.run_case(*self.CASES[2])


# ---------------------------------------------------------------------------------------------
# Stage 3: pseudofermion terms
# ---------------------------------------------------------------------------------------------
def rotate_active(U, bd, eps, gen, n_links=None):
    """U with active links multiplied by exp(i eps X), X Gaussian: all of them, or `n_links` drawn at
    random; the other links are untouched."""
    mask = bd.active_links
    if n_links is not None:
        idx = mask.reshape(-1).nonzero().squeeze(-1)
        idx = idx[torch.randperm(idx.numel(), generator=gen)[:n_links]]
        mask = torch.zeros_like(mask).reshape(-1)
        mask[idx] = True
        mask = mask.reshape(bd.active_links.shape)
    X = random_algebra(U.shape, generator=gen, dtype=U.dtype)
    X = bd.link_mask(mask, U) * X
    return SUN_mul(expi(eps * X), U)


class TermTest(unittest.TestCase):
    """2D 8x8 SU(3), 4x4 blocks, kappa = 0.2, Haar links."""

    KAPPA = 0.2
    B = 1024
    STEP = 0.05  # largest link rotation angle of the move U0 -> U1

    def setUp(self):
        self.bd = BlockDecomposition(G2_8, 4)

    def test_stochastic_determinant_ratios(self):
        """<exp(-(S(U1) - S(U0)))>_{phi ~ exp(-S(U0))} = det M(U1) / det M(U0) for both terms, with
        U1 = exp(-i eps F) U0 moved on the active links only, F the term's own force averaged over 256
        independent pseudofermions (an estimate of -grad log det M; a random direction would leave
        the ratio within the noise of 1). The exact ratios come from dense matrices, and for each
        U1 the two ratios multiply to the full |det D1|^2 / |det D0|^2."""
        bd = self.bd
        U0, gen = haar_links(G2_8, 1, 21)
        B = self.B
        U0b = U0.repeat(B, 1, 1, 1, 1, 1)
        msgs = []
        for name, make in (("block", lambda: BlockTwoFlavour(self.KAPPA, bd, solver_tol=1e-12)),
                           ("Schur", lambda: SchurTwoFlavour(self.KAPPA, bd, solver_tol=1e-12))):
            probe = make()
            probe.refresh(U0.repeat(256, 1, 1, 1, 1, 1), gen)
            F = probe.force(U0.repeat(256, 1, 1, 1, 1, 1)).mean(0, keepdim=True)
            U1 = SUN_mul(expi(-(self.STEP / F.abs().max().item()) * F), U0)
            inactive = (bd.link_mask(~bd.active_links, U0) > 0).expand_as(U0)
            self.assertEqual((U1 - U0)[inactive].abs().max().item(), 0.0)
            exact = {}
            for tag, U in (("0", U0), ("1", U1)):
                ops = DDOperators(U, self.KAPPA, bd, tol=1e-14)
                exact["D" + tag] = 2 * torch.linalg.slogdet(dense_matrix(ops.full))[1].item()
                exact["bl" + tag] = 2 * torch.linalg.slogdet(dense_matrix(ops.block))[1].item()
                MR = dense_matrix(_Wrapped(ops.R, G2_8, bd.exterior_boundary))
                exact["R" + tag] = 2 * torch.linalg.slogdet(MR)[1].item()
            r_bl, r_R, r_D = (exact[k + "1"] - exact[k + "0"] for k in ("bl", "R", "D"))
            self.assertLess(abs(r_D - (r_bl + r_R)), 1e-10)
            ref = r_bl if name == "block" else r_R
            term = make()
            term.refresh(U0b, gen)
            S0 = term.action(U0b)
            self.assertLess((S0 - term._refresh_action).abs().max().item(), 1e-8)
            w = torch.exp(-term.action_difference(U0b, U1.repeat(B, 1, 1, 1, 1, 1)))
            m, e = w.mean().item(), (w.std() / math.sqrt(B)).item()
            msgs.append(f"{name}: log det ratios D {r_D:.6f} = blocks {r_bl:.6f} + R {r_R:.6f}; "
                        f"<exp(-dS)> = {m:.5f}({e:.5f}), exact {math.exp(ref):.5f}, "
                        f"{abs(m - math.exp(ref)) / e:.2f} sigma, exact - 1 = {abs(math.exp(ref) - 1) / e:.1f} sigma; "
                        f"<S(U0)> = {S0.mean().item():.1f} (n = {term.n_components})")
            self.assertLess(abs(m - math.exp(ref)), 4 * e, msg=msgs[-1])
            self.assertGreater(abs(math.exp(ref) - 1.0), 5 * e, msg=msgs[-1])  # the test discriminates
        say("\n".join(msgs))

    def check_force(self, term, U, gen, label, n_links=6, h=2e-3, rtol=1e-6, geometry=G2_8):
        bd = self.bd
        F = term.force(U)
        inactive = (bd.link_mask(~bd.active_links, U) > 0).expand_as(F)
        self.assertEqual(F[inactive].abs().max().item(), 0.0)
        active = bd.active_links.nonzero()
        pick = active[torch.randperm(active.shape[0], generator=gen)[:n_links]]
        worst = 0.0
        for link in pick.tolist():
            T = random_algebra((geometry.N, geometry.N), generator=gen, dtype=U.dtype)
            where = (0,) + tuple(link)

            def S(t):
                V = U.clone()
                V[where] = expi(t * T) @ U[where]
                return term.action(V).item()

            fd = (-S(2 * h) + 8 * S(h) - 8 * S(-h) + S(-2 * h)) / (12 * h)
            an = trace_product(F[where], T).real.item()
            worst = max(worst, abs(fd - an) / max(abs(fd), 1e-8))
        say(f"{label}: force vs 4th-order finite differences on {n_links} active links, max rel. err {worst:.1e}")
        self.assertLess(worst, rtol)

    def test_forces(self):
        U, gen = haar_links(G2_8, 1, 31)
        block = BlockTwoFlavour(self.KAPPA, self.bd, solver_tol=1e-13, action_tol=1e-13, reuse_guess=False)
        schur = SchurTwoFlavour(self.KAPPA, self.bd, solver_tol=1e-13, action_tol=1e-13)
        for term, label in ((block, "block term"), (schur, "Schur term")):
            term.refresh(U, gen)
            self.check_force(term, U, gen, label)
        ham = DDHMCHamiltonian(5.0, self.KAPPA, self.bd)
        self.check_force(ham.gauge, U, gen, "masked gauge term")

    def test_forces_vs_autograd(self):
        """Every active link at once: the forces against autograd of the actions built from dense
        matrices (exact.assemble_hopping, torch.linalg.solve), U(theta) = exp(i theta_a T^a) U with
        an orthonormal basis T^a of su(N), so dS/dtheta_a = tr(F T^a). Schur term
        S_R = |P(phi - D^-1 D_dOmega phi)|^2; block term in its full-lattice form
        (even_odd=False), S = phi^dag (D_bl^dag D_bl)^-1 phi."""
        bd, G, kappa = self.bd, G2_8, self.KAPPA
        U, gen = haar_links(G, 1, 33)
        N = G.N
        basis = []
        for j in range(N):
            for k in range(j + 1, N):
                E = torch.zeros(N, N, dtype=U.dtype)
                E[j, k], E[k, j] = 1.0, 1.0
                basis.append(E / math.sqrt(2.0))
                E = torch.zeros(N, N, dtype=U.dtype)
                E[j, k], E[k, j] = -1j, 1j
                basis.append(E / math.sqrt(2.0))
        for m in range(1, N):
            d = torch.zeros(N, dtype=U.dtype)
            d[:m] = 1.0
            d[m] = -m
            basis.append(torch.diag(d) / math.sqrt(m * (m + 1)))
        basis = torch.stack(basis)  # (N^2 - 1, N, N), tr(T^a T^b) = delta_ab
        gram = torch.einsum("aij,bji->ab", basis, basis)
        self.assertLess((gram - torch.eye(N * N - 1, dtype=gram.dtype)).abs().max().item(), 1e-14)

        def links(theta):
            X = torch.einsum("...a,aij->...ij", theta.to(U.dtype), basis)
            return torch.matmul(torch.linalg.matrix_exp(1j * X), U)

        inblock = bd.link_mask(bd.inblock_links, U)
        cut = bd.link_mask(bd.cut_links, U)
        n = G.volume * G.spin * G.N
        eye = torch.eye(n, dtype=U.dtype)

        def site_mask(m):
            return m.reshape(-1).repeat_interleave(G.spin * G.N).to(U.dtype)

        black, white, P = site_mask(bd.black), site_mask(bd.white), site_mask(bd.exterior_boundary)

        def dense_ops(theta):
            V = links(theta)
            D = eye - kappa * assemble_hopping(V, G)[0]
            Dbl = eye - kappa * assemble_hopping(V * inblock, G)[0]
            Ddo = black[:, None] * (-kappa * assemble_hopping(V * cut, G)[0]) * white[None, :]
            return D, Dbl, Ddo

        schur = SchurTwoFlavour(kappa, bd, solver_tol=1e-13, action_tol=1e-13)
        block = BlockTwoFlavour(kappa, bd, even_odd=False, solver_tol=1e-13, action_tol=1e-13, reuse_guess=False)
        for term in (schur, block):
            term.refresh(U, gen)
        phi_R = schur.phi.reshape(-1)
        phi_b = block.phi.reshape(-1)

        def actions(theta):
            D, Dbl, Ddo = dense_ops(theta)
            chi = P * (phi_R - torch.linalg.solve(D, Ddo @ phi_R))
            S_R = (chi.conj() * chi).real.sum()
            y = torch.linalg.solve(Dbl.conj().T, phi_b)
            S_b = (y.conj() * y).real.sum()
            return S_R, S_b

        theta = torch.zeros(U.shape[:-2] + (N * N - 1,), dtype=torch.float64, requires_grad=True)
        S_R, S_b = actions(theta)
        self.assertLess(abs(S_R.item() - schur.action(U).item()), 1e-9)
        self.assertLess(abs(S_b.item() - block.action(U).item()), 1e-9)
        active = bd.link_mask(bd.active_links, U)[..., 0, 0]  # (1, D, lattice)
        for S, term, label in ((S_R, schur, "Schur"), (S_b, block, "block (full lattice)")):
            g, = torch.autograd.grad(S, theta, retain_graph=True)
            F_auto = torch.einsum("...a,aij->...ij", g.to(U.dtype), basis) * active[..., None, None]
            F = term.force(U)
            err = (F - F_auto).abs().max().item() / F_auto.abs().max().item()
            say(f"{label} force vs autograd on all {int(bd.active_links.sum())} active links: rel. err {err:.1e}")
            self.assertLess(err, 1e-9)

    def test_forces_4d(self):
        """The same force checks in 4D: 8^4 SU(2) with 4^4 blocks (the smallest 4D case with active
        links, active fraction 3 * 2^3 / 4^4 = 0.094), kappa = 0.12, Haar links."""
        G = Geometry(4, 8, 8, 2)
        self.bd = BlockDecomposition(G, 4)
        U, gen = haar_links(G, 1, 32)
        for term, label in ((BlockTwoFlavour(0.12, self.bd, solver_tol=1e-13, action_tol=1e-13, reuse_guess=False),
                             "4D block term"),
                            (SchurTwoFlavour(0.12, self.bd, solver_tol=1e-13, action_tol=1e-13), "4D Schur term")):
            term.refresh(U, gen)
            self.check_force(term, U, gen, label, n_links=4, geometry=G)


# ---------------------------------------------------------------------------------------------
# Stage 4: DD-HMC
# ---------------------------------------------------------------------------------------------
def dd_hamiltonian(bd, beta, kappa, tol=1e-12, n_block=2, n_gauge=2):
    return DDHMCHamiltonian(beta, kappa, bd, n_block_substeps=n_block, n_gauge_substeps=n_gauge,
                            block_kwargs={"solver_tol": tol, "reuse_guess": False},
                            schur_kwargs={"solver_tol": tol})


class DDHMCTest(unittest.TestCase):
    """2D 8x8 SU(3), 4x4 blocks, beta = 5.0, kappa = 0.2."""

    BETA, KAPPA = 5.0, 0.2

    def setUp(self):
        self.bd = BlockDecomposition(G2_8, 4)

    def inactive(self, U):
        return (self.bd.link_mask(~self.bd.active_links, U) > 0).expand_as(U)

    def test_translation_is_a_symmetry(self):
        U, gen = haar_links(G2_8, 3, 41)
        Ut, shifts = random_translation(U, gen)
        self.assertEqual(tuple(shifts.shape), (3, 2))
        ld = log_det_two_flavour(U, self.KAPPA, G2_8, method="assembled")
        ldt = log_det_two_flavour(Ut, self.KAPPA, G2_8, method="assembled")
        self.assertLess((ld - ldt).abs().max().item(), 1e-10)
        p = mean_plaquette(plaquette_fields(U, 2), 3)
        pt = mean_plaquette(plaquette_fields(Ut, 2), 3)
        self.assertLess((p - pt).abs().max().item(), 1e-13)
        self.assertGreater((U - Ut).abs().max().item(), 0.1)

    def test_reversibility_and_frozen_links(self):
        U, gen = haar_links(G2_8, 2, 51)
        ham = dd_hamiltonian(self.bd, self.BETA, self.KAPPA, tol=1e-13)
        hmc = DDHMC(G2_8, ham, tau=0.5, n_steps=4)
        ham.refresh(U, gen)
        P = hmc.momenta(U, gen)
        self.assertEqual(P[self.inactive(P)].abs().max().item(), 0.0)
        U1, P1, _ = hmc.integrate(U, P)
        U2, P2, _ = hmc.integrate(U1, -P1)
        frozen = (U1 - U)[self.inactive(U)].abs().max().item()
        err_U, err_P = (U2 - U).abs().max().item(), (P2 + P).abs().max().item()
        moved = (U1 - U).abs().max().item()
        say(f"reversibility: |U2 - U| {err_U:.1e}, |P2 + P| {err_P:.1e}; inactive links moved by {frozen:.1e}, "
            f"active by up to {moved:.2f}")
        self.assertEqual(frozen, 0.0)
        self.assertGreater(moved, 0.05)
        self.assertLess(err_U, 1e-9)
        self.assertLess(err_P, 1e-9)

    def test_dH_scales_as_eps_squared(self):
        U, gen = haar_links(G2_8, 4, 61)
        ham = dd_hamiltonian(self.bd, self.BETA, self.KAPPA)
        hmc = DDHMC(G2_8, ham, tau=0.5, n_steps=4)
        ham.refresh(U, gen)
        P = hmc.momenta(U, gen)
        dHs = []
        for n in (4, 8, 16):
            U1, P1, _ = hmc.integrate(U, P, n_steps=n)
            dHs.append(ham.energy_difference(U, P, U1, P1).abs().mean().item())
        say(f"|dH| at 4, 8, 16 steps: {dHs}, ratios {dHs[0] / dHs[1]:.2f}, {dHs[1] / dHs[2]:.2f}")
        self.assertTrue(3.0 < dHs[0] / dHs[1] < 5.5 and 3.0 < dHs[1] / dHs[2] < 5.5, msg=str(dHs))

    def test_cycles(self):
        """Short run from a cold start: inactive links frozen in every trajectory, translations
        between them, <exp(-dH)> = 1 within 4 sigma, good acceptance."""
        B = 16
        gen = torch.Generator().manual_seed(71)
        U = SUN_identity(G2_8.link_shape(B)[:-1], dtype=torch.cdouble)
        ham = dd_hamiltonian(self.bd, self.BETA, self.KAPPA, tol=1e-10)
        hmc = DDHMC(G2_8, ham, tau=1.0, n_steps=2)
        x = []
        for i in range(12):
            U, _ = random_translation(U, gen)
            U_new, acc, dH, st = hmc.trajectory(U, gen)
            self.assertTrue(torch.equal(U_new[self.inactive(U)], U[self.inactive(U)]))
            U = U_new
            if i >= 4:
                x.append(st["exp_minus_dH"])
        x = torch.cat(x)
        m, e = x.mean().item(), (x.std() / math.sqrt(x.numel())).item()
        acc = (x.clamp(max=1.0)).mean().item()
        say(f"DD-HMC cycles: <exp(-dH)> = {m:.4f}({e:.4f}), <min(1, exp(-dH))> = {acc:.3f}, "
            f"plaquette {mean_plaquette(plaquette_fields(U, 2), 3).mean().item():.4f}, "
            f"Schur solves {ham.schur.stats['solves']}, iterations/solve {ham.schur.stats['iterations'] / ham.schur.stats['solves']:.1f}")
        self.assertLess(abs(m - 1.0), 4 * e)
        self.assertGreater(acc, 0.7)


class DDHMCExactComparisonTest(unittest.TestCase):
    """DD-HMC against the plain N_f = 2 HMC, the exact-determinant Metropolis and quenched HMC,
    2D 8x8 SU(3), 4x4 blocks, beta = 5.0, kappa = 0.22, B = 32 chains from a cold start (as
    ExactComparisonTest of test_fermions_hmc_fermions.py, on the smallest 2D lattice with active
    links). Observables: plaquette and log det(D^dagger D) (exact, dense). Slow: NEOQCD_SLOW=1."""

    B = 32
    BETA, KAPPA = 5.0, 0.22

    def chain_stats(self, x):
        m = torch.stack(x).mean(0)
        return m.mean().item(), (m.std() / math.sqrt(self.B)).item()

    def observables(self, U):
        return (mean_plaquette(plaquette_fields(U, 2), 3),
                log_det_two_flavour(U, self.KAPPA, G2_8, method="assembled"))

    def run_hmc(self, hmc, cycle, n_therm, n_meas, seed):
        gen = torch.Generator().manual_seed(seed)
        U = SUN_identity(G2_8.link_shape(self.B)[:-1], dtype=torch.cdouble)
        plaq, logdet, expdH = [], [], []
        for i in range(n_therm + n_meas):
            U, _, _, st = hmc.cycle(U, gen) if cycle else hmc.trajectory(U, gen)
            if i >= n_therm:
                p, l = self.observables(U)
                plaq.append(p)
                logdet.append(l)
                expdH.append(st["exp_minus_dH"])
        return self.chain_stats(plaq), self.chain_stats(logdet), self.chain_stats(expdH)

    def run_metropolis(self, n_therm, n_meas, seed):
        gen = torch.Generator().manual_seed(seed)
        m = ExactMetropolis(self.BETA, self.KAPPA, G2_8, eps=0.45, links_per_update=1)
        m.reset(SUN_identity(G2_8.link_shape(self.B)[:-1], dtype=torch.cdouble))
        plaq, logdet = [], []
        for sweep in range(n_therm + n_meas):
            for _ in range(G2_8.D * G2_8.volume):
                m.step(gen)
            if sweep >= n_therm:
                p, l = self.observables(m.U)
                plaq.append(p)
                logdet.append(l)
        return self.chain_stats(plaq), self.chain_stats(logdet), m.acceptance

    @unittest.skipUnless(SLOW, "~15 min on one CPU thread: NEOQCD_SLOW=1")
    def test_kappa_022(self):
        bd = BlockDecomposition(G2_8, 4)
        dd = DDHMC(G2_8, dd_hamiltonian(bd, self.BETA, self.KAPPA, tol=1e-10), tau=1.0, n_steps=2)
        (p_d, e_pd), (l_d, e_ld), (x, e_x) = self.run_hmc(dd, True, 30, 120, 300)
        plain = H.HMC(G2_8, TwoFlavourHamiltonian(self.BETA, self.KAPPA, G2_8, n_gauge_substeps=3), tau=1.0,
                      n_steps=2)
        (p_h, e_ph), (l_h, e_lh), _ = self.run_hmc(plain, False, 15, 60, 500)
        quenched = H.HMC(G2_8, H.WilsonGaugeHamiltonian(self.BETA, G2_8), tau=1.0, n_steps=2)
        (p_q, e_pq), (l_q, e_lq), _ = self.run_hmc(quenched, False, 15, 60, 600)
        (p_m, e_pm), (l_m, e_lm), acc_m = self.run_metropolis(40, 60, 400)
        msg = (f"2D 8x8 SU(3), 4x4 blocks, beta={self.BETA} kappa={self.KAPPA}:\n"
               f"  <P>       DD-HMC {p_d:.4f}({e_pd:.4f})  HMC {p_h:.4f}({e_ph:.4f})  exact {p_m:.4f}({e_pm:.4f})  "
               f"quenched {p_q:.4f}({e_pq:.4f})\n"
               f"  <log det> DD-HMC {l_d:.4f}({e_ld:.4f})  HMC {l_h:.4f}({e_lh:.4f})  exact {l_m:.4f}({e_lm:.4f})  "
               f"quenched {l_q:.4f}({e_lq:.4f})\n"
               f"  DD <exp(-dH)> = {x:.4f}({e_x:.4f}); Metropolis acceptance {acc_m:.3f}; "
               f"|l_dd - l_exact| / |l_exact - l_quenched| = {abs(l_d - l_m) / abs(l_m - l_q):.3f}")
        say(msg)
        self.assertLess(abs(x - 1.0), 4 * e_x, msg=msg)
        for a, ea in ((p_h, e_ph), (p_m, e_pm)):
            self.assertLess(abs(p_d - a), 4 * math.hypot(e_pd, ea), msg=msg)
        for a, ea in ((l_h, e_lh), (l_m, e_lm)):
            self.assertLess(abs(l_d - a), 4 * math.hypot(e_ld, ea), msg=msg)
        self.assertGreater(abs(l_m - l_q), 4 * math.hypot(e_lm, e_lq), msg=msg)
        self.assertLess(abs(l_d - l_m), 0.3 * abs(l_m - l_q), msg=msg)


if __name__ == "__main__":
    unittest.main()
