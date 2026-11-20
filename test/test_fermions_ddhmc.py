"""Tests of the domain-decomposition HMC (neoqcd.fermions.ddhmc), Luescher hep-lat/0409106.

CPU, complex128, fixed seeds. Stage 1: block geometry. Stage 2: block operators and the exact
factorisation det D = det D_Omega det D_Omega* det R with dense matrices. Stage 3: the
pseudofermion terms (stochastic determinant ratios, forces vs finite differences). Stage 4: the
DD-HMC update. Prints behind NEOQCD_TEST_VERBOSE, slow comparisons behind NEOQCD_SLOW=1. The
translations against the paper (eq. 3.9, appendix C) are in test_fermions_ddhmc_translation.py.
"""

import itertools
import math
import os
import unittest
import warnings

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
    ddhmc_step,
    random_translation,
    translate,
)
from neoqcd.fermions.dirac import dense_matrix
from neoqcd.fermions.exact import ExactMetropolis, assemble_hopping, log_det_two_flavour
from neoqcd.fermions.lattice import Geometry, even_odd_masks
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

    def run_case(self, geometry, b, kappa, seed, time_sign=-1.0):
        bd = BlockDecomposition(geometry, b)
        U, _ = haar_links(geometry, 1, seed)
        ops = DDOperators(U, kappa, bd, time_sign=time_sign, tol=1e-14)
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

    def test_time_sign_plus_and_su2(self):
        """Periodic time boundary (time_sign = +1) and SU(2), 2D 4x4 with 2x2 blocks."""
        self.run_case(G2_4, 2, 0.2, 14, time_sign=1.0)
        self.run_case(Geometry(2, 4, 4, 2), 2, 0.2, 15)
        self.run_case(Geometry(2, 4, 4, 2), 2, 0.2, 16, time_sign=1.0)

    def test_per_chain_kappa(self):
        """A (B,) tensor of kappa: every chain's actions and forces equal those of its scalar kappa."""
        bd = BlockDecomposition(G2_8, 4)
        U, gen = haar_links(G2_8, 2, 17)
        kappas = (0.15, 0.2)
        kt = torch.tensor(kappas, dtype=torch.float64)
        for make in (lambda k: SchurTwoFlavour(k, bd, solver_tol=1e-13, action_tol=1e-13),
                     lambda k: BlockTwoFlavour(k, bd, solver_tol=1e-13, action_tol=1e-13, reuse_guess=False)):
            batch = make(kt)
            batch.refresh(U, gen)
            S, F = batch.action(U), batch.force(U)
            for b, k in enumerate(kappas):
                one = make(k)
                one.phi = batch.phi[b:b + 1].clone()
                self.assertLess(abs(one.action(U[b:b + 1]).item() - S[b].item()), 1e-9 * abs(S[b].item()))
                self.assertLess((one.force(U[b:b + 1]) - F[b:b + 1]).abs().max().item(), 1e-9 * F.abs().max().item())

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
        (even_odd=False), S = phi^dag (D_bl^dag D_bl)^-1 phi, and in its even-odd form
        (production default), S = phi^dag (D_hat^dag D_hat)^-1 phi with
        D_hat = 1_ee - kappa^2 H_bl,eo H_bl,oe (made invertible on the full space by 1_oo)."""
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
        even_m, _ = even_odd_masks(G)
        even = site_mask(even_m.reshape(G.lattice))
        odd = 1 - even

        def dense_ops(theta):
            V = links(theta)
            D = eye - kappa * assemble_hopping(V, G)[0]
            Dbl = eye - kappa * assemble_hopping(V * inblock, G)[0]
            Ddo = black[:, None] * (-kappa * assemble_hopping(V * cut, G)[0]) * white[None, :]
            Hbl = assemble_hopping(V * inblock, G)[0]
            Dhat = torch.diag(odd) + torch.diag(even) - kappa ** 2 * (even[:, None] * (Hbl @ (odd[:, None] * Hbl)) * even[None, :])
            return D, Dbl, Ddo, Dhat

        schur = SchurTwoFlavour(kappa, bd, solver_tol=1e-13, action_tol=1e-13)
        block = BlockTwoFlavour(kappa, bd, even_odd=False, solver_tol=1e-13, action_tol=1e-13, reuse_guess=False)
        block_eo = BlockTwoFlavour(kappa, bd, solver_tol=1e-13, action_tol=1e-13, reuse_guess=False)
        for term in (schur, block, block_eo):
            term.refresh(U, gen)
        phi_R = schur.phi.reshape(-1)
        phi_b = block.phi.reshape(-1)
        phi_e = block_eo.phi.reshape(-1)
        self.assertEqual(phi_e[odd.real > 0].abs().max().item(), 0.0)

        def actions(theta):
            D, Dbl, Ddo, Dhat = dense_ops(theta)
            chi = P * (phi_R - torch.linalg.solve(D, Ddo @ phi_R))
            S_R = (chi.conj() * chi).real.sum()
            y = torch.linalg.solve(Dbl.conj().T, phi_b)
            S_b = (y.conj() * y).real.sum()
            z = torch.linalg.solve(Dhat.conj().T, phi_e)
            S_e = (z.conj() * z).real.sum()
            return S_R, S_b, S_e

        theta = torch.zeros(U.shape[:-2] + (N * N - 1,), dtype=torch.float64, requires_grad=True)
        S_R, S_b, S_e = actions(theta)
        self.assertLess(abs(S_R.item() - schur.action(U).item()), 1e-9)
        self.assertLess(abs(S_b.item() - block.action(U).item()), 1e-9)
        self.assertLess(abs(S_e.item() - block_eo.action(U).item()), 1e-9)
        active = bd.link_mask(bd.active_links, U)[..., 0, 0]  # (1, D, lattice)
        for S, term, label in ((S_R, schur, "Schur"), (S_b, block, "block (full lattice)"),
                                 (S_e, block_eo, "block (even-odd)")):
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


class BookkeepingTest(unittest.TestCase):
    """Cached device masks, the TwoFlavourWilson stats keys and the on_unconverged policy of the
    Schur term (2D 8x8 SU(3), 4x4 blocks, kappa = 0.2)."""

    KAPPA = 0.2

    def setUp(self):
        self.bd = BlockDecomposition(G2_8, 4)

    def test_masks_cached_per_device_and_dtype(self):
        bd = self.bd
        U, _ = haar_links(G2_8, 2, 5)
        m = bd.link_mask(bd.active_links, U)
        self.assertIs(m, bd.link_mask(bd.active_links, U))
        self.assertEqual((m.dtype, tuple(m.shape)), (torch.float64, (1, 2) + G2_8.lattice + (1, 1)))
        self.assertTrue(torch.equal(m[0, ..., 0, 0], bd.active_links.double()))
        m32 = bd.link_mask(bd.active_links, U.to(torch.complex64))
        self.assertEqual(m32.dtype, torch.float32)
        self.assertIsNot(m32, m)
        psi = torch.zeros(G2_8.wilson_shape(1), dtype=torch.cdouble)
        f = bd.field_mask(bd.exterior_boundary, psi)
        self.assertIs(f, bd.field_mask(bd.exterior_boundary, psi))
        self.assertTrue(torch.equal(f[0, ..., 0, 0], bd.exterior_boundary.double()))
        n = len(bd._mask_cache)
        inactive = bd.link_mask(~bd.active_links, U)  # a temporary: converted, not cached
        self.assertEqual(len(bd._mask_cache), n)
        self.assertTrue(torch.equal(inactive + m, torch.ones_like(m)))
        # built on the requested device, same masks
        on = BlockDecomposition(G2_8, 4, device=torch.device("cpu"))
        self.assertEqual(on.device, torch.device("cpu"))
        for name in BlockDecomposition.SITE_MASKS + BlockDecomposition.LINK_MASKS:
            self.assertTrue(torch.equal(getattr(on, name), getattr(bd, name)), name)

    def test_schur_stats_match_two_flavour_wilson(self):
        from neoqcd.fermions.pseudofermion import TwoFlavourWilson
        U, gen = haar_links(G2_8, 2, 6)
        schur = SchurTwoFlavour(self.KAPPA, self.bd, solver_tol=1e-11)
        block = BlockTwoFlavour(self.KAPPA, self.bd, solver_tol=1e-11)
        want = set(TwoFlavourWilson(self.KAPPA, G2_8).stats)
        self.assertEqual(set(schur.stats), want)
        self.assertEqual(set(block.stats), want)
        calls = []
        original = schur._solve

        def wrapped(*a, **k):  # what main_hmc's SolveTimer does: an instance attribute
            calls.append(1)
            return original(*a, **k)

        schur._solve = wrapped
        schur.refresh(U, gen)
        self.assertEqual((schur.stats["solves"], schur.stats["refresh_calls"]), (2, 1))  # R eta: two block solves
        schur.action(U)
        self.assertEqual((schur.stats["solves"], schur.stats["action_calls"]), (3, 1))
        schur.force(U)
        self.assertEqual((schur.stats["solves"], schur.stats["force_calls"]), (5, 1))
        self.assertEqual(len(calls), 5)
        s = schur.stats
        self.assertEqual(tuple(s["cg_iterations_per_sample"].shape), (2,))
        self.assertLessEqual(int(s["cg_iterations_per_sample"].max()), s["cg_iterations"])
        self.assertEqual(tuple(s["last_residual"].shape), (2,))
        self.assertLess(s["last_residual"].max().item(), 1e-10)
        self.assertEqual(s["unconverged_solves"], 0)
        schur.reset_stats()
        self.assertEqual(schur.stats["solves"], 0)

    def test_on_unconverged_policy(self):
        from neoqcd.fermions.pseudofermion import UnconvergedSolveWarning
        U, gen = haar_links(G2_8, 2, 7)
        with self.assertRaises(ValueError):
            SchurTwoFlavour(self.KAPPA, self.bd, on_unconverged="maybe")
        term = SchurTwoFlavour(self.KAPPA, self.bd, max_iter=2)
        with self.assertRaisesRegex(RuntimeError, "SchurTwoFlavour refresh: CG not converged"):
            term.refresh(U, gen)
        term = SchurTwoFlavour(self.KAPPA, self.bd, max_iter=2, on_unconverged="warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            term.refresh(U, gen)
            term.action(U)
            term.force(U)
            term.force(U)
        kinds = [str(x.message).split(":")[0] for x in w if issubclass(x.category, UnconvergedSolveWarning)]
        self.assertEqual(kinds, ["SchurTwoFlavour refresh", "SchurTwoFlavour action", "SchurTwoFlavour force"])
        self.assertEqual(term.stats["unconverged_solves"], 2 + 1 + 4)
        term = SchurTwoFlavour(self.KAPPA, self.bd, max_iter=2, on_unconverged="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            term.refresh(U, gen)
            term.action(U)
        self.assertEqual(term.stats["unconverged_solves"], 3)
        self.assertGreater(term.stats["last_residual"].max().item(), 1e-6)
        # through the Hamiltonian's kwargs, and the standalone operators still raise
        ham = DDHMCHamiltonian(5.0, self.KAPPA, self.bd, schur_kwargs={"max_iter": 2, "on_unconverged": "ignore"},
                               block_kwargs={"max_iter": 2, "on_unconverged": "ignore"})
        ham.refresh(U, gen)
        self.assertEqual(ham.schur.stats["unconverged_solves"], 2)
        ops = DDOperators(U, self.KAPPA, self.bd, tol=1e-12, max_iter=2)
        with self.assertRaisesRegex(RuntimeError, "DD solve not converged"):
            ops.full_solve(torch.randn(G2_8.wilson_shape(2), dtype=torch.cdouble, generator=gen))


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

    def test_4d_trajectory_su2(self):
        """One 4D trajectory, 8^4 SU(2), 4^4 blocks, kappa = 0.12: reversibility, inactive links
        unchanged (~8 s)."""
        G = Geometry(4, 8, 8, 2)
        bd = BlockDecomposition(G, 4)
        U, gen = haar_links(G, 1, 52)
        ham = dd_hamiltonian(bd, 2.0, 0.12, tol=1e-12, n_block=1, n_gauge=1)
        hmc = DDHMC(G, ham, tau=0.5, n_steps=2)
        ham.refresh(U, gen)
        P = hmc.momenta(U, gen)
        U1, P1, _ = hmc.integrate(U, P)
        U2, P2, _ = hmc.integrate(U1, -P1)
        inactive = (bd.link_mask(~bd.active_links, U) > 0).expand_as(U)
        self.assertEqual((U1 - U)[inactive].abs().max().item(), 0.0)
        self.assertGreater((U1 - U).abs().max().item(), 0.05)
        self.assertLess((U2 - U).abs().max().item(), 1e-9)
        self.assertLess((P2 + P).abs().max().item(), 1e-9)

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
            f"Schur solves {ham.schur.stats['solves']}, iterations/solve {ham.schur.stats['cg_iterations'] / ham.schur.stats['solves']:.1f}")
        self.assertLess(abs(m - 1.0), 4 * e)
        self.assertGreater(acc, 0.7)


class DDHMCUpdateTest(unittest.TestCase):
    """Metropolis step, `cycle` / `ddhmc_step`, coverage of the translations. 2D 8x8 SU(3), 4x4
    blocks, beta = 5.0, kappa = 0.2, a coarse step (tau = 1.5, one step) so that some chains are
    rejected."""

    BETA, KAPPA = 5.0, 0.2

    def setUp(self):
        self.bd = BlockDecomposition(G2_8, 4)

    def make(self, translation="random", **kw):
        ham = dd_hamiltonian(self.bd, self.BETA, self.KAPPA, tol=1e-10, n_block=1, n_gauge=1)
        return ham, DDHMC(G2_8, ham, tau=1.5, n_steps=1, translation=translation, **kw)

    def inactive(self, U):
        return (self.bd.link_mask(~self.bd.active_links, U) > 0).expand_as(U)

    def test_metropolis_step(self):
        B = 8
        gen = torch.Generator().manual_seed(81)
        ham, hmc = self.make()
        U = SUN_identity(G2_8.link_shape(B)[:-1], dtype=torch.cdouble)
        U, _, _, _ = hmc.cycle(U, gen)
        state = gen.get_state()
        U_new, acc, dH, st = hmc.trajectory(U, gen)
        # replay the random stream: refresh, momenta, then the Metropolis uniforms
        g2 = torch.Generator()
        g2.set_state(state)
        ham.refresh(U, g2)
        hmc.momenta(U, g2)
        r = torch.rand(B, generator=g2, dtype=dH.dtype)
        self.assertTrue(torch.equal(acc, torch.log(r) < -dH))
        self.assertTrue(bool(acc.any()) and bool((~acc).any()), msg=f"{acc.tolist()} {dH.tolist()}")
        for b in range(B):
            if acc[b]:
                self.assertGreater((U_new[b] - U[b]).abs().max().item(), 1e-3)
            else:
                self.assertTrue(torch.equal(U_new[b], U[b]))
        self.assertEqual(st["acceptance"], acc.double().mean().item())
        p = mean_plaquette(plaquette_fields(U_new, 2), 3)
        self.assertTrue(torch.equal(st["plaquette"], p))
        say(f"Metropolis: accepted {acc.tolist()}, dH {[round(x, 3) for x in dH.tolist()]}")

    def check_cycle_output(self, U_in, U_new, acc, st):
        self.assertEqual(set(st), {"exp_minus_dH", "acceptance", "force_evals", "plaquette", "shifts"})
        self.assertEqual(tuple(st["shifts"].shape), (U_in.shape[0], 2))
        # links in SU(N)
        eye = torch.eye(3, dtype=U_new.dtype)
        self.assertLess((U_new @ U_new.conj().transpose(-1, -2) - eye).abs().max().item(), 1e-12)
        self.assertLess((torch.linalg.det(U_new) - 1).abs().max().item(), 1e-12)
        # the translation comes before the trajectory: rejected chains and the inactive links of
        # all chains equal the TRANSLATED input
        Ut = translate(U_in, st["shifts"])
        inact = self.inactive(U_new)
        self.assertTrue(torch.equal(U_new[inact], Ut[inact]))
        for b in range(U_in.shape[0]):
            if not acc[b]:
                self.assertTrue(torch.equal(U_new[b], Ut[b]))

    def test_cycle_and_step(self):
        B = 8
        gen = torch.Generator().manual_seed(82)
        ham, hmc = self.make()
        U = SUN_haar(G2_8.link_shape(B)[:-1], generator=gen)
        moved = False
        for i in range(3):
            step = hmc.cycle if i % 2 == 0 else (lambda V, g: ddhmc_step(hmc, V, g))
            U_new, acc, dH, st = step(U, gen)
            self.check_cycle_output(U, U_new, acc, st)
            moved |= bool((st["shifts"] != 0).any())
            U = U_new
        self.assertTrue(moved)
        # the random shifts are uniform on the lattice, not on the block grid: every residue
        # modulo the block size occurs (a grid-only translation never changes the active set)
        s = hmc.next_shifts(256, gen)
        for mu in range(2):
            self.assertEqual(set((s[:, mu] % self.bd.block_size[mu]).tolist()), set(range(self.bd.block_size[mu])))
            self.assertEqual(set(s[:, mu].tolist()), set(range(G2_8.lattice[mu])))

    def test_cycle_refuses_defect(self):
        ham, hmc = self.make()
        w = {(0, 1): torch.ones(G2_8.lattice, dtype=torch.float64)}
        ham.gauge.term.plaquette_weights = w
        U = SUN_identity(G2_8.link_shape(1)[:-1], dtype=torch.cdouble)
        with self.assertRaises(ValueError):
            hmc.cycle(U)
        hmc.translation = None  # no translation: allowed, shifts are zero
        U_new, _, _, st = hmc.cycle(U, torch.Generator().manual_seed(1))
        self.assertEqual(int(st["shifts"].abs().sum()), 0)
        with self.assertRaises(ValueError):
            DDHMC(G2_8, ham, tau=1.0, n_steps=1, translation="diagonal")

    @staticmethod
    def coverage(bd, positions):
        """Number of positions s at which each link of the ORIGINAL field is active: after
        U'(n) = U(n - s) the link at x sits at x + s."""
        D = bd.active_links.shape[0]
        count = torch.zeros(bd.active_links.shape, dtype=torch.long)
        for s in positions:
            count += torch.roll(bd.active_links, shifts=tuple(-int(v) for v in s), dims=tuple(range(1, D + 1))).long()
        return count

    def test_coverage_all_translations(self):
        """Over all translations every link is active in exactly active_fraction x (number of
        translations) of them (cubic blocks: the same fraction in every direction); with shifts
        restricted to multiples of the block size it fails (the grid does not move)."""
        for g, b in ((G2_8, 4), (Geometry(2, 12, 12, 3), 6), (Geometry(4, 8, 8, 2), 4)):
            bd = BlockDecomposition(g, b)
            all_s = list(itertools.product(*[range(n) for n in g.lattice]))
            c = self.coverage(bd, all_s)
            expect = bd.active_fraction * len(all_s)
            self.assertAlmostEqual(expect, round(expect), places=9)
            self.assertTrue(bool((c == round(expect)).all()), msg=f"{g.lattice} {b}")
            grid = [s for s in all_s if all(v % bd.block_size[mu] == 0 for mu, v in enumerate(s))]
            cg = self.coverage(bd, grid)
            self.assertFalse(bool((cg == cg.reshape(-1)[0]).all()))
            # a fixed diagonal step (1, ..., 1) over b cycles is not uniform either
            diag = [tuple([k] * g.D) for k in range(bd.block_size[0])]
            cd = self.coverage(bd, diag)
            self.assertFalse(bool((cd == cd.reshape(-1)[0]).all()))

    def test_odometer_translation(self):
        """translation="odometer" (ours, not the paper's): one random vector, then the
        block-relative offsets; after a full period of prod(b) cycles every link has been active
        the same number of times. The paper's scheme ("luscher") is tested in
        test_fermions_ddhmc_translation.py."""
        B = 3
        ham, hmc = self.make(translation="odometer")
        self.assertEqual(hmc.n_regular, 15)
        gen = torch.Generator().manual_seed(83)
        period = hmc.n_regular + 1
        pos = torch.zeros(B, 2, dtype=torch.long)
        positions = [[] for _ in range(B)]
        firsts = []
        for k in range(2 * period):
            s = hmc.next_shifts(B, gen)
            if k % period == 0:
                pos = s.clone()
                firsts.append(s.clone())
            else:
                pos = pos + s
            for b in range(B):
                positions[b].append(tuple(pos[b].tolist()))
        self.assertFalse(torch.equal(firsts[0], firsts[1]))  # a new random vector every period
        expect = self.bd.active_fraction * period
        for b in range(B):
            for half in (positions[b][:period], positions[b][period:]):
                c = self.coverage(self.bd, half)
                self.assertTrue(bool((c == round(expect)).all()))
        # a shorter period (n_regular = 3) is accepted and restarts after 4 cycles
        _, h2 = self.make(translation="odometer", n_regular=3)
        seq = [h2.next_shifts(1, gen) for _ in range(5)]
        self.assertEqual([tuple(x[0].tolist()) for x in seq[1:4]], [(1, 0), (1, 0), (1, 0)])
        # the cycle applies the sequence
        U = SUN_identity(G2_8.link_shape(1)[:-1], dtype=torch.cdouble)
        _, h3 = self.make(translation="odometer")
        U1, acc, _, st = h3.cycle(U, gen)
        self.check_cycle_output(U, U1, acc, st)


class DDHMCExactComparisonTest(unittest.TestCase):
    """DD-HMC, with the random and with the paper's ("luscher") translations, against the plain
    N_f = 2 HMC, the exact-determinant Metropolis and quenched HMC, 2D 8x8 SU(3), 4x4 blocks,
    beta = 5.0, kappa = 0.22, B = 32 chains from a cold start (as ExactComparisonTest of
    test_fermions_hmc_fermions.py, on the smallest 2D lattice with active links). Observables:
    plaquette and log det(D^dagger D) (exact, dense). Slow: NEOQCD_SLOW=1."""

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

    @unittest.skipUnless(SLOW, "~20 min on one CPU thread: NEOQCD_SLOW=1")
    def test_kappa_022(self):
        bd = BlockDecomposition(G2_8, 4)
        dd = {}
        for translation, seed in (("random", 300), ("luscher", 310)):
            hmc = DDHMC(G2_8, dd_hamiltonian(bd, self.BETA, self.KAPPA, tol=1e-10), tau=1.0, n_steps=2,
                        translation=translation)
            dd[translation] = self.run_hmc(hmc, True, 30, 120, seed)
        plain = H.HMC(G2_8, TwoFlavourHamiltonian(self.BETA, self.KAPPA, G2_8, n_gauge_substeps=3), tau=1.0,
                      n_steps=2)
        (p_h, e_ph), (l_h, e_lh), _ = self.run_hmc(plain, False, 15, 60, 500)
        quenched = H.HMC(G2_8, H.WilsonGaugeHamiltonian(self.BETA, G2_8), tau=1.0, n_steps=2)
        (p_q, e_pq), (l_q, e_lq), _ = self.run_hmc(quenched, False, 15, 60, 600)
        (p_m, e_pm), (l_m, e_lm), acc_m = self.run_metropolis(40, 60, 400)
        msg = (f"2D 8x8 SU(3), 4x4 blocks, beta={self.BETA} kappa={self.KAPPA}:\n"
               f"  <P>       HMC {p_h:.4f}({e_ph:.4f})  exact {p_m:.4f}({e_pm:.4f})  quenched {p_q:.4f}({e_pq:.4f})\n"
               f"  <log det> HMC {l_h:.4f}({e_lh:.4f})  exact {l_m:.4f}({e_lm:.4f})  quenched {l_q:.4f}({e_lq:.4f})\n"
               f"  Metropolis acceptance {acc_m:.3f}")
        for translation, ((p_d, e_pd), (l_d, e_ld), (x, e_x)) in dd.items():
            msg += (f"\n  DD-HMC ({translation}): <P> {p_d:.4f}({e_pd:.4f})  <log det> {l_d:.4f}({e_ld:.4f})  "
                    f"<exp(-dH)> = {x:.4f}({e_x:.4f}); "
                    f"|l_dd - l_exact| / |l_exact - l_quenched| = {abs(l_d - l_m) / abs(l_m - l_q):.3f}")
        say(msg)
        self.assertGreater(abs(l_m - l_q), 4 * math.hypot(e_lm, e_lq), msg=msg)
        for (p_d, e_pd), (l_d, e_ld), (x, e_x) in dd.values():
            self.assertLess(abs(x - 1.0), 4 * e_x, msg=msg)
            for a, ea in ((p_h, e_ph), (p_m, e_pm)):
                self.assertLess(abs(p_d - a), 4 * math.hypot(e_pd, ea), msg=msg)
            for a, ea in ((l_h, e_lh), (l_m, e_lm)):
                self.assertLess(abs(l_d - a), 4 * math.hypot(e_ld, ea), msg=msg)
            self.assertLess(abs(l_d - l_m), 0.3 * abs(l_m - l_q), msg=msg)


if __name__ == "__main__":
    unittest.main()
