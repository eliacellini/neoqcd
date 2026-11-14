"""Tests of the N_f = 2 pseudofermion action (neoqcd.fermions.pseudofermion) and the exact
dense-determinant tools (neoqcd.fermions.exact). Phase 1 of fermions_plan.md.

CPU, complex128, fixed seeds. Lattices: 2^3 x 4 SU(3) (even-odd dimension 192), 2D 4 x 4 SU(3)
(48) and 2D 4 x 4 SU(2) (32). Haar-random links are the strong-coupling limit; kappa = 0.12 in
4D and 0.20 in 2D are below the critical values there (see test_fermions_solver), so every CG
converges in a few tens of iterations.
"""

import math
import unittest
import warnings

import torch

from neoqcd.fermions._gauge import SUN_dagger, SUN_haar, SUN_mul, SUN_trace
from neoqcd.fermions.algebra import algebra_basis, expi, random_algebra, trace_product
from neoqcd.fermions.dirac import WilsonDirac, dense_matrix
from neoqcd.fermions.exact import assemble_even_odd, assemble_hopping, log_det_two_flavour
from neoqcd.fermions.hmc import link_sum
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.pseudofermion import (
    TwoFlavourWilson,
    UnconvergedSolveWarning,
    convergence_limit,
    hopping_derivative,
)

G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4, SU(3)
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4, SU(3)
G2_SU2 = Geometry(2, 4, 4, 2)  # 2D 4 x 4, SU(2)
C = torch.complex128


def haar_links(geometry, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=gen), gen


def replicate(U, K):
    """K copies of a single gauge field (batch 1) as a batch of K."""
    return U.expand((K,) + tuple(U.shape[1:])).contiguous()


def fd_derivative(f, U, X, t=1e-3):
    """4th-order central difference of f(exp(i s X) U) at s = 0."""
    g = lambda s: f(SUN_mul(expi(s * X), U))
    return (-g(2 * t) + 8 * g(t) - 8 * g(-t) + g(-2 * t)) / (12 * t)


class ExactToolsTest(unittest.TestCase):
    """The batched dense assembly from the links agrees with dense_matrix of the sparse operators."""

    def test_assembly_matches_dense_matrix(self):
        for geo, kappa, seed in ((G4, 0.12, 1), (G2, 0.20, 2), (G2_SU2, 0.20, 3)):
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    U, _ = haar_links(geo, 2, seed)
                    k = torch.tensor([kappa, 0.7 * kappa], dtype=torch.float64)
                    W = WilsonDirac(U, k, geo, time_sign)
                    A_eo = assemble_even_odd(U, k, geo, time_sign)
                    H = assemble_hopping(U, geo, time_sign=time_sign)
                    for b in range(2):
                        M_eo = dense_matrix(W.even_odd(), geo, batch_index=b)
                        M = dense_matrix(W, geo, batch_index=b)
                        self.assertLess((A_eo[b] - M_eo).abs().max().item(), 1e-14)
                        D_full = torch.eye(M.shape[0], dtype=C) - k[b] * H[b]
                        self.assertLess((D_full - M).abs().max().item(), 1e-14)
                        # log det(D^dagger D) = 2 log|det D| = 2 log|det D_hat|
                        ref = 2.0 * torch.linalg.slogdet(M)[1]
                        for method in ("dense_matrix", "assembled"):
                            ld = log_det_two_flavour(U, k, geo, time_sign, method=method)
                            self.assertLess(abs(ld[b].item() - ref.item()), 1e-10)


class RefreshTest(unittest.TestCase):
    """1. phi = D_hat^dagger eta: S_pf(U) = eta^dagger eta sample by sample, so <S_pf> = n and
    Var(S_pf) = n (a sum of n unit exponentials), n the number of complex components."""

    def run_case(self, geo, kappa, K, even_odd):
        U1, gen = haar_links(geo, 1, 10 + geo.D + int(even_odd))
        U = replicate(U1, K)
        S = TwoFlavourWilson(kappa, geo, even_odd=even_odd, action_tol=1e-13)
        S.refresh(U, gen)
        n = S.n_components
        expect_n = geo.volume * geo.spin * geo.N // (2 if even_odd else 1)
        self.assertEqual(n, expect_n)
        if even_odd:  # phi lives on even sites
            odd = S.operator(U).odd(S.phi)
            self.assertEqual(odd.abs().max().item(), 0.0)
        s = S.action(U)
        # the solve reproduces eta^dagger eta to solver precision, sample by sample
        self.assertLess(((s - S._refresh_action).abs() / s).max().item(), 1e-11)
        mean, var = s.mean().item(), s.var().item()
        self.assertLess(abs(mean - n), 4.0 * math.sqrt(n / K), msg=f"<S_pf> = {mean}, n = {n}")
        # Var of the sample variance of a sum of n Exp(1): ~ (Var + 2 n^2) / K, Var = n (approx)
        self.assertLess(abs(var - n), 4.0 * math.sqrt((6.0 * n + 2.0 * n * n) / K), msg=f"Var = {var}")
        print(f"\n{geo} even_odd={even_odd}: <S_pf> = {mean:.2f} +- {math.sqrt(n / K):.2f}, n = {n}")

    def test_refresh_distribution(self):
        self.run_case(G4, 0.12, 500, True)
        self.run_case(G2, 0.20, 4000, True)
        self.run_case(G2, 0.20, 2000, False)


class StochasticEstimatorTest(unittest.TestCase):
    """2. <exp(-[S_pf(U') - S_pf(U)])>_{phi ~ exp(-S_pf(U))} = det(D^dagger D)(U') / det(D^dagger D)(U).

    U' = exp(i c X) U with X the unit-norm average of the pseudofermion force over 16 independent
    phi at U (only a choice of direction, along which log det changes fast: for a random X the
    determinant ratio is within a few per cent of 1, comparable with the statistical error).
    c = +-0.2 gives ratios ~0.4 and ~2, both far from 1: the check is discriminating.
    The full-lattice variant must give the same ratio (4D only).
    """

    def run_case(self, geo, kappa, K, seed, variants):
        U1, gen = haar_links(geo, 1, seed)
        probe = TwoFlavourWilson(kappa, geo)
        U16 = replicate(U1, 16)
        probe.refresh(U16, gen)
        X1 = probe.force(U16).mean(0, keepdim=True)
        X1 = X1 / math.sqrt(link_sum(trace_product(X1, X1)).item())
        U = replicate(U1, K)
        for c in (0.2, -0.2):
            V1 = SUN_mul(expi(c * X1), U1)
            ld = log_det_two_flavour(torch.cat((U1, V1)), kappa, geo)
            ratio = math.exp((ld[1] - ld[0]).item())
            V = replicate(V1, K)
            for even_odd in variants:
                with self.subTest(geometry=geo, c=c, even_odd=even_odd):
                    S = TwoFlavourWilson(kappa, geo, even_odd=even_odd)
                    S.refresh(U, gen)
                    w = torch.exp(-S.action_difference(U, V))
                    est, err = w.mean().item(), w.std().item() / math.sqrt(K)
                    print(f"\n{geo} c={c:+.1f} even_odd={even_odd}: det ratio exact {ratio:.4f}, "
                          f"stochastic {est:.4f} +- {err:.4f}")
                    self.assertGreater(abs(ratio - 1.0), 10 * err)  # discriminating
                    self.assertLess(abs(est - ratio), 4 * err)

    def test_determinant_ratio_4d(self):
        self.run_case(G4, 0.12, 400, 21, (True, False))

    def test_determinant_ratio_2d(self):
        self.run_case(G2, 0.20, 1000, 22, (True,))


class ForceTest(unittest.TestCase):
    """3. tr(F X) = d/dt S_pf(exp(i t X) U)|_0 at fixed phi, against 4th-order finite differences of
    `action`, along random algebra directions (all links) and single-link generators."""

    LINKS = {4: [(0, 0, 0, 0, 0, 0), (1, 1, 2, 1, 0, 1), (0, 3, 3, 1, 1, 0), (1, 2, 0, 0, 1, 1)],
             2: [(0, 0, 0, 0), (1, 1, 3, 2), (0, 1, 1, 0), (1, 0, 2, 3)]}

    def cases(self):
        k2 = torch.tensor([0.20, 0.17], dtype=torch.float64)
        k4 = torch.tensor([0.12, 0.10], dtype=torch.float64)
        yield "4D eo", G4, k4, True, -1.0
        yield "4D full", G4, k4, False, -1.0
        yield "2D eo", G2, k2, True, -1.0
        yield "2D full", G2, k2, False, -1.0
        yield "2D eo periodic", G2, k2, True, 1.0
        yield "2D SU(2) eo", G2_SU2, k2, True, -1.0

    def test_force_against_finite_differences(self):
        for i, (name, geo, kappa, even_odd, time_sign) in enumerate(self.cases()):
            with self.subTest(case=name):
                U, gen = haar_links(geo, 2, 30 + i)
                S = TwoFlavourWilson(kappa, geo, time_sign=time_sign, even_odd=even_odd,
                                     solver_tol=1e-13, action_tol=1e-13)
                S.refresh(U, gen)
                F = S.force(U)
                self.assertEqual(F.shape, U.shape)
                self.assertLess((F - SUN_dagger(F)).abs().max().item(), 1e-13)
                self.assertLess(SUN_trace(F).abs().max().item(), 1e-13)
                for _ in range(2):
                    X = random_algebra(U.shape, generator=gen)
                    pred = link_sum(trace_product(F, X))
                    fd = fd_derivative(S.action, U, X)
                    rel = ((pred - fd).abs() / fd.abs()).max().item()
                    self.assertLess(rel, 1e-6, msg=f"pred={pred} fd={fd}")
                T = algebra_basis(geo.N)
                worst = 0.0
                for link in self.LINKS[geo.D]:
                    b = link[0]
                    for a in range(geo.N * geo.N - 1):
                        X = torch.zeros_like(U)
                        X[link] = T[a]
                        pred = trace_product(F[link], T[a]).item()
                        fd = fd_derivative(S.action, U, X)[b].item()
                        err = abs(pred - fd) / max(abs(fd), 1e-3)
                        worst = max(worst, err)
                        self.assertLess(abs(pred - fd), 1e-6 * max(abs(fd), 1.0), msg=f"link={link} a={a}")
                print(f"\n{name}: worst single-link relative deviation {worst:.1e}")

    def test_hopping_derivative_bilinear(self):
        """Re[a^dagger H b] for arbitrary fields: the building block on its own, 4D and 2D."""
        for geo, seed in ((G4, 41), (G2, 42)):
            with self.subTest(geometry=geo):
                U, gen = haar_links(geo, 2, seed)
                a = torch.randn(geo.wilson_shape(2), dtype=C, generator=gen)
                b = torch.randn(geo.wilson_shape(2), dtype=C, generator=gen)
                f = lambda V: (a.conj() * WilsonDirac(V, 0.1, geo).hopping(b)).real.reshape(2, -1).sum(1)
                G = hopping_derivative(WilsonDirac(U, 0.1, geo), a, b)
                for _ in range(2):
                    X = random_algebra(U.shape, generator=gen)
                    pred = link_sum(trace_product(G, X))
                    fd = fd_derivative(f, U, X)
                    self.assertLess(((pred - fd).abs() / fd.abs()).max().item(), 1e-9)


class ForceExpectationTest(unittest.TestCase):
    """4. <tr(F X)>_phi = -d/dt log det(D^dagger D)(exp(i t X) U): the pseudofermion force is an
    unbiased estimator of the exact two-flavour force. 2^3 x 4, the exact side from dense
    determinants (finite differences of slogdet)."""

    def test_force_expectation_matches_log_det_gradient(self):
        geo, kappa, K = G4, 0.12, 300
        U1, gen = haar_links(geo, 1, 51)
        for direction in ("random", "force"):
            with self.subTest(direction=direction):
                self.check_direction(geo, kappa, K, U1, gen, direction)

    def check_direction(self, geo, kappa, K, U1, gen, direction):
        if direction == "random":
            X1 = random_algebra(U1.shape, generator=gen)
        else:  # unit-norm average force of 16 independent phi: a direction where log det moves fast
            probe = TwoFlavourWilson(kappa, geo)
            probe.refresh(replicate(U1, 16), gen)
            X1 = probe.force(replicate(U1, 16)).mean(0, keepdim=True)
            X1 = 10.0 * X1 / math.sqrt(link_sum(trace_product(X1, X1)).item())
        exact = -fd_derivative(lambda V: log_det_two_flavour(V, kappa, geo), U1, X1).item()
        S = TwoFlavourWilson(kappa, geo)
        U = replicate(U1, K)
        S.refresh(U, gen)
        proj = link_sum(trace_product(S.force(U), X1))
        est, err = proj.mean().item(), proj.std().item() / math.sqrt(K)
        print(f"\n{direction} X: <tr(F X)> = {est:.3f} +- {err:.3f}, -d log det(D^dag D) = {exact:.3f}")
        if direction == "force":
            self.assertGreater(abs(exact), 10 * err)  # discriminating
        self.assertLess(abs(est - exact), 4 * err)



class ConvergencePolicyTest(unittest.TestCase):
    """on_unconverged of the reference term: raise (default), warn once per kind, ignore."""

    def setUp(self):
        self.U, _ = haar_links(G4, 2, 90)
        self.U2, _ = haar_links(G4, 2, 91)

    def _term(self, **kw):
        term = TwoFlavourWilson(0.12, G4, max_iter=2, **kw)
        term.refresh(self.U, torch.Generator().manual_seed(92))
        return term

    def test_limit(self):
        eps = torch.finfo(torch.float64).eps
        self.assertEqual(convergence_limit(1e-8, torch.float64), 1e-7)
        self.assertAlmostEqual(convergence_limit(1e-16, torch.float64), 1e4 * eps)
        self.assertAlmostEqual(convergence_limit(1e-8, torch.float32), 1e4 * torch.finfo(torch.float32).eps)
        with self.assertRaises(ValueError):
            TwoFlavourWilson(0.12, G4, on_unconverged="sometimes")

    def test_raise_is_default(self):
        term = self._term()
        self.assertEqual(term.on_unconverged, "raise")
        with self.assertRaisesRegex(RuntimeError, r"TwoFlavourWilson action: CG not converged, relative residual "
                                                  r".*sample\(s\) \[0, 1\], iterations \[2, 2\], max_iter 2"):
            term.action(self.U2)
        with self.assertRaisesRegex(RuntimeError, "TwoFlavourWilson force: CG not converged"):
            term.force(self.U2)
        self.assertEqual(term.stats["unconverged_solves"], 2)
        # equal tolerances: still named by the calling method
        same = self._term(solver_tol=1e-10, action_tol=1e-10)
        with self.assertRaisesRegex(RuntimeError, "force: CG not converged"):
            same.force(self.U2)
        # converged solves pass
        ok = TwoFlavourWilson(0.12, G4, solver_tol=1e-12)
        ok.refresh(self.U, torch.Generator().manual_seed(92))
        ok.action(self.U2)
        ok.force(self.U2)
        self.assertEqual(ok.stats["unconverged_solves"], 0)

    def test_warn_once_per_kind_and_ignore(self):
        term = self._term(on_unconverged="warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                term.action(self.U2)
                term.force(self.U2)
        msgs = [str(w.message) for w in caught if issubclass(w.category, UnconvergedSolveWarning)]
        self.assertEqual(len(msgs), 2)
        self.assertTrue(msgs[0].startswith("TwoFlavourWilson action: CG not converged"))
        self.assertTrue(msgs[1].startswith("TwoFlavourWilson force: CG not converged"))
        self.assertEqual(term.stats["unconverged_solves"], 4)
        quiet = self._term(on_unconverged="ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quiet.action(self.U2)
        self.assertEqual(quiet.stats["unconverged_solves"], 1)

    def test_count_without_tolerance_uses_looser_tolerance(self):
        """Subclasses that call `_count(result)` themselves get the check at max(solver_tol, action_tol)."""
        from neoqcd.fermions.solver import SolverResult
        term = TwoFlavourWilson(0.12, G4, solver_tol=1e-6, action_tol=1e-12)
        res = lambda r: SolverResult(torch.zeros(1), torch.tensor([3]), torch.tensor([r], dtype=torch.float64))
        term._count(res(5e-6))  # above action_tol, below 10 solver_tol
        with self.assertRaisesRegex(RuntimeError, "TwoFlavourWilson solve: CG not converged"):
            term._count(res(2e-5))
        with self.assertRaisesRegex(RuntimeError, "not converged"):
            term._count(res(float("nan")))


if __name__ == "__main__":
    unittest.main()
