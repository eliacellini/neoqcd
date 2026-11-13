"""Tests of the RHMC for rooted staggered fermions: multishift CG (neoqcd.fermions.solver), rational
approximations (neoqcd.fermions.rational) and the rooted pseudofermion term, HMC and exact sampler
(neoqcd.fermions.rhmc). Track C3 of fermions_plan.md ("Fase 6": N_f = 2 RHMC).

CPU, complex128, fixed seeds. Lattices: 2D 4 x 4 SU(3) (M_ee dimension 24), 2D 4 x 4 SU(2), 2^3 x 4
SU(3) (48). Exponents gamma = 1/2 (N_f = 2 in 4D, N_f = 1 in 2D) and 1/4 (as n_pf = 2 fields of
gamma = 1/2, or directly); gamma = 1 is checked against StaggeredPseudofermion.
Small masses (SmallMassTest): Zolotarev with exact coefficients and Remez at kappa = 4e4 - 1.6e5,
over-specified degrees, x^{+1/2}; the full default constructors at 4D m = 0.01, 0.02 and 2D
m = 0.01 (~45 s of Remez) run only with NEOQCD_SLOW=1.
Prints behind NEOQCD_TEST_VERBOSE.
"""

import math
import os
import time
import unittest
from fractions import Fraction

import torch

from neoqcd.fermions import hmc as H
from neoqcd.fermions._gauge import (
    SUN_dagger,
    SUN_haar,
    SUN_identity,
    SUN_mul,
    SUN_trace,
    mean_plaquette,
    plaquette_fields,
)
from neoqcd.fermions.algebra import algebra_basis, algebra_components, expi, random_algebra, trace_product
from neoqcd.fermions.hmc import link_sum
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.rational import (
    log_grid,
    max_relative_error,
    rational_approximation,
    remez_approximation,
    zolotarev_inverse_sqrt,
    zolotarev_sqrt,
)
from neoqcd.fermions.rhmc import (
    ExactRootedMetropolis,
    RootedStaggeredHamiltonian,
    RootedStaggeredPseudofermion,
    rational_apply,
    rooted_effective_action,
    staggered_rational_force,
    staggered_spectral_bounds,
)
from neoqcd.fermions.solver import cg, multishift_cg
from neoqcd.fermions.staggered import StaggeredDirac, staggered_dense_matrix
from neoqcd.fermions.staggered_exact import (
    ExactStaggeredMetropolis,
    assemble_staggered_even_odd,
    log_det_staggered,
)
from neoqcd.fermions.staggered_pseudofermion import StaggeredPseudofermion, staggered_hopping_derivative

G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4, SU(3)
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4, SU(3)
G2_SU2 = Geometry(2, 4, 4, 2)  # 2D 4 x 4, SU(2)
C = torch.complex128
MASSES = torch.tensor([0.3, 0.5], dtype=torch.float64)
HALF = Fraction(1, 2)
QUARTER = Fraction(1, 4)

VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))
SLOW = bool(os.environ.get("NEOQCD_SLOW"))


def say(msg):
    if VERBOSE:
        print("\n" + msg)


def haar_links(geometry, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=gen), gen


def replicate(U, K):
    return U.expand((K,) + tuple(U.shape[1:])).contiguous()


def fd_derivative(f, U, X, t=1e-3):
    """4th-order central difference of f(exp(i s X) U) at s = 0."""
    g = lambda s: f(SUN_mul(expi(s * X), U))
    return (-g(2 * t) + 8 * g(t) - 8 * g(-t) + g(-2 * t)) / (12 * t)


def even_sites(geo):
    eo = StaggeredDirac(SUN_identity(geo.link_shape(1)[:-1], dtype=C), 1.0, geo).even_odd()
    return eo.support.reshape(-1).nonzero().squeeze(-1)


def to_dense(field, geo, b):
    """Even-site entries of field[b] as a vector in the order of `staggered_dense_matrix`."""
    return field[b].reshape(geo.volume, geo.N)[even_sites(geo)].reshape(-1)


def force_direction(geo, mass, U1, gen, n=16, norm=1.0):
    """Unit-norm (times `norm`) average N_f = 2^(D/2) pseudofermion force of n independent phi at U1:
    a direction along which log det M_ee changes fast."""
    probe = StaggeredPseudofermion(mass, geo)
    Un = replicate(U1, n)
    probe.refresh(Un, gen)
    X1 = probe.force(Un).mean(0, keepdim=True)
    return norm * X1 / math.sqrt(link_sum(trace_product(X1, X1)).item())


def check_certified(test, approx, n_grid=200001):
    """Certified error >= the dense-grid error (compensated evaluation), and error + rounding >= the
    dense-grid error of the plain float64 evaluation; returns (dense, plain)."""
    grid = log_grid(approx.lower, approx.upper, n_grid)
    dense = max_relative_error(approx, grid)
    plain = max_relative_error(approx, grid, plain=True)
    test.assertLessEqual(dense, approx.error, msg=repr(approx))
    test.assertLessEqual(plain, approx.error + approx.rounding, msg=repr(approx))
    test.assertTrue((approx.shifts >= 0).all())
    return dense, plain


def matrix_function(M, power):
    """Dense M^power (M hermitian positive definite, batched) by eigendecomposition."""
    lam, V = torch.linalg.eigh(M)
    return V @ torch.diag_embed(lam.pow(float(power)).to(M.dtype)) @ V.conj().transpose(-1, -2)


# ---------------------------------------------------------------------------------------------
# 1. multishift CG
# ---------------------------------------------------------------------------------------------

class MultishiftCGTest(unittest.TestCase):
    """All (M_ee + sigma_k) x_k = b from one Krylov sequence agree with independent CG solves on
    M_ee + sigma_k and with dense solves; per-sample mass (different iteration counts), zero shift
    included, per-sample shifts, b = 0 for one sample."""

    SHIFTS = [0.0, 1e-3, 0.05, 1.0, 30.0]

    def test_against_cg_and_dense(self):
        masses = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        for geo, seed in ((G4, 1), (G2, 2)):
            with self.subTest(geometry=geo):
                U, gen = haar_links(geo, 3, seed)
                eo = StaggeredDirac(U, masses, geo).even_odd()
                b = eo.even(torch.randn(geo.staggered_shape(3), dtype=C, generator=gen))
                res = multishift_cg(eo.apply, b, self.SHIFTS, tol=1e-12, max_iter=3000)
                self.assertEqual(tuple(res.x.shape), (len(self.SHIFTS),) + tuple(b.shape))
                self.assertLess(res.residual.max().item(), 2e-12)
                worst_cg, worst_dense = 0.0, 0.0
                for k, s in enumerate(self.SHIFTS):
                    ref = cg(lambda v: eo.apply(v) + s * v, b, tol=1e-13, max_iter=3000)
                    dev = ((ref.x - res.x[k]).reshape(3, -1).norm(dim=1) / ref.x.reshape(3, -1).norm(dim=1)).max().item()
                    worst_cg = max(worst_cg, dev)
                    self.assertLess(dev, 1e-9, msg=f"shift {s}")
                    # the iteration count of shift k is that of CG on M + sigma_k (to +- 2)
                    self.assertLess((ref.iterations - res.iterations[k]).abs().max().item(), 3)
                    for i in range(3):
                        M = staggered_dense_matrix(eo, geo, batch_index=i)
                        xd = torch.linalg.solve(M + s * torch.eye(M.shape[0], dtype=C), to_dense(b, geo, i))
                        dev = ((xd - to_dense(res.x[k], geo, i)).norm() / xd.norm()).item()
                        worst_dense = max(worst_dense, dev)
                        self.assertLess(dev, 1e-9)
                # odd sites untouched
                self.assertEqual(eo.odd(res.x[0]).abs().max().item(), 0.0)
                say(f"multishift {geo}: iterations per shift/sample {res.iterations.tolist()}, "
                    f"max true residual {res.residual.max().item():.1e}, max rel. dev. from CG {worst_cg:.1e}, "
                    f"from dense {worst_dense:.1e}")

    def test_per_sample_shifts_and_zero_rhs(self):
        U, gen = haar_links(G2, 2, 3)
        eo = StaggeredDirac(U, MASSES, G2).even_odd()
        b = eo.even(torch.randn(G2.staggered_shape(2), dtype=C, generator=gen))
        b[1] = 0.0
        shifts = torch.tensor([[0.5, 0.01, 2.0], [0.0, 0.3, 0.7]], dtype=torch.float64)
        res = multishift_cg(eo.apply, b, shifts, tol=1e-12)
        self.assertEqual(res.x[:, 1].abs().max().item(), 0.0)
        self.assertEqual(res.iterations[:, 1].abs().max().item(), 0)
        M = staggered_dense_matrix(eo, G2, batch_index=0)
        for k in range(3):
            s = shifts[0, k].item()
            xd = torch.linalg.solve(M + s * torch.eye(M.shape[0], dtype=C), to_dense(b, G2, 0))
            self.assertLess(((xd - to_dense(res.x[k], G2, 0)).norm() / xd.norm()).item(), 1e-9)
        with self.assertRaises(ValueError):
            multishift_cg(eo.apply, b, [-0.1, 1.0])


# ---------------------------------------------------------------------------------------------
# 2. rational approximations
# ---------------------------------------------------------------------------------------------

class RationalApproximationTest(unittest.TestCase):
    """Certified max relative error below the requested tolerance, checked again on a dense log
    grid of 200001 points, for gamma = 1/2 (Zolotarev; Remez for the heat bath x^1/4) on the default
    interval of 2D and 4D at m = 0.1, and gamma = 1/4 (Remez) in 2D; Zolotarev and Remez agree for x^-1/2; r(M) phi against the dense
    matrix function."""

    def test_errors_below_tolerance(self):
        lines = []
        for geo, gammas in ((G2, (HALF, QUARTER)), (G4, (HALF, 1))):
            lo, hi = staggered_spectral_bounds(0.1, geo.D)
            for gamma in gammas:
                cases = (("action", -gamma, 1e-10), ("heat bath", gamma / 2, 1e-10), ("force", -gamma, 1e-6))
                if gamma == 1:  # x^-1 is exact: only the heat bath x^{+1/2} (Zolotarev reciprocal)
                    cases = cases[1:2]
                for name, power, tol in cases:
                    t0 = time.time()
                    approx = rational_approximation(power, lo, hi, tol=tol)
                    dense, plain = check_certified(self, approx)
                    lines.append(f"  D={geo.D} [{lo:.4g}, {hi:.4g}] x^{str(power):5s} {name:9s} tol {tol:.0e}: "
                                 f"{approx.method:9s} degree {approx.degree:2d}, certified {approx.error:.3e}, "
                                 f"dense grid {dense:.3e}, plain float64 {plain:.3e} (cancellation "
                                 f"{approx.cancellation:.3g}, rounding {approx.rounding:.1e}) ({time.time() - t0:.1f}s)")
                    self.assertLessEqual(approx.error, tol)
                    if power < 0:
                        self.assertTrue((approx.residues > 0).all() and approx.constant >= 0)
                    # one degree less does not reach the tolerance: the search returns the lowest
                    if approx.degree > 1 and approx.method != "exact":
                        lower = rational_approximation(power, lo, hi, degree=approx.degree - 1, method=approx.method)
                        self.assertGreater(lower.error, tol)
        say("rational approximations (default spectral interval, m = 0.1):\n" + "\n".join(lines))

    def test_zolotarev_is_minimax(self):
        lo, hi = staggered_spectral_bounds(0.1, 2)
        for n in (6, 8):
            z = zolotarev_inverse_sqrt(lo, hi, n)
            r = remez_approximation(Fraction(-1, 2), lo, hi, n)
            say(f"x^-1/2 degree {n}: Zolotarev {z.error:.6e}, Remez {r.error:.6e}")
            self.assertLess(abs(z.error / r.error - 1.0), 1e-2)

    def test_matrix_function(self):
        """r(M_ee) phi (multishift CG) against M_ee^power phi from the eigendecomposition of the dense
        M_ee, 2D 4 x 4 SU(3) Haar links, masses [0.3, 0.5]; the spectrum lies in the interval."""
        U, gen = haar_links(G2, 2, 5)
        eo = StaggeredDirac(U, MASSES, G2).even_odd()
        M = assemble_staggered_even_odd(U, MASSES, G2)
        lo, hi = staggered_spectral_bounds(MASSES, G2.D)
        lam = torch.linalg.eigvalsh(M)
        self.assertGreater(lam.min().item(), lo)
        self.assertLess(lam.max().item(), hi)
        phi = eo.even(torch.randn(G2.staggered_shape(2), dtype=C, generator=gen))
        worst = 0.0
        for gamma in (HALF, QUARTER):
            for power in (-gamma, gamma / 2):
                approx = rational_approximation(power, lo, hi, tol=1e-12)
                out, _, res = rational_apply(approx, eo.apply, phi, tol=1e-13)
                for b in range(2):
                    ref = matrix_function(M[b], power) @ to_dense(phi, G2, b)
                    dev = ((ref - to_dense(out, G2, b)).norm() / ref.norm()).item()
                    worst = max(worst, dev)
                    self.assertLess(dev, 1e-9, msg=f"power {power}")
        say(f"r(M) phi vs dense M^power phi: max rel. dev. {worst:.1e} "
            f"(spectrum [{lam.min().item():.3f}, {lam.max().item():.3f}] in [{lo:.3f}, {hi:.3f}])")


class SmallMassTest(unittest.TestCase):
    """Small masses (spectral condition number kappa = 4e4 at 4D m = 0.02 and 2D m = 0.01, 1.6e5 at
    4D m = 0.01), where float64 Zolotarev coefficients had an error floor ~1.5e-12 and the Remez
    system at 50 digits did not converge (degree >= 17, kappa >~ 2e4). Fast cases: Zolotarev
    coefficients against an independent formula and the achieved error at degree 22; Remez x^{1/4}
    of degree 17 at 4D m = 0.02 (the constructor with that heat-bath degree, a refresh and the
    action on Haar links); over-specified Remez degrees; x^{+1/2}; validation of powers;
    immutability of the cached approximations. NEOQCD_SLOW=1: the full default constructors."""

    def test_zolotarev_coefficients(self):
        """Shifts b_k = lower c_{2k-1} against c_l = sc^2(u_l | 1 - 1/kappa) from the imaginary
        transformation sc(u | m') = Im sn(i u | 1 - m') and K' = pi / (2 agm(1, kappa^-1/2)) in 60
        digits; float64 coefficients had relative errors 1e-10 (kappa = 1.6e5) to 1e-8 (1.6e7)."""
        import mpmath as mp

        for lo, hi, n in ((1e-4 / 1.01, 16.0001 * 1.01, 22), (1e-6, 16.0, 30)):
            z = zolotarev_inverse_sqrt(lo, hi, n)
            with mp.workdps(60):
                kappa = mp.mpf(hi) / mp.mpf(lo)
                Kp = mp.pi / (2 * mp.agm(1, 1 / mp.sqrt(kappa)))
                ref = sorted(float(mp.mpf(lo) * mp.im(mp.ellipfun("sn", 1j * l * Kp / (2 * n + 1), m=1 / kappa)) ** 2)
                             for l in range(1, 2 * n + 1, 2))
            dev = max(abs(a / b - 1) for a, b in zip(sorted(z.shifts), ref))
            say(f"Zolotarev kappa {hi / lo:.3g} degree {n}: max rel. dev. of the shifts {dev:.1e}, error {z.error:.3e}")
            self.assertLess(dev, 1e-13)

    def test_zolotarev_small_mass(self):
        """4D m = 0.01: degree 22 reaches 3.5e-13 (the float64 coefficients never got below 1.5e-12);
        the search returns degree 22 for 1e-12; x^{+1/2} certified against plain and compensated
        dense grids (cancellation ~8e3)."""
        lo, hi = staggered_spectral_bounds(0.01, 4)
        z = zolotarev_inverse_sqrt(lo, hi, 22)
        dense, _ = check_certified(self, z)
        self.assertLess(z.error, 5e-13)
        a = rational_approximation(-HALF, lo, hi, tol=1e-12)
        self.assertEqual(a.degree, 22)
        r = zolotarev_sqrt(lo, hi, 22)
        dense_r, plain_r = check_certified(self, r)
        say(f"4D m=0.01 Zolotarev degree 22: x^-1/2 certified {z.error:.3e} (dense {dense:.3e}); x^+1/2 certified "
            f"{r.error:.3e} (dense {dense_r:.3e}, plain float64 {plain_r:.3e}, cancellation {r.cancellation:.3g})")
        self.assertLess(r.error, 1e-12)

    def test_remez_small_mass(self):
        """x^{1/4} degree 17 on the 4D m = 0.02 interval (raised 'the error does not alternate' at 50
        digits); then the constructor with this heat-bath degree (action and force by tolerance), a
        refresh and the action at the refresh links on Haar fields: S_pf = |eta|^2 up to the
        approximation and solver errors, all multishift solves converged."""
        lo, hi = staggered_spectral_bounds(0.02, 4)
        t0 = time.time()
        a = remez_approximation(QUARTER, lo, hi, 17)
        t_build = time.time() - t0
        dense, plain = check_certified(self, a)
        say(f"4D m=0.02 x^1/4 Remez degree 17: certified {a.error:.3e}, dense {dense:.3e}, plain {plain:.3e} "
            f"(cancellation {a.cancellation:.3g}), {t_build:.1f}s")
        self.assertLess(a.error, 2e-11)
        self.assertTrue((a.residues < 0).all() and a.constant > 0)
        S = RootedStaggeredPseudofermion(0.02, G4, heatbath_degree=17)
        self.assertIs(S.heatbath_approx, a)  # cached
        self.assertLessEqual(S.action_approx.error, 1e-12)
        self.assertEqual(S.action_approx.degree, 20)
        U, gen = haar_links(G4, 2, 71)
        S.refresh(U, gen)
        s = S.action(U)
        eta2 = sum((e.abs() ** 2).reshape(2, -1).sum(1) for e in S.eta)
        dev = ((s - eta2).abs() / eta2).max().item()
        say(f"4D m=0.02 refresh + action: |S - |eta|^2| / |eta|^2 = {dev:.1e}, M_ee applications per solve "
            f"{S.stats['cg_iterations'] / S.stats['solves']:.0f}, last true residual {S.stats['last_residual'].max().item():.1e}")
        self.assertLess(dev, 1e-9)

    def test_over_specified_degree(self):
        """Degrees whose minimax error is below the float64 resolution (kappa = 66 and 5) return a
        float64-level approximation instead of raising."""
        for m, D, power, n in ((0.5, 4, QUARTER, 16), (1.0, 2, -QUARTER, 14)):
            lo, hi = staggered_spectral_bounds(m, D)
            a = remez_approximation(power, lo, hi, n)
            dense, plain = check_certified(self, a)
            say(f"over-specified x^{power} degree {n} on [{lo:.3g}, {hi:.3g}]: certified {a.error:.2e}, dense "
                f"{dense:.2e}, plain {plain:.2e}")
            self.assertLess(a.error, 1e-13)

    def test_powers(self):
        lo, hi = staggered_spectral_bounds(0.3, 2)
        for power in (0, 1, Fraction(3, 2), -2, Fraction(-5, 4), 0.0):
            with self.subTest(power=power):
                with self.assertRaises(ValueError):
                    rational_approximation(power, lo, hi, tol=1e-8)
        with self.assertRaises(ValueError):
            remez_approximation(Fraction(-1), lo, hi, 4)
        with self.assertRaises(ValueError):
            rational_approximation(-1, lo, hi, degree=3, method="remez")
        self.assertEqual(rational_approximation(-1, lo, hi, tol=1e-8).method, "exact")
        with self.assertRaises(RuntimeError):  # below the float64 floor of the certified error
            rational_approximation(-HALF, lo, hi, tol=1e-16)

    def test_immutable(self):
        lo, hi = staggered_spectral_bounds(0.3, 2)
        a = rational_approximation(-HALF, lo, hi, degree=5)
        shifts = a.shifts.copy()
        with self.assertRaises(AttributeError):
            a.error = 0.0
        with self.assertRaises(AttributeError):
            a.residues = a.residues * 2
        with self.assertRaises(ValueError):
            a.residues[0] = 1.0
        with self.assertRaises(ValueError):
            a.shifts[:] = 0.0
        b = rational_approximation(-HALF, lo, hi, degree=5)
        self.assertIs(a, b)
        self.assertTrue((b.shifts == shifts).all())

    @unittest.skipUnless(SLOW, "NEOQCD_SLOW=1: default constructors at small mass (~45 s)")
    def test_default_constructors_small_mass(self):
        """RootedStaggeredPseudofermion with all defaults (action and heat bath 1e-12, force 1e-7) at
        4D m = 0.02 and 0.01 and 2D m = 0.01; certified against dense grids; refresh and action."""
        lines = []
        for geo, m, seed in ((G4, 0.02, 81), (G2, 0.01, 82), (G4, 0.01, 83)):
            with self.subTest(D=geo.D, m=m):
                t0 = time.time()
                S = RootedStaggeredPseudofermion(m, geo)
                t_build = time.time() - t0
                for name, a, tol in (("action", S.action_approx, 1e-12), ("heat bath", S.heatbath_approx, 1e-12),
                                     ("force", S.force_approx, 1e-7)):
                    dense, plain = check_certified(self, a)
                    self.assertLessEqual(a.error, tol)
                    lines.append(f"  D={geo.D} m={m} {name:9s} x^{str(a.power):5s} {a.method:9s} degree {a.degree:2d} "
                                 f"certified {a.error:.3e} dense {dense:.3e} plain {plain:.3e} "
                                 f"(cancellation {a.cancellation:.3g})")
                U, gen = haar_links(geo, 2, seed)
                S.refresh(U, gen)
                s = S.action(U)
                eta2 = sum((e.abs() ** 2).reshape(2, -1).sum(1) for e in S.eta)
                dev = ((s - eta2).abs() / eta2).max().item()
                lines.append(f"  D={geo.D} m={m}: constructor {t_build:.1f}s, |S - |eta|^2| / |eta|^2 = {dev:.1e}, "
                             f"M_ee applications per solve {S.stats['cg_iterations'] / S.stats['solves']:.0f}")
                self.assertLess(dev, 1e-9)
        say("default constructors at small mass:\n" + "\n".join(lines))


# ---------------------------------------------------------------------------------------------
# 3. exactness of the rooted action
# ---------------------------------------------------------------------------------------------

class RootedActionTest(unittest.TestCase):
    """gamma = 1, n_pf = 1 reproduces StaggeredPseudofermion (action and force at the same phi);
    refresh distribution: S_pf at the refresh links is sum |eta|^2 up to the approximation error and
    <S_pf> = Var(S_pf) = n_pf V/2 N."""

    def test_gamma_one_is_unrooted(self):
        for geo, seed in ((G2, 11), (G4, 12)):
            with self.subTest(geometry=geo):
                U, gen = haar_links(geo, 2, seed)
                V, _ = haar_links(geo, 2, seed + 100)
                ref = StaggeredPseudofermion(MASSES, geo, solver_tol=1e-13, action_tol=1e-13)
                ref.refresh(U, gen)
                S = RootedStaggeredPseudofermion(MASSES, geo, exponent=1, solver_tol=1e-13)
                self.assertEqual(S.action_approx.method, "exact")
                S.phi = [ref.phi]
                dS = (S.action(V) - ref.action(V)).abs().max().item()
                scale = ref.action(V).abs().max().item()
                dF = (S.force(V) - ref.force(V)).abs().max().item()
                say(f"gamma = 1 vs StaggeredPseudofermion {geo}: |dS| = {dS:.1e} (S ~ {scale:.0f}), |dF| = {dF:.1e}")
                self.assertLess(dS, 1e-9 * scale)
                self.assertLess(dF, 1e-9)

    def test_refresh_distribution(self):
        U1, gen = haar_links(G2, 1, 13)
        K = 1000
        U = replicate(U1, K)
        S = RootedStaggeredPseudofermion(0.3, G2, exponent=HALF, n_pf=2)
        S.refresh(U, gen)
        n = S.n_components * S.n_pf
        s = S.action(U)
        eta2 = sum((e.abs() ** 2).reshape(K, -1).sum(1) for e in S.eta)
        self.assertEqual(S.operator(U).odd(S.phi[0]).abs().max().item(), 0.0)
        self.assertLess(((s - eta2).abs() / eta2).max().item(), 1e-10)
        mean, var = s.mean().item(), s.var().item()
        say(f"refresh n_pf = 2: <S_pf> = {mean:.2f} (n = {n}), Var = {var:.1f}, max |S - |eta|^2| / |eta|^2 = "
            f"{((s - eta2).abs() / eta2).max().item():.1e}")
        self.assertLess(abs(mean - n), 4.0 * math.sqrt(n / K))
        self.assertLess(abs(var - n), 4.0 * math.sqrt((6.0 * n + 2.0 * n * n) / K))


class BoundsAndSolverChecksTest(unittest.TestCase):
    """User spectral bounds must contain [m_min^2, m_max^2 + D^2] (ValueError; unsafe_bounds=True
    accepts them, and the optional power-iteration check catches lambda_max above the interval);
    the mass is a private copy; unconverged multishift solves raise in refresh, action and force."""

    def test_spectral_bounds_validation(self):
        with self.assertRaises(ValueError):  # free-field bounds at m = 0.3: silently wrong before
            RootedStaggeredPseudofermion(0.3, G4, spectral_bounds=(0.5, 2.0))
        with self.assertRaises(ValueError):  # lower bound above m^2
            RootedStaggeredPseudofermion(MASSES, G2, spectral_bounds=(0.1, 10.0))
        with self.assertRaises(ValueError):  # upper bound below m_max^2 + D^2 = 4.25
            RootedStaggeredPseudofermion(MASSES, G2, spectral_bounds=(0.05, 4.2))
        req = staggered_spectral_bounds(MASSES, G2.D, 0.0)
        RootedStaggeredPseudofermion(MASSES, G2, spectral_bounds=req)  # the rigorous interval itself
        RootedStaggeredPseudofermion(MASSES, G2, spectral_bounds=(0.05, 10.0))
        # unsafe bounds: accepted; the run-time check finds lambda_max ~ m^2 + 3 > 2 on Haar links
        S = RootedStaggeredPseudofermion(0.3, G2, spectral_bounds=(0.05, 2.0), unsafe_bounds=True,
                                         spectrum_check_iterations=30)
        U, gen = haar_links(G2, 2, 91)
        with self.assertRaisesRegex(RuntimeError, "spectral bound"):
            S.refresh(U, gen)
        # default bounds pass the check; the estimate is below the true lambda_max
        S = RootedStaggeredPseudofermion(MASSES, G2, spectrum_check_iterations=30)
        S.refresh(U, gen)
        est = S.stats["lambda_max_estimate"]
        lam = torch.linalg.eigvalsh(assemble_staggered_even_odd(U, MASSES, G2)).max(dim=-1).values
        say(f"power iteration lambda_max {est.tolist()}, exact {lam.tolist()}, upper bound {S.spectral_bounds[1]:.3f}")
        self.assertTrue((est <= lam * (1 + 1e-12)).all() and (est > 0.9 * lam).all())

    def test_mass_is_copied(self):
        m = MASSES.clone()
        S = RootedStaggeredPseudofermion(m, G2)
        m[0] = 0.01  # in place, after construction
        self.assertEqual(S.mass[0].item(), 0.3)
        S.mass[0] = 0.02  # the property returns a copy
        self.assertEqual(S.mass[0].item(), 0.3)
        U, _ = haar_links(G2, 2, 92)
        self.assertTrue(torch.equal(S.dirac(U).mass.reshape(-1), MASSES))

    def test_unconverged_solves_raise(self):
        U, gen = haar_links(G2, 2, 93)
        with self.assertRaisesRegex(RuntimeError, "refresh: multishift CG not converged"):
            RootedStaggeredPseudofermion(MASSES, G2, max_iter=3).refresh(U, gen)
        ok = RootedStaggeredPseudofermion(MASSES, G2)
        ok.refresh(U, gen)
        bad = RootedStaggeredPseudofermion(MASSES, G2, max_iter=3)
        bad.phi = ok.phi
        with self.assertRaisesRegex(RuntimeError, "action: multishift CG not converged"):
            bad.action(U)
        with self.assertRaisesRegex(RuntimeError, "force: multishift CG not converged"):
            bad.force(U)


class StochasticEstimatorTest(unittest.TestCase):
    """<exp(-[S_pf(U') - S_pf(U)])>_{phi drawn at U} = (det M_ee(U') / det M_ee(U))^gamma, exact from
    dense determinants, 2D 4 x 4 SU(3), m = 0.3, U' = exp(i c X) U along the force direction with
    |ratio - 1| > 5 sigma. gamma = 1/2 with n_pf = 1 and 2, gamma = 1/4."""

    def test_determinant_ratio(self):
        geo, mass, K = G2, 0.3, 1500
        U1, gen = haar_links(geo, 1, 22)
        X1 = force_direction(geo, mass, U1, gen)
        U = replicate(U1, K)
        for gamma, n_pf, c in ((HALF, 1, 0.4), (HALF, 2, -0.4), (QUARTER, 1, 0.6)):
            with self.subTest(gamma=gamma, n_pf=n_pf, c=c):
                V1 = SUN_mul(expi(c * X1), U1)
                ld = log_det_staggered(torch.cat((U1, V1)), mass, geo)
                ratio = math.exp(float(gamma) * (ld[1] - ld[0]).item())
                S = RootedStaggeredPseudofermion(mass, geo, exponent=gamma, n_pf=n_pf)
                S.refresh(U, gen)
                w = torch.exp(-S.action_difference(U, replicate(V1, K)))
                est, err = w.mean().item(), w.std().item() / math.sqrt(K)
                msg = (f"gamma={gamma} n_pf={n_pf} c={c:+.2f}: (det ratio)^gamma exact {ratio:.4f}, stochastic "
                       f"{est:.4f} +- {err:.4f} (|ratio - 1| = {abs(ratio - 1) / err:.1f} sigma, "
                       f"|est - ratio| = {abs(est - ratio) / err:.2f} sigma)")
                say(msg)
                self.assertGreater(abs(ratio - 1.0), 5 * err, msg=msg)
                self.assertLess(abs(est - ratio), 4 * err, msg=msg)


# ---------------------------------------------------------------------------------------------
# 4. force
# ---------------------------------------------------------------------------------------------

class ForceTest(unittest.TestCase):
    """tr(F X) = d/dt S_F(exp(i t X) U)|_0 with S_F the action of the SAME (force) rational function,
    against 4th-order finite differences: random directions and single-link generators including
    time links at t = 0. Per-batch mass [0.3, 0.5]; gamma = 1/2 (n_pf 1) and 1/2 with n_pf = 2."""

    LINKS = {4: [(0, 0, 0, 0, 0, 0), (1, 0, 0, 1, 1, 0), (1, 1, 2, 1, 0, 1), (0, 3, 3, 1, 1, 0)],
             2: [(0, 0, 0, 0), (1, 0, 0, 3), (1, 1, 3, 2), (0, 1, 1, 0)]}

    def cases(self):
        yield "2D SU(3)", G2, -1.0, HALF, 1
        yield "2D SU(3) periodic", G2, 1.0, HALF, 2
        yield "2D SU(2)", G2_SU2, -1.0, QUARTER, 1
        yield "4D SU(3)", G4, -1.0, HALF, 1

    def test_force_against_finite_differences(self):
        for i, (name, geo, time_sign, gamma, n_pf) in enumerate(self.cases()):
            with self.subTest(case=name):
                U, gen = haar_links(geo, 2, 30 + i)
                S = RootedStaggeredPseudofermion(MASSES, geo, exponent=gamma, n_pf=n_pf, time_sign=time_sign,
                                                 solver_tol=1e-13)
                S.refresh(U, gen)
                F = S.force(U)
                self.assertEqual(F.shape, U.shape)
                self.assertLess((F - SUN_dagger(F)).abs().max().item(), 1e-13)
                self.assertLess(SUN_trace(F).abs().max().item(), 1e-13)
                action = lambda V: S.action(V, "force")
                worst_random = 0.0
                for _ in range(2):
                    X = random_algebra(U.shape, generator=gen)
                    pred = link_sum(trace_product(F, X))
                    fd = fd_derivative(action, U, X)
                    rel = ((pred - fd).abs() / fd.abs()).max().item()
                    worst_random = max(worst_random, rel)
                    self.assertLess(rel, 1e-6, msg=f"pred={pred} fd={fd}")
                T = algebra_basis(geo.N)
                worst = 0.0
                for link in self.LINKS[geo.D]:
                    b = link[0]
                    for a in range(geo.N * geo.N - 1):
                        X = torch.zeros_like(U)
                        X[link] = T[a]
                        pred = trace_product(F[link], T[a]).item()
                        fd = fd_derivative(action, U, X)[b].item()
                        worst = max(worst, abs(pred - fd) / max(abs(fd), 1.0))
                        self.assertLess(abs(pred - fd), 1e-6 * max(abs(fd), 1.0), msg=f"link={link} a={a}")
                say(f"{name} gamma={gamma} n_pf={n_pf} (force degree {S.force_approx.degree}): random directions "
                    f"rel. dev. {worst_random:.1e}, single links {worst:.1e}")

    def test_summed_outer_products(self):
        """staggered_rational_force equals the sum of staggered_hopping_derivative over the poles."""
        U, gen = haar_links(G4, 2, 41)
        eo = StaggeredDirac(U, MASSES, G4).even_odd()
        X = torch.stack([eo.even(torch.randn(G4.staggered_shape(2), dtype=C, generator=gen)) for _ in range(3)])
        c = [0.7, 0.2, 1.3]
        ref = sum(2.0 * ck * staggered_hopping_derivative(eo, X[k], eo.odd(eo.staggered.hopping(X[k])))
                  for k, ck in enumerate(c))
        self.assertLess((staggered_rational_force(eo, X, c) - ref).abs().max().item(), 1e-13)


class AutogradForceTest(unittest.TestCase):
    """(a) The force against autograd of S_F = sum_j phi_j^dagger r_F(M) phi_j with r_F(M) built from
    dense solves (same rational function, 1e-10); (b) the force of the accurate (action) rational
    function against the exact gradient of phi^dagger M^-gamma phi through the dense matrix function
    (Daleckii-Krein: dS = tr(G dM), G = V (Lambda o V^dagger phi phi^dagger V) V^dagger with the
    divided differences Lambda_ij = f[lambda_i, lambda_j], f(x) = x^-gamma; 1e-8 relative). Links
    U(omega) = (1 + i omega.T) U, so dS/d omega_a = tr(F T^a)."""

    def test_force_against_autograd(self):
        for geo, seed, gamma, n_pf in ((G2, 61, HALF, 1), (G2_SU2, 62, HALF, 2), (G4, 63, QUARTER, 1)):
            with self.subTest(geometry=geo, gamma=gamma, n_pf=n_pf):
                U, gen = haar_links(geo, 2, seed)
                S = RootedStaggeredPseudofermion(MASSES, geo, exponent=gamma, n_pf=n_pf, solver_tol=1e-14,
                                                 action_solver_tol=1e-14)
                S.refresh(U, gen)
                T = algebra_basis(geo.N)
                omega = torch.zeros(U.shape[:-2] + (T.shape[0],), dtype=torch.float64, requires_grad=True)
                Uw = U + 1j * torch.matmul(torch.einsum("...a,aij->...ij", omega.to(C), T), U)
                eo = StaggeredDirac(Uw, MASSES, geo).even_odd()
                Ms = [staggered_dense_matrix(eo, geo, batch_index=b) for b in range(2)]
                g = float(gamma) / n_pf
                # (a) same rational function, dense solves
                approx = S.force_approx
                total = 0.0
                for b in range(2):
                    eye = torch.eye(Ms[b].shape[0], dtype=C)
                    for phi in S.phi:
                        v = to_dense(phi, geo, b)
                        rv = approx.constant * v
                        for a, s in zip(approx.residues, approx.shifts):
                            rv = rv + float(a) * torch.linalg.solve(Ms[b] + float(s) * eye, v)
                        total = total + (v.conj() * rv).sum().real
                grad_a, = torch.autograd.grad(total, omega, retain_graph=True)
                comp = algebra_components(S.force(U))
                dev_a = (comp - grad_a).abs().max().item()
                scale = grad_a.abs().max().item()
                # (b) exact matrix function M^-g (Daleckii-Krein), against the accurate rational force
                surrogate = 0.0
                for b in range(2):
                    M0 = Ms[b].detach()
                    lam, Vec = torch.linalg.eigh(M0)
                    f, fp = lam.pow(-g), -g * lam.pow(-g - 1)
                    dl = lam[:, None] - lam[None, :]
                    close = dl.abs() < 1e-10
                    L = torch.where(close, 0.5 * (fp[:, None] + fp[None, :]), (f[:, None] - f[None, :]) / torch.where(close, torch.ones_like(dl), dl))
                    for phi in S.phi:
                        v = to_dense(phi, geo, b)
                        P = Vec.conj().T @ torch.outer(v, v.conj()) @ Vec
                        G = Vec @ (L.to(C) * P) @ Vec.conj().T
                        surrogate = surrogate + torch.trace(G @ Ms[b]).real
                grad_b, = torch.autograd.grad(surrogate, omega)
                comp_b = algebra_components(S.force(U, approximation="action"))
                dev_b = (comp_b - grad_b).abs().max().item()
                say(f"{geo} gamma={gamma} n_pf={n_pf}: |F_a - autograd(r_F)_a| = {dev_a:.1e}, "
                    f"|F(r_S)_a - exact x^-g gradient_a| = {dev_b:.1e} (max |F_a| = {scale:.2f})")
                self.assertLess(dev_a, 1e-10 * max(scale, 1.0))
                self.assertLess(dev_b, 1e-8 * max(scale, 1.0))


# ---------------------------------------------------------------------------------------------
# 5. HMC
# ---------------------------------------------------------------------------------------------

class ReversibilityTest(unittest.TestCase):
    """Forward, momentum flip, forward returns to (U, -P); multishift CG always from zero, tol 1e-14
    (see test_fermions_staggered_hmc.ReversibilityTest for why the tolerance must be tight)."""

    def test_reversibility(self):
        for geo, seed in ((G4, 101), (G2_SU2, 102)):
            U, gen = haar_links(geo, 2, seed)
            P = H.random_momenta(U.shape, generator=gen)
            ham = RootedStaggeredHamiltonian(5.0, MASSES, geo, exponent=HALF, n_gauge_substeps=2, solver_tol=1e-14)
            ham.refresh(U, gen)
            for integrator in H.INTEGRATORS:
                with self.subTest(geometry=geo, integrator=integrator):
                    hmc = H.HMC(geo, ham, tau=0.5, n_steps=4, integrator=integrator)
                    U1, P1, _ = hmc.integrate(U, P)
                    U2, P2, _ = hmc.integrate(U1, -P1)
                    dU, dP = (U2 - U).abs().max().item(), (P2 + P).abs().max().item()
                    say(f"{geo} {integrator}: |U2 - U| = {dU:.1e}, |P2 + P| = {dP:.1e}")
                    self.assertGreater((U1 - U).abs().max().item(), 0.1)
                    self.assertLess(dU, 1e-12)
                    self.assertLess(dP, 1e-12)


class EnergyScalingTest(unittest.TestCase):
    """|dH| ~ eps^2 for the two-level RHMC (gamma = 1/2), leapfrog and Omelyan, 2^3 x 4 SU(3). dH
    uses the accurate action, the MD the cheaper force function, so dH has an eps-independent part
    of the order of the force approximation error: force degree fixed at a 1e-10 tolerance here."""

    STEPS = (8, 16, 32)

    def test_dH_scales_as_eps_squared(self):
        U, gen = haar_links(G4, 4, 111)
        P = H.random_momenta(U.shape, generator=gen)
        ham = RootedStaggeredHamiltonian(5.0, 0.3, G4, exponent=HALF, n_gauge_substeps=2, solver_tol=1e-12,
                                         force_approx_tol=1e-10)
        ham.refresh(U, gen)
        mean_dH = {}
        for integrator in H.INTEGRATORS:
            with self.subTest(integrator=integrator):
                hmc = H.HMC(G4, ham, tau=0.5, n_steps=1, integrator=integrator)
                dH = []
                for n in self.STEPS:
                    U1, P1, _ = hmc.integrate(U, P, n_steps=n)
                    dH.append(ham.energy_difference(U, P, U1, P1).abs().mean().item())
                x = torch.tensor([math.log(1.0 / n) for n in self.STEPS], dtype=torch.double)
                y = torch.tensor([math.log(d) for d in dH], dtype=torch.double)
                slope = (((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()).item()
                say(f"RHMC {integrator}: <|dH|> = {['%.2e' % d for d in dH]}, slope {slope:.3f}")
                self.assertGreater(slope, 1.85, msg=f"dH={dH}")
                self.assertLess(slope, 2.15, msg=f"dH={dH}")
                mean_dH[integrator] = dH
        self.assertLess(mean_dH["omelyan"][-1], 0.25 * mean_dH["leapfrog"][-1])


class ExactComparisonTest(unittest.TestCase):
    """RHMC at gamma = 1/2 against the exact-determinant Metropolis for det(M_ee)^(1/2) exp(-S_g),
    2D 4 x 4 SU(3), beta = 5.0, m = 0.1, B = 32 chains from a cold start. Exact references at
    gamma = 0 (quenched), 1/2 and 1 (ExactRootedMetropolis; gamma = 1 is ExactStaggeredMetropolis).
    Observables plaquette and log det M_ee, computed exactly on every configuration. Errors from the
    spread of the chain means. The rooted HMC must agree with the exact gamma = 1/2 value within
    4 sigma, <exp(-dH)> = 1 within 4 sigma, and in <log det> it must sit within 0.3 of the gap from
    the exact gamma = 1/2 value to each neighbour (gamma = 0 and gamma = 1); both gaps must be
    > 4 sigma."""

    B = 32
    BETA = 5.0
    MASS = 0.1

    def chain_stats(self, x):
        m = torch.stack(x).mean(0)
        return m.mean().item(), (m.std() / math.sqrt(self.B)).item()

    def observables(self, U):
        return mean_plaquette(plaquette_fields(U, G2.D), G2.N), log_det_staggered(U, self.MASS, G2)

    def run_rhmc(self, n_therm, n_meas, seed, n_steps, n_gauge_substeps):
        gen = torch.Generator().manual_seed(seed)
        ham = RootedStaggeredHamiltonian(self.BETA, self.MASS, G2, exponent=HALF, n_gauge_substeps=n_gauge_substeps)
        U = SUN_identity(G2.link_shape(self.B)[:-1], dtype=C)
        hmc = H.HMC(G2, ham, tau=1.0, n_steps=n_steps, integrator="omelyan")
        plaq, logdet, expdH, acc = [], [], [], []
        for i in range(n_therm + n_meas):
            U, accepted, dH, stats = hmc.trajectory(U, gen)
            if i >= n_therm:
                p, l = self.observables(U)
                plaq.append(p)
                logdet.append(l)
                expdH.append(stats["exp_minus_dH"])
                acc.append(stats["acceptance"])
        return (self.chain_stats(plaq), self.chain_stats(logdet), self.chain_stats(expdH), sum(acc) / len(acc),
                ham.fermion)

    def run_metropolis(self, gamma, n_therm, n_meas, seed):
        gen = torch.Generator().manual_seed(seed)
        if gamma == 1:
            m = ExactStaggeredMetropolis(self.BETA, self.MASS, G2, eps=0.45, links_per_update=1)
        else:
            m = ExactRootedMetropolis(self.BETA, self.MASS, G2, exponent=gamma, eps=0.45, links_per_update=1)
        m.reset(SUN_identity(G2.link_shape(self.B)[:-1], dtype=C))
        n_links = G2.D * G2.volume
        plaq, logdet = [], []
        for sweep in range(n_therm + n_meas):
            for _ in range(n_links):
                m.step(gen)
            if sweep >= n_therm:
                p, l = self.observables(m.U)
                plaq.append(p)
                logdet.append(l)
                if m.log_det_value is not None:
                    self.assertLess((m.log_det_value - l).abs().max().item(), 1e-10)
        return self.chain_stats(plaq), self.chain_stats(logdet), m.acceptance

    def test_rooted_hmc_against_exact(self):
        t0 = time.time()
        (p_h, e_ph), (l_h, e_lh), (x, e_x), acc, fermion = self.run_rhmc(10, 50, 300, n_steps=5, n_gauge_substeps=2)
        t1 = time.time()
        ref = {}
        for gamma, seed in ((HALF, 400), (Fraction(0), 500), (Fraction(1), 600)):
            ref[gamma] = self.run_metropolis(gamma, 40, 60, seed)
        t2 = time.time()
        (p_m, e_pm), (l_m, e_lm), acc_m = ref[HALF]
        (p_0, e_p0), (l_0, e_l0), _ = ref[Fraction(0)]
        (p_1, e_p1), (l_1, e_l1), _ = ref[Fraction(1)]
        st = fermion.stats
        gap0, gap1 = abs(l_m - l_0), abs(l_m - l_1)
        msg = (f"2D 4x4 SU(3) rooted staggered gamma=1/2 beta={self.BETA} m={self.MASS} "
               f"(action degree {fermion.action_approx.degree}, force degree {fermion.force_approx.degree}, "
               f"heat bath degree {fermion.heatbath_approx.degree}):\n"
               f"  <P>       RHMC {p_h:.4f}({e_ph:.4f})  exact g=1/2 {p_m:.4f}({e_pm:.4f})  "
               f"exact g=0 {p_0:.4f}({e_p0:.4f})  exact g=1 {p_1:.4f}({e_p1:.4f})\n"
               f"  <log det> RHMC {l_h:.4f}({e_lh:.4f})  exact g=1/2 {l_m:.4f}({e_lm:.4f})  "
               f"exact g=0 {l_0:.4f}({e_l0:.4f})  exact g=1 {l_1:.4f}({e_l1:.4f})\n"
               f"  <exp(-dH)> = {x:.4f}({e_x:.4f}), RHMC acceptance {acc:.3f}, Metropolis acceptance {acc_m:.3f}, "
               f"M_ee applications per multishift solve {st['cg_iterations'] / st['solves']:.1f}\n"
               f"  sigmas: <P> RHMC-exact {abs(p_h - p_m) / math.hypot(e_ph, e_pm):.2f}, <log det> RHMC-exact "
               f"{abs(l_h - l_m) / math.hypot(e_lh, e_lm):.2f}, exact(1/2)-exact(0) {gap0 / math.hypot(e_lm, e_l0):.2f}, "
               f"exact(1/2)-exact(1) {gap1 / math.hypot(e_lm, e_l1):.2f}, <exp(-dH)>-1 {abs(x - 1.0) / e_x:.2f}\n"
               f"  |l_rhmc - l_exact| / gap to g=0: {abs(l_h - l_m) / gap0:.3f}, to g=1: {abs(l_h - l_m) / gap1:.3f}\n"
               f"  time: RHMC {t1 - t0:.1f}s, three exact samplers {t2 - t1:.1f}s")
        say(msg)
        self.assertGreater(acc, 0.7, msg=msg)
        self.assertGreater(acc_m, 0.3, msg=msg)
        self.assertLess(abs(x - 1.0), 4 * e_x, msg=msg)
        self.assertLess(abs(p_h - p_m), 4 * math.hypot(e_ph, e_pm), msg=msg)
        self.assertLess(abs(l_h - l_m), 4 * math.hypot(e_lh, e_lm), msg=msg)
        self.assertGreater(gap0, 4 * math.hypot(e_lm, e_l0), msg=msg)
        self.assertGreater(gap1, 4 * math.hypot(e_lm, e_l1), msg=msg)
        self.assertLess(abs(l_h - l_m), 0.3 * gap0, msg=msg)
        self.assertLess(abs(l_h - l_m), 0.3 * gap1, msg=msg)

    def test_effective_action(self):
        """rooted_effective_action against an independent computation: S_g = beta sum_x sum_{mu<nu}
        (1 - Re tr P_{mu nu}(x) / N) from explicit link products, and log det M_ee = log det D of
        the full (not even-odd) dense staggered matrix (det D = det M_ee for D = m + K, K
        antihermitian with chiral pairs), for gamma = 1/2 and 1/4."""
        U, _ = haar_links(G2, 3, 7)
        N, D = G2.N, G2.D
        retr = 0.0
        for mu in range(D):
            for nu in range(mu + 1, D):
                # U_nu(x + mu) is torch.roll(., 1, dim of mu) in this code base (shift(., mu, +1))
                Umu, Unu = U[:, mu], U[:, nu]
                P = SUN_mul(SUN_mul(Umu, torch.roll(Unu, 1, dims=1 + mu)),
                            SUN_mul(SUN_dagger(torch.roll(Umu, 1, dims=1 + nu)), SUN_dagger(Unu)))
                retr = retr + SUN_trace(P).real.reshape(3, -1).sum(1)
        S_g = self.BETA * (G2.volume * D * (D - 1) / 2 - retr / N)
        ld = []
        for b in range(3):
            Dm = staggered_dense_matrix(StaggeredDirac(U, self.MASS, G2), G2, batch_index=b)
            self.assertEqual(Dm.shape[0], G2.volume * N)
            sign, logabs = torch.linalg.slogdet(Dm)
            self.assertLess(abs(sign.item() - 1.0), 1e-10)
            ld.append(logabs)
        ld = torch.stack(ld)
        for gamma in (HALF, QUARTER):
            S = rooted_effective_action(U, self.BETA, self.MASS, G2, gamma)
            expected = S_g - float(gamma) * ld
            self.assertLess((S - expected).abs().max().item(), 1e-10 * expected.abs().max().item())


if __name__ == "__main__":
    unittest.main()
