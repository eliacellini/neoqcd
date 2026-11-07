"""Tests of the staggered pseudofermion action and HMC (neoqcd.fermions.staggered_pseudofermion)
and of the exact dense-determinant tools (neoqcd.fermions.staggered_exact). Track C2 of
fermions_plan.md: one even-site staggered field, N_f = 2^(D/2) (4 in 4D, 2 in 2D).

CPU, complex128, fixed seeds. Lattices: 2D 4 x 4 SU(3) (M_ee dimension 24), 2D 4 x 4 SU(2) (16),
2^3 x 4 SU(3) (48). Masses 0.3 and 0.5 on Haar-random links, 0.3 and 0.1 in the exact comparison.
Prints behind NEOQCD_TEST_VERBOSE.
"""

import math
import os
import unittest

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
from neoqcd.fermions.staggered import StaggeredDirac, staggered_dense_matrix
from neoqcd.fermions.staggered_exact import (
    ExactStaggeredMetropolis,
    assemble_staggered_even_odd,
    assemble_staggered_hopping,
    log_det_staggered,
)
from neoqcd.fermions.staggered_pseudofermion import (
    StaggeredHamiltonian,
    StaggeredPseudofermion,
    staggered_hopping_derivative,
)

G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4, SU(3)
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4, SU(3)
G2_SU2 = Geometry(2, 4, 4, 2)  # 2D 4 x 4, SU(2)
C = torch.complex128
MASSES = torch.tensor([0.3, 0.5], dtype=torch.float64)

VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))


def haar_links(geometry, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=gen), gen


def replicate(U, K):
    return U.expand((K,) + tuple(U.shape[1:])).contiguous()


def fd_derivative(f, U, X, t=1e-3):
    """4th-order central difference of f(exp(i s X) U) at s = 0."""
    g = lambda s: f(SUN_mul(expi(s * X), U))
    return (-g(2 * t) + 8 * g(t) - 8 * g(-t) + g(-2 * t)) / (12 * t)


def su_n_violation(U):
    eye = SUN_identity(U.shape[:-1], dtype=U.dtype)
    unit = (SUN_mul(SUN_dagger(U), U) - eye).abs().max().item()
    det = (torch.linalg.det(U) - 1.0).abs().max().item()
    return unit, det


def force_direction(geo, mass, U1, gen, n=16, norm=1.0):
    """Unit-norm (times `norm`) average pseudofermion force of n independent phi at U1: a direction
    along which log det M_ee changes fast."""
    probe = StaggeredPseudofermion(mass, geo)
    Un = replicate(U1, n)
    probe.refresh(Un, gen)
    X1 = probe.force(Un).mean(0, keepdim=True)
    return norm * X1 / math.sqrt(link_sum(trace_product(X1, X1)).item())


class ExactToolsTest(unittest.TestCase):
    """The batched dense assembly agrees with `staggered_dense_matrix` of the sparse operators;
    log det M_ee = log det D (one staggered field: N_f = 2^(D/2) flavours)."""

    def test_assembly_and_log_det(self):
        for geo, seed in ((G4, 1), (G2, 2), (G2_SU2, 3)):
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    U, _ = haar_links(geo, 2, seed)
                    op = StaggeredDirac(U, MASSES, geo, time_sign)
                    A = assemble_staggered_even_odd(U, MASSES, geo, time_sign)
                    K = assemble_staggered_hopping(U, geo, time_sign=time_sign)
                    ld = log_det_staggered(U, MASSES, geo, time_sign)
                    ld_ref = log_det_staggered(U, MASSES, geo, time_sign, method="dense_matrix")
                    self.assertLess((ld - ld_ref).abs().max().item(), 1e-10)
                    for b in range(2):
                        self.assertLess((A[b] - staggered_dense_matrix(op.even_odd(), geo, batch_index=b)).abs().max().item(), 1e-14)
                        Dfull = staggered_dense_matrix(op, geo, batch_index=b)
                        eye = torch.eye(Dfull.shape[0], dtype=C)
                        self.assertLess((MASSES[b] * eye + K[b] - Dfull).abs().max().item(), 1e-14)
                        sign, ld_D = torch.linalg.slogdet(Dfull)
                        self.assertLess(abs(sign.item() - 1.0), 1e-10)
                        self.assertLess(abs(ld_D.item() - ld[b].item()), 1e-10)


class RefreshTest(unittest.TestCase):
    """1. phi = (D^dagger chi)_e lives on even sites and has density ~ exp(-phi^dagger M_ee^-1 phi):
    <S_pf> = n and Var(S_pf) = n at the refresh links (S_pf = chi^dagger P chi, P a rank-n
    projector, so a sum of n unit exponentials in distribution)."""

    def test_refresh_distribution(self):
        for geo, K in ((G2, 4000), (G4, 1000)):
            with self.subTest(geometry=geo):
                U1, gen = haar_links(geo, 1, 10 + geo.D)
                U = replicate(U1, K)
                S = StaggeredPseudofermion(0.3, geo, action_tol=1e-13)
                S.refresh(U, gen)
                n = S.n_components
                self.assertEqual(n, geo.volume * geo.N // 2)
                self.assertEqual(S.n_flavours, 2 ** (geo.D // 2))
                self.assertEqual(S.operator(U).odd(S.phi).abs().max().item(), 0.0)
                s = S.action(U)
                mean, var = s.mean().item(), s.var().item()
                if VERBOSE:
                    print(f"\n{geo}: <S_pf> = {mean:.2f} +- {math.sqrt(n / K):.2f}, Var = {var:.1f}, n = {n}")
                self.assertLess(abs(mean - n), 4.0 * math.sqrt(n / K))
                self.assertLess(abs(var - n), 4.0 * math.sqrt((6.0 * n + 2.0 * n * n) / K))
                # the cached refresh action is the solved one
                self.assertLess((S.action_difference(U, U)).abs().max().item(), 1e-9)


class ActionCacheTest(unittest.TestCase):
    """The cached S_pf at the refresh links is used only while those links are unchanged: after an
    in-place `U.copy_(U2)` (same object, new `_version`) `action_difference` re-solves."""

    def test_cache_invalidated_by_in_place_change(self):
        U, gen = haar_links(G4, 2, 21)
        U2, _ = haar_links(G4, 2, 22)
        V, _ = haar_links(G4, 2, 23)
        S = StaggeredPseudofermion(MASSES, G4, action_tol=1e-13)
        S.refresh(U, gen)
        d1 = S.action_difference(U, V)
        self.assertLess((d1 - (S.action(V) - S.action(U.clone()))).abs().max().item(), 1e-9)
        U.copy_(U2)
        d2 = S.action_difference(U, V)
        honest = S.action(V) - S.action(U2)
        if VERBOSE:
            print(f"\nafter U.copy_: action_difference {d2.tolist()}, honest {honest.tolist()}")
        self.assertGreater((d1 - honest).abs().max().item(), 1e-3)
        self.assertLess((d2 - honest).abs().max().item(), 1e-9)


class StochasticEstimatorTest(unittest.TestCase):
    """1. <exp(-[S_pf(U') - S_pf(U)])>_{phi drawn at U} = det M_ee(U') / det M_ee(U), exact from
    dense determinants. U' = exp(i c X) U with X a unit-norm force-averaged direction; c is chosen
    so that the ratio is far from 1 (asserted > 5 sigma away)."""

    def run_case(self, geo, mass, K, seed, cs):
        U1, gen = haar_links(geo, 1, seed)
        X1 = force_direction(geo, mass, U1, gen)
        U = replicate(U1, K)
        for c in cs:
            with self.subTest(geometry=geo, c=c):
                V1 = SUN_mul(expi(c * X1), U1)
                ld = log_det_staggered(torch.cat((U1, V1)), mass, geo)
                ratio = math.exp((ld[1] - ld[0]).item())
                S = StaggeredPseudofermion(mass, geo)
                S.refresh(U, gen)
                w = torch.exp(-S.action_difference(U, replicate(V1, K)))
                est, err = w.mean().item(), w.std().item() / math.sqrt(K)
                msg = (f"{geo} m={mass} c={c:+.2f}: det ratio exact {ratio:.4f}, stochastic {est:.4f} +- {err:.4f} "
                       f"(|ratio - 1| = {abs(ratio - 1) / err:.1f} sigma, |est - ratio| = {abs(est - ratio) / err:.2f} sigma)")
                if VERBOSE:
                    print("\n" + msg)
                self.assertGreater(abs(ratio - 1.0), 5 * err, msg=msg)
                self.assertLess(abs(est - ratio), 4 * err, msg=msg)

    def test_determinant_ratio_2d(self):
        self.run_case(G2, 0.3, 2000, 22, (0.3, -0.3))

    def test_determinant_ratio_4d(self):
        self.run_case(G4, 0.3, 600, 21, (0.2, -0.2))


class ForceTest(unittest.TestCase):
    """2. tr(F X) = d/dt S_pf(exp(i t X) U)|_0 at fixed phi, against 4th-order finite differences,
    random directions (all links) and single-link generators, including time links at t = 0
    (the ones carrying the antiperiodic sign). Per-batch mass [0.3, 0.5]."""

    # (b, mu, t, x[, y, z]); the first two of each list are time links at t = 0
    LINKS = {4: [(0, 0, 0, 0, 0, 0), (1, 0, 0, 1, 1, 0), (1, 1, 2, 1, 0, 1), (0, 3, 3, 1, 1, 0)],
             2: [(0, 0, 0, 0), (1, 0, 0, 3), (1, 1, 3, 2), (0, 1, 1, 0)]}

    def cases(self):
        yield "2D SU(3)", G2, -1.0
        yield "2D SU(3) periodic", G2, 1.0
        yield "2D SU(2)", G2_SU2, -1.0
        yield "4D SU(3)", G4, -1.0

    def test_force_against_finite_differences(self):
        for i, (name, geo, time_sign) in enumerate(self.cases()):
            with self.subTest(case=name):
                U, gen = haar_links(geo, 2, 30 + i)
                S = StaggeredPseudofermion(MASSES, geo, time_sign=time_sign, solver_tol=1e-13, action_tol=1e-13)
                S.refresh(U, gen)
                F = S.force(U)
                self.assertEqual(F.shape, U.shape)
                self.assertLess((F - SUN_dagger(F)).abs().max().item(), 1e-13)
                self.assertLess(SUN_trace(F).abs().max().item(), 1e-13)
                worst_random = 0.0
                for _ in range(2):
                    X = random_algebra(U.shape, generator=gen)
                    pred = link_sum(trace_product(F, X))
                    fd = fd_derivative(S.action, U, X)
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
                        fd = fd_derivative(S.action, U, X)[b].item()
                        worst = max(worst, abs(pred - fd) / max(abs(fd), 1.0))
                        self.assertLess(abs(pred - fd), 1e-6 * max(abs(fd), 1.0), msg=f"link={link} a={a}")
                if VERBOSE:
                    print(f"\n{name}: random directions rel. dev. {worst_random:.1e}, "
                          f"single links {worst:.1e}")

    def test_hopping_derivative_bilinear(self):
        """Re[a^dagger K b] for arbitrary fields, both boundary conditions."""
        for geo, seed in ((G4, 41), (G2, 42), (G2_SU2, 43)):
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    U, gen = haar_links(geo, 2, seed)
                    a = torch.randn(geo.staggered_shape(2), dtype=C, generator=gen)
                    b = torch.randn(geo.staggered_shape(2), dtype=C, generator=gen)
                    f = lambda V: (a.conj() * StaggeredDirac(V, 0.3, geo, time_sign).hopping(b)).real.reshape(2, -1).sum(1)
                    G = staggered_hopping_derivative(StaggeredDirac(U, 0.3, geo, time_sign), a, b)
                    for _ in range(2):
                        X = random_algebra(U.shape, generator=gen)
                        pred = link_sum(trace_product(G, X))
                        fd = fd_derivative(f, U, X)
                        self.assertLess(((pred - fd).abs() / fd.abs()).max().item(), 1e-9)


class AutogradForceTest(unittest.TestCase):
    """3. The force against autograd of S_pf = phi^dagger M_ee^-1 phi computed with a dense solve
    (torch.linalg.solve on `staggered_dense_matrix` of M_ee). The links are parametrised as
    U(omega) = (1 + i sum_a omega_a T^a) U, which has the same first derivative at omega = 0 as
    exp(i omega.T) U, so d S / d omega_a = tr(F T^a)."""

    def test_force_against_autograd(self):
        for geo, seed in ((G2, 61), (G2_SU2, 62), (G4, 63)):
            with self.subTest(geometry=geo):
                U, gen = haar_links(geo, 2, seed)
                S = StaggeredPseudofermion(MASSES, geo, solver_tol=1e-14)
                S.refresh(U, gen)
                F = S.force(U)
                T = algebra_basis(geo.N)
                omega = torch.zeros(U.shape[:-2] + (T.shape[0],), dtype=torch.float64, requires_grad=True)
                Uw = U + 1j * torch.matmul(torch.einsum("...a,aij->...ij", omega.to(C), T), U)
                eo = StaggeredDirac(Uw, MASSES, geo).even_odd()
                sites = eo.support.reshape(-1).nonzero().squeeze(-1)
                total = 0.0
                for b in range(2):
                    M = staggered_dense_matrix(eo, geo, batch_index=b)
                    phi = S.phi[b].reshape(geo.volume, geo.N)[sites].reshape(-1)
                    x = torch.linalg.solve(M, phi)
                    total = total + (phi.conj() * x).sum().real
                total.backward()
                ref = omega.grad
                comp = algebra_components(F)
                dev = (comp - ref).abs().max().item()
                scale = ref.abs().max().item()
                if VERBOSE:
                    print(f"\n{geo}: max |F_a - autograd_a| = {dev:.1e} (max |F_a| = {scale:.2f})")
                self.assertLess(dev, 1e-10 * max(scale, 1.0))


class ForceExpectationTest(unittest.TestCase):
    """<tr(F X)>_phi = -d/dt log det M_ee(exp(i t X) U): the force is an unbiased estimate of the
    exact force of N_f = 2^(D/2) flavours (pins the normalisation). 2D 4 x 4 SU(3)."""

    def test_force_expectation(self):
        geo, mass, K = G2, 0.3, 2000
        U1, gen = haar_links(geo, 1, 71)
        X1 = force_direction(geo, mass, U1, gen, norm=3.0)
        exact = -fd_derivative(lambda V: log_det_staggered(V, mass, geo), U1, X1).item()
        S = StaggeredPseudofermion(mass, geo)
        U = replicate(U1, K)
        S.refresh(U, gen)
        proj = link_sum(trace_product(S.force(U), X1))
        est, err = proj.mean().item(), proj.std().item() / math.sqrt(K)
        if VERBOSE:
            print(f"\n<tr(F X)> = {est:.3f} +- {err:.3f}, -d log det M_ee = {exact:.3f}")
        self.assertGreater(abs(exact), 10 * err)
        self.assertLess(abs(est - exact), 4 * err)


class ReversibilityTest(unittest.TestCase):
    """4. Forward, momentum flip, forward returns to (U, -P), on 2^3 x 4 SU(3) and 2D 4 x 4 SU(2).

    With `reuse_guess=False` the CG starts from zero, but the force is still a smooth function of
    U only up to the solver tolerance: in finite precision, once CG has lost orthogonality its
    iterates are not a smooth function of the operator, so the reversed run does not retrace the
    forward one beyond O(tol), even with identical forward and backward iteration counts. This
    is generic, not specific to these tiny lattices (on 4^4: |U2 - U| = 1.2e-10 at tol 1e-8,
    1.7e-15 at 1e-14). Hence tol = 1e-14 here, which closes the round trip to ~1e-15.
    """

    def test_reversibility(self):
        for geo, seed in ((G4, 101), (G2_SU2, 102)):
            U, gen = haar_links(geo, 2, seed)
            P = H.random_momenta(U.shape, generator=gen)
            for reuse, tol, bound in ((False, 1e-14, 1e-12), (True, 1e-12, 1e-10)):
                ham = StaggeredHamiltonian(5.0, MASSES, geo, n_gauge_substeps=2, reuse_guess=reuse, solver_tol=tol)
                ham.refresh(U, gen)
                for integrator in H.INTEGRATORS:
                    with self.subTest(geometry=geo, reuse_guess=reuse, integrator=integrator):
                        hmc = H.HMC(geo, ham, tau=0.5, n_steps=4, integrator=integrator)
                        U1, P1, _ = hmc.integrate(U, P)
                        U2, P2, _ = hmc.integrate(U1, -P1)
                        dU, dP = (U2 - U).abs().max().item(), (P2 + P).abs().max().item()
                        if VERBOSE:
                            print(f"\n{geo} reuse_guess={reuse} {integrator}: |U2 - U| = {dU:.1e}, |P2 + P| = {dP:.1e}")
                        self.assertGreater((U1 - U).abs().max().item(), 0.1)
                        self.assertLess(dU, bound)
                        self.assertLess(dP, bound)


class EnergyScalingTest(unittest.TestCase):
    """4. |dH| ~ eps^2 for the two-level staggered HMC, leapfrog and Omelyan, 2^3 x 4 SU(3)."""

    STEPS = (8, 16, 32)

    def test_dH_scales_as_eps_squared(self):
        U, gen = haar_links(G4, 4, 111)
        P = H.random_momenta(U.shape, generator=gen)
        ham = StaggeredHamiltonian(5.0, 0.3, G4, n_gauge_substeps=2, solver_tol=1e-12)
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
                if VERBOSE:
                    print(f"\n{integrator}: <|dH|> = {['%.2e' % d for d in dH]}, slope {slope:.3f}")
                self.assertGreater(slope, 1.85, msg=f"dH={dH}")
                self.assertLess(slope, 2.15, msg=f"dH={dH}")
                mean_dH[integrator] = dH
        self.assertLess(mean_dH["omelyan"][-1], 0.25 * mean_dH["leapfrog"][-1])


class GroupManifoldTest(unittest.TestCase):
    """4. Links stay in SU(N) over staggered trajectories; solver bookkeeping (two action solves
    per trajectory: S_pf at the refresh links, cached, and at the end)."""

    def test_links_stay_in_sun(self):
        for geo, seed in ((G4, 121), (G2_SU2, 122)):
            with self.subTest(geometry=geo):
                U, gen = haar_links(geo, 4, seed)
                ham = StaggeredHamiltonian(5.0, 0.3, geo, n_gauge_substeps=2)
                hmc = H.HMC(geo, ham, tau=0.5, n_steps=3, integrator="omelyan")
                n_traj, n_acc = 3, 0
                for _ in range(n_traj):
                    U, accepted, dH, stats = hmc.trajectory(U, gen)
                    n_acc += int(accepted.sum())
                    unit, det = su_n_violation(U)
                    self.assertLess(unit, 1e-13)
                    self.assertLess(det, 1e-13)
                st = ham.fermion.stats
                self.assertGreater(n_acc, 0)
                self.assertEqual(st["refresh_calls"], n_traj)
                self.assertEqual(st["action_calls"], 2 * n_traj)
                self.assertEqual(st["force_calls"], n_traj * (2 * 3 + 1))
                self.assertEqual(st["solves"], st["force_calls"] + st["action_calls"])
                self.assertLess(st["last_residual"].max().item(), 1e-12)


class ExactComparisonTest(unittest.TestCase):
    """5. Staggered HMC (N_f = 2 in 2D) against the exact-determinant Metropolis, 2D 4 x 4 SU(3),
    beta = 5.0, B = 32 chains from a cold start (Metropolis: 40 thermalisation sweeps, HMC: 15
    trajectories). Observables: plaquette and log det M_ee, computed exactly on every
    configuration of both samplers; <exp(-dH)> = 1. Errors from the spread of the chain means.
    Omelyan, tau = 1: m = 0.3 with 4 outer steps x 3 gauge substeps, m = 0.1 with 10 x 2 (the
    fermion force at m = 0.1 on these rough fields makes 2-6 outer steps unstable, |dH| >> 1).
    m = 0.3 is a mild check; at m = 0.1 the exact sampler must differ from a quenched HMC in
    <log det> by > 4 sigma and the HMC must sit within 30% of that gap from the exact value.
    """

    B = 32
    BETA = 5.0

    def chain_stats(self, x):
        m = torch.stack(x).mean(0)
        return m.mean().item(), (m.std() / math.sqrt(self.B)).item()

    def observables(self, U, mass):
        return mean_plaquette(plaquette_fields(U, G2.D), G2.N), log_det_staggered(U, mass, G2)

    def run_hmc(self, ham, mass, n_therm, n_meas, seed, n_steps):
        gen = torch.Generator().manual_seed(seed)
        U = SUN_identity(G2.link_shape(self.B)[:-1], dtype=C)
        hmc = H.HMC(G2, ham, tau=1.0, n_steps=n_steps, integrator="omelyan")
        plaq, logdet, expdH, acc = [], [], [], []
        for i in range(n_therm + n_meas):
            U, accepted, dH, stats = hmc.trajectory(U, gen)
            if i >= n_therm:
                p, l = self.observables(U, mass)
                plaq.append(p)
                logdet.append(l)
                expdH.append(stats["exp_minus_dH"])
                acc.append(stats["acceptance"])
        return self.chain_stats(plaq), self.chain_stats(logdet), self.chain_stats(expdH), sum(acc) / len(acc)

    def run_metropolis(self, mass, n_therm, n_meas, seed):
        gen = torch.Generator().manual_seed(seed)
        m = ExactStaggeredMetropolis(self.BETA, mass, G2, eps=0.45, links_per_update=1)
        m.reset(SUN_identity(G2.link_shape(self.B)[:-1], dtype=C))
        n_links = G2.D * G2.volume
        plaq, logdet = [], []
        for sweep in range(n_therm + n_meas):
            for _ in range(n_links):
                m.step(gen)
            if sweep >= n_therm:
                p, l = self.observables(m.U, mass)
                plaq.append(p)
                logdet.append(l)
                self.assertLess((m.log_det_value - l).abs().max().item(), 1e-10)
        return self.chain_stats(plaq), self.chain_stats(logdet), m.acceptance

    def run_case(self, mass, discriminating, n_steps, n_gauge_substeps):
        ham = StaggeredHamiltonian(self.BETA, mass, G2, n_gauge_substeps=n_gauge_substeps)
        (p_h, e_ph), (l_h, e_lh), (x, e_x), acc = self.run_hmc(ham, mass, 15, 60, 300, n_steps)
        (p_m, e_pm), (l_m, e_lm), acc_m = self.run_metropolis(mass, 40, 60, 400)
        (p_q, e_pq), (l_q, e_lq), _, _ = self.run_hmc(H.WilsonGaugeHamiltonian(self.BETA, G2), mass, 15, 60, 500, 2)
        st = ham.fermion.stats
        msg = (f"2D 4x4 SU(3) staggered beta={self.BETA} m={mass}:\n"
               f"  <P>          HMC {p_h:.4f}({e_ph:.4f})  exact Metropolis {p_m:.4f}({e_pm:.4f})  "
               f"quenched HMC {p_q:.4f}({e_pq:.4f})\n"
               f"  <log det>    HMC {l_h:.4f}({e_lh:.4f})  exact Metropolis {l_m:.4f}({e_lm:.4f})  "
               f"quenched HMC {l_q:.4f}({e_lq:.4f})\n"
               f"  <exp(-dH)> = {x:.4f}({e_x:.4f}), HMC acceptance {acc:.3f}, Metropolis acceptance {acc_m:.3f}, "
               f"CG iterations per solve {st['cg_iterations'] / st['solves']:.1f}\n"
               f"  sigmas: <P> HMC-exact {abs(p_h - p_m) / math.hypot(e_ph, e_pm):.2f}, "
               f"<log det> HMC-exact {abs(l_h - l_m) / math.hypot(e_lh, e_lm):.2f}, "
               f"exact-quenched {abs(l_m - l_q) / math.hypot(e_lm, e_lq):.2f}, "
               f"<exp(-dH)>-1 {abs(x - 1.0) / e_x:.2f}; |l_hmc - l_exact| / |l_exact - l_quenched| = "
               f"{abs(l_h - l_m) / abs(l_m - l_q):.3f}")
        if VERBOSE:
            print("\n" + msg)
        self.assertGreater(acc, 0.8, msg=msg)
        self.assertGreater(acc_m, 0.3, msg=msg)
        self.assertLess(abs(x - 1.0), 4 * e_x, msg=msg)
        self.assertLess(abs(p_h - p_m), 4 * math.hypot(e_ph, e_pm), msg=msg)
        self.assertLess(abs(l_h - l_m), 4 * math.hypot(e_lh, e_lm), msg=msg)
        if discriminating:
            self.assertGreater(abs(l_m - l_q), 4 * math.hypot(e_lm, e_lq), msg=msg)
            self.assertLess(abs(l_h - l_m), 0.3 * abs(l_m - l_q), msg=msg)

    def test_mass_03(self):
        self.run_case(0.3, discriminating=False, n_steps=4, n_gauge_substeps=3)

    def test_mass_01(self):
        self.run_case(0.1, discriminating=True, n_steps=10, n_gauge_substeps=2)


if __name__ == "__main__":
    unittest.main()
