"""Tests of the pure-gauge HMC of neoqcd.fermions.hmc (track B1 of fermions_plan.md).

CPU, complex128, fixed seeds, 2^4 lattices with SU(2) and SU(3).
"""

import math
import os
import unittest
from types import SimpleNamespace

import torch

from neoqcd.fermions import hmc as H
from neoqcd.fermions._gauge import (
    HBOR,
    SUN_dagger,
    SUN_haar,
    SUN_identity,
    SUN_mul,
    SUN_trace,
    Wilson_action,
    create_mask,
    mean_plaquette,
    plaquette_fields,
)
from neoqcd.fermions.algebra import (
    algebra_basis,
    algebra_components,
    expi,
    random_algebra,
    trace_product,
    unitarize,
)
from neoqcd.fermions.lattice import Geometry

GROUPS = (2, 3)
VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))  # print the end-to-end summaries


def geometry(N):
    return Geometry(D=4, T=2, L=2, N=N)


def haar_links(g, B, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(g.link_shape(B)[:-1], generator=gen), gen


def wilson_reference(g, beta, defect_par=1.0):
    return Wilson_action(SimpleNamespace(D=g.D, N=g.N, device=torch.device("cpu")), beta, defect_par)


def defect_mask(g):
    """A defect on the time links at t = 0, x = 0 (all y, z), in the layout of utils.Defect."""
    d = torch.zeros((g.D,) + g.lattice, dtype=torch.double)
    d[0, 0, 0] = 1.0
    return torch.stack((1.0 - d, d), dim=0)


def su_n_violation(U):
    eye = SUN_identity(U.shape[:-1], dtype=U.dtype)
    unit = (SUN_mul(SUN_dagger(U), U) - eye).abs().max().item()
    det = (torch.linalg.det(U) - 1.0).abs().max().item()
    return unit, det


class AlgebraTest(unittest.TestCase):
    def test_basis_is_orthonormal_hermitian_traceless(self):
        for N in GROUPS:
            T = algebra_basis(N)
            self.assertEqual(T.shape[0], N * N - 1)
            gram = trace_product(T[:, None], T[None, :])
            self.assertLess((gram - torch.eye(N * N - 1, dtype=gram.dtype)).abs().max().item(), 1e-14)
            self.assertLess((T - SUN_dagger(T)).abs().max().item(), 1e-15)
            self.assertLess(SUN_trace(T).abs().max().item(), 1e-15)

    def test_exponential_matches_matrix_exp(self):
        gen = torch.Generator().manual_seed(11)
        for N in GROUPS:
            # small and large arguments: c1 = tr(Q^2)/2 on both sides of the series threshold 3
            for scale in (0.0, 1e-4, 0.3, 2.0):
                Q = scale * random_algebra((200, N, N), generator=gen)
                ref = torch.linalg.matrix_exp(1j * Q)
                self.assertLess((expi(Q) - ref).abs().max().item(), 1e-13, msg=f"N={N} scale={scale}")
            # around the threshold itself: Q normalised so that c1 = tr(Q^2)/2 is exactly 2.9, 3.0, 3.1
            for c1 in (2.9, 3.0, 3.1):
                Q = random_algebra((200, N, N), generator=gen)
                Q = Q * torch.sqrt(c1 / (0.5 * trace_product(Q, Q)))[..., None, None]
                self.assertLess((0.5 * trace_product(Q, Q) - c1).abs().max().item(), 1e-13)
                ref = torch.linalg.matrix_exp(1j * Q)
                self.assertLess((expi(Q) - ref).abs().max().item(), 1e-13, msg=f"N={N} c1={c1}")

    def test_su2_closed_exponential_gradient_at_zero(self):
        # the SU(2) closed form must be differentiable at Q = 0: d exp(iQ)/dQ = i at Q = 0
        Q = torch.zeros((5, 2, 2), dtype=torch.cdouble, requires_grad=True)
        X = random_algebra((5, 2, 2), generator=torch.Generator().manual_seed(13))
        out = trace_product(expi(Q, method="closed"), 1j * X).sum()  # Re tr(exp(iQ) i X)
        (grad,) = torch.autograd.grad(out, Q)
        self.assertTrue(torch.isfinite(grad.real).all() and torch.isfinite(grad.imag).all())
        # first order: Re tr(i Q i X) = -tr(Q X) for hermitian Q, X, so d out / dQ = -X^T (conj grad)
        dir_deriv = (grad.conj() * X).sum().real.item()
        self.assertLess(abs(dir_deriv + trace_product(X, X).sum().item()), 1e-12)

    def test_unitarize_projects_onto_sun(self):
        gen = torch.Generator().manual_seed(12)
        for N in GROUPS:
            U = SUN_haar((50, N), generator=gen)
            self.assertLess((unitarize(U) - U).abs().max().item(), 1e-14)
            noisy = U + 1e-6 * torch.randn(U.shape, generator=gen, dtype=U.dtype)
            unit, det = su_n_violation(unitarize(noisy))
            self.assertLess(max(unit, det), 1e-14)
            self.assertLess((unitarize(noisy) - U).abs().max().item(), 1e-5)


class MomentumTest(unittest.TestCase):
    """1. P = sum_a p_a T^a, tr(T^a T^b) = delta_ab, p_a ~ N(0, 1): <tr P^2> = N^2 - 1 per link."""

    def test_momentum_normalisation(self):
        for N in GROUPS:
            with self.subTest(N=N):
                g = geometry(N)
                gen = torch.Generator().manual_seed(100 + N)
                # B = 4096 configurations of 2^4 x 4 links: M = 262144 links, so the 5 sigma
                # tolerance on the unit covariance, 5 sqrt(2 / M), is about 1.4%
                B = 4096
                P = H.random_momenta(g.link_shape(B), generator=gen)
                self.assertLess((P - SUN_dagger(P)).abs().max().item(), 1e-15)
                self.assertLess(SUN_trace(P).abs().max().item(), 1e-14)

                n_gen = N * N - 1
                p = algebra_components(P).reshape(-1, n_gen)
                M = p.shape[0]
                trP2 = trace_product(P, P).reshape(-1)
                # the components reproduce P and tr P^2 = sum_a p_a^2 exactly
                self.assertLess((trP2 - (p ** 2).sum(-1)).abs().max().item(), 1e-12)
                self.assertLess((torch.einsum("...a,aij->...ij", p.to(P.dtype), algebra_basis(N))
                                 - P.reshape(-1, N, N)).abs().max().item(), 1e-14)
                # <tr P^2> = N^2 - 1, Var(tr P^2) = 2 (N^2 - 1)
                sigma = math.sqrt(2.0 * n_gen / M)
                self.assertLess(abs(trP2.mean().item() - n_gen), 5 * sigma)
                # components: zero mean, unit covariance
                self.assertLess(p.mean(0).abs().max().item(), 5 / math.sqrt(M))
                cov = p.T @ p / M
                self.assertLess((cov - torch.eye(n_gen, dtype=cov.dtype)).abs().max().item(),
                                5 * math.sqrt(2.0 / M))
                # kinetic energy = (1/2) sum tr P^2 per configuration
                K = H.kinetic_energy(P)
                self.assertEqual(K.shape, (B,))
                self.assertLess((K.sum() - 0.5 * trP2.sum()).abs().item(), 1e-8)


class GaugeForceTest(unittest.TestCase):
    """2. tr(F X) = d/dt S_g(exp(i t X) U) at t = 0, from finite differences of Wilson_action."""

    T_FD = 1e-3

    def fd_derivative(self, action, U, X):
        t = self.T_FD
        f = lambda s: action(SUN_mul(expi(s * X), U))
        return (-f(2 * t) + 8 * f(t) - 8 * f(-t) + f(-2 * t)) / (12 * t)

    def cases(self, N):
        """(name, geometry, WilsonGaugeAction, reference action V -> (3,)), batch B = 3."""
        g = geometry(N)
        beta = torch.tensor([5.0, 2.3, 6.7], dtype=torch.double)
        mask = defect_mask(g)
        yield "plain", g, H.WilsonGaugeAction(beta, g), lambda V: wilson_reference(g, beta)(V, None)
        w = H.defect_plaquette_weights(mask, 0.37, g.D)
        yield ("defect", g, H.WilsonGaugeAction(beta[1], g, w),
               lambda V: wilson_reference(g, beta[1], 0.37)(V, mask))
        # per-batch beta (B,) together with (lattice-shaped) defect weights
        yield ("batch-beta+defect", g, H.WilsonGaugeAction(beta, g, w),
               lambda V: wilson_reference(g, beta, 0.37)(V, mask))
        # per-entry plaquette weights (B,) + lattice: one defect parameter per configuration,
        # reference = Wilson_action entry by entry
        c = torch.tensor([0.37, 0.8, 0.05], dtype=torch.double)
        wb = H.batched_defect_plaquette_weights(mask, c, g.D)

        def per_entry(V):
            return torch.cat([wilson_reference(g, beta[b].item(), c[b].item())(V[b:b + 1], mask)
                              for b in range(V.shape[0])])
        yield "per-entry-weights", g, H.WilsonGaugeAction(beta, g, wb), per_entry

    def test_batched_defect_weights_match_scalar_helper(self):
        for N in GROUPS:
            g = geometry(N)
            mask = defect_mask(g)
            c = torch.tensor([0.37, 1.0, 0.0, 0.8], dtype=torch.double)
            wb = H.batched_defect_plaquette_weights(mask, c, g.D)
            self.assertEqual(len(wb), g.D * (g.D - 1) // 2)
            for b in range(c.numel()):
                ws = H.defect_plaquette_weights(mask, c[b].item(), g.D)
                self.assertEqual(set(wb), set(ws))
                for plane in ws:
                    self.assertEqual(wb[plane].shape, (c.numel(),) + g.lattice)
                    self.assertTrue(torch.equal(wb[plane][b], ws[plane]), msg=f"N={N} b={b} {plane}")
            # the defect is visible (weights != 1 somewhere) and absent for c = 1
            self.assertLess(min(w[0].min().item() for w in wb.values()), 1.0)
            self.assertTrue(all(torch.equal(w[1], torch.ones_like(w[1])) for w in wb.values()))

    def test_action_matches_wilson_action(self):
        for N in GROUPS:
            for name, g, S, ref_action in self.cases(N):
                with self.subTest(N=N, case=name):
                    U, _ = haar_links(g, 3, 20 + N)
                    ref = ref_action(U)
                    self.assertLess((S.action(U) - ref).abs().max().item(), 1e-12 * ref.abs().max().item())
                    V, _ = haar_links(g, 3, 30 + N)
                    diff = S.action_difference(U, V)
                    ref_diff = ref_action(V) - ref
                    self.assertLess((diff - ref_diff).abs().max().item(), 1e-10)

    def test_force_against_finite_differences(self):
        for N in GROUPS:
            for name, g, S, action in self.cases(N):
                with self.subTest(N=N, case=name):
                    U, gen = haar_links(g, 3, 40 + N)
                    F = S.force(U)
                    self.assertLess((F - SUN_dagger(F)).abs().max().item(), 1e-12)
                    self.assertLess(SUN_trace(F).abs().max().item(), 1e-12)
                    # global random directions in the algebra, all links at once
                    for _ in range(3):
                        X = random_algebra(U.shape, generator=gen)
                        pred = H.link_sum(trace_product(F, X))
                        fd = self.fd_derivative(action, U, X)
                        rel = ((pred - fd).abs() / fd.abs()).max().item()
                        self.assertLess(rel, 1e-6, msg=f"pred={pred} fd={fd}")
                    # single links, every generator: tr(F T^a) at that link
                    T = algebra_basis(N)
                    for link in [(0, 0, 0, 0, 0, 0), (1, 2, 1, 0, 1, 1), (2, 3, 0, 1, 0, 1), (1, 0, 0, 0, 1, 0)]:
                        b = link[0]
                        for a in range(N * N - 1):
                            X = torch.zeros_like(U)
                            X[link] = T[a]
                            pred = trace_product(F[link], T[a]).item()
                            fd = self.fd_derivative(action, U, X)[b].item()
                            self.assertLess(abs(pred - fd), 1e-6 * max(abs(fd), 1.0), msg=f"link={link} a={a}")


class ReversibilityTest(unittest.TestCase):
    """3. Forward, momentum flip, forward again: back to (U, -P) at machine precision."""

    def hamiltonians(self, g):
        yield "one-level", H.WilsonGaugeHamiltonian(5.0, g)
        # stand-in for phase 1: an outer "fermion" level (here part of the gauge action) and an
        # inner gauge level with 3 substeps
        yield "two-level", H.Hamiltonian([(H.WilsonGaugeAction(1.5, g), 1), (H.WilsonGaugeAction(3.5, g), 3)])

    def test_reversibility(self):
        for N in GROUPS:
            g = geometry(N)
            U, gen = haar_links(g, 4, 50 + N)
            P = H.random_momenta(U.shape, generator=gen)
            for hname, ham in self.hamiltonians(g):
                for integrator in H.INTEGRATORS:
                    with self.subTest(N=N, hamiltonian=hname, integrator=integrator):
                        hmc = H.HMC(g, ham, tau=1.0, n_steps=10, integrator=integrator)
                        U1, P1, _ = hmc.integrate(U, P)
                        U2, P2, _ = hmc.integrate(U1, -P1)
                        self.assertGreater((U1 - U).abs().max().item(), 0.1)  # it did move
                        self.assertLess((U2 - U).abs().max().item(), 1e-10)
                        self.assertLess((P2 + P).abs().max().item(), 1e-10)
                        H0, H2 = ham.energy(U, P), ham.energy(U2, P2)
                        self.assertLess(((H2 - H0).abs() / H0.abs()).max().item(), 1e-13)

    def test_per_level_integrators_reversibility_and_force_counts(self):
        # level 0: n_substeps 2, n_steps 4 -> n = 8 outer steps; level 1: n_substeps 3.
        # Merged kicks: an outer leapfrog step costs 1 level-0 force and has 1 drift, an outer
        # Omelyan step costs 2 and has 2 drifts; each drift runs 3 inner steps (1 force each for
        # leapfrog, 2 for Omelyan); inner kicks merge across outer drifts (no move in between),
        # so every level gets one extra evaluation at the very end:
        #   [leapfrog, omelyan]: [8 + 1, 8 * 3 * 2 + 1] = [9, 49]
        #   [omelyan, leapfrog]: [2 * 8 + 1, 16 * 3 * 1 + 1] = [17, 49]
        expected = {("leapfrog", "omelyan"): [9, 49], ("omelyan", "leapfrog"): [17, 49]}
        for N in GROUPS:
            g = geometry(N)
            U, gen = haar_links(g, 4, 80 + N)
            P = H.random_momenta(U.shape, generator=gen)
            ham = H.Hamiltonian([(H.WilsonGaugeAction(1.5, g), 2), (H.WilsonGaugeAction(3.5, g), 3)])
            for integrator, counts in expected.items():
                with self.subTest(N=N, integrator=integrator):
                    hmc = H.HMC(g, ham, tau=1.0, n_steps=4, integrator=list(integrator))
                    U1, P1, evals = hmc.integrate(U, P)
                    self.assertEqual(evals, counts)
                    U2, P2, evals2 = hmc.integrate(U1, -P1)
                    self.assertEqual(evals2, counts)
                    self.assertGreater((U1 - U).abs().max().item(), 0.1)
                    self.assertLess((U2 - U).abs().max().item(), 1e-10)
                    self.assertLess((P2 + P).abs().max().item(), 1e-10)


class EnergyScalingTest(unittest.TestCase):
    """4. |dH| ~ eps^2 for leapfrog and Omelyan, one and two time-scale levels."""

    STEPS = (10, 20, 40)

    def test_dH_scales_as_eps_squared(self):
        for N in GROUPS:
            g = geometry(N)
            U, gen = haar_links(g, 8, 60 + N)
            P = H.random_momenta(U.shape, generator=gen)
            hams = (("one-level", H.WilsonGaugeHamiltonian(5.0, g)),
                    ("two-level", H.Hamiltonian([(H.WilsonGaugeAction(1.5, g), 1),
                                                 (H.WilsonGaugeAction(3.5, g), 3)])))
            for hname, ham in hams:
                mean_dH = {}
                for integrator in H.INTEGRATORS:
                    with self.subTest(N=N, hamiltonian=hname, integrator=integrator):
                        hmc = H.HMC(g, ham, tau=1.0, n_steps=1, integrator=integrator)
                        dH = []
                        for n in self.STEPS:
                            U1, P1, evals = hmc.integrate(U, P, n_steps=n)
                            dH.append(ham.energy_difference(U, P, U1, P1).abs().mean().item())
                            # the site-by-site difference agrees with the difference of the totals
                            direct = (ham.energy(U1, P1) - ham.energy(U, P)).abs().mean().item()
                            self.assertLess(abs(direct - dH[-1]), 1e-9)
                        # force evaluations with merged kicks: n + 1 (leapfrog), 2 n + 1 (Omelyan)
                        per_step = 1 if integrator == "leapfrog" else 2
                        expected = [per_step * n + 1]
                        if hname == "two-level":
                            expected.append(3 * per_step * n * (2 if integrator == "omelyan" else 1) + 1)
                        self.assertEqual(evals, expected)
                        x = torch.tensor([math.log(1.0 / n) for n in self.STEPS], dtype=torch.double)
                        y = torch.tensor([math.log(d) for d in dH], dtype=torch.double)
                        slope = (((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()).item()
                        self.assertGreater(slope, 1.9, msg=f"dH={dH}")
                        self.assertLess(slope, 2.1, msg=f"dH={dH}")
                        mean_dH[integrator] = dH
                # at equal step count the Omelyan error is several times smaller
                self.assertLess(mean_dH["omelyan"][-1], 0.25 * mean_dH["leapfrog"][-1])


class GroupManifoldTest(unittest.TestCase):
    """5. Links stay in SU(N) after trajectories (reunitarised at the end of each one)."""

    def test_links_stay_in_sun(self):
        for N in GROUPS:
            with self.subTest(N=N):
                g = geometry(N)
                U, gen = haar_links(g, 4, 70 + N)
                hmc = H.HMC(g, H.WilsonGaugeHamiltonian(5.0, g), tau=1.0, n_steps=8, integrator="omelyan")
                # before reunitarisation the drift is at rounding level
                P = H.random_momenta(U.shape, generator=gen)
                U1, _, _ = hmc.integrate(U, P)
                self.assertLess(max(su_n_violation(U1)), 1e-12)
                n_acc = 0
                for _ in range(5):
                    U, accepted, dH, stats = hmc.trajectory(U, gen)
                    n_acc += int(accepted.sum())
                    unit, det = su_n_violation(U)
                    self.assertLess(unit, 1e-13)
                    self.assertLess(det, 1e-13)
                    self.assertEqual(accepted.shape, (4,))
                    self.assertEqual(dH.shape, (4,))
                self.assertGreater(n_acc, 0)


class EndToEndTest(unittest.TestCase):
    """6. <exp(-dH)> = 1 and <P> from HMC equal to <P> from the heat bath HBOR, 2^4, B = 64 chains.

    Errors from the spread of the 64 independent chain means.
    """

    B = 64

    def chain_stats(self, x):
        m = x.mean(0)  # time average per chain
        return m.mean().item(), (m.std() / math.sqrt(self.B)).item()

    def run_case(self, N, beta, integrator, n_steps, n_therm, n_meas):
        g = geometry(N)
        gen = torch.Generator().manual_seed(1000 + N)
        U = SUN_identity(g.link_shape(self.B)[:-1], dtype=torch.cdouble)
        hmc = H.HMC(g, H.WilsonGaugeHamiltonian(beta, g), tau=1.0, n_steps=n_steps, integrator=integrator)
        plaq, expdH, acc = [], [], []
        for i in range(n_therm + n_meas):
            U, accepted, dH, stats = hmc.trajectory(U, gen)
            if i >= n_therm:
                plaq.append(stats["plaquette"])
                expdH.append(stats["exp_minus_dH"])
                acc.append(stats["acceptance"])
        p_hmc, e_hmc = self.chain_stats(torch.stack(plaq))
        x_mean, x_err = self.chain_stats(torch.stack(expdH))

        torch.manual_seed(2000 + N)  # HBOR draws from the global generator
        hbor = HBOR(SimpleNamespace(N=N, D=g.D, orsteps=1, device=torch.device("cpu")), beta=beta)
        mask = create_mask(g.D, g.T, g.L)
        V = SUN_identity(g.link_shape(self.B)[:-1], dtype=torch.cdouble)
        plaq_hb = []
        for i in range(n_therm + n_meas):
            V = hbor(V, mask, None)
            if i >= n_therm:
                plaq_hb.append(mean_plaquette(plaquette_fields(V, g.D), N))
        p_hb, e_hb = self.chain_stats(torch.stack(plaq_hb))

        sigma = math.hypot(e_hmc, e_hb)
        msg = (f"SU({N}) beta={beta} {integrator}: <P>_HMC = {p_hmc:.5f}({e_hmc:.5f}), "
               f"<P>_HB = {p_hb:.5f}({e_hb:.5f}), <exp(-dH)> = {x_mean:.5f}({x_err:.5f}), "
               f"acceptance {sum(acc) / len(acc):.3f}")
        if VERBOSE:
            print("\n" + msg)
        self.assertGreater(sum(acc) / len(acc), 0.8, msg=msg)
        self.assertLess(abs(x_mean - 1.0), 4 * x_err, msg=msg)
        self.assertLess(abs(p_hmc - p_hb), 4 * sigma, msg=msg)

    def test_su3_omelyan_against_heatbath(self):
        self.run_case(N=3, beta=5.0, integrator="omelyan", n_steps=5, n_therm=40, n_meas=200)

    def test_su2_leapfrog_against_heatbath(self):
        self.run_case(N=2, beta=2.0, integrator="leapfrog", n_steps=8, n_therm=40, n_meas=200)


if __name__ == "__main__":
    unittest.main()
