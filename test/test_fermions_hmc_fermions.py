"""Tests of the N_f = 2 Wilson HMC (pseudofermion outer level, gauge inner level), phase 1 of
fermions_plan.md, against an exact-determinant Metropolis (neoqcd.fermions.exact).

CPU, complex128, fixed seeds. Integrator properties on 2^3 x 4 SU(3); the comparison with the
exact sampler on 2D 4 x 4 SU(3), B = 32 chains (on 2^3 x 4 a dense 192 x 192 determinant per
proposal and chain makes the Metropolis too slow for a unit test: ~35 ms per proposal, 128
proposals per sweep).
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
    mean_plaquette,
    plaquette_fields,
)
from neoqcd.fermions.exact import ExactMetropolis, log_det_two_flavour
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.pseudofermion import TwoFlavourHamiltonian

G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4, SU(3)
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4, SU(3)

VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))  # print the diagnostics


def haar_links(geometry, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=gen), gen


def su_n_violation(U):
    eye = SUN_identity(U.shape[:-1], dtype=U.dtype)
    unit = (SUN_mul(SUN_dagger(U), U) - eye).abs().max().item()
    det = (torch.linalg.det(U) - 1.0).abs().max().item()
    return unit, det


class ReversibilityTest(unittest.TestCase):
    """5a. Forward, momentum flip, forward: back to (U, -P), phi fixed within the trajectory.

    With `reuse_guess=False` the force is a deterministic function of U, so the integrator is
    reversible to rounding whatever the CG tolerance; with the chronological guess the force
    depends on the history at the level of the solver tolerance.
    """

    def test_reversibility(self):
        U, gen = haar_links(G4, 3, 101)
        P = H.random_momenta(U.shape, generator=gen)
        for reuse, tol, bound in ((False, 1e-8, 1e-12), (True, 1e-12, 1e-10)):
            ham = TwoFlavourHamiltonian(5.0, 0.12, G4, n_gauge_substeps=2, reuse_guess=reuse, solver_tol=tol)
            ham.refresh(U, gen)
            for integrator in H.INTEGRATORS:
                with self.subTest(reuse_guess=reuse, integrator=integrator):
                    hmc = H.HMC(G4, ham, tau=0.5, n_steps=4, integrator=integrator)
                    U1, P1, _ = hmc.integrate(U, P)
                    U2, P2, _ = hmc.integrate(U1, -P1)
                    dU, dP = (U2 - U).abs().max().item(), (P2 + P).abs().max().item()
                    if VERBOSE:
                        print(f"\nreuse_guess={reuse} tol={tol:g} {integrator}: |U2 - U| = {dU:.1e}, |P2 + P| = {dP:.1e}")
                    self.assertGreater((U1 - U).abs().max().item(), 0.1)
                    self.assertLess(dU, bound)
                    self.assertLess(dP, bound)
                    H0, H2 = ham.energy(U, P), ham.energy(U2, P2)
                    self.assertLess(((H2 - H0).abs() / H0.abs()).max().item(), 1e-12)


class EnergyScalingTest(unittest.TestCase):
    """5b. |dH| ~ eps^2 for the two-level fermionic HMC, leapfrog and Omelyan, and the number of
    force evaluations (merged kicks) on each level."""

    STEPS = (8, 16, 32)

    def test_dH_scales_as_eps_squared(self):
        U, gen = haar_links(G4, 4, 111)
        P = H.random_momenta(U.shape, generator=gen)
        n_sub = 2
        ham = TwoFlavourHamiltonian(5.0, 0.12, G4, n_gauge_substeps=n_sub, solver_tol=1e-12)
        ham.refresh(U, gen)
        mean_dH = {}
        for integrator in H.INTEGRATORS:
            with self.subTest(integrator=integrator):
                hmc = H.HMC(G4, ham, tau=0.5, n_steps=1, integrator=integrator)
                dH = []
                for n in self.STEPS:
                    calls = ham.fermion.stats["force_calls"]
                    U1, P1, evals = hmc.integrate(U, P, n_steps=n)
                    dH.append(ham.energy_difference(U, P, U1, P1).abs().mean().item())
                    per_step = 1 if integrator == "leapfrog" else 2
                    self.assertEqual(evals[0], per_step * n + 1)
                    self.assertEqual(evals[1], n_sub * per_step * n * per_step + 1)
                    self.assertEqual(ham.fermion.stats["force_calls"] - calls, evals[0])
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
    """7. Links stay in SU(3) over fermionic trajectories; refresh, solves and the cached
    S_pf(U_old) = eta^dagger eta are used as intended (one action solve per trajectory)."""

    def test_links_stay_in_sun_and_solver_bookkeeping(self):
        U, gen = haar_links(G4, 4, 121)
        ham = TwoFlavourHamiltonian(5.0, 0.12, G4, n_gauge_substeps=2)
        hmc = H.HMC(G4, ham, tau=0.5, n_steps=3, integrator="omelyan")
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
        self.assertEqual(st["action_calls"], n_traj)
        self.assertEqual(st["force_calls"], n_traj * (2 * 3 + 1))
        self.assertEqual(st["solves"], st["force_calls"] + st["action_calls"])
        self.assertLess(st["last_residual"].max().item(), 1e-12)
        if VERBOSE:
            print(f"\nstats: {st['solves']} solves, {st['cg_iterations']} CG iterations "
                  f"({st['cg_iterations'] / st['solves']:.1f} per solve, max over the batch)")


class ExactComparisonTest(unittest.TestCase):
    """6. Fermionic HMC against the exact-determinant Metropolis, 2D 4 x 4 SU(3), beta = 5.0,
    B = 32 chains from a cold start. Observables: plaquette and log det(D^dagger D) (computed
    exactly on every configuration of both samplers; it is the most sensitive probe of the
    fermion weight). Also <exp(-dH)> = 1. Errors from the spread of the 32 chain means.

    kappa = 0.12 is the point of the plan; in 2D it is a heavy quark (free kappa_c = 1/4) and the
    fermion effect is below the errors, so a quenched HMC run is printed as reference. At kappa = 0.22
    the test discriminates dynamical from quenched: <log det> of the exact sampler differs from the
    quenched one by many sigma (asserted), <P> by about 4 sigma, and the HMC <log det> must lie
    within 30% of that quenched gap from the exact one. It does not pin the normalisation of the
    determinant: an HMC with S_pf and its force halved (N_f = 1) passes these assertions with this
    statistics. N_f = 2 versus 1 is pinned by StochasticEstimatorTest and ForceExpectationTest in
    test_fermions_pseudofermion.py.

    The Metropolis chains start cold and are thermalised for 40 sweeps (tau_exp ~ 5 sweeps at
    eps = 0.45), the HMC chains for 15 trajectories.
    """

    B = 32
    BETA = 5.0

    def chain_stats(self, x):
        m = torch.stack(x).mean(0)
        return m.mean().item(), (m.std() / math.sqrt(self.B)).item()

    def observables(self, U, kappa):
        return mean_plaquette(plaquette_fields(U, G2.D), G2.N), log_det_two_flavour(U, kappa, G2, method="assembled")

    def run_hmc(self, ham, kappa, n_therm, n_meas, seed):
        gen = torch.Generator().manual_seed(seed)
        U = SUN_identity(G2.link_shape(self.B)[:-1], dtype=torch.cdouble)
        hmc = H.HMC(G2, ham, tau=1.0, n_steps=2, integrator="omelyan")
        plaq, logdet, expdH, acc = [], [], [], []
        for i in range(n_therm + n_meas):
            U, accepted, dH, stats = hmc.trajectory(U, gen)
            if i >= n_therm:
                p, l = self.observables(U, kappa)
                plaq.append(p)
                logdet.append(l)
                expdH.append(stats["exp_minus_dH"])
                acc.append(stats["acceptance"])
        return self.chain_stats(plaq), self.chain_stats(logdet), self.chain_stats(expdH), sum(acc) / len(acc)

    def run_metropolis(self, kappa, n_therm, n_meas, seed):
        gen = torch.Generator().manual_seed(seed)
        m = ExactMetropolis(self.BETA, kappa, G2, eps=0.45, links_per_update=1)
        m.reset(SUN_identity(G2.link_shape(self.B)[:-1], dtype=torch.cdouble))
        n_links = G2.D * G2.volume
        plaq, logdet = [], []
        for sweep in range(n_therm + n_meas):
            for _ in range(n_links):
                m.step(gen)
            if sweep >= n_therm:
                p, l = self.observables(m.U, kappa)
                plaq.append(p)
                logdet.append(l)
                # the cached log det of the sampler is the exact one of its current links
                self.assertLess((m.log_det_value - l).abs().max().item(), 1e-10)
        return self.chain_stats(plaq), self.chain_stats(logdet), m.acceptance

    def run_case(self, kappa, discriminating):
        ham = TwoFlavourHamiltonian(self.BETA, kappa, G2, n_gauge_substeps=3)
        (p_h, e_ph), (l_h, e_lh), (x, e_x), acc = self.run_hmc(ham, kappa, 15, 60, 300)
        (p_m, e_pm), (l_m, e_lm), acc_m = self.run_metropolis(kappa, 40, 60, 400)
        (p_q, e_pq), (l_q, e_lq), _, _ = self.run_hmc(H.WilsonGaugeHamiltonian(self.BETA, G2), kappa, 15, 60, 500)
        st = ham.fermion.stats
        msg = (f"2D 4x4 SU(3) beta={self.BETA} kappa={kappa}:\n"
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

    def test_kappa_012(self):
        self.run_case(0.12, discriminating=False)

    def test_kappa_022(self):
        self.run_case(0.22, discriminating=True)


if __name__ == "__main__":
    unittest.main()
