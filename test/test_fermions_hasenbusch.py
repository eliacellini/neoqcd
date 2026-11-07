"""Tests of Hasenbusch mass preconditioning (neoqcd.fermions.hasenbusch) for the N_f = 2 even-odd
Wilson HMC: exactness of the factorisation, forces of the heavy and ratio terms, HMC with the
Hasenbusch Hamiltonian against the single-pseudofermion HMC, and the CG cost.

CPU, complex128, fixed seeds. Lattices: 2^3 x 4 SU(3) (even-odd dimension 192) and 2D 4 x 4 SU(3)
(48). Set NEOQCD_TEST_VERBOSE=1 to print the diagnostics.
"""

import math
import os
import unittest

import torch

from neoqcd.fermions import hmc as H
from neoqcd.fermions._gauge import SUN_dagger, SUN_haar, SUN_identity, SUN_mul, SUN_trace, mean_plaquette, plaquette_fields
from neoqcd.fermions.algebra import algebra_basis, expi, random_algebra, trace_product
from neoqcd.fermions.exact import assemble_even_odd, log_det_two_flavour
from neoqcd.fermions.hasenbusch import HasenbuschHamiltonian, HeavyWilsonTerm, RatioWilsonTerm
from neoqcd.fermions.hmc import link_sum
from neoqcd.fermions.lattice import Geometry, parity
from neoqcd.fermions.pseudofermion import TwoFlavourHamiltonian, TwoFlavourWilson

G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4, SU(3)
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4, SU(3)
C = torch.complex128
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


def rms_force(F):
    """sqrt(<tr F^2>) per link, averaged over links and batch."""
    return math.sqrt(trace_product(F, F).mean().item())


class ExactnessTest(unittest.TestCase):
    """1. <exp(-[S_1 + S_2](U') + [S_1 + S_2](U))>_{phi_1, phi_2 at U} = det M(U') / det M(U) with
    M = D_hat^dagger D_hat at the target kappa, from dense determinants. Each factor is also checked
    on its own (heavy: det M'(U') / det M'(U); ratio: the quotient of the two). U' = exp(i c X) U
    with X the unit-norm average single-term force at kappa over 16 phi (as in the phase 1 test)."""

    def test_refresh_gives_eta_norm(self):
        U, gen = haar_links(G2, 4, 201)
        S = RatioWilsonTerm(0.20, 0.15, G2, action_tol=1e-13)
        S.refresh(U, gen)
        s = S.action(U)
        self.assertEqual(S.operator(U).odd(S.phi).abs().max().item(), 0.0)
        self.assertLess(((s - S._refresh_action).abs() / s).max().item(), 1e-11)

    def test_determinant_ratio_2d(self):
        geo, kappa, kappa_h, K = G2, 0.20, 0.15, 1000
        U1, gen = haar_links(geo, 1, 202)
        probe = TwoFlavourWilson(kappa, geo)
        probe.refresh(replicate(U1, 16), gen)
        X1 = probe.force(replicate(U1, 16)).mean(0, keepdim=True)
        X1 = X1 / math.sqrt(link_sum(trace_product(X1, X1)).item())
        U = replicate(U1, K)
        for c in (0.2, -0.2):
            V1 = SUN_mul(expi(c * X1), U1)
            pair = torch.cat((U1, V1))
            ld = log_det_two_flavour(pair, kappa, geo)
            ld_h = log_det_two_flavour(pair, kappa_h, geo)
            exact = {"product": math.exp((ld[1] - ld[0]).item()),
                     "heavy": math.exp((ld_h[1] - ld_h[0]).item())}
            exact["ratio"] = exact["product"] / exact["heavy"]
            V = replicate(V1, K)
            heavy = HeavyWilsonTerm(kappa_h, geo)
            ratio = RatioWilsonTerm(kappa, kappa_h, geo)
            heavy.refresh(U, gen)
            ratio.refresh(U, gen)
            d1 = heavy.action_difference(U, V)
            d2 = ratio.action_difference(U, V)
            self.assertEqual(ratio.stats["action_calls"], 1)  # S_2(U) from the refresh cache
            for name, d in (("product", d1 + d2), ("heavy", d1), ("ratio", d2)):
                with self.subTest(c=c, term=name):
                    w = torch.exp(-d)
                    est, err = w.mean().item(), w.std().item() / math.sqrt(K)
                    if VERBOSE:
                        print(f"\nc={c:+.1f} {name}: exact {exact[name]:.4f}, stochastic {est:.4f} +- {err:.4f} "
                              f"({abs(est - exact[name]) / err:.2f} sigma, |exact - 1| = {abs(exact[name] - 1) / err:.1f} sigma)")
                    if name == "product":
                        self.assertGreater(abs(exact[name] - 1.0), 5 * err)  # discriminating
                    self.assertLess(abs(est - exact[name]), 4 * err)


def dense_actions(U, phi_h, phi_r, kappa, kappa_h, geo):
    """S_1 = |D_hat'^-dagger phi_1|^2 and S_2 = |D_hat^-dagger D_hat'^dagger phi_2|^2 from dense
    even-odd matrices (differentiable in U by autograd), shape (B,) each."""
    B = U.shape[0]
    even = (parity(geo).reshape(-1) == 0).nonzero().squeeze(-1)

    def ev(phi):
        return phi.reshape(B, geo.volume, -1)[:, even].reshape(B, -1, 1)

    Dl = assemble_even_odd(U, kappa, geo)
    Dh = assemble_even_odd(U, kappa_h, geo)
    dag = lambda A: A.conj().transpose(-1, -2)
    y1 = torch.linalg.solve(dag(Dh), ev(phi_h))
    y2 = torch.linalg.solve(dag(Dl), torch.matmul(dag(Dh), ev(phi_r)))
    norm = lambda y: (y.conj() * y).real.reshape(B, -1).sum(1)
    return norm(y1), norm(y2)


class ForceTest(unittest.TestCase):
    """2. Forces of the heavy and ratio terms against 4th-order finite differences of their actions
    (random directions on all links and single-link generators, including time links at t = 0,
    which cross the time boundary) and against autograd through dense solves. Per-batch kappa."""

    LINKS = {4: [(0, 0, 0, 0, 0, 0), (1, 0, 0, 1, 0, 1), (1, 1, 2, 1, 0, 1), (0, 3, 3, 1, 1, 0)],
             2: [(0, 0, 0, 0), (1, 0, 0, 2), (1, 1, 3, 2), (0, 1, 1, 0)]}

    def cases(self):
        yield "2D", G2, (0.20, 0.17), (0.15, 0.12), -1.0, 301
        yield "2D periodic", G2, (0.20, 0.17), (0.15, 0.12), 1.0, 302
        yield "4D", G4, (0.12, 0.10), (0.09, 0.07), -1.0, 303

    def terms(self, geo, k, kh, time_sign):
        k = torch.tensor(k, dtype=torch.float64)
        kh = torch.tensor(kh, dtype=torch.float64)
        kw = dict(time_sign=time_sign, solver_tol=1e-13, action_tol=1e-13)
        return k, kh, {"heavy": HeavyWilsonTerm(kh, geo, **kw), "ratio": RatioWilsonTerm(k, kh, geo, **kw)}

    def test_force_against_finite_differences(self):
        for name, geo, k, kh, ts, seed in self.cases():
            U, gen = haar_links(geo, 2, seed)
            _, _, terms = self.terms(geo, k, kh, ts)
            for tname, S in terms.items():
                with self.subTest(case=name, term=tname):
                    S.refresh(U, gen)
                    F = S.force(U)
                    self.assertEqual(F.shape, U.shape)
                    self.assertLess((F - SUN_dagger(F)).abs().max().item(), 1e-13)
                    self.assertLess(SUN_trace(F).abs().max().item(), 1e-13)
                    for _ in range(2):
                        X = random_algebra(U.shape, generator=gen)
                        pred = link_sum(trace_product(F, X))
                        fd = fd_derivative(S.action, U, X)
                        self.assertLess(((pred - fd).abs() / fd.abs()).max().item(), 1e-6, msg=f"{pred} {fd}")
                    T = algebra_basis(geo.N)
                    worst = 0.0
                    for link in self.LINKS[geo.D]:
                        for a in range(geo.N * geo.N - 1):
                            X = torch.zeros_like(U)
                            X[link] = T[a]
                            pred = trace_product(F[link], T[a]).item()
                            fd = fd_derivative(S.action, U, X)[link[0]].item()
                            worst = max(worst, abs(pred - fd) / max(abs(fd), 1.0))
                            self.assertLess(abs(pred - fd), 1e-6 * max(abs(fd), 1.0), msg=f"link={link} a={a}")
                    if VERBOSE:
                        print(f"\n{name} {tname}: worst single-link deviation {worst:.1e}")

    def test_force_against_autograd(self):
        for name, geo, k, kh, ts, seed in self.cases():
            if ts != -1.0:
                continue  # assemble_even_odd is called with the default antiperiodic sign
            U, gen = haar_links(geo, 2, seed + 10)
            kt, kht, terms = self.terms(geo, k, kh, ts)
            for S in terms.values():
                S.refresh(U, gen)
            T = algebra_basis(geo.N)
            omega = torch.zeros(U.shape[:-2] + (T.shape[0],), dtype=torch.float64, requires_grad=True)
            Uw = U + 1j * torch.matmul(torch.einsum("...a,aij->...ij", omega.to(C), T), U)
            s1, s2 = dense_actions(Uw, terms["heavy"].phi, terms["ratio"].phi, kt, kht, geo)
            # the dense actions agree with the sparse ones
            with torch.no_grad():
                self.assertLess(((s1 - terms["heavy"].action(U)).abs() / s1).max().item(), 1e-11)
                self.assertLess(((s2 - terms["ratio"].action(U)).abs() / s2).max().item(), 1e-11)
            for tname, s in (("heavy", s1), ("ratio", s2)):
                with self.subTest(case=name, term=tname):
                    (ref,) = torch.autograd.grad(s.sum(), omega, retain_graph=True)
                    comp = torch.einsum("...ij,aji->...a", terms[tname].force(U), T).real
                    dev = (comp - ref).abs().max().item()
                    scale = ref.abs().max().item()
                    if VERBOSE:
                        print(f"\n{name} {tname}: max |F_a - autograd_a| = {dev:.1e} (max |F_a| = {scale:.2f})")
                    self.assertLess(dev, 1e-10 * max(scale, 1.0))


class IntegratorTest(unittest.TestCase):
    """3a. Reversibility and |dH| ~ eps^2 with the Hasenbusch Hamiltonian, 2^3 x 4 SU(3), both the
    nested (three-level) and the shared (two-level) layouts."""

    def test_reversibility(self):
        U, gen = haar_links(G4, 3, 401)
        P = H.random_momenta(U.shape, generator=gen)
        for layout in ("nested", "shared"):
            ham = HasenbuschHamiltonian(5.0, 0.12, 0.09, G4, n_heavy_substeps=2, n_gauge_substeps=2,
                                        layout=layout, reuse_guess=False, solver_tol=1e-12)
            ham.refresh(U, gen)
            with self.subTest(layout=layout):
                hmc = H.HMC(G4, ham, tau=0.5, n_steps=3, integrator="omelyan")
                U1, P1, _ = hmc.integrate(U, P)
                U2, P2, _ = hmc.integrate(U1, -P1)
                dU, dP = (U2 - U).abs().max().item(), (P2 + P).abs().max().item()
                if VERBOSE:
                    print(f"\n{layout}: |U2 - U| = {dU:.1e}, |P2 + P| = {dP:.1e}")
                self.assertGreater((U1 - U).abs().max().item(), 0.1)
                self.assertLess(dU, 1e-10)
                self.assertLess(dP, 1e-10)

    def test_dH_scales_as_eps_squared(self):
        U, gen = haar_links(G4, 4, 402)
        P = H.random_momenta(U.shape, generator=gen)
        ham = HasenbuschHamiltonian(5.0, 0.12, 0.09, G4, n_heavy_substeps=2, n_gauge_substeps=2, solver_tol=1e-12)
        ham.refresh(U, gen)
        hmc = H.HMC(G4, ham, tau=0.5, n_steps=1, integrator="omelyan")
        steps, dH = (4, 8, 16), []
        for n in steps:
            calls_r, calls_h = ham.ratio.stats["force_calls"], ham.heavy.stats["force_calls"]
            U1, P1, evals = hmc.integrate(U, P, n_steps=n)
            dH.append(ham.energy_difference(U, P, U1, P1).abs().mean().item())
            self.assertEqual(evals[0], 2 * n + 1)
            self.assertEqual(ham.ratio.stats["force_calls"] - calls_r, evals[0])
            self.assertEqual(ham.heavy.stats["force_calls"] - calls_h, evals[1])
        x = torch.tensor([math.log(1.0 / n) for n in steps], dtype=torch.double)
        y = torch.tensor([math.log(d) for d in dH], dtype=torch.double)
        slope = (((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()).item()
        if VERBOSE:
            print(f"\n<|dH|> = {['%.2e' % d for d in dH]} for n_steps {steps}, slope {slope:.3f}")
        self.assertGreater(slope, 1.8, msg=f"dH={dH}")
        self.assertLess(slope, 2.2, msg=f"dH={dH}")


class HMCComparisonTest(unittest.TestCase):
    """3b, 4. 2D 4 x 4 SU(3), beta = 5.0, kappa = 0.22 (kappa' = 0.18), B = 32 chains from a cold
    start: <exp(-dH)> = 1 and <P> equal to the single-pseudofermion HMC within errors; rms forces of
    the ratio and heavy terms against the single-term force on the thermalised configurations; CG
    iterations per trajectory with and without preconditioning (reported, not asserted)."""

    BETA, KAPPA, KAPPA_H, B = 5.0, 0.22, 0.18, 32
    N_THERM, N_MEAS = 15, 50

    @classmethod
    def run_chain(cls, ham, n_steps, seed):
        gen = torch.Generator().manual_seed(seed)
        U = SUN_identity(G2.link_shape(cls.B)[:-1], dtype=torch.cdouble)
        hmc = H.HMC(G2, ham, tau=1.0, n_steps=n_steps, integrator="omelyan")
        plaq, expdH, acc = [], [], []
        for i in range(cls.N_THERM + cls.N_MEAS):
            if i == cls.N_THERM:
                for t in ham.terms:
                    if hasattr(t, "reset_stats"):
                        t.reset_stats()
            U, accepted, dH, stats = hmc.trajectory(U, gen)
            if i >= cls.N_THERM:
                plaq.append(mean_plaquette(plaquette_fields(U, G2.D), G2.N))
                expdH.append(stats["exp_minus_dH"])
                acc.append(stats["acceptance"])

        def chain_stats(x):
            m = torch.stack(x).mean(0)
            return m.mean().item(), (m.std() / math.sqrt(cls.B)).item()

        cg = sum(t.stats["cg_iterations"] for t in ham.terms if hasattr(t, "stats"))
        return {"U": U, "plaq": chain_stats(plaq), "expdH": chain_stats(expdH), "acc": sum(acc) / len(acc),
                "cg_per_traj": cg / cls.N_MEAS}

    @classmethod
    def setUpClass(cls):
        cls.plain = TwoFlavourHamiltonian(cls.BETA, cls.KAPPA, G2, n_gauge_substeps=3)
        cls.hb = HasenbuschHamiltonian(cls.BETA, cls.KAPPA, cls.KAPPA_H, G2, n_heavy_substeps=2, n_gauge_substeps=2)
        cls.res_plain = cls.run_chain(cls.plain, 2, 501)
        cls.res_hb = cls.run_chain(cls.hb, 1, 502)

    def test_plaquette_and_exp_dH(self):
        a, b = self.res_plain, self.res_hb
        sig = abs(a["plaq"][0] - b["plaq"][0]) / math.hypot(a["plaq"][1], b["plaq"][1])
        msg = (f"plain:     <P> = {a['plaq'][0]:.4f}({a['plaq'][1]:.4f}), <exp(-dH)> = {a['expdH'][0]:.4f}"
               f"({a['expdH'][1]:.4f}), acceptance {a['acc']:.3f}, CG iterations / trajectory {a['cg_per_traj']:.0f}\n"
               f"Hasenbusch: <P> = {b['plaq'][0]:.4f}({b['plaq'][1]:.4f}), <exp(-dH)> = {b['expdH'][0]:.4f}"
               f"({b['expdH'][1]:.4f}), acceptance {b['acc']:.3f}, CG iterations / trajectory {b['cg_per_traj']:.0f}"
               f" (ratio {self.hb.ratio.stats['cg_iterations'] / self.N_MEAS:.0f}, heavy "
               f"{self.hb.heavy.stats['cg_iterations'] / self.N_MEAS:.0f}); <P> difference {sig:.2f} sigma")
        if VERBOSE:
            print("\n" + msg)
        for r in (a, b):
            self.assertGreater(r["acc"], 0.7, msg=msg)
            self.assertLess(abs(r["expdH"][0] - 1.0), 4 * r["expdH"][1], msg=msg)
        self.assertLess(sig, 4.0, msg=msg)

    def test_force_magnitudes(self):
        gen = torch.Generator().manual_seed(503)
        U = self.res_hb["U"]
        single = TwoFlavourWilson(self.KAPPA, G2)
        ratio = RatioWilsonTerm(self.KAPPA, self.KAPPA_H, G2)
        heavy = HeavyWilsonTerm(self.KAPPA_H, G2)
        gauge = H.WilsonGaugeAction(self.BETA, G2)
        rms = {}
        for name, t in (("single", single), ("ratio", ratio), ("heavy", heavy)):
            t.refresh(U, gen)
            rms[name] = rms_force(t.force(U))
        rms["gauge"] = rms_force(gauge.force(U))
        msg = "rms force per link: " + ", ".join(f"{k} {v:.3f}" for k, v in rms.items())
        if VERBOSE:
            print("\n" + msg)
        self.assertLess(rms["ratio"], rms["single"], msg=msg)

    def test_cost_at_equal_acceptance(self):
        """Mean min(1, exp(-dH)) and CG iterations per trajectory against n_steps on the thermalised
        configurations of the plain chain (same U, P for every setting). Reported only."""
        gen = torch.Generator().manual_seed(504)
        U = self.res_plain["U"]
        P = H.random_momenta(U.shape, generator=gen)
        lines = []
        settings = [("plain", lambda: TwoFlavourHamiltonian(self.BETA, self.KAPPA, G2, n_gauge_substeps=3), n)
                    for n in (1, 2, 3)]
        settings += [(f"Hasenbusch nested n_heavy={nh}",
                      lambda nh=nh: HasenbuschHamiltonian(self.BETA, self.KAPPA, self.KAPPA_H, G2,
                                                          n_heavy_substeps=nh, n_gauge_substeps=2), n)
                     for nh in (1, 2) for n in (1, 2)]
        settings += [("Hasenbusch shared", lambda: HasenbuschHamiltonian(
            self.BETA, self.KAPPA, self.KAPPA_H, G2, n_gauge_substeps=3, layout="shared"), n) for n in (1, 2, 3)]
        for name, make, n in settings:
            ham = make()
            ham.refresh(U, torch.Generator().manual_seed(505))
            hmc = H.HMC(G2, ham, tau=1.0, n_steps=n, integrator="omelyan")
            U1, P1, evals = hmc.integrate(U, P)
            dH = ham.energy_difference(U, P, U1, P1)
            acc = torch.clamp(torch.exp(-dH), max=1.0).mean().item()
            if isinstance(ham, HasenbuschHamiltonian):
                cg = {"ratio": ham.ratio.stats["cg_iterations"], "heavy": ham.heavy.stats["cg_iterations"]}
            else:
                cg = {"single": ham.fermion.stats["cg_iterations"]}
            lines.append(f"{name:31s} n_steps={n}: <min(1, e^-dH)> = {acc:.3f}, <|dH|> = {dH.abs().mean().item():.3f}, "
                         f"CG iterations {sum(cg.values())} {cg}, force evals {evals}")
            self.assertTrue(math.isfinite(acc))
        if VERBOSE:
            print("\n" + "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
