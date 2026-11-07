"""Tests of the parametric joint action with staggered fermions (neoqcd.fermions.action,
`QCDAction(..., fermions="staggered", mass=m)`): S(U, phi; lambda) = S_g(U; beta, bc) +
phi^dagger M_ee(m)^-1 phi, lambda in {beta, mass, bc}, and the NE-MCMC / REX drivers on top of it.

CPU, complex128, fixed seeds; 2D 4 x 4 SU(3) (one even-site staggered field, N_f = 2).
References: an independent `StaggeredPseudofermion` at the same phi, `theory.Wilson_action` (with
`defect_par`), and for Jarzynski the exact reweighting
    Z(m2)/Z(m1) = < det M_ee(m2) / det M_ee(m1) >_{m1}
with the dense determinant of `staggered_exact.log_det_staggered` on every configuration of an
independent equilibrium sample at m1 (plain staggered HMC).
"""

import math
import unittest
from types import SimpleNamespace

import torch

from neoqcd.fermions import hmc as H
from neoqcd.fermions._gauge import SUN_haar, SUN_identity, Wilson_action
from neoqcd.fermions.action import ParametricStaggeredPseudofermion, QCDAction
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.rex import FermionREX
from neoqcd.fermions.snf import FermionNEMCMC, FieldState, effective_sample_size, jarzynski_free_energy
from neoqcd.fermions.staggered_exact import log_det_staggered
from neoqcd.fermions.staggered_pseudofermion import StaggeredHamiltonian, StaggeredPseudofermion

G2 = Geometry(2, 4, 4, 3)
BETA, MASS, BC = 5.0, 0.3, 0.4
MASS2 = 0.2
TOL = 1e-10
N_SUB = 3
HMC_KW = dict(tau=1.0, n_steps=4, n_traj=1, integrator="omelyan")


def haar_links(batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(G2.link_shape(batch)[:-1], generator=gen), gen


def defect_mask():
    d = torch.zeros((G2.D,) + G2.lattice, dtype=torch.double)
    d[0, 0, 0:2] = 1.0  # two time links at t = 0
    return torch.stack((1.0 - d, d), dim=0)


def wilson_reference(beta, bc=1.0):
    return Wilson_action(SimpleNamespace(D=G2.D, N=G2.N, device=torch.device("cpu")), beta, bc)


def make(name, bc=BC, **kw):
    return QCDAction(G2, name, beta=BETA, bc=bc, defect_mask=defect_mask(), fermions="staggered", mass=MASS,
                     n_gauge_substeps=N_SUB, **kw)


def fresh_pf(mass, phi, U):
    """S_pf(U, phi; mass) from an independent StaggeredPseudofermion."""
    term = StaggeredPseudofermion(mass, G2)
    term.phi = phi
    return term.action(U)


def rel(a, b):
    return (a - b).abs().max().item() / b.abs().max().item()


class ConstructionTest(unittest.TestCase):
    def test_parameter_names(self):
        with self.assertRaisesRegex(ValueError, "staggered"):
            QCDAction(G2, "kappa", beta=BETA, fermions="staggered", mass=MASS)
        with self.assertRaisesRegex(ValueError, "wilson"):
            QCDAction(G2, "mass", beta=BETA, kappa=0.18)
        with self.assertRaisesRegex(ValueError, "mass"):
            QCDAction(G2, "beta", beta=BETA, kappa=0.18, mass=MASS)
        with self.assertRaisesRegex(ValueError, "kappa"):
            QCDAction(G2, "beta", beta=BETA, kappa=0.18, fermions="staggered", mass=MASS)
        with self.assertRaisesRegex(ValueError, "need mass"):
            QCDAction(G2, "beta", beta=BETA, fermions="staggered")
        with self.assertRaisesRegex(ValueError, "fermions"):
            QCDAction(G2, "beta", beta=BETA, kappa=0.18, fermions="overlap")
        act = make("mass")
        self.assertIsInstance(act.fermion, ParametricStaggeredPseudofermion)
        self.assertEqual(act.parameter, MASS)
        self.assertEqual(act.fermion_parameter_name, "mass")
        self.assertEqual(act.fermion.n_components, G2.volume * G2.N // 2)


class ActionValueTest(unittest.TestCase):
    """1. action(U) = S_g + S_pf against independent objects; action_difference in beta, mass, bc."""

    def setUp(self):
        self.U, self.gen = haar_links(4, 11)
        self.U2, _ = haar_links(4, 12)

    def test_action_after_refresh(self):
        U = self.U
        for name in ("beta", "mass", "bc"):
            with self.subTest(parameter=name):
                act = make(name)
                act.refresh(U, self.gen)
                S = act.action(U)
                S_g = wilson_reference(BETA, BC)(U, defect_mask())
                S_pf = fresh_pf(MASS, act.phi, U)
                self.assertLess(rel(S, S_g + S_pf), TOL)
                # <S_pf> = n per configuration for a heat bath: a loose sanity check on 4 configurations
                n = act.fermion.n_components
                self.assertLess(abs(S_pf.mean().item() - n), 6 * math.sqrt(n / 4))

    def test_beta(self):
        U = self.U
        act = make("beta")
        act.refresh(U, self.gen)
        b1, b2 = 4.7, 5.35
        solves = act.fermion.stats["solves"]
        dS = act.action_difference(U, b1, b2)
        self.assertEqual(act.fermion.stats["solves"], solves)
        ref = wilson_reference(b2, BC)(U, defect_mask()) - wilson_reference(b1, BC)(U, defect_mask())
        self.assertLess(rel(dS, ref), TOL)
        self.assertLess(rel(dS, act.action(U, b2) - act.action(U, b1)), TOL)

    def test_mass(self):
        U = self.U
        act = make("mass")
        act.refresh(U, self.gen)
        m1, m2 = 0.35, 0.2
        dS = act.action_difference(U, m1, m2)
        ref = fresh_pf(m2, act.phi, U) - fresh_pf(m1, act.phi, U)
        self.assertLess(rel(dS, ref), TOL)
        self.assertGreater(ref.abs().min().item(), 1e-2)
        self.assertEqual(act.parameter, MASS)  # the current mass is untouched
        self.assertEqual(act.fermion.mass, MASS)
        self.assertLess(rel(dS, act.action(U, m2) - act.action(U, m1)), TOL)
        # batched masses, one per configuration
        mt1 = torch.tensor([0.25, 0.3, 0.35, 0.4], dtype=torch.double)
        mt2 = mt1 - 0.1
        dSb = act.action_difference(U, mt1, mt2)
        refb = torch.stack([fresh_pf(float(mt2[i]), act.phi[i:i + 1], U[i:i + 1])[0]
                            - fresh_pf(float(mt1[i]), act.phi[i:i + 1], U[i:i + 1])[0] for i in range(4)])
        self.assertLess(rel(dSb, refb), 1e-9)

    def test_bc(self):
        U = self.U
        act = make("bc")
        act.refresh(U, self.gen)
        c1, c2 = 0.7, 0.15
        solves = act.fermion.stats["solves"]
        dS = act.action_difference(U, c1, c2)
        self.assertEqual(act.fermion.stats["solves"], solves)  # gauge only
        ref = wilson_reference(BETA, c2)(U, defect_mask()) - wilson_reference(BETA, c1)(U, defect_mask())
        self.assertLess(rel(dS, ref), TOL)
        self.assertLess(rel(dS, act.action(U, c2) - act.action(U, c1)), TOL)
        self.assertGreater(dS.abs().min().item(), 1e-3)

    def test_refresh_cache_after_mass_change(self):
        """The S_pf value at the refresh links is cached at the mass of its first use; after a mass
        change the HMC accept step (fermion.action_difference(U_refresh, U_new)) must not reuse it."""
        U, U2 = self.U, self.U2
        act = make("mass")
        act.refresh(U, self.gen)
        d1 = act.fermion.action_difference(U, U2)  # fills the cache at MASS
        ref1 = fresh_pf(MASS, act.phi, U2) - fresh_pf(MASS, act.phi, U)
        self.assertLess(rel(d1, ref1), TOL)
        solves = act.fermion.stats["solves"]
        act.fermion.action_difference(U, U2)  # same mass: the cache is used (one solve, not two)
        self.assertEqual(act.fermion.stats["solves"] - solves, 1)
        act.set_parameter(MASS2)
        d2 = act.fermion.action_difference(U, U2)
        ref2 = fresh_pf(MASS2, act.phi, U2) - fresh_pf(MASS2, act.phi, U)  # honest: two solves
        self.assertLess(rel(d2, ref2), TOL)
        stale = fresh_pf(MASS2, act.phi, U2) - fresh_pf(MASS, act.phi, U)  # what a stale cache gives
        self.assertGreater((stale - ref2).abs().min().item(), 1e-2)
        # mutation check: the base class with the mass changed in place does reuse the stale value
        base = StaggeredPseudofermion(MASS, G2)
        base.refresh(U, torch.Generator().manual_seed(5))
        base.action_difference(U, U2)
        base.mass = MASS2
        d_base = base.action_difference(U, U2)
        honest = fresh_pf(MASS2, base.phi, U2) - fresh_pf(MASS2, base.phi, U)
        self.assertGreater((d_base - honest).abs().min().item(), 1e-2)
        # back to MASS: recomputed at MASS (not the MASS2 value)
        act.set_parameter(MASS)
        self.assertLess(rel(act.fermion.action_difference(U, U2), ref1), TOL)
        # a new phi drops the cache
        act.phi = act.phi.clone()
        self.assertIsNone(act.fermion._refresh_action)


class HamiltonianTest(unittest.TestCase):
    """2. The Hamiltonian at a parameter value has the matching beta, mass and bc."""

    def test_hamiltonian_parameters(self):
        U, gen = haar_links(4, 31)
        values = {"beta": 5.6, "mass": 0.2, "bc": 0.25}
        for name, value in values.items():
            with self.subTest(parameter=name):
                act = make(name)
                act.refresh(U, gen)
                ham = act.hamiltonian(value)
                self.assertEqual(act.parameter, value)
                beta = value if name == "beta" else BETA
                mass = value if name == "mass" else MASS
                bc = value if name == "bc" else BC
                self.assertIs(ham.fermion, act.fermion)
                self.assertEqual(ham.fermion.mass, mass)
                # two levels as StaggeredHamiltonian: fermion outer, gauge inner with N_SUB substeps
                self.assertEqual([(terms, n) for terms, n in ham.levels], [([act.fermion], 1), ([ham.gauge], N_SUB)])
                S_g_ref = wilson_reference(beta, bc)(U, defect_mask())
                self.assertLess(rel(ham.gauge.action(U), S_g_ref), TOL)
                S_pf_ref = fresh_pf(mass, act.phi, U)
                self.assertLess(rel(ham.fermion.action(U), S_pf_ref), TOL)
                self.assertLess(rel(ham.action(U), S_g_ref + S_pf_ref), TOL)
                self.assertLess(rel(ham.action(U), act.action(U)), TOL)


# ---- NE-MCMC -------------------------------------------------------------------------------
B = 64
N_THERM = 20


def plain_hmc(mass, seed, n_therm, n_meas, every=1):
    gen = torch.Generator().manual_seed(seed)
    ham = StaggeredHamiltonian(BETA, mass, G2, n_gauge_substeps=N_SUB)
    hmc = H.HMC(G2, ham, tau=HMC_KW["tau"], n_steps=HMC_KW["n_steps"], integrator=HMC_KW["integrator"])
    U = SUN_identity(G2.link_shape(B)[:-1], dtype=torch.cdouble)
    out = []
    for i in range(n_therm + n_meas * every):
        U, _, _, _ = hmc.trajectory(U, gen)
        if i >= n_therm and (i - n_therm + 1) % every == 0:
            out.append(U)
    return out


def linear_protocol(a, b, n):
    return [a + (b - a) * (i + 1) / n for i in range(n)]


class JarzynskiMassTest(unittest.TestCase):
    """3. <exp(-W)> = Z(m2)/Z(m1) for the mass protocol 0.3 -> 0.2 in 1 and 8 steps, and
    W + Q = S_final - S_0 per chain (S_final from an independent action object)."""

    @classmethod
    def setUpClass(cls):
        ref = plain_hmc(MASS, 1001, N_THERM, 20)
        cls.ref_ratio = torch.stack([torch.exp(log_det_staggered(U, MASS2, G2) - log_det_staggered(U, MASS, G2))
                                     for U in ref])  # (20, B)
        cls.starts = plain_hmc(MASS, 2002, N_THERM, 4, every=3)

    def test_mass_protocol(self):
        m = self.ref_ratio.mean(0)
        r, e_r = m.mean().item(), (m.std() / math.sqrt(m.numel())).item()
        dF_ref, e_ref = -math.log(r), e_r / r
        lines = [f"mass: {MASS} -> {MASS2}, exact reweighting Z_n/Z_0 = {r:.4f}({e_r:.4f}), "
                 f"dF = {dF_ref:.4f}({e_ref:.4f}), |ratio - 1|/sigma = {abs(r - 1) / e_r:.1f}"]
        results = {}
        for n_step in (1, 8):
            gen = torch.Generator().manual_seed(3000 + n_step)
            act = make("mass", bc=1.0)
            ne = FermionNEMCMC(act, linear_protocol(MASS, MASS2, n_step), **HMC_KW)
            Ws = []
            accs = []
            for U in self.starts:
                state, W, Q, _ = ne.forward(FieldState(U), gen)
                Ws.append(W)
                accs += [a for info in ne.last_run["block_info"] for a in info["acceptance"]]
                check = make("mass", bc=1.0)
                check.set_parameter(MASS2)
                check.phi = state.phi
                S_final = check.action(state.U)
                self.assertLess((W + Q - (S_final - ne.last_run["S0"])).abs().max().item(),
                                1e-10 * S_final.abs().max().item())
            W = torch.cat(Ws)
            dF, e = jarzynski_free_energy(W)
            results[n_step] = (dF, e)
            lines.append(f"  n_step = {n_step}: <exp(-W)> = {math.exp(-dF):.4f}({math.exp(-dF) * e:.4f}), "
                         f"dF = {dF:.4f}({e:.4f}), <W> = {W.mean().item():.4f}, "
                         f"ESS = {effective_sample_size(W):.3f}, "
                         f"(dF - dF_ref)/sigma = {(dF - dF_ref) / math.hypot(e, e_ref):+.2f}, "
                         f"acceptance {sum(accs) / len(accs):.3f}")
        msg = "\n".join(lines)
        print("\n" + msg)
        self.assertGreater(abs(r - 1.0), 5 * e_r, msg=msg)
        for dF, e in results.values():
            self.assertLess(abs(dF - dF_ref), 3 * math.hypot(e, e_ref), msg=msg)

    def test_bookkeeping_beta_bc(self):
        """beta and bc protocols with staggered blocks: W + Q = S_final - S_0 per chain, no solve in
        the jumps."""
        U0 = self.starts[0][:16]
        for name, (start, protocol) in {"beta": (BETA, [5.1, 5.2]), "bc": (1.0, [0.6, 0.2])}.items():
            with self.subTest(parameter=name):
                act = make(name, bc=1.0)
                act.set_parameter(start)
                ne = FermionNEMCMC(act, protocol, **HMC_KW)
                state, W, Q, obs = ne.forward(FieldState(U0), torch.Generator().manual_seed(77))
                check = make(name, bc=1.0)
                check.set_parameter(protocol[-1])
                check.phi = state.phi
                S_final = check.action(state.U)
                self.assertLess((W + Q + ne.last_run["S0"] - S_final).abs().max().item(),
                                1e-10 * S_final.abs().max().item())
                self.assertTrue(torch.equal(obs[:, -1, 0], W))


class REXSmokeTest(unittest.TestCase):
    """FermionREX accepts the staggered action through its fermion keyword arguments (kappa=None)."""

    def test_staggered_rex_runs(self):
        rex = FermionREX(G2, BETA, None, defect_mask(), [0.0, 0.5, 1.0], n_chains=2, n_steps=4,
                         fermions="staggered", mass=MASS, swap_seed=3)
        self.assertIsInstance(rex.qcd.fermion, ParametricStaggeredPseudofermion)
        out = rex.run(3, torch.Generator().manual_seed(4))
        self.assertEqual(len(out), 3)
        self.assertGreater(sum(o["hmc_acceptance"] for o in out) / 3, 0.3)
        # the swap table is gauge-only: no solve
        solves = rex.qcd.fermion.stats["solves"]
        from neoqcd.fermions.rex import swap_action_difference
        swap_action_difference(rex.qcd, rex.U, rex.parameters(), rex.ladder)
        self.assertEqual(rex.qcd.fermion.stats["solves"], solves)

    def test_shift_invariance_of_the_determinant(self):
        """`periodic_translation` shifts the bc = 1 replica by one site: exact for staggered fermions
        too, since one-site shifts (with the staggered sign factors) are a symmetry, det M_ee unchanged."""
        U, _ = haar_links(4, 41)
        ld = log_det_staggered(U, MASS, G2)
        for mu in range(G2.D):
            for s in (1, -1):
                d = log_det_staggered(torch.roll(U, s, dims=2 + mu), MASS, G2) - ld
                self.assertLess(d.abs().max().item(), 1e-12)


if __name__ == "__main__":
    unittest.main()
