"""Tests of the parametric joint action S(U, phi; lambda) = S_g(U; beta, bc) + S_pf(U, phi; kappa)
of neoqcd.fermions.action (phase 3 of fermions_plan.md), for lambda in {beta, kappa, bc}.

CPU, complex128, fixed seeds; 2D 4 x 4 SU(3) and 4D 2^3 x 4 SU(3), Haar-random links (the
identities hold on any configuration). References: `theory.Wilson_action` (with `defect_par`),
`mcmc.DefectMCMCAdapter.action_difference`, an independent `TwoFlavourWilson` and the dense
even-odd matrix of `exact.assemble_even_odd`.
"""

import unittest
from types import SimpleNamespace

import torch

from neoqcd.fermions._gauge import SUN_haar, Wilson_action, create_mask
from neoqcd.fermions.action import QCDAction, bc_plaquette_weights
from neoqcd.fermions.exact import _site_tables, assemble_even_odd
from neoqcd.fermions.hmc import defect_plaquette_weights
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.pseudofermion import TwoFlavourWilson
from neoqcd.mcmc import DefectMCMCAdapter

G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4, SU(3)
G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4, SU(3)
B = 4
BETA, KAPPA, BC = 5.0, 0.18, 0.4
TOL = 1e-10


def haar_links(g, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(g.link_shape(batch)[:-1], generator=gen), gen


def defect_mask(g):
    """A few time links at t = 0 (x = 0, 1 and all other coordinates 0), layout of utils.Defect."""
    d = torch.zeros((g.D,) + g.lattice, dtype=torch.double)
    d[(0, 0, slice(0, 2)) + (0,) * (g.D - 2)] = 1.0
    return torch.stack((1.0 - d, d), dim=0)


def wilson_reference(g, beta, bc=1.0):
    return Wilson_action(SimpleNamespace(D=g.D, N=g.N, device=torch.device("cpu")), beta, bc)


def dense_pf_action(U, phi, kappa, g):
    """phi^dagger (D_hat^dagger D_hat)^-1 phi from the dense even-odd matrix (dense LU)."""
    _, _, par = _site_tables(g, U.device)
    even = (par == 0).nonzero().squeeze(-1)
    b = phi.reshape(phi.shape[0], g.volume, g.spin, g.N)[:, even].reshape(phi.shape[0], -1)
    Dh = assemble_even_odd(U, kappa, g)
    y = torch.linalg.solve(Dh.conj().transpose(-1, -2), b)  # D_hat^-dagger phi
    return (y.conj() * y).real.sum(-1)  # |D_hat^-dagger phi|^2


def make(g, name, **kw):
    return QCDAction(g, name, beta=BETA, kappa=KAPPA, bc=kw.pop("bc", BC), defect_mask=defect_mask(g), **kw)


class ActionValueTest(unittest.TestCase):
    """1a. action(U) = S_g + eta^dagger eta right after the refresh, and S_pf against the dense matrix."""

    def test_action_after_refresh(self):
        for g in (G2, G4):
            for name in ("beta", "kappa", "bc"):
                with self.subTest(geometry=g, parameter=name):
                    U, gen = haar_links(g, B, 11)
                    act = make(g, name)
                    act.refresh(U, gen)
                    eta2 = act.fermion._refresh_action
                    S_g = wilson_reference(g, BETA, BC)(U, defect_mask(g))
                    S = act.action(U)
                    self.assertLess((S - (S_g + eta2)).abs().max().item(), TOL * S.abs().max().item())
                    S_pf = act.fermion_action(U)
                    ref = dense_pf_action(U, act.phi, KAPPA, g)
                    self.assertLess((S_pf - ref).abs().max().item(), TOL * ref.abs().max().item())


class ActionDifferenceTest(unittest.TestCase):
    """1b. action_difference at fixed (U, phi) for the three parameters."""

    def setUp(self):
        self.cases = []
        for g in (G2, G4):
            U, gen = haar_links(g, B, 21)
            self.cases.append((g, U, gen))

    def test_beta(self):
        for g, U, gen in self.cases:
            with self.subTest(geometry=g):
                act = make(g, "beta")
                act.refresh(U, gen)
                b1, b2 = 4.7, 5.35
                solves = act.fermion.stats["solves"]
                dS = act.action_difference(U, b1, b2)
                self.assertEqual(act.fermion.stats["solves"], solves)  # no solve for a beta jump
                ref_g = wilson_reference(g, b2, BC)(U, defect_mask(g)) - wilson_reference(g, b1, BC)(U, defect_mask(g))
                self.assertLess((dS - ref_g).abs().max().item(), TOL * ref_g.abs().max().item())
                ref = act.action(U, b2) - act.action(U, b1)  # two full actions (with solves)
                self.assertLess((dS - ref).abs().max().item(), TOL * ref.abs().max().item())
                # batched beta: one value per configuration
                bt1 = torch.linspace(4.5, 5.0, B, dtype=torch.double)
                bt2 = bt1 + 0.3
                dSb = act.action_difference(U, bt1, bt2)
                refb = torch.stack([act.action_difference(U[i:i + 1], float(bt1[i]), float(bt2[i]))[0] for i in range(B)])
                self.assertLess((dSb - refb).abs().max().item(), 1e-12 * refb.abs().max().item())

    def test_kappa(self):
        for g, U, gen in self.cases:
            with self.subTest(geometry=g):
                act = make(g, "kappa")
                act.refresh(U, gen)
                k1, k2 = 0.16, 0.19
                dS = act.action_difference(U, k1, k2)
                pf = []
                for k in (k1, k2):
                    term = TwoFlavourWilson(k, g)
                    term.phi = act.phi
                    pf.append(term.action(U))
                ref = pf[1] - pf[0]
                self.assertLess((dS - ref).abs().max().item(), TOL * ref.abs().max().item())
                dense = dense_pf_action(U, act.phi, k2, g) - dense_pf_action(U, act.phi, k1, g)
                self.assertLess((dS - dense).abs().max().item(), TOL * dense.abs().max().item())
                # the gauge part does not depend on kappa; the current kappa is untouched
                self.assertEqual(act.parameter, KAPPA)
                self.assertEqual(act.fermion.kappa, KAPPA)
                full = act.action(U, k2) - act.action(U, k1)
                self.assertLess((dS - full).abs().max().item(), TOL * full.abs().max().item())

    def test_bc(self):
        for g, U, gen in self.cases:
            with self.subTest(geometry=g):
                act = make(g, "bc")
                act.refresh(U, gen)
                c1, c2 = 0.7, 0.15
                mask = defect_mask(g)
                solves = act.fermion.stats["solves"]
                dS = act.action_difference(U, c1, c2)
                self.assertEqual(act.fermion.stats["solves"], solves)  # gauge only
                ref_w = wilson_reference(g, BETA, c2)(U, mask) - wilson_reference(g, BETA, c1)(U, mask)
                self.assertLess((dS - ref_w).abs().max().item(), TOL * ref_w.abs().max().item())
                pars = SimpleNamespace(D=g.D, N=g.N, T=g.T, L=g.L, device=torch.device("cpu"), orsteps=1,
                                       mask=create_mask(g.D, g.T, g.L))
                adapter = DefectMCMCAdapter(pars, BETA, mask, B)
                for full in (False, True):
                    ref_a = adapter.action_difference(U, c1, c2, full_lattice=full)
                    self.assertLess((dS - ref_a).abs().max().item(), TOL * ref_a.abs().max().item())
                # the fermion part is unchanged: the full-action difference is the gauge one
                full_diff = act.action(U, c2) - act.action(U, c1)
                self.assertLess((dS - full_diff).abs().max().item(), TOL * full_diff.abs().max().item())
                self.assertGreater(dS.abs().min().item(), 1e-3)
                # weights: scalar bc as hmc.defect_plaquette_weights, batched bc entry by entry
                w = bc_plaquette_weights(mask, c2, g.D)
                w_ref = defect_plaquette_weights(mask, c2, g.D)
                bcs = torch.tensor([0.1, 0.5, 0.9, 1.0], dtype=torch.double)
                wb = bc_plaquette_weights(mask, bcs, g.D)
                for key in w_ref:
                    self.assertTrue(torch.equal(w[key], w_ref[key]))
                    for i in range(4):
                        self.assertTrue(torch.allclose(wb[key][i], bc_plaquette_weights(mask, float(bcs[i]), g.D)[key],
                                                       rtol=0, atol=1e-15))
                dSb = act.action_difference(U, torch.full((B,), c1, dtype=torch.double), bcs)
                refb = torch.stack([act.action_difference(U[i:i + 1], c1, float(bcs[i]))[0] for i in range(B)])
                self.assertLess((dSb - refb).abs().max().item(), 1e-12)


class HamiltonianTest(unittest.TestCase):
    """2. The Hamiltonian at a parameter value has the matching beta, kappa and bc."""

    def test_hamiltonian_parameters(self):
        g = G2
        U, gen = haar_links(g, B, 31)
        values = {"beta": 5.6, "kappa": 0.2, "bc": 0.25}
        for name, value in values.items():
            with self.subTest(parameter=name):
                act = make(g, name)
                act.refresh(U, gen)
                ham = act.hamiltonian(value)
                self.assertEqual(act.parameter, value)
                beta = value if name == "beta" else BETA
                kappa = value if name == "kappa" else KAPPA
                bc = value if name == "bc" else BC
                self.assertIs(ham.fermion, act.fermion)  # shared: carries phi
                self.assertEqual(ham.fermion.kappa, kappa)
                S_g_ref = wilson_reference(g, beta, bc)(U, defect_mask(g))
                self.assertLess((ham.gauge.action(U) - S_g_ref).abs().max().item(), TOL * S_g_ref.abs().max().item())
                S_pf_ref = dense_pf_action(U, act.phi, kappa, g)
                self.assertLess((ham.fermion.action(U) - S_pf_ref).abs().max().item(), TOL * S_pf_ref.abs().max().item())
                S = ham.action(U)
                self.assertLess((S - act.action(U)).abs().max().item(), TOL * S.abs().max().item())
                self.assertLess((S - (S_g_ref + S_pf_ref)).abs().max().item(), TOL * S.abs().max().item())
                # the HMC acceptance step uses action_difference of the fermion term: the refresh cache
                # (eta^dagger eta) must not be used once kappa has changed
                act.set_parameter(values[name])
                act.refresh(U, gen)
                if name == "kappa":
                    act.set_parameter(0.15)
                    U2, _ = haar_links(g, B, 32)
                    d = act.fermion.action_difference(U, U2)
                    ref = dense_pf_action(U2, act.phi, 0.15, g) - dense_pf_action(U, act.phi, 0.15, g)
                    self.assertLess((d - ref).abs().max().item(), TOL * ref.abs().max().item())


if __name__ == "__main__":
    unittest.main()
