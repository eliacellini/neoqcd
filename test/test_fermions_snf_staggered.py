"""Tests of the fermionic SNF (neoqcd.fermions.snf_layers) with staggered fermions:
`QCDAction(..., fermions="staggered", mass=m)`, S_pf = phi^dagger M_ee(m)^-1 phi, mass protocol.

CPU, complex128, fixed seeds; 2D 4 x 4 SU(3) (one even-site staggered field, N_f = 2), beta = 5,
m = 0.3 (the setting of test_fermions_action_staggered, whose helpers are reused); layers: the
pure-gauge stout `CouplingLayer` (isotropic rho, D = 2).

(a) dS_pf/drho through a stout layer: custom Function vs 4th-order finite differences and vs
    autograd through a dense solve of M_ee (`staggered_exact.assemble_staggered_even_odd`);
(b) W + Q = S_final - S_0 - logJ for a 2-step mass protocol with rho != 0;
(c) rho = 0 reproduces `FermionNEMCMC` exactly;
(d) Jarzynski with layers, m 0.3 -> 0.2 in 4 steps, rho = 0.005, against exact reweighting
    Z(m2)/Z(m1) = < det M_ee(m2) / det M_ee(m1) >_{m1} (`log_det_staggered`).
"""

import math
import unittest

import torch

from neoqcd.fermions._gauge import FlowPars, create_mask
from neoqcd.fermions.action import QCDAction
from neoqcd.fermions.hasenbusch import RatioWilsonTerm
from neoqcd.fermions.lattice import parity
from neoqcd.fermions.pseudofermion import TwoFlavourWilson
from neoqcd.fermions.snf import FermionNEMCMC, FieldState, effective_sample_size, jarzynski_free_energy
from neoqcd.fermions.snf_layers import FermionSNF, PseudofermionAction, joint_action, make_gauge_layers, pseudofermion_action
from neoqcd.fermions.staggered_exact import assemble_staggered_even_odd
from neoqcd.fermions.staggered_pseudofermion import StaggeredPseudofermion
from test.test_fermions_action_staggered import (BETA, G2, HMC_KW, MASS, MASS2, N_SUB, defect_mask, haar_links,
                                                 linear_protocol, mass_ensembles, record_jump_solves)


def flow_pars(protocol, batch_size):
    return FlowPars(D=G2.D, T=G2.T, L=G2.L, N=G2.N, mask=create_mask(G2.D, G2.T, G2.L), protocol=list(protocol),
                    batch_size=batch_size, device="cpu")


def make_action(name, **kw):
    return QCDAction(G2, name, beta=BETA, bc=1.0, defect_mask=defect_mask(), fermions="staggered", mass=MASS,
                     n_gauge_substeps=N_SUB, **kw)


def make_snf(name, start, protocol, rho, B, **kw):
    act = make_action(name, **kw)
    act.set_parameter(start)
    fp = flow_pars(protocol, B)
    layers = make_gauge_layers(fp, "smearing", rho=rho)
    return FermionSNF(act, protocol, layers, fp, **HMC_KW), act


def dense_pf_action(U, phi, mass):
    """S_pf = phi_e^dagger M_ee^-1 phi_e from the dense M_ee, differentiable in U by autograd."""
    B = U.shape[0]
    even = (parity(G2).reshape(-1) == 0).nonzero().squeeze(-1)
    p = phi.reshape(B, G2.volume, -1)[:, even].reshape(B, -1, 1)
    M = assemble_staggered_even_odd(U, mass, G2)
    y = torch.linalg.solve(M, p)
    return (p.conj() * y).real.reshape(B, -1).sum(1)


def fd4(f, h):
    return (8.0 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12.0 * h)


class StaggeredFunctionTest(unittest.TestCase):
    """(a) The custom Function with a staggered term: value, gradient through a stout layer."""

    @classmethod
    def setUpClass(cls):
        cls.U, gen = haar_links(4, 41)
        cls.fermion = StaggeredPseudofermion(MASS, G2, solver_tol=1e-13, action_tol=1e-14, reuse_guess=False)
        cls.fermion.refresh(cls.U, gen)

    def test_value(self):
        U, f = self.U, self.fermion
        S = pseudofermion_action(U, f)
        ref = dense_pf_action(U, f.phi, MASS)
        self.assertLess((S - f.action(U)).abs().max().item(), 1e-12 * S.abs().max().item())
        self.assertLess((S - ref).abs().max().item(), 1e-10 * S.abs().max().item())

    def test_gradient_through_layer(self):
        U, fermion = self.U, self.fermion
        fp = flow_pars([MASS], U.shape[0])
        layer = make_gauge_layers(fp, "smearing", rho=0.0)[0]
        rho0 = torch.tensor([0.05, 0.08, 0.03, 0.06], dtype=torch.float64).view(fp.rho_shape)
        with torch.no_grad():
            layer.rho_layer.copy_(rho0)

        def L_custom():
            U_new, _, _ = layer(U, fp, None, beta=MASS, delta_beta=0.0)
            return pseudofermion_action(U_new, fermion).sum()

        def L_dense():
            U_new, _, _ = layer(U, fp, None, beta=MASS, delta_beta=0.0)
            return dense_pf_action(U_new, fermion.phi, MASS).sum()

        g_custom = torch.autograd.grad(L_custom(), layer.rho_layer)[0].reshape(-1)
        g_dense = torch.autograd.grad(L_dense(), layer.rho_layer)[0].reshape(-1)
        g_fd = torch.zeros_like(g_custom)
        for k in range(g_fd.numel()):
            def f(h, k=k):
                with torch.no_grad():
                    r = rho0.clone().reshape(-1)
                    r[k] += h
                    layer.rho_layer.copy_(r.view(fp.rho_shape))
                    return L_custom().item()
            g_fd[k] = fd4(f, 1e-3)
        with torch.no_grad():
            layer.rho_layer.copy_(rho0)
        rel_fd = ((g_custom - g_fd).abs() / g_fd.abs()).max().item()
        rel_dense = ((g_custom - g_dense).abs() / g_dense.abs()).max().item()
        print(f"\nstaggered dS_pf/drho through a stout layer: custom {g_custom.tolist()}\n  FD {g_fd.tolist()}\n"
              f"  dense {g_dense.tolist()}\n  max rel diff vs FD {rel_fd:.1e}, vs dense autograd {rel_dense:.1e}")
        self.assertTrue(bool((g_custom.abs() > 1e-3).all()))
        self.assertLess(rel_fd, 1e-6)
        self.assertLess(rel_dense, 1e-10)

    def test_joint_action_mass(self):
        """joint_action uses the mass of `parameter` and keeps the current one."""
        U = self.U
        act = make_action("mass")
        act.phi = self.fermion.phi
        S = joint_action(act, U, 0.25)
        self.assertEqual(act.fermion.mass, MASS)
        ref = act.gauge_action(U) + dense_pf_action(U, act.phi, 0.25)
        self.assertLess((S - ref).abs().max().item(), 1e-10 * ref.abs().max().item())

    def test_unknown_fermion_type(self):
        class Other:
            pass
        with self.assertRaisesRegex(TypeError, "unsupported fermion term"):
            PseudofermionAction.apply(self.U, Other())
        # the Wilson branch is still dispatched
        w = TwoFlavourWilson(0.18, G2, reuse_guess=False)
        w.refresh(self.U, torch.Generator().manual_seed(1))
        self.assertEqual(pseudofermion_action(self.U, w).shape, (4,))
        # a Hasenbusch ratio term is a TwoFlavourWilson subclass but must not be taken for one
        ratio = RatioWilsonTerm(0.18, 0.12, G2)
        with self.assertRaisesRegex(TypeError, "Hasenbusch"):
            PseudofermionAction.apply(self.U, ratio)

    def test_joint_action_restores_mass_on_error(self):
        """If the solve raises inside joint_action, the current mass is restored (try/finally)."""
        act = make_action("mass")
        act.phi = self.fermion.phi

        def failing_solve(op, tol):
            self.assertEqual(act.fermion.mass, 0.25)  # the solve runs at the mass of `parameter`
            raise RuntimeError("solver failure")
        act.fermion._solve = failing_solve
        with self.assertRaisesRegex(RuntimeError, "solver failure"):
            joint_action(act, self.U, 0.25)
        self.assertEqual(act.fermion.mass, MASS)
        self.assertEqual(act.parameter, MASS)


class StaggeredSNFGradientTest(unittest.TestCase):
    """dW/drho through a 2-step FermionSNF.forward, staggered: the backward of the last layer's
    S_pf runs after the last HMC block has redrawn phi (and overwritten the CG solution X), so it
    must use the force saved by the forward. Reference: 4th-order finite differences of W in the
    last layer's rho, each evaluation a full run with the same seeds (the HMC blocks before the last
    layer do not depend on it, and the last block does not enter W, so the runs differ only in the
    last layer's work)."""

    def test_last_layer_gradient_after_phi_redraw(self):
        U0, _ = haar_links(4, 77)
        protocol = [0.25, 0.2]

        def run(rho_last=None, grad=False, saved_phi=None):
            snf, act = make_snf("mass", MASS, protocol, 0.03, U0.shape[0], solver_tol=1e-13, action_tol=1e-14,
                                reuse_guess=False)
            layers = [b.layer for b in snf.blocks]
            if rho_last is not None:
                with torch.no_grad():
                    layers[-1].rho_layer.copy_(rho_last)
            if saved_phi is not None:
                snf.blocks[-1].register_forward_pre_hook(lambda mod, args: saved_phi.append(args[0].phi))
            with torch.set_grad_enabled(grad):
                _, W, _, _, _ = snf(FieldState(U0), torch.Generator().manual_seed(5))
            if not grad:
                return W.sum().item()
            g = torch.autograd.grad(W.sum(), layers[-1].rho_layer)[0]
            return g.reshape(-1), layers[-1].rho_layer.detach().clone(), act.phi

        phi_at_layer = []
        g, rho0, phi_end = run(grad=True, saved_phi=phi_at_layer)
        self.assertFalse(torch.equal(phi_at_layer[0], phi_end))  # phi redrawn after the last layer
        g_fd = torch.zeros_like(g)
        for k in range(g.numel()):
            def f(h, k=k):
                r = rho0.clone().reshape(-1)
                r[k] += h
                return run(r.view(rho0.shape))
            g_fd[k] = fd4(f, 1e-3)
        rel_fd = ((g - g_fd).abs() / g_fd.abs()).max().item()
        print(f"\nstaggered dW/drho (last layer) through FermionSNF: autograd {g.tolist()}\n  FD {g_fd.tolist()}\n"
              f"  max rel diff {rel_fd:.1e}")
        self.assertTrue(bool((g.abs() > 1e-3).all()))
        self.assertLess(rel_fd, 1e-6)


class Ensembles:
    ref_ratio = None
    starts = None

    @classmethod
    def build(cls):
        if cls.starts is None:
            cls.ref_ratio, cls.starts = mass_ensembles()  # shared with test_fermions_action_staggered


class StaggeredSNFTest(unittest.TestCase):
    """(b) bookkeeping, (c) rho = 0 limit, (d) Jarzynski with layers in the mass."""

    @classmethod
    def setUpClass(cls):
        Ensembles.build()

    def test_bookkeeping(self):
        U0 = Ensembles.starts[0][:8]
        protocol = [0.25, 0.2]
        snf, act = make_snf("mass", MASS, protocol, 0.04, U0.shape[0])
        jump_solves = record_jump_solves(snf, act.fermion)
        with torch.no_grad():
            state, W, Q, logJ, obs = snf(FieldState(U0), torch.Generator().manual_seed(11))
        self.assertEqual(jump_solves, [1, 1])  # a mass jump costs exactly one solve
        S0 = snf.last_run["S0"]
        check = make_action("mass")
        check.set_parameter(protocol[-1])
        check.phi = state.phi
        S_final = check.action(state.U)
        resid = (W + Q - (S_final - S0 - logJ)).abs().max().item()
        scale = S_final.abs().max().item()
        print(f"\nstaggered mass: max |W + Q - (S_final - S_0 - logJ)| = {resid:.2e} (scale {scale:.1f}), "
              f"<logJ> = {logJ.mean().item():+.4f}, <W> = {W.mean().item():+.4f}")
        self.assertLess(resid, 1e-10 * scale)
        self.assertGreater(logJ.abs().min().item(), 1e-3)
        W_parts = sum(snf.last_run["jump_work"]) + sum(snf.last_run["layer_work"])
        self.assertLess((W_parts - W).abs().max().item(), 1e-12 * scale)
        self.assertTrue(torch.equal(obs[:, -1, 0], W))

    def test_rho_zero_is_nemcmc(self):
        U = Ensembles.starts[3][:16]
        protocol = linear_protocol(MASS, MASS2, 4)
        snf, _ = make_snf("mass", MASS, protocol, 0.0, U.shape[0], reuse_guess=False)
        with torch.no_grad():
            st1, W1, Q1, logJ1, _ = snf(FieldState(U), torch.Generator().manual_seed(21))
        act = make_action("mass", reuse_guess=False)
        ne = FermionNEMCMC(act, protocol, **HMC_KW)
        st2, W2, Q2, _ = ne.forward(FieldState(U), torch.Generator().manual_seed(21))
        dW = (W1 - W2).abs().max().item()
        print(f"\nstaggered rho = 0: max |W_SNF - W_NEMCMC| = {dW:.1e}, max |Q diff| = {(Q1 - Q2).abs().max().item():.1e}, "
              f"max |logJ| = {logJ1.abs().max().item():.1e}, max |U_SNF - U_NEMCMC| = {(st1.U - st2.U).abs().max().item():.1e}")
        self.assertLess(dW, 1e-12)
        self.assertLess((Q1 - Q2).abs().max().item(), 1e-10)
        self.assertLess(logJ1.abs().max().item(), 1e-12)
        self.assertLess((st1.U - st2.U).abs().max().item(), 1e-12)

    def test_jarzynski_with_layers(self):
        m = Ensembles.ref_ratio.mean(0)
        r, e_r = m.mean().item(), (m.std() / math.sqrt(m.numel())).item()
        dF_ref, e_ref = -math.log(r), e_r / r
        protocol = linear_protocol(MASS, MASS2, 4)
        gen = torch.Generator().manual_seed(4004)
        Ws, logJs = [], []
        for U in Ensembles.starts:
            snf, _ = make_snf("mass", MASS, protocol, 0.005, U.shape[0])
            with torch.no_grad():
                _, W, _, logJ, _ = snf(FieldState(U), gen)
            Ws.append(W)
            logJs.append(logJ)
        W = torch.cat(Ws)
        dF, e = jarzynski_free_energy(W)
        msg = (f"staggered mass {MASS} -> {MASS2}, 4 steps, stout rho = 0.005: dF = {dF:.4f}({e:.4f}), "
               f"exact dF = {dF_ref:.4f}({e_ref:.4f}), (dF - dF_ref)/sigma = {(dF - dF_ref) / math.hypot(e, e_ref):+.2f}, "
               f"<W> = {W.mean().item():.4f}, ESS = {effective_sample_size(W):.3f}, "
               f"<logJ> = {torch.cat(logJs).mean().item():+.4f}")
        print("\n" + msg)
        self.assertGreater(abs(r - 1.0), 5 * e_r, msg=msg)
        self.assertLess(abs(dF - dF_ref), 3 * math.hypot(e, e_ref), msg=msg)
        self.assertGreater(torch.cat(logJs).abs().min().item(), 1e-4, msg=msg)


if __name__ == "__main__":
    unittest.main()
