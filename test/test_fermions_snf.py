"""Tests of the fermionic SNF (neoqcd.fermions.snf_layers, phase 4 step 2 of fermions_plan.md):
the custom autograd Function of S_pf, the work bookkeeping with deterministic gauge layers,
Jarzynski with layers against exact reweighting, the rho = 0 limit, a short training run; and the
version-keyed refresh cache of `TwoFlavourWilson.action_difference` (phase-1 review item).

CPU, complex128, fixed seeds; 2D 4 x 4 SU(3), beta = 5, kappa = 0.18 (the setting of
test_fermions_nemcmc, whose equilibrium ensembles are reused); layers: the pure-gauge stout
`CouplingLayer` (isotropic rho, D = 2).
"""

import math
import unittest

import torch

from neoqcd.fermions._gauge import FlowPars, create_mask
from neoqcd.fermions.action import QCDAction
from neoqcd.fermions.algebra import (algebra_basis, algebra_components, expi, project_hermitian_traceless,
                                     random_algebra)
from neoqcd.fermions.exact import assemble_even_odd
from neoqcd.fermions.lattice import parity
from neoqcd.fermions.pseudofermion import TwoFlavourWilson
from neoqcd.fermions.snf import FermionNEMCMC, FieldState, effective_sample_size, jarzynski_free_energy
from neoqcd.fermions.snf_layers import (FermionLayerBlock, FermionSNF, PseudofermionAction,
                                        joint_action, make_gauge_layers, pseudofermion_action)
from test.test_fermions_nemcmc import (BETA, G2, HMC_KW, KAPPA, KAPPA2, N_SUB, Ensembles,
                                       chain_mean_error, defect_mask, linear_protocol)


def flow_pars(protocol, batch_size):
    return FlowPars(D=G2.D, T=G2.T, L=G2.L, N=G2.N, mask=create_mask(G2.D, G2.T, G2.L), protocol=list(protocol),
                    batch_size=batch_size, device="cpu")


def make_action(name, **kw):
    return QCDAction(G2, name, beta=BETA, kappa=KAPPA, defect_mask=defect_mask(), n_gauge_substeps=N_SUB, **kw)


def make_snf(name, start, protocol, rho, B, **kw):
    act = make_action(name, **kw)
    act.set_parameter(start)
    fp = flow_pars(protocol, B)
    layers = make_gauge_layers(fp, "smearing", rho=rho)
    return FermionSNF(act, protocol, layers, fp, **HMC_KW), act


def dense_pf_action(U, phi, kappa):
    """S_pf = |D_hat^-dagger phi|^2 from the dense even-odd matrix, differentiable in U by autograd."""
    B = U.shape[0]
    even = (parity(G2).reshape(-1) == 0).nonzero().squeeze(-1)
    p = phi.reshape(B, G2.volume, -1)[:, even].reshape(B, -1, 1)
    Dh = assemble_even_odd(U, kappa, G2)
    y = torch.linalg.solve(Dh.conj().transpose(-1, -2), p)
    return (y.conj() * y).real.reshape(B, -1).sum(1)


def fd4(f, h):
    """4th-order central difference of a scalar function at 0."""
    return (8.0 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12.0 * h)


class PseudofermionFunctionTest(unittest.TestCase):
    """1. The custom Function: value, directional derivatives along e^{itT} U, gradient through a
    stout layer against 4th-order finite differences and against autograd of a dense solve."""

    @classmethod
    def setUpClass(cls):
        Ensembles.build()
        cls.U = Ensembles.starts[1][:4].clone()
        cls.fermion = TwoFlavourWilson(KAPPA, G2, solver_tol=1e-13, action_tol=1e-14, reuse_guess=False)
        cls.fermion.refresh(cls.U, torch.Generator().manual_seed(5))

    def test_value_and_group_derivative(self):
        U, fermion = self.U, self.fermion
        S = pseudofermion_action(U, fermion)
        self.assertLess((S - fermion.action(U)).abs().max().item(), 1e-12 * S.abs().max().item())
        self.assertLess((S - dense_pf_action(U, fermion.phi, KAPPA)).abs().max().item(), 1e-10 * S.abs().max().item())
        T = random_algebra(U.shape, torch.Generator().manual_seed(6))

        def S_of(t):
            return pseudofermion_action(torch.matmul(expi(t * T), U), fermion)

        t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        S_t = pseudofermion_action(torch.matmul(expi(t * T), U), fermion)
        grad = torch.autograd.grad(S_t.sum(), t)[0].item()
        fd = fd4(lambda h: S_of(torch.tensor(h, dtype=torch.float64)).sum().item(), 1e-3)
        print(f"\nd/dt S_pf(e^(itT) U): autograd {grad:.12e}, 4th-order FD {fd:.12e}, rel {abs(grad - fd) / abs(fd):.1e}")
        self.assertLess(abs(grad - fd), 1e-7 * abs(fd))

    def test_gradient_through_layer(self):
        U, fermion = self.U, self.fermion
        fp = flow_pars([KAPPA], U.shape[0])
        layer = make_gauge_layers(fp, "smearing", rho=0.0)[0]
        rho0 = torch.tensor([0.05, 0.08, 0.03, 0.06], dtype=torch.float64).view(fp.rho_shape)
        with torch.no_grad():
            layer.rho_layer.copy_(rho0)

        def L_custom():
            U_new, _, _ = layer(U, fp, None, beta=KAPPA, delta_beta=0.0)
            return pseudofermion_action(U_new, fermion).sum()

        def L_dense():
            U_new, _, _ = layer(U, fp, None, beta=KAPPA, delta_beta=0.0)
            return dense_pf_action(U_new, fermion.phi, KAPPA).sum()

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
        print(f"\ndS_pf/drho through a stout layer: custom {g_custom.tolist()}\n  FD {g_fd.tolist()}\n"
              f"  dense {g_dense.tolist()}\n  max rel diff vs FD {rel_fd:.1e}, vs dense autograd {rel_dense:.1e}")
        self.assertTrue(bool((g_custom.abs() > 1e-3).all()))
        self.assertLess(rel_fd, 1e-6)
        self.assertLess(rel_dense, 1e-10)

    def test_force_saved_at_forward(self):
        """The backward uses phi and kappa of the forward even if the fermion changes afterwards."""
        U, fermion = self.U, self.fermion
        f2 = TwoFlavourWilson(KAPPA, G2, action_tol=1e-14, reuse_guess=False)
        f2.phi = fermion.phi.clone()
        T = random_algebra(U.shape, torch.Generator().manual_seed(7))
        t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        S = pseudofermion_action(torch.matmul(expi(t * T), U), f2)
        f2.kappa = 0.2
        f2.refresh(U, torch.Generator().manual_seed(8))
        g = torch.autograd.grad(S.sum(), t)[0]
        t2 = torch.zeros((), dtype=torch.float64, requires_grad=True)
        g_ref = torch.autograd.grad(pseudofermion_action(torch.matmul(expi(t2 * T), U), fermion).sum(), t2)[0]
        self.assertLess(abs(g - g_ref).item(), 1e-10 * abs(g_ref).item())


class RefreshCacheTest(unittest.TestCase):
    """0. action_difference after U.copy_ must not use the stale eta^dagger eta of the refresh."""

    def test_in_place_change(self):
        Ensembles.build()
        U = Ensembles.starts[0][:4].clone()
        U2 = Ensembles.starts[1][:4].clone()
        U3 = Ensembles.starts[2][:4].clone()
        f = TwoFlavourWilson(KAPPA, G2, reuse_guess=False)
        f.refresh(U, torch.Generator().manual_seed(9))
        cached = f.action_difference(U, U2)
        honest0 = f.action(U2) - f.action(U)
        self.assertLess((cached - honest0).abs().max().item(), 1e-9 * honest0.abs().max().item())
        U.copy_(U3)
        stale = f.action_difference(U, U2)
        honest = f.action(U2) - f.action(U3)
        self.assertGreater((honest - honest0).abs().max().item(), 1e-3)
        self.assertLess((stale - honest).abs().max().item(), 1e-9 * honest.abs().max().item())


class BookkeepingTest(unittest.TestCase):
    """2. W + Q = S_final - S_0 - logJ per chain (S_final recomputed by an independent action object)
    for 2-step protocols in kappa, beta, bc with stout layers of nonzero rho; layer works equal
    S(U') - S(U) - logJ; accumulators and int_obs consistent."""

    PROTOCOLS = {"kappa": (KAPPA, [0.195, 0.21]), "beta": (BETA, [5.15, 5.3]), "bc": (1.0, [0.5, 0.0])}

    def test_accounting(self):
        Ensembles.build()
        U0 = Ensembles.starts[0][:8]
        for name, (start, protocol) in self.PROTOCOLS.items():
            with self.subTest(parameter=name):
                snf, act = make_snf(name, start, protocol, 0.04, U0.shape[0])
                with torch.no_grad():
                    state, W, Q, logJ, obs = snf(FieldState(U0), torch.Generator().manual_seed(11))
                S0 = snf.last_run["S0"]
                check = make_action(name)
                check.set_parameter(protocol[-1])
                check.phi = state.phi
                S_final = check.action(state.U)
                scale = S_final.abs().max().item()
                resid = (W + Q - (S_final - S0 - logJ)).abs().max().item()
                print(f"\n{name}: max |W + Q - (S_final - S_0 - logJ)| = {resid:.2e} (scale {scale:.1f}), "
                      f"<logJ> = {logJ.mean().item():+.4f}, <W> = {W.mean().item():+.4f}")
                self.assertLess(resid, 1e-10 * scale)
                self.assertGreater(logJ.abs().min().item(), 1e-3)
                self.assertTrue(torch.equal(state.work, W) and torch.equal(state.heat, Q)
                                and torch.equal(state.logJ, logJ))
                self.assertLess((obs[:, :, 0] + obs[:, :, 1] - (obs[:, :, 3] - S0[:, None] - obs[:, :, 2])).abs().max().item(),
                                1e-10 * scale)
                self.assertTrue(torch.equal(obs[:, -1, 0], W))
                W_parts = sum(snf.last_run["jump_work"]) + sum(snf.last_run["layer_work"])
                self.assertLess((W_parts - W).abs().max().item(), 1e-12 * scale)

    def test_layer_work(self):
        """One FermionLayerBlock: dW = S(U') - S(U) - logJ with both actions from scratch."""
        Ensembles.build()
        U = Ensembles.starts[0][:4]
        act = make_action("kappa")
        act.refresh(U, torch.Generator().manual_seed(12))
        fp = flow_pars([KAPPA], U.shape[0])
        layer = make_gauge_layers(fp, "smearing", rho=0.05)[0]
        blk = FermionLayerBlock(layer, act, fp, KAPPA, 0.0)
        with torch.no_grad():
            st, dW, S_new, dlogJ = blk(FieldState(U, act.phi), 0.2)
            ref = make_action("kappa")
            ref.phi = act.phi
            S_U, S_Up = ref.action(U, 0.2), ref.action(st.U, 0.2)
        self.assertLess((dW - (S_Up - S_U - dlogJ)).abs().max().item(), 1e-10 * S_U.abs().max().item())
        self.assertLess((S_new - S_Up).abs().max().item(), 1e-10 * S_U.abs().max().item())
        self.assertGreater((st.U - U).abs().max().item(), 1e-3)


class HaarJacobianTest(unittest.TestCase):
    """2b. The layer's dlogJ is the exact Haar log-Jacobian (SU(3)): one stout `CouplingLayer` at
    rho = 0.05 on one random 2D 4 x 4 configuration; the Jacobian of the whole layer map in
    orthonormal tangent coordinates (input U = e^{iX} U_0, output components of
    -i U'(X) U'(0)^dagger projected on su(3), both in the basis tr(T^a T^b) = delta_ab) by dense
    autograd. Left-invariant coordinates on both sides, so log |det J| is the Haar log-Jacobian;
    the full 256 x 256 determinant (block-triangular over the mask blocks) is exact."""

    def test_logdet(self):
        gen = torch.Generator().manual_seed(31)
        U0 = expi(random_algebra((1, G2.D, G2.T, G2.L, G2.N, G2.N), gen))
        fp = flow_pars([KAPPA], 1)
        layer = make_gauge_layers(fp, "smearing", rho=0.05)[0]
        T = algebra_basis(G2.N)
        with torch.no_grad():
            U0p, _, dlogJ = layer(U0, fp, None, beta=KAPPA, delta_beta=0.0)
        dlogJ = FermionLayerBlock(layer, make_action("kappa"), fp, KAPPA, 0.0)._full_logdet(
            torch.as_tensor(dlogJ, dtype=torch.float64).reshape(-1))[0].item()
        n = U0[..., 0, 0].numel() * T.shape[0]

        def f(x):
            X = torch.einsum("...a,aij->...ij", x.view(U0.shape[:-2] + (T.shape[0],)).to(U0.dtype), T)
            Up, _, _ = layer(torch.matmul(expi(X), U0), fp, None, beta=KAPPA, delta_beta=0.0)
            Y = project_hermitian_traceless(-1j * torch.matmul(Up, U0p.conj().transpose(-1, -2)))
            return algebra_components(Y).reshape(-1)

        J = torch.autograd.functional.jacobian(f, torch.zeros(n, dtype=torch.float64), vectorize=True)
        sign, logdet = torch.linalg.slogdet(J)
        print(f"\nHaar log-Jacobian, stout rho = 0.05, 4x4 SU(3): dense {logdet.item():.15f}, layer {dlogJ:.15f}, "
              f"diff {logdet.item() - dlogJ:.1e}")
        self.assertEqual(sign.item(), 1.0)
        self.assertGreater(abs(dlogJ), 1e-2)
        self.assertLess(abs(logdet.item() - dlogJ), 1e-10)


RHO_J = 0.005


class JarzynskiTest(unittest.TestCase):
    """3. kappa 0.18 -> 0.21 in 4 steps with untrained stout layers (rho = RHO_J): <exp(-W)> = Z(0.21)/Z(0.18)
    from exact reweighting; and rho = 0 reproduces FermionNEMCMC exactly."""

    @classmethod
    def setUpClass(cls):
        Ensembles.build()

    def test_jarzynski_with_layers(self):
        r, e_r = chain_mean_error(Ensembles.ref_det_ratio)
        dF_ref, e_ref = -math.log(r), e_r / r
        protocol = linear_protocol(KAPPA, KAPPA2, 4)
        gen = torch.Generator().manual_seed(4004)
        Ws, logJs = [], []
        for U in Ensembles.starts:
            snf, _ = make_snf("kappa", KAPPA, protocol, RHO_J, U.shape[0])
            with torch.no_grad():
                _, W, _, logJ, _ = snf(FieldState(U), gen)
            Ws.append(W)
            logJs.append(logJ)
        W = torch.cat(Ws)
        dF, e = jarzynski_free_energy(W)
        msg = (f"kappa {KAPPA} -> {KAPPA2}, 4 steps, stout rho = {RHO_J}: dF = {dF:.4f}({e:.4f}), "
               f"exact dF = {dF_ref:.4f}({e_ref:.4f}), (dF - dF_ref)/sigma = {(dF - dF_ref) / math.hypot(e, e_ref):+.2f}, "
               f"<W> = {W.mean().item():.4f}, ESS = {effective_sample_size(W):.3f}, <logJ> = {torch.cat(logJs).mean().item():+.4f}")
        print("\n" + msg)
        self.assertLess(abs(dF - dF_ref), 3 * math.hypot(e, e_ref), msg=msg)
        self.assertGreater(torch.cat(logJs).abs().min().item(), 1e-3, msg=msg)

    def test_rho_zero_is_nemcmc(self):
        U = Ensembles.starts[3][:16]
        protocol = linear_protocol(KAPPA, KAPPA2, 4)
        phi0 = None
        for name in ("kappa", "beta"):
            with self.subTest(parameter=name):
                start, prot = (KAPPA, protocol) if name == "kappa" else (BETA, linear_protocol(BETA, 5.3, 4))
                snf, _ = make_snf(name, start, prot, 0.0, U.shape[0], reuse_guess=False)
                with torch.no_grad():
                    st1, W1, Q1, logJ1, _ = snf(FieldState(U, phi=phi0), torch.Generator().manual_seed(21))
                act = make_action(name, reuse_guess=False)
                act.set_parameter(start)
                ne = FermionNEMCMC(act, prot, **HMC_KW)
                st2, W2, Q2, _ = ne.forward(FieldState(U, phi=phi0), torch.Generator().manual_seed(21))
                dW = (W1 - W2).abs().max().item()
                print(f"\nrho = 0, {name}: max |W_SNF - W_NEMCMC| = {dW:.1e}, max |logJ| = {logJ1.abs().max().item():.1e}, "
                      f"max |U_SNF - U_NEMCMC| = {(st1.U - st2.U).abs().max().item():.1e}")
                self.assertLess(dW, 1e-12)
                self.assertLess((Q1 - Q2).abs().max().item(), 1e-10)
                self.assertLess(logJ1.abs().max().item(), 1e-12)
                self.assertLess((st1.U - st2.U).abs().max().item(), 1e-12)


class TrainingSmokeTest(unittest.TestCase):
    """4. 20 Adam steps on mean(W) for kappa 0.18 -> 0.21 in 2 steps: finite gradients, and the
    mean work at a fixed seed on fixed starts (common random numbers) decreases by more than one
    standard error of the paired difference."""

    def test_training(self):
        Ensembles.build()
        U = Ensembles.starts[2][:16]
        protocol = linear_protocol(KAPPA, KAPPA2, 2)
        snf, _ = make_snf("kappa", KAPPA, protocol, 0.01, U.shape[0])
        params = [p for p in snf.parameters() if p.requires_grad]
        self.assertEqual(len(params), 2)
        opt = torch.optim.Adam(params, lr=5e-3)

        def evaluate():
            with torch.no_grad():
                W = torch.cat([snf(FieldState(Us), torch.Generator().manual_seed(99))[1]
                               for Us in (Ensembles.starts[0], Ensembles.starts[1])])
            return W

        W_before = evaluate()
        gen = torch.Generator().manual_seed(123)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            _, W, _, _, _ = snf(FieldState(U), gen)
            loss = snf.loss(W)
            loss.backward()
            for p in params:
                self.assertTrue(bool(torch.isfinite(p.grad).all()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in params), 0.0)
            opt.step()
            losses.append(loss.item())
        W_after = evaluate()
        diff = W_after - W_before
        d, e = diff.mean().item(), (diff.std() / math.sqrt(diff.numel())).item()
        print(f"\ntraining: losses first/last 5 {[round(x, 4) for x in losses[:5]]} / {[round(x, 4) for x in losses[-5:]]}\n"
              f"  <W> fixed seed before {W_before.mean().item():.4f}, after {W_after.mean().item():.4f}, "
              f"paired diff {d:+.4f}({e:.4f}) = {d / e:+.2f} sigma; rho = {[p.detach().reshape(-1).tolist() for p in params]}")
        # Trained rho may cross zero: the stout layer uses |rho| (Smearing._mask_rho), so the sign
        # of the printed rho is not meaningful, only its modulus.
        self.assertLess(d, -e)


if __name__ == "__main__":
    unittest.main()
