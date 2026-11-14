"""The fast path (neoqcd.fermions.fast) beyond the plain HMC: QCDAction(fast=True) and
ParametricFastTwoFlavourWilson, FermionNEMCMC, FermionSNF / PseudofermionAction, FermionREX, the
signed stout layer, and the --fast option of the three drivers.

Every fast path is compared with the reference path (the ground truth) at the same seeds: the
heat-bath draws are the same, so actions, forces, works and swap tables agree to the solver
tolerance and the Metropolis decisions are the same.

CPU, complex128; 2D 4 x 4 SU(3), beta = 5, kappa = 0.18, smooth random links (B = 4), tight
solver tolerances (solver_tol 1e-12, action_tol 1e-13).
"""

import importlib.util
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from neoqcd.fermions._gauge import CouplingLayer, FlowPars, create_mask  # noqa: E402
from neoqcd.fermions.action import (ParametricFastTwoFlavourWilson, ParametricTwoFlavourWilson,  # noqa: E402
                                    QCDAction)
from neoqcd.fermions.algebra import (algebra_basis, algebra_components, expi, project_hermitian_traceless,  # noqa: E402
                                     random_algebra)
from neoqcd.fermions.hasenbusch import RatioWilsonTerm  # noqa: E402
from neoqcd.fermions.io import read_observables  # noqa: E402
from neoqcd.fermions.lattice import Geometry  # noqa: E402
from neoqcd.fermions.rex import FermionREX, FermionREXTarget, swap_action_difference  # noqa: E402
from neoqcd.fermions.snf import FermionNEMCMC, FieldState  # noqa: E402
from neoqcd.fermions.snf_layers import (FermionLayerBlock, FermionSNF, SignedCouplingLayer,  # noqa: E402
                                        _fermion_kind, make_gauge_layers, pseudofermion_action)

G2 = Geometry(2, 4, 4, 3)
BETA, KAPPA = 5.0, 0.18
B = 4
TOL = dict(solver_tol=1e-12, action_tol=1e-13)
HMC_KW = dict(tau=0.5, n_steps=2, n_traj=1, integrator="omelyan")
PROTOCOLS = {"kappa": (KAPPA, [0.19, 0.2]), "beta": (BETA, [5.15, 5.3]), "bc": (1.0, [0.5, 0.0])}


def links(seed, batch=B, scale=0.3):
    return expi(scale * random_algebra(G2.link_shape(batch), torch.Generator().manual_seed(seed)))


def defect_mask():
    d = torch.zeros((G2.D,) + G2.lattice, dtype=torch.double)
    d[0, 0, 0:2] = 1.0
    return torch.stack((1.0 - d, d), dim=0)


def make_action(name, fast, **kw):
    kw = dict(TOL, **kw)
    return QCDAction(G2, name, beta=BETA, kappa=KAPPA, defect_mask=defect_mask(), n_gauge_substeps=2, fast=fast, **kw)


def flow_pars(protocol, batch_size):
    return FlowPars(D=G2.D, T=G2.T, L=G2.L, N=G2.N, mask=create_mask(G2.D, G2.T, G2.L), protocol=list(protocol),
                    batch_size=batch_size, device="cpu")


def fd4(f, h):
    return (8.0 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12.0 * h)


def amax(x):
    return x.abs().max().item()


# ---- 1. QCDAction(fast=True) ---------------------------------------------------------------------
class QCDActionFastTest(unittest.TestCase):

    def test_actions_forces_jumps(self):
        U = links(1)
        for name, (_, protocol) in PROTOCOLS.items():
            with self.subTest(parameter=name):
                ref, fast = make_action(name, False), make_action(name, True)
                self.assertIsInstance(fast.fermion, ParametricFastTwoFlavourWilson)
                self.assertNotIsInstance(fast.fermion, ParametricTwoFlavourWilson)
                for a in (ref, fast):
                    a.refresh(U, torch.Generator().manual_seed(2))
                self.assertEqual(tuple(fast.phi.shape), tuple(ref.phi.shape))
                self.assertLess(amax(fast.phi - ref.phi), 1e-13)
                S_r, S_f = ref.action(U), fast.action(U)
                self.assertLess(amax(S_f - S_r), 1e-9)
                for p in protocol:  # the action at another parameter value and the jump
                    self.assertLess(amax(fast.action(U, p) - ref.action(U, p)), 1e-9)
                    self.assertLess(amax(fast.action_difference(U, fast.parameter, p)
                                         - ref.action_difference(U, ref.parameter, p)), 1e-9)
                h_r, h_f = ref.hamiltonian(protocol[0]), fast.hamiltonian(protocol[0])
                # the fast two-level Hamiltonian: fast fermion term outside, the same gauge term inside
                self.assertIs(h_f.fermion, fast.fermion)
                self.assertEqual([(len(t), n) for t, n in h_f.levels], [(len(t), n) for t, n in h_r.levels])
                self.assertIs(h_f.levels[0][0][0], fast.fermion)
                self.assertLess(amax(h_f.gauge.action(U) - h_r.gauge.action(U)), 1e-12)
                F_r, F_f = h_r.fermion.force(U), h_f.fermion.force(U)
                self.assertGreater(amax(F_r), 1e-2)
                self.assertLess(amax(F_f - F_r), 1e-9)

    def test_kappa_jump_costs_one_solve(self):
        U = links(3)
        for fast in (False, True):
            act = make_action("kappa", fast)
            ne = FermionNEMCMC(act, [0.19], **HMC_KW)
            act.refresh(U, torch.Generator().manual_seed(4))
            S = act.action(U)
            before = act.fermion.stats["solves"]
            dW, S_new = ne._jump(U, KAPPA, 0.19, S)
            self.assertEqual(act.fermion.stats["solves"] - before, 1, fast)
            self.assertLess(amax(S_new - act.action(U, 0.19)), 1e-9)

    def test_refresh_cache(self):
        """eta^dagger eta of the refresh is used only at the refresh links (same _version) and the
        refresh kappa; set_phi drops it; same solve counts as the reference."""
        U, U2 = links(5), links(6)
        counts = {}
        for fast in (False, True):
            f = make_action("kappa", fast).fermion
            f.refresh(U, torch.Generator().manual_seed(7))
            n = []
            for step in ("cached", "other_kappa", "back", "in_place", "set_phi"):
                if step == "other_kappa":
                    f.set_kappa(0.2)
                elif step == "back":
                    f.set_kappa(KAPPA)
                elif step == "in_place":
                    U = U.clone()
                    f.refresh(U, torch.Generator().manual_seed(7))
                    U.mul_(1.0)  # bumps _version
                elif step == "set_phi":
                    f.refresh(U, torch.Generator().manual_seed(7))
                    f.set_phi(f.phi)
                s0 = f.stats["solves"]
                d = f.action_difference(U, U2)
                n.append(f.stats["solves"] - s0)
                honest = f.action(U2) - f.action(U)
                self.assertLess(amax(d - honest), 1e-9, (fast, step))
            counts[fast] = n
        self.assertEqual(counts[True], counts[False])
        self.assertEqual(counts[True], [1, 2, 1, 2, 2])

    def test_phi_round_trip_and_private_kappa(self):
        U = links(8)
        act = make_action("kappa", True)
        act.refresh(U, torch.Generator().manual_seed(9))
        phi = act.phi
        act.phi = phi
        self.assertTrue(torch.equal(act.phi, phi))
        bad = phi.clone()
        bad[:, 0, 1] = 1.0  # site (0, 1) is odd
        with self.assertRaises(ValueError):
            act.phi = bad
        k = torch.full((B,), 0.19, dtype=torch.float64)
        act.set_parameter(k)
        k.fill_(0.3)
        self.assertTrue(torch.equal(act.fermion.kappa, torch.full((B,), 0.19, dtype=torch.float64)))
        ref = make_action("kappa", False)
        ref.phi = phi
        ref.set_parameter(torch.full((B,), 0.19, dtype=torch.float64))
        self.assertLess(amax(act.action(U) - ref.action(U)), 1e-9)  # per-configuration kappa

    def test_options_checked(self):
        with self.assertRaises(ValueError):
            QCDAction(G2, "mass", beta=BETA, fermions="staggered", mass=0.3, fast=True)
        with self.assertRaises(ValueError):
            QCDAction(G2, "beta", beta=BETA, kappa=KAPPA, precision="mixed")
        with self.assertRaises(ValueError):
            QCDAction(G2, "beta", beta=BETA, kappa=KAPPA, fast=True, even_odd=False)
        act = QCDAction(G2, "beta", beta=BETA, kappa=KAPPA, fast=True, precision="mixed", check_every=3,
                        colour_product="broadcast")
        self.assertEqual((act.fermion.precision, act.fermion.check_every, act.fermion.colour_product),
                         ("mixed", 3, "broadcast"))

    def test_mixed_precision_action(self):
        U = links(10)
        ref = make_action("kappa", False)
        mixed = make_action("kappa", True, precision="mixed")
        for a in (ref, mixed):
            a.refresh(U, torch.Generator().manual_seed(11))
        self.assertLess(amax(mixed.action(U, 0.2) - ref.action(U, 0.2)), 1e-9)


# ---- 2. FermionNEMCMC ------------------------------------------------------------------------------
class NEMCMCFastTest(unittest.TestCase):

    def test_same_run(self):
        U0 = links(12)
        for name, (start, protocol) in PROTOCOLS.items():
            with self.subTest(parameter=name):
                out = {}
                for fast in (False, True):
                    act = make_action(name, fast)
                    act.set_parameter(start)
                    ne = FermionNEMCMC(act, protocol, **HMC_KW)
                    state, W, Q, obs = ne.forward(FieldState(U0.clone()), torch.Generator().manual_seed(13))
                    acc = [i["acceptance"] for i in ne.last_run["block_info"]]
                    out[fast] = (state, W, Q, obs, acc)
                (s_r, W_r, Q_r, o_r, a_r), (s_f, W_f, Q_f, o_f, a_f) = out[False], out[True]
                self.assertEqual(a_f, a_r)
                self.assertLess(amax(W_f - W_r), 1e-8)
                self.assertLess(amax(Q_f - Q_r), 1e-8)
                self.assertLess(amax(o_f - o_r), 1e-8)
                self.assertLess(amax(s_f.U - s_r.U), 1e-10)
                self.assertEqual(tuple(s_f.phi.shape), tuple(s_r.phi.shape))
                self.assertLess(amax(s_f.phi - s_r.phi), 1e-10)
                if name != "beta":  # the fermionic work is nonzero in kappa; a real test
                    self.assertGreater(amax(W_r), 1e-3)

    def test_state_continues(self):
        """A run continued from a FieldState of the fast path equals the reference continuation
        (phi carried through FieldState round-trips)."""
        U0 = links(14)
        res = {}
        for fast in (False, True):
            act = make_action("kappa", fast)
            ne = FermionNEMCMC(act, [0.19], **HMC_KW)
            gen = torch.Generator().manual_seed(15)
            st, _, _, _ = ne.forward(FieldState(U0.clone()), gen)
            ne2 = FermionNEMCMC(act, [0.2], initial_parameter=0.19, **HMC_KW)
            st2, W2, _, _ = ne2.forward(st, gen)
            res[fast] = (st2.work, W2)
        self.assertLess(amax(res[True][0] - res[False][0]), 1e-8)
        self.assertLess(amax(res[True][1] - res[False][1]), 1e-8)


# ---- 3. SNF layers ---------------------------------------------------------------------------------
class SNFFastTest(unittest.TestCase):

    def test_kind_dispatch(self):
        self.assertEqual(_fermion_kind(make_action("beta", True).fermion), "wilson_fast")
        self.assertEqual(_fermion_kind(make_action("beta", False).fermion), "wilson")
        stag = QCDAction(G2, "mass", beta=BETA, fermions="staggered", mass=0.3)
        self.assertEqual(_fermion_kind(stag.fermion), "staggered")
        with self.assertRaises(TypeError):
            _fermion_kind(RatioWilsonTerm(KAPPA, 0.1, G2))

    def test_function_value_and_gradient_through_layer(self):
        U = links(16)
        ref, fast = make_action("kappa", False).fermion, make_action("kappa", True).fermion
        for f in (ref, fast):
            f.refresh(U, torch.Generator().manual_seed(17))
        fp = flow_pars([KAPPA], B)
        layer = make_gauge_layers(fp, "smearing", rho=0.0)[0]
        rho0 = torch.tensor([0.05, 0.08, 0.03, 0.06], dtype=torch.float64).view(fp.rho_shape)
        with torch.no_grad():
            layer.rho_layer.copy_(rho0)

        def loss(fermion):
            U_new, _, _ = layer(U, fp, None, beta=KAPPA, delta_beta=0.0)
            return pseudofermion_action(U_new, fermion).sum()

        L_r, L_f = loss(ref), loss(fast)
        self.assertLess(abs(L_f.item() - L_r.item()), 1e-9)
        g_r = torch.autograd.grad(loss(ref), layer.rho_layer)[0].reshape(-1)
        g_f = torch.autograd.grad(loss(fast), layer.rho_layer)[0].reshape(-1)
        g_fd = torch.zeros_like(g_f)
        for k in range(g_fd.numel()):
            def f(h, k=k):
                with torch.no_grad():
                    r = rho0.clone().reshape(-1)
                    r[k] += h
                    layer.rho_layer.copy_(r.view(fp.rho_shape))
                    return loss(fast).item()
            g_fd[k] = fd4(f, 1e-3)
        with torch.no_grad():
            layer.rho_layer.copy_(rho0)
        print(f"\nfast dS_pf/drho through a stout layer: {g_f.tolist()}\n  reference {g_r.tolist()}\n  FD {g_fd.tolist()}")
        self.assertTrue(bool((g_f.abs() > 1e-3).all()))
        self.assertLess(amax(g_f - g_r), 1e-9 * max(1.0, amax(g_r)))
        self.assertLess(((g_f - g_fd).abs() / g_fd.abs()).max().item(), 1e-6)
        # the fast Function stats: one solve per forward
        s0 = fast.stats["solves"]
        loss(fast)
        self.assertEqual(fast.stats["solves"] - s0, 1)

    def test_flow_same_run_and_gradient(self):
        U0 = links(18)
        for name, (start, protocol) in PROTOCOLS.items():
            with self.subTest(parameter=name):
                out = {}
                for fast in (False, True):
                    act = make_action(name, fast)
                    act.set_parameter(start)
                    fp = flow_pars(protocol, B)
                    torch.manual_seed(0)
                    snf = FermionSNF(act, protocol, make_gauge_layers(fp, "smearing", rho=0.04), fp, **HMC_KW)
                    state, W, Q, logJ, obs = snf(FieldState(U0.clone()), torch.Generator().manual_seed(19))
                    snf.loss(W).backward()
                    grads = torch.cat([p.grad.reshape(-1) for p in snf.parameters()])
                    acc = [i["acceptance"] for i in snf.last_run["block_info"]]
                    out[fast] = (W.detach(), Q, logJ, obs, grads, acc, state)
                r, f = out[False], out[True]
                self.assertEqual(f[5], r[5])
                for k in range(4):
                    self.assertLess(amax(f[k] - r[k]), 1e-8, k)
                self.assertGreater(amax(r[4]), 1e-3)
                self.assertLess(amax(f[4] - r[4]), 1e-8 * max(1.0, amax(r[4])))
                self.assertLess(amax(f[6].phi - r[6].phi), 1e-10)


# ---- 4. signed stout layer (identity start) --------------------------------------------------------
class SignedLayerTest(unittest.TestCase):

    def test_identity_and_gradient_at_zero(self):
        U = links(20)
        act = make_action("kappa", True)
        act.refresh(U, torch.Generator().manual_seed(21))
        fp = flow_pars([KAPPA], B)
        layer = make_gauge_layers(fp, "smearing_signed", rho=0.0)[0]
        plain = make_gauge_layers(fp, "smearing", rho=0.0)[0]
        self.assertIsInstance(layer, SignedCouplingLayer)
        with torch.no_grad():
            U1, _, lj = layer(U, fp, None, beta=KAPPA, delta_beta=0.0)
        self.assertLess(amax(U1 - U), 1e-14)
        self.assertLess(amax(torch.as_tensor(lj)), 1e-14)

        blk = FermionLayerBlock(layer, act, fp, KAPPA, 0.0)
        blk_plain = FermionLayerBlock(plain, act, fp, KAPPA, 0.0)
        _, dW, _, _ = blk(FieldState(U, act.phi), KAPPA)
        g = torch.autograd.grad(dW.sum(), layer.rho_layer)[0].reshape(-1)
        _, dW_p, _, _ = blk_plain(FieldState(U, act.phi), KAPPA)
        g_plain = torch.autograd.grad(dW_p.sum(), plain.rho_layer)[0].reshape(-1)
        self.assertEqual(amax(g_plain), 0.0)  # |rho| at 0: untrainable
        g_fd = torch.zeros_like(g)
        for k in range(g.numel()):
            def f(h, k=k):
                with torch.no_grad():
                    r = torch.zeros(g.numel(), dtype=torch.float64)
                    r[k] = h
                    layer.rho_layer.copy_(r.view(fp.rho_shape))
                    return blk(FieldState(U, act.phi), KAPPA)[1].sum().item()
            g_fd[k] = fd4(f, 1e-4)
        with torch.no_grad():
            layer.rho_layer.zero_()
        print(f"\nsigned layer d<W>/drho at rho = 0: {g.tolist()}\n  FD {g_fd.tolist()}")
        self.assertGreater(amax(g), 1e-2)
        self.assertLess(((g - g_fd).abs() / g_fd.abs().clamp(min=1e-3)).max().item(), 1e-6)

    def test_same_as_stout_for_positive_rho(self):
        U = links(22)
        fp = flow_pars([KAPPA], B)
        a = make_gauge_layers(fp, "smearing_signed", rho=0.05)[0]
        b = make_gauge_layers(fp, "smearing", rho=0.05)[0]
        with torch.no_grad():
            Ua, _, la = a(U, fp, None, beta=KAPPA, delta_beta=0.0)
            Ub, _, lb = b(U, fp, None, beta=KAPPA, delta_beta=0.0)
        self.assertTrue(torch.equal(Ua, Ub) and torch.equal(torch.as_tensor(la), torch.as_tensor(lb)))
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))

    def test_haar_logdet_negative_rho(self):
        """The layer's dlogJ at negative rho is the exact Haar log-Jacobian (dense, as
        test_fermions_snf.HaarJacobianTest at positive rho)."""
        gen = torch.Generator().manual_seed(31)
        U0 = expi(random_algebra((1, G2.D, G2.T, G2.L, G2.N, G2.N), gen))
        fp = flow_pars([KAPPA], 1)
        layer = make_gauge_layers(fp, "smearing_signed", rho=0.0)[0]
        with torch.no_grad():
            layer.rho_layer.copy_(torch.tensor([-0.05, 0.03, -0.04, -0.02], dtype=torch.float64).view(fp.rho_shape))
        T = algebra_basis(G2.N)
        with torch.no_grad():
            U0p, _, dlogJ = layer(U0, fp, None, beta=KAPPA, delta_beta=0.0)
        dlogJ = float(torch.as_tensor(dlogJ).reshape(-1)[0])
        n = U0[..., 0, 0].numel() * T.shape[0]

        def f(x):
            X = torch.einsum("...a,aij->...ij", x.view(U0.shape[:-2] + (T.shape[0],)).to(U0.dtype), T)
            Up, _, _ = layer(torch.matmul(expi(X), U0), fp, None, beta=KAPPA, delta_beta=0.0)
            Y = project_hermitian_traceless(-1j * torch.matmul(Up, U0p.conj().transpose(-1, -2)))
            return algebra_components(Y).reshape(-1)

        J = torch.autograd.functional.jacobian(f, torch.zeros(n, dtype=torch.float64), vectorize=True)
        sign, logdet = torch.linalg.slogdet(J)
        print(f"\nHaar log-Jacobian, signed stout (negative rho): dense {logdet.item():.15f}, layer {dlogJ:.15f}")
        self.assertEqual(sign.item(), 1.0)
        self.assertGreater(abs(dlogJ), 1e-2)
        self.assertLess(abs(logdet.item() - dlogJ), 1e-10)

    def test_rejects_hyper(self):
        fp = flow_pars([KAPPA], 1)
        fp.fit_train = 1
        with self.assertRaises(ValueError):
            SignedCouplingLayer(fp, torch.zeros(fp.rho_shape, dtype=torch.float64), 1.0)
        self.assertTrue(issubclass(SignedCouplingLayer, CouplingLayer))


# ---- 5. REX ----------------------------------------------------------------------------------------
class REXFastTest(unittest.TestCase):

    def test_same_run(self):
        runs = {}
        for fast in (False, True):
            rex = FermionREX(G2, BETA, KAPPA, defect_mask(), [0.0, 0.5, 1.0], n_chains=2, tau=0.5, n_steps=2,
                             n_gauge_substeps=2, periodic_translation=True, U=links(23, batch=6), swap_seed=24,
                             fast=fast, **TOL)
            self.assertEqual(type(rex.qcd.fermion).__name__,
                             "ParametricFastTwoFlavourWilson" if fast else "ParametricTwoFlavourWilson")
            gen = torch.Generator().manual_seed(25)
            stats = rex.run(4, gen)
            table = swap_action_difference(rex.qcd, rex.U, rex.parameters(), rex.ladder)
            runs[fast] = (rex, stats, table)
        (r, s_r, t_r), (f, s_f, t_f) = runs[False], runs[True]
        for a, b in zip(s_r, s_f):
            self.assertEqual(a["link_accepted"], b["link_accepted"])
            self.assertEqual(a["hmc_acceptance"], b["hmc_acceptance"])
            for k in a["swap_dS"]:
                self.assertLess(amax(a["swap_dS"][k] - b["swap_dS"][k]), 1e-10)
        for name in ("labels_history", "trajectory_labels_history"):
            for a, b in zip(getattr(r, name), getattr(f, name)):
                self.assertTrue(torch.equal(a, b), name)
        for name in ("plaquette_history", "defect_action_history", "exp_minus_dH_history"):
            for a, b in zip(getattr(r, name), getattr(f, name)):
                self.assertLess(amax(a - b), 1e-8, name)
        self.assertLess(amax(t_f - t_r), 1e-10)

    def test_target_trajectory(self):
        U = links(26)
        out = {}
        for fast in (False, True):
            qcd = make_action("bc", fast)
            qcd.set_parameter(torch.tensor([0.0, 0.5, 1.0, 1.0], dtype=torch.float64))
            target = FermionREXTarget(qcd, tau=0.5, n_steps=2)
            U1, acc, dH, _ = target.trajectory(U.clone(), torch.Generator().manual_seed(27))
            out[fast] = (U1, acc, dH, target.action_table(U1, torch.tensor([0.0, 0.5, 1.0])))
        self.assertTrue(torch.equal(out[True][1], out[False][1]))
        self.assertLess(amax(out[True][0] - out[False][0]), 1e-10)
        self.assertLess(amax(out[True][2] - out[False][2]), 1e-8)
        self.assertLess(amax(out[True][3] - out[False][3]), 1e-10)


# ---- 6. drivers ------------------------------------------------------------------------------------
def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "main" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SNF = _load("main_snf_fermions")
REX = _load("main_rex_fermions")
LATTICE = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "5.0", "--kappa", "0.18",
           "--tau", "0.5", "--n-steps", "2", "--n-gauge-substeps", "2", "--device", "cpu", "--quiet",
           "--solver-tol", "1e-12", "--action-tol", "1e-13"]


class DriversFastTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _compare(self, a, b, name, fields, atol=1e-8):
        ca, cb = read_observables(a / name), read_observables(b / name)
        self.assertEqual(sorted(ca), sorted(cb))
        for k in fields:
            np.testing.assert_allclose(cb[k], ca[k], rtol=0, atol=atol, err_msg=f"{name}:{k}")

    def test_snf_driver(self):
        base = LATTICE + ["--protocol", "bc", "--final", "0.0", "--n-step", "2", "--mode", "snf",
                          "--layers", "smearing", "--rho-init", "0.01", "--defect-shape", "line",
                          "--defect-size", "2", "--batch", "4", "--therm", "2", "--seed", "5",
                          "--train-steps", "3", "--eval-samples", "8", "--checkpoint-every", "1"]
        ref, fast, part = self.tmp / "ref", self.tmp / "fast", self.tmp / "part"
        SNF.main(base + ["--out", str(ref)])
        SNF.main(base + ["--out", str(fast), "--fast"])
        self._compare(ref, fast, "samples.csv", ("work", "heat", "logJ", "S0", "S_final", "plaquette", "solves"))
        self._compare(ref, fast, "training.csv", ("loss", "work_mean", "grad_norm", "param_abs_mean", "acceptance"))
        self._compare(ref, fast, "blocks.csv", ("acceptance", "exp_mdH", "work_mean", "plaquette_mean"))
        meta = __import__("json").loads((fast / "run_meta.json").read_text())
        self.assertTrue(meta["parameters"]["fast"])
        self.assertEqual(meta["run"]["fermion_operator"], "half_lattice_fast")
        self.assertEqual(meta["run"]["colour_product"], "matmul")
        # interrupted fast run (inside training, then inside evaluation) resumes bitwise
        for stop in (2, 5):
            calls = {"n": 0}
            original = SNF.effective_sample_size

            def failing(W, original=original, stop=stop):
                calls["n"] += 1
                if calls["n"] == stop:
                    raise RuntimeError("simulated wall time")
                return original(W)
            SNF.effective_sample_size = failing
            try:
                with self.assertRaises(RuntimeError):
                    SNF.main(base + ["--out", str(part), "--fast"] + (["--resume"] if stop == 5 else []))
            finally:
                SNF.effective_sample_size = original
        # a fast checkpoint is not resumed by a reference run (and vice versa)
        with self.assertRaisesRegex(SystemExit, "differ"):
            SNF.main(base + ["--out", str(part), "--resume"])
        with self.assertRaisesRegex(SystemExit, "differ"):
            SNF.main(base + ["--out", str(ref), "--resume", "--fast"])
        SNF.main(base + ["--out", str(part), "--resume", "--fast"])
        for name, field in (("training.csv", "loss"), ("samples.csv", "work"), ("samples.csv", "S_final")):
            np.testing.assert_array_equal(read_observables(fast / name)[field], read_observables(part / name)[field])

    def test_nemcmc_kappa_driver(self):
        base = LATTICE + ["--protocol", "kappa", "--final", "0.2", "--n-step", "2", "--mode", "nemcmc",
                          "--batch", "4", "--eval-samples", "8", "--therm", "3", "--seed", "3"]
        ref, fast, mixed = self.tmp / "ref", self.tmp / "fast", self.tmp / "mixed"
        SNF.main(base + ["--out", str(ref)])
        SNF.main(base + ["--out", str(fast), "--fast", "--check-every", "3"])
        SNF.main(base + ["--out", str(mixed), "--fast", "--precision", "mixed"])
        for other in (fast, mixed):
            self._compare(ref, other, "samples.csv", ("work", "heat", "S0", "S_final", "plaquette", "solves"))
        a, b = SNF.physics_key(SNF.parse_args(base + ["--fast"])), SNF.physics_key(SNF.parse_args(base))
        self.assertNotEqual(a, b)
        self.assertNotIn("fast", b)  # reference keys unchanged: old checkpoints still resume

    def test_args(self):
        base = LATTICE + ["--protocol", "beta", "--final", "5.2", "--n-step", "1"]
        for extra in (["--precision", "mixed"], ["--check-every", "3"], ["--colour-product", "matmul"],
                      ["--fast", "--inner-tol", "1e-5"], ["--fast", "--check-every", "0"]):
            with self.assertRaises(SystemExit):
                SNF.parse_args(base + extra)
        stag = [a for a in base if a not in ("--kappa", "0.18")] + ["--fermions", "staggered", "--mass", "0.3"]
        for extra in (["--fast"], ["--on-unconverged", "warn"]):
            with self.assertRaises(SystemExit):
                SNF.parse_args(stag + extra)
        with self.assertRaises(SystemExit):  # untrainable identity start of the |rho| layer
            SNF.parse_args(LATTICE + ["--protocol", "beta", "--final", "5.2", "--n-step", "1", "--mode", "snf",
                                      "--rho-init", "0", "--train-steps", "1"])
        args = SNF.parse_args(LATTICE + ["--protocol", "beta", "--final", "5.2", "--n-step", "1", "--mode", "snf",
                                         "--layers", "smearing_signed", "--rho-init", "0", "--train-steps", "1",
                                         "--lr-warmup", "4"])
        self.assertEqual([SNF.learning_rate(args, s) for s in range(5)],
                         [args.lr * (s + 1) / 4 for s in range(4)] + [args.lr])

    def test_signed_identity_start_driver(self):
        out = self.tmp / "signed"
        SNF.main(LATTICE + ["--protocol", "kappa", "--final", "0.2", "--n-step", "2", "--mode", "snf",
                            "--layers", "smearing_signed", "--rho-init", "0", "--lr", "1e-4", "--lr-warmup", "2",
                            "--batch", "4", "--train-steps", "3", "--eval-samples", "4", "--seed", "7",
                            "--fast", "--out", str(out)])
        t = read_observables(out / "training.csv")
        self.assertEqual(t["param_abs_mean"][0], 0.0)  # the identity at step 0
        self.assertGreater(t["param_abs_mean"][1], 0.0)  # and it moves
        self.assertEqual(t["logJ_mean"][0], 0.0)
        np.testing.assert_allclose(t["lr"], [5e-5, 1e-4, 1e-4])

    def test_rex_driver(self):
        common = LATTICE + ["--ladder", "0,0.5,1", "--defect-shape", "line", "--defect-size", "2",
                            "--n-chains", "2", "--therm", "2", "--seed", "11", "--periodic-translation",
                            "--steps", "4", "--checkpoint-every", "2"]
        ref, fast = self.tmp / "ref", self.tmp / "fast"
        REX.main(common + ["--out", str(ref)])
        REX.main(common + ["--out", str(fast), "--fast"])
        self._compare(ref, fast, "observables.csv", ("plaquette", "exp_mdH", "swap_accepted", "swap_proposed",
                                                     "hmc_acceptance", "solves"))
        with np.load(ref / "replica_history.npz") as a, np.load(fast / "replica_history.npz") as b:
            for name in ("labels", "trajectory_labels"):
                np.testing.assert_array_equal(a[name], b[name])
            for name in ("plaquette", "defect_action", "exp_minus_dH"):
                np.testing.assert_allclose(a[name], b[name], rtol=0, atol=1e-8)
        with self.assertRaisesRegex(SystemExit, "differ"):
            REX.main(common + ["--steps", "6", "--out", str(fast), "--resume"])
        with self.assertRaisesRegex(SystemExit, "differ"):
            REX.main(common + ["--steps", "6", "--out", str(ref), "--resume", "--fast"])
        REX.main(common + ["--steps", "6", "--out", str(fast), "--resume", "--fast"])  # its own: fine


@unittest.skipIf(os.environ.get("NEOQCD_SKIP_GLOO"), "NEOQCD_SKIP_GLOO set")
class DistributedDriverFastTest(unittest.TestCase):
    """main_rex_fermions_distributed.py --fast on three Gloo ranks reproduces the reference run,
    and a resume across fast / reference is refused."""

    def test_fast_vs_reference(self):
        from test.test_fermions_rex_distributed import WILSON, DriverExit, run_distributed
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            common = WILSON + ["--n-chains", "2", "--therm", "1", "--seed", "11", "--periodic-translation",
                               "--checkpoint-every", "2", "--steps", "3", "--solver-tol", "1e-12",
                               "--action-tol", "1e-13"]
            ref, fast = tmp / "ref", tmp / "fast"
            run_distributed(common + ["--out", str(ref)], tmp / "res_ref")
            run_distributed(common + ["--out", str(fast), "--fast"], tmp / "res_fast")
            a, b = read_observables(ref / "observables.csv"), read_observables(fast / "observables.csv")
            np.testing.assert_allclose(b["plaquette"], a["plaquette"], rtol=0, atol=1e-8)
            for name in ("swap_accepted", "swap_proposed", "solves"):
                np.testing.assert_array_equal(b[name], a[name], err_msg=name)
            with np.load(ref / "replica_history.npz") as ha, np.load(fast / "replica_history.npz") as hb:
                np.testing.assert_array_equal(ha["labels"], hb["labels"])
            self.assertEqual(a["plaquette"].size, 4 * 3)
            with self.assertRaises(DriverExit) as cm:
                run_distributed(common + ["--steps", "5", "--out", str(ref), "--resume", "--fast"], tmp / "res_x")
            self.assertIn("'fast': True", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
