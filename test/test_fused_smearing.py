"""The fused stout step (neoqcd.fused_smearing) against the torch path of smearing_jacobian and Smearing.

On the CPU the kernels run as their emulation (the same source built with the host C++ compiler); on CUDA the
same comparisons run on the GPU. Both paths compute the same functions, in the same order where it matters, so
they agree to float64 rounding:
- the stout step alone (stout_links against smearing_step), link by link: staple sums of six random SU(3)
  matrices times rho = 1e-3 and 0.1, staple sums scaled to c1 below, above and on both sides of C1_SERIES
  (Taylor series and closed forms), Q = 0, Q with two equal eigenvalues and with two 1e-6 and 1e-10 apart: U' to
  TOL_LINK, log|det J| to TOL_LOGDET relative (absolute below |log|det J|| = 1);
- the coefficients of order 2 (exp_coefficients) to TOL_COEFF relative to the largest of each kind;
- the inverse (stout_links_inverse against inverse_smearing_step(fixed_iterations=10)): U, log|det| and
  residual, and inverse(forward) = identity for small rho;
- whole NEO-REX defect smearing layers (main_pt_obc.DefectSNFProtocolLayer, hyper smearing in class mode, 1 and
  2 smearing steps) and a whole-lattice hyper smearing layer, with Smearing.use_fused_kernels on and off: maps
  and log J per configuration, forward and fixed-iteration inverse, to TOL_LAYER; grad mode with an input that
  requires grad (rho from the hypernetwork being trained, or the configurations), the CPU without the emulation
  and the adaptive inverse stay torch, grad mode alone does not.
The deviations are printed. CudaGraph (GPU only) replays the captured layer with the kernels against the eager
torch layer; CudaSource checks the CUDA halves of both kernel sources (fused.py, fused_smearing.py) with clang's
CUDA front end, where available.
"""

import functools
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import neoqcd.fused as fused
import neoqcd.fused_smearing as fs
import neoqcd.smearing_jacobian as sj
import neoqcd.sun_utils as sun
from neoqcd.flow import FlowPars
from neoqcd.smearing import CouplingLayer, Smearing
from neoqcd.utils import Defect, create_mask

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("main_pt_obc_fused", ROOT / "main" / "main_pt_obc.py")
PT_OBC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PT_OBC)

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
TOL_LINK = 1e-13
TOL_LOGDET = 1e-12
TOL_COEFF = 1e-13
TOL_LAYER = 1e-13


def _cpu_backend_or_skip():
    ok, why = fs.backend_status("cpu")
    if not ok:
        raise unittest.SkipTest(f"no CPU build of the smearing kernels: {why}")


def _random_su3(n, g, device):
    return sun.SUN_haar((n, 3), dtype=torch.cdouble, generator=g).to(device)


def _staple_sum(n, rho, g, device):
    # rho times the sum of six random SU(3) matrices: a staple sum of a link with rho on its six staples
    return rho * sum(_random_su3(n, g, device) for _ in range(6))


def _batches(g, device):
    # (U, C) batches of the stout step by name: staple sums at rho = 1e-3 and 0.1; scaled to c1 below
    # C1_SERIES ('small'), above ('large'), both ('mixed', the first 16 above); degenerate: U = 1 with
    # C = s diag(-i, -i, i), for which Q = s diag(-2/3, -2/3, 4/3) has two equal eigenvalues and
    # c1 = 4 s^2/3 >= C1_SERIES, and C = 0 (Q = 0); and Q with two eigenvalues 1e-6, 1e-10 apart
    n, n_large, n_zero = 64, 16, 4
    U = _random_su3(n, g, device)

    def scaled(U, c1_low, c1_high):
        C = _staple_sum(U.shape[0], 1.0, g, device)
        c1 = c1_low + (c1_high - c1_low) * torch.rand(U.shape[0], generator=g, dtype=torch.double)
        return C * torch.sqrt(c1.to(device) / sj.StepQuantities(U, C).c1)[:, None, None]

    small, large = (0.0, 2.9), (3.1, 8.0)
    unit = torch.eye(3, dtype=torch.cdouble, device=device).expand(n_large, 3, 3)
    s = torch.linspace(1.5, 4.0, n_large, dtype=torch.double, device=device)
    diagonal = torch.diag_embed(torch.tensor([-1j, -1j, 1j], dtype=torch.cdouble, device=device) * s[:, None])
    batches = {
        "rho 1e-3": (U, _staple_sum(n, 1e-3, g, device)),
        "rho 0.1": (U, _staple_sum(n, 0.1, g, device)),
        "small c1": (U, scaled(U, *small)),
        "large c1": (U, scaled(U, *large)),
        "mixed c1": (U, torch.cat((scaled(U[:n_large], *large), scaled(U[n_large:], *small)))),
        "degenerate": (torch.cat((unit, U[n_large:])),
                       torch.cat((diagonal, torch.zeros(n_zero, 3, 3, dtype=torch.cdouble, device=device),
                                  scaled(U[n_large + n_zero:], *small)))),
    }
    # C = -i Q0 U gives Q = -Q0: Q0 = V diag(a, a + d, -2a - d) V^dagger, V Haar-random
    V, W = _random_su3(n, g, device), _random_su3(n, g, device)
    a = torch.tensor([0.5, -0.5, 1.5, -1.5], dtype=torch.double).repeat(n // 4).to(device)
    for d in (1e-6, 1e-10):
        eig = torch.stack((a, a + d, -2 * a - d), dim=-1).to(torch.cdouble)
        Q0 = V @ torch.diag_embed(eig) @ V.mH
        batches[f"near-degenerate {d:g}"] = (W, -1j * Q0 @ W)
    return batches


def _report(text):
    print(f"\n  {text}", end=" ", flush=True)


class StoutStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _cpu_backend_or_skip()

    def setUp(self):
        self._grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def tearDown(self):
        torch.set_grad_enabled(self._grad)

    def test_stout_step_link_by_link(self):
        for device in DEVICES:
            g = torch.Generator().manual_seed(3)
            for name, (U, C) in _batches(g, device).items():
                with self.subTest(device=device, batch=name):
                    Up0, logdet0 = sj.smearing_step(U, C)
                    Up, logdet = fs.stout_links(U, C)
                    dev = (Up - Up0).abs().max().item()
                    rel = ((logdet - logdet0).abs() / logdet0.abs().clamp(min=1.0)).max().item()
                    c1 = sj.StepQuantities(U, C).c1
                    _report(f"stout step, {name} ({device}, c1 in [{c1.min().item():.2g}, {c1.max().item():.2g}]): "
                            f"max |U' fused - torch| {dev:.1e}, log|det J| {rel:.1e}")
                    self.assertLessEqual(dev, TOL_LINK)
                    self.assertLessEqual(rel, TOL_LOGDET)
                    if name == "mixed c1":
                        self.assertTrue(bool((c1[:16] >= sj.C1_SERIES).all() and (c1[16:] < sj.C1_SERIES).all()))

    def test_batch_shapes_and_strided_inputs(self):
        g = torch.Generator().manual_seed(4)
        U, C = _random_su3(24, g, "cpu"), _staple_sum(24, 0.1, g, "cpu")
        Up, logdet = fs.stout_links(U, C)
        # a (2, 12) batch, transposed (non-contiguous) and lazily conjugated inputs: the same numbers
        Up2, logdet2 = fs.stout_links(U.reshape(2, 12, 3, 3), C.reshape(2, 12, 3, 3))
        self.assertTrue(torch.equal(Up2.reshape(Up.shape), Up) and torch.equal(logdet2.reshape(-1), logdet))
        Ut = U.mT.contiguous().mT
        Cc = C.conj().resolve_conj().conj()
        self.assertFalse(Ut.is_contiguous())
        self.assertTrue(Cc.is_conj())
        Up3, logdet3 = fs.stout_links(Ut, Cc)
        self.assertTrue(torch.equal(Up3, Up) and torch.equal(logdet3, logdet))
        # no links
        Up4, logdet4 = fs.stout_links(U[:0], C[:0])
        self.assertEqual((tuple(Up4.shape), tuple(logdet4.shape)), ((0, 3, 3), (0,)))
        with self.assertRaises(ValueError):
            fs.stout_links(U.to(torch.complex64), C.to(torch.complex64))

    def test_coefficients(self):
        """f, b1, b2, d11, d10, d00 against smearing_jacobian.exp_coefficients(order=2): c1 below and above
        C1_SERIES, c0 of both signs, c0 = 0, |c0| = c0max (two equal eigenvalues) and within 1e-12 of it."""
        g = torch.Generator().manual_seed(7)
        c1 = torch.cat((3 * torch.rand(256, generator=g, dtype=torch.double),
                        3 + 12 * torch.rand(256, generator=g, dtype=torch.double),
                        torch.tensor([0.0, 1e-8, 2.999, 3.0, 3.001], dtype=torch.double)))
        c0max = 2 * (c1 / 3) ** 1.5
        c0 = c0max * (2 * torch.rand(c1.shape, generator=g, dtype=torch.double) - 1)
        c0 = torch.cat((c0, 0 * c0max, c0max, -c0max, c0max * (1 - 1e-12), -c0max * (1 - 1e-12)))
        c1 = c1.repeat(6)
        for device in DEVICES:
            got = fs.exp_coefficients(c0.to(device), c1.to(device))
            want = sj.exp_coefficients(c0.to(device), c1.to(device), order=2)
            for name, x, y in zip(("f", "b1", "b2", "d11", "d10", "d00"), got, want):
                for part, sel in (("series", c1 < sj.C1_SERIES), ("closed forms", c1 >= sj.C1_SERIES)):
                    with self.subTest(device=device, coefficient=name, part=part):
                        sel = sel.to(device)
                        dev = ((x[sel] - y[sel]).abs().max() / y[sel].abs().max()).item()
                        _report(f"coefficients {name} ({part}, {device}): max deviation {dev:.1e} relative")
                        self.assertLessEqual(dev, TOL_COEFF)

    def test_inverse_step(self):
        """stout_links_inverse against inverse_smearing_step(fixed_iterations=n) (U, log|det|, residual per
        link), converged or not, and the inverse of the forward step: the identity once converged (in 10
        iterations at rho = 1e-3 on Haar-random links, the production default; 40 at rho = 0.03, 0.1)."""
        for device in DEVICES:
            g = torch.Generator().manual_seed(5)
            for rho in (1e-3, 0.03, 0.1):
                U, C = _random_su3(64, g, device), _staple_sum(64, rho, g, device)
                Up, logdet = sj.smearing_step(U, C)
                for n in (10, 40):
                    with self.subTest(device=device, rho=rho, iterations=n):
                        U0, ld0, iterations, res0 = sj.inverse_smearing_step(Up, C, fixed_iterations=n)
                        res0_links = (sun.SUN_mul(sj.StepQuantities(U0, C).X, U0) - Up).abs().amax((-2, -1))
                        U1, ld1, res1 = fs.stout_links_inverse(Up, C, n)
                        dU = (U1 - U0).abs().max().item()
                        dld = (ld1 - ld0).abs().max().item()
                        dres = (res1 - res0_links).abs().max().item()
                        back, inv = (U1 - U).abs().max().item(), (ld1 + logdet).abs().max().item()
                        _report(f"inverse step, rho {rho:g}, {n} iterations ({device}): max |U fused - torch| {dU:.1e}, "
                                f"log|det| {dld:.1e}, residual {dres:.1e} (torch residual {res0.item():.1e}); "
                                f"inverse(forward) - 1: {back:.1e}, log|det| sum {inv:.1e}")
                        self.assertEqual(iterations, n)
                        self.assertLessEqual(dU, TOL_LINK)
                        self.assertLessEqual(dld, TOL_LINK)
                        self.assertLessEqual(dres, 1e-15)
                        self.assertEqual(res1.amax().item() == 0.0, res0.item() == 0.0)
                        if rho == 1e-3 or n == 40:
                            self.assertLessEqual(res0.item(), 1e-15)
                            self.assertLessEqual(back, 1e-14)
                            self.assertLessEqual(inv, 1e-13)

    def test_inverse_edge_cases(self):
        g = torch.Generator().manual_seed(6)
        U, C = _random_su3(16, g, "cpu"), _staple_sum(16, 1e-2, g, "cpu")
        Up, _ = sj.smearing_step(U, C)
        # no iteration: U = U', and its residual
        U1, ld1, res1 = fs.stout_links_inverse(Up, C, 0)
        self.assertTrue(torch.equal(U1, Up))
        torch.testing.assert_close(ld1, -sj.smearing_step(Up, C)[1], rtol=0, atol=1e-13)
        torch.testing.assert_close(res1, (sun.SUN_mul(sj.StepQuantities(Up, C).X, Up) - Up).abs().amax((-2, -1)),
                                   rtol=0, atol=1e-16)
        # a NaN staple sum gives a NaN residual on its link only, which the deferred check then rejects
        C_nan = C.clone()
        C_nan[3, 1, 2] = float("nan")
        _, _, res = fs.stout_links_inverse(Up, C_nan, 10)
        self.assertTrue(bool(res[3].isnan()) and bool(torch.isfinite(torch.cat((res[:3], res[4:]))).all()))
        with sj.DeferredChecks() as checks:
            fs.inverse_smearing_step(Up, C_nan, 10)
        self.assertFalse(checks.resolve()[1] <= 1e-12)
        # the drop-in of inverse_smearing_step: its outputs, the residual recorded and no c1
        with sj.DeferredChecks() as checks:
            U2, ld2, iterations, residual = fs.inverse_smearing_step(Up, C, 10)
        self.assertEqual(iterations, 10)
        self.assertIsNone(checks.c1_max)
        self.assertEqual(checks.residual_max.item(), residual.item())
        U3, ld3, res3 = fs.stout_links_inverse(Up, C, 10)
        self.assertTrue(torch.equal(U2, U3) and torch.equal(ld2, ld3) and residual.item() == res3.max().item())
        # no links
        U4, ld4, iterations, residual = fs.inverse_smearing_step(Up[:0], C[:0], 10)
        self.assertEqual((tuple(U4.shape), tuple(ld4.shape), residual.item()), ((0, 3, 3), (0,), 0.0))
        with self.assertRaises(ValueError):
            fs.stout_links_inverse(Up, C, -1)


class BackendChoiceTest(unittest.TestCase):
    """fused_smearing.backend_for (one warning per reason, fused_strict raises) and needs_autograd."""

    @classmethod
    def setUpClass(cls):
        _cpu_backend_or_skip()

    def test_fallback_warnings(self):
        owner = SimpleNamespace(N=3, use_nn=False, fused_emulate=True, fused_strict=False)
        x = torch.zeros(1, 4, 2, 2, 2, 2, 3, 3, dtype=torch.complex64)
        for what, change, key in (("complex64", {}, "complex128"), ("SU(2)", {"N": 2}, "SU(3)"),
                                  ("use_nn", {"use_nn": True}, "use_nn")):
            with self.subTest(what):
                o = SimpleNamespace(**{**vars(owner), **change})
                cfgs = x if what == "complex64" else x.to(torch.complex128)
                fused._WARNED.clear()
                with warnings.catch_warnings(record=True) as caught, torch.no_grad():
                    warnings.simplefilter("always")
                    for _ in range(2):
                        self.assertIsNone(fs.backend_for(o, cfgs))
                messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
                self.assertEqual(len(messages), 1)
                self.assertIn(key, messages[0])
                o.fused_strict = True
                with self.assertRaises(fused.BackendUnavailable), torch.no_grad():
                    fs.backend_for(o, cfgs)
        # grad mode alone changes nothing: needs_autograd decides per step, on its inputs
        self.assertTrue(torch.is_grad_enabled())
        self.assertIs(fs.backend_for(owner, x.to(torch.complex128)), fs.load_backend("cpu"))
        U = torch.zeros(2, 3, 3, dtype=torch.complex128)
        self.assertFalse(fs.needs_autograd(U, U))
        self.assertTrue(fs.needs_autograd(U, U.clone().requires_grad_()))
        with torch.no_grad():
            self.assertFalse(fs.needs_autograd(U, U.clone().requires_grad_()))


def _perturb(module, scale, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=g, dtype=parameter.dtype))


def _neorex_layer(rho_init, perturbation, steps=1, seed=0, device="cpu"):
    # The defect smearing layer of NEO-REX as main_pt_obc.py builds it (--nf-layer-type smearing, hyper
    # smearing in class mode), with `steps` smearing steps; its hypernetwork perturbed so that rho depends on
    # the coupling and differs between the classes
    flags = ("--cpu --algorithm neorex --D 4 --T 4 --L 6 --N 3 --beta 6 --defect-size 1 --time-slice 2 "
             "--space-slice 2 --parameter-min 0 --parameter-max 1 --batch-size 2 --nf-layer-type smearing "
             f"--use-hyper-smearing --hyper-smearing-mode class --hyper-rho-init {rho_init} "
             "--hyper-no-normalize-by-nstep").split()
    with mock.patch.object(sys, "argv", ["main_pt_obc", *flags]):
        args = PT_OBC.parse_args()
    defect = Defect(args.D, args.T, args.L, args.defect_size, args.time_slice, args.space_slice)
    with mock.patch.object(PT_OBC, "FlowPars", functools.partial(FlowPars, smearing_steps_per_layer=steps)):
        pars = PT_OBC._build_neorex_flow_pars(args, torch.device(device), create_mask(args.D, args.T, args.L).to(device),
                                              defect)
    torch.manual_seed(seed)
    layer = PT_OBC.DefectSNFProtocolLayer(pars).to(device)
    _perturb(layer, perturbation, seed)
    return layer


def _smooth_links(shape, spread, seed):
    # SU(3) links exp(iH), H traceless hermitian of size ~spread: closer to a thermalized lattice than Haar
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(*shape, 3, 3, dtype=torch.cdouble, generator=g)
    H = 0.5 * (A + A.conj().transpose(-1, -2))
    H = H - torch.diagonal(H, dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(3, dtype=torch.cdouble)
    return torch.linalg.matrix_exp(1j * spread * H)


def _fused(smearing, on=True):
    smearing.use_fused_kernels = on
    smearing.fused_emulate = on
    smearing.fused_strict = on


class LayerTest(unittest.TestCase):
    """Whole smearing layers with Smearing.use_fused_kernels on and off (instance attributes here; the program
    sets the class attribute)."""

    @classmethod
    def setUpClass(cls):
        _cpu_backend_or_skip()

    def setUp(self):
        self.bc = torch.tensor([0.2, 0.7], dtype=torch.double)
        self.delta = torch.tensor([0.5, -0.5], dtype=torch.double)

    def _rho_range(self, layer):
        smearing = layer.layer.smearing
        rho = smearing.hyper_smearing(beta=self.bc, delta_beta=self.delta)
        active = rho[rho != 0].abs()
        return active.min().item(), active.max().item()

    def _compare_layer(self, layer, x, name, inverse_iterations=10):
        smearing = layer.layer.smearing
        with torch.no_grad():
            _fused(smearing, False)
            y0, ld0 = layer(x, self.bc, self.delta)
            xi0, ldi0 = layer.inverse(y0, self.bc, self.delta, fixed_iterations=inverse_iterations)
            self.assertIsNone(smearing.fused_used)
            _fused(smearing, True)
            y1, ld1 = layer(x, self.bc, self.delta)
            self.assertEqual(smearing.fused_used, "cpu")
            xi1, ldi1 = layer.inverse(y0, self.bc, self.delta, fixed_iterations=inverse_iterations)
            self.assertEqual(smearing.fused_used, "cpu")
            stats = smearing.last_inverse_stats
            _fused(smearing, False)
        self.assertEqual(set(stats["iterations"]), {inverse_iterations})
        self.assertGreater((y0 - x).abs().max().item(), 1e-5)
        dy, dld = (y1 - y0).abs().max().item(), (ld1 - ld0).abs().max().item()
        dxi, dldi = (xi1 - xi0).abs().max().item(), (ldi1 - ldi0).abs().max().item()
        back = (xi1 - x).abs().max().item()
        lo, hi = self._rho_range(layer)
        # the largest c1 of the torch forward (above C1_SERIES its first pass is repeated with the closed forms)
        with torch.no_grad(), sj.DeferredChecks() as checks:
            layer._run_blocks("forward", layer.flow_pars.defect.cut_defect(x, buffer=layer.flow_pars.patch_buffer),
                              self.bc, self.delta)
        _report(f"{name}, |rho| in [{lo:.1e}, {hi:.1e}], max c1 {checks.c1_max.item():.2g}, log J "
                f"[{', '.join(f'{v:.3g}' for v in ld0.tolist())}]: forward map {dy:.1e}, log J {dld:.1e}; "
                f"inverse map {dxi:.1e}, log J {dldi:.1e}; inverse(forward) - 1 {back:.1e}")
        for what, d in (("forward map", dy), ("forward log J", dld), ("inverse map", dxi), ("inverse log J", dldi)):
            self.assertLessEqual(d, TOL_LAYER, f"{name}: {what} deviates by {d:.2e}")
        self.assertLessEqual(back, 1e-11)
        return y0

    def test_neorex_defect_layer(self):
        cases = (
            ("rho ~1e-3, 1 step", 1e-3, 1e-4, 1, 10),
            ("rho ~0.02, 1 step", 0.02, 0.005, 1, 10),
            ("rho ~0.02, 2 steps", 0.02, 0.005, 2, 10),
            ("rho ~0.1, 1 step", 0.1, 0.02, 1, 30),
            ("rho ~0.3, 1 step", 0.3, 0.05, 1, 80),
        )
        for name, rho_init, perturbation, steps, iterations in cases:
            for config in ("haar", "smooth"):
                with self.subTest(case=name, config=config):
                    layer = _neorex_layer(rho_init, perturbation, steps=steps, seed=1)
                    self.assertEqual(layer.layer.smearing.hyper_smearing.mode, "class")
                    if config == "haar":
                        x = sun.SUN_haar((2, 4, 4, 6, 6, 6, 3), dtype=torch.cdouble,
                                         generator=torch.Generator().manual_seed(2))
                    else:
                        x = _smooth_links((2, 4, 4, 6, 6, 6), 0.3, seed=2)
                    self._compare_layer(layer, x, f"NEO-REX class layer, {name}, {config}", iterations)

    def test_neorex_defect_layer_heavy_smearing_forward(self):
        """|rho| ~ 3: c1 >= C1_SERIES on part of the links, where the torch forward repeats its call with the
        closed forms and the kernel takes them link by link (the inverse would need more iterations)."""
        layer = _neorex_layer(6.0, 0.3, seed=1)
        smearing = layer.layer.smearing
        x = sun.SUN_haar((2, 4, 4, 6, 6, 6, 3), dtype=torch.cdouble, generator=torch.Generator().manual_seed(2))
        small = layer.flow_pars.defect.cut_defect(x, buffer=layer.flow_pars.patch_buffer)
        with torch.no_grad():
            with sj.DeferredChecks() as checks:
                layer._run_blocks("forward", small, self.bc, self.delta)
            y0, ld0 = layer(x, self.bc, self.delta)
            _fused(smearing)
            y1, ld1 = layer(x, self.bc, self.delta)
            self.assertEqual(smearing.fused_used, "cpu")
            _fused(smearing, False)
        c1_max = checks.c1_max.item()
        dy, dld = (y1 - y0).abs().max().item(), ((ld1 - ld0).abs() / ld0.abs()).max().item()
        lo, hi = self._rho_range(layer)
        _report(f"NEO-REX class layer, heavy smearing, |rho| in [{lo:.2g}, {hi:.2g}], max c1 {c1_max:.2g}: forward map "
                f"{dy:.1e}, log J {dld:.1e} relative")
        self.assertGreater(c1_max, sj.C1_SERIES)
        self.assertLessEqual(dy, TOL_LAYER)
        self.assertLessEqual(dld, 1e-13)

    def test_whole_lattice_hyper_smearing(self):
        from test.test_cnf_adapters import _flow_pars
        pars = _flow_pars("hyper-smearing")
        pars.mask = create_mask(2, 4, 4)
        layer = CouplingLayer(pars, torch.zeros(pars.rho_shape, dtype=torch.double), time=1.0)
        _perturb(layer.smearing.hyper_smearing, 0.05, seed=3)
        x = sun.SUN_haar((2, 2, 4, 4, 3), dtype=torch.cdouble, generator=torch.Generator().manual_seed(3))
        beta = torch.tensor([0.3, 0.8], dtype=torch.double)
        smearing = layer.smearing
        with torch.no_grad():
            y0, _, ld0 = layer(x, pars, None, beta=beta)
            xi0, _, ldi0 = layer.inverse(y0, pars, None, beta=beta)
            _fused(smearing)
            y1, _, ld1 = layer(x, pars, None, beta=beta)
            xi1, _, ldi1 = layer.inverse(y0, pars, None, beta=beta)
            self.assertEqual(smearing.fused_used, "cpu")
        deviations = [(a - b).abs().max().item() for a, b in ((y1, y0), (ld1, ld0), (xi1, xi0), (ldi1, ldi0))]
        _report("whole-lattice hyper smearing D=2: forward map, log J, inverse map, log J: "
                + ", ".join(f"{d:.1e}" for d in deviations))
        self.assertLessEqual(max(deviations), TOL_LAYER)

    def test_torch_paths(self):
        """Flag off: torch, fused_used untouched. Grad mode with rho requiring grad (the hypernetwork being
        trained), the CPU without the emulation and the adaptive inverse: torch, the same numbers as with the
        flag off. Grad mode with nothing requiring grad: the kernels. A failed residual check repeats the inverse
        with the adaptive torch iteration."""
        layer = _neorex_layer(0.05, 0.02, seed=4)
        smearing = layer.layer.smearing
        x = _smooth_links((2, 4, 4, 6, 6, 6), 0.3, seed=4)
        self.assertFalse(Smearing.use_fused_kernels)
        with torch.no_grad():
            y0, ld0 = layer(x, self.bc, self.delta)
            xa0, lda0 = layer.inverse(y0, self.bc, self.delta, fixed_iterations=0)
        self.assertIsNone(smearing.fused_used)
        # grad mode, rho from the trainable hypernetwork: autograd records the torch step
        _fused(smearing)
        y1, ld1 = layer(x, self.bc, self.delta)
        self.assertEqual(smearing.fused_used, "torch")
        self.assertTrue(ld1.requires_grad)
        self.assertTrue(torch.equal(y1.detach(), y0) and torch.equal(ld1.detach(), ld0))
        x1, ldi1 = layer.inverse(y0, self.bc, self.delta)
        self.assertEqual(smearing.fused_used, "torch")
        self.assertTrue(ldi1.requires_grad)
        # grad mode with no input requiring grad (main_pt_obc.py runs much without torch.no_grad()): the
        # kernels, the numbers of the kernels under no_grad
        with torch.no_grad():
            y_k, ld_k = layer(x, self.bc, self.delta)
            x_k, ldi_k = layer.inverse(y0, self.bc, self.delta)
        self.assertEqual(smearing.fused_used, "cpu")
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        self.assertTrue(torch.is_grad_enabled())
        y2, ld2 = layer(x, self.bc, self.delta)
        self.assertEqual(smearing.fused_used, "cpu")
        x2, ldi2 = layer.inverse(y0, self.bc, self.delta)
        self.assertEqual(smearing.fused_used, "cpu")
        self.assertFalse(ld2.requires_grad or ldi2.requires_grad)
        self.assertTrue(torch.equal(y2, y_k) and torch.equal(ld2, ld_k) and torch.equal(x2, x_k) and torch.equal(ldi2, ldi_k))
        # ... and torch again when the configurations require grad
        y3, ld3 = layer(x.clone().requires_grad_(), self.bc, self.delta)
        self.assertEqual(smearing.fused_used, "torch")
        self.assertTrue(ld3.requires_grad)
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
        with torch.no_grad():
            # the adaptive inverse
            xa1, lda1 = layer.inverse(y0, self.bc, self.delta, fixed_iterations=0)
            self.assertEqual(smearing.fused_used, "torch")
            self.assertTrue(torch.equal(xa1, xa0) and torch.equal(lda1, lda0))
            # a failed check of the fixed-iteration inverse with the kernels: the adaptive torch inverse
            with mock.patch.object(sj.DeferredChecks, "resolve", return_value=(True, 1.0)):
                xf, ldf = layer.inverse(y0, self.bc, self.delta)
            self.assertEqual(smearing.fused_used, "torch")
            self.assertTrue(torch.equal(xf, xa0) and torch.equal(ldf, lda0))
            # the CPU without the emulation: torch, silently
            smearing.fused_emulate = False
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                y2, ld2 = layer(x, self.bc, self.delta)
            self.assertEqual(smearing.fused_used, "torch")
            self.assertTrue(torch.equal(y2, y0) and torch.equal(ld2, ld0))
        _fused(smearing, False)


@unittest.skipUnless(CUDA, "needs a GPU")
class CudaGraph(unittest.TestCase):
    """main_pt_obc --cuda-graphs --fused-kernels: the defect smearing layer captured with the kernel launches
    inside, replayed against the eager torch layer; the times of the eager torch, eager fused and captured fused
    forward and inverse are printed."""

    @staticmethod
    def _timed(fn, repeats=10):
        import time
        fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / repeats * 1e3

    def test_captured_layer_with_the_kernels_matches_eager_torch(self):
        from neoqcd.cnf_io import build_cnf
        from test.test_cnf_adapters import _defect_config
        ok, why = fs.backend_status("cuda")
        if not ok:
            self.skipTest(f"no CUDA build of the smearing kernels: {why}")
        device = torch.device("cuda")
        _, pars, _ = build_cnf(_defect_config(batch_size=16, D=4, T=8, L=8, defect_size=2, time_slice=4,
                                              space_slice=3, hyper_smearing_mode="class"), device="cuda")
        torch.manual_seed(1)
        eager, graphed = PT_OBC.DefectSNFProtocolLayer(pars).to(device), PT_OBC.DefectSNFProtocolLayer(pars).to(device)
        fused_eager = PT_OBC.DefectSNFProtocolLayer(pars).to(device)
        with torch.no_grad():
            for parameter in eager.layer.smearing.hyper_smearing.mlp.parameters():
                parameter.add_(0.02 * torch.randn_like(parameter))
        graphed.load_state_dict(eager.state_dict())
        fused_eager.load_state_dict(eager.state_dict())
        graphed.use_cuda_graphs = True
        for layer in (graphed, fused_eager):
            layer.layer.smearing.use_fused_kernels = True
            layer.layer.smearing.fused_strict = True
        shape = (16, 4, 8, 8, 8, 8)
        x = _smooth_links(shape, 0.3, seed=1).to(device)
        bc = torch.linspace(0.1, 0.8, 16, dtype=torch.float64, device=device)
        delta = torch.full_like(bc, 1 / 28)
        with torch.no_grad():
            for trial in range(2):  # the second replay must see the new inputs
                if trial:
                    x = _smooth_links(shape, 0.3, seed=2).to(device)
                    bc = bc.flip(0)
                y_e, ld_e = eager(x, bc, delta)
                y_g, ld_g = graphed(x, bc, delta)
                self.assertTrue(graphed.use_cuda_graphs, "capture failed and fell back to eager")
                self.assertEqual(graphed.layer.smearing.fused_used, "cuda")
                self.assertLess((y_g - y_e).abs().max().item(), TOL_LAYER)
                self.assertLess((ld_g - ld_e).abs().max().item(), 1e-11)
                xi_e, ldi_e = eager.inverse(y_e, bc, delta)
                xi_g, ldi_g = graphed.inverse(y_e, bc, delta)
                self.assertTrue(graphed.use_cuda_graphs, "capture failed and fell back to eager")
                self.assertEqual(graphed.layer.smearing.fused_used, "cuda")
                self.assertLess((xi_g - xi_e).abs().max().item(), TOL_LAYER)
                self.assertLess((ldi_g - ldi_e).abs().max().item(), 1e-11)
                self.assertLess((xi_g - x).abs().max().item(), 1e-10)
            self.assertEqual(len(graphed._graphs), 2)
            times = {name: (self._timed(lambda: layer(x, bc, delta)), self._timed(lambda: layer.inverse(y_e, bc, delta)))
                     for name, layer in (("torch eager", eager), ("fused eager", fused_eager), ("fused graph", graphed))}
            self.assertEqual(fused_eager.layer.smearing.fused_used, "cuda")
        _report("defect layer 8^4, batch 16, forward / inverse ms: "
                + ", ".join(f"{name} {f:.2f} / {i:.2f}" for name, (f, i) in times.items()))


# The CUDA runtime as far as the kernel sources use it, for clang's CUDA front end without a CUDA toolkit
CUDA_STUB = r"""
#pragma once
#include <stddef.h>
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#include "__clang_cuda_builtin_vars.h"
typedef struct CUstream_st* cudaStream_t;
typedef int cudaError_t;
struct cudaFuncAttributes { int maxThreadsPerBlock, numRegs; };
struct dim3 { unsigned x, y, z;
  __host__ __device__ dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {} };
extern "C" int cudaConfigureCall(dim3 g, dim3 b, size_t s = 0, cudaStream_t st = 0);
extern "C" cudaError_t cudaGetDevice(int*);
extern "C" cudaError_t cudaSetDevice(int);
extern "C" cudaError_t cudaGetLastError(void);
extern "C" cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, const void*);
extern "C" const char* cudaGetErrorString(cudaError_t);
#include <math.h>
__device__ double log(double);
__device__ double cos(double);
__device__ double sin(double);
__device__ double sqrt(double);
__device__ double acos(double);
__device__ double pow(double, double);
__device__ double fabs(double);
__device__ double hypot(double, double);
"""


class CudaSource(unittest.TestCase):
    def test_clang_cuda_front_end(self):
        """The host and device halves of the CUDA builds of fused.py and fused_smearing.py parse and
        type-check, every template instantiated (clang -x cuda with a stub runtime header): catches errors
        before the first GPU job."""
        clang = shutil.which("clang++")
        if clang is None:
            self.skipTest("no clang++")
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "stub"
            stub.mkdir()
            (stub / "cuda_runtime.h").write_text(CUDA_STUB)
            src = Path(tmp) / "k.cu"
            src.write_text("__attribute__((global)) void k(int* x) { x[0] = 1; }\n")
            base = [clang, "-x", "cuda", "--cuda-gpu-arch=sm_80", "-nocudainc", "-nocudalib", "-fsyntax-only"]
            res = subprocess.run(base + ["--cuda-host-only", str(src)], capture_output=True, text=True)
            if res.returncode != 0:
                self.skipTest(f"clang++ has no CUDA front end: {res.stderr.strip()[:200]}")
            for name, source in (("fused", fused.KERNEL_SOURCE), ("fused_smearing", fs.KERNEL_SOURCE)):
                src.write_text(source)
                for side in ("--cuda-host-only", "--cuda-device-only"):
                    with self.subTest(source=name, side=side):
                        res = subprocess.run(base + [side, "-D__CUDACC__", f"-I{stub}", "-std=c++14", "-Wall", str(src)],
                                             capture_output=True, text=True, timeout=300)
                        self.assertEqual(res.returncode, 0, f"{side}: {res.stderr}")
                        self.assertNotIn("warning", res.stderr, f"{side}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
