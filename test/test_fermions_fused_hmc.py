"""Tests of neoqcd.fermions.fused_hmc (fused molecular-dynamics kernels) against the torch code.

Targets: "cpu" (the kernel source compiled for the host and run as a loop over the CUDA thread
index: same index arithmetic; skipped without a C++ compiler) and "cuda" (the twins, skipped without
a CUDA device or nvcc; they run in job/debug_bench_fermions.sh). Equality is to rounding (the
kernels sum in another order), on Haar-random links with batch > 1 and unequal extents.
"""

import copy
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neoqcd.fermions import fused as F  # noqa: E402
from neoqcd.fermions import fused_hmc as M  # noqa: E402
from neoqcd.fermions._gauge import SUN_haar  # noqa: E402
from neoqcd.fermions.algebra import random_algebra  # noqa: E402
from neoqcd.fermions.hmc import WilsonGaugeAction, batched_defect_plaquette_weights, defect_plaquette_weights  # noqa: E402
from neoqcd.fermions.lattice import Geometry  # noqa: E402

G46 = Geometry(4, 6, 4, 3)  # 4^3 x 6, SU(3)
G42 = Geometry(4, 4, 2, 2)  # 2^3 x 4, SU(2)
G2 = Geometry(2, 6, 4, 3)  # 2D 6 x 4 (T = 6, L = 4), SU(3)
G22 = Geometry(2, 4, 6, 2)  # 2D 4 x 6, SU(2)
GEOMETRIES = (G46, G42, G2, G22)
C, C64 = torch.complex128, torch.complex64

_SAVED_CACHE = None
_TMP = None


def setUpModule():
    """Build the kernel libraries in a temporary cache unless NEOQCD_KERNEL_CACHE is set."""
    global _SAVED_CACHE, _TMP
    _SAVED_CACHE = os.environ.get("NEOQCD_KERNEL_CACHE")
    if _SAVED_CACHE is None:
        _TMP = tempfile.TemporaryDirectory(prefix="neoqcd_kernels_")
        os.environ["NEOQCD_KERNEL_CACHE"] = _TMP.name


def tearDownModule():
    if _SAVED_CACHE is None:
        os.environ.pop("NEOQCD_KERNEL_CACHE", None)
        _TMP.cleanup()


def links(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=g)


def momenta(geometry, batch, seed):
    return random_algebra(geometry.link_shape(batch), generator=torch.Generator().manual_seed(seed))


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def defect_mask(geometry, seed):
    """A random gauge-only defect: about a quarter of the links get the factor bc."""
    g = torch.Generator().manual_seed(seed)
    on = (torch.rand((geometry.D,) + tuple(geometry.lattice), generator=g) < 0.25).to(torch.float64)
    return torch.stack([1.0 - on, on])


def md_reason(target):
    if target == "cuda" and not torch.cuda.is_available():
        return "no CUDA device"
    ok, why = M.md_backend_status(target)
    return None if ok else why


class _Target:
    target = "cpu"
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        reason = md_reason(cls.target)
        if reason is not None:
            raise unittest.SkipTest(f"{cls.target} MD kernels unavailable: {reason}")

    def dev(self, t):
        return t.to(self.device)


class _GaugeForceEquality(_Target):
    def test_force_equals_torch(self):
        """Force and kick vs WilsonGaugeAction: SU(2) and SU(3), 4D and 2D, unequal extents, batch 3,
        scalar and per-sample beta; relative 1e-14."""
        for geo in GEOMETRIES:
            U = self.dev(links(geo, 3, 1))
            P = self.dev(momenta(geo, 3, 2))
            for beta in (5.6, torch.tensor([5.6, 5.0, 6.1], dtype=torch.float64)):
                with self.subTest(geo=geo, beta=type(beta).__name__):
                    ref = WilsonGaugeAction(beta, geo).force(U)
                    act = M.FusedWilsonGaugeAction(beta, geo, target=self.target, strict=True)
                    got = act.force(U)
                    self.assertEqual(act.kernel_used, self.target)
                    self.assertEqual(got.dtype, C)
                    self.assertLess(rel(got, ref), 1e-14)
                    kick = act.force_kick(U, P, 0.37)
                    self.assertLess((kick - (P - 0.37 * ref)).abs().max().item(), 1e-14)

    def test_defect_weights(self):
        """Plaquette weights of a gauge-only defect: per-sample (B, lattice) fields, lattice fields,
        and a mixture of both layouts; the weights change the force (non-trivial test)."""
        for geo in GEOMETRIES:
            U = self.dev(links(geo, 3, 3))
            mask = self.dev(defect_mask(geo, 4))
            bcs = torch.tensor([0.0, 0.4, 0.9], dtype=torch.float64)
            layouts = {
                "batched": batched_defect_plaquette_weights(mask, self.dev(bcs), geo.D),
                "lattice": defect_plaquette_weights(mask, 0.3, geo.D),
            }
            mixed = dict(layouts["lattice"])
            first = min(mixed)
            mixed[first] = layouts["batched"][first]
            layouts["mixed"] = mixed
            plain = WilsonGaugeAction(5.6, geo).force(U)
            for name, w in layouts.items():
                with self.subTest(geo=geo, layout=name):
                    ref = WilsonGaugeAction(5.6, geo, w).force(U)
                    self.assertGreater(rel(ref, plain), 1e-3)
                    act = M.FusedWilsonGaugeAction(5.6, geo, w, target=self.target, strict=True)
                    self.assertLess(rel(act.force(U), ref), 1e-14)
                    P = self.dev(momenta(geo, 3, 5))
                    self.assertLess((act.force_kick(U, P, -0.2) - (P + 0.2 * ref)).abs().max().item(), 1e-14)

    def test_weights_cache_follows_the_weights(self):
        """The packed weights follow an in-place change of a weight field and a new weights dict."""
        geo = G46
        U = self.dev(links(geo, 2, 6))
        mask = self.dev(defect_mask(geo, 7))
        w = batched_defect_plaquette_weights(mask, self.dev(torch.tensor([0.2, 0.5], dtype=torch.float64)), geo.D)
        act = M.FusedWilsonGaugeAction(5.6, geo, w, target=self.target, strict=True)
        act.force(U)
        w[(0, 1)].mul_(0.5)
        self.assertLess(rel(act.force(U), WilsonGaugeAction(5.6, geo, w).force(U)), 1e-14)
        w2 = batched_defect_plaquette_weights(mask, self.dev(torch.tensor([0.9, 0.1], dtype=torch.float64)), geo.D)
        act.plaquette_weights = w2
        self.assertLess(rel(act.force(U), WilsonGaugeAction(5.6, geo, w2).force(U)), 1e-14)

    def test_strided_conj_and_neg_views(self):
        """Non-contiguous, offset, lazily conjugated and negated links and momenta are read as their values."""
        geo = G46
        U = self.dev(links(geo, 2, 8))
        P = self.dev(momenta(geo, 2, 9))
        ref = WilsonGaugeAction(5.6, geo).force(U)
        act = M.FusedWilsonGaugeAction(5.6, geo, target=self.target, strict=True)
        big = torch.zeros((3,) + tuple(U.shape), dtype=C, device=U.device)
        big[1] = U
        views = {
            "offset": big[1],
            "transposed": U.transpose(-1, -2).contiguous().transpose(-1, -2),
            "conj": U.conj().resolve_conj().conj(),
            "neg": torch._neg_view(-U),
            "batch_strided": torch.stack([U[0], U[0], U[1]])[::2][[0, 1]],
        }
        self.assertTrue(views["conj"].is_conj())
        self.assertTrue(views["neg"].is_neg())
        self.assertFalse(views["transposed"].is_contiguous())
        for name, V in views.items():
            with self.subTest(view=name):
                want = ref if name != "batch_strided" else WilsonGaugeAction(5.6, geo).force(V.contiguous())
                self.assertLess(rel(act.force(V), want), 1e-14)
        Pv = P.conj().resolve_conj().conj()  # P hermitian: the same values through a lazy conj view
        self.assertLess((act.force_kick(U, Pv, 0.1) - (P - 0.1 * ref)).abs().max().item(), 1e-14)

    def test_autograd_inputs_refused(self):
        geo = G46
        U = self.dev(links(geo, 2, 10)).requires_grad_()
        act = M.FusedWilsonGaugeAction(5.6, geo, target=self.target, strict=True)
        with self.assertRaisesRegex(RuntimeError, "not differentiable"):
            act.force(U)
        with torch.no_grad():
            self.assertLess(rel(act.force(U), WilsonGaugeAction(5.6, geo).force(U.detach())), 1e-14)


class CpuGaugeForceTest(_GaugeForceEquality, unittest.TestCase):
    target, device = "cpu", "cpu"


class CudaGaugeForceTest(_GaugeForceEquality, unittest.TestCase):
    target, device = "cuda", "cuda"


class FallbackTest(unittest.TestCase):
    def test_torch_target_is_the_parent(self):
        geo = G46
        U = links(geo, 2, 11)
        act = M.FusedWilsonGaugeAction(5.6, geo, target=None)
        self.assertTrue(torch.equal(act.force(U), WilsonGaugeAction(5.6, geo).force(U)))
        self.assertEqual(act.kernel_used, "torch")

    def test_unsupported_inputs_fall_back_with_one_warning_or_raise(self):
        """complex64 links, D = 3 and a CUDA target on CPU tensors run the torch force (one warning per
        reason); strict=True raises BackendUnavailable."""
        U64 = links(G46, 2, 12).to(C64)
        G3 = Geometry(3, 4, 4, 3)
        U3 = links(G3, 2, 13)
        cases = [("cpu", G46, U64), ("cpu", G3, U3), ("cuda", G46, links(G46, 2, 14))]
        if md_reason("cpu") is not None:
            cases = cases[2:]
        for target, geo, U in cases:
            with self.subTest(target=target, geo=geo, dtype=U.dtype):
                F._WARNED.clear()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    act = M.FusedWilsonGaugeAction(5.6, geo, target=target)
                    got = act.force(U)
                    act.force(U)
                self.assertTrue(torch.equal(got, WilsonGaugeAction(5.6, geo).force(U)))
                self.assertEqual(act.kernel_used, "torch")
                self.assertEqual(len([w for w in caught if "fused MD kernel" in str(w.message)]), 1)
                with self.assertRaises(F.BackendUnavailable):
                    M.FusedWilsonGaugeAction(5.6, geo, target=target, strict=True).force(U)

    def test_copy_and_pickle(self):
        if md_reason("cpu") is not None:
            self.skipTest(md_reason("cpu"))
        geo = G46
        U = links(geo, 2, 15)
        act = M.FusedWilsonGaugeAction(5.6, geo, target="cpu", strict=True)
        want = act.force(U)
        for clone in (copy.deepcopy(act), pickle.loads(pickle.dumps(act))):
            self.assertTrue(torch.equal(clone.force(U), want))
            self.assertEqual(clone.kernel_used, "cpu")

    def test_library_name_and_cache_key(self):
        """The MD library is cached apart from the hop library (its own name and source hash)."""
        if md_reason("cpu") is not None:
            self.skipTest(md_reason("cpu"))
        path = Path(M.load_md_backend("cpu").path)
        self.assertTrue(path.name.startswith("neoqcd_md_cpu_"), path.name)
        self.assertNotEqual(F._build_spec("cpu")[3], F._build_spec("cpu", backend_class=M._MDBackend)[3])
        self.assertEqual(F._build_spec("cpu")[3].name.split("_")[:3], ["neoqcd", "hop", "cpu"])


class CudaSourceTest(unittest.TestCase):
    def test_clang_cuda_front_end(self):
        """Host and device halves of the CUDA build of MD_KERNEL_SOURCE parse and type-check (clang -x
        cuda with the stub runtime header of test_fermions_fused), warnings included."""
        from test.test_fermions_fused import _CUDA_STUB
        clang = shutil.which("clang++")
        if clang is None:
            self.skipTest("no clang++")
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.cu"
            probe.write_text("__attribute__((global)) void k(int* x) { x[0] = 1; }\n")
            res = subprocess.run([clang, "-x", "cuda", "--cuda-host-only", "--cuda-gpu-arch=sm_80", "-nocudainc",
                                  "-nocudalib", "-fsyntax-only", str(probe)], capture_output=True, text=True)
            if res.returncode != 0:
                self.skipTest(f"clang++ has no CUDA front end: {res.stderr.strip()[:200]}")
            stub = Path(tmp) / "stub"
            stub.mkdir()
            (stub / "cuda_runtime.h").write_text(_CUDA_STUB + _MATH_STUB)
            src = Path(tmp) / "md.cu"
            src.write_text(M.MD_KERNEL_SOURCE)
            for side in ("--cuda-host-only", "--cuda-device-only"):
                cmd = [clang, "-x", "cuda", side, "--cuda-gpu-arch=sm_80", "-nocudainc", "-nocudalib", "-D__CUDACC__",
                       "-D__CUDACC_VER_MAJOR__=12", "-D__CUDACC_VER_MINOR__=1", f"-I{stub}", "-std=c++14", "-Wall",
                       "-fsyntax-only", str(src)]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                self.assertEqual(res.returncode, 0, f"{side}: {res.stderr}")
                self.assertNotIn("warning", res.stderr, f"{side}: {res.stderr}")


# device declarations of the libm functions the kernels call (cuda_runtime.h provides them under nvcc)
_MATH_STUB = r"""
__attribute__((device)) double sqrt(double); __attribute__((device)) double sin(double);
__attribute__((device)) double cos(double); __attribute__((device)) double acos(double);
__attribute__((device)) double pow(double, double); __attribute__((device)) double fabs(double);
__attribute__((device)) double exp(double); __attribute__((device)) double log(double);
"""


if __name__ == "__main__":
    unittest.main()
