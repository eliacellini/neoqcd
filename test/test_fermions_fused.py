"""Tests of neoqcd.fermions.fused against neoqcd.fermions.fast and the even-odd reference.

Kernels: "fast" (the parent's hop: fallback and "auto" without CUDA), "loop" (pure torch, always), "cpu" (the CUDA kernel source compiled for the host: same
index arithmetic and thread decomposition, run as a loop over the CUDA thread index; skipped
without a C++ compiler), "cuda" (skipped without a CUDA device or nvcc). The CUDA-only parts of
the source (kernel, launch, device handling) are checked for syntax and semantics with clang's
CUDA front end when available (`CudaSourceTest`).
"""

import os
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
from neoqcd.fermions import hmc as H  # noqa: E402
from neoqcd.fermions._gauge import SUN_haar  # noqa: E402
from neoqcd.fermions.dirac import WilsonDirac  # noqa: E402
from neoqcd.fermions.fast import (  # noqa: E402
    FastTwoFlavourHamiltonian,
    FastTwoFlavourWilson,
    HalfLatticeWilson,
    lean_cg,
    to_half,
)
from neoqcd.fermions.fused import (  # noqa: E402
    BackendUnavailable,
    FusedHalfLatticeWilson,
    FusedTwoFlavourHamiltonian,
    FusedTwoFlavourWilson,
    backend_status,
)
from neoqcd.fermions.lattice import Geometry  # noqa: E402

G4s = Geometry(4, 4, 2, 3)  # 2^3 x 4
G44 = Geometry(4, 4, 4, 3)  # 4^4
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4
G46 = Geometry(4, 6, 4, 3)  # 4^3 x 6
GEOMETRIES = (G4s, G44, G2, G46)
C, C64 = torch.complex128, torch.complex64
TOL = {C: 1e-14, C64: 1e-6}

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


def field(geometry, batch, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(geometry.wilson_shape(batch), dtype=C, generator=g)


def rel(a, b):
    return ((a.to(b.dtype) - b).abs().max() / b.abs().max()).item()


def kappas(*values):
    return torch.tensor(values, dtype=torch.float64)


def cuda_reason():
    if not torch.cuda.is_available():
        return "no CUDA device"
    ok, why = backend_status("cuda")
    return None if ok else why


def cpu_reason():
    ok, why = backend_status("cpu")
    return None if ok else why


class _KernelEquality:
    """Shared assertions: the fused operator equals HalfLatticeWilson (and the even-odd reference)."""

    kernel = None
    device = "cpu"
    link_layout = "site"

    def make(self, U, k, geo, ts, **kw):
        op = FusedHalfLatticeWilson(U, k, geo, time_sign=ts, kernel=self.kernel, strict=True,
                                    link_layout=self.link_layout, **kw)
        self.assertEqual(op.kernel, self.kernel)
        return op

    def test_hops_operator_and_normal(self):
        """hop (both target parities, dagger), apply, apply_dagger, normal, prepare_rhs, reconstruct:
        1e-14 vs HalfLatticeWilson (complex128), 1e-6 (complex64); per-batch kappa, both time signs."""
        dev = torch.device(self.device)
        k = kappas(0.11, 0.13)
        for geo in GEOMETRIES:
            U = links(geo, 2, 1).to(dev)
            psi = field(geo, 2, 2).to(dev)
            e, o = to_half(psi, geo, 0), to_half(psi, geo, 1)
            for ts in (-1.0, 1.0):
                ref = HalfLatticeWilson(U, k, geo, time_sign=ts)
                op = self.make(U, k, geo, ts)
                for dtype in (C, C64):
                    r, f = ref.to(dtype), op.to(dtype)
                    self.assertEqual(f.kernel, self.kernel)
                    xe, xo = e.to(dtype), o.to(dtype)
                    cases = [(f"hop t={t} dagger={d}", (lambda m, t=t, d=d: m.hop(xo if t == 0 else xe, t, d)))
                             for t in (0, 1) for d in (False, True)]
                    cases += [(name, (lambda m, name=name: getattr(m, name)(xe)))
                              for name in ("apply", "apply_dagger", "normal")]
                    cases += [("prepare_rhs_half", lambda m: m.prepare_rhs_half(xe, xo)),
                              ("reconstruct_odd", lambda m: m.reconstruct_odd(xe, xo))]
                    for name, fn in cases:
                        with self.subTest(lattice=geo.lattice, time_sign=ts, dtype=dtype, case=name):
                            want = fn(r)
                            got = fn(f)
                            self.assertEqual(got.dtype, dtype)
                            self.assertEqual(tuple(got.shape), tuple(want.shape))
                            self.assertLess(rel(got, want), TOL[dtype])
                # and against the full-lattice even-odd reference (complex128)
                eo = WilsonDirac(U, k, geo, time_sign=ts).even_odd()
                for name in ("apply", "apply_dagger", "normal"):
                    with self.subTest(lattice=geo.lattice, time_sign=ts, reference=name):
                        self.assertLess(rel(getattr(op, name)(e), to_half(getattr(eo, name)(psi), geo)), 1e-13)
                self.assertLess(rel(op.reconstruct(e, psi), ref.reconstruct(e, psi)), 1e-14)
                self.assertLess(rel(op.prepare_rhs(psi), ref.prepare_rhs(psi)), 1e-14)

    def test_shared_links_and_scalar_kappa(self):
        """Links with batch 1 act on a batch-3 field (link batch stride 0); scalar kappa."""
        dev = torch.device(self.device)
        geo = G46
        U = links(geo, 1, 3).to(dev)
        x = to_half(field(geo, 3, 4), geo).to(dev)
        ref = HalfLatticeWilson(U, 0.125, geo)
        op = self.make(U, 0.125, geo, -1.0)
        for name in ("normal", "apply_dagger"):
            self.assertLess(rel(getattr(op, name)(x), getattr(ref, name)(x)), 1e-14)
        self.assertLess(rel(op.hop(x, 1, True), ref.hop(x, 1, True)), 1e-14)

    def test_su2(self):
        """N = 2 instantiations (4D 2^3 x 4 and 2D 4 x 4), normal and daggered hops."""
        dev = torch.device(self.device)
        for geo in (Geometry(4, 4, 2, 2), Geometry(2, 4, 4, 2)):
            with self.subTest(lattice=geo.lattice):
                U = links(geo, 2, 13).to(dev)
                x = to_half(field(geo, 2, 14), geo).to(dev)
                k = kappas(0.12, 0.14)
                ref = HalfLatticeWilson(U, k, geo)
                op = self.make(U, k, geo, -1.0)
                self.assertLess(rel(op.normal(x), ref.normal(x)), 1e-14)
                self.assertLess(rel(op.hop(x, 1, True), ref.hop(x, 1, True)), 1e-14)

    def test_lean_cg_and_conjugate_input(self):
        dev = torch.device(self.device)
        geo = G44
        U = links(geo, 2, 5).to(dev)
        k = kappas(0.12, 0.14)
        b = to_half(field(geo, 2, 6), geo).to(dev)
        ref = HalfLatticeWilson(U, k, geo)
        op = self.make(U, k, geo, -1.0)
        x_ref = lean_cg(ref.normal, b, tol=1e-12, max_iter=500).x
        res = lean_cg(op.normal, b, tol=1e-12, max_iter=500)
        self.assertLess(res.residual.max().item(), 1e-11)
        self.assertLess(rel(res.x, x_ref), 1e-10)
        # a lazily conjugated field is resolved before the kernel reads its memory
        bc = b.conj()
        self.assertLess(rel(op.normal(bc), ref.normal(bc.resolve_conj())), 1e-14)

    @staticmethod
    def _views(x):
        """(name, view): the values of x as a non-contiguous tensor, at a storage offset, with a
        lazy conjugate bit, a negative bit or both (memory != values), and an expanded (stride 0)
        batch (the values of sample 0 for every sample). The kernel reads raw memory."""
        B = x.shape[0]
        buf = torch.zeros((B + 1,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        buf[1:] = x
        out = [("strided", x.transpose(2, 3).contiguous().transpose(2, 3)),
               ("offset", buf[1:]),
               ("conj", x.conj().resolve_conj().conj()),  # memory conj(x), conj bit set
               ("neg", torch._neg_view(-x)),  # memory -x, neg bit set
               ("neg_conj", torch._neg_view((-x).conj().resolve_conj()).conj())]  # memory -conj(x)
        if B > 1:
            out.append(("expanded", x[:1].expand_as(x)))
        return out

    def test_strided_offset_expanded_and_lazy_inputs(self):
        """Fields (and the fused `add`) given as non-contiguous, offset, expanded, conj- or
        neg-bit views give the result of their materialised values (1e-14)."""
        dev = torch.device(self.device)
        geo = G46
        k = kappas(0.11, 0.13)
        U = links(geo, 2, 50).to(dev)
        psi = field(geo, 2, 51).to(dev)
        e, o = to_half(psi, geo, 0), to_half(psi, geo, 1)
        ref = HalfLatticeWilson(U, k, geo)
        op = self.make(U, k, geo, -1.0)
        for dtype in (C, C64):
            r, f = ref.to(dtype), op.to(dtype)
            for (name, ve), (_, vo) in zip(self._views(e.to(dtype)), self._views(o.to(dtype))):
                plain_e = ve.resolve_conj().resolve_neg().clone()
                plain_o = vo.resolve_conj().resolve_neg().clone()
                cases = [("hop0", lambda m, a, b: m.hop(b, 0)), ("hop1_dag", lambda m, a, b: m.hop(a, 1, True)),
                         ("apply", lambda m, a, b: m.apply(a)), ("normal", lambda m, a, b: m.normal(a)),
                         ("prepare_rhs_half", lambda m, a, b: m.prepare_rhs_half(a, b)),
                         ("reconstruct_odd", lambda m, a, b: m.reconstruct_odd(a, b))]
                for case, fn in cases:
                    with self.subTest(dtype=dtype, view=name, case=case):
                        want = fn(r, plain_e, plain_o)
                        self.assertLess(rel(fn(f, ve, vo), want), TOL[dtype])
                        # the view as the only odd input of the fused axpy (`add` of the kernel)
                        if case == "reconstruct_odd":
                            self.assertLess(rel(fn(f, plain_e, vo), want), TOL[dtype])

    def test_non_contiguous_add_of_the_kernel(self):
        """The compiled hop with scale and add given as views: out = scale H psi + add."""
        if self.kernel not in ("cpu", "cuda"):
            self.skipTest("compiled kernels only")
        dev = torch.device(self.device)
        geo = G4s
        k = kappas(0.12, 0.13)
        U = links(geo, 2, 52).to(dev)
        psi = field(geo, 2, 53).to(dev)
        e, o = to_half(psi, geo, 0), to_half(psi, geo, 1)
        ref = HalfLatticeWilson(U, k, geo)
        op = self.make(U, k, geo, -1.0)
        for name, add in self._views(e):
            with self.subTest(view=name):
                plain = add.resolve_conj().resolve_neg().clone()
                want = ref.hop(o, 0).mul_(ref._k).add_(plain)
                got = op._hop_kernel(o, 0, scale=op._scale_k, add=add)
                self.assertLess(rel(got, want), 1e-14)

    def test_compiled_hop_refuses_autograd_and_foreign_add(self):
        """The compiled kernel writes through raw pointers: with grad mode on, inputs that require
        grad are an error (not a silently detached output); `add` on another device is an error."""
        if self.kernel not in ("cpu", "cuda"):
            self.skipTest("compiled kernels only")
        dev = torch.device(self.device)
        geo = G4s
        U = links(geo, 2, 54).to(dev)
        x = to_half(field(geo, 2, 55), geo).to(dev)
        op = self.make(U, 0.12, geo, -1.0)
        xg = x.clone().requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "not differentiable.*psi"):
            op.hop(xg, 1)
        with self.assertRaisesRegex(RuntimeError, "not differentiable.*add"):
            op.reconstruct_odd(x, xg)
        with torch.no_grad():  # no graph is being recorded: allowed
            self.assertLess(rel(op.hop(xg, 1), HalfLatticeWilson(U, 0.12, geo).hop(x, 1)), 1e-14)
        Ug = U.clone().requires_grad_(True)
        op_g = self.make(Ug, 0.12, geo, -1.0)  # gathered in grad mode: the links require grad
        with self.assertRaisesRegex(RuntimeError, "not differentiable.*links"):
            op_g.normal(x)
        with torch.no_grad():
            self.assertLess(rel(op_g.normal(x), HalfLatticeWilson(U, 0.12, geo).normal(x)), 1e-14)
        foreign = torch.device("meta") if dev.type == "cpu" else torch.device("cpu")
        with self.assertRaisesRegex(ValueError, "add on"):
            op._hop_kernel(x, 1, scale=op._scale_k, add=x.to(foreign))


class LoopKernelTest(_KernelEquality, unittest.TestCase):
    kernel = "loop"


class CpuEmulationKernelTest(_KernelEquality, unittest.TestCase):
    """The CUDA kernel source compiled for the host: tests its index arithmetic without a GPU."""

    kernel = "cpu"

    @classmethod
    def setUpClass(cls):
        reason = cpu_reason()
        if reason is not None:
            raise unittest.SkipTest(f"host build of the kernel source unavailable: {reason}")


class CpuEmulationSoaKernelTest(CpuEmulationKernelTest):
    """As CpuEmulationKernelTest with the site-fastest link layout (the default)."""

    link_layout = "soa"


class CpuEmulationSiteKernelTest(CpuEmulationKernelTest):
    """As CpuEmulationKernelTest with the site-major link layout (still selectable)."""

    link_layout = "site"


class CudaKernelTest(_KernelEquality, unittest.TestCase):
    kernel = "cuda"
    device = "cuda"

    @classmethod
    def setUpClass(cls):
        reason = cuda_reason()
        if reason is not None:
            raise unittest.SkipTest(f"cuda kernel not testable here: {reason}")


class CudaSoaKernelTest(CudaKernelTest):
    link_layout = "soa"


class CudaSiteKernelTest(CudaKernelTest):
    link_layout = "site"


_CUDA_STUB = r"""
#pragma once
#include <stddef.h>
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __grid_constant__
#include "__clang_cuda_builtin_vars.h"
typedef struct CUstream_st* cudaStream_t;
typedef int cudaError_t;
struct dim3 { unsigned x, y, z;
  __host__ __device__ dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {} };
extern "C" int cudaConfigureCall(dim3 g, dim3 b, size_t s = 0, cudaStream_t st = 0);
extern "C" cudaError_t cudaGetDevice(int*);
extern "C" cudaError_t cudaSetDevice(int);
extern "C" cudaError_t cudaGetLastError(void);
extern "C" const char* cudaGetErrorString(cudaError_t);
"""


def _clang_cuda_check(tmp, side):
    clang = shutil.which("clang++")
    if clang is None:
        return None, "no clang++"
    stub = Path(tmp) / "stub"
    stub.mkdir(exist_ok=True)
    (stub / "cuda_runtime.h").write_text(_CUDA_STUB)
    src = Path(tmp) / "k.cu"
    src.write_text(F.KERNEL_SOURCE)
    cmd = [clang, "-x", "cuda", side, "--cuda-gpu-arch=sm_80", "-nocudainc", "-nocudalib", "-D__CUDACC__",
           "-D__CUDACC_VER_MAJOR__=12", "-D__CUDACC_VER_MINOR__=1", f"-I{stub}", "-std=c++14", "-Wall",
           "-fsyntax-only", str(src)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return res.returncode, res.stderr


class CudaSourceTest(unittest.TestCase):
    def test_clang_cuda_front_end(self):
        """Host and device halves of the CUDA build parse and type-check (clang -x cuda, stub
        runtime header): catches syntax and template errors before the first GPU job."""
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.cu"
            probe.write_text("__attribute__((global)) void k(int* x) { x[0] = 1; }\n")
            clang = shutil.which("clang++")
            if clang is None:
                self.skipTest("no clang++")
            res = subprocess.run([clang, "-x", "cuda", "--cuda-host-only", "--cuda-gpu-arch=sm_80", "-nocudainc",
                                  "-nocudalib", "-fsyntax-only", str(probe)], capture_output=True, text=True)
            if res.returncode != 0:
                self.skipTest(f"clang++ has no CUDA front end: {res.stderr.strip()[:200]}")
            for side in ("--cuda-host-only", "--cuda-device-only"):
                code, err = _clang_cuda_check(tmp, side)
                self.assertEqual(code, 0, f"{side}: {err}")
                self.assertNotIn("warning", err, f"{side}: {err}")


class FallbackTest(unittest.TestCase):
    def test_unavailable_backend_falls_back_with_one_warning(self):
        geo = G4s
        U = links(geo, 2, 7)
        x = to_half(field(geo, 2, 8), geo)
        ref = HalfLatticeWilson(U, 0.12, geo)
        F._WARNED.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            op = FusedHalfLatticeWilson(U, 0.12, geo, kernel="cuda")  # CPU links: not a CUDA device
            op2 = FusedHalfLatticeWilson(U, 0.12, geo, kernel="cuda")
        msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(msgs), 1, msgs)
        self.assertIn("'fast'", msgs[0])
        for o in (op, op2):
            self.assertEqual(o.kernel, "fast")
            self.assertFalse(o.compiled)
            self.assertEqual(o.colour_product, "matmul")
            self.assertIsNotNone(o.kernel_fallback_reason)
            self.assertLess(rel(o.normal(x), ref.normal(x)), 1e-14)
        with self.assertRaises(BackendUnavailable):
            FusedHalfLatticeWilson(U, 0.12, geo, kernel="cuda", strict=True)
        # auto on the CPU is the parent's best mode (matmul), silently; never the slow "loop"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            op_auto = FusedHalfLatticeWilson(U, 0.12, geo)
            self.assertEqual(op_auto.kernel, "fast")
            self.assertEqual(op_auto.colour_product, "matmul")
            self.assertIsNone(op_auto.kernel_fallback_reason)
            self.assertEqual(F.resolve_kernel("auto", "cpu", geo), ("fast", None))
        self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])
        self.assertLess(rel(op_auto.normal(x), ref.normal(x)), 1e-14)
        self.assertLess(rel(op_auto.to(C64).normal(x.to(C64)), ref.to(C64).normal(x.to(C64))), 1e-6)
        # "loop" stays selectable explicitly
        self.assertEqual(FusedHalfLatticeWilson(U, 0.12, geo, kernel="loop").kernel, "loop")
        self.assertEqual(F.resolve_kernel("loop", "cpu", geo), ("loop", None))
        # default link layout is "soa"; "site" selectable
        self.assertEqual(op_auto.link_layout, "soa")
        self.assertEqual(F.DEFAULT_LINK_LAYOUT, "soa")
        self.assertEqual(FusedHalfLatticeWilson(U, 0.12, geo, link_layout="site").link_layout, "site")
        with self.assertRaises(ValueError):
            FusedHalfLatticeWilson(U, 0.12, geo, kernel="triton")
        with self.assertRaises(ValueError):
            FusedHalfLatticeWilson(U, 0.12, geo, link_layout="aos")

    def test_unsupported_colour_falls_back(self):
        geo = Geometry(4, 4, 2, 4)  # N = 4: no compiled instantiation (random complex links suffice)
        U = torch.randn(geo.link_shape(1), dtype=C, generator=torch.Generator().manual_seed(9))
        x = to_half(field(geo, 1, 10), geo)
        F._WARNED.clear()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            op = FusedHalfLatticeWilson(U, 0.1, geo, kernel="cpu")
        self.assertEqual(op.kernel, "fast")
        self.assertIn("N=4", op.kernel_fallback_reason)
        self.assertLess(rel(op.normal(x), HalfLatticeWilson(U, 0.1, geo).normal(x)), 1e-14)

    def test_backend_status_without_cuda(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA present")
        ok, why = backend_status("cuda")
        self.assertFalse(ok)
        self.assertIn("CUDA", why)

    def test_wrong_inputs_rejected(self):
        reason = cpu_reason()
        if reason is not None:
            self.skipTest(reason)
        geo = G4s
        op = FusedHalfLatticeWilson(links(geo, 2, 11), 0.12, geo, kernel="cpu", strict=True)
        x = to_half(field(geo, 2, 12), geo)
        with self.assertRaises(TypeError):
            op.normal(x.to(C64))
        with self.assertRaises(ValueError):
            op.hop(to_half(field(geo, 3, 12), geo), 0)  # batch 3 vs link batch 2


class FusedTermTest(unittest.TestCase):
    def test_term_matches_fast_term(self):
        """Action and force of FusedTwoFlavourWilson equal FastTwoFlavourWilson (same draws)."""
        geo = G46
        U = links(geo, 2, 20)
        kernels = ["fast", "loop"] + (["cpu"] if cpu_reason() is None else [])
        fast = FastTwoFlavourWilson(kappas(0.12, 0.13), geo, solver_tol=1e-12, action_tol=1e-13)
        fast.refresh(U, torch.Generator().manual_seed(21))
        S_ref, F_ref = fast.action(U), fast.force(U)
        for kernel in kernels:
            with self.subTest(kernel=kernel):
                term = FusedTwoFlavourWilson(kappas(0.12, 0.13), geo, kernel="auto" if kernel == "fast" else kernel,
                                             solver_tol=1e-12, action_tol=1e-13)
                term.refresh(U, torch.Generator().manual_seed(21))
                self.assertEqual(term.operator(U).kernel, kernel)
                self.assertLess(rel(term.phi, fast.phi), 1e-14)
                self.assertLess((term.action(U) - S_ref).abs().max().item(), 1e-8)
                self.assertLess(rel(term.force(U), F_ref), 1e-9)

    def test_operator_cache_key_and_alias(self):
        geo = G4s
        U = links(geo, 1, 22)
        term = FusedTwoFlavourWilson(0.12, geo, colour_product="fused")
        self.assertEqual(term.colour_product, "auto")
        op = term.operator(U)
        self.assertIsInstance(op, FusedHalfLatticeWilson)
        self.assertIs(term.operator(U), op)
        self.assertEqual(op.kernel, "fast")
        self.assertEqual(op.link_layout, "soa")
        term.link_layout = "site"  # a different link layout: new operator
        op_site = term.operator(U)
        self.assertIsNot(op_site, op)
        self.assertEqual(op_site.link_layout, "site")
        term.kernel = "loop"  # a different request: new operator
        self.assertIsNot(term.operator(U), op_site)
        with self.assertRaises(ValueError):
            FusedTwoFlavourWilson(0.12, geo, link_layout="aos")
        with self.assertRaises(ValueError):
            FusedTwoFlavourWilson(0.12, geo, kernel="nope")
        self.assertEqual(FusedHalfLatticeWilson(U, 0.12, geo, colour_product="fused").colour_product, "matmul")

    def test_hmc_trajectory_matches_fast(self):
        """Two trajectories (2D, batch 3): dH equal to FastTwoFlavourHamiltonian to 1e-8, links to 1e-10."""
        geo = G2
        kw = dict(n_gauge_substeps=2, solver_tol=1e-12, action_tol=1e-13)
        U0 = links(geo, 3, 30)
        runs = [("fast", FastTwoFlavourHamiltonian, {}), ("loop", FusedTwoFlavourHamiltonian, {"kernel": "loop"})]
        if cpu_reason() is None:
            runs.append(("cpu", FusedTwoFlavourHamiltonian, {"kernel": "cpu"}))
        out = {}
        for name, cls, extra in runs:
            ham = cls(2.0, 0.2, geo, **kw, **extra)
            hmc = H.HMC(geo, ham, tau=0.5, n_steps=4)
            gen = torch.Generator().manual_seed(31)
            U = U0.clone()
            dHs = []
            for _ in range(2):
                U, acc, dH, _ = hmc.trajectory(U, gen)
                dHs.append(dH)
            out[name] = (U, torch.stack(dHs))
        for name in out:
            if name == "fast":
                continue
            with self.subTest(kernel=name):
                self.assertLess((out[name][1] - out["fast"][1]).abs().max().item(), 1e-8)
                self.assertLess((out[name][0] - out["fast"][0]).abs().max().item(), 1e-10)

    def test_hmc_4d_mixed_precision(self):
        """One trajectory on 2^3 x 4 with mixed-precision solves through the fused operator."""
        geo = G4s
        U0 = links(geo, 2, 40)
        out = {}
        runs = [("fast", FastTwoFlavourHamiltonian, {}), ("loop", FusedTwoFlavourHamiltonian, {"kernel": "loop"})]
        if cpu_reason() is None:
            runs.append(("cpu", FusedTwoFlavourHamiltonian, {"kernel": "cpu"}))
        for name, cls, extra in runs:
            ham = cls(5.0, 0.12, geo, n_gauge_substeps=1, solver_tol=1e-12, action_tol=1e-13,
                      precision="mixed", **extra)
            hmc = H.HMC(geo, ham, tau=0.3, n_steps=3)
            U, acc, dH, _ = hmc.trajectory(U0.clone(), torch.Generator().manual_seed(41))
            out[name] = (U, dH)
        for name in [n for n in out if n != "fast"]:
            with self.subTest(kernel=name):
                self.assertLess((out[name][1] - out["fast"][1]).abs().max().item(), 1e-8)
                self.assertLess((out[name][0] - out["fast"][0]).abs().max().item(), 1e-10)


if __name__ == "__main__":
    unittest.main()
