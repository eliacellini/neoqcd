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
The deviations are printed. CudaSource checks the CUDA halves of both kernel sources (fused.py, fused_smearing.py) with clang's
CUDA front end, where available.
"""

import shutil
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

import neoqcd.fused as fused
import neoqcd.fused_smearing as fs
import neoqcd.smearing_jacobian as sj
import neoqcd.sun_utils as sun

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
TOL_LINK = 1e-13
TOL_LOGDET = 1e-12
TOL_COEFF = 1e-13


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
