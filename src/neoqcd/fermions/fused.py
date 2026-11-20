"""Opt-in fused hop kernels for the half-lattice Wilson operator of `fast.py`.

`FusedHalfLatticeWilson` is a `HalfLatticeWilson` (same tables, gathered links, site order,
boundary phases, dtype handling, force) whose hop H (and H^dagger) runs through one of
    kernel="cuda"  a custom CUDA kernel (one pass per output site, no temporaries), compiled
                   at first use with nvcc and called through ctypes (below);
    kernel="cpu"   the SAME kernel source compiled for the host with the system C++ compiler
                   (single-threaded loop over the CUDA thread index): the CPU emulation path
                   that tests the kernel's index arithmetic without a GPU;
    kernel="loop"  a pure-torch low-memory variant (runs everywhere, no compiler; slow: only for
                   tests and as an explicit choice, never chosen automatically);
    kernel="auto"  "cuda" on a CUDA device when the kernel builds and loads, otherwise "fast".
The resolved kernel "fast" is the parent's own hop (`HalfLatticeWilson.hop` with its resolved
`colour_product`: broadcast_sum on CUDA, matmul on CPU), the best non-compiled mode measured.
A compiled backend that cannot be built or loaded (no nvcc, no compiler, unsupported D or N)
falls back to "fast" (silently for "auto", with one RuntimeWarning per reason for an explicit
"cuda" / "cpu"; strict=True raises instead); nothing here raises at import, and nothing in the
package uses this module unless asked. `HalfLatticeWilson` / `EvenOddWilsonDirac` stay the ground
truth (test/test_fermions_fused.py). `resolve_kernel` gives the resolution without building an
operator (main/main_hmc.py records it).

Measured (A100 64 GB, torch 2.0, CUDA 12.1; job 64456003, commit 6858689)
--------------------------------------------------------------------------
The library compiled with nvcc and loaded through ctypes at the first attempt; all CUDA equality
tests passed (largest relative deviation 2.55e-16 in complex128). Normal operator D_hat^dag D_hat
in ms, complex128 / complex64:
    lattice  batch   broadcast_sum    fused "site"    fused "soa"
    8^4      4       0.588 / 0.476    0.099 / 0.078   0.085 / 0.080
    16^4     16      53.66 / 32.42    3.935 / 1.661   2.992 / 1.523
    16^4     64      222.2 / 141.8    16.23 / 6.544   11.99 / 5.979
The site-fastest link layout "soa" is faster by 25-35% in double precision at 16^4 and by 3-9%
in single: it is the default. Lean CG at 16^4, B = 16: 66.15 ms per iteration (broadcast_sum)
vs 5.41 ms (fused soa); full solve to 1e-10: 2256.6 ms (lean CG, double), 219.8 ms (fused,
double), 139.7 ms (fused, mixed precision). Best throughput 3.5e8 site-updates/s in double,
7.0e8 in single. The pure-torch "loop" is slower than broadcast_sum at small sizes (3.4 ms vs
0.59 ms at 8^4 B4) and only 1.3x faster at large ones, hence "fast" as the fallback.

Why a custom kernel
-------------------
`HalfLatticeWilson.hop` makes several passes over memory (projection at the source, gather of the
2D projected neighbours, colour product with a temporary 3x the output, reconstruction,
concatenation, sum over terms) and is memory-bandwidth bound on the A100. The fused kernel reads,
per output site, the 2D neighbour 4-spinors and the 2D links once and writes the output once.

Backend choice (Leonardo, cineca-ai/4.1.1: torch 2.0.0, CUDA 12.1, Python 3.11)
---------------------------------------------------------------------------------
No triton, cupy, numba or pycuda in the environment; nvcc 12.1 (= torch.version.cuda), gcc 12.2
and ninja are available after `module load profile/deeplrn cineca-ai/`. The kernel is compiled
with plain nvcc into a shared library with an extern "C" entry point and called through ctypes
with raw device pointers and the current torch stream, rather than through
torch.utils.cpp_extension: no torch headers (seconds of nvcc instead of a minute-long torch
build), no ninja, no pybind, no build lock that a killed job leaves behind (the library is written
to a temporary name and renamed atomically), and the identical source also compiles with any host
C++ compiler (the "cpu" emulation, testable on a laptop). The library links the CUDA runtime
statically; it shares the device's primary context with torch, sets the device of the tensors
before the launch and restores the previous one.
Build cache: $NEOQCD_KERNEL_CACHE, default ~/.cache/neoqcd/kernels, one file per source hash,
compiler and GPU architecture. nvcc is looked up in $CUDA_HOME/bin, $CUDA_PATH/bin, then PATH;
the host compiler in $CXX, then c++, g++, clang++.

Kernel design (KERNEL_SOURCE)
-----------------------------
Layouts (all contiguous, complex as interleaved (re, im) pairs):
    psi    (B, V/2, S, N)          source field, sites of parity 1 - t in `site_tables` order;
    links  (BL, V/2, 2D, N, N)     `HalfLatticeWilson._links[t]` as is: entry [i, k, a, c] is
                                   M_k(x_i)[a, c] with M = L_mu(x)^T (k < D, forward) or
                                   conj(L_mu(x - mu)) (k >= D, backward), boundary phase included;
                                   the colour vector of the term is chi_c = sum_a M[a, c] h_a;
                                   BL = B, or 1 (links shared by the batch: batch stride 0);
                                   link_layout="soa": the same numbers as (BL, 2D, N, N, V/2)
                                   (site index fastest), a copy built once per configuration;
    nbr    (V/2, 2D) int32         nbr[i, k] = position (in the source half field) of x_i + mu
                                   (k < D) or x_i - mu (k >= D), = hop_index // 2D of `site_tables`;
    out    (B, V/2, S, N).
One CUDA thread per output (b, i, c) (sample, target site, colour): tid = (b V/2 + i) N + c, 128
threads per block. For each term k (fully unrolled: D and N are template parameters; the spin
tables are in the kernel parameter space) n = nbr[i, k]; for each contracted colour a and
half-spinor row j < H = S/2 the projection is done on the fly from the neighbour 4-spinor,
    h_j[a] = psi[b, n, j, a] + proj_coef[k, j] psi[b, n, H + proj_perm[k, j], a],
and accumulated into chi_j = sum_a h_j[a] M_k[a, c]; then the reconstruction adds chi_j to the
upper spin j and rec_coef[k, j] chi_{rec_perm[k, j]} to the lower spin H + j of the per-thread
accumulator (S complex registers). The tables are those of `spin_projection_tables` (dagger:
both coefficients negated, resolved on the host). The output is written once,
    out[b, i, s, c] = scale[b] acc_s + add[b, i, s, c]
(scale and add optional: the axpy of D_hat x = x - kappa^2 H_eo H_oe x, of prepare_rhs and of the
odd reconstruction are fused into the second hop). Per site and hop: 2D neighbour 4-spinors and
2D links read (the neighbour reads of the N colour threads of a site hit L1), one spinor written.
Precision: complex128 and complex64 (double / float arithmetic, no fast-math). Instantiated for
D in (2, 4) and N in (2, 3) (other geometries fall back to "fast").
Link reads: with "site" a warp (about 11 sites x 3 colours) reads, per (k, a), 11 runs of 3
complex numbers 1152 bytes apart (complex128), the rest of each run being used by the following
(k, a) through L1; with "soa" it reads 3 runs of 11 consecutive complex numbers. Which is faster
on the A100 is measured by main/bench_fermions.py (rows half_lattice_fused_cuda[_soa]): "soa"
(numbers at the top).

Pure-torch variant ("loop")
---------------------------
Accumulates the 2D terms one at a time into a preallocated output with out= / in-place ops: per
term, project the source field at the source sites (index_select on spin + addcmul, H N per
site), gather the neighbours (one index_select of half spinors), colour product as N
mul/addcmul_ on strided link slices, add to the upper half, reconstruct the lower half
(index_select + addcmul_). Temporaries are half fields; no (2D, H, N, N) product tensor.
"""

import copy
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import torch

from neoqcd.fermions.fast import FastTwoFlavourWilson, HalfLatticeWilson
from neoqcd.fermions.hmc import Hamiltonian, WilsonGaugeAction

__all__ = [
    "KERNELS",
    "LINK_LAYOUTS",
    "KERNEL_SOURCE",
    "BackendUnavailable",
    "backend_status",
    "load_backend",
    "resolve_kernel",
    "FusedHalfLatticeWilson",
    "FusedTwoFlavourWilson",
    "FusedTwoFlavourHamiltonian",
]

KERNELS = ("auto", "cuda", "cpu", "loop")
RESOLVED_KERNELS = ("cuda", "cpu", "loop", "fast")  # "fast": the parent's hop (fallback)
_SUPPORTED_D = (2, 4)
_SUPPORTED_N = (2, 3)
LINK_LAYOUTS = ("site", "soa")
DEFAULT_LINK_LAYOUT = "soa"  # measured faster on the A100 (module docstring)
_MAX_TERMS, _MAX_HALF = 8, 2

KERNEL_SOURCE = r"""
// neoqcd fused half-lattice Wilson hop: one source for CUDA (nvcc) and the host (C++ compiler).
// Conventions and layouts: module docstring of neoqcd/fermions/fused.py.
#include <stdint.h>
#include <stddef.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define NQ_HD __host__ __device__ __forceinline__
#define NQ_UNROLL _Pragma("unroll")
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 7))
#define NQ_GRID_CONSTANT __grid_constant__  // the kernel argument stays in parameter space
#else
#define NQ_GRID_CONSTANT
#endif
#else
#define NQ_HD inline
#define NQ_UNROLL
#endif

#define NQ_MAX_TERMS 8  // 2D, D <= 4
#define NQ_MAX_HALF 2   // H = S / 2

// Arguments of one hop (mirrored by the ctypes structure _HopArgs in fused.py; the host checks
// sizeof through nq_args_size()).
struct NqHopArgs {
    void* out;                     // (B, V2, S, N) complex
    const void* psi;               // (B, V2, S, N) complex, source parity
    const void* links;             // M_k(x_i)[a][c]: layout 0 (BL, V2, 2D, N, N), layout 1 (BL, 2D, N, N, V2)
    const int32_t* nbr;            // (V2, 2D) neighbour positions in the source half field
    const void* add;               // NULL or (B, V2, S, N) complex
    const void* scale;             // NULL or (B or 1,) real
    int64_t batch;                 // B (field batch)
    int64_t half_volume;           // V2
    int64_t link_batch_stride;     // complex elements between link samples (0: shared links)
    int64_t scale_batch_stride;    // 0 (one scale for all samples) or 1
    int32_t D, N, dtype, device;   // dtype 0: complex128, 1: complex64
    int32_t link_layout;           // 0: site-major (HalfLatticeWilson._links), 1: site-fastest
    int32_t reserved;
    int32_t proj_perm[NQ_MAX_TERMS][NQ_MAX_HALF];      // in 0..H-1 (lower half)
    double proj_coef[NQ_MAX_TERMS][NQ_MAX_HALF][2];    // (re, im), +-1 or +-i
    int32_t rec_perm[NQ_MAX_TERMS][NQ_MAX_HALF];       // in 0..H-1 (row of chi)
    double rec_coef[NQ_MAX_TERMS][NQ_MAX_HALF][2];
};

template <typename T> struct alignas(2 * sizeof(T)) NqC { T re, im; };

template <typename T> NQ_HD NqC<T> nq_make(T re, T im) { NqC<T> z; z.re = re; z.im = im; return z; }
template <typename T> NQ_HD NqC<T> nq_add(NqC<T> a, NqC<T> b) { return nq_make<T>(a.re + b.re, a.im + b.im); }
template <typename T> NQ_HD NqC<T> nq_mul(NqC<T> a, NqC<T> b) {
    return nq_make<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
template <typename T> NQ_HD NqC<T> nq_coef(const double c[2]) { return nq_make<T>((T)c[0], (T)c[1]); }

// Offset of M_k(x_i)[col][c] within one link sample (K = 2D terms, V2 sites).
template <int LAYOUT, int K, int N>
NQ_HD int64_t nq_link_offset(int64_t i, int k, int col, int c, int64_t V2) {
    return LAYOUT == 0 ? i * (K * N * N) + (k * N + col) * N + c   // (V2, 2D, N, N)
                       : ((int64_t)(k * N + col) * N + c) * V2 + i;  // (2D, N, N, V2): coalesced in i
}

// One output (b, i, c): sample b, target site i (position in the target half field), colour c.
template <typename T, int D, int N, int LAYOUT>
NQ_HD void nq_hop_site(const NqHopArgs& a, int64_t b, int64_t i, int c) {
    constexpr int S = (D == 4) ? 4 : 2;  // 2^(D / 2) spin components
    constexpr int H = S / 2;             // half-spinor components
    constexpr int K = 2 * D;             // hop terms: k < D forward along mu = k, k >= D backward
    typedef NqC<T> C;
    const C* psi = static_cast<const C*>(a.psi) + b * a.half_volume * (S * N);
    const C* links = static_cast<const C*>(a.links) + b * a.link_batch_stride;
    const int32_t* nbr = a.nbr + i * K;
    C acc[S];
    NQ_UNROLL
    for (int s = 0; s < S; ++s) acc[s] = nq_make<T>(0, 0);
    NQ_UNROLL
    for (int k = 0; k < K; ++k) {
        const C* src = psi + (int64_t)nbr[k] * (S * N);  // neighbour 4-spinor (S, N)
        C chi[H];
        NQ_UNROLL
        for (int j = 0; j < H; ++j) chi[j] = nq_make<T>(0, 0);
        NQ_UNROLL
        for (int col = 0; col < N; ++col) {  // contracted colour index a
            const C m = links[nq_link_offset<LAYOUT, K, N>(i, k, col, c, a.half_volume)];  // M_k[a][c]
            NQ_UNROLL
            for (int j = 0; j < H; ++j) {
                // projection on the fly: h_j[a] = psi_j[a] + proj_coef[k][j] psi_{H + proj_perm[k][j]}[a]
                const C h = nq_add(src[j * N + col],
                                   nq_mul(nq_coef<T>(a.proj_coef[k][j]), src[(H + a.proj_perm[k][j]) * N + col]));
                chi[j] = nq_add(chi[j], nq_mul(h, m));
            }
        }
        // reconstruction: upper j += chi_j; lower H + j += rec_coef[k][j] chi_{rec_perm[k][j]}
        NQ_UNROLL
        for (int j = 0; j < H; ++j) {
            acc[j] = nq_add(acc[j], chi[j]);
            const C sel = (a.rec_perm[k][j] == 0) ? chi[0] : chi[H - 1];  // H <= 2: no dynamic register index
            acc[H + j] = nq_add(acc[H + j], nq_mul(nq_coef<T>(a.rec_coef[k][j]), sel));
        }
    }
    const int64_t base = (b * a.half_volume + i) * (S * N) + c;
    C* out = static_cast<C*>(a.out) + base;
    const T sc = a.scale ? static_cast<const T*>(a.scale)[b * a.scale_batch_stride] : (T)1;
    const C* add = a.add ? static_cast<const C*>(a.add) + base : (const C*)0;
    NQ_UNROLL
    for (int s = 0; s < S; ++s) {
        C v = nq_make<T>(sc * acc[s].re, sc * acc[s].im);
        if (add) v = nq_add(v, add[s * N]);
        out[s * N] = v;
    }
}

// Flat thread index -> (b, i, c), c fastest: shared by the CUDA kernel and the host loop.
template <typename T, int D, int N, int LAYOUT>
NQ_HD void nq_hop_thread(const NqHopArgs& a, int64_t tid) {
    const int c = (int)(tid % N);
    const int64_t site = tid / N;
    nq_hop_site<T, D, N, LAYOUT>(a, site / a.half_volume, site % a.half_volume, c);
}

#ifdef __CUDACC__
template <typename T, int D, int N, int LAYOUT>
__global__ void nq_hop_kernel(const NQ_GRID_CONSTANT NqHopArgs a, int64_t total) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) nq_hop_thread<T, D, N, LAYOUT>(a, tid);
}

template <typename T, int D, int N, int LAYOUT>
static int nq_launch(const NqHopArgs& a, void* stream) {
    const int64_t total = a.batch * a.half_volume * N;
    if (total == 0) return 0;
    const int threads = 128;
    const int64_t blocks = (total + threads - 1) / threads;
    nq_hop_kernel<T, D, N, LAYOUT><<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(a, total);
    return (int)cudaGetLastError();
}
#else
template <typename T, int D, int N, int LAYOUT>
static int nq_launch(const NqHopArgs& a, void* stream) {
    (void)stream;
    const int64_t total = a.batch * a.half_volume * N;
    for (int64_t tid = 0; tid < total; ++tid) nq_hop_thread<T, D, N, LAYOUT>(a, tid);
    return 0;
}
#endif

template <typename T>
static int nq_dispatch(const NqHopArgs& a, void* stream) {
#define NQ_CASE(DD, NN) \
    if (a.D == DD && a.N == NN) \
        return a.link_layout == 0 ? nq_launch<T, DD, NN, 0>(a, stream) : nq_launch<T, DD, NN, 1>(a, stream);
    NQ_CASE(4, 3) NQ_CASE(4, 2) NQ_CASE(2, 3) NQ_CASE(2, 2)
#undef NQ_CASE
    return -1;
}

extern "C" {
int nq_args_size(void) { return (int)sizeof(NqHopArgs); }

int nq_is_cuda(void) {
#ifdef __CUDACC__
    return 1;
#else
    return 0;
#endif
}

// Returns 0 on success, -1 for an unsupported (dtype, D, N), a cudaError_t code otherwise.
int nq_hop(const NqHopArgs* a, void* stream) {
#ifdef __CUDACC__
    int previous = -1;
    int err = (int)cudaGetDevice(&previous);
    if (err) return err;
    if (previous != a->device && (err = (int)cudaSetDevice(a->device))) return err;
#endif
    int code = -1;
    if (a->dtype == 0) code = nq_dispatch<double>(*a, stream);
    else if (a->dtype == 1) code = nq_dispatch<float>(*a, stream);
#ifdef __CUDACC__
    if (previous != a->device) cudaSetDevice(previous);
#endif
    return code;
}

const char* nq_error_string(int code) {
    if (code == -1) return "unsupported (dtype, D, N)";
#ifdef __CUDACC__
    return cudaGetErrorString((cudaError_t)code);
#else
    return code == 0 ? "no error" : "unknown error";
#endif
}
}
"""


# ---- ctypes mirror of NqHopArgs ----------------------------------------------------------------
class _HopArgs(ctypes.Structure):
    _fields_ = [
        ("out", ctypes.c_void_p),
        ("psi", ctypes.c_void_p),
        ("links", ctypes.c_void_p),
        ("nbr", ctypes.c_void_p),
        ("add", ctypes.c_void_p),
        ("scale", ctypes.c_void_p),
        ("batch", ctypes.c_int64),
        ("half_volume", ctypes.c_int64),
        ("link_batch_stride", ctypes.c_int64),
        ("scale_batch_stride", ctypes.c_int64),
        ("D", ctypes.c_int32),
        ("N", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("device", ctypes.c_int32),
        ("link_layout", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("proj_perm", (ctypes.c_int32 * _MAX_HALF) * _MAX_TERMS),
        ("proj_coef", ((ctypes.c_double * 2) * _MAX_HALF) * _MAX_TERMS),
        ("rec_perm", (ctypes.c_int32 * _MAX_HALF) * _MAX_TERMS),
        ("rec_coef", ((ctypes.c_double * 2) * _MAX_HALF) * _MAX_TERMS),
    ]


_DTYPE_CODE = {torch.complex128: 0, torch.complex64: 1}


def _raw(t):
    """t as plain contiguous memory for a raw pointer: lazy conjugate and negative bits resolved
    (a neg view such as torch._neg_view(x) shares x's memory un-negated), strides, storage offsets
    and expanded (stride 0) dimensions materialised. A no-op for a plain contiguous tensor."""
    return t.resolve_conj().resolve_neg().contiguous()


# ---- build and load ------------------------------------------------------------------------------
class BackendUnavailable(RuntimeError):
    """The requested kernel backend cannot be built or loaded here (the message says why)."""


_BACKENDS = {}  # key -> _Backend or BackendUnavailable
_WARNED = set()


def _warn_once(message):
    if message not in _WARNED:
        _WARNED.add(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _cache_dir():
    root = os.environ.get("NEOQCD_KERNEL_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "neoqcd" / "kernels"


def _find_nvcc():
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root and (Path(root) / "bin" / "nvcc").is_file():
            return str(Path(root) / "bin" / "nvcc")
    return shutil.which("nvcc")


def _find_cxx():
    candidates = [os.environ.get("CXX")] if os.environ.get("CXX") else []
    for name in candidates + ["c++", "g++", "clang++"]:
        path = shutil.which(name)
        if path:
            return path
    return None


def _version(compiler):
    try:
        res = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=60)
        return res.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _build(target, device=None):
    """Compile KERNEL_SOURCE for target ("cuda" or "cpu") into the cache; returns the library path."""
    if target == "cuda":
        compiler = _find_nvcc()
        if compiler is None:
            raise BackendUnavailable("nvcc not found (set CUDA_HOME or load a cuda module)")
        major, minor = torch.cuda.get_device_capability(device)
        arch = f"{major}{minor}"
        flags = ["-O3", "-std=c++14", "-shared", "-Xcompiler", "-fPIC",
                 f"-gencode=arch=compute_{arch},code=sm_{arch}",
                 f"-gencode=arch=compute_{arch},code=compute_{arch}"]
        suffix = ".cu"
    elif target == "cpu":
        compiler = _find_cxx()
        if compiler is None:
            raise BackendUnavailable("no host C++ compiler found (c++, g++, clang++ or $CXX)")
        flags = ["-O2", "-std=c++14", "-shared", "-fPIC"]
        suffix = ".cpp"
    else:
        raise ValueError(f"unknown target {target!r}")
    tag = hashlib.sha256("\n".join([KERNEL_SOURCE, compiler, _version(compiler)] + flags).encode()).hexdigest()[:16]
    cache = _cache_dir()
    lib = cache / f"neoqcd_hop_{target}_{tag}.so"
    if lib.is_file():
        return lib
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="neoqcd_hop_") as tmp:
        src = Path(tmp) / f"neoqcd_hop{suffix}"
        src.write_text(KERNEL_SOURCE)
        partial = cache / f".{lib.name}.{os.getpid()}.tmp"
        cmd = [compiler] + flags + ["-o", str(partial), str(src)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.SubprocessError) as exc:
            raise BackendUnavailable(f"{target} build failed to run: {exc}")
        if res.returncode != 0:
            try:
                partial.unlink()
            except OSError:
                pass
            msg = (res.stderr or res.stdout).strip().splitlines()
            raise BackendUnavailable(f"{target} build failed ({' '.join(cmd[:1])}): " + " | ".join(msg[-6:]))
        os.replace(partial, lib)  # atomic: concurrent builds and killed jobs leave no half file
    return lib


class _Backend:
    """A loaded kernel library: `hop(args, stream)` raises RuntimeError on a nonzero return code."""

    def __init__(self, target, path):
        self.target = target
        self.path = str(path)
        lib = ctypes.CDLL(self.path)
        lib.nq_args_size.restype = ctypes.c_int
        lib.nq_is_cuda.restype = ctypes.c_int
        lib.nq_hop.argtypes = [ctypes.POINTER(_HopArgs), ctypes.c_void_p]
        lib.nq_hop.restype = ctypes.c_int
        lib.nq_error_string.argtypes = [ctypes.c_int]
        lib.nq_error_string.restype = ctypes.c_char_p
        if lib.nq_args_size() != ctypes.sizeof(_HopArgs):
            raise BackendUnavailable(f"argument layout mismatch: library {lib.nq_args_size()} bytes, "
                                     f"ctypes {ctypes.sizeof(_HopArgs)}")
        if bool(lib.nq_is_cuda()) != (target == "cuda"):
            raise BackendUnavailable(f"library {self.path} is not a {target} build")
        self._lib = lib

    def hop(self, args, stream):
        code = self._lib.nq_hop(ctypes.byref(args), stream)
        if code != 0:
            raise RuntimeError(f"neoqcd fused hop ({self.target}) failed: code {code}: "
                               f"{self._lib.nq_error_string(code).decode(errors='replace')}")


def load_backend(target, device=None):
    """The kernel library for target "cuda" or "cpu" (built at first use, cached per process and
    on disk); raises BackendUnavailable with the reason."""
    if target == "cuda":
        if not torch.cuda.is_available():
            raise BackendUnavailable("CUDA is not available to torch")
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        if device.type != "cuda":
            raise BackendUnavailable(f"the cuda kernel needs a CUDA device, got {device}")
        key = ("cuda", torch.cuda.get_device_capability(device))
    elif target == "cpu":
        key = ("cpu",)
    else:
        raise ValueError(f"unknown target {target!r}")
    hit = _BACKENDS.get(key)
    if isinstance(hit, _Backend):
        return hit
    if isinstance(hit, BackendUnavailable):
        raise hit
    try:
        backend = _Backend(target, _build(target, device))
    except BackendUnavailable as exc:
        _BACKENDS[key] = exc
        raise
    except Exception as exc:  # noqa: BLE001 - any loading problem means: not available
        err = BackendUnavailable(f"{target} backend failed to load: {type(exc).__name__}: {exc}")
        _BACKENDS[key] = err
        raise err
    _BACKENDS[key] = backend
    return backend


def backend_status(target, device=None):
    """(available, reason): reason is the library path when available."""
    try:
        return True, load_backend(target, device).path
    except BackendUnavailable as exc:
        return False, str(exc)


def resolve_kernel(kernel, device, geometry, strict=False):
    """(resolved, reason) of a requested kernel for links on `device` (module docstring).

    resolved is one of RESOLVED_KERNELS; reason is None unless a compiled backend was requested
    (explicitly or by "auto" on a CUDA device) and is unavailable, in which case resolved is
    "fast". An explicit "cuda" / "cpu" that falls back warns once per reason; strict=True raises
    BackendUnavailable instead of falling back."""
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    dev = torch.device(device)
    explicit = kernel != "auto"
    if kernel == "auto":
        if dev.type != "cuda":
            return "fast", None
        kernel = "cuda"
    if kernel == "loop":
        return "loop", None
    reason = None
    if kernel == "cuda" and dev.type != "cuda":
        reason = f"the cuda kernel needs CUDA tensors, the links are on {dev}"
    elif kernel == "cpu" and dev.type != "cpu":
        reason = f"the cpu kernel needs CPU tensors, the links are on {dev}"
    elif geometry.D not in _SUPPORTED_D or geometry.N not in _SUPPORTED_N:
        reason = (f"the {kernel} kernel supports D in {_SUPPORTED_D}, N in {_SUPPORTED_N}; "
                  f"got D={geometry.D}, N={geometry.N}")
    else:
        ok, why = backend_status(kernel, dev)
        if not ok:
            reason = why
    if reason is None:
        return kernel, None
    if strict:
        raise BackendUnavailable(reason)
    if explicit:
        _warn_once(f"neoqcd fused hop: kernel {kernel!r} unavailable ({reason}); "
                   f"using the 'fast' operator (HalfLatticeWilson.hop)")
    return "fast", reason


# ---- operator ----------------------------------------------------------------------------------
class FusedHalfLatticeWilson(HalfLatticeWilson):
    """`HalfLatticeWilson` with the hop through a fused kernel (module docstring).

    Extra args:
        kernel: "auto" (default: "cuda" on a CUDA device, the parent's "fast" hop elsewhere or
            when the CUDA kernel is unavailable), "cuda", "cpu" (host build of the CUDA source,
            for testing) or "loop" (pure torch, explicit only). An explicit compiled backend that
            is not available falls back to "fast" with a RuntimeWarning (strict=True raises
            BackendUnavailable instead). The kernel in use is the attribute `kernel` (one of
            RESOLVED_KERNELS), the reason of a fallback `kernel_fallback_reason` (None if none).
        link_layout: of the compiled kernels: "soa" (default, measured faster) uses a
            site-fastest copy (B, 2D, N, N, V/2) of the gathered links, built at the first
            compiled hop of each dtype (+ the memory of the gathered links), so that the link
            reads of consecutive sites are coalesced; "site" reads the parent's gathered links
            (B, V/2, 2D, N, N) in place (no extra memory).
        colour_product: of the parent, used by `hopping_derivative` (the force) only; "fused" is
            accepted as an alias of "auto" (so that colour_product="fused" selects this class
            through `FusedTwoFlavourWilson`).

    Same methods as the parent (`hop`, `hop_eo`, `hop_oe`, `apply`, `apply_dagger`, `normal`,
    `to`, `prepare_rhs[_half]`, `reconstruct[_odd]`, `hopping_derivative`); with a compiled
    kernel the axpy of `apply`, `prepare_rhs_half` and `reconstruct_odd` is fused into the hop.
    """

    def __init__(self, U, kappa, geometry=None, time_sign=-1.0, dtype=None, colour_product="auto",
                 kernel="auto", strict=False, link_layout=DEFAULT_LINK_LAYOUT):
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
        if link_layout not in LINK_LAYOUTS:
            raise ValueError(f"link_layout must be one of {LINK_LAYOUTS}, got {link_layout!r}")
        self.link_layout = link_layout
        if colour_product == "fused":
            colour_product = "auto"
        self.kernel_requested = kernel
        self._backend = None
        super().__init__(U, kappa, geometry, time_sign, dtype=dtype, colour_product=colour_product)
        self.kernel, self.kernel_fallback_reason = resolve_kernel(kernel, self.device, self.geometry, strict)
        if self.kernel in ("cuda", "cpu"):
            self._backend = load_backend(self.kernel, self.device)

    @property
    def compiled(self):
        """True when the hop runs through a compiled kernel ("cuda" or "cpu")."""
        return self._backend is not None

    # ---- per-dtype state ---------------------------------------------------------------------
    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        geo = self.geometry
        D, H = geo.D, self._spin["H"]
        K = 2 * D
        dev = self.U.device
        if not hasattr(self, "_nbr"):
            # neighbour position of term k at target site i: hop_index = pos * 2D + k
            self._nbr = [(self._tables["hop_index"][t].view(-1, K) // K) for t in (0, 1)]
            self._nbr32 = [n.to(torch.int32).contiguous() for n in self._nbr]
            self._nbr_k = [[n[:, k].contiguous() for k in range(K)] for n in self._nbr]
            rp = self._spin["rec_perm"].view(K, H) - torch.arange(K).view(K, 1) * H  # row of chi
            self._rec_local = rp.to(torch.int32)
            self._rperm_k = [rp[k].to(dev) for k in range(K)]
            self._pperm_k = [self._pperm.view(K, H)[k].contiguous() for k in range(K)]
        real = self._k.dtype
        self._scale_k = self.kappa.to(real).contiguous()
        self._scale_neg_k2 = (-(self.kappa ** 2)).to(real).contiguous()
        self._kargs = {}  # (target parity, dagger) -> _HopArgs template (pointers set per call)
        self._links_soa = None

    def _kernel_links(self, t):
        """Links of target parity t in the layout of the compiled kernel."""
        if self.link_layout == "site":
            return self._links[t]
        if self._links_soa is None:
            self._links_soa = [l.permute(0, 2, 3, 4, 1).contiguous() for l in self._links]
        return self._links_soa[t]

    def to(self, dtype):
        if dtype == self._dtype:
            return self
        new = copy.copy(self)
        new._set_dtype(dtype)
        return new

    # ---- kernels -----------------------------------------------------------------------------
    def _args_template(self, t, dagger):
        key = (t, dagger)
        args = self._kargs.get(key)
        if args is not None:
            return args
        geo = self.geometry
        D, H, N = geo.D, self._spin["H"], geo.N
        K = 2 * D
        sp = self._spin
        sign = -1.0 if dagger else 1.0
        pc = sp["proj_coef"].reshape(K, H) * sign
        rc = sp["rec_coef"].reshape(K, H) * sign
        pp = sp["proj_perm"].reshape(K, H)
        args = _HopArgs()
        links = self._links[t]
        args.half_volume = links.shape[1]
        args.link_batch_stride = 0 if links.shape[0] == 1 else links[0].numel()
        args.link_layout = LINK_LAYOUTS.index(self.link_layout)
        args.D, args.N = D, N
        args.dtype = _DTYPE_CODE[self._dtype]
        args.device = links.device.index if links.device.type == "cuda" and links.device.index is not None else 0
        for k in range(K):
            for j in range(H):
                args.proj_perm[k][j] = int(pp[k, j])
                args.proj_coef[k][j][0] = float(pc[k, j].real)
                args.proj_coef[k][j][1] = float(pc[k, j].imag)
                args.rec_perm[k][j] = int(self._rec_local[k, j])
                args.rec_coef[k][j][0] = float(rc[k, j].real)
                args.rec_coef[k][j][1] = float(rc[k, j].imag)
        self._kargs[key] = args
        return args

    def _hop_kernel(self, psi, t, dagger=False, scale=None, add=None):
        """Compiled hop: out = scale * H psi + add (scale (B,) or (1,) real, add a field; both optional).

        The kernel reads raw memory and writes a fresh tensor, so it is not differentiable: with
        grad mode on, an input that requires grad (psi, add, the links, the scale) is an error
        rather than a silently detached result (use kernel "fast" / "loop", or torch.no_grad()).
        Inputs may be any strided view: lazy conjugate and negative bits are resolved and the
        memory made contiguous before the pointers are taken."""
        self._check(psi)
        links = self._kernel_links(t)
        if torch.is_grad_enabled():
            needs = [name for name, v in (("psi", psi), ("add", add), ("links", links), ("scale", scale))
                     if v is not None and v.requires_grad]
            if needs:
                raise RuntimeError(f"FusedHalfLatticeWilson ({self.kernel} kernel) is not differentiable, but "
                                   f"{', '.join(needs)} require(s) grad with grad mode on: use kernel='fast' or "
                                   "'loop' for autograd, or call it under torch.no_grad()")
        psi = _raw(psi)
        links = _raw(links)
        B = psi.shape[0]
        if tuple(psi.shape[1:]) != tuple(self.field_shape[1:]):
            raise ValueError(f"field of shape {tuple(psi.shape)} does not match {self.field_shape}")
        if psi.device != links.device:
            raise ValueError(f"field on {psi.device}, operator on {links.device}")
        if links.shape[0] != 1 and links.shape[0] != B:
            raise ValueError(f"field batch {B} does not match the link batch {links.shape[0]}")
        out = torch.empty_like(psi, memory_format=torch.contiguous_format)
        a = _HopArgs.from_buffer_copy(self._args_template(t, dagger))
        a.out, a.psi, a.links, a.nbr = out.data_ptr(), psi.data_ptr(), links.data_ptr(), self._nbr32[t].data_ptr()
        a.batch = B
        if scale is not None:
            scale = _raw(scale)
            if scale.shape[0] not in (1, B):
                raise ValueError(f"scale of shape {tuple(scale.shape)} for batch {B}")
            a.scale = scale.data_ptr()
            a.scale_batch_stride = 0 if scale.shape[0] == 1 else 1
        if add is not None:
            if add.device != psi.device:
                raise ValueError(f"add on {add.device}, field and operator on {psi.device}")
            self._check(add)
            add = _raw(add)
            if add.shape != psi.shape:
                raise ValueError(f"add of shape {tuple(add.shape)} differs from {tuple(psi.shape)}")
            a.add = add.data_ptr()
        stream = torch.cuda.current_stream(psi.device).cuda_stream if psi.device.type == "cuda" else None
        self._backend.hop(a, stream)
        return out

    def _hop_loop(self, psi, t, dagger=False):
        """Pure-torch low-memory hop: the 2D terms accumulated one at a time (module docstring)."""
        self._check(psi)
        geo = self.geometry
        D, H, N, S = geo.D, self._spin["H"], geo.N, geo.spin
        B, V2 = psi.shape[:2]
        links = self._links[t]
        pc = self._pcoef[dagger]  # (2D, H, 1)
        rc = self._rcoef[dagger].view(2 * D, H, 1)
        out = psi.new_zeros((B, V2, S, N))
        upper, lower = out[:, :, :H], out[:, :, H:]
        tmp = psi.new_empty((B, V2, H, N))
        proj = psi.new_empty((B, V2, H, N))
        h = psi.new_empty((B, V2, H, N))
        chi = psi.new_empty((B, V2, H, N))
        u = psi[:, :, :H]
        for k in range(2 * D):
            torch.index_select(psi, 2, self._pperm_k[k], out=tmp)  # the lower components g-permuted
            torch.addcmul(u, tmp, pc[k], out=proj)  # half spinors at every source site
            torch.index_select(proj, 1, self._nbr_k[t][k], out=h)  # at the neighbours of the targets
            M = links[:, :, k]  # (BL, V2, N, N): chi_c = sum_a h_a M[a, c]
            torch.mul(h[..., 0:1], M[:, :, None, 0, :], out=chi)
            for a in range(1, N):
                chi.addcmul_(h[..., a:a + 1], M[:, :, None, a, :])
            upper.add_(chi)
            torch.index_select(chi, 2, self._rperm_k[k], out=tmp)
            lower.addcmul_(tmp, rc[k])
        return out

    def hop(self, psi, target_parity, dagger=False):
        """H (dagger: H^dagger) from the parity 1 - t field psi to parity t, by `kernel`."""
        if self.kernel == "loop":
            return self._hop_loop(psi, target_parity, dagger)
        if self.kernel == "fast":
            return super().hop(psi, target_parity, dagger)
        return self._hop_kernel(psi, target_parity, dagger)

    # ---- operator with the axpy fused --------------------------------------------------------
    def apply(self, x, dagger=False):
        if not self.compiled:
            return super().apply(x, dagger)
        w = self._hop_kernel(x, 1, dagger)
        return self._hop_kernel(w, 0, dagger, scale=self._scale_neg_k2, add=x)

    def prepare_rhs_half(self, b_e, b_o):
        if not self.compiled:
            return super().prepare_rhs_half(b_e, b_o)
        return self._hop_kernel(b_o, 0, scale=self._scale_k, add=b_e)

    def reconstruct_odd(self, x_e, b_o):
        if not self.compiled:
            return super().reconstruct_odd(x_e, b_o)
        return self._hop_kernel(x_e, 1, scale=self._scale_k, add=b_o)


# ---- pseudofermion term ------------------------------------------------------------------------
class FusedTwoFlavourWilson(FastTwoFlavourWilson):
    """`FastTwoFlavourWilson` with `FusedHalfLatticeWilson` operators (no change to fast.py).

    Extra args `kernel` (KERNELS, default "auto"), `link_layout` (LINK_LAYOUTS, default "soa")
    and `strict` (raise BackendUnavailable instead of falling back); colour_product="fused" is
    accepted (the force then uses colour_product "auto"). Everything else (heat bath, action,
    force, solvers, caches, convergence policy, stats, mixed precision through `to`) is the
    parent's; the operator cache key includes the kernel and the link layout.
    """

    def __init__(self, kappa, geometry, kernel="auto", colour_product="auto", link_layout=DEFAULT_LINK_LAYOUT,
                 strict=False, **kwargs):
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
        if link_layout not in LINK_LAYOUTS:
            raise ValueError(f"link_layout must be one of {LINK_LAYOUTS}, got {link_layout!r}")
        self.kernel = kernel
        self.link_layout = link_layout
        self.strict = bool(strict)
        if colour_product == "fused":
            colour_product = "auto"
        super().__init__(kappa, geometry, colour_product=colour_product, **kwargs)

    def _operator_key(self):
        return super()._operator_key() + (self.kernel, self.link_layout)

    def operator(self, U):
        c = self._op_cache
        key = self._operator_key()
        if c is not None and c[0] is U and c[1] == U._version and self._same_key(c[2], key):
            return c[3]
        op = FusedHalfLatticeWilson(U, self.kappa, self.geometry, self.time_sign,
                                    colour_product=self.colour_product, kernel=self.kernel,
                                    link_layout=self.link_layout, strict=self.strict)
        op._low = None
        self._op_cache = (U, U._version, key, op)
        return op


class FusedTwoFlavourHamiltonian(Hamiltonian):
    """`FastTwoFlavourHamiltonian` with `FusedTwoFlavourWilson` (fermion outside, gauge inside)."""

    def __init__(self, beta, kappa, geometry, n_gauge_substeps=1, plaquette_weights=None, **fermion_kwargs):
        self.fermion = FusedTwoFlavourWilson(kappa, geometry, **fermion_kwargs)
        self.gauge = WilsonGaugeAction(beta, geometry, plaquette_weights)
        super().__init__([([self.fermion], 1), ([self.gauge], n_gauge_substeps)])
