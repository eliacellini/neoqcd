"""Opt-in fused kernels for the molecular dynamics of the N_f = 2 Wilson HMC (beyond the hop).

`fused.py` runs the half-lattice Wilson hop through one compiled kernel. This module adds, with the
same build machinery (`fused._build_spec` / `fused.load_backend` with `_MDBackend`: kernel source in
a string, plain nvcc or the host C++ compiler, ctypes, on-disk cache keyed on source, compiler and
GPU architecture, atomic writes), kernels for the rest of a trajectory:

    gauge force    `FusedWilsonGaugeAction`: the staple sum of every link (with the plaquette
                   weights of the gauge-only defect), Omega = U A and its projection, in one pass;
                   optionally the momentum kick P - c F in the same pass (F never stored).

The torch implementations (`hmc.WilsonGaugeAction`, ...) stay the ground truth: the kernels agree
with them to rounding (test/test_fermions_fused_hmc.py), never bitwise, so a run with the kernels is
a different (equally valid) chain from the same seed. Nothing here is used unless asked for
(`main_hmc.py --kernel ... --md-kernels ...`).

Targets: "cuda" (compiled with nvcc, run on torch's current stream of the tensors' device, made
current through torch as for the hop) and "cpu" (the same source built for the host and run as a
loop over the CUDA thread index: the CPU emulation the tests use). complex128 only for the forces
(the molecular dynamics stays in double precision: reversibility); D in (2, 4), N in (2, 3).
Anything else (another dtype, D, N, device, an input that requires grad with grad mode on, ...)
runs the torch code (once-per-reason RuntimeWarning when a compiled kernel was requested
explicitly; strict=True raises instead), except autograd inputs, which are an error as in
`fused.FusedHalfLatticeWilson`.

Kernel conventions (MD_KERNEL_SOURCE)
-------------------------------------
Links and momenta: (B, D, V, N, N) complex, contiguous (the lattice flattened in lexicographic
order (t, x_1, ...), last index fastest), complex numbers as (re, im) pairs. The neighbour
"x + mu" of the code (torch.roll(f, +1, axis_mu): array index n - e_mu) and "x - mu" (n + e_mu)
are computed in the kernel from the site index, the extents and the strides (periodic).

Gauge force (`nq_gauge_force`), one thread per link (b, mu, x), tid = (b D + mu) V + x:
    A = sum_{nu != mu} [ w_{mu nu}(x) U_nu(x + mu) U_mu(x + nu)^dag U_nu(x)^dag
                       + w_{mu nu}(x - nu) U_nu(x + mu - nu)^dag U_mu(x - nu)^dag U_nu(x - nu) ],
    Omega = U_mu(x) A,  Z = Omega - Omega^dag - tr(Omega - Omega^dag) / N,
    F = -i beta / (2N) Z                                       (= WilsonGaugeAction.force);
the weights are a (BW, P, V) real tensor (BW = 1 or B, plane p = nu (nu - 1) / 2 + mu for mu < nu,
the order of `WilsonGaugeAction._weighted_retr`), absent without a defect; beta is (B or 1,).
Written: F, or (kick) P - c F.
"""

import ctypes
import warnings

import torch

from neoqcd.fermions import fused as _fused
from neoqcd.fermions.hmc import WilsonGaugeAction

__all__ = [
    "MD_KERNEL_SOURCE",
    "MD_TARGETS",
    "md_backend_status",
    "load_md_backend",
    "gauge_force",
    "FusedWilsonGaugeAction",
]

MD_TARGETS = ("cuda", "cpu")
_SUPPORTED_D = (2, 4)
_SUPPORTED_N = (2, 3)
_MAX_D = 4

MD_KERNEL_SOURCE = r"""
// neoqcd fused molecular-dynamics kernels (gauge force, ...): one source for CUDA (nvcc) and the
// host (C++ compiler). Conventions and layouts: module docstring of neoqcd/fermions/fused_hmc.py.
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define MD_HD __host__ __device__ __forceinline__
#define MD_UNROLL _Pragma("unroll")
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 7))
#define MD_GRID_CONSTANT __grid_constant__
#else
#define MD_GRID_CONSTANT
#endif
#else
#define MD_HD static inline
#define MD_UNROLL
#endif

#define MD_THREADS 128
#define MD_MAX_D 4

// ---- complex numbers and N x N matrices (row-major) ---------------------------------------------
template <typename T> struct alignas(2 * sizeof(T)) MdC { T re, im; };

template <typename T> MD_HD MdC<T> c_make(T re, T im) { MdC<T> z; z.re = re; z.im = im; return z; }
template <typename T> MD_HD MdC<T> c_add(MdC<T> a, MdC<T> b) { return c_make<T>(a.re + b.re, a.im + b.im); }
template <typename T> MD_HD MdC<T> c_sub(MdC<T> a, MdC<T> b) { return c_make<T>(a.re - b.re, a.im - b.im); }
template <typename T> MD_HD MdC<T> c_conj(MdC<T> a) { return c_make<T>(a.re, -a.im); }
template <typename T> MD_HD MdC<T> c_mul(MdC<T> a, MdC<T> b) {
    return c_make<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
template <typename T> MD_HD MdC<T> c_scale(T s, MdC<T> a) { return c_make<T>(s * a.re, s * a.im); }

template <typename T, int N> MD_HD void m_load(MdC<T>* M, const MdC<T>* p) {
    MD_UNROLL
    for (int e = 0; e < N * N; ++e) M[e] = p[e];
}

template <typename T, int N> MD_HD void m_store(MdC<T>* p, const MdC<T>* M) {
    MD_UNROLL
    for (int e = 0; e < N * N; ++e) p[e] = M[e];
}

// out = op(A) op(B), op the dagger when DA (DB) is set
template <typename T, int N, bool DA, bool DB>
MD_HD void m_mul(MdC<T>* out, const MdC<T>* A, const MdC<T>* B) {
    MD_UNROLL
    for (int i = 0; i < N; ++i) {
        MD_UNROLL
        for (int k = 0; k < N; ++k) {
            MdC<T> s = c_make<T>(0, 0);
            MD_UNROLL
            for (int j = 0; j < N; ++j) {
                const MdC<T> x = DA ? c_conj(A[N * j + i]) : A[N * i + j];
                const MdC<T> y = DB ? c_conj(B[N * k + j]) : B[N * j + k];
                s = j == 0 ? c_mul(x, y) : c_add(s, c_mul(x, y));
            }
            out[N * i + k] = s;
        }
    }
}

// S += w M (w = 1 without weights)
template <typename T, int N, bool WEIGHTED>
MD_HD void m_accumulate(MdC<T>* S, const MdC<T>* M, T w) {
    MD_UNROLL
    for (int e = 0; e < N * N; ++e) S[e] = c_add(S[e], WEIGHTED ? c_scale(w, M[e]) : M[e]);
}

// ---- lattice neighbours (code convention: "x + mu" is array index n - e_mu, periodic) -----------
MD_HD int64_t md_fwd(int64_t site, int mu, const int64_t* extent, const int64_t* stride) {
    const int64_t c = (site / stride[mu]) % extent[mu];
    return c == 0 ? site + (extent[mu] - 1) * stride[mu] : site - stride[mu];
}

MD_HD int64_t md_bwd(int64_t site, int mu, const int64_t* extent, const int64_t* stride) {
    const int64_t c = (site / stride[mu]) % extent[mu];
    return c == extent[mu] - 1 ? site - (extent[mu] - 1) * stride[mu] : site + stride[mu];
}

// ---- gauge force ----------------------------------------------------------------------------------
// Arguments of nq_gauge_force (mirrored by the ctypes structure _GaugeArgs in fused_hmc.py; the host
// checks sizeof through nq_gauge_args_size()).
struct NqGaugeArgs {
    const void* U;            // (B, D, V, N, N) complex: links
    const void* weights;      // NULL or (BW, P, V) real: plaquette weights, plane p = nu (nu - 1) / 2 + mu
    const void* beta;         // (B or 1,) real
    const void* mom;          // NULL (write F) or (B, D, V, N, N) complex momenta: write P - coef F
    void* out;                // (B, D, V, N, N) complex
    void* reserved_ptr;
    double coef;              // kick coefficient (with mom)
    double reserved;
    int64_t batch, volume;
    int64_t extent[MD_MAX_D], stride[MD_MAX_D];
    int64_t weights_bstride, beta_bstride;  // per-sample strides in real elements (0: shared)
    int32_t D, N, dtype, device;
};

template <typename T, int D, int N, bool WEIGHTED, bool KICK>
MD_HD void md_gauge_link(const NqGaugeArgs& a, int64_t tid) {
    typedef MdC<T> C;
    constexpr int NN = N * N;
    const int64_t V = a.volume;
    const int64_t x = tid % V;
    const int64_t bm = tid / V;
    const int mu = (int)(bm % D);
    const int64_t b = bm / D;
    const C* Ub = static_cast<const C*>(a.U) + b * ((int64_t)D * V * NN);
    const T* w = WEIGHTED ? static_cast<const T*>(a.weights) + b * a.weights_bstride : (const T*)0;
    C A[NN], t1[NN], t2[NN], L1[NN], L2[NN];
    MD_UNROLL
    for (int e = 0; e < NN; ++e) A[e] = c_make<T>(0, 0);
    MD_UNROLL
    for (int nu = 0; nu < D; ++nu) {
        if (nu == mu) continue;
        const int lo = mu < nu ? mu : nu, hi = mu < nu ? nu : mu;
        const int64_t plane = (int64_t)(hi * (hi - 1) / 2 + lo) * V;
        const int64_t xpmu = md_fwd(x, mu, a.extent, a.stride);
        const int64_t xpnu = md_fwd(x, nu, a.extent, a.stride);
        const int64_t xmnu = md_bwd(x, nu, a.extent, a.stride);
        const int64_t xpmumnu = md_bwd(xpmu, nu, a.extent, a.stride);
        // up = U_nu(x + mu) U_mu(x + nu)^dag U_nu(x)^dag
        m_load<T, N>(L1, Ub + ((int64_t)nu * V + xpmu) * NN);
        m_load<T, N>(L2, Ub + ((int64_t)mu * V + xpnu) * NN);
        m_mul<T, N, false, true>(t1, L1, L2);
        m_load<T, N>(L1, Ub + ((int64_t)nu * V + x) * NN);
        m_mul<T, N, false, true>(t2, t1, L1);
        m_accumulate<T, N, WEIGHTED>(A, t2, WEIGHTED ? w[plane + x] : (T)1);
        // down = U_nu(x + mu - nu)^dag U_mu(x - nu)^dag U_nu(x - nu)
        m_load<T, N>(L1, Ub + ((int64_t)nu * V + xpmumnu) * NN);
        m_load<T, N>(L2, Ub + ((int64_t)mu * V + xmnu) * NN);
        m_mul<T, N, true, true>(t1, L1, L2);
        m_load<T, N>(L1, Ub + ((int64_t)nu * V + xmnu) * NN);
        m_mul<T, N, false, false>(t2, t1, L1);
        m_accumulate<T, N, WEIGHTED>(A, t2, WEIGHTED ? w[plane + xmnu] : (T)1);
    }
    // Omega = U_mu(x) A; Z = Omega - Omega^dag minus its trace / N; F = -i beta / (2N) Z
    m_load<T, N>(L1, Ub + ((int64_t)mu * V + x) * NN);
    m_mul<T, N, false, false>(t1, L1, A);
    MD_UNROLL
    for (int i = 0; i < N; ++i) {
        MD_UNROLL
        for (int j = 0; j < N; ++j) t2[N * i + j] = c_sub(t1[N * i + j], c_conj(t1[N * j + i]));
    }
    C tr = t2[0];
    MD_UNROLL
    for (int i = 1; i < N; ++i) tr = c_add(tr, t2[(N + 1) * i]);
    const C trN = c_make<T>(tr.re / (T)N, tr.im / (T)N);
    const T s = ((T)0.5 / (T)N) * static_cast<const T*>(a.beta)[b * a.beta_bstride];
    C* out = static_cast<C*>(a.out) + tid * NN;
    const C* P = KICK ? static_cast<const C*>(a.mom) + tid * NN : (const C*)0;
    MD_UNROLL
    for (int e = 0; e < NN; ++e) {
        const C z = (e % (N + 1) == 0) ? c_sub(t2[e], trN) : t2[e];
        const C f = c_make<T>(s * z.im, -(s * z.re));  // (-i s) z
        out[e] = KICK ? c_sub(P[e], c_scale((T)a.coef, f)) : f;
    }
}

// ---- launch helpers (CUDA kernel, or a host loop over the same thread index) ---------------------
#ifdef __CUDACC__
template <typename T, int D, int N, bool WEIGHTED, bool KICK>
__global__ void md_gauge_kernel(const MD_GRID_CONSTANT NqGaugeArgs a, int64_t total) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) md_gauge_link<T, D, N, WEIGHTED, KICK>(a, tid);
}

template <typename T, int D, int N, bool WEIGHTED, bool KICK>
static int md_gauge_launch(const NqGaugeArgs& a, void* stream) {
    const int64_t total = a.batch * (int64_t)D * a.volume;
    if (total == 0) return 0;
    const int64_t blocks = (total + MD_THREADS - 1) / MD_THREADS;
    md_gauge_kernel<T, D, N, WEIGHTED, KICK><<<(unsigned)blocks, MD_THREADS, 0, (cudaStream_t)stream>>>(a, total);
    return (int)cudaGetLastError();
}
#else
template <typename T, int D, int N, bool WEIGHTED, bool KICK>
static int md_gauge_launch(const NqGaugeArgs& a, void* stream) {
    (void)stream;
    const int64_t total = a.batch * (int64_t)D * a.volume;
    for (int64_t tid = 0; tid < total; ++tid) md_gauge_link<T, D, N, WEIGHTED, KICK>(a, tid);
    return 0;
}
#endif

template <typename T, int D, int N>
static int md_gauge_modes(const NqGaugeArgs& a, void* stream) {
    const bool weighted = a.weights != 0, kick = a.mom != 0;
    if (weighted) return kick ? md_gauge_launch<T, D, N, true, true>(a, stream) : md_gauge_launch<T, D, N, true, false>(a, stream);
    return kick ? md_gauge_launch<T, D, N, false, true>(a, stream) : md_gauge_launch<T, D, N, false, false>(a, stream);
}

static int md_gauge_dispatch(const NqGaugeArgs& a, void* stream) {
    if (a.dtype != 0) return -1;  // complex128 only
#define MD_CASE(DD, NN) if (a.D == DD && a.N == NN) return md_gauge_modes<double, DD, NN>(a, stream);
    MD_CASE(4, 3) MD_CASE(4, 2) MD_CASE(2, 3) MD_CASE(2, 2)
#undef MD_CASE
    return -1;
}

// ---- entry points ----------------------------------------------------------------------------------
// The caller (fused_hmc._launch) makes the tensors' device current through torch, so the current
// device normally equals `device` and nothing is switched (as nq_hop in fused.py: this library links
// its own static CUDA runtime, and switching to a device whose context torch never created would
// create one behind torch's back). Only when they differ is the device set for the launch and set
// back, both return codes checked.
#ifdef __CUDACC__
#define MD_DEVICE_GUARD_BEGIN(dev) \
    int current = -1; int err = (int)cudaGetDevice(&current); if (err) return err; \
    const int switched = (current != (dev)); \
    if (switched && (err = (int)cudaSetDevice(dev))) return err;
#define MD_DEVICE_GUARD_END(code) \
    if (switched) { err = (int)cudaSetDevice(current); if ((code) == 0 && err) (code) = err; }
#else
#define MD_DEVICE_GUARD_BEGIN(dev) (void)(dev);
#define MD_DEVICE_GUARD_END(code) (void)(code);
#endif

extern "C" {
int nq_md_is_cuda(void) {
#ifdef __CUDACC__
    return 1;
#else
    return 0;
#endif
}

int nq_gauge_args_size(void) { return (int)sizeof(NqGaugeArgs); }

// Returns 0 on success, -1 for an unsupported (dtype, D, N), a cudaError_t code otherwise.
int nq_gauge_force(const NqGaugeArgs* a, void* stream) {
    MD_DEVICE_GUARD_BEGIN(a->device)
    int code = md_gauge_dispatch(*a, stream);
    MD_DEVICE_GUARD_END(code)
    return code;
}

const char* nq_md_error_string(int code) {
    if (code == -1) return "unsupported (dtype, D, N)";
#ifdef __CUDACC__
    return cudaGetErrorString((cudaError_t)code);
#else
    return code == 0 ? "no error" : "unknown error";
#endif
}
}
"""


# ---- ctypes mirrors of the argument structures -------------------------------------------------
class _GaugeArgs(ctypes.Structure):
    _fields_ = [
        ("U", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("beta", ctypes.c_void_p),
        ("mom", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("reserved_ptr", ctypes.c_void_p),
        ("coef", ctypes.c_double),
        ("reserved", ctypes.c_double),
        ("batch", ctypes.c_int64),
        ("volume", ctypes.c_int64),
        ("extent", ctypes.c_int64 * _MAX_D),
        ("stride", ctypes.c_int64 * _MAX_D),
        ("weights_bstride", ctypes.c_int64),
        ("beta_bstride", ctypes.c_int64),
        ("D", ctypes.c_int32),
        ("N", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("device", ctypes.c_int32),
    ]


_ENTRY_POINTS = (
    # (function, argument structure, size function)
    ("nq_gauge_force", _GaugeArgs, "nq_gauge_args_size"),
)


class _MDBackend(_fused._Backend):
    """The loaded molecular-dynamics library: `call(name, args, stream)` raises RuntimeError on a
    nonzero return code. Built, cached and loaded by `fused.load_backend(..., backend_class=_MDBackend)`."""

    SOURCE = MD_KERNEL_SOURCE
    NAME = "neoqcd_md"

    def __init__(self, target, path):
        self.target = target
        self.path = str(path)
        self.rebuilt_from = None
        lib = ctypes.CDLL(self.path)
        lib.nq_md_is_cuda.restype = ctypes.c_int
        lib.nq_md_error_string.argtypes = [ctypes.c_int]
        lib.nq_md_error_string.restype = ctypes.c_char_p
        for name, struct, size_name in _ENTRY_POINTS:
            size = getattr(lib, size_name)
            size.restype = ctypes.c_int
            if size() != ctypes.sizeof(struct):
                raise _fused.BackendUnavailable(f"argument layout mismatch ({name}): library {size()} bytes, "
                                                f"ctypes {ctypes.sizeof(struct)}")
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.POINTER(struct), ctypes.c_void_p]
            fn.restype = ctypes.c_int
        if bool(lib.nq_md_is_cuda()) != (target == "cuda"):
            raise _fused.BackendUnavailable(f"library {self.path} is not a {target} build")
        self._lib = lib

    def call(self, name, args, stream):
        code = getattr(self._lib, name)(ctypes.byref(args), stream)
        if code != 0:
            raise RuntimeError(f"neoqcd fused MD kernel {name} ({self.target}) failed: code {code}: "
                               f"{self._lib.nq_md_error_string(code).decode(errors='replace')}")


def load_md_backend(target, device=None):
    """The MD kernel library for target "cuda" or "cpu" (`fused.load_backend` with `_MDBackend`:
    built at first use, cached per process and on disk; BackendUnavailable with the reason)."""
    return _fused.load_backend(target, device, _MDBackend)


def md_backend_status(target, device=None):
    """(available, library path or the reason why not)."""
    return _fused.backend_status(target, device, _MDBackend)


# ---- helpers -------------------------------------------------------------------------------------
_raw = _fused._raw


def _device_index(device):
    if device.type != "cuda":
        return 0
    return device.index if device.index is not None else torch.cuda.current_device()


def _launch(backend, name, args, device):
    """Run entry point `name` on the tensors' device: torch's current stream of that device, made
    current through torch first when it is not (as `FusedHalfLatticeWilson._hop_kernel`)."""
    if device.type != "cuda":
        backend.call(name, args, None)
    elif torch.cuda.current_device() == device.index:
        backend.call(name, args, torch.cuda.current_stream(device).cuda_stream)
    else:
        with torch.cuda.device(device):
            backend.call(name, args, torch.cuda.current_stream(device).cuda_stream)


def _refuse_autograd(what, **inputs):
    """The kernels read raw memory: with grad mode on, an input that requires grad is an error
    rather than a silently detached result."""
    if not torch.is_grad_enabled():
        return
    needs = [name for name, v in inputs.items() if torch.is_tensor(v) and v.requires_grad]
    if needs:
        raise RuntimeError(f"{what} (fused MD kernel) is not differentiable, but {', '.join(needs)} require(s) "
                           "grad with grad mode on: use the torch path or call it under torch.no_grad()")


def _lattice_args(args, geometry):
    lattice = tuple(geometry.lattice)
    stride = 1
    strides = [0] * len(lattice)
    for mu in range(len(lattice) - 1, -1, -1):
        strides[mu] = stride
        stride *= lattice[mu]
    for mu in range(_MAX_D):
        args.extent[mu] = lattice[mu] if mu < len(lattice) else 1
        args.stride[mu] = strides[mu] if mu < len(lattice) else 1


def _target_of(device):
    return "cuda" if torch.device(device).type == "cuda" else "cpu"


def kernel_supports(geometry, dtype=torch.complex128):
    """None if the force / update kernels support this geometry and dtype, else the reason."""
    if geometry.D not in _SUPPORTED_D or geometry.N not in _SUPPORTED_N:
        return f"the MD kernels support D in {_SUPPORTED_D}, N in {_SUPPORTED_N}; got D={geometry.D}, N={geometry.N}"
    if dtype != torch.complex128:
        return f"the force and update kernels are complex128 only (molecular dynamics in double), got {dtype}"
    return None


# ---- gauge force -----------------------------------------------------------------------------------
def _beta_tensor(beta, B, device):
    t = torch.as_tensor(beta, dtype=torch.float64).to(device).reshape(-1).contiguous()
    if t.numel() not in (1, B):
        raise ValueError(f"beta of {t.numel()} values for a batch of {B}")
    return t


def plaquette_weight_tensor(weights, geometry, B, device):
    """The plaquette weights {(mu, nu): field} of `WilsonGaugeAction` as one (BW, P, V) float64
    tensor (plane p = nu (nu - 1) / 2 + mu, mu < nu; BW = 1 when no field has a batch axis, else B),
    or None without weights."""
    if weights is None:
        return None
    D, V = geometry.D, geometry.volume
    lattice = tuple(geometry.lattice)
    fields = []
    for nu in range(1, D):
        for mu in range(nu):
            w = torch.as_tensor(weights[(mu, nu)]).to(device=device, dtype=torch.float64)
            if tuple(w.shape) == lattice:
                w = w.reshape(1, V)
            elif w.dim() == len(lattice) + 1 and tuple(w.shape[1:]) == lattice:
                w = w.reshape(w.shape[0], V)
            else:
                raise ValueError(f"plaquette weight ({mu}, {nu}) of shape {tuple(w.shape)} for lattice {lattice}")
            fields.append(w)
    BW = max(w.shape[0] for w in fields)
    if BW not in (1, B) or any(w.shape[0] not in (1, BW) for w in fields):
        raise ValueError(f"plaquette weights with batch sizes {[w.shape[0] for w in fields]} for a batch of {B}")
    return torch.stack([w.expand(BW, V) for w in fields], dim=1).contiguous()


def gauge_force(U, beta, geometry, weights=None, mom=None, coef=0.0, backend=None):
    """The force of `WilsonGaugeAction` (beta, plaquette weights as a `plaquette_weight_tensor`) through
    the kernel; with `mom`, the kicked momenta mom - coef F instead (F never stored). U complex128
    (B, D, lattice, N, N) on the device of `backend` (default: the library of U's device)."""
    _refuse_autograd("gauge_force", U=U, mom=mom)
    reason = kernel_supports(geometry, U.dtype)
    if reason is not None:
        raise ValueError(reason)
    dev = U.device
    backend = load_md_backend(_target_of(dev), dev) if backend is None else backend
    B = U.shape[0]
    if tuple(U.shape) != tuple(geometry.link_shape(B)):
        raise ValueError(f"links of shape {tuple(U.shape)} do not match {geometry}")
    U = _raw(U)
    beta_t = _beta_tensor(beta, B, dev)
    a = _GaugeArgs()
    a.U = U.data_ptr()
    a.beta = beta_t.data_ptr()
    a.beta_bstride = 0 if beta_t.numel() == 1 else 1
    if weights is not None:
        if weights.dtype != torch.float64 or weights.device != dev or weights.shape[0] not in (1, B):
            raise ValueError("weights: a (1 or B, P, V) float64 tensor on the links' device")
        weights = _raw(weights)
        a.weights = weights.data_ptr()
        a.weights_bstride = 0 if weights.shape[0] == 1 else weights[0].numel()
    if mom is not None:
        if mom.shape != U.shape or mom.dtype != U.dtype or mom.device != dev:
            raise ValueError("mom must have the shape, dtype and device of the links")
        mom = _raw(mom)
        a.mom = mom.data_ptr()
        a.coef = float(coef)
    out = torch.empty_like(U, memory_format=torch.contiguous_format)
    a.out = out.data_ptr()
    a.batch, a.volume = B, geometry.volume
    _lattice_args(a, geometry)
    a.D, a.N, a.dtype, a.device = geometry.D, geometry.N, 0, _device_index(dev)
    _launch(backend, "nq_gauge_force", a, dev)
    return out


class _KernelChoice:
    """Lazy per-device choice between a compiled MD kernel and the torch code (module docstring)."""

    def __init__(self, target, strict, what):
        if target is not None and target not in MD_TARGETS:
            raise ValueError(f"target must be one of {MD_TARGETS} or None, got {target!r}")
        self.target = target
        self.strict = bool(strict)
        self.what = what
        self._backends = {}
        self.last_used = None  # "cuda", "cpu" or "torch": the path of the last call

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_backends"] = {}  # ctypes handles: re-acquired at the first call
        return state

    def backend(self, device, geometry, dtype):
        """The loaded library for tensors on `device` of `dtype`, or None (torch code)."""
        if self.target is None:
            return None
        device = torch.device(device)
        key = (str(device), dtype)
        if key in self._backends:
            return self._backends[key]
        reason = None
        if _target_of(device) != self.target:
            reason = f"the {self.target} kernel needs {'CUDA' if self.target == 'cuda' else 'CPU'} tensors, got {device}"
        else:
            reason = kernel_supports(geometry, dtype)
        backend = None
        if reason is None:
            ok, why = md_backend_status(self.target, device)
            if ok:
                backend = load_md_backend(self.target, device)
            else:
                reason = why
        if reason is not None:
            if self.strict:
                raise _fused.BackendUnavailable(f"{self.what}: {reason}")
            _fused._warn_once(f"neoqcd fused MD kernel ({self.what}): {self.target} unavailable ({reason}); "
                              "using the torch code")
        self._backends[key] = backend
        return backend


class FusedWilsonGaugeAction(WilsonGaugeAction):
    """`WilsonGaugeAction` whose force runs through the gauge-force kernel (module docstring).

    Extra args: `target` "cuda", "cpu" (the host build of the same source) or None (torch: this is
    then the parent class exactly), `strict` (raise BackendUnavailable instead of falling back to
    the torch force). Action, action difference and staple sums are the parent's. The plaquette
    weights (any of the parent's layouts) are packed once per device and weights dict
    (`plaquette_weight_tensor`). `force_kick(U, P, c)` returns P - c F(U) in one pass.
    The path of the last force is the attribute `kernel_used` ("cuda", "cpu" or "torch").
    """

    def __init__(self, beta, geometry, plaquette_weights=None, target=None, strict=False):
        super().__init__(beta, geometry, plaquette_weights)
        self._choice = _KernelChoice(target, strict, "gauge force")
        self._weights_cache = None

    @property
    def kernel_used(self):
        return self._choice.last_used

    def _packed_weights(self, U):
        w = self.plaquette_weights
        if w is None:
            return None
        key = (id(w), tuple((k, id(v), getattr(v, "_version", None)) for k, v in sorted(w.items())),
               str(U.device), U.shape[0])
        c = self._weights_cache
        if c is not None and c[0] == key:
            return c[1]
        packed = plaquette_weight_tensor(w, self.geometry, U.shape[0], U.device)
        self._weights_cache = (key, packed)
        return packed

    def _backend(self, U):
        return self._choice.backend(U.device, self.geometry, U.dtype)

    def force(self, U):
        backend = self._backend(U)
        if backend is None:
            self._choice.last_used = "torch"
            return super().force(U)
        self._choice.last_used = backend.target
        return gauge_force(U, self.beta, self.geometry, self._packed_weights(U), backend=backend)

    def force_kick(self, U, P, coefficient):
        """P - coefficient * force(U), with the kick fused into the force kernel when it runs."""
        backend = self._backend(U)
        if backend is None or not (P.dtype == U.dtype and P.shape == U.shape and P.device == U.device):
            self._choice.last_used = "torch"
            return P - coefficient * super().force(U)
        self._choice.last_used = backend.target
        return gauge_force(U, self.beta, self.geometry, self._packed_weights(U), mom=P, coef=coefficient,
                           backend=backend)
