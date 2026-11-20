"""Fused CUDA kernel of the heat bath + overrelaxation sweep (mcmc.HBOR) of SU(3) in D = 4.

Ported from snf_SUN, branch perf-main at a3bfc7b (snf_code/fused.py), with the changes that
O-REX/NEO-REX need (listed below). The file keeps the snf_SUN name and layout, so that the two can be
compared file by file.

With HBOR.use_fused_kernels (main_pt_obc.py --fused-kernels) the sweep of HBOR.__call__ runs through
one compiled kernel per masked block: each thread reads one link and the 18 links of its 6 staples
once, builds the staple sum (with the defect weights), does the Cabibbo-Marinari heat bath or
overrelaxation of its 3 SU(2) subgroups in registers and writes the link once. The torch sweep makes
thousands of small elementwise kernels per sweep, memory bound, plus the gathers of the staples.

Same mathematics, same random numbers: the kernel consumes rand[b, 6 phb + r, mu, x] (the tensor of
HBOR._rand_shape, drawn as HBOR.__call__ draws it) for the subgroup phb of the link U_mu(x), as
HBOR._su3_selected_hb does, with the operations of HBOR in the same order where it matters. Results
agree with the torch sweep to float64 rounding (test/test_fused.py); the update amplifies rounding, so
runs with and without the kernel are statistically, not bitwise, equivalent. A heat-bath acceptance
decided the other way by a rounding difference (probability ~1e-15 per decision) gives an O(1)
difference on that link: an equally valid update of the chain.

Changes with respect to snf_SUN:
- the link factors kappa of the defect may differ per chain: (B, D V), or (1, D V) shared (the
  per-chain parameter of the batched O-REX/NEO-REX sweeps, HBOR._sweep_kappa);
- beta may differ per chain: (B,) or one value (HBOR._beta_for_staples);
- the reverse sweep of the symmetric updates (HBOR.__call__(reverse=True)): the overrelaxation sweeps
  first, then the heat bath, each over the blocks backwards, and in every link the SU(2) subgroups in
  the order 2, 1, 0;
- a vanishing staple sum (k = 0: the links of a defect at bc = 0) gives an SU(2) element from the Haar
  measure out of the first four random numbers, always accepted (mcmc._su2_haar_from_uniform), with or
  without the defect, as HBOR.heatbath_standard does; snf_SUN applies the acceptance test there;
- the weights of the lower staples are multiplied in the order of HBOR._gathered_staples;
- after the kernel the updated links are re-unitarized by torch, as HBOR._sweep does;
- snf_init, called by load_backend once per device: initializes the CUDA runtime of the library (nvcc links
  it statically) and loads the kernels there (cudaFuncGetAttributes), so that nothing is loaded lazily
  (CUDA_MODULE_LOADING=LAZY) at a later launch, which may be inside a CUDA graph capture of the caller
  (the smearing kernels of fused_smearing.py, whose library has the same hook, snf_stout_init).

The kernel runs only without gradients, for SU(3) in D = 4, complex128, on CUDA, or on the CPU with
HBOR.fused_emulate (the host build of the same source: the CPU emulation the tests compare with HBOR
on a laptop). Otherwise, or when the kernel cannot be built or loaded, HBOR's own torch sweep runs
(with one RuntimeWarning per reason; HBOR.fused_strict raises BackendUnavailable instead).
HBOR.fused_used records the path of the last call: 'cuda', 'cpu' (the emulation) or 'torch'.

Kernel (KERNEL_SOURCE). One thread per link (sample b, site x of mask m = eo + 2 mu), 128 threads per
block, one launch per mask, all launched by one call into the library (snf_hbor). The links of a mask
are updated in place: none of them is a staple link of another link of the same mask (checked once
per sun.MaskedLinks). Neighbours are computed in the kernel from the site index and the extents of
the lattice, periodic, with the shifts of sun.MaskedLinks (torch.roll(ids, 1, mu)[x] is the site
"x + mu"), so the same kernel serves the whole lattice and the protocol patch (whose masks list the
sites of the patch only). Only MaskedLinks.sites (one int64 per link) is read from the tables.
Layout: the configurations as they are, (B, D, T, L, L, L, 3, 3) complex128, contiguous.

Backends. Plain CUDA C compiled with nvcc into a shared library with an extern "C" entry point, called
through ctypes with raw device pointers and the current torch stream: no torch headers, no ninja,
seconds of nvcc. Libraries are cached in $NEOQCD_KERNEL_CACHE (default ~/.cache/neoqcd/kernels), one
per source hash, compiler and GPU architecture, written under a temporary name and renamed atomically,
with the compiler output (nvcc -Xptxas -v: registers and spills) in a .log beside it. nvcc is looked
up in $CUDA_HOME/bin, $CUDA_PATH/bin, then PATH; the host compiler in $CXX, then c++, g++, clang++.
"""

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import warnings
import weakref
from pathlib import Path

import torch

import neoqcd.sun_utils as sun

KERNEL_SOURCE = r"""
// neoqcd: heat bath + overrelaxation of mcmc.HBOR, SU(3) in D = 4, one source for CUDA (nvcc) and the
// host C++ compiler (CPU emulation). Ported from snf_SUN (snf_code/fused.py); conventions:
// src/neoqcd/fused.py.
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define SNF_HD __host__ __device__ __forceinline__
#define SNF_UNROLL _Pragma("unroll")
#else
#define SNF_HD static inline
#define SNF_UNROLL
#endif

#define SNF_D 4          // dimensions
#define SNF_MASKS 8      // masks m = eo + 2 mu
#define SNF_NRAND 18     // random numbers per link: 6 per SU(2) subgroup of the heat bath
#define SNF_THREADS 128  // threads per block

// Arguments of one sweep (mirrored by the ctypes structure _Args in fused.py; the host checks the
// size through snf_args_size()).
struct SnfArgs {
    void* links;                      // (B, D, V, 3, 3) complex: the configurations, updated in place
    const void* rand;                 // (B, 18, D, V) real: the random numbers of HBOR.__call__
    const void* kappa;                // (B or 1, D, V) real: link factors kappa_mu(x), else NULL
    const void* beta;                 // (B or 1,) real: the coupling
    const int64_t* sites[SNF_MASKS];  // sites x of the links U_mu(x) of mask m (MaskedLinks.sites)
    int64_t n[SNF_MASKS];             // number of links of mask m
    int64_t batch, volume;            // B, V
    int64_t extent[SNF_D];            // extents (T, L, L, L) of the lattice of the field
    int64_t kappa_bstride, beta_bstride;  // per-sample strides: 0 (shared) or D V, 1
    int32_t orsteps, has_defect, reverse, dtype, device, pad;  // dtype 0: complex128
};

// aligned as a pair: one 16-byte load per complex128
template <typename T> struct alignas(2 * sizeof(T)) SnfC { T re, im; };

template <typename T> SNF_HD SnfC<T> c_make(T re, T im) { SnfC<T> z; z.re = re; z.im = im; return z; }
template <typename T> SNF_HD SnfC<T> c_add(SnfC<T> a, SnfC<T> b) { return c_make<T>(a.re + b.re, a.im + b.im); }
template <typename T> SNF_HD SnfC<T> c_sub(SnfC<T> a, SnfC<T> b) { return c_make<T>(a.re - b.re, a.im - b.im); }
template <typename T> SNF_HD SnfC<T> c_conj(SnfC<T> a) { return c_make<T>(a.re, -a.im); }
template <typename T> SNF_HD SnfC<T> c_neg(SnfC<T> a) { return c_make<T>(-a.re, -a.im); }
// the product of c10::complex
template <typename T> SNF_HD SnfC<T> c_mul(SnfC<T> a, SnfC<T> b) {
    return c_make<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
// real times complex (torch promotes the real factor to complex: the same numbers)
template <typename T> SNF_HD SnfC<T> c_scale(T s, SnfC<T> a) { return c_make<T>(s * a.re, s * a.im); }

template <typename T> SNF_HD void m_load(SnfC<T> M[9], const SnfC<T>* p) {
    SNF_UNROLL
    for (int e = 0; e < 9; ++e) M[e] = p[e];
}

// out = op(A) op(B), op the dagger when DA (DB) is set: out_ik = sum_j op(A)_ij op(B)_jk accumulated
// over j = 0, 1, 2 as sun.SUN_mul does
template <typename T, bool DA, bool DB>
SNF_HD void m_mul(SnfC<T> out[9], const SnfC<T> A[9], const SnfC<T> B[9]) {
    SNF_UNROLL
    for (int i = 0; i < 3; ++i) {
        SNF_UNROLL
        for (int k = 0; k < 3; ++k) {
            SnfC<T> s = c_make<T>(0, 0);
            SNF_UNROLL
            for (int j = 0; j < 3; ++j) {
                const SnfC<T> a = DA ? c_conj(A[3 * j + i]) : A[3 * i + j];
                const SnfC<T> b = DB ? c_conj(B[3 * k + j]) : B[3 * j + k];
                s = j == 0 ? c_mul(a, b) : c_add(s, c_mul(a, b));
            }
            out[3 * i + k] = s;
        }
    }
}

// S += K Q (staples + weight * upper in HBOR._gathered_staples; K = 1 without the defect)
template <typename T, bool DEFECT>
SNF_HD void m_accumulate(SnfC<T> S[9], const SnfC<T> Q[9], T K) {
    SNF_UNROLL
    for (int e = 0; e < 9; ++e) S[e] = c_add(S[e], DEFECT ? c_scale(K, Q[e]) : Q[e]);
}

// (U S)_{pq} for rows p and columns q of the subgroup (mcmc.SU3_compute_RK)
template <typename T> SNF_HD SnfC<T> m_dot(const SnfC<T> U[9], int p, const SnfC<T> S[9], int q) {
    SnfC<T> s = c_mul(U[3 * p], S[q]);
    s = c_add(s, c_mul(U[3 * p + 1], S[3 + q]));
    return c_add(s, c_mul(U[3 * p + 2], S[6 + q]));
}

// HBOR.heatbath_standard of the subgroup matrix R = (sa, sb) = k V: the SU(2) element (c0, c1) that
// left-multiplies the link, returned with its acceptance. q[r * stride], r < 6: the random numbers of
// the subgroup. A vanishing staple sum (k = 0) gives the Haar element of mcmc._su2_haar_from_uniform,
// always accepted.
template <typename T>
SNF_HD bool snf_heatbath(SnfC<T> sa, SnfC<T> sb, const T* q, int64_t stride, T beta, SnfC<T>* c0, SnfC<T>* c1) {
    const T two_pi = (T)6.283185307179586;  // 2.0 * torch.pi
    const T k = sqrt(((sa.re * sa.re + sa.im * sa.im) + sb.re * sb.re) + sb.im * sb.im);
    const T q0 = q[0], q1 = q[stride], q2 = q[2 * stride], q3 = q[3 * stride], q4 = q[4 * stride], q5 = q[5 * stride];
    if (!(k > 0)) {
        const T tiny = (T)2.2250738585072014e-308;  // torch.finfo(torch.float64).tiny
        const T u0 = q0 > tiny ? q0 : tiny, u2 = q2 > tiny ? q2 : tiny;
        const T r0 = sqrt((T)-2 * log(u0)), r1 = sqrt((T)-2 * log(u2));
        const T g0 = r0 * cos(two_pi * q1), g1 = r0 * sin(two_pi * q1);
        const T g2 = r1 * cos(two_pi * q3), g3 = r1 * sin(two_pi * q3);
        const T norm = sqrt(((g0 * g0 + g1 * g1) + g2 * g2) + g3 * g3);
        *c0 = c_make<T>(g0 / norm, g1 / norm);
        *c1 = c_make<T>(g2 / norm, g3 / norm);
        return true;
    }
    // xi = (N / 2.) / (beta k): float / tensor is Tensor.__rtruediv__, reciprocal(beta k) * 1.5
    const T xi = ((T)1 / (beta * k)) * (T)1.5;
    const T cs = cos(two_pi * ((T)1 - q1));
    const T lam2 = ((T)-0.5 * xi) * (log((T)1 - q0) + (cs * cs) * log((T)1 - q2));
    // rejected: the link is kept (HBOR computes the element anyway and discards it)
    if (!((T)1 - lam2 >= q3 * q3)) return false;
    const T x0 = (T)1 - (T)2 * lam2;
    const T ct = (T)1 - (T)2 * q4;
    const T stheta = sqrt((T)1 - ct * ct);
    const T mod = sqrt((T)1 - x0 * x0);
    const SnfC<T> A = c_make<T>(x0, mod * ct);
    const T sm = stheta * mod;
    const SnfC<T> B = c_make<T>(sm * cos(two_pi * q5), sm * sin(two_pi * q5));
    // (A, B) V^dagger, V = R / k
    const SnfC<T> v0 = c_conj(sa), v1 = c_neg(sb);
    const T scale = (T)1 / k;
    *c0 = c_scale(scale, c_sub(c_mul(A, v0), c_mul(B, c_conj(v1))));
    *c1 = c_scale(scale, c_add(c_mul(A, v1), c_mul(B, c_conj(v0))));
    return true;
}

// HBOR._su3_selected_over on one subgroup: (c0, c1) = (R^dagger / k)^2, not applied when k = 0
template <typename T>
SNF_HD bool snf_overrelax(SnfC<T> sa, SnfC<T> sb, SnfC<T>* c0, SnfC<T>* c1) {
    const T k2 = ((sa.re * sa.re + sa.im * sa.im) + sb.re * sb.re) + sb.im * sb.im;
    if (!(k2 > 0)) return false;
    const SnfC<T> v0 = c_conj(sa), v1 = c_neg(sb);
    const T scale = (T)1 / k2;
    *c0 = c_scale(scale, c_sub(c_mul(v0, v0), c_mul(v1, c_conj(v1))));
    *c1 = c_scale(scale, c_add(c_mul(v0, v1), c_mul(v1, c_conj(v0))));
    return true;
}

// Wilson cooling of the subgroup (topology.cool_torch): the SU(2) element V = W^dagger, W = R / k, that
// maximizes Re Tr(V R); no change for k = 0
template <typename T>
SNF_HD bool snf_cool(SnfC<T> sa, SnfC<T> sb, SnfC<T>* c0, SnfC<T>* c1) {
    const T k2 = ((sa.re * sa.re + sa.im * sa.im) + sb.re * sb.re) + sb.im * sb.im;
    if (!(k2 > 0)) return false;
    const T scale = (T)1 / sqrt(k2);
    *c0 = c_scale(scale, c_conj(sa));
    *c1 = c_scale(scale, c_neg(sb));
    return true;
}

// One link: sample b, site x = sites[i] of the mask of direction mu, tid = b n + i. MODE 0: heat
// bath, 1: overrelaxation, 2: cooling.
template <typename T, int MODE, bool DEFECT>
SNF_HD void snf_link(const SnfArgs& a, int mu, const int64_t* sites, int64_t n, int64_t tid) {
    typedef SnfC<T> C;
    const int64_t b = tid / n;
    const int32_t x = (int32_t)sites[tid - b * n];
    const int32_t V = (int32_t)a.volume;
    C* field = static_cast<C*>(a.links) + b * ((int64_t)SNF_D * V * 9);

    // Neighbours as in sun.MaskedLinks: torch.roll(ids, 1, axis d)[x] = "x + d" is the site with
    // coordinate c_d - 1, torch.roll(ids, -1, axis d)[x] = "x - d" the one with c_d + 1, periodic
    int32_t fwd[SNF_D], bwd[SNF_D];
    int32_t stride = 1;
    SNF_UNROLL
    for (int d = SNF_D - 1; d >= 0; --d) {
        const int32_t L = (int32_t)a.extent[d];
        const int32_t c = (x / stride) % L;
        fwd[d] = c == 0 ? (L - 1) * stride : -stride;
        bwd[d] = c == L - 1 ? -(L - 1) * stride : stride;
        stride *= L;
    }
    const int32_t fwd_mu = mu == 0 ? fwd[0] : mu == 1 ? fwd[1] : mu == 2 ? fwd[2] : fwd[3];
    const T* kappa = DEFECT ? static_cast<const T*>(a.kappa) + b * a.kappa_bstride : (const T*)0;
    const T* kmu = DEFECT ? kappa + (int64_t)mu * V : (const T*)0;

    // Staple sum (HBOR._gathered_staples, sun.staples_at): for nu != mu in increasing order,
    // upper(x) = U_nu(x+mu) U_mu^dagger(x+nu) U_nu^dagger(x), weight
    // kappa_mu(x) kappa_nu(x+mu) kappa_mu(x+nu) kappa_nu(x), and
    // lower(x) = U_nu^dagger(y+mu) U_mu^dagger(y) U_nu(y), y = x - nu, weight
    // kappa_mu(x) kappa_nu(y+mu) kappa_mu(y) kappa_nu(y), multiplied left to right
    C S[9];
    SNF_UNROLL
    for (int e = 0; e < 9; ++e) S[e] = c_make<T>(0, 0);
    SNF_UNROLL
    for (int nu = 0; nu < SNF_D; ++nu) {
        if (nu == mu) continue;
        const int32_t y = x + bwd[nu];
        T K_upper = 1, K_lower = 1;
        if (DEFECT) {
            const T* knu = kappa + (int64_t)nu * V;
            K_upper = ((kmu[x] * knu[x + fwd_mu]) * kmu[x + fwd[nu]]) * knu[x];
            K_lower = ((kmu[x] * knu[y + fwd_mu]) * kmu[y]) * knu[y];
        }
        C A[9], B[9], P[9];
        m_load(A, field + ((int64_t)nu * V + x + fwd_mu) * 9);
        m_load(B, field + ((int64_t)mu * V + x + fwd[nu]) * 9);
        m_mul<T, false, true>(P, A, B);
        m_load(A, field + ((int64_t)nu * V + x) * 9);
        m_mul<T, false, true>(B, P, A);
        m_accumulate<T, DEFECT>(S, B, K_upper);
        m_load(A, field + ((int64_t)nu * V + y + fwd_mu) * 9);
        m_load(B, field + ((int64_t)mu * V + y) * 9);
        m_mul<T, true, true>(P, A, B);
        m_load(A, field + ((int64_t)nu * V + y) * 9);
        m_mul<T, false, false>(B, P, A);
        m_accumulate<T, DEFECT>(S, B, K_lower);
    }

    // The 3 Cabibbo-Marinari subgroups in turn (0, 1, 2, or 2, 1, 0 in the reverse sweep), each on
    // the link updated by the previous ones; rand[b, 6 phb + r, mu, x] = q[(6 phb + r) D V]
    C* link = field + ((int64_t)mu * V + x) * 9;
    C U[9];
    m_load(U, link);
    const int64_t rstride = (int64_t)SNF_D * V;
    const T* q = MODE == 0 ? static_cast<const T*>(a.rand) + (b * SNF_NRAND * SNF_D + mu) * (int64_t)V + x : (const T*)0;
    const T beta = static_cast<const T*>(a.beta)[b * a.beta_bstride];
    SNF_UNROLL
    for (int s = 0; s < 3; ++s) {
        const int phb = a.reverse ? 2 - s : s;
        const int r0 = phb == 0 ? 1 : 0, r1 = phb == 2 ? 1 : 2;
        const C R00 = m_dot(U, r0, S, r0), R01 = m_dot(U, r0, S, r1);
        const C R10 = m_dot(U, r1, S, r0), R11 = m_dot(U, r1, S, r1);
        const C sa = c_scale((T)0.5, c_add(R00, c_conj(R11)));
        const C sb = c_scale((T)0.5, c_sub(R01, c_conj(R10)));
        C c0, c1;
        const bool apply = MODE == 0 ? snf_heatbath<T>(sa, sb, q + 6 * phb * rstride, rstride, beta, &c0, &c1)
                         : MODE == 1 ? snf_overrelax<T>(sa, sb, &c0, &c1)
                                     : snf_cool<T>(sa, sb, &c0, &c1);
        if (!apply) continue;
        // mcmc.SU2_left_multiply_SU3
        SNF_UNROLL
        for (int j = 0; j < 3; ++j) {
            const C u0 = U[3 * r0 + j], u1 = U[3 * r1 + j];
            U[3 * r0 + j] = c_add(c_mul(c0, u0), c_mul(c1, u1));
            U[3 * r1 + j] = c_add(c_mul(c_neg(c_conj(c1)), u0), c_mul(c_conj(c0), u1));
        }
    }
    SNF_UNROLL
    for (int e = 0; e < 9; ++e) link[e] = U[e];
}

#ifdef __CUDACC__
template <typename T, int MODE, bool DEFECT>
__global__ void __launch_bounds__(SNF_THREADS) snf_kernel(const SnfArgs a, int mu, const int64_t* sites, int64_t n) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < a.batch * n) snf_link<T, MODE, DEFECT>(a, mu, sites, n, tid);
}
#endif

// The links of mask m: one CUDA launch on `stream`, or a loop over the thread index on the host
template <typename T, int MODE, bool DEFECT>
static int snf_mask(const SnfArgs& a, int m, void* stream) {
    const int64_t n = a.n[m], total = a.batch * n;
    if (total == 0) return 0;
#ifdef __CUDACC__
    const int64_t blocks = (total + SNF_THREADS - 1) / SNF_THREADS;
    snf_kernel<T, MODE, DEFECT><<<(unsigned)blocks, SNF_THREADS, 0, (cudaStream_t)stream>>>(a, m / 2, a.sites[m], n);
    return (int)cudaGetLastError();
#else
    (void)stream;
    for (int64_t tid = 0; tid < total; ++tid) snf_link<T, MODE, DEFECT>(a, m / 2, a.sites[m], n, tid);
    return 0;
#endif
}

// HBOR.__call__: heat bath of the masks m = 0 .. 2D - 1 (m = eo + 2 mu), then orsteps sweeps of
// overrelaxation in the same order; reversed: the overrelaxation sweeps, then the heat bath, each
// over the masks 2D - 1 .. 0
template <typename T, bool DEFECT>
static int snf_update(const SnfArgs& a, void* stream) {
    int err = 0;
    if (!a.reverse) {
        for (int m = 0; m < SNF_MASKS && !err; ++m) err = snf_mask<T, 0, DEFECT>(a, m, stream);
        for (int o = 0; o < a.orsteps && !err; ++o)
            for (int m = 0; m < SNF_MASKS && !err; ++m) err = snf_mask<T, 1, DEFECT>(a, m, stream);
    } else {
        for (int o = 0; o < a.orsteps && !err; ++o)
            for (int m = SNF_MASKS - 1; m >= 0 && !err; --m) err = snf_mask<T, 1, DEFECT>(a, m, stream);
        for (int m = SNF_MASKS - 1; m >= 0 && !err; --m) err = snf_mask<T, 0, DEFECT>(a, m, stream);
    }
    return err;
}

// Arguments of snf_plaquette (mirrored by the ctypes structure _PlaqArgs in fused.py)
struct SnfPlaqArgs {
    const void* links;               // (B, D, V, 3, 3) complex: the configurations
    void* out;                       // (B, V) real: the sum over the planes of Re Tr P at each site
    int64_t batch, volume;           // B, V
    int64_t extent[SNF_D];           // extents (T, L, L, L)
    int32_t dtype, device;           // dtype 0: complex128
};

// sun.plaquette_SUN at one site, before the average: for nu = 1 .. D - 1 and mu < nu in that order,
// sun.plaquette_retr: A = U_mu(x) U_nu(x + mu), B = U_nu(x) U_mu(x + nu), Re Tr(A B^dagger) =
// sum_e Re(A_e conj(B_e)), with "x + d" the site of coordinate c_d - 1 as in snf_link
template <typename T>
SNF_HD void snf_plaquette_site(const SnfPlaqArgs& a, int64_t tid) {
    typedef SnfC<T> C;
    const int64_t b = tid / a.volume;
    const int32_t x = (int32_t)(tid - b * a.volume);
    const int32_t V = (int32_t)a.volume;
    const C* field = static_cast<const C*>(a.links) + b * ((int64_t)SNF_D * V * 9);
    int32_t fwd[SNF_D];
    int32_t stride = 1;
    SNF_UNROLL
    for (int d = SNF_D - 1; d >= 0; --d) {
        const int32_t L = (int32_t)a.extent[d];
        const int32_t c = (x / stride) % L;
        fwd[d] = c == 0 ? (L - 1) * stride : -stride;
        stride *= L;
    }
    T total = 0;
    SNF_UNROLL
    for (int nu = 1; nu < SNF_D; ++nu) {
        SNF_UNROLL
        for (int mu = 0; mu < nu; ++mu) {
            C U[9], W[9], A[9], B[9];
            m_load(U, field + ((int64_t)mu * V + x) * 9);
            m_load(W, field + ((int64_t)nu * V + x + fwd[mu]) * 9);
            m_mul<T, false, false>(A, U, W);
            m_load(U, field + ((int64_t)nu * V + x) * 9);
            m_load(W, field + ((int64_t)mu * V + x + fwd[nu]) * 9);
            m_mul<T, false, false>(B, U, W);
            T r = A[0].re * B[0].re + A[0].im * B[0].im;
            SNF_UNROLL
            for (int e = 1; e < 9; ++e) r += A[e].re * B[e].re + A[e].im * B[e].im;
            total = (nu == 1 && mu == 0) ? r : total + r;
        }
    }
    static_cast<T*>(a.out)[tid] = total;
}

#ifdef __CUDACC__
template <typename T>
__global__ void __launch_bounds__(SNF_THREADS) snf_plaquette_kernel(const SnfPlaqArgs a) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < a.batch * a.volume) snf_plaquette_site<T>(a, tid);
}
#endif

// topology.cool: `steps` sweeps of Wilson cooling over the masks m = 0 .. 2D - 1, the standard action
template <typename T>
static int snf_cooling(const SnfArgs& a, int steps, void* stream) {
    int err = 0;
    for (int s = 0; s < steps && !err; ++s)
        for (int m = 0; m < SNF_MASKS && !err; ++m) err = snf_mask<T, 2, false>(a, m, stream);
    return err;
}

// The clover topological charge density at one site (topology.clover_charge_torch):
// C_{mu nu} = the four plaquettes around x in the (mu, nu) plane, A = (C - C^dagger) - Tr(C - C^dagger) / 3,
// F = A / (8 i), q = [Tr F01 F23 - Tr F02 F13 + Tr F03 F12] / (4 pi^2) = -[Tr A01 A23 - ...] / (256 pi^2).
// "x + d" is the site of coordinate c_d - 1, "x - d" the one of c_d + 1, as in snf_link.
template <typename T>
SNF_HD void snf_clover_a(SnfC<T> A[9], const SnfC<T>* field, int32_t V, int mu, int nu, int32_t x,
                         const int32_t fwd[SNF_D], const int32_t bwd[SNF_D]) {
    typedef SnfC<T> C;
    C L[9], M[9], P[9], S[9];
    const C* Umu = field + (int64_t)mu * V * 9;
    const C* Unu = field + (int64_t)nu * V * 9;
    const int32_t pm = fwd[mu], pn = fwd[nu], mm = bwd[mu], mn = bwd[nu];
    // leaf 1: U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
    m_load(L, Umu + (int64_t)x * 9);
    m_load(M, Unu + (int64_t)(x + pm) * 9);
    m_mul<T, false, false>(P, L, M);
    m_load(M, Umu + (int64_t)(x + pn) * 9);
    m_mul<T, false, true>(L, P, M);
    m_load(M, Unu + (int64_t)x * 9);
    m_mul<T, false, true>(S, L, M);
    // leaf 2: U_nu(x) U_mu^dag(x+nu-mu) U_nu^dag(x-mu) U_mu(x-mu)
    m_load(L, Unu + (int64_t)x * 9);
    m_load(M, Umu + (int64_t)(x + pn + mm) * 9);
    m_mul<T, false, true>(P, L, M);
    m_load(M, Unu + (int64_t)(x + mm) * 9);
    m_mul<T, false, true>(L, P, M);
    m_load(M, Umu + (int64_t)(x + mm) * 9);
    m_mul<T, false, false>(P, L, M);
    SNF_UNROLL
    for (int e = 0; e < 9; ++e) S[e] = c_add(S[e], P[e]);
    // leaf 3: U_mu^dag(x-mu) U_nu^dag(x-mu-nu) U_mu(x-mu-nu) U_nu(x-nu)
    m_load(L, Umu + (int64_t)(x + mm) * 9);
    m_load(M, Unu + (int64_t)(x + mm + mn) * 9);
    m_mul<T, true, true>(P, L, M);
    m_load(M, Umu + (int64_t)(x + mm + mn) * 9);
    m_mul<T, false, false>(L, P, M);
    m_load(M, Unu + (int64_t)(x + mn) * 9);
    m_mul<T, false, false>(P, L, M);
    SNF_UNROLL
    for (int e = 0; e < 9; ++e) S[e] = c_add(S[e], P[e]);
    // leaf 4: U_nu^dag(x-nu) U_mu(x-nu) U_nu(x-nu+mu) U_mu^dag(x)
    m_load(L, Unu + (int64_t)(x + mn) * 9);
    m_load(M, Umu + (int64_t)(x + mn) * 9);
    m_mul<T, true, false>(P, L, M);
    m_load(M, Unu + (int64_t)(x + mn + pm) * 9);
    m_mul<T, false, false>(L, P, M);
    m_load(M, Umu + (int64_t)x * 9);
    m_mul<T, false, true>(P, L, M);
    SNF_UNROLL
    for (int e = 0; e < 9; ++e) S[e] = c_add(S[e], P[e]);
    // A = (C - C^dagger) - Tr(C - C^dagger) / 3
    SNF_UNROLL
    for (int i = 0; i < 3; ++i) {
        SNF_UNROLL
        for (int k = 0; k < 3; ++k) A[3 * i + k] = c_sub(S[3 * i + k], c_conj(S[3 * k + i]));
    }
    const SnfC<T> tr3 = c_scale((T)1 / (T)3, c_add(c_add(A[0], A[4]), A[8]));
    A[0] = c_sub(A[0], tr3);
    A[4] = c_sub(A[4], tr3);
    A[8] = c_sub(A[8], tr3);
}

// Re Tr(A B)
template <typename T> SNF_HD T snf_retr_mul(const SnfC<T> A[9], const SnfC<T> B[9]) {
    T r = 0;
    SNF_UNROLL
    for (int i = 0; i < 3; ++i) {
        SNF_UNROLL
        for (int j = 0; j < 3; ++j) r += A[3 * i + j].re * B[3 * j + i].re - A[3 * i + j].im * B[3 * j + i].im;
    }
    return r;
}

template <typename T>
SNF_HD void snf_charge_site(const SnfPlaqArgs& a, int64_t tid) {
    typedef SnfC<T> C;
    const int64_t b = tid / a.volume;
    const int32_t x = (int32_t)(tid - b * a.volume);
    const int32_t V = (int32_t)a.volume;
    const C* field = static_cast<const C*>(a.links) + b * ((int64_t)SNF_D * V * 9);
    int32_t fwd[SNF_D], bwd[SNF_D];
    int32_t stride = 1;
    SNF_UNROLL
    for (int d = SNF_D - 1; d >= 0; --d) {
        const int32_t L = (int32_t)a.extent[d];
        const int32_t c = (x / stride) % L;
        fwd[d] = c == 0 ? (L - 1) * stride : -stride;
        bwd[d] = c == L - 1 ? -(L - 1) * stride : stride;
        stride *= L;
    }
    C A[9], B[9];
    snf_clover_a<T>(A, field, V, 0, 1, x, fwd, bwd);
    snf_clover_a<T>(B, field, V, 2, 3, x, fwd, bwd);
    T total = snf_retr_mul<T>(A, B);
    snf_clover_a<T>(A, field, V, 0, 2, x, fwd, bwd);
    snf_clover_a<T>(B, field, V, 1, 3, x, fwd, bwd);
    total -= snf_retr_mul<T>(A, B);
    snf_clover_a<T>(A, field, V, 0, 3, x, fwd, bwd);
    snf_clover_a<T>(B, field, V, 1, 2, x, fwd, bwd);
    total += snf_retr_mul<T>(A, B);
    const T pi = (T)3.141592653589793;
    static_cast<T*>(a.out)[tid] = -total / ((T)256 * pi * pi);
}

#ifdef __CUDACC__
template <typename T>
__global__ void __launch_bounds__(SNF_THREADS) snf_charge_kernel(const SnfPlaqArgs a) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < a.batch * a.volume) snf_charge_site<T>(a, tid);
}
#endif

extern "C" {
int snf_args_size(void) { return (int)sizeof(SnfArgs); }

// `steps` sweeps of Wilson cooling of the links of a->sites (in place). Returns 0 on success, -1 for an
// unsupported dtype, a cudaError_t code otherwise.
int snf_cool(const SnfArgs* a, int steps, void* stream) {
#ifdef __CUDACC__
    int previous = -1;
    int err = (int)cudaGetDevice(&previous);
    if (err) return err;
    if (previous != a->device && (err = (int)cudaSetDevice(a->device))) return err;
#endif
    int code = a->dtype == 0 ? snf_cooling<double>(*a, steps, stream) : -1;
#ifdef __CUDACC__
    if (previous != a->device) cudaSetDevice(previous);
#endif
    return code;
}

// The clover charge density of every site of every sample, one thread per site.
int snf_topological_charge(const SnfPlaqArgs* a, void* stream) {
    if (a->dtype != 0) return -1;
    const int64_t total = a->batch * a->volume;
    if (total == 0) return 0;
#ifdef __CUDACC__
    int previous = -1;
    int err = (int)cudaGetDevice(&previous);
    if (err) return err;
    if (previous != a->device && (err = (int)cudaSetDevice(a->device))) return err;
    const int64_t blocks = (total + SNF_THREADS - 1) / SNF_THREADS;
    snf_charge_kernel<double><<<(unsigned)blocks, SNF_THREADS, 0, (cudaStream_t)stream>>>(*a);
    err = (int)cudaGetLastError();
    if (previous != a->device) cudaSetDevice(previous);
    return err;
#else
    (void)stream;
    for (int64_t tid = 0; tid < total; ++tid) snf_charge_site<double>(*a, tid);
    return 0;
#endif
}
int snf_plaq_args_size(void) { return (int)sizeof(SnfPlaqArgs); }

// The plaquette sums of every site of every sample, one thread per site. Returns 0 on success, -1 for
// an unsupported dtype, a cudaError_t code otherwise.
int snf_plaquette(const SnfPlaqArgs* a, void* stream) {
    if (a->dtype != 0) return -1;
    const int64_t total = a->batch * a->volume;
    if (total == 0) return 0;
#ifdef __CUDACC__
    int previous = -1;
    int err = (int)cudaGetDevice(&previous);
    if (err) return err;
    if (previous != a->device && (err = (int)cudaSetDevice(a->device))) return err;
    const int64_t blocks = (total + SNF_THREADS - 1) / SNF_THREADS;
    snf_plaquette_kernel<double><<<(unsigned)blocks, SNF_THREADS, 0, (cudaStream_t)stream>>>(*a);
    err = (int)cudaGetLastError();
    if (previous != a->device) cudaSetDevice(previous);
    return err;
#else
    (void)stream;
    for (int64_t tid = 0; tid < total; ++tid) snf_plaquette_site<double>(*a, tid);
    return 0;
#endif
}

int snf_is_cuda(void) {
#ifdef __CUDACC__
    return 1;
#else
    return 0;
#endif
}

// Initializes the CUDA runtime of this library on `device` and loads the kernels there (cudaFuncGetAttributes
// forces the lazy loading of CUDA_MODULE_LOADING=LAZY): called once per device when the library is loaded
// (load_backend in fused.py), so that neither happens later inside a CUDA graph capture. Returns 0 on success,
// a cudaError_t code otherwise; nothing on the host build.
int snf_init(int device) {
#ifdef __CUDACC__
    int previous = -1;
    int err = (int)cudaGetDevice(&previous);
    if (err) return err;
    if (previous != device && (err = (int)cudaSetDevice(device))) return err;
    cudaFuncAttributes attr;
    err = (int)cudaFuncGetAttributes(&attr, (const void*)snf_kernel<double, 0, false>);
    if (!err) err = (int)cudaFuncGetAttributes(&attr, (const void*)snf_kernel<double, 0, true>);
    if (!err) err = (int)cudaFuncGetAttributes(&attr, (const void*)snf_kernel<double, 1, false>);
    if (!err) err = (int)cudaFuncGetAttributes(&attr, (const void*)snf_kernel<double, 1, true>);
    if (!err) err = (int)cudaFuncGetAttributes(&attr, (const void*)snf_plaquette_kernel<double>);
    if (!err) err = (int)cudaFuncGetAttributes(&attr, (const void*)snf_kernel<double, 2, false>);
    if (!err) err = (int)cudaFuncGetAttributes(&attr, (const void*)snf_charge_kernel<double>);
    if (previous != device) cudaSetDevice(previous);
    return err;
#else
    (void)device;
    return 0;
#endif
}

// Returns 0 on success, -1 for an unsupported dtype, a cudaError_t code otherwise.
int snf_hbor(const SnfArgs* a, void* stream) {
#ifdef __CUDACC__
    int previous = -1;
    int err = (int)cudaGetDevice(&previous);
    if (err) return err;
    if (previous != a->device && (err = (int)cudaSetDevice(a->device))) return err;
#endif
    int code = -1;
    if (a->dtype == 0) code = a->has_defect ? snf_update<double, true>(*a, stream) : snf_update<double, false>(*a, stream);
#ifdef __CUDACC__
    if (previous != a->device) cudaSetDevice(previous);
#endif
    return code;
}

const char* snf_error_string(int code) {
    if (code == -1) return "unsupported dtype";
#ifdef __CUDACC__
    return cudaGetErrorString((cudaError_t)code);
#else
    return code == 0 ? "no error" : "unknown error";
#endif
}
}
"""

_MASKS = 8


class _Args(ctypes.Structure):
    # ctypes mirror of SnfArgs
    _fields_ = [
        ('links', ctypes.c_void_p),
        ('rand', ctypes.c_void_p),
        ('kappa', ctypes.c_void_p),
        ('beta', ctypes.c_void_p),
        ('sites', ctypes.c_void_p * _MASKS),
        ('n', ctypes.c_int64 * _MASKS),
        ('batch', ctypes.c_int64),
        ('volume', ctypes.c_int64),
        ('extent', ctypes.c_int64 * 4),
        ('kappa_bstride', ctypes.c_int64),
        ('beta_bstride', ctypes.c_int64),
        ('orsteps', ctypes.c_int32),
        ('has_defect', ctypes.c_int32),
        ('reverse', ctypes.c_int32),
        ('dtype', ctypes.c_int32),
        ('device', ctypes.c_int32),
        ('pad', ctypes.c_int32),
    ]

class _PlaqArgs(ctypes.Structure):
    # ctypes mirror of SnfPlaqArgs
    _fields_ = [
        ('links', ctypes.c_void_p),
        ('out', ctypes.c_void_p),
        ('batch', ctypes.c_int64),
        ('volume', ctypes.c_int64),
        ('extent', ctypes.c_int64 * 4),
        ('dtype', ctypes.c_int32),
        ('device', ctypes.c_int32),
    ]

# ----------------------------------------------------------------------------------------------
# Build and load
# ----------------------------------------------------------------------------------------------

class BackendUnavailable(RuntimeError):
    """The kernel cannot be built or loaded here (the message says why)."""


def check_int32(count, what):
    # The kernels hold the sites in int32: fail loudly instead of wrapping around
    if count > 2**31 - 1:
        raise ValueError(f'{what} = {count}: beyond the int32 indices of the fused kernels')

# target key -> _Backend, or the BackendUnavailable of a failed attempt (not retried in the process)
_BACKENDS = {}
_WARNED = set()


def _warn_once(message):
    if message not in _WARNED:
        _WARNED.add(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _cache_dir():
    root = os.environ.get('NEOQCD_KERNEL_CACHE')
    return Path(root) if root else Path.home() / '.cache' / 'neoqcd' / 'kernels'


def _find_nvcc():
    for var in ('CUDA_HOME', 'CUDA_PATH'):
        root = os.environ.get(var)
        if root and (Path(root) / 'bin' / 'nvcc').is_file():
            return str(Path(root) / 'bin' / 'nvcc')
    return shutil.which('nvcc')


def _find_cxx():
    names = ([os.environ['CXX']] if os.environ.get('CXX') else []) + ['c++', 'g++', 'clang++']
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def _version(compiler):
    try:
        return subprocess.run([compiler, '--version'], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def _build(target, device=None, source=None, name='neoqcd_hbor', rebuild=False):
    # Compiles `source` (default KERNEL_SOURCE) for target 'cuda' or 'cpu' into the cache as library
    # `name`; returns the library path (the cached one, unless rebuild). Also used by
    # fused_smearing.py for its own kernel source.
    source = KERNEL_SOURCE if source is None else source
    if target == 'cuda':
        compiler = _find_nvcc()
        if compiler is None:
            raise BackendUnavailable('nvcc not found (set CUDA_HOME or load a cuda module)')
        arch = '{}{}'.format(*torch.cuda.get_device_capability(device))
        flags = ['-O3', '-std=c++14', '-shared', '-Xcompiler', '-fPIC', '-Xptxas', '-v',
                 f'-gencode=arch=compute_{arch},code=sm_{arch}', f'-gencode=arch=compute_{arch},code=compute_{arch}']
        suffix = '.cu'
    elif target == 'cpu':
        compiler = _find_cxx()
        if compiler is None:
            raise BackendUnavailable('no host C++ compiler found (c++, g++, clang++ or $CXX)')
        flags = ['-O2', '-std=c++14', '-shared', '-fPIC']
        suffix = '.cpp'
    else:
        raise ValueError(f'unknown target {target!r}')
    tag = hashlib.sha256('\n'.join([source, compiler, _version(compiler)] + flags).encode()).hexdigest()[:16]
    cache = _cache_dir()
    lib = cache / f'{name}_{target}_{tag}.so'
    if lib.is_file() and not rebuild:
        return lib
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'{name}_') as tmp:
        src = Path(tmp) / f'{name}{suffix}'
        src.write_text(source)
        # a name of its own (not only the pid, which jobs on other nodes sharing the cache may have too)
        fd, partial = tempfile.mkstemp(dir=cache, prefix=f'.{lib.name}.', suffix='.tmp')
        os.close(fd)
        partial = Path(partial)
        cmd = [compiler] + flags + ['-o', str(partial), str(src)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.SubprocessError) as exc:
            partial.unlink(missing_ok=True)
            raise BackendUnavailable(f'{target} build failed to run: {exc}')
        if res.returncode != 0:
            partial.unlink(missing_ok=True)
            msg = (res.stderr or res.stdout).strip().splitlines()
            raise BackendUnavailable(f'{target} build failed ({compiler}): ' + ' | '.join(msg[-6:]))
        lib.with_suffix('.log').write_text(' '.join(cmd) + '\n' + res.stdout + res.stderr)
        # atomic: concurrent builds and killed jobs leave no half-written library
        os.replace(partial, lib)
    return lib


class _Backend:
    # A loaded kernel library; run(args, stream) raises RuntimeError on a nonzero return code. SOURCE and
    # NAME: what _build compiles (a subclass with its own source binds its own entry points); INIT: the entry
    # point int(int device) that prepare() calls once per CUDA device
    SOURCE = KERNEL_SOURCE
    NAME = 'neoqcd_hbor'
    INIT = 'snf_init'

    def __init__(self, target, path):
        self.target = target
        self.path = str(path)
        lib = ctypes.CDLL(self.path)
        lib.snf_args_size.restype = ctypes.c_int
        lib.snf_is_cuda.restype = ctypes.c_int
        lib.snf_init.argtypes = [ctypes.c_int]
        lib.snf_init.restype = ctypes.c_int
        lib.snf_hbor.argtypes = [ctypes.POINTER(_Args), ctypes.c_void_p]
        lib.snf_hbor.restype = ctypes.c_int
        lib.snf_plaq_args_size.restype = ctypes.c_int
        lib.snf_plaquette.argtypes = [ctypes.POINTER(_PlaqArgs), ctypes.c_void_p]
        lib.snf_plaquette.restype = ctypes.c_int
        lib.snf_cool.argtypes = [ctypes.POINTER(_Args), ctypes.c_int, ctypes.c_void_p]
        lib.snf_cool.restype = ctypes.c_int
        lib.snf_topological_charge.argtypes = [ctypes.POINTER(_PlaqArgs), ctypes.c_void_p]
        lib.snf_topological_charge.restype = ctypes.c_int
        lib.snf_error_string.argtypes = [ctypes.c_int]
        lib.snf_error_string.restype = ctypes.c_char_p
        if lib.snf_plaq_args_size() != ctypes.sizeof(_PlaqArgs):
            raise BackendUnavailable(f'argument layout mismatch (plaquette): library {lib.snf_plaq_args_size()} '
                                     f'bytes, ctypes {ctypes.sizeof(_PlaqArgs)}')
        if lib.snf_args_size() != ctypes.sizeof(_Args):
            raise BackendUnavailable(f'argument layout mismatch: library {lib.snf_args_size()} bytes, '
                                     f'ctypes {ctypes.sizeof(_Args)}')
        if bool(lib.snf_is_cuda()) != (target == 'cuda'):
            raise BackendUnavailable(f'library {self.path} is not a {target} build')
        self._lib = lib
        self._prepared = set()

    @property
    def log(self):
        # the compiler output of the build (nvcc: registers and spills of the kernels)
        log = Path(self.path).with_suffix('.log')
        return log.read_text() if log.is_file() else ''

    def error_string(self, code):
        return self._lib.snf_error_string(code).decode(errors='replace')

    def prepare(self, device):
        # Initializes the CUDA runtime of the library on `device` and loads its kernels there (INIT), once per
        # device; nothing for the CPU build. load_backend calls it, so that the first launch of a kernel inside
        # a CUDA graph capture (the smearing layer with --cuda-graphs) loads nothing lazily.
        if self.target != 'cuda':
            return
        index = device.index if device is not None and device.index is not None else torch.cuda.current_device()
        if index in self._prepared:
            return
        code = getattr(self._lib, self.INIT)(index)
        if code != 0:
            raise BackendUnavailable(f'{self.NAME}: the kernels do not load on cuda:{index}: code {code}: '
                                     f'{self.error_string(code)}')
        self._prepared.add(index)

    def run(self, args, stream):
        code = self._lib.snf_hbor(ctypes.byref(args), stream)
        if code != 0:
            raise RuntimeError(f'fused HBOR ({self.target}) failed: code {code}: {self.error_string(code)}')

    def run_plaquette(self, args, stream):
        code = self._lib.snf_plaquette(ctypes.byref(args), stream)
        if code != 0:
            raise RuntimeError(f'fused plaquette ({self.target}) failed: code {code}: {self.error_string(code)}')

    def run_cool(self, args, steps, stream):
        code = self._lib.snf_cool(ctypes.byref(args), int(steps), stream)
        if code != 0:
            raise RuntimeError(f'fused cooling ({self.target}) failed: code {code}: {self.error_string(code)}')

    def run_charge(self, args, stream):
        code = self._lib.snf_topological_charge(ctypes.byref(args), stream)
        if code != 0:
            raise RuntimeError(f'fused charge ({self.target}) failed: code {code}: {self.error_string(code)}')


def _load(backend_class, target, device):
    # The library from the cache, or built; a cached library that does not load (truncated, or built on a
    # node with other system libraries) is built again, in its place, once
    path = _build(target, device, backend_class.SOURCE, backend_class.NAME)
    try:
        return backend_class(target, path)
    except OSError:
        path = _build(target, device, backend_class.SOURCE, backend_class.NAME, rebuild=True)
        return backend_class(target, path)


def load_backend(target, device=None, backend_class=_Backend):
    """The kernel library for target 'cuda' or 'cpu' (the emulation), built at first use and cached
    in the process and on disk, its kernels loaded on `device` (_Backend.prepare); raises
    BackendUnavailable with the reason."""
    if target == 'cuda':
        if not torch.cuda.is_available():
            raise BackendUnavailable('CUDA is not available to torch')
        device = torch.device('cuda', torch.cuda.current_device()) if device is None else torch.device(device)
        key = (backend_class.NAME, 'cuda', torch.cuda.get_device_capability(device))
    elif target == 'cpu':
        key = (backend_class.NAME, 'cpu')
    else:
        raise ValueError(f'unknown target {target!r}')
    hit = _BACKENDS.get(key)
    if isinstance(hit, Exception):
        raise hit
    if hit is None:
        try:
            hit = _load(backend_class, target, device)
        except BackendUnavailable as exc:
            hit = exc
        except Exception as exc:  # any loading problem means: not available
            hit = BackendUnavailable(f'{target} backend failed to load: {type(exc).__name__}: {exc}')
        _BACKENDS[key] = hit
        if isinstance(hit, Exception):
            raise hit
    # once per device (a set lookup afterwards): the library may be shared by devices of one capability
    hit.prepare(device)
    return hit


def backend_status(target, device=None, backend_class=_Backend):
    """(available, library path or the reason why not)."""
    try:
        return True, load_backend(target, device, backend_class).path
    except BackendUnavailable as exc:
        return False, str(exc)


def enable_hbor_kernel(device, rank=0, world_size=1):
    """Turn the kernel on for every HBOR of the process (the class attribute HBOR.use_fused_kernels)
    and load its library for `device`: rank 0 builds it first, the other ranks then load it from the
    shared cache instead of compiling it concurrently (a barrier in between when torch.distributed is
    initialized). Returns (available, library path or the reason why not); where it is not available
    the torch sweep runs, with one RuntimeWarning. Used by the --fused-kernels options of the mains."""
    import torch.distributed as dist
    from neoqcd.mcmc import HBOR
    HBOR.use_fused_kernels = True
    device = torch.device(device)
    status = (False, f'device {device.type}: the kernel runs on CUDA (the CPU build is for the tests)')
    if device.type == 'cuda' and rank == 0:
        status = backend_status('cuda', device)
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    if device.type == 'cuda' and rank != 0:
        status = backend_status('cuda', device)
    return status


# MaskedLinks -> None, or why its masks cannot be updated in place: a link of a mask among the staple
# links of the same mask (HBOR computes all the staples of a mask from the links before its update)
_IN_PLACE = weakref.WeakKeyDictionary()


def _in_place_problem(links):
    if links not in _IN_PLACE:
        problem = None
        for m, (updated, per_nu) in enumerate(zip(links.links, links.staple_links)):
            if bool(torch.isin(updated, torch.cat([s.flatten() for s in per_nu])).any()):
                problem = f'a link of mask {m} is a staple link of the same mask'
                break
        _IN_PLACE[links] = problem
    return _IN_PLACE[links]

# ----------------------------------------------------------------------------------------------
# The sweep
# ----------------------------------------------------------------------------------------------

_BETAS = {}


def _beta_tensor(hbor, batch, device):
    # The coupling as a contiguous float64 tensor on the device and its per-sample stride
    beta = hbor.beta
    if torch.is_tensor(beta):
        beta = beta.detach().to(device=device, dtype=torch.float64).reshape(-1).contiguous()
        if beta.numel() == 1:
            return beta, 0
        if beta.numel() != batch:
            raise ValueError(f'beta has shape ({beta.numel()},), expected batch size {batch}')
        return beta, 1
    key = (float(beta), str(device))
    if key not in _BETAS:
        _BETAS[key] = torch.tensor([float(beta)], dtype=torch.float64, device=device)
    return _BETAS[key], 0


def _backend_for(hbor, cfgs, links):
    # The kernel library for this call, or None for HBOR's torch sweep
    if cfgs.device.type == 'cuda':
        target = 'cuda'
    elif cfgs.device.type == 'cpu' and getattr(hbor, 'fused_emulate', False):
        target = 'cpu'
    else:
        return None
    lattice = tuple(cfgs.shape[2:2 + hbor.D])
    if hbor.N != 3 or hbor.D != 4:
        reason = f'the kernel is written for SU(3) in D = 4, not N = {hbor.N}, D = {hbor.D}'
    elif cfgs.dtype != torch.complex128:
        reason = f'the kernel is instantiated for complex128 only, not {cfgs.dtype}'
    elif links.V != torch.Size(lattice).numel() or len(links.sites) != 2 * hbor.D:
        reason = f'the masked links do not belong to the lattice {lattice}'
    elif any(s.device != cfgs.device or s.dtype != torch.int64 or not s.is_contiguous() for s in links.sites):
        reason = f'the sites of the masks are not contiguous int64 on {cfgs.device}'
    else:
        reason = _in_place_problem(links)
    if reason is None:
        try:
            return load_backend(target, cfgs.device)
        except BackendUnavailable as exc:
            reason = str(exc)
    if getattr(hbor, 'fused_strict', False):
        raise BackendUnavailable(reason)
    _warn_once(f'fused HBOR: {reason}; using the torch sweep')
    return None


def hbor_call(hbor, cfgs, mask, defect, rand=None, reverse=False):
    """The sweep of HBOR.__call__ through the kernel: one HB sweep and hbor.orsteps OR sweeps over
    the links selected by `mask` (reversed with reverse=True), out of place. Returns None when the
    kernel does not apply (an input that requires gradients in grad mode, CPU without
    hbor.fused_emulate, another N, D or dtype, a kernel that cannot be built): the caller then runs
    its torch sweep. Grad mode alone does not stop the kernel: main_pt_obc.py sweeps in grad mode, and
    without an input that requires gradients the torch sweep builds no graph either."""
    if torch.is_grad_enabled() and any(
            torch.is_tensor(t) and t.requires_grad
            for t in (cfgs, rand, defect, hbor.beta, getattr(hbor, "defect_par", None))):
        return None
    links = mask if isinstance(mask, sun.MaskedLinks) else hbor._links_for_mask(mask)
    backend = _backend_for(hbor, cfgs, links)
    if backend is None:
        return None
    batch, device = int(cfgs.shape[0]), cfgs.device
    shape = hbor._rand_shape(cfgs)
    if rand is None:
        rand = torch.rand(shape, device=device, dtype=cfgs.real.dtype)
    elif tuple(rand.shape) != tuple(shape):
        raise ValueError(f'rand has shape {tuple(rand.shape)}, expected {tuple(shape)}')
    rand = rand.to(device=device, dtype=torch.float64).contiguous()
    # out of place, as HBOR's sweep: a contiguous copy, updated in place by the kernel
    out = torch.empty(cfgs.shape, dtype=cfgs.dtype, device=device)
    out.copy_(cfgs)
    kappa = None
    if defect is not None and hbor._defect_parameter_uses_defect():
        kappa = hbor._sweep_kappa(cfgs, defect).to(device=device, dtype=torch.float64).contiguous()
        if kappa.shape[0] not in (1, batch):
            raise ValueError(f'link factors of batch {kappa.shape[0]}, expected 1 or {batch}')
    beta, beta_bstride = _beta_tensor(hbor, batch, device)

    a = _Args()
    a.links, a.rand = out.data_ptr(), rand.data_ptr()
    a.kappa = None if kappa is None else kappa.data_ptr()
    a.beta = beta.data_ptr()
    for m, sites in enumerate(links.sites):
        a.sites[m], a.n[m] = sites.data_ptr(), sites.numel()
    check_int32(links.V * hbor.D, 'links of the lattice')
    a.batch, a.volume = batch, links.V
    a.extent[:] = cfgs.shape[2:2 + hbor.D]
    a.kappa_bstride = hbor.D * links.V if (kappa is not None and kappa.shape[0] > 1) else 0
    a.beta_bstride = beta_bstride
    a.orsteps = int(hbor.orsteps)
    a.has_defect = int(kappa is not None)
    a.reverse = int(bool(reverse))
    a.dtype = 0
    a.device = device.index if device.index is not None else 0
    stream = torch.cuda.current_stream(device).cuda_stream if device.type == 'cuda' else None
    backend.run(a, stream)

    # Re-unitarize the updated links only (all of them on a whole-lattice sweep), as HBOR._sweep does
    flat = sun.flat_links(out, hbor.D)
    updated = links.all_links
    if int(updated.numel()) == int(flat.shape[1]):
        flat = sun.SU3_unitarize(flat)
    else:
        flat.index_copy_(1, updated, sun.SU3_unitarize(flat.index_select(1, updated)))
    hbor.fused_used = backend.target
    return flat.reshape(cfgs.shape)


def plaquette(cfgs, N=3, emulate=False):
    """sun.plaquette_SUN through one kernel: the mean plaquette Re Tr P / N of each configuration over the
    sites and the D(D-1)/2 planes, (B,) float64, the same products per site (rounding aside). One read of
    the links instead of the rolled copies and broadcast products of plaquette_SUN (~90 GB of memory
    traffic for 64 chains at 16^4; main_pt_obc.py measures it for the replica history every step).
    None when the kernel does not apply: not SU(3) in D = 4 complex128, gradients, the CPU without
    `emulate`, or no kernel library (then one warning)."""
    if int(N) != 3:
        return None
    out = _site_kernel(cfgs, emulate, 'run_plaquette', 'fused plaquette: {}; using sun.plaquette_SUN')
    return None if out is None else out.mean(dim=1) / 6.0 / float(N)


def _site_backend(cfgs, emulate, warning):
    # The library for a kernel over the sites of SU(3) configurations in D = 4 (complex128, no gradients),
    # or None (then one warning if the library is missing)
    if (cfgs.dim() != 8 or int(cfgs.shape[1]) != 4 or tuple(cfgs.shape[-2:]) != (3, 3)
            or cfgs.dtype != torch.complex128 or (torch.is_grad_enabled() and cfgs.requires_grad)):
        return None, None
    if cfgs.device.type == 'cuda':
        target = 'cuda'
    elif cfgs.device.type == 'cpu' and emulate:
        target = 'cpu'
    else:
        return None, None
    try:
        return load_backend(target, cfgs.device), target
    except BackendUnavailable as exc:
        _warn_once(warning.format(exc))
        return None, None


def _site_kernel(cfgs, emulate, method, warning):
    # One value per site and sample, (B, V) float64, from the kernel `method` of the library (SnfPlaqArgs)
    backend, target = _site_backend(cfgs, emulate, warning)
    if backend is None:
        return None
    links = cfgs.detach().contiguous()
    batch, volume = int(links.shape[0]), int(torch.Size(links.shape[2:6]).numel())
    check_int32(4 * volume, 'links of the lattice')
    out = torch.empty((batch, volume), dtype=torch.float64, device=links.device)
    a = _PlaqArgs()
    a.links, a.out = links.data_ptr(), out.data_ptr()
    a.batch, a.volume = batch, volume
    a.extent[:] = links.shape[2:6]
    a.dtype = 0
    a.device = links.device.index if links.device.index is not None else 0
    stream = torch.cuda.current_stream(links.device).cuda_stream if target == 'cuda' else None
    getattr(backend, method)(a, stream)
    return out


def clover_charge(cfgs, emulate=False):
    """topology.clover_charge_torch through one kernel (snf_topological_charge): the clover topological
    charge of each configuration, (B,) float64. None when the kernel does not apply (see plaquette)."""
    out = _site_kernel(cfgs, emulate, 'run_charge', 'fused clover charge: {}; using topology.clover_charge_torch')
    return None if out is None else out.sum(dim=1)


_COOLING_LINKS = {}


def cooling(cfgs, steps, emulate=False):
    """topology.cool_torch through the kernel (snf_cool, MODE 2 of snf_link): `steps` sweeps of Wilson
    cooling of every link (checkerboard masks m = eo + 2 mu in order, SU(2) subgroups 0, 1, 2), on a copy.
    None when the kernel does not apply."""
    backend, target = _site_backend(cfgs, emulate, 'fused cooling: {}; using topology.cool_torch')
    if backend is None:
        return None
    T, L = int(cfgs.shape[2]), int(cfgs.shape[3])
    if tuple(cfgs.shape[3:6]) != (L, L, L):
        return None  # the checkerboard masks of create_mask: T x L^3
    key = (T, L, str(cfgs.device))
    links = _COOLING_LINKS.get(key)
    if links is None:
        from neoqcd.utils import create_mask
        links = sun.MaskedLinks(create_mask(4, T, L).to(cfgs.device), 4)
        _COOLING_LINKS[key] = links
    out = cfgs.detach().contiguous().clone()
    a = _Args()
    a.links, a.rand, a.kappa, a.beta = out.data_ptr(), None, None, None
    for m, sites in enumerate(links.sites):
        a.sites[m], a.n[m] = sites.data_ptr(), sites.numel()
    check_int32(links.V * 4, 'links of the lattice')
    a.batch, a.volume = int(out.shape[0]), links.V
    a.extent[:] = out.shape[2:6]
    a.kappa_bstride = a.beta_bstride = 0
    a.orsteps, a.has_defect, a.reverse, a.dtype = 0, 0, 0, 0
    a.device = out.device.index if out.device.index is not None else 0
    stream = torch.cuda.current_stream(out.device).cuda_stream if target == 'cuda' else None
    backend.run_cool(a, steps, stream)
    return out
