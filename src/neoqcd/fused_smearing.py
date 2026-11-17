"""Fused kernels of the stout smearing step of SU(3) (smearing_jacobian.smearing_step) and of its inverse.

Ported from snf_SUN, branch perf-main at a3bfc7b (snf_code/fused_smearing.py), cut down to the stout step on given
links and staple sums, with a new kernel for the inverse of the step (the changes are listed below). The file
keeps the snf_SUN name and layout, so that the two can be compared file by file; the device functions of the step
and of its coefficients are snf_SUN's.

With Smearing.use_fused_kernels (main_pt_obc.py --fused-kernels) the SU(3) path of Smearing on the selected links
runs, where no gradient is recorded, the stout step of every masked update through one kernel launch
(stout_links, in Smearing._forward_blocks: U' and log|det J| of every link) and the fixed-iteration inverse of
Smearing.inverse (the NEO-REX default, 10 iterations) through another (inverse_smearing_step, in
Smearing._inverse_blocks). Each thread reads one link and its staple sum and does the whole step in registers:
Om, Q, the coefficients of exp(iQ), U' = exp(iQ) U and the Jacobian on the tangent space. The torch step makes
hundreds of small elementwise kernels per masked update (the 9x9 complex Jacobian and its slogdet), memory bound.
The staple sums (with rho), the gathers and the write-backs of the links stay in torch for now.

Same mathematics as the torch path, link by link, in the same order where it matters:
- Om = C U^dagger, Q = i/2 (Om^dagger - Om - Tr(...)/3), Q^2, c1 = Tr(Q^2)/2, c0 = Tr(Q^3)/3 as in
  smearing_jacobian.StepQuantities; X = f0 + f1 Q + f2 Q^2 and U' = X U;
- the coefficients of smearing_jacobian.exp_coefficients: the Taylor series truncated after N_SERIES terms for
  c1 < C1_SERIES, the closed forms of Morningstar and Peardon (mp_coefficients, order 1, with the xi functions
  summed from XI_TAYLOR below |w| = XI_SERIES) above, chosen link by link in the kernel: no host check and no
  DeferredChecks of c1 (the torch path under DeferredChecks takes the series everywhere and repeats the call
  when some c1 >= C1_SERIES). The series is the same truncated polynomial, summed term by term by the recursion
  of smearing_jacobian.series_polynomials, (a, b, c)_{n+1} = (c0 c_n, a_n + c1 c_n, b_n), and its derivatives in
  c1 and c0, instead of the table of monomials: ~5e-16 from series_coefficients for c1 < 3;
- log|det J|: not the 9x9 complex matrix of smearing_step but the real 8x8 Jacobian on the tangent space, which
  has the same determinant. See "Jacobian" below.
The constants C1_SERIES, N_SERIES, XI_SERIES and the coefficients XI_TAYLOR are taken from smearing_jacobian
when the source is generated. Results agree with the torch path to float64 rounding (test/test_fused_smearing.py,
CPU emulation): link by link U' within 1e-15 and log|det J| within 1.5e-14 relative, on both sides of C1_SERIES
and on degenerate Q; the inverse within 3e-16 (U) and 3e-15 (log|det|) of inverse_smearing_step; whole NEO-REX
defect layers (class mode, |rho| up to 0.1, 1 and 2 steps) within 3e-16 (maps) and 1.5e-14 (log J per
configuration), forward and inverse, and within 3e-15 at |rho| ~ 3 (closed forms on part of the links).
log|det J| of one link is as accurate as the 9x9 slogdet (both within ~cond(J) eps of an exact value; snf_SUN
measured 1.4e-12 from an mpmath reference for a link with a singular value 1.7e-4 of J).

Jacobian. For dU = i y_b e_b U (e_b a basis of the hermitian traceless matrices, y real), U' changes
by dU' = i (A y)_a e_a U', and |det J| = |det A| (eq. (eq:J8) of snf_SUN's notes/su3_stout_jacobian.tex). With
X = exp(iQ) and dX its change, -i dU' U'^dagger = X (e_b - i X^dagger dX) X^dagger: A = Ad_X (1 + B),
Ad_X : e_b -> X e_b X^dagger a rotation of the 8 coordinates (det 1), and B_ab the coordinates of -i X^dagger dX,
hermitian and traceless. So log|det J| = log|det(1 + B)|, one 8x8 real determinant instead of the 9x9 complex one.
Column b of B, from dQ = -(M + M^dagger)/2 + Re Tr(M)/3, M = e_b Om^dagger (eq. (eq:dQ-Y) with
Y = i e_b), dc1 = Tr(Q dQ), dc0 = Tr(Q^2 dQ), df_j = b1_j dc1 + b2_j dc0 and
dX = sum_j df_j Q^j + f1 dQ + f2 (Q dQ + dQ Q):
- the part sum_j df_j X^dagger Q^j is a polynomial in Q; with X^dagger = sum_j f_j^* Q^j and
  Q^3 = c1 Q + c0 (Cayley-Hamilton) it is g_I + g_Q Q + g_Q2 Q^2, and only Q and Q^2 have
  coordinates: its column is dc1 v1 + dc0 v0, with two vectors v1, v0 fixed per link;
- the rest is X^dagger H, H = f1 dQ + f2 (P + P^dagger), P = Q dQ: two 3x3 products per column.
The basis: e_b = lambda_1, ..., lambda_7 (Gell-Mann) and diag(1, 1, -2) (sqrt 3 lambda_8), and the
coordinates e^a(H) = Tr(e_a H)/Tr(e_a e_a): the multiplications by e_b are exact. 1 + B is reduced by
an LU decomposition with partial pivoting, column by column as the columns are built (left-looking,
so that only the 28 multipliers of L are kept), and log|det| = log|product of the pivots|.
Cost per link, in complex multiplications: Q, Q^2, X, U' ~110, coefficients ~120
(series), Jacobian ~8 x 75, LU ~50 (real); the 9x9 complex Jacobian of StepQuantities.jacobian would
need ~1100 to be built and ~250 for its LU, with 81 complex entries live. The closed forms cost a few
transcendental functions more, on the links with c1 >= C1_SERIES only (none for small rho).

Inverse (new). stout_links_inverse(U', C, n) is smearing_jacobian.inverse_smearing_step(U', C, fixed_iterations=n)
link by link: the fixed point of sun.invert_left_update, U_0 = U', U_{k+1} = X(U_k)^dagger U' with X(U) = exp(iQ)
from Om = C U^dagger (StepQuantities(U, C).X), n times in registers with the device functions of the step (Om, Q,
the coefficients, X). Then, as there, the residual |X(U) U - U'| at the final U, here per link (the largest
modulus of its 9 entries, NaN if any is NaN), and the log|det| of the inverse map, minus log|det J| of the step
at the final U (snf_stout_step, computed after U and the residual are written, so that U' and C are not live
during the Jacobian). inverse_smearing_step takes the largest residual on the device and records it in the active
smearing_jacobian.DeferredChecks, as the torch function of the same name does: Smearing.inverse checks it once
per call and repeats the call with its adaptive torch iteration when it is above the tolerance.

Kernels (KERNEL_SOURCE). One thread per link, 128 threads per block, one launch per call: snf_stout_links (the
step), snf_stout_links_inverse (the inverse), snf_stout_coefficients (the coefficients of order 2 alone, for the
tests). Layout: the links and the staple sums as (..., 3, 3) complex128, contiguous. Registers during the
Jacobian: Om, Q, X (27 complex), f1, f2, v1, v0, the 28 multipliers of L and one column, and the products of the
current column, ~150 doubles (nvcc -Xptxas -v in the build log tells the registers and spills).
$NEOQCD_SMEARING_MIN_BLOCKS sets the __launch_bounds__ of the kernels (MIN_BLOCKS).

CUDA graphs. main_pt_obc.py --cuda-graphs captures the defect smearing layer (DefectSNFProtocolLayer._graphed runs
Smearing._forward_blocks and _inverse_blocks inside torch.cuda.graph), so with --fused-kernels the kernel launches
are captured with the torch kernels around them. They go to the current torch stream (the capture stream), with
no allocation, synchronization or host read of device data around them (the outputs are torch tensors from the
graph's pool, the residual is reduced and recorded on the device). The library has its own CUDA runtime (nvcc
links cudart statically): its first runtime call initializes it, and with lazy module loading
(CUDA_MODULE_LOADING=LAZY, the default of recent CUDA) the first launch of a kernel loads its module; either
inside a capture (global mode in torch 2.0) could invalidate it. So the runtime is initialized and the kernels
loaded when the library is loaded, on every device it is used on (snf_stout_init: cudaFuncGetAttributes on every
kernel, called by fused.load_backend), and _graphed runs the layer eagerly twice before its capture anyway.
To be verified on the GPU: test/test_fused_smearing.py CudaGraph (capture and replay against the eager torch
layer).

Changes with respect to snf_SUN:
- only the stout step on given links and staple sums (snf_stout_links, stout_links) and the coefficients of
  order 2 (snf_stout_coefficients, exp_coefficients, for the tests) are ported. Not ported: the layer kernel
  (staple sums from the field with |rho|, the masked updates in place: snf_stout_layer), the backward of the step
  and of the layer (snf_stout_links_backward, snf_stout_layer_backward) and the classes FusedSmearing,
  FusedSmearingLayer and FusedSmearingStep. The staple sums of NEO-REX take a signed rho per link and staple
  from the hypernetwork: a later step;
- the closed forms clamp |c0|/c0max one representable step below 1, as smearing_jacobian.mp_coefficients does
  (snf_SUN clamps at 1);
- new: the inverse of the step (snf_stout_links_inverse; stout_links_inverse, inverse_smearing_step);
- new: snf_stout_init (see "CUDA graphs");
- $NEOQCD_SMEARING_MIN_BLOCKS instead of $SNF_SMEARING_MIN_BLOCKS; the library is neoqcd_stout_<target>_<hash>.so.

The kernels have no backward: the torch step runs where autograd records the step, i.e. in grad mode with an
input of the step (the links U or U', or the staple sums C, through rho) that requires grad (needs_autograd; in
NEO-REX the training of the hypernetwork; its evaluation runs under torch.no_grad()). Grad mode alone does not
disable them. They run for SU(3), complex128, on CUDA, or on the CPU with Smearing.fused_emulate (the host build
of the same source: the CPU emulation the tests compare with the torch path on a laptop). Otherwise, or when the
library cannot be built or loaded, the torch step runs (one RuntimeWarning per reason; Smearing.fused_strict
raises fused.BackendUnavailable instead). Smearing.fused_used records the path of the last call: 'cuda', 'cpu'
(the emulation), or 'torch' if any step of it ran in torch.

Build, cache and CPU emulation: those of fused.py (fused._build, fused.load_backend with this module's _Backend).
"""

import ctypes
import os

import torch

import neoqcd.fused as fused
import neoqcd.smearing_jacobian as sj


def _horner(coefficients, x):
    # C expression of sum_n coefficients[n] x^n by Horner's rule, the coefficients exact (%.17g)
    expr = f'(T){coefficients[-1]:.17g}'
    for c in coefficients[-2::-1]:
        expr = f'({expr} * {x} + (T){c:.17g})'
    return expr


_SOURCE = r"""
// neoqcd: the stout smearing step of smearing_jacobian.smearing_step on given links and staple sums, its
// inverse at fixed staple sums, and the coefficients of exp(iQ), SU(3); one source for CUDA (nvcc) and the
// host C++ compiler (CPU emulation). Ported from snf_SUN (snf_code/fused_smearing.py); conventions:
// src/neoqcd/fused_smearing.py.
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

#define SNF_THREADS @THREADS@  // threads per block
#define SNF_MIN_BLOCKS @MIN_BLOCKS@  // __launch_bounds__: blocks per SM the registers must allow (1: up to 255 each)
#define SNF_GEN 8        // generators of SU(3)
// smearing_jacobian.C1_SERIES, smearing_jacobian.N_SERIES, smearing_jacobian.XI_SERIES
#define SNF_C1_SERIES @C1_SERIES@
#define SNF_N_SERIES @N_SERIES@
#define SNF_XI_SERIES @XI_SERIES@

// Arguments of the stout step alone on n given links and staple sums (snf_stout_links; mirrored by the ctypes
// structure _LinksArgs in fused_smearing.py, as the structures below: the host checks their sizes through
// snf_stout_*_args_size())
struct SnfStoutLinksArgs {
    const void* U;                    // (n, 3, 3) complex: links
    const void* C;                    // (n, 3, 3) complex: staple sums
    void* Up;                         // (n, 3, 3) complex: smeared links
    void* logdet;                     // (n,) real: log|det J|
    int64_t n;
    int32_t dtype, device;
};

// Arguments of the inverse of the stout step on n smeared links at their staple sums (snf_stout_links_inverse,
// _InverseArgs)
struct SnfStoutInverseArgs {
    const void* Up;                   // (n, 3, 3) complex: smeared links U'
    const void* C;                    // (n, 3, 3) complex: staple sums
    void* U;                          // (n, 3, 3) complex: links U, X(U) U = U' (written)
    void* logdet;                     // (n,) real: log|det| of the inverse map, -log|det J| at U (written)
    void* residual;                   // (n,) real: max over the entries of |X(U) U - U'| (written)
    int64_t n;
    int32_t iterations, dtype, device, unused;
};

// Arguments of the coefficients alone (snf_stout_coefficients, for the tests): f, b1, b2, d11, d10, d00
// of smearing_jacobian.exp_coefficients(c0, c1, order=2) at n points (_CoeffArgs)
struct SnfStoutCoeffArgs {
    const void* c0;                   // (n,) real
    const void* c1;                   // (n,) real
    void* out;                        // (n, 6, 3) complex
    int64_t n;
    int32_t dtype, device;
};

// Complex and 3x3 helpers, as in fused.py. Aligned as a pair: one 16-byte load per complex128
template <typename T> struct alignas(2 * sizeof(T)) SnfC { T re, im; };

template <typename T> SNF_HD SnfC<T> c_make(T re, T im) { SnfC<T> z; z.re = re; z.im = im; return z; }
template <typename T> SNF_HD SnfC<T> c_add(SnfC<T> a, SnfC<T> b) { return c_make<T>(a.re + b.re, a.im + b.im); }
template <typename T> SNF_HD SnfC<T> c_sub(SnfC<T> a, SnfC<T> b) { return c_make<T>(a.re - b.re, a.im - b.im); }
template <typename T> SNF_HD SnfC<T> c_conj(SnfC<T> a) { return c_make<T>(a.re, -a.im); }
// the product of c10::complex
template <typename T> SNF_HD SnfC<T> c_mul(SnfC<T> a, SnfC<T> b) {
    return c_make<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
// real times complex (torch promotes the real factor to complex: the same numbers)
template <typename T> SNF_HD SnfC<T> c_scale(T s, SnfC<T> a) { return c_make<T>(s * a.re, s * a.im); }
// i a
template <typename T> SNF_HD SnfC<T> c_i(SnfC<T> a) { return c_make<T>(-a.im, a.re); }

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

// ---------------------------------------------------------------------------------------------------
// Coefficients of exp(iQ) = f0 + f1 Q + f2 Q^2 and their derivatives b1_j = df_j/dc1, b2_j = df_j/dc0
// (smearing_jacobian.exp_coefficients, order 1); with ORDER 2 also d11_j = d2f_j/dc1^2, d10_j = d2f_j/dc1 dc0,
// d00_j = d2f_j/dc0^2 (order 2, for the backward). The operations of order 1 are the same with ORDER 2.
// ---------------------------------------------------------------------------------------------------

// The Taylor series truncated after SNF_N_SERIES terms (smearing_jacobian.series_polynomials):
// exp(iQ) = sum_n (iQ)^n/n!, Q^n = a_n + b_n Q + c_n Q^2, (a, b, c)_{n+1} = (c0 c_n, a_n + c1 c_n, b_n),
// summed as s = (a, b, c)_n/n! with its derivatives p = ds/dc1 and q = ds/dc0, and for ORDER 2
// pp = dp/dc1, pq = dp/dc0, qq = dq/dc0 (the recursion differentiated once more)
template <typename T, int ORDER>
SNF_HD void snf_series(T c0, T c1, SnfC<T> f[3], SnfC<T> b1[3], SnfC<T> b2[3], SnfC<T> d11[3], SnfC<T> d10[3],
                       SnfC<T> d00[3]) {
    T s0 = 1, s1 = 0, s2 = 0, p0 = 0, p1 = 0, p2 = 0, q0 = 0, q1 = 0, q2 = 0;
    T pp0 = 0, pp1 = 0, pp2 = 0, pq0 = 0, pq1 = 0, pq2 = 0, qq0 = 0, qq1 = 0, qq2 = 0;
    T fr[3] = {0, 0, 0}, fi[3] = {0, 0, 0}, b1r[3] = {0, 0, 0}, b1i[3] = {0, 0, 0}, b2r[3] = {0, 0, 0}, b2i[3] = {0, 0, 0};
    T d11r[3] = {0, 0, 0}, d11i[3] = {0, 0, 0}, d10r[3] = {0, 0, 0}, d10i[3] = {0, 0, 0}, d00r[3] = {0, 0, 0}, d00i[3] = {0, 0, 0};
    SNF_UNROLL
    for (int n = 0; n < SNF_N_SERIES; ++n) {
        // i^n: real for even n, imaginary for odd n, sign + for n % 4 < 2 (sg x exact)
        const T sg = n % 4 < 2 ? (T)1 : (T)-1;
        if (n % 2 == 0) {
            fr[0] += sg * s0; fr[1] += sg * s1; fr[2] += sg * s2;
            b1r[0] += sg * p0; b1r[1] += sg * p1; b1r[2] += sg * p2;
            b2r[0] += sg * q0; b2r[1] += sg * q1; b2r[2] += sg * q2;
            if (ORDER == 2) {
                d11r[0] += sg * pp0; d11r[1] += sg * pp1; d11r[2] += sg * pp2;
                d10r[0] += sg * pq0; d10r[1] += sg * pq1; d10r[2] += sg * pq2;
                d00r[0] += sg * qq0; d00r[1] += sg * qq1; d00r[2] += sg * qq2;
            }
        } else {
            fi[0] += sg * s0; fi[1] += sg * s1; fi[2] += sg * s2;
            b1i[0] += sg * p0; b1i[1] += sg * p1; b1i[2] += sg * p2;
            b2i[0] += sg * q0; b2i[1] += sg * q1; b2i[2] += sg * q2;
            if (ORDER == 2) {
                d11i[0] += sg * pp0; d11i[1] += sg * pp1; d11i[2] += sg * pp2;
                d10i[0] += sg * pq0; d10i[1] += sg * pq1; d10i[2] += sg * pq2;
                d00i[0] += sg * qq0; d00i[1] += sg * qq1; d00i[2] += sg * qq2;
            }
        }
        const T inv = (T)1 / (T)(n + 1);
        if (ORDER == 2) {
            // from the terms n of s, p, q before they are advanced
            const T w0 = (c0 * pp2) * inv, w1 = (pp0 + (T)2 * p2 + c1 * pp2) * inv, w2 = pp1 * inv;
            const T x0 = (p2 + c0 * pq2) * inv, x1 = (pq0 + q2 + c1 * pq2) * inv, x2 = pq1 * inv;
            const T y0 = ((T)2 * q2 + c0 * qq2) * inv, y1 = (qq0 + c1 * qq2) * inv, y2 = qq1 * inv;
            pp0 = w0; pp1 = w1; pp2 = w2;
            pq0 = x0; pq1 = x1; pq2 = x2;
            qq0 = y0; qq1 = y1; qq2 = y2;
        }
        const T t0 = (c0 * s2) * inv, t1 = (s0 + c1 * s2) * inv, t2 = s1 * inv;
        const T u0 = (c0 * p2) * inv, u1 = (p0 + s2 + c1 * p2) * inv, u2 = p1 * inv;
        const T v0 = (s2 + c0 * q2) * inv, v1 = (q0 + c1 * q2) * inv, v2 = q1 * inv;
        s0 = t0; s1 = t1; s2 = t2;
        p0 = u0; p1 = u1; p2 = u2;
        q0 = v0; q1 = v1; q2 = v2;
    }
    SNF_UNROLL
    for (int j = 0; j < 3; ++j) {
        f[j] = c_make<T>(fr[j], fi[j]);
        b1[j] = c_make<T>(b1r[j], b1i[j]);
        b2[j] = c_make<T>(b2r[j], b2i[j]);
        if (ORDER == 2) {
            d11[j] = c_make<T>(d11r[j], d11i[j]);
            d10[j] = c_make<T>(d10r[j], d10i[j]);
            d00[j] = c_make<T>(d00r[j], d00i[j]);
        }
    }
}

// xi0 = sin(w)/w and xi1 = xi0'/w (smearing_jacobian.xi_functions): Taylor series (smearing_jacobian.XI_TAYLOR,
// Horner's rule in w^2) below |w| = SNF_XI_SERIES, closed forms above
template <typename T>
SNF_HD void snf_xi(T w, T* xi0, T* xi1) {
    const T w2 = w * w;
    if (fabs(w) < (T)SNF_XI_SERIES) {
        *xi0 = @XI0@;
        *xi1 = @XI1@;
    } else {
        const T s = sin(w);
        *xi0 = s / w;
        *xi1 = cos(w) / w2 - s / (w2 * w);
    }
}

// xi2 = -xi1'/w = (xi0 + 3 xi1)/w^2 (smearing_jacobian.xi_functions), from xi0 and xi1 of snf_xi
template <typename T>
SNF_HD T snf_xi2(T w, T xi0, T xi1) {
    const T w2 = w * w;
    if (fabs(w) < (T)SNF_XI_SERIES) return @XI2@;
    return (xi0 + (T)3 * xi1) / w2;
}

// The closed forms of Morningstar and Peardon (smearing_jacobian.mp_coefficients), for c1 >= C1_SERIES:
// evaluated at |c0|, then f_j(-c0) = (-1)^j f_j(c0)^*, with one more sign per derivative in c0
template <typename T, int ORDER>
SNF_HD void snf_mp(T c0, T c1, SnfC<T> f[3], SnfC<T> b1[3], SnfC<T> b2[3], SnfC<T> d11[3], SnfC<T> d10[3],
                   SnfC<T> d00[3]) {
    typedef SnfC<T> C;
    const bool neg = c0 < 0;
    const T c0max = (T)2 * pow(c1 / (T)3, (T)1.5);
    // |c0|/c0max clamped one representable step below 1, as smearing_jacobian.mp_coefficients does
    // (snf_SUN: at 1)
    const T ratio_max = (T)1 - (T)2.220446049250313e-16;  // 1 - torch.finfo(torch.float64).eps
    T ratio = fabs(c0) / c0max;
    if (ratio > ratio_max) ratio = ratio_max;
    const T theta = acos(ratio);
    const T u = sqrt(c1 / (T)3) * cos(theta / (T)3);
    const T w = sqrt(c1) * sin(theta / (T)3);
    const T u2 = u * u, w2 = w * w;
    T xi0, xi1;
    snf_xi(w, &xi0, &xi1);
    const T cw = cos(w);
    const C eu = c_make<T>(cos(u), sin(u));
    const C e2u = c_mul(eu, eu), emu = c_conj(eu);

    // h_j, r1_j = dh_j/du, r2_j = (1/w) dh_j/dw, the expressions of mp_coefficients expanded into
    // real and imaginary parts
    C h[3], r1[3], r2[3];
    h[0] = c_add(c_scale(u2 - w2, e2u), c_mul(emu, c_make<T>((T)8 * u2 * cw, (T)2 * u * ((T)3 * u2 + w2) * xi0)));
    h[1] = c_sub(c_scale((T)2 * u, e2u), c_mul(emu, c_make<T>((T)2 * u * cw, -((T)3 * u2 - w2) * xi0)));
    h[2] = c_sub(e2u, c_mul(emu, c_make<T>(cw, (T)3 * u * xi0)));
    // 2 (u + i (u2 - w2)) e2u + 2 emu (4 u (2 - i u) cw + i (9 u2 + w2 - i u (3 u2 + w2)) xi0)
    r1[0] = c_add(c_scale((T)2, c_mul(c_make<T>(u, u2 - w2), e2u)),
                  c_scale((T)2, c_mul(emu, c_make<T>((T)8 * u * cw + u * ((T)3 * u2 + w2) * xi0,
                                                     -(T)4 * u2 * cw + ((T)9 * u2 + w2) * xi0))));
    // 2 (1 + 2 i u) e2u + emu (-2 (1 - i u) cw + i (6 u + i (w2 - 3 u2)) xi0)
    r1[1] = c_add(c_scale((T)2, c_mul(c_make<T>(1, (T)2 * u), e2u)),
                  c_mul(emu, c_make<T>(-(T)2 * cw + ((T)3 * u2 - w2) * xi0, (T)2 * u * cw + (T)6 * u * xi0)));
    // 2 i e2u + i emu (cw - 3 (1 - i u) xi0)
    r1[2] = c_add(c_scale((T)2, c_i(e2u)), c_i(c_mul(emu, c_make<T>(cw - (T)3 * xi0, (T)3 * u * xi0))));
    // -2 e2u + 2 i u emu (cw + (1 + 4 i u) xi0 + 3 u2 xi1)
    r2[0] = c_add(c_scale((T)-2, e2u),
                  c_scale((T)2 * u, c_i(c_mul(emu, c_make<T>(cw + xi0 + (T)3 * u2 * xi1, (T)4 * u * xi0)))));
    // -i emu (cw + (1 + 2 i u) xi0 - 3 u2 xi1)
    r2[1] = c_i(c_mul(emu, c_make<T>(-(cw + xi0 - (T)3 * u2 * xi1), -(T)2 * u * xi0)));
    // emu (xi0 - 3 i u xi1)
    r2[2] = c_mul(emu, c_make<T>(xi0, -(T)3 * u * xi1));

    // ORDER 2: the second derivatives r1_u = dr1/du, r1_w = (1/w) dr1/dw, r2_w = (1/w) dr2/dw of
    // mp_coefficients, each written as in smearing_jacobian.py (its expression in the comment)
    C r1u[3], r1w[3], r2w[3];
    if (ORDER == 2) {
        const T xi2 = snf_xi2(w, xi0, xi1);
        // 2 e2u (1 + 4 i u - 2 u2 + 2 w2) - 2 emu (4 (-2 + 4 i u + u2) cw + i (18 i u2 + 3 u2 u + 2 i w2 + u (-18 + w2)) xi0)
        r1u[0] = c_sub(c_scale((T)2, c_mul(e2u, c_make<T>((T)1 - (T)2 * u2 + (T)2 * w2, (T)4 * u))),
                       c_scale((T)2, c_mul(emu, c_make<T>((T)4 * ((T)-2 + u2) * cw - ((T)18 * u2 + (T)2 * w2) * xi0,
                                                          (T)16 * u * cw + ((T)3 * u2 * u + u * (w2 - (T)18)) * xi0))));
        // 8 e2u (i - u) + emu (2 (2 i + u) cw + (12 u - 3 i u2 + i (6 + w2)) xi0)
        r1u[1] = c_add(c_scale((T)8, c_mul(e2u, c_make<T>(-u, 1))),
                       c_mul(emu, c_make<T>((T)2 * u * cw + (T)12 * u * xi0, (T)4 * cw + ((T)6 + w2 - (T)3 * u2) * xi0)));
        // -4 e2u + emu (cw + 3 i (2 i + u) xi0)
        r1u[2] = c_add(c_scale((T)-4, e2u), c_mul(emu, c_make<T>(cw - (T)6 * xi0, (T)3 * u * xi0)));
        // -4 i e2u + 2 emu ((i + u) cw + (i - 7 u + 4 i u2) xi0 + 3 u2 (3 i + u) xi1)
        r1w[0] = c_add(c_scale((T)-4, c_i(e2u)),
                       c_scale((T)2, c_mul(emu, c_make<T>(u * cw - (T)7 * u * xi0 + (T)3 * u2 * u * xi1,
                                                          cw + ((T)1 + (T)4 * u2) * xi0 + (T)9 * u2 * xi1))));
        // emu (-cw + (1 - 2 i u) xi0 + 3 u (2 i + u) xi1)
        r1w[1] = c_mul(emu, c_make<T>(-cw + xi0 + (T)3 * u2 * xi1, (T)-2 * u * xi0 + (T)6 * u * xi1));
        // -emu (i xi0 + 3 (i + u) xi1)
        r1w[2] = c_scale((T)-1, c_mul(emu, c_make<T>((T)3 * u * xi1, xi0 + (T)3 * xi1)));
        // 2 i emu u ((1 + 4 i u) xi1 - xi0 - 3 u2 xi2)
        r2w[0] = c_scale((T)2 * u, c_i(c_mul(emu, c_make<T>(xi1 - xi0 - (T)3 * u2 * xi2, (T)4 * u * xi1))));
        // i emu (-(1 + 2 i u) xi1 + xi0 - 3 u2 xi2)
        r2w[1] = c_i(c_mul(emu, c_make<T>(xi0 - xi1 - (T)3 * u2 * xi2, (T)-2 * u * xi1)));
        // emu (xi1 + 3 i u xi2)
        r2w[2] = c_mul(emu, c_make<T>(xi1, (T)3 * u * xi2));
    }

    const T den = (T)9 * u2 - w2;
    const T a = (T)0.5 * ((T)3 * u2 - w2);
    SNF_UNROLL
    for (int j = 0; j < 3; ++j) {
        const C fj = c_scale((T)1 / den, h[j]);
        const C f_u = c_scale((T)1 / den, c_sub(r1[j], c_scale((T)18 * u, fj)));
        const C f_w = c_scale((T)1 / den, c_add(r2[j], c_scale((T)2, fj)));
        C b1j = c_scale((T)1 / den, c_add(c_scale(u, f_u), c_scale(a, f_w)));
        C b2j = c_scale((T)1 / den, c_sub(c_scale((T)0.5, f_u), c_scale((T)1.5 * u, f_w)));
        C d11j, d10j, d00j;
        if (ORDER == 2) {
            // the second derivatives of mp_coefficients (order 2), from the unsigned b1, b2
            const T id = (T)1 / den;
            const C f_uu = c_scale(id, c_sub(c_sub(r1u[j], c_scale((T)18, fj)), c_scale((T)36 * u, f_u)));
            const C f_uw = c_scale(id, c_add(c_sub(r1w[j], c_scale((T)18 * u, f_w)), c_scale((T)2, f_u)));
            const C f_ww = c_scale(id, c_add(r2w[j], c_scale((T)4, f_w)));
            const C b1_u = c_scale(id, c_sub(c_add(c_add(c_add(f_u, c_scale(u, f_uu)), c_scale((T)3 * u, f_w)), c_scale(a, f_uw)),
                                             c_scale((T)18 * u, b1j)));
            const C b1_w = c_scale(id, c_add(c_add(c_sub(c_scale(u, f_uw), f_w), c_scale(a, f_ww)), c_scale((T)2, b1j)));
            const C b2_u = c_scale(id, c_sub(c_sub(c_sub(c_scale((T)0.5, f_uu), c_scale((T)1.5, f_w)), c_scale((T)1.5 * u, f_uw)),
                                             c_scale((T)18 * u, b2j)));
            const C b2_w = c_scale(id, c_add(c_sub(c_scale((T)0.5, f_uw), c_scale((T)1.5 * u, f_ww)), c_scale((T)2, b2j)));
            d11j = c_scale(id, c_add(c_scale(u, b1_u), c_scale(a, b1_w)));
            d10j = c_scale(id, c_sub(c_scale((T)0.5, b1_u), c_scale((T)1.5 * u, b1_w)));
            d00j = c_scale(id, c_sub(c_scale((T)0.5, b2_u), c_scale((T)1.5 * u, b2_w)));
        }
        C fjs = fj;
        if (neg) {
            // (-1)^j conj, and one more sign for each c0-derivative
            const T sj = j == 1 ? (T)-1 : (T)1;
            fjs = c_scale(sj, c_conj(fj));
            b1j = c_scale(sj, c_conj(b1j));
            b2j = c_scale(-sj, c_conj(b2j));
            if (ORDER == 2) {
                d11j = c_scale(sj, c_conj(d11j));
                d10j = c_scale(-sj, c_conj(d10j));
                d00j = c_scale(sj, c_conj(d00j));
            }
        }
        f[j] = fjs;
        b1[j] = b1j;
        b2[j] = b2j;
        if (ORDER == 2) {
            d11[j] = d11j;
            d10[j] = d10j;
            d00[j] = d00j;
        }
    }
}

// smearing_jacobian.exp_coefficients: the series below C1_SERIES, the closed forms above (d11, d10, d00 are
// written only with ORDER 2)
template <typename T, int ORDER>
SNF_HD void snf_coefficients(T c0, T c1, SnfC<T> f[3], SnfC<T> b1[3], SnfC<T> b2[3], SnfC<T> d11[3], SnfC<T> d10[3],
                             SnfC<T> d00[3]) {
    if (c1 < (T)SNF_C1_SERIES) snf_series<T, ORDER>(c0, c1, f, b1, b2, d11, d10, d00);
    else snf_mp<T, ORDER>(c0, c1, f, b1, b2, d11, d10, d00);
}

// ---------------------------------------------------------------------------------------------------
// The tangent-space basis e_0..e_7 = lambda_1..lambda_7, diag(1, 1, -2), and its coordinates
// ---------------------------------------------------------------------------------------------------

// z times the entry kind of a generator: 0 zero, 1 one, 2 minus one, 3 i, 4 -i, 5 -2 (exact)
template <typename T> SNF_HD SnfC<T> c_unit(int kind, SnfC<T> z) {
    switch (kind) {
    case 1: return z;
    case 2: return c_make<T>(-z.re, -z.im);
    case 3: return c_i(z);
    case 4: return c_make<T>(z.im, -z.re);
    case 5: return c_make<T>((T)-2 * z.re, (T)-2 * z.im);
    default: return c_make<T>(0, 0);
    }
}

// Row r of e_b has one entry, (e_b)_{r, src} = kind (c_unit), or none (kind 0)
SNF_HD void snf_gen_row(int b, int r, int* src, int* kind) {
    int s = 0, k = 0;
    switch (b) {
    case 0: s = r == 0 ? 1 : 0; k = r < 2 ? 1 : 0; break;                // E01 + E10
    case 1: s = r == 0 ? 1 : 0; k = r == 0 ? 4 : r == 1 ? 3 : 0; break;  // -i E01 + i E10
    case 2: s = r; k = r == 0 ? 1 : r == 1 ? 2 : 0; break;               // E00 - E11
    case 3: s = r == 0 ? 2 : 0; k = r == 1 ? 0 : 1; break;               // E02 + E20
    case 4: s = r == 0 ? 2 : 0; k = r == 0 ? 4 : r == 2 ? 3 : 0; break;  // -i E02 + i E20
    case 5: s = r == 1 ? 2 : 1; k = r == 0 ? 0 : 1; break;               // E12 + E21
    case 6: s = r == 1 ? 2 : 1; k = r == 0 ? 0 : r == 1 ? 4 : 3; break;  // -i E12 + i E21
    default: s = r; k = r == 2 ? 5 : 1; break;                           // diag(1, 1, -2)
    }
    *src = s;
    *kind = k;
}

// Tr(e_a G) / Tr(e_a e_a), complex (real for hermitian G)
template <typename T> SNF_HD SnfC<T> snf_coord(int a, const SnfC<T> G[9]) {
    typedef SnfC<T> C;
    switch (a) {
    case 0: return c_scale((T)0.5, c_add(G[1], G[3]));            // G01 + G10
    case 1: return c_scale((T)0.5, c_i(c_sub(G[1], G[3])));       // i (G01 - G10)
    case 2: return c_scale((T)0.5, c_sub(G[0], G[4]));            // G00 - G11
    case 3: return c_scale((T)0.5, c_add(G[2], G[6]));            // G02 + G20
    case 4: return c_scale((T)0.5, c_i(c_sub(G[2], G[6])));       // i (G02 - G20)
    case 5: return c_scale((T)0.5, c_add(G[5], G[7]));            // G12 + G21
    case 6: return c_scale((T)0.5, c_i(c_sub(G[5], G[7])));       // i (G12 - G21)
    default: {                                                    // G00 + G11 - 2 G22
        const C t = c_sub(c_add(G[0], G[4]), c_scale((T)2, G[8]));
        return c_make<T>(t.re / (T)6, t.im / (T)6);
    }
    }
}

// ---------------------------------------------------------------------------------------------------
// Pieces of the stout step of one link, shared by the step (snf_stout_step) and the fixed point of its
// inverse (snf_left_factor); in snf_SUN also by the backward (not ported)
// ---------------------------------------------------------------------------------------------------

// Q from Om = C U^dagger (StepQuantities): Z = Om^dagger - Om, Q = i/2 (Z - Tr(Z)/3), Q2 = Q Q,
// c1 = Re Tr(Q^2)/2, c0 = Re Tr(Q Q^2)/3
template <typename T>
SNF_HD void snf_q(const SnfC<T> Om[9], SnfC<T> Q[9], SnfC<T> Q2[9], T* c1_out, T* c0_out) {
    typedef SnfC<T> C;
    {
        C Z[9];
        SNF_UNROLL
        for (int i = 0; i < 3; ++i) {
            SNF_UNROLL
            for (int j = 0; j < 3; ++j) Z[3 * i + j] = c_sub(c_conj(Om[3 * j + i]), Om[3 * i + j]);
        }
        const C tr = c_add(c_add(Z[0], Z[4]), Z[8]);
        const C tr3 = c_make<T>(tr.re / (T)3, tr.im / (T)3);
        SNF_UNROLL
        for (int e = 0; e < 9; ++e) {
            const C z = (e % 4 == 0) ? c_sub(Z[e], tr3) : Z[e];
            Q[e] = c_make<T>((T)-0.5 * z.im, (T)0.5 * z.re);
        }
    }
    m_mul<T, false, false>(Q2, Q, Q);
    const T c1 = (T)0.5 * ((Q2[0].re + Q2[4].re) + Q2[8].re);
    T c0 = 0;
    SNF_UNROLL
    for (int i = 0; i < 3; ++i) {
        SNF_UNROLL
        for (int j = 0; j < 3; ++j) c0 += Q[3 * i + j].re * Q2[3 * j + i].re - Q[3 * i + j].im * Q2[3 * j + i].im;
    }
    *c1_out = c1;
    *c0_out = c0 / (T)3;
}

// X = f0 + f1 Q + f2 Q^2
template <typename T>
SNF_HD void snf_exp(const SnfC<T> f[3], const SnfC<T> Q[9], const SnfC<T> Q2[9], SnfC<T> X[9]) {
    SNF_UNROLL
    for (int e = 0; e < 9; ++e) {
        const SnfC<T> x = c_mul(f[1], Q[e]);
        X[e] = c_add(e % 4 == 0 ? c_add(f[0], x) : x, c_mul(f[2], Q2[e]));
    }
}

// The polynomial part of the columns of B (see fused_smearing.py): with F_j = f_j^*, X^dagger Q^j has
// the coefficients bQ[j] of Q and bQ2[j] of Q^2, and the column b gets dc1 v1 + dc0 v0, with
// v1 = g1Q e(Q) + g1Q2 e(Q^2), v0 = g0Q e(Q) + g0Q2 e(Q^2), g1Q = Im(sum_j b1_j bQ[j]), g1Q2 = Im(sum_j
// b1_j bQ2[j]), g0Q, g0Q2 the same with b2, e(.) the coordinates. gq = (g1Q, g1Q2, g0Q, g0Q2).
template <typename T>
SNF_HD void snf_poly(const SnfC<T> f[3], const SnfC<T> b1[3], const SnfC<T> b2[3], T c1, T c0, const SnfC<T> Q[9],
                     const SnfC<T> Q2[9], T v1[SNF_GEN], T v0[SNF_GEN], T gq[4]) {
    typedef SnfC<T> C;
    const C F0 = c_conj(f[0]), F1 = c_conj(f[1]), F2 = c_conj(f[2]);
    const C F0c = c_add(F0, c_scale(c1, F2));
    const C bQ[3] = {F1, F0c, c_add(c_scale(c1, F1), c_scale(c0, F2))};
    const C bQ2[3] = {F2, F1, F0c};
    T g1Q = 0, g1Q2 = 0, g0Q = 0, g0Q2 = 0;
    SNF_UNROLL
    for (int j = 0; j < 3; ++j) {
        g1Q += c_mul(b1[j], bQ[j]).im;
        g1Q2 += c_mul(b1[j], bQ2[j]).im;
        g0Q += c_mul(b2[j], bQ[j]).im;
        g0Q2 += c_mul(b2[j], bQ2[j]).im;
    }
    SNF_UNROLL
    for (int a = 0; a < SNF_GEN; ++a) {
        const T qa = snf_coord(a, Q).re, q2a = snf_coord(a, Q2).re;
        v1[a] = g1Q * qa + g1Q2 * q2a;
        v0[a] = g0Q * qa + g0Q2 * q2a;
    }
    gq[0] = g1Q; gq[1] = g1Q2; gq[2] = g0Q; gq[3] = g0Q2;
}

// S = (s M + (s M)^dagger) + d 1, for real s and d, built hermitian (the lower triangle the conjugate of
// the upper one, the diagonal real: the compiler keeps 9 numbers)
template <typename T> SNF_HD void m_herm(T s, T d, const SnfC<T> M[9], SnfC<T> S[9]) {
    SNF_UNROLL
    for (int r = 0; r < 3; ++r) {
        SNF_UNROLL
        for (int c = 0; c < 3; ++c) {
            if (c < r) continue;
            const SnfC<T> z = c_add(M[3 * r + c], c_conj(M[3 * c + r]));
            S[3 * r + c] = c == r ? c_make<T>(s * z.re + d, 0) : c_make<T>(s * z.re, s * z.im);
            if (c > r) S[3 * c + r] = c_conj(S[3 * r + c]);
        }
    }
}

// L(Y) = -(M + M^dagger)/2 + Re Tr(M)/3, M = Y Om^dagger: the change dQ of Q for a change Y of the
// hermitian generator of dU = i Y U (eq. (eq:dQ-Y)), hermitian; for Y = e_b (dQ of the columns of B)
// M is read off Om (snf_dq), otherwise one product
template <typename T> SNF_HD void snf_dq_of(const SnfC<T> M[9], SnfC<T> dQ[9]) {
    m_herm((T)-0.5, ((M[0].re + M[4].re) + M[8].re) / (T)3, M, dQ);
}

template <typename T> SNF_HD void snf_dq(int b, const SnfC<T> Om[9], SnfC<T> dQ[9]) {
    // (e_b Om^dagger)_{rc} = kind (Om^dagger)_{src, c} = kind Om_{c, src}^*
    SnfC<T> M[9];
    SNF_UNROLL
    for (int r = 0; r < 3; ++r) {
        int src, kind;
        snf_gen_row(b, r, &src, &kind);
        SNF_UNROLL
        for (int c = 0; c < 3; ++c) M[3 * r + c] = c_unit(kind, c_conj(Om[3 * c + src]));
    }
    snf_dq_of(M, dQ);
}

// Direction b of the columns of B: dQ = -(M + M^dagger)/2 + Re Tr(M)/3, M = e_b Om^dagger, P = Q dQ,
// dc1 = Re Tr(Q dQ) = Re Tr P, dc0 = Re Tr(Q^2 dQ) = Re Tr(Q P); returns dQ, Pp = P + P^dagger and
// H = f1 dQ + f2 Pp
template <typename T>
SNF_HD void snf_direction(int b, const SnfC<T> Om[9], const SnfC<T> Q[9], const SnfC<T> f[3], SnfC<T> dQ[9],
                          SnfC<T> Pp[9], SnfC<T> H[9], T* dc1_out, T* dc0_out) {
    snf_dq(b, Om, dQ);
    SnfC<T> P[9];
    m_mul<T, false, false>(P, Q, dQ);
    const T dc1 = (P[0].re + P[4].re) + P[8].re;
    T dc0 = 0;
    SNF_UNROLL
    for (int i = 0; i < 3; ++i) {
        SNF_UNROLL
        for (int j = 0; j < 3; ++j) dc0 += Q[3 * i + j].re * P[3 * j + i].re - Q[3 * i + j].im * P[3 * j + i].im;
    }
    m_herm((T)1, (T)0, P, Pp);
    SNF_UNROLL
    for (int e = 0; e < 9; ++e) H[e] = c_add(c_mul(f[1], dQ[e]), c_mul(f[2], Pp[e]));
    *dc1_out = dc1;
    *dc0_out = dc0;
}

// Column b of 1 + B: B_ab = dc1 v1_a + dc0 v0_a + Im e^a(X^dagger H) (see fused_smearing.py), and dc1, dc0
template <typename T>
SNF_HD void snf_column(int b, const SnfC<T> Om[9], const SnfC<T> Q[9], const SnfC<T> X[9], const SnfC<T> f[3],
                       const T v1[SNF_GEN], const T v0[SNF_GEN], T v[SNF_GEN], T* dc1_out, T* dc0_out) {
    SnfC<T> dQ[9], Pp[9], H[9], G[9];
    T dc1, dc0;
    snf_direction(b, Om, Q, f, dQ, Pp, H, &dc1, &dc0);
    *dc1_out = dc1;
    *dc0_out = dc0;
    m_mul<T, true, false>(G, X, H);
    SNF_UNROLL
    for (int a = 0; a < SNF_GEN; ++a) {
        const T bab = (dc1 * v1[a] + dc0 * v0[a]) + snf_coord(a, G).im;
        v[a] = a == b ? (T)1 + bab : bab;
    }
}

// ---------------------------------------------------------------------------------------------------
// The stout step of one link (smearing_jacobian.smearing_step): U' = exp(iQ) U, written to `out` (9 complex,
// possibly the link itself) before the Jacobian, so that it does not hold registers; returns log|det J|
// (0 without `jacobian`: U' only)
// ---------------------------------------------------------------------------------------------------
template <typename T>
SNF_HD T snf_stout_step(const SnfC<T> U[9], const SnfC<T> Cs[9], SnfC<T>* out, bool jacobian = true) {
    typedef SnfC<T> C;
    // Om = C U^dagger, Q (StepQuantities)
    C Om[9], Q[9], Q2[9];
    m_mul<T, false, true>(Om, Cs, U);
    T c1, c0;
    snf_q(Om, Q, Q2, &c1, &c0);

    C f[3], b1[3], b2[3], unused[3];
    snf_coefficients<T, 1>(c0, c1, f, b1, b2, unused, unused, unused);

    // X = f0 + f1 Q + f2 Q^2, U' = X U
    C X[9];
    snf_exp(f, Q, Q2, X);
    {
        C Up[9];
        m_mul<T, false, false>(Up, X, U);
        SNF_UNROLL
        for (int e = 0; e < 9; ++e) out[e] = Up[e];
    }
    if (!jacobian) return 0;

    // Jacobian (see fused_smearing.py): log|det(1 + B)|, the polynomial part of the columns
    T v1[SNF_GEN], v0[SNF_GEN], gq[4];
    snf_poly(f, b1, b2, c1, c0, Q, Q2, v1, v0, gq);

    // LU with partial pivoting, column by column (left-looking): L[i][k] the multipliers, piv[k] the row
    // exchanged with row k at step k, det the product of the pivots. Every loop has a constant trip count
    // (the triangular ranges as conditions), so that nvcc unrolls them all and L, piv, v stay in registers
    T L[SNF_GEN][SNF_GEN];
    int piv[SNF_GEN];
    T det = 1;
    bool singular = false;
    SNF_UNROLL
    for (int b = 0; b < SNF_GEN; ++b) {
        T v[SNF_GEN], dc1, dc0;
        snf_column(b, Om, Q, X, f, v1, v0, v, &dc1, &dc0);
        // the earlier row exchanges, then the elimination with the earlier columns of L
        SNF_UNROLL
        for (int k = 0; k < SNF_GEN; ++k) {
            SNF_UNROLL
            for (int r = 0; r < SNF_GEN; ++r) {
                if (k < b && r > k && piv[k] == r) { const T t = v[k]; v[k] = v[r]; v[r] = t; }
            }
        }
        SNF_UNROLL
        for (int k = 0; k < SNF_GEN; ++k) {
            SNF_UNROLL
            for (int i = 0; i < SNF_GEN; ++i) {
                if (k < b && i > k) v[i] -= L[i][k] * v[k];
            }
        }
        // pivot: the largest |v_r|, r >= b, exchanged with row b (in v and in the earlier columns of L)
        int p = b;
        T vmax = fabs(v[b]);
        SNF_UNROLL
        for (int r = 0; r < SNF_GEN; ++r) {
            if (r > b && fabs(v[r]) > vmax) { vmax = fabs(v[r]); p = r; }
        }
        piv[b] = p;
        SNF_UNROLL
        for (int r = 0; r < SNF_GEN; ++r) {
            if (r > b && p == r) {
                const T t = v[b]; v[b] = v[r]; v[r] = t;
                SNF_UNROLL
                for (int k = 0; k < SNF_GEN; ++k) {
                    if (k < b) { const T l = L[b][k]; L[b][k] = L[r][k]; L[r][k] = l; }
                }
            }
        }
        singular = singular || v[b] == (T)0;
        det *= v[b];
        const T inv = (T)1 / v[b];
        SNF_UNROLL
        for (int i = 0; i < SNF_GEN; ++i) {
            if (i > b) L[i][b] = v[i] * inv;
        }
    }
    // an exactly singular Jacobian: -inf, as slogdet
    return singular ? -(T)INFINITY : log(fabs(det));
}

// ---------------------------------------------------------------------------------------------------
// The inverse of the stout step of one link at fixed staple sum (new in neoqcd)
// ---------------------------------------------------------------------------------------------------

// X(U) = exp(iQ) of the stout step of U with staple sum Cs (StepQuantities(U, C).X): the device functions of
// snf_stout_step, without U' and the Jacobian
template <typename T>
SNF_HD void snf_left_factor(const SnfC<T> U[9], const SnfC<T> Cs[9], SnfC<T> X[9]) {
    typedef SnfC<T> C;
    C Om[9], Q[9], Q2[9];
    m_mul<T, false, true>(Om, Cs, U);
    T c1, c0;
    snf_q(Om, Q, Q2, &c1, &c0);
    C f[3], b1[3], b2[3], unused[3];
    snf_coefficients<T, 1>(c0, c1, f, b1, b2, unused, unused, unused);
    snf_exp(f, Q, Q2, X);
}

// smearing_jacobian.inverse_smearing_step(U', C, fixed_iterations) on one link: U_0 = U', U_{k+1} =
// X(U_k)^dagger U' (sun.invert_left_update), `iterations` times; writes U, the residual max_ij |(X(U) U -
// U')_ij| (NaN if an entry is NaN, as torch's amax) and -log|det J| of the step at U
template <typename T>
SNF_HD void snf_inverse_one(const SnfStoutInverseArgs& a, int64_t tid) {
    typedef SnfC<T> C;
    C U[9], S[9];
    {
        C Up[9];
        m_load(Up, static_cast<const C*>(a.Up) + 9 * tid);
        m_load(S, static_cast<const C*>(a.C) + 9 * tid);
        SNF_UNROLL
        for (int e = 0; e < 9; ++e) U[e] = Up[e];
        for (int k = 0; k < a.iterations; ++k) {
            C X[9];
            snf_left_factor<T>(U, S, X);
            m_mul<T, true, false>(U, X, Up);
        }
        // the residual (sun.SUN_mul(left_factor(U), U) - Up)
        C X[9], XU[9];
        snf_left_factor<T>(U, S, X);
        m_mul<T, false, false>(XU, X, U);
        T residual = 0;
        bool nan = false;
        SNF_UNROLL
        for (int e = 0; e < 9; ++e) {
            const T d = hypot(XU[e].re - Up[e].re, XU[e].im - Up[e].im);
            nan = nan || d != d;
            residual = d > residual ? d : residual;
        }
        static_cast<T*>(a.residual)[tid] = nan ? (T)NAN : residual;
        C* out = static_cast<C*>(a.U) + 9 * tid;
        SNF_UNROLL
        for (int e = 0; e < 9; ++e) out[e] = U[e];
    }
    // log|det J| of the step at U (smearing_step(U, C)[1]); its U' is not needed
    C unused[9];
    static_cast<T*>(a.logdet)[tid] = -snf_stout_step<T>(U, S, unused);
}

// ---------------------------------------------------------------------------------------------------
// One thread per link: the entry points of snf_SUN kept here (snf_stout_links, snf_stout_coefficients) and
// the inverse
// ---------------------------------------------------------------------------------------------------

template <typename T>
SNF_HD void snf_one(const SnfStoutLinksArgs& a, int64_t tid) {
    typedef SnfC<T> C;
    C U[9], S[9];
    m_load(U, static_cast<const C*>(a.U) + 9 * tid);
    m_load(S, static_cast<const C*>(a.C) + 9 * tid);
    static_cast<T*>(a.logdet)[tid] = snf_stout_step<T>(U, S, static_cast<C*>(a.Up) + 9 * tid);
}

template <typename T>
SNF_HD void snf_coefficients_one(const SnfStoutCoeffArgs& a, int64_t tid) {
    typedef SnfC<T> C;
    C k[6][3];
    snf_coefficients<T, 2>(static_cast<const T*>(a.c0)[tid], static_cast<const T*>(a.c1)[tid], k[0], k[1], k[2], k[3],
                           k[4], k[5]);
    C* out = static_cast<C*>(a.out) + 18 * tid;
    for (int d = 0; d < 6; ++d)
        for (int j = 0; j < 3; ++j) out[3 * d + j] = k[d][j];
}

#ifdef __CUDACC__
template <typename T>
__global__ void __launch_bounds__(SNF_THREADS, SNF_MIN_BLOCKS) snf_links_kernel(const SnfStoutLinksArgs a) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < a.n) snf_one<T>(a, tid);
}

template <typename T>
__global__ void __launch_bounds__(SNF_THREADS, SNF_MIN_BLOCKS) snf_inverse_kernel(const SnfStoutInverseArgs a) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < a.n) snf_inverse_one<T>(a, tid);
}

template <typename T>
__global__ void __launch_bounds__(SNF_THREADS) snf_coefficients_kernel(const SnfStoutCoeffArgs a) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < a.n) snf_coefficients_one<T>(a, tid);
}
#endif

// n links: one CUDA launch on `stream`, or a loop over the thread index on the host
template <typename T>
static int snf_links(const SnfStoutLinksArgs& a, void* stream) {
    if (a.n == 0) return 0;
#ifdef __CUDACC__
    const int64_t blocks = (a.n + SNF_THREADS - 1) / SNF_THREADS;
    snf_links_kernel<T><<<(unsigned)blocks, SNF_THREADS, 0, (cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
#else
    (void)stream;
    for (int64_t tid = 0; tid < a.n; ++tid) snf_one<T>(a, tid);
    return 0;
#endif
}

template <typename T>
static int snf_links_inverse(const SnfStoutInverseArgs& a, void* stream) {
    if (a.n == 0) return 0;
#ifdef __CUDACC__
    const int64_t blocks = (a.n + SNF_THREADS - 1) / SNF_THREADS;
    snf_inverse_kernel<T><<<(unsigned)blocks, SNF_THREADS, 0, (cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
#else
    (void)stream;
    for (int64_t tid = 0; tid < a.n; ++tid) snf_inverse_one<T>(a, tid);
    return 0;
#endif
}

template <typename T>
static int snf_coefficients_all(const SnfStoutCoeffArgs& a, void* stream) {
    if (a.n == 0) return 0;
#ifdef __CUDACC__
    const int64_t blocks = (a.n + SNF_THREADS - 1) / SNF_THREADS;
    snf_coefficients_kernel<T><<<(unsigned)blocks, SNF_THREADS, 0, (cudaStream_t)stream>>>(a);
    return (int)cudaGetLastError();
#else
    (void)stream;
    for (int64_t tid = 0; tid < a.n; ++tid) snf_coefficients_one<T>(a, tid);
    return 0;
#endif
}

// Loads the module of every kernel on the current device (cudaFuncGetAttributes forces the lazy loading of
// CUDA_MODULE_LOADING=LAZY); with the cudaGetDevice of SNF_ON_DEVICE before it, the first calls into the CUDA
// runtime of this library, which initialize it. Nothing on the host build.
static int snf_load_kernels(void) {
#ifdef __CUDACC__
    cudaFuncAttributes attr;
    int err = (int)cudaFuncGetAttributes(&attr, (const void*)snf_links_kernel<double>);
    if (!err) err = (int)cudaFuncGetAttributes(&attr, (const void*)snf_inverse_kernel<double>);
    if (!err) err = (int)cudaFuncGetAttributes(&attr, (const void*)snf_coefficients_kernel<double>);
    return err;
#else
    return 0;
#endif
}

#ifdef __CUDACC__
// runs f on `device`, restoring the current device
#define SNF_ON_DEVICE(device, call)                                               \
    int previous = -1;                                                            \
    int err = (int)cudaGetDevice(&previous);                                      \
    if (err) return err;                                                          \
    if (previous != (device) && (err = (int)cudaSetDevice(device))) return err;   \
    const int code = (call);                                                      \
    if (previous != (device)) cudaSetDevice(previous);                            \
    return code;
#else
#define SNF_ON_DEVICE(device, call) return (call);
#endif

extern "C" {
int snf_stout_links_args_size(void) { return (int)sizeof(SnfStoutLinksArgs); }
int snf_stout_inverse_args_size(void) { return (int)sizeof(SnfStoutInverseArgs); }
int snf_stout_coeff_args_size(void) { return (int)sizeof(SnfStoutCoeffArgs); }

int snf_stout_is_cuda(void) {
#ifdef __CUDACC__
    return 1;
#else
    return 0;
#endif
}

// Initializes the CUDA runtime of this library on `device` and loads the kernels there (snf_load_kernels):
// called once per device when the library is loaded (fused.load_backend), so that neither happens later inside
// a CUDA graph capture. Returns 0 on success, a cudaError_t code otherwise.
int snf_stout_init(int device) {
    SNF_ON_DEVICE(device, snf_load_kernels())
}

// Returns 0 on success, -1 for an unsupported dtype, a cudaError_t code otherwise.
int snf_stout_links(const SnfStoutLinksArgs* a, void* stream) {
    SNF_ON_DEVICE(a->device, a->dtype == 0 ? snf_links<double>(*a, stream) : -1)
}

int snf_stout_links_inverse(const SnfStoutInverseArgs* a, void* stream) {
    SNF_ON_DEVICE(a->device, a->dtype == 0 ? snf_links_inverse<double>(*a, stream) : -1)
}

int snf_stout_coefficients(const SnfStoutCoeffArgs* a, void* stream) {
    SNF_ON_DEVICE(a->device, a->dtype == 0 ? snf_coefficients_all<double>(*a, stream) : -1)
}

const char* snf_stout_error_string(int code) {
    if (code == -1) return "unsupported dtype";
#ifdef __CUDACC__
    return cudaGetErrorString((cudaError_t)code);
#else
    return code == 0 ? "no error" : "unknown error";
#endif
}
}
"""

# Blocks of 128 threads per SM that the registers of a thread must allow (__launch_bounds__): 1 lets nvcc use up
# to 255 registers per thread (fewer spills, 2 blocks per SM on an A100), 3 caps them at 168, 4 at 128 (more
# threads to hide latency, more spills). Read once, from $NEOQCD_SMEARING_MIN_BLOCKS, for tuning on the GPU.
MIN_BLOCKS = int(os.environ.get('NEOQCD_SMEARING_MIN_BLOCKS', '1'))
# threads per block (SNF_THREADS)
THREADS = 128

KERNEL_SOURCE = (_SOURCE.replace('@C1_SERIES@', f'{float(sj.C1_SERIES):.17g}')
                 .replace('@MIN_BLOCKS@', str(MIN_BLOCKS))
                 .replace('@N_SERIES@', str(int(sj.N_SERIES)))
                 .replace('@THREADS@', str(THREADS))
                 .replace('@XI_SERIES@', f'{float(sj.XI_SERIES):.17g}')
                 .replace('@XI0@', _horner(sj.XI_TAYLOR[0], 'w2'))
                 .replace('@XI1@', _horner(sj.XI_TAYLOR[1], 'w2'))
                 .replace('@XI2@', _horner(sj.XI_TAYLOR[2], 'w2')))


class _LinksArgs(ctypes.Structure):
    # ctypes mirror of SnfStoutLinksArgs
    _fields_ = [
        ('U', ctypes.c_void_p),
        ('C', ctypes.c_void_p),
        ('Up', ctypes.c_void_p),
        ('logdet', ctypes.c_void_p),
        ('n', ctypes.c_int64),
        ('dtype', ctypes.c_int32),
        ('device', ctypes.c_int32),
    ]


class _InverseArgs(ctypes.Structure):
    # ctypes mirror of SnfStoutInverseArgs
    _fields_ = [
        ('Up', ctypes.c_void_p),
        ('C', ctypes.c_void_p),
        ('U', ctypes.c_void_p),
        ('logdet', ctypes.c_void_p),
        ('residual', ctypes.c_void_p),
        ('n', ctypes.c_int64),
        ('iterations', ctypes.c_int32),
        ('dtype', ctypes.c_int32),
        ('device', ctypes.c_int32),
        ('unused', ctypes.c_int32),
    ]


class _CoeffArgs(ctypes.Structure):
    # ctypes mirror of SnfStoutCoeffArgs
    _fields_ = [
        ('c0', ctypes.c_void_p),
        ('c1', ctypes.c_void_p),
        ('out', ctypes.c_void_p),
        ('n', ctypes.c_int64),
        ('dtype', ctypes.c_int32),
        ('device', ctypes.c_int32),
    ]

# ----------------------------------------------------------------------------------------------
# Build and load: fused.py's, with this source
# ----------------------------------------------------------------------------------------------

class _Backend(fused._Backend):
    # The loaded smearing library: run_links(args, stream) for the stout step, run_inverse(args, stream) for its
    # inverse, run_coefficients(args, stream) for the coefficients of order 2 alone; RuntimeError on a nonzero
    # return code. prepare(device) (fused._Backend) calls INIT once per device.
    SOURCE = KERNEL_SOURCE
    NAME = 'neoqcd_stout'
    INIT = 'snf_stout_init'

    def __init__(self, target, path):
        self.target = target
        self.path = str(path)
        lib = ctypes.CDLL(self.path)
        sizes = (('snf_stout_links_args_size', _LinksArgs), ('snf_stout_inverse_args_size', _InverseArgs),
                 ('snf_stout_coeff_args_size', _CoeffArgs))
        for name in [name for name, _ in sizes] + ['snf_stout_is_cuda']:
            getattr(lib, name).restype = ctypes.c_int
        for name, args in (('snf_stout_links', _LinksArgs), ('snf_stout_links_inverse', _InverseArgs),
                           ('snf_stout_coefficients', _CoeffArgs)):
            getattr(lib, name).argtypes = [ctypes.POINTER(args), ctypes.c_void_p]
            getattr(lib, name).restype = ctypes.c_int
        lib.snf_stout_init.argtypes = [ctypes.c_int]
        lib.snf_stout_init.restype = ctypes.c_int
        lib.snf_stout_error_string.argtypes = [ctypes.c_int]
        lib.snf_stout_error_string.restype = ctypes.c_char_p
        for name, args in sizes:
            if getattr(lib, name)() != ctypes.sizeof(args):
                raise fused.BackendUnavailable(f'argument layout mismatch ({name}): library '
                                               f'{getattr(lib, name)()} bytes, ctypes {ctypes.sizeof(args)}')
        if bool(lib.snf_stout_is_cuda()) != (target == 'cuda'):
            raise fused.BackendUnavailable(f'library {self.path} is not a {target} build')
        self._lib = lib
        self._prepared = set()

    def error_string(self, code):
        return self._lib.snf_stout_error_string(code).decode(errors='replace')

    def _check(self, code):
        if code != 0:
            raise RuntimeError(f'fused smearing ({self.target}) failed: code {code}: {self.error_string(code)}')

    def run_links(self, args, stream):
        self._check(self._lib.snf_stout_links(ctypes.byref(args), stream))

    def run_inverse(self, args, stream):
        self._check(self._lib.snf_stout_links_inverse(ctypes.byref(args), stream))

    def run_coefficients(self, args, stream):
        self._check(self._lib.snf_stout_coefficients(ctypes.byref(args), stream))


def load_backend(target, device=None):
    # The smearing library for target 'cuda' or 'cpu' (the emulation): fused.load_backend
    return fused.load_backend(target, device, _Backend)


def backend_status(target, device=None):
    # (available, library path or the reason why not)
    return fused.backend_status(target, device, _Backend)


def _stream(device):
    return torch.cuda.current_stream(device).cuda_stream if device.type == 'cuda' else None


def _device_index(device):
    return device.index if device.index is not None else (torch.cuda.current_device() if device.type == 'cuda' else 0)


def _target(device):
    return 'cuda' if device.type == 'cuda' else 'cpu'


def _plain(x, device):
    # x detached, on `device`, contiguous, with its lazy conjugation or negation applied: the memory the kernels
    # read through data_ptr
    return x.detach().to(device).resolve_conj().resolve_neg().contiguous()


def _check_links(what, A, B):
    if A.dtype != torch.complex128 or B.dtype != torch.complex128 or A.shape != B.shape or A.shape[-2:] != (3, 3):
        raise ValueError(f'{what} takes complex128 links and staple sums of the same shape (..., 3, 3)')

# ----------------------------------------------------------------------------------------------
# The stout step, its inverse, the coefficients
# ----------------------------------------------------------------------------------------------

def stout_links(U, C, backend=None):
    """smearing_jacobian.smearing_step(U, C) through the kernel, on the device of U (CUDA, or the CPU emulation):
    (U', log|det J|) for links U and staple sums C of shape (..., 3, 3), complex128, without gradients.
    `backend`: a library from load_backend (default: the one for the device of U)."""
    _check_links('stout_links', U, C)
    backend = load_backend(_target(U.device), U.device) if backend is None else backend
    U, C = _plain(U, U.device), _plain(C, U.device)
    Up = torch.empty_like(U)
    logdet = torch.empty(U.shape[:-2], dtype=torch.float64, device=U.device)
    a = _LinksArgs()
    a.U, a.C, a.Up, a.logdet = U.data_ptr(), C.data_ptr(), Up.data_ptr(), logdet.data_ptr()
    a.n = logdet.numel()
    a.dtype, a.device = 0, _device_index(U.device)
    backend.run_links(a, _stream(U.device))
    return Up, logdet


def stout_links_inverse(Up, C, iterations, backend=None):
    """The inverse of stout_links at fixed staple sums, smearing_jacobian.inverse_smearing_step(Up, C,
    fixed_iterations=iterations) link by link, through the kernel on the device of Up: (U, log|det| of the
    inverse map, residual) for smeared links Up and staple sums C of shape (..., 3, 3), complex128, without
    gradients; the log|det| is minus log|det J| of the step at U, the residual max_ij |(X(U) U - Up)_ij| per link
    (NaN if any entry is NaN), both (...) float64."""
    _check_links('stout_links_inverse', Up, C)
    iterations = int(iterations)
    if not 0 <= iterations < 2**31:
        raise ValueError(f'iterations = {iterations}: expected a non-negative int32')
    backend = load_backend(_target(Up.device), Up.device) if backend is None else backend
    Up, C = _plain(Up, Up.device), _plain(C, Up.device)
    U = torch.empty_like(Up)
    logdet = torch.empty(Up.shape[:-2], dtype=torch.float64, device=Up.device)
    residual = torch.empty_like(logdet)
    a = _InverseArgs()
    a.Up, a.C, a.U = Up.data_ptr(), C.data_ptr(), U.data_ptr()
    a.logdet, a.residual = logdet.data_ptr(), residual.data_ptr()
    a.n = logdet.numel()
    a.iterations, a.dtype, a.device = iterations, 0, _device_index(Up.device)
    backend.run_inverse(a, _stream(Up.device))
    return U, logdet, residual


def inverse_smearing_step(Up, C, fixed_iterations, backend=None):
    """smearing_jacobian.inverse_smearing_step(Up, C, fixed_iterations=n) through the kernel, with its outputs:
    (U, log|det| of the inverse map, n, the largest residual |X(U) U - Up| as a device scalar). As there, the
    residual is recorded by the active smearing_jacobian.DeferredChecks scope, if any, for the caller to check
    once; nothing here waits for the device."""
    U, logdet, residual = stout_links_inverse(Up, C, fixed_iterations, backend)
    residual = residual.amax() if residual.numel() else torch.zeros((), dtype=torch.float64, device=Up.device)
    if sj._DEFERRED is not None:
        sj._DEFERRED.note_residual(residual)
    return U, logdet, int(fixed_iterations), residual


def exp_coefficients(c0, c1, device=None):
    """smearing_jacobian.exp_coefficients(c0, c1, order=2) through the kernel's own functions (snf_coefficients),
    on `device` (default: that of c0): the list [f, b1, b2, d11, d10, d00], each (..., 3) complex128. For the
    tests of the coefficients on chosen (c0, c1)."""
    device = c0.device if device is None else torch.device(device)
    backend = load_backend(_target(device), device)
    c0, c1 = (x.detach().to(device=device, dtype=torch.float64).contiguous() for x in torch.broadcast_tensors(c0, c1))
    out = torch.empty(c0.shape + (6, 3), dtype=torch.complex128, device=device)
    a = _CoeffArgs()
    a.c0, a.c1, a.out = c0.data_ptr(), c1.data_ptr(), out.data_ptr()
    a.n = c0.numel()
    a.dtype, a.device = 0, _device_index(device)
    backend.run_coefficients(a, _stream(device))
    return list(out.unbind(-2))

# ----------------------------------------------------------------------------------------------
# Smearing's choice of path
# ----------------------------------------------------------------------------------------------

def backend_for(owner, cfgs):
    """The kernel library for a call of `owner` (a smearing.Smearing) on cfgs, or None for its torch step: None
    on the CPU without owner.fused_emulate; for another N, the staples of a network (use_nn), another dtype, or
    a library that cannot be built or loaded, None with one RuntimeWarning per reason (owner.fused_strict
    raises fused.BackendUnavailable instead). Whether autograd needs the torch step is decided per step
    (needs_autograd)."""
    if cfgs.device.type == 'cuda':
        target = 'cuda'
    elif cfgs.device.type == 'cpu' and getattr(owner, 'fused_emulate', False):
        target = 'cpu'
    else:
        return None
    if owner.N != 3:
        reason = f'the kernels are written for SU(3), not N = {owner.N}'
    elif owner.use_nn:
        reason = 'the kernels replace the stout step of the selected-links path, which use_nn does not take'
    elif cfgs.dtype != torch.complex128:
        reason = f'the kernels are instantiated for complex128 only, not {cfgs.dtype}'
    else:
        try:
            return load_backend(target, cfgs.device)
        except fused.BackendUnavailable as exc:
            reason = str(exc)
    if getattr(owner, 'fused_strict', False):
        raise fused.BackendUnavailable(reason)
    fused._warn_once(f'fused smearing: {reason}; using the torch stout step')
    return None


def needs_autograd(*inputs):
    """True when autograd must record the stout step on these inputs (grad mode, and one of them requires
    grad: the configurations, or rho through the staple sums, e.g. from the hypernetwork while it is
    trained): the kernels have no backward, so the torch step runs. Grad mode alone is not enough: the
    program runs much of its no-gradient work without torch.no_grad()."""
    return torch.is_grad_enabled() and any(x.requires_grad for x in inputs)
