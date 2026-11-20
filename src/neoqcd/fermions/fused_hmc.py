"""Opt-in fused kernels for the molecular dynamics of the N_f = 2 Wilson HMC (beyond the hop).

`fused.py` runs the half-lattice Wilson hop through one compiled kernel. This module moves the rest
of a trajectory to compiled kernels, with the same build machinery (`fused.load_backend` with
`_MDBackend`: kernel source in a string, plain nvcc or the host C++ compiler, ctypes, on-disk cache
keyed on source, compiler and GPU architecture, atomic writes, a bad cached file rebuilt once):

    component      class / function                what runs in the kernel
    gauge_force    FusedWilsonGaugeAction          staple sum with the plaquette weights of the
                                                   gauge-only defect, Omega = U A, projection;
                                                   optionally the kick P - c F (F never stored) and
                                                   the move exp(i h P') U in the same pass
    update         FusedHMC (_FusedMDState)        P' = P - c F and U' = exp(i h P') U per link
                                                   (exact SU(3) / SU(2) exponential, md_update)
    fermion_force  FusedForceTwoFlavourWilson      the pseudofermion force per link from the
                                                   half-lattice solve (hopping derivative, 2 kappa^2),
                                                   optionally the kick (fermion_force)
    cg             FusedCGStep (lean_cg step)      the CG linear algebra between the hops: three
                                                   passes per iteration (FusedCGStep)

`FusedMDHamiltonian` is the fused-hop Hamiltonian of `fused.py` with a set of components on;
`main_hmc.py --kernel cuda|cpu|auto --md-kernels all|none|<list>` selects them. The torch code stays
the ground truth: the kernels agree with it to rounding (test/test_fermions_fused_hmc*.py), never
bitwise, so a run with them is a different, equally valid chain from the same seed. Nothing here
runs unless asked for; with every component off the classes are their parents exactly.

Targets: "cuda" (compiled with nvcc, run on torch's current stream of the tensors' device, made
current through torch as for the hop, return codes checked) and "cpu" (the same source built for
the host and run as a loop over the CUDA thread index: the CPU emulation the tests use; not a fast
CPU path). complex128 only for the forces and the update (the molecular dynamics stays in double:
reversibility), complex128 and complex64 for the CG (the inner solves of mixed precision); D in
(2, 4), N in (2, 3). Anything else (dtype, D, N, a device of the other kind) runs the torch code,
with one RuntimeWarning per reason (strict=True raises BackendUnavailable instead); an input that
requires grad with grad mode on is an error, as in `fused.FusedHalfLatticeWilson`. Inputs are made
plain first (`fused._raw`: lazy conj/neg bits resolved, strides and offsets materialised); every
output is a new tensor (no in-place write to U or P: the operator caches keyed on the links'
identity and _version stay valid), except the CG vectors x, r, p of `lean_cg`, which are local.

Kernel conventions (MD_KERNEL_SOURCE)
-------------------------------------
Links, momenta, forces: (B, D, V, N, N) complex, contiguous (the lattice flattened in lexicographic
order (t, x_1, ...), last index fastest). The neighbour "x + mu" of the code (torch.roll(f, +1,
axis_mu): array index n - e_mu) and "x - mu" (n + e_mu) are computed in the kernel from the site
index, the extents and the strides (periodic). One thread per link (or per chunk for the CG), 128
threads per block; the host build loops over the same thread index.

Gauge force (`nq_gauge_force`), tid = (b D + mu) V + x:
    A = sum_{nu != mu} [ w_{mu nu}(x) U_nu(x + mu) U_mu(x + nu)^dag U_nu(x)^dag
                       + w_{mu nu}(x - nu) U_nu(x + mu - nu)^dag U_mu(x - nu)^dag U_nu(x - nu) ],
    Omega = U_mu(x) A,  Z = Omega - Omega^dag - tr(Omega - Omega^dag) / N,
    F = -i beta / (2N) Z                                       (= WilsonGaugeAction.force);
the weights are a (BW, P, V) float64 tensor (BW = 1 or B, plane p = nu (nu - 1) / 2 + mu for
mu < nu, the order of `WilsonGaugeAction._weighted_retr`; every layout of the parent is packed by
`plaquette_weight_tensor`), absent without a defect; beta is (B or 1,). Written: F, or P - c F, or
(P - c F, exp(i h (P - c F)) U).

Exponential (`md_expi`, as `algebra.expi` "auto"): SU(3) exp(iQ) = f0 + f1 Q + f2 Q^2 with the
coefficients of `smearing_jacobian.exp_coefficients`: for c1 = tr(Q^2)/2 < C1_SERIES the Taylor
series truncated after N_SERIES terms (the recursion of series_polynomials), above it the closed
forms of Morningstar and Peardon at |c0| with the |c0|/c0max clamp and xi0 from its Taylor series
below |w| = XI_SERIES (constants and XI_TAYLOR taken from smearing_jacobian when the source is
generated); SU(2) the closed form cos q + i sin(q)/q Q. Update (`nq_md_update`): per link P' =
P - c F and / or U' = exp(i h P') U.

Fermion force (`nq_fermion_force`), tid = ((b D + mu) 2 + t) V/2 + i for the i-th site x of parity
t: the half spinors h = R_k^dag psi of the projectors 1 - gamma_mu (k = mu) and 1 + gamma_mu
(k = D + mu) of a(x), b(x + mu), b(x), a(x + mu) (the `spin_projection_tables` of fast.py),
O1 = sum_j h(b(x + mu))_j h(a(x))_j^dag, O2 = sum_j h(b(x))_j h(a(x + mu))_j^dag, Q = L O1 - O2 L^dag
with L from the gathered links of the operator (boundary phase included), F = 2 kappa^2 P_ht(i Q):
`HalfLatticeWilson.hopping_derivative` link by link. Inputs (a_e, a_o, b_e, b_o) = (Y, Z, X, W) of
`FastTwoFlavourWilson.force_from_solution`.

CG (`nq_cg`), per iteration after A p (the operator stays the fused hop): op 0 the partial sums of
Re <p, Ap>; op 1 alpha (the pAp <= tiny guard), x += alpha p, r -= alpha Ap and the partial sums of
|r|^2; op 2 beta (the rr <= tiny guard), p = beta p + r, iterations += !converged,
converged |= |r|^2 <= target. Thread t of sample b handles the elements t, t + nchunk, ...
(nchunk = n/8: coalesced on the GPU), partial sums in double in a fixed order, summed by torch:
deterministic. On the CPU emulation this access pattern is cache-hostile (the serial loop jumps by
nchunk), so the cg component is slower than torch there; it is designed for the GPU, where it
replaces about 25 small launches per iteration by 5.
"""

import ctypes

import torch

from neoqcd.fermions import fused as _fused
from neoqcd.fermions._gauge import C1_SERIES, N_SERIES, XI_SERIES, XI_TAYLOR, SUN_mul
from neoqcd.fermions.algebra import expi
from neoqcd.fermions.fused import FusedTwoFlavourWilson
from neoqcd.fermions.hmc import HMC, OMELYAN_LAMBDA, Hamiltonian, WilsonGaugeAction, _MDState

__all__ = [
    "MD_KERNEL_SOURCE",
    "MD_TARGETS",
    "MD_COMPONENTS",
    "parse_md_kernels",
    "FusedMDHamiltonian",
    "md_backend_status",
    "load_md_backend",
    "gauge_force",
    "md_update",
    "fermion_force",
    "FusedCGStep",
    "FusedWilsonGaugeAction",
    "FusedForceTwoFlavourWilson",
    "FusedHMC",
]

MD_TARGETS = ("cuda", "cpu")
# the components that can run through the kernels (main_hmc.py --md-kernels)
MD_COMPONENTS = ("gauge_force", "update", "fermion_force", "cg")
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

// ---- exp(i Q) of hermitian traceless Q (algebra.expi, method "auto") ----------------------------
// SU(3): exp(iQ) = f0 + f1 Q + f2 Q^2 with the coefficients of smearing_jacobian.exp_coefficients:
// below c1 = Tr(Q^2)/2 = MD_C1_SERIES the Taylor series truncated after MD_N_SERIES terms, summed by
// the recursion of smearing_jacobian.series_polynomials, Q^n = a_n + b_n Q + c_n Q^2,
// (a, b, c)_{n+1} = (c0 c_n, a_n + c1 c_n, b_n), as s = (a, b, c)_n / n!; above, the closed forms of
// Morningstar and Peardon (smearing_jacobian.mp_coefficients, f only) at |c0|, then
// f_j(-c0) = (-1)^j f_j(c0)^*; xi0 = sin(w)/w from its Taylor series (XI_TAYLOR) below |w| = MD_XI_SERIES.
#define MD_C1_SERIES @C1_SERIES@
#define MD_N_SERIES @N_SERIES@
#define MD_XI_SERIES @XI_SERIES@

template <typename T>
MD_HD void md_su3_coefficients(T c0, T c1, MdC<T> f[3]) {
    if (c1 < (T)MD_C1_SERIES) {
        T s0 = 1, s1 = 0, s2 = 0;
        T fr[3] = {0, 0, 0}, fi[3] = {0, 0, 0};
        for (int n = 0; n < MD_N_SERIES; ++n) {
            const T sg = n % 4 < 2 ? (T)1 : (T)-1;  // i^n: real for even n, imaginary for odd n
            if (n % 2 == 0) { fr[0] += sg * s0; fr[1] += sg * s1; fr[2] += sg * s2; }
            else { fi[0] += sg * s0; fi[1] += sg * s1; fi[2] += sg * s2; }
            const T inv = (T)1 / (T)(n + 1);
            const T t0 = (c0 * s2) * inv, t1 = (s0 + c1 * s2) * inv, t2 = s1 * inv;
            s0 = t0; s1 = t1; s2 = t2;
        }
        MD_UNROLL
        for (int j = 0; j < 3; ++j) f[j] = c_make<T>(fr[j], fi[j]);
        return;
    }
    typedef MdC<T> C;
    const bool neg = c0 < 0;
    const T c0max = (T)2 * pow(c1 / (T)3, (T)1.5);
    const T ratio_max = (T)1 - (T)2.220446049250313e-16;  // 1 - torch.finfo(torch.float64).eps
    T ratio = fabs(c0) / c0max;
    if (ratio > ratio_max) ratio = ratio_max;
    const T theta = acos(ratio);
    const T u = sqrt(c1 / (T)3) * cos(theta / (T)3);
    const T w = sqrt(c1) * sin(theta / (T)3);
    const T u2 = u * u, w2 = w * w;
    const T xi0 = fabs(w) < (T)MD_XI_SERIES ? @XI0@ : sin(w) / w;
    const T cw = cos(w);
    const C eu = c_make<T>(cos(u), sin(u));
    const C e2u = c_mul(eu, eu), emu = c_conj(eu);
    C h[3];
    // (u2 - w2) e2u + emu (8 u2 cw + 2 i u (3 u2 + w2) xi0)
    h[0] = c_add(c_scale(u2 - w2, e2u), c_mul(emu, c_make<T>((T)8 * u2 * cw, (T)2 * u * ((T)3 * u2 + w2) * xi0)));
    // 2 u e2u - emu (2 u cw - i (3 u2 - w2) xi0)
    h[1] = c_sub(c_scale((T)2 * u, e2u), c_mul(emu, c_make<T>((T)2 * u * cw, -((T)3 * u2 - w2) * xi0)));
    // e2u - emu (cw + 3 i u xi0)
    h[2] = c_sub(e2u, c_mul(emu, c_make<T>(cw, (T)3 * u * xi0)));
    const T den = (T)9 * u2 - w2;
    MD_UNROLL
    for (int j = 0; j < 3; ++j) {
        C fj = c_make<T>(h[j].re / den, h[j].im / den);
        if (neg) fj = c_scale(j == 1 ? (T)-1 : (T)1, c_conj(fj));
        f[j] = fj;
    }
}

// X = exp(i Q) for N = 3 (Cayley-Hamilton) or N = 2 (closed form cos q + i sin(q)/q Q, q^2 = Tr(Q^2)/2)
template <typename T, int N>
MD_HD void md_expi(MdC<T>* X, const MdC<T>* Q) {
    typedef MdC<T> C;
    if (N == 3) {
        C Q2[9];
        m_mul<T, 3, false, false>(Q2, Q, Q);
        const T c1 = (T)0.5 * ((Q2[0].re + Q2[4].re) + Q2[8].re);
        T c0 = 0;  // Re Tr(Q Q^2) / 3
        MD_UNROLL
        for (int i = 0; i < 3; ++i) {
            MD_UNROLL
            for (int j = 0; j < 3; ++j) c0 += Q[3 * i + j].re * Q2[3 * j + i].re - Q[3 * i + j].im * Q2[3 * j + i].im;
        }
        C f[3];
        md_su3_coefficients<T>(c0 / (T)3, c1, f);
        MD_UNROLL
        for (int e = 0; e < 9; ++e) {
            const C x = c_mul(f[1], Q[e]);
            X[e] = c_add(e % 4 == 0 ? c_add(f[0], x) : x, c_mul(f[2], Q2[e]));
        }
    } else {
        T q2 = 0;  // Re Tr(Q Q) / 2
        MD_UNROLL
        for (int i = 0; i < N; ++i) {
            MD_UNROLL
            for (int j = 0; j < N; ++j) q2 += Q[N * i + j].re * Q[N * j + i].re - Q[N * i + j].im * Q[N * j + i].im;
        }
        q2 *= (T)0.5;
        const T q = q2 > 0 ? sqrt(q2) : (T)0;
        const T cq = cos(q), sq = q > 0 ? sin(q) / q : (T)1;
        MD_UNROLL
        for (int e = 0; e < N * N; ++e) {
            const C iq = c_make<T>(-(sq * Q[e].im), sq * Q[e].re);  // i (sin(q)/q) Q
            X[e] = e % (N + 1) == 0 ? c_add(c_make<T>(cq, 0), iq) : iq;
        }
    }
}

// U' = exp(i h P) U for one link
template <typename T, int N>
MD_HD void md_move_link(MdC<T>* Uout, const MdC<T>* P, const MdC<T>* U, T h) {
    MdC<T> Q[N * N], X[N * N], L[N * N];
    MD_UNROLL
    for (int e = 0; e < N * N; ++e) Q[e] = c_scale(h, P[e]);
    md_expi<T, N>(X, Q);
    m_load<T, N>(L, U);
    MdC<T> out[N * N];
    m_mul<T, N, false, false>(out, X, L);
    m_store<T, N>(Uout, out);
}

// ---- link and momentum update ----------------------------------------------------------------------
// Arguments of nq_md_update (_UpdateArgs): per link, P' = P - coef F (with F) and U' = exp(i h P') U
// (with U); P' = P without F.
struct NqUpdateArgs {
    const void* U;            // NULL (no move) or (n, N, N) complex links
    const void* P;            // (n, N, N) complex momenta
    const void* F;            // NULL (no kick) or (n, N, N) complex force
    void* P_out;              // with F: (n, N, N) P - coef F
    void* U_out;              // with U: (n, N, N) exp(i h P') U
    void* reserved_ptr;
    double coef, h;
    int64_t n;                // links
    int32_t N, dtype, device, reserved;
};

template <typename T, int N, bool KICK, bool MOVE>
MD_HD void md_update_link(const NqUpdateArgs& a, int64_t tid) {
    typedef MdC<T> C;
    constexpr int NN = N * N;
    const C* P = static_cast<const C*>(a.P) + tid * NN;
    C Pn[NN];
    if (KICK) {
        const C* Fp = static_cast<const C*>(a.F) + tid * NN;
        C* Po = static_cast<C*>(a.P_out) + tid * NN;
        MD_UNROLL
        for (int e = 0; e < NN; ++e) {
            Pn[e] = c_sub(P[e], c_scale((T)a.coef, Fp[e]));
            Po[e] = Pn[e];
        }
    } else {
        m_load<T, N>(Pn, P);
    }
    if (MOVE) md_move_link<T, N>(static_cast<C*>(a.U_out) + tid * NN, Pn, static_cast<const C*>(a.U) + tid * NN, (T)a.h);
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
    void* links_out;          // NULL, or (with mom) (B, D, V, N, N): exp(i h (P - coef F)) U (kick + move)
    double coef;              // kick coefficient (with mom)
    double h;                 // move step (with links_out)
    int64_t batch, volume;
    int64_t extent[MD_MAX_D], stride[MD_MAX_D];
    int64_t weights_bstride, beta_bstride;  // per-sample strides in real elements (0: shared)
    int32_t D, N, dtype, device;
};

template <typename T, int D, int N, bool WEIGHTED, bool KICK, bool MOVE>
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
        t1[e] = KICK ? c_sub(P[e], c_scale((T)a.coef, f)) : f;
        out[e] = t1[e];
    }
    if (MOVE) md_move_link<T, N>(static_cast<C*>(a.links_out) + tid * NN, t1, Ub + ((int64_t)mu * V + x) * NN, (T)a.h);
}

// ---- launch helpers (CUDA kernel, or a host loop over the same thread index) ---------------------
// One generic launcher: FN is a functor class with a static `run(args, tid)`; one thread per index
// tid < total (the host build loops over the same index).
#ifdef __CUDACC__
template <typename FN, typename ARGS>
__global__ void md_kernel(const MD_GRID_CONSTANT ARGS a, int64_t total) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) FN::run(a, tid);
}

template <typename FN, typename ARGS>
static int md_launch(const ARGS& a, int64_t total, void* stream) {
    if (total == 0) return 0;
    const int64_t blocks = (total + MD_THREADS - 1) / MD_THREADS;
    md_kernel<FN, ARGS><<<(unsigned)blocks, MD_THREADS, 0, (cudaStream_t)stream>>>(a, total);
    return (int)cudaGetLastError();
}
#else
template <typename FN, typename ARGS>
static int md_launch(const ARGS& a, int64_t total, void* stream) {
    (void)stream;
    for (int64_t tid = 0; tid < total; ++tid) FN::run(a, tid);
    return 0;
}
#endif

template <typename T, int D, int N, bool WEIGHTED, bool KICK, bool MOVE> struct MdGaugeFn {
    static MD_HD void run(const NqGaugeArgs& a, int64_t tid) { md_gauge_link<T, D, N, WEIGHTED, KICK, MOVE>(a, tid); }
};

template <typename T, int D, int N, bool WEIGHTED>
static int md_gauge_kick(const NqGaugeArgs& a, void* stream) {
    const int64_t total = a.batch * (int64_t)D * a.volume;
    if (a.mom == 0) return md_launch<MdGaugeFn<T, D, N, WEIGHTED, false, false> >(a, total, stream);
    if (a.links_out == 0) return md_launch<MdGaugeFn<T, D, N, WEIGHTED, true, false> >(a, total, stream);
    return md_launch<MdGaugeFn<T, D, N, WEIGHTED, true, true> >(a, total, stream);
}

template <typename T, int D, int N>
static int md_gauge_modes(const NqGaugeArgs& a, void* stream) {
    if (a.links_out != 0 && a.mom == 0) return -1;  // the move needs the kick
    return a.weights != 0 ? md_gauge_kick<T, D, N, true>(a, stream) : md_gauge_kick<T, D, N, false>(a, stream);
}

static int md_gauge_dispatch(const NqGaugeArgs& a, void* stream) {
    if (a.dtype != 0) return -1;  // complex128 only
#define MD_CASE(DD, NN) if (a.D == DD && a.N == NN) return md_gauge_modes<double, DD, NN>(a, stream);
    MD_CASE(4, 3) MD_CASE(4, 2) MD_CASE(2, 3) MD_CASE(2, 2)
#undef MD_CASE
    return -1;
}

// ---- pseudofermion force (HalfLatticeWilson.hopping_derivative, times 2 kappa^2) --------------------
// Arguments of nq_fermion_force (_FermionForceArgs).
struct NqFermionForceArgs {
    const void* a[2];             // (B, V2, S, N) complex: a_e, a_o (half-lattice fields)
    const void* b[2];             // (B, V2, S, N) complex: b_e, b_o
    const void* links[2];         // (BL, V2, 2D, N, N) complex: HalfLatticeWilson._links[t]; entry k < D is L_mu(x)^T
    const int32_t* nbr[2];        // (V2, 2D): entry k < D is the position of x_i + mu in the other parity
    const int64_t* sites[2];      // (V2,): full-lattice site of position i of parity t
    const void* scale;            // (B or 1,) real: 2 kappa^2
    const void* mom;              // NULL (write F) or (B, D, V, N, N) complex momenta: write P - coef F
    void* out;                    // (B, D, V, N, N) complex
    double coef;
    double reserved;
    int64_t batch, half_volume, link_batch_stride, scale_batch_stride;
    int32_t D, N, dtype, device;
    int32_t proj_perm[8][2];      // spin_projection_tables (not daggered): term k, row j
    double proj_coef[8][2][2];    // (re, im)
};

// One link (b, mu, x), x the i-th site of parity t: with the half spinors h = R_k^dag psi of the
// projector of term k (k = mu: 1 - gamma_mu, k = D + mu: 1 + gamma_mu),
//   O1 = sum_j h_mu(b(x + mu))_j h_mu(a(x))_j^dag,  O2 = sum_j h_{D+mu}(b(x))_j h_{D+mu}(a(x + mu))_j^dag,
//   Q = L O1 - O2 L^dag,  F = scale P_ht(i Q)  (the formula of HalfLatticeWilson.hopping_derivative).
template <typename T, int D, int N, bool KICK>
MD_HD void md_fforce_link(const NqFermionForceArgs& a, int64_t tid) {
    typedef MdC<T> C;
    constexpr int S = (D == 4) ? 4 : 2;
    constexpr int H = S / 2;
    constexpr int K = 2 * D;
    constexpr int NN = N * N;
    const int64_t V2 = a.half_volume;
    const int64_t i = tid % V2;
    int64_t r = tid / V2;
    const int t = (int)(r % 2);
    r /= 2;
    const int mu = (int)(r % D);
    const int64_t b = r / D;
    const int n = 1 - t;
    const int64_t y = a.nbr[t][i * K + mu];
    const C* ax = static_cast<const C*>(a.a[t]) + (b * V2 + i) * (S * N);
    const C* ay = static_cast<const C*>(a.a[n]) + (b * V2 + y) * (S * N);
    const C* bx = static_cast<const C*>(a.b[t]) + (b * V2 + i) * (S * N);
    const C* by = static_cast<const C*>(a.b[n]) + (b * V2 + y) * (S * N);
    const int kf = mu, kb = D + mu;
    C haf[H][N], hbf[H][N], hbb[H][N], hab[H][N];
    MD_UNROLL
    for (int j = 0; j < H; ++j) {
        const C cf = c_make<T>((T)a.proj_coef[kf][j][0], (T)a.proj_coef[kf][j][1]);
        const C cb = c_make<T>((T)a.proj_coef[kb][j][0], (T)a.proj_coef[kb][j][1]);
        const int pf = (H + a.proj_perm[kf][j]) * N, pb = (H + a.proj_perm[kb][j]) * N;
        MD_UNROLL
        for (int c = 0; c < N; ++c) {
            haf[j][c] = c_add(ax[j * N + c], c_mul(cf, ax[pf + c]));
            hbf[j][c] = c_add(by[j * N + c], c_mul(cf, by[pf + c]));
            hbb[j][c] = c_add(bx[j * N + c], c_mul(cb, bx[pb + c]));
            hab[j][c] = c_add(ay[j * N + c], c_mul(cb, ay[pb + c]));
        }
    }
    C O1[NN], O2[NN];
    MD_UNROLL
    for (int c1 = 0; c1 < N; ++c1) {
        MD_UNROLL
        for (int c2 = 0; c2 < N; ++c2) {
            C s1 = c_mul(hbf[0][c1], c_conj(haf[0][c2])), s2 = c_mul(hbb[0][c1], c_conj(hab[0][c2]));
            MD_UNROLL
            for (int j = 1; j < H; ++j) {
                s1 = c_add(s1, c_mul(hbf[j][c1], c_conj(haf[j][c2])));
                s2 = c_add(s2, c_mul(hbb[j][c1], c_conj(hab[j][c2])));
            }
            O1[N * c1 + c2] = s1;
            O2[N * c1 + c2] = s2;
        }
    }
    // Lt[a][c] = L[c][a]
    C Lt[NN];
    m_load<T, N>(Lt, static_cast<const C*>(a.links[t]) + b * a.link_batch_stride + (i * K + mu) * NN);
    C Q[NN];
    MD_UNROLL
    for (int row = 0; row < N; ++row) {
        MD_UNROLL
        for (int c = 0; c < N; ++c) {
            C s1 = c_mul(Lt[row], O1[c]);                     // sum_a L[row][a] O1[a][c]
            C s2 = c_mul(O2[N * row], c_conj(Lt[c]));        // sum_a O2[row][a] conj(L[c][a])
            MD_UNROLL
            for (int k = 1; k < N; ++k) {
                s1 = c_add(s1, c_mul(Lt[N * k + row], O1[N * k + c]));
                s2 = c_add(s2, c_mul(O2[N * row + k], c_conj(Lt[N * k + c])));
            }
            Q[N * row + c] = c_sub(s1, s2);
        }
    }
    // M = i Q; F = scale (M + M^dag) / 2 minus the trace / N
    C G[NN];
    MD_UNROLL
    for (int row = 0; row < N; ++row) {
        MD_UNROLL
        for (int c = 0; c < N; ++c) {
            const C m = c_make<T>(-Q[N * row + c].im, Q[N * row + c].re);
            const C md = c_make<T>(-Q[N * c + row].im, -Q[N * c + row].re);  // conj(i Q[c][row])
            G[N * row + c] = c_scale((T)0.5, c_add(m, md));
        }
    }
    C tr = G[0];
    MD_UNROLL
    for (int k = 1; k < N; ++k) tr = c_add(tr, G[(N + 1) * k]);
    const C trN = c_make<T>(tr.re / (T)N, tr.im / (T)N);
    const T sc = static_cast<const T*>(a.scale)[b * a.scale_batch_stride];
    const int64_t V = 2 * V2;
    const int64_t link = (b * D + mu) * V + a.sites[t][i];
    C* out = static_cast<C*>(a.out) + link * NN;
    const C* P = KICK ? static_cast<const C*>(a.mom) + link * NN : (const C*)0;
    MD_UNROLL
    for (int e = 0; e < NN; ++e) {
        const C g = (e % (N + 1) == 0) ? c_sub(G[e], trN) : G[e];
        const C f = c_scale(sc, g);
        out[e] = KICK ? c_sub(P[e], c_scale((T)a.coef, f)) : f;
    }
}

template <typename T, int D, int N, bool KICK> struct MdFermionForceFn {
    static MD_HD void run(const NqFermionForceArgs& a, int64_t tid) { md_fforce_link<T, D, N, KICK>(a, tid); }
};

static int md_fforce_dispatch(const NqFermionForceArgs& a, void* stream) {
    if (a.dtype != 0) return -1;  // complex128 only
    const int64_t total = a.batch * (int64_t)a.D * 2 * a.half_volume;
#define MD_CASE(DD, NN) \
    if (a.D == DD && a.N == NN) \
        return a.mom ? md_launch<MdFermionForceFn<double, DD, NN, true> >(a, total, stream) \
                     : md_launch<MdFermionForceFn<double, DD, NN, false> >(a, total, stream);
    MD_CASE(4, 3) MD_CASE(4, 2) MD_CASE(2, 3) MD_CASE(2, 2)
#undef MD_CASE
    return -1;
}

// ---- CG linear algebra (fast.lean_cg, one loop iteration after A p) ---------------------------------
// Arguments of nq_cg (_CGArgs). op 0: dot partials Re <p, Ap>; op 1: alpha = rr / pAp (0 where
// pAp <= tiny), x += alpha p, r -= alpha Ap, partials of |r|^2; op 2: beta = rr_new / rr (0 where
// rr <= tiny), p = beta p + r, and per sample iterations += !converged,
// converged_out = converged | (rr_new <= target). Thread t of sample b handles the elements
// e = t, t + nchunk, ... (coalesced across a warp); its partial sums are in double, in a fixed order
// (deterministic), and torch sums the (B, nchunk) partials.
struct NqCGArgs {
    void* x;                      // (B, n) complex (op 1: updated in place)
    void* r;                      // (B, n) complex (op 1: updated in place; op 2: read)
    void* p;                      // (B, n) complex (op 0, 1: read; op 2: updated in place)
    const void* Ap;               // (B, n) complex
    const void* rr;               // (B,) real (field precision): |r|^2 before the iteration
    const double* pAp;            // (B,) Re <p, Ap> (op 1)
    const double* rr_new;         // (B,) |r|^2 after the update (op 2)
    const void* target;           // (B,) real (field precision) (op 2)
    int64_t* iterations;          // (B,) (op 2)
    const uint8_t* converged_in;  // (B,) bool (op 2)
    uint8_t* converged_out;       // (B,) bool (op 2)
    double* partials;             // (B, nchunk) (op 0, 1)
    double tiny;                  // torch.finfo(real dtype).tiny
    int64_t batch, n, nchunk;
    int32_t op, dtype, device, reserved;
};

template <typename T>
MD_HD void md_cg_dot(const NqCGArgs& a, int64_t tid) {
    typedef MdC<T> C;
    const int64_t b = tid / a.nchunk, t = tid % a.nchunk;
    const C* P = static_cast<const C*>(a.p) + b * a.n;
    const C* A = static_cast<const C*>(a.Ap) + b * a.n;
    double s = 0;
    for (int64_t e = t; e < a.n; e += a.nchunk) s += (double)P[e].re * (double)A[e].re + (double)P[e].im * (double)A[e].im;
    a.partials[tid] = s;
}

template <typename T>
MD_HD void md_cg_xr(const NqCGArgs& a, int64_t tid) {
    typedef MdC<T> C;
    const int64_t b = tid / a.nchunk, t = tid % a.nchunk;
    const T pap = (T)a.pAp[b];
    const T alpha = pap > (T)a.tiny ? static_cast<const T*>(a.rr)[b] / pap : (T)0;
    C* X = static_cast<C*>(a.x) + b * a.n;
    C* R = static_cast<C*>(a.r) + b * a.n;
    const C* P = static_cast<const C*>(a.p) + b * a.n;
    const C* A = static_cast<const C*>(a.Ap) + b * a.n;
    double s = 0;
    for (int64_t e = t; e < a.n; e += a.nchunk) {
        X[e] = c_add(X[e], c_scale(alpha, P[e]));
        const C rv = c_sub(R[e], c_scale(alpha, A[e]));
        R[e] = rv;
        s += (double)rv.re * (double)rv.re + (double)rv.im * (double)rv.im;
    }
    a.partials[tid] = s;
}

template <typename T>
MD_HD void md_cg_p(const NqCGArgs& a, int64_t tid) {
    typedef MdC<T> C;
    const int64_t b = tid / a.nchunk, t = tid % a.nchunk;
    const T rr = static_cast<const T*>(a.rr)[b];
    const T rn = (T)a.rr_new[b];
    const T beta = rr > (T)a.tiny ? rn / rr : (T)0;
    C* P = static_cast<C*>(a.p) + b * a.n;
    const C* R = static_cast<const C*>(a.r) + b * a.n;
    for (int64_t e = t; e < a.n; e += a.nchunk) P[e] = c_add(c_scale(beta, P[e]), R[e]);
    if (t == 0) {
        const uint8_t conv = a.converged_in[b];
        a.iterations[b] += conv ? 0 : 1;
        a.converged_out[b] = (uint8_t)(conv || rn <= static_cast<const T*>(a.target)[b]);
    }
}

template <typename T, int OP> struct MdCGFn {
    static MD_HD void run(const NqCGArgs& a, int64_t tid) {
        if (OP == 0) md_cg_dot<T>(a, tid);
        else if (OP == 1) md_cg_xr<T>(a, tid);
        else md_cg_p<T>(a, tid);
    }
};

template <typename T>
static int md_cg_ops(const NqCGArgs& a, void* stream) {
    if (a.op == 0) return md_launch<MdCGFn<T, 0> >(a, a.batch * a.nchunk, stream);
    if (a.op == 1) return md_launch<MdCGFn<T, 1> >(a, a.batch * a.nchunk, stream);
    if (a.op == 2) return md_launch<MdCGFn<T, 2> >(a, a.batch * a.nchunk, stream);
    return -1;
}

static int md_cg_dispatch(const NqCGArgs& a, void* stream) {
    if (a.dtype == 0) return md_cg_ops<double>(a, stream);
    if (a.dtype == 1) return md_cg_ops<float>(a, stream);
    return -1;
}

template <typename T, int N, bool KICK, bool MOVE> struct MdUpdateFn {
    static MD_HD void run(const NqUpdateArgs& a, int64_t tid) { md_update_link<T, N, KICK, MOVE>(a, tid); }
};

template <typename T, int N>
static int md_update_modes(const NqUpdateArgs& a, void* stream) {
    const bool kick = a.F != 0, move = a.U != 0;
    if (kick && (a.P_out == 0)) return -1;
    if (move && (a.U_out == 0)) return -1;
    if (kick && move) return md_launch<MdUpdateFn<T, N, true, true> >(a, a.n, stream);
    if (kick) return md_launch<MdUpdateFn<T, N, true, false> >(a, a.n, stream);
    if (move) return md_launch<MdUpdateFn<T, N, false, true> >(a, a.n, stream);
    return 0;
}

static int md_update_dispatch(const NqUpdateArgs& a, void* stream) {
    if (a.dtype != 0) return -1;  // complex128 only
    if (a.N == 3) return md_update_modes<double, 3>(a, stream);
    if (a.N == 2) return md_update_modes<double, 2>(a, stream);
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
int nq_update_args_size(void) { return (int)sizeof(NqUpdateArgs); }
int nq_fforce_args_size(void) { return (int)sizeof(NqFermionForceArgs); }
int nq_cg_args_size(void) { return (int)sizeof(NqCGArgs); }

int nq_cg(const NqCGArgs* a, void* stream) {
    MD_DEVICE_GUARD_BEGIN(a->device)
    int code = md_cg_dispatch(*a, stream);
    MD_DEVICE_GUARD_END(code)
    return code;
}

int nq_fermion_force(const NqFermionForceArgs* a, void* stream) {
    MD_DEVICE_GUARD_BEGIN(a->device)
    int code = md_fforce_dispatch(*a, stream);
    MD_DEVICE_GUARD_END(code)
    return code;
}

int nq_md_update(const NqUpdateArgs* a, void* stream) {
    MD_DEVICE_GUARD_BEGIN(a->device)
    int code = md_update_dispatch(*a, stream);
    MD_DEVICE_GUARD_END(code)
    return code;
}

// Every entry point returns 0 on success, -1 for an unsupported (dtype, D, N) or argument
// combination, a cudaError_t code otherwise.
int nq_gauge_force(const NqGaugeArgs* a, void* stream) {
    MD_DEVICE_GUARD_BEGIN(a->device)
    int code = md_gauge_dispatch(*a, stream);
    MD_DEVICE_GUARD_END(code)
    return code;
}

const char* nq_md_error_string(int code) {
    if (code == -1) return "unsupported (dtype, D, N) or argument combination";
#ifdef __CUDACC__
    return cudaGetErrorString((cudaError_t)code);
#else
    return code == 0 ? "no error" : "unknown error";
#endif
}
}
"""


def _horner(coefficients, x):
    """C expression of sum_n coefficients[n] x^n by Horner's rule, the coefficients exact (%.17g)."""
    expr = f"(T){coefficients[-1]:.17g}"
    for c in coefficients[-2::-1]:
        expr = f"({expr} * {x} + (T){c:.17g})"
    return expr


MD_KERNEL_SOURCE = (MD_KERNEL_SOURCE.replace("@C1_SERIES@", repr(float(C1_SERIES)))
                    .replace("@N_SERIES@", str(int(N_SERIES)))
                    .replace("@XI_SERIES@", repr(float(XI_SERIES)))
                    .replace("@XI0@", _horner(XI_TAYLOR[0], "w2")))


# ---- ctypes mirrors of the argument structures -------------------------------------------------
class _GaugeArgs(ctypes.Structure):
    _fields_ = [
        ("U", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("beta", ctypes.c_void_p),
        ("mom", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("links_out", ctypes.c_void_p),
        ("coef", ctypes.c_double),
        ("h", ctypes.c_double),
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


class _UpdateArgs(ctypes.Structure):
    _fields_ = [
        ("U", ctypes.c_void_p),
        ("P", ctypes.c_void_p),
        ("F", ctypes.c_void_p),
        ("P_out", ctypes.c_void_p),
        ("U_out", ctypes.c_void_p),
        ("reserved_ptr", ctypes.c_void_p),
        ("coef", ctypes.c_double),
        ("h", ctypes.c_double),
        ("n", ctypes.c_int64),
        ("N", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("device", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class _FermionForceArgs(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_void_p * 2),
        ("b", ctypes.c_void_p * 2),
        ("links", ctypes.c_void_p * 2),
        ("nbr", ctypes.c_void_p * 2),
        ("sites", ctypes.c_void_p * 2),
        ("scale", ctypes.c_void_p),
        ("mom", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("coef", ctypes.c_double),
        ("reserved", ctypes.c_double),
        ("batch", ctypes.c_int64),
        ("half_volume", ctypes.c_int64),
        ("link_batch_stride", ctypes.c_int64),
        ("scale_batch_stride", ctypes.c_int64),
        ("D", ctypes.c_int32),
        ("N", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("device", ctypes.c_int32),
        ("proj_perm", (ctypes.c_int32 * 2) * 8),
        ("proj_coef", ((ctypes.c_double * 2) * 2) * 8),
    ]


class _CGArgs(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_void_p),
        ("r", ctypes.c_void_p),
        ("p", ctypes.c_void_p),
        ("Ap", ctypes.c_void_p),
        ("rr", ctypes.c_void_p),
        ("pAp", ctypes.c_void_p),
        ("rr_new", ctypes.c_void_p),
        ("target", ctypes.c_void_p),
        ("iterations", ctypes.c_void_p),
        ("converged_in", ctypes.c_void_p),
        ("converged_out", ctypes.c_void_p),
        ("partials", ctypes.c_void_p),
        ("tiny", ctypes.c_double),
        ("batch", ctypes.c_int64),
        ("n", ctypes.c_int64),
        ("nchunk", ctypes.c_int64),
        ("op", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("device", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


_ENTRY_POINTS = (
    # (function, argument structure, size function)
    ("nq_gauge_force", _GaugeArgs, "nq_gauge_args_size"),
    ("nq_md_update", _UpdateArgs, "nq_update_args_size"),
    ("nq_fermion_force", _FermionForceArgs, "nq_fforce_args_size"),
    ("nq_cg", _CGArgs, "nq_cg_args_size"),
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


def gauge_force(U, beta, geometry, weights=None, mom=None, coef=0.0, h=None, backend=None):
    """The force of `WilsonGaugeAction` (beta, plaquette weights as a `plaquette_weight_tensor`) through
    the kernel; with `mom`, the kicked momenta mom - coef F instead (F never stored); with `mom` and
    `h`, the pair (mom - coef F, exp(i h (mom - coef F)) U) (kick and move in one pass). U complex128
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
    links_out = None
    if h is not None:
        if mom is None:
            raise ValueError("the fused move (h) needs the kick (mom)")
        links_out = torch.empty_like(U, memory_format=torch.contiguous_format)
        a.links_out = links_out.data_ptr()
        a.h = float(h)
    a.batch, a.volume = B, geometry.volume
    _lattice_args(a, geometry)
    a.D, a.N, a.dtype, a.device = geometry.D, geometry.N, 0, _device_index(dev)
    _launch(backend, "nq_gauge_force", a, dev)
    return out if links_out is None else (out, links_out)


def md_update(P, U=None, F=None, coef=0.0, h=None, backend=None):
    """Link and momentum update through the kernel, per link: P' = P - coef F (with F, else P' = P)
    and U' = exp(i h P') U (with U and h; `algebra.expi` "auto": Cayley-Hamilton for SU(3) with the
    coefficients of exp_coefficients, closed form for SU(2)). Returns (P' or None without F,
    U' or None without U). Tensors (..., N, N) of one shape, complex128, one device."""
    _refuse_autograd("md_update", P=P, U=U, F=F)
    N = P.shape[-1]
    if P.dtype != torch.complex128 or N not in _SUPPORTED_N or P.shape[-2] != N:
        raise ValueError(f"md_update: complex128 (..., N, N) with N in {_SUPPORTED_N}, got {P.dtype} {tuple(P.shape)}")
    for name, t in (("U", U), ("F", F)):
        if t is not None and (t.shape != P.shape or t.dtype != P.dtype or t.device != P.device):
            raise ValueError(f"md_update: {name} must have the shape, dtype and device of P")
    if (U is None) != (h is None):
        raise ValueError("md_update: the move needs both U and h")
    dev = P.device
    backend = load_md_backend(_target_of(dev), dev) if backend is None else backend
    P = _raw(P)
    a = _UpdateArgs()
    a.P = P.data_ptr()
    a.n = P.numel() // (N * N)
    a.N, a.dtype, a.device = N, 0, _device_index(dev)
    P_out = U_out = None
    if F is not None:
        F = _raw(F)
        P_out = torch.empty_like(P)
        a.F, a.P_out, a.coef = F.data_ptr(), P_out.data_ptr(), float(coef)
    if U is not None:
        U = _raw(U)
        U_out = torch.empty_like(U)
        a.U, a.U_out, a.h = U.data_ptr(), U_out.data_ptr(), float(h)
    _launch(backend, "nq_md_update", a, dev)
    return P_out, U_out


_FORCE_TABLES = {}


def _force_tables(op):
    """(nbr int32 (V/2, 2D) per target parity, sites int64 (V/2,) per parity) of a HalfLatticeWilson,
    cached per geometry and device (the neighbour positions of `site_tables`' hop_index)."""
    key = (op.geometry, str(op.device))
    hit = _FORCE_TABLES.get(key)
    if hit is None:
        K = 2 * op.geometry.D
        tab = op._tables
        nbr = [(tab["hop_index"][t].view(-1, K) // K).to(torch.int32).contiguous() for t in (0, 1)]
        sites = [tab["sites"][t].to(torch.int64).contiguous() for t in (0, 1)]
        hit = _FORCE_TABLES[key] = (nbr, sites)
    return hit


def fermion_force(op, a_e, a_o, b_e, b_o, scale, mom=None, coef=0.0, backend=None):
    """scale * `HalfLatticeWilson.hopping_derivative`(a_e, a_o, b_e, b_o) through the kernel, shape
    (B, D, lattice, N, N); with `mom` the kicked momenta mom - coef F instead. `op` is the
    complex128 half-lattice operator (its gathered links, site and spin tables), the fields are
    half-lattice (B, V/2, S, N) complex128, `scale` a (B,) or (1,) real tensor (2 kappa^2)."""
    _refuse_autograd("fermion_force", a_e=a_e, a_o=a_o, b_e=b_e, b_o=b_o, mom=mom, links=op.U)
    geo = op.geometry
    reason = kernel_supports(geo, op.dtype)
    if reason is not None:
        raise ValueError(reason)
    dev = op.device
    backend = load_md_backend(_target_of(dev), dev) if backend is None else backend
    D, H, N = geo.D, op._spin["H"], geo.N
    K = 2 * D
    fields = [a_e, a_o, b_e, b_o]
    B = a_e.shape[0]
    for f in fields:
        if tuple(f.shape) != (B, geo.volume // 2, geo.spin, N) or f.dtype != op.dtype or f.device != dev:
            raise ValueError(f"fermion_force: half-lattice complex128 fields of shape {(B, geo.volume // 2, geo.spin, N)} "
                             f"on {dev}, got {tuple(f.shape)} {f.dtype} {f.device}")
    a_e, a_o, b_e, b_o = [_raw(f) for f in fields]
    links = [_raw(l) for l in op._links]
    if links[0].shape[0] not in (1, B):
        raise ValueError(f"field batch {B} does not match the link batch {links[0].shape[0]}")
    nbr, sites = _force_tables(op)
    scale = _raw(torch.as_tensor(scale, dtype=torch.float64, device=dev).reshape(-1))
    if scale.numel() not in (1, B):
        raise ValueError(f"scale of {scale.numel()} values for a batch of {B}")
    a = _FermionForceArgs()
    for t in (0, 1):
        a.a[t] = (a_e, a_o)[t].data_ptr()
        a.b[t] = (b_e, b_o)[t].data_ptr()
        a.links[t] = links[t].data_ptr()
        a.nbr[t] = nbr[t].data_ptr()
        a.sites[t] = sites[t].data_ptr()
    a.scale = scale.data_ptr()
    a.scale_batch_stride = 0 if scale.numel() == 1 else 1
    a.batch, a.half_volume = B, geo.volume // 2
    a.link_batch_stride = 0 if links[0].shape[0] == 1 else links[0][0].numel()
    a.D, a.N, a.dtype, a.device = D, N, 0, _device_index(dev)
    pp = op._spin["proj_perm"].reshape(K, H)
    pc = op._spin["proj_coef"].reshape(K, H)
    for k in range(K):
        for j in range(H):
            a.proj_perm[k][j] = int(pp[k, j])
            a.proj_coef[k][j][0] = float(pc[k, j].real)
            a.proj_coef[k][j][1] = float(pc[k, j].imag)
    shape = geo.link_shape(B)
    if mom is not None:
        if tuple(mom.shape) != tuple(shape) or mom.dtype != op.dtype or mom.device != dev:
            raise ValueError(f"mom must be ({shape}) complex128 on {dev}")
        mom = _raw(mom)
        a.mom = mom.data_ptr()
        a.coef = float(coef)
    out = torch.empty(shape, dtype=op.dtype, device=dev)
    a.out = out.data_ptr()
    _launch(backend, "nq_fermion_force", a, dev)
    return out


_CG_DTYPES = {torch.complex128: (0, torch.float64), torch.complex64: (1, torch.float32)}
CG_ELEMENTS_PER_THREAD = 8  # complex elements summed by one thread of the reductions


def _plain(t):
    return t.is_contiguous() and not t.is_conj() and not t.is_neg()


class FusedCGStep:
    """One iteration of `fast.lean_cg` after the operator, in three kernel passes (`step` of
    lean_cg / mixed_precision_cg): Re <p, Ap> (partials), the x and r update with |r|^2 (partials),
    the p update with the per-sample iteration count and convergence flag; torch sums the
    (B, n / 8) double partials. Same arithmetic as the torch body (alpha, beta, the guards for
    pAp, rr <= tiny, NaN), the dot products summed in double in another order: rounding-level
    differences. complex128 and complex64 (the inner solves of mixed precision). Returns None (the
    torch body runs) for inputs it cannot write in place (not contiguous, conj/neg bits, another
    device or dtype)."""

    def __init__(self, backend, device):
        self.backend = backend
        self.device = torch.device(device)
        self.calls = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["backend"] = None  # ctypes handles: re-acquired at the first call
        return state

    def _get_backend(self):
        if self.backend is None:
            self.backend = load_md_backend(_target_of(self.device), self.device)
        return self.backend

    def __call__(self, apply_normal, x, r, p, rr, target, converged, iterations):
        if x.dtype not in _CG_DTYPES or x.device.type != self.device.type:
            return None
        code, real = _CG_DTYPES[x.dtype]
        if not (all(_plain(t) and t.dtype == x.dtype and t.shape == x.shape for t in (x, r, p))
                and rr.dtype == real and iterations.dtype == torch.int64 and iterations.is_contiguous()
                and converged.dtype == torch.bool):
            return None
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x, r, p)):
            return None
        backend = self._get_backend()
        Ap = _raw(apply_normal(p))
        dev = x.device
        B = x.shape[0]
        n = x[0].numel()
        nchunk = max(1, min(n, -(-n // CG_ELEMENTS_PER_THREAD)))
        partials = torch.empty((B, nchunk), dtype=torch.float64, device=dev)
        rr_c = _raw(rr)
        a = _CGArgs()
        a.x, a.r, a.p, a.Ap = x.data_ptr(), r.data_ptr(), p.data_ptr(), Ap.data_ptr()
        a.rr = rr_c.data_ptr()
        a.partials = partials.data_ptr()
        a.tiny = float(torch.finfo(real).tiny)
        a.batch, a.n, a.nchunk = B, n, nchunk
        a.dtype, a.device = code, _device_index(dev)
        a.op = 0
        _launch(backend, "nq_cg", a, dev)
        pAp = partials.sum(dim=1)
        a.pAp = pAp.data_ptr()
        a.op = 1
        _launch(backend, "nq_cg", a, dev)
        rr_new = partials.sum(dim=1)
        target_c = _raw(target.to(device=dev, dtype=real))
        conv_in = _raw(converged)
        conv_out = torch.empty_like(conv_in)
        a.rr_new = rr_new.data_ptr()
        a.target = target_c.data_ptr()
        a.iterations = iterations.data_ptr()
        a.converged_in = conv_in.data_ptr()
        a.converged_out = conv_out.data_ptr()
        a.op = 2
        _launch(backend, "nq_cg", a, dev)
        self.calls += 1
        return rr_new.to(real), conv_out


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

    @property
    def fuses_kicks(self):
        """True when the force kernel is requested: `FusedHMC` then kicks through `force_kick`."""
        return self._choice.target is not None

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

    def force_kick_move(self, U, P, coefficient, h):
        """(P', exp(i h P') U) with P' = P - coefficient * force(U), in one kernel pass; None when the
        kernel does not run here (the caller then kicks and moves separately)."""
        backend = self._backend(U)
        if backend is None or not (P.dtype == U.dtype and P.shape == U.shape and P.device == U.device):
            return None
        self._choice.last_used = backend.target
        return gauge_force(U, self.beta, self.geometry, self._packed_weights(U), mom=P, coef=coefficient, h=h,
                           backend=backend)


# ---- pseudofermion term -----------------------------------------------------------------------------
class FusedForceTwoFlavourWilson(FusedTwoFlavourWilson):
    """`fused.FusedTwoFlavourWilson` (fused hop) whose force runs through the fermion-force kernel.

    The force is the parent's, 2 kappa^2 grad Re[(Y + Z)^dag H (X + W)] from the half fields X (the
    solve), W = H_oe X, Y = X - kappa^2 H_eo W (the axpy fused into the compiled hop), Z = (H^dag)_oe Y,
    with `HalfLatticeWilson.hopping_derivative` and the factor 2 kappa^2 replaced by one kernel pass
    (`fermion_force`). `force_kick(U, P, c)` is P - c F(U) with the kick in the same pass.
    Extra args `md_target` ("cuda", "cpu" or None: the parent's torch force) and `md_strict`; the
    rest as the parent (kernel, link_layout, strict, solver settings). The path of the last force
    is `force_kernel_used` ("cuda", "cpu" or "torch").
    """

    def __init__(self, kappa, geometry, md_target=None, md_strict=False, cg_target=None, **kwargs):
        self._force_choice = _KernelChoice(md_target, md_strict, "fermion force")
        self._cg_choice = _KernelChoice(cg_target, md_strict, "CG linear algebra")
        self._cg_steps = {}
        super().__init__(kappa, geometry, **kwargs)

    @property
    def cg_kernel_used(self):
        """"cuda" / "cpu" when the last solve ran its iterations through `FusedCGStep`, else "torch"."""
        return self._cg_choice.last_used

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cg_steps"] = {}
        return state

    def _solve(self, op, tol):
        """The parent's solve with the CG iterations through `FusedCGStep` when "cg" is requested
        (`fast.FastTwoFlavourWilson._solve` reads `_cg_step`)."""
        self._cg_step = None
        self._cg_choice.last_used = "torch"
        if self._cg_choice.target is not None:
            backend = self._cg_choice.backend(op.device, self.geometry, torch.complex128)
            if backend is not None:
                key = str(op.device)
                if key not in self._cg_steps:
                    self._cg_steps[key] = FusedCGStep(backend, op.device)
                self._cg_step = self._cg_steps[key]
            self._cg_choice.last_used = "torch" if backend is None else backend.target
        try:
            return super()._solve(op, tol)
        finally:
            self._cg_step = None

    @property
    def force_kernel_used(self):
        return self._force_choice.last_used

    @property
    def fuses_kicks(self):
        """True when the force kernel is requested: `FusedHMC` then kicks through `force_kick`."""
        return self._force_choice.target is not None

    def _force_fields(self, op, X):
        W = op.hop(X, 1)
        if getattr(op, "compiled", False):
            Y = op._hop_kernel(W, 0, scale=op._scale_neg_k2, add=X)
        else:
            Y = op.hop(W, 0).mul_(-op._k2).add_(X)
        Z = op.hop(Y, 1, dagger=True)
        return Y, Z, X, W

    def _force_backend(self, op):
        return self._force_choice.backend(op.device, self.geometry, op.dtype)

    def force_from_solution(self, op, X):
        """The force of `FastTwoFlavourWilson.force_from_solution` (kernel when it runs here)."""
        backend = self._force_backend(op)
        if backend is None:
            self._force_choice.last_used = "torch"
            return FusedTwoFlavourWilson.force_from_solution(op, X)
        self._force_choice.last_used = backend.target
        return fermion_force(op, *self._force_fields(op, X), 2.0 * op.kappa ** 2, backend=backend)

    def force_kick(self, U, P, coefficient):
        """P - coefficient * force(U): one force solve, then the kernel with the kick fused."""
        op = self.operator(U)
        backend = self._force_backend(op)
        if backend is None or tuple(P.shape) != tuple(U.shape) or P.dtype != op.dtype or P.device != op.device:
            return P - coefficient * self.force(U)
        self.stats["force_calls"] += 1
        X = self._solve_as("force", op, self.solver_tol)
        self._force_choice.last_used = backend.target
        return fermion_force(op, *self._force_fields(op, X), 2.0 * op.kappa ** 2, mom=P, coef=coefficient,
                             backend=backend)


# ---- integrator ------------------------------------------------------------------------------------
def _default_exp(method, N):
    """True if `algebra.expi(Q, method)` is the exponential of the update kernel for SU(N)."""
    return method == "auto" or (method, N) in (("closed", 2), ("cayley_hamilton", 3))


class _FusedMDState(_MDState):
    """`hmc._MDState` with the momentum kicks and the position updates through the kernels.

    Same lazy kicks and the same sequence of updates as the parent (level 0 first at a flush). A
    level whose only term fuses its kicks (`fuses_kicks`: the gauge or fermion force kernel is
    requested) kicks through the term's `force_kick` in one pass;
    other levels compute F with `level_force` and kick through the update kernel. At a move, the
    last pending level is kicked together with the move: in the force kernel when its term has
    `force_kick_move` (the gauge force: F, the kick and exp(i h P) U in one pass), else by the
    update kernel (kick + move in one pass). New tensors are written every time (no in-place
    update of P or U: the trajectory keeps its initial P and U, and the operator caches keyed on
    U's identity see a new tensor). `counts` records which path every kick and move took.
    """

    def __init__(self, hamiltonian, U, P, exp_method, update_backend, fuse_force_kicks=True):
        super().__init__(hamiltonian, U, P, exp_method)
        self.backend = update_backend
        self.fuse_force_kicks = bool(fuse_force_kicks)
        self.counts = {"kick_fused": 0, "kick_kernel": 0, "kick_torch": 0, "move_fused": 0, "move_kernel": 0,
                       "move_torch": 0}

    def _single_term(self, level):
        """The only term of `level` when it fuses its kicks (`fuses_kicks`), else None."""
        terms = self.hamiltonian.levels[level][0]
        if len(terms) == 1 and self.fuse_force_kicks and getattr(terms[0], "fuses_kicks", False):
            return terms[0]
        return None

    def _kick(self, level, c):
        term = self._single_term(level)
        if term is not None and hasattr(term, "force_kick"):
            self.P = term.force_kick(self.U, self.P, c)
            self.counts["kick_fused"] += 1
        else:
            F = self.hamiltonian.level_force(self.U, level)
            if self.backend is not None and F.dtype == self.P.dtype and F.shape == self.P.shape:
                self.P = md_update(self.P, F=F, coef=c, backend=self.backend)[0]
                self.counts["kick_kernel"] += 1
            else:
                self.P = self.P - c * F
                self.counts["kick_torch"] += 1
        self.force_evals[level] += 1
        self.pending[level] = 0.0

    def flush(self):
        for level, c in enumerate(self.pending):
            if c != 0.0:
                self._kick(level, c)

    def move(self, h):
        pending = [level for level, c in enumerate(self.pending) if c != 0.0]
        for level in pending[:-1]:
            self._kick(level, self.pending[level])
        if self.backend is None or not _default_exp(self.exp_method, self.U.shape[-1]):
            self.flush()
            self.U = SUN_mul(expi(h * self.P, self.exp_method), self.U)
            self.counts["move_torch"] += 1
            return
        if pending:
            level = pending[-1]
            c = self.pending[level]
            term = self._single_term(level)
            if term is not None and hasattr(term, "force_kick_move"):
                done = term.force_kick_move(self.U, self.P, c, h)
                if done is not None:
                    self.P, self.U = done
                    self.force_evals[level] += 1
                    self.pending[level] = 0.0
                    self.counts["kick_fused"] += 1
                    self.counts["move_fused"] += 1
                    return
            if term is not None and hasattr(term, "force_kick"):
                self._kick(level, c)
            else:
                F = self.hamiltonian.level_force(self.U, level)
                self.P, self.U = md_update(self.P, U=self.U, F=F, coef=c, h=h, backend=self.backend)
                self.force_evals[level] += 1
                self.pending[level] = 0.0
                self.counts["kick_kernel"] += 1
                self.counts["move_kernel"] += 1
                return
        self.U = md_update(self.P, U=self.U, h=h, backend=self.backend)[1]
        self.counts["move_kernel"] += 1


class FusedHMC(HMC):
    """`hmc.HMC` whose molecular dynamics runs the kicks and the position updates through the update
    kernel (and the fused force kicks of the terms that have them): `_FusedMDState`.

    Extra args: `target` "cuda", "cpu" (host build) or None (the parent's integrator exactly),
    `strict` (raise BackendUnavailable instead of the torch updates when the kernel cannot run for
    the links: device, dtype, N), `fuse_force_kicks` (default True: use the terms' `force_kick` /
    `force_kick_move`; False: every level's force is computed and kicked by the update kernel).
    The update kernel runs for exp_method "auto" (and the SU(N)-specific method it means); any other
    method runs the torch updates. Metropolis, refresh, energies and the final reunitarisation are
    the parent's. `last_counts` holds the paths of the last trajectory's kicks and moves.
    """

    def __init__(self, geometry, hamiltonian, tau, n_steps, integrator="omelyan", exp_method="auto",
                 omelyan_lambda=OMELYAN_LAMBDA, target=None, strict=False, fuse_force_kicks=True):
        super().__init__(geometry, hamiltonian, tau, n_steps, integrator=integrator, exp_method=exp_method,
                         omelyan_lambda=omelyan_lambda)
        self._choice = _KernelChoice(target, strict, "link and momentum update")
        self.fuse_force_kicks = bool(fuse_force_kicks)
        self.last_counts = None

    @property
    def kernel_used(self):
        return self._choice.last_used

    @torch.no_grad()
    def integrate(self, U, P, tau=None, n_steps=None):
        """As `HMC.integrate`, with `_FusedMDState` (the parent's own integrator when target is None)."""
        if self._choice.target is None:
            self._choice.last_used = "torch"
            return super().integrate(U, P, tau, n_steps)
        backend = self._choice.backend(U.device, self.geometry, U.dtype)
        self._choice.last_used = "torch" if backend is None else backend.target
        tau = self.tau if tau is None else float(tau)
        n = (self.n_steps if n_steps is None else int(n_steps)) * self.hamiltonian.levels[0][1]
        state = _FusedMDState(self.hamiltonian, U, P, self.exp_method, backend, self.fuse_force_kicks)
        h = tau / n
        for _ in range(n):
            self._step(state, 0, h)
        state.flush()
        self.last_counts = dict(state.counts)
        return state.U, state.P, state.force_evals


# ---- Hamiltonian ----------------------------------------------------------------------------------
def parse_md_kernels(spec):
    """--md-kernels value -> sorted tuple of MD_COMPONENTS: "none" (or "", None) is (), "all" every
    component, else a comma-separated list (ValueError on an unknown name)."""
    if spec is None:
        return ()
    if isinstance(spec, (list, tuple, set, frozenset)):
        names = [str(x).strip() for x in spec]
    else:
        names = [x.strip() for x in str(spec).split(",")]
    names = [x for x in names if x and x != "none"]
    if "all" in names:
        if len(names) > 1:
            raise ValueError(f"md kernels: 'all' combined with {names}")
        return tuple(MD_COMPONENTS)
    bad = [x for x in names if x not in MD_COMPONENTS]
    if bad:
        raise ValueError(f"md kernels: unknown component(s) {bad}; choose from {MD_COMPONENTS}, 'all' or 'none'")
    return tuple(sorted(set(names), key=MD_COMPONENTS.index))


class FusedMDHamiltonian(Hamiltonian):
    """`fused.FusedTwoFlavourHamiltonian` (fused hop) with the MD kernels of `md_kernels`.

    Levels as there: the fermion term outside, the gauge term inside with `n_gauge_substeps`.
    `md_kernels`: components of MD_COMPONENTS (`parse_md_kernels`) run through the compiled target
    `md_target` ("cuda" or "cpu"; None: none of them) with `md_strict`: "gauge_force" for the gauge
    term (`FusedWilsonGaugeAction`), "fermion_force" for the fermion term
    (`FusedForceTwoFlavourWilson`; with the component off it is `FusedTwoFlavourWilson`'s force),
    "cg" for the CG iterations of the fermion solves (`FusedCGStep`). The "update" component is the integrator's
    (`FusedHMC(target=...)`, `md_hmc_target(ham)`). Other keyword arguments go to
    `FusedTwoFlavourWilson` (kernel, link_layout, strict, solver settings, ...).
    """

    def __init__(self, beta, kappa, geometry, n_gauge_substeps=1, plaquette_weights=None, md_kernels=(),
                 md_target=None, md_strict=False, **fermion_kwargs):
        self.md_kernels = parse_md_kernels(md_kernels)
        self.md_target = md_target
        self.md_strict = bool(md_strict)

        def target(component):
            return md_target if component in self.md_kernels else None

        self.fermion = FusedForceTwoFlavourWilson(kappa, geometry, md_target=target("fermion_force"),
                                                  md_strict=md_strict, cg_target=target("cg"), **fermion_kwargs)
        self.gauge = FusedWilsonGaugeAction(beta, geometry, plaquette_weights, target=target("gauge_force"),
                                            strict=md_strict)
        super().__init__([([self.fermion], 1), ([self.gauge], n_gauge_substeps)])

    def md_hmc_target(self):
        """The `FusedHMC` target of the "update" component (None when it is off)."""
        return self.md_target if "update" in self.md_kernels else None
