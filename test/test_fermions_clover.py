"""Tests of the Sheikholeslami-Wohlert (clover) N_f = 2 Wilson fermions (neoqcd.fermions.clover).

CPU, complex128, fixed seeds. Lattices: 2^3 x 4 and 2^4 SU(3) (T = 2: the antiperiodic sign and
the clover leaves wrap the time direction), 2D 4 x 4 SU(3) (one plane), and smooth backgrounds on
2D L x L and 4D T x 2^3 for the continuum checks. NEOQCD_TEST_VERBOSE=1 prints the numbers;
NEOQCD_SLOW=1 enables the HMC against the exact clover Metropolis.
"""

import math
import os
import time
import unittest

import torch

from neoqcd.fermions import hmc as H
from neoqcd.fermions._gauge import SUN_dagger, SUN_haar, SUN_identity, SUN_mul, SUN_trace, plaquette_fields
from neoqcd.fermions.algebra import algebra_basis, expi, project_hermitian_traceless, random_algebra, trace_product
from neoqcd.fermions.clover import (
    CloverDeterminantTerm,
    CloverHamiltonian,
    CloverWilsonDirac,
    ExactCloverMetropolis,
    TwoFlavourClover,
    assemble_clover_even_odd,
    clover_derivative,
    clover_field_strength,
    clover_leaves,
    clover_term,
    log_det_clover_two_flavour,
    planes,
)
from neoqcd.fermions.dirac import WilsonDirac, dense_matrix, gamma_matrices
from neoqcd.fermions.exact import assemble_hopping
from neoqcd.fermions.hmc import link_sum
from neoqcd.fermions.lattice import Geometry, gauge_transform_field, gauge_transform_links, shift
from neoqcd.fermions.solver import solve

G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4, SU(3)
G16 = Geometry(4, 2, 2, 3)  # 2^4, SU(3)
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4, SU(3)
C = torch.complex128
VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))
SLOW = bool(os.environ.get("NEOQCD_SLOW"))


def say(msg):
    if VERBOSE:
        print("\n" + msg)


def haar_links(geometry, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=gen), gen


def replicate(U, K):
    return U.expand((K,) + tuple(U.shape[1:])).contiguous()


def fd_derivative(f, U, X, t=1e-3):
    """4th-order central difference of f(exp(i s X) U) at s = 0."""
    g = lambda s: f(SUN_mul(expi(s * X), U))
    return (-g(2 * t) + 8 * g(t) - 8 * g(-t) + g(-2 * t)) / (12 * t)


def single_link_check(test, F, f, U, links, tol, N):
    """tr(F[link] T^a) against finite differences of f along single-link generators."""
    T = algebra_basis(N)
    worst = 0.0
    for link in links:
        b = link[0]
        for a in range(N * N - 1):
            X = torch.zeros_like(U)
            X[link] = T[a]
            pred = trace_product(F[link], T[a]).item()
            fd = fd_derivative(f, U, X)[b].item()
            worst = max(worst, abs(pred - fd) / max(abs(fd), 1.0))
            test.assertLess(abs(pred - fd), tol * max(abs(fd), 1.0), msg=f"link={link} a={a}")
    return worst


# links (b, mu, t, x, ...) including time links at t = 0 (they carry the boundary sign)
LINKS = {G4: [(0, 0, 0, 0, 0, 0), (1, 0, 0, 1, 1, 0), (1, 1, 2, 1, 0, 1), (0, 3, 3, 1, 1, 0)],
         G16: [(0, 0, 0, 0, 0, 0), (1, 0, 1, 1, 0, 1), (1, 2, 0, 1, 1, 0), (0, 3, 1, 0, 1, 1)],
         G2: [(0, 0, 0, 0), (1, 0, 0, 2), (1, 1, 3, 2), (0, 1, 1, 0)]}


def gamma5_dense(geo):
    _, g5 = gamma_matrices(geo.D)
    return torch.kron(torch.kron(torch.eye(geo.volume, dtype=C), g5), torch.eye(geo.N, dtype=C))


def block_diagonal(T):
    """Dense block diagonal from site matrices (lattice, SN, SN) (one gauge field)."""
    return torch.block_diag(*T.reshape((-1,) + T.shape[-2:]))


# ---- 1. clover term -------------------------------------------------------------------------
class CloverTermTest(unittest.TestCase):

    def test_hermitian_traceless_chiral_blocks(self):
        for geo, seed in ((G4, 1), (G16, 2), (G2, 3)):
            with self.subTest(geometry=geo):
                U, _ = haar_links(geo, 2, seed)
                F = clover_field_strength(U, geo)
                self.assertLess((F + SUN_dagger(F)).abs().max().item(), 1e-15)
                self.assertLess(SUN_trace(F).abs().max().item(), 1e-15)
                kappa = torch.tensor([0.12, 0.10], dtype=torch.float64)
                T = clover_term(U, kappa, 1.3, geo)
                SN = geo.spin * geo.N
                h = SN // 2
                self.assertEqual(T.shape, (2,) + geo.lattice + (SN, SN))
                self.assertLess((T - SUN_dagger(T)).abs().max().item(), 1e-15)
                # block diagonal in chirality (gamma_5 = diag(1, 1, -1, -1) or sigma_3): exactly 0
                self.assertEqual(T[..., :h, h:].abs().max().item(), 0.0)
                self.assertEqual(T[..., h:, :h].abs().max().item(), 0.0)
                # each chiral block is traceless (sigma blocks are traceless in spin)
                for blk in (T[..., :h, :h], T[..., h:, h:]):
                    self.assertLess(SUN_trace(blk).abs().max().item(), 1e-14)
                # [T, gamma_5] = 0
                _, g5 = gamma_matrices(geo.D)
                G5 = torch.kron(g5, torch.eye(geo.N, dtype=C))
                self.assertLess((T @ G5 - G5 @ T).abs().max().item(), 1e-15)
                # nonzero, and proportional to kappa c_sw per batch entry
                T1 = clover_term(U, 1.0, 1.0, geo)
                for b in range(2):
                    self.assertGreater(T1[b].abs().max().item(), 0.1)
                    self.assertLess((T[b] - 1.3 * kappa[b] * T1[b]).abs().max().item(), 1e-15)

    def test_leaves_are_the_four_plaquettes_around_x(self):
        """Re tr Q_{mu nu}(x) = sum of the sun_utils plaquettes at x, x - mu, x - nu, x - mu - nu."""
        for geo, seed in ((G4, 4), (G2, 5)):
            U, _ = haar_links(geo, 2, seed)
            Q = clover_leaves(U, geo)
            for p, ((mu, nu), P) in enumerate(zip(planes(geo.D), plaquette_fields(U, geo.D))):
                Pm = shift(P, mu, -1)
                ref = P + Pm + shift(P, nu, -1) + shift(Pm, nu, -1)
                self.assertLess((SUN_trace(Q[:, p]).real - ref).abs().max().item(), 1e-13)

    def test_gauge_covariance(self):
        for geo, seed in ((G4, 6), (G16, 7)):
            with self.subTest(geometry=geo):
                U, gen = haar_links(geo, 2, seed)
                omega = SUN_haar((2,) + geo.lattice + (geo.N,), generator=gen)
                Ug = gauge_transform_links(U, omega)
                F, Fg = clover_field_strength(U, geo), clover_field_strength(Ug, geo)
                om = omega[:, None]
                self.assertLess((Fg - SUN_mul(SUN_mul(om, F), SUN_dagger(om))).abs().max().item(), 1e-13)
                psi = torch.randn(geo.wilson_shape(2), dtype=C, generator=gen)
                for time_sign in (-1.0, 1.0):
                    D, Dg = (CloverWilsonDirac(V, 0.12, 1.0, geo, time_sign) for V in (U, Ug))
                    lhs = Dg.apply(gauge_transform_field(psi, omega))
                    rhs = gauge_transform_field(D.apply(psi), omega)
                    self.assertLess((lhs - rhs).abs().max().item(), 1e-13)

    def test_time_sign_does_not_enter(self):
        """Every leaf has 0 or 2 time links at the same time slice: no boundary sign, also at T = 2."""
        for geo, seed in ((G16, 8), (G4, 9), (G2, 10)):
            U, _ = haar_links(geo, 2, seed)
            T_ap = clover_term(U, 0.12, 1.0, geo, time_sign=-1.0)
            T_p = clover_term(U, 0.12, 1.0, geo, time_sign=1.0)
            self.assertLess((T_ap - T_p).abs().max().item(), 1e-15)
            # while the leaves of a time plane built from the sign-flipped links individually
            # contain the flipped link (the sign cancels inside each leaf, not by accident of T)
            Q_ap, Q_p = clover_leaves(U, geo, -1.0), clover_leaves(U, geo, 1.0)
            self.assertLess((Q_ap - Q_p).abs().max().item(), 1e-15)

    def test_csw_zero_is_wilson(self):
        U, gen = haar_links(G4, 2, 11)
        k = torch.tensor([0.12, 0.09], dtype=torch.float64)
        psi = torch.randn(G4.wilson_shape(2), dtype=C, generator=gen)
        for time_sign in (-1.0, 1.0):
            Wc, W = CloverWilsonDirac(U, k, 0.0, G4, time_sign), WilsonDirac(U, k, G4, time_sign)
            self.assertEqual((Wc.apply(psi) - W.apply(psi)).abs().max().item(), 0.0)
            self.assertEqual((Wc.even_odd().apply(psi) - W.even_odd().apply(psi)).abs().max().item(), 0.0)

    def test_dense_assembly_and_gamma5_hermiticity(self):
        for geo, seed in ((G4, 12), (G16, 13), (G2, 14)):
            U, _ = haar_links(geo, 2, seed)
            k = torch.tensor([0.12, 0.10], dtype=torch.float64)
            csw = torch.tensor([1.0, 1.7], dtype=torch.float64)
            for time_sign in (-1.0, 1.0):
                with self.subTest(geometry=geo, time_sign=time_sign):
                    w = CloverWilsonDirac(U, k, csw, geo, time_sign)
                    Hd = assemble_hopping(U, geo, time_sign=time_sign)
                    G5 = gamma5_dense(geo)
                    D_hat, ld_odd = assemble_clover_even_odd(U, k, csw, geo, time_sign)
                    for b in range(2):
                        M = dense_matrix(w, geo, batch_index=b)
                        n = M.shape[0]
                        ref = torch.eye(n, dtype=C) + block_diagonal(w.T[b]) - k[b] * Hd[b]
                        self.assertLess((M - ref).abs().max().item(), 1e-14)
                        self.assertLess((M.conj().T - G5 @ M @ G5).abs().max().item(), 1e-14)
                        Mh = dense_matrix(w.even_odd(), geo, batch_index=b)
                        self.assertLess((Mh - D_hat[b]).abs().max().item(), 1e-14)
                        g5h = G5[:Mh.shape[0], :Mh.shape[0]]  # gamma_5 on the even sites
                        self.assertLess((Mh.conj().T - g5h @ Mh @ g5h).abs().max().item(), 1e-14)

    def test_apply_dagger_and_batch_broadcast(self):
        U, gen = haar_links(G16, 2, 15)
        w = CloverWilsonDirac(U, 0.12, 1.0, G16)
        a = torch.randn(G16.wilson_shape(2), dtype=C, generator=gen)
        b = torch.randn(G16.wilson_shape(2), dtype=C, generator=gen)
        for op in (w, w.even_odd()):
            lhs = (a.conj() * op.apply(b)).sum((1, 2, 3, 4, 5, 6))
            rhs = (op.apply_dagger(a).conj() * b).sum((1, 2, 3, 4, 5, 6))
            if op is not w:  # D_hat lives on even sites
                lhs = (op.even(a).conj() * op.apply(op.even(b))).sum((1, 2, 3, 4, 5, 6))
                rhs = (op.apply_dagger(op.even(a)).conj() * op.even(b)).sum((1, 2, 3, 4, 5, 6))
            self.assertLess((lhs - rhs).abs().max().item(), 1e-12)
        one = w.batch_select(1)
        many = one.apply(torch.stack([b[1]] * 3))
        self.assertLess((many - w.apply(b)[1:2]).abs().max().item(), 1e-14)


class ContinuumTest(unittest.TestCase):
    """Smooth backgrounds U_mu(x) = exp(a A_mu(x + a mu / 2)), A anti-hermitian, on a unit torus
    with a = 1/T. The physical coordinate of array index n is x = -a n (x + mu is index n - 1).

    (a) Fhat_{mu nu}(x) = a^2 F_{mu nu}(x) + O(a^4), F = d_mu A_nu - d_nu A_mu + [A_mu, A_nu]:
        relative error falls by ~4 when a halves (abelian and non-abelian backgrounds).
    (b) Tree-level O(a) improvement, which fixes the sign and normalisation of T relative to the
        hopping term: with the naive D_n = (1/2) sum gamma_mu (nabla_mu + nabla*_mu) and the
        Laplacian sum nabla*_mu nabla_mu built here independently from the links,
            (1/2) (D_n^2 - sum nabla* nabla) psi + T / (2 kappa) |_{c_sw = 1} psi = O(a^4),
        i.e. c_sw = 1 cancels the Pauli term of the Wilson term. The ratio |residual| / |T psi|
        falls by ~4 per halving of a; with the opposite sign of T it stays at ~2.
    2D: fields on L x L. 4D: fields depending on x_0 only, on T x 2^3 (all six planes have F != 0,
    the spatial ones from the commutator).
    """

    @staticmethod
    def background(D, T, abelian, seed):
        L = T if D == 2 else 2
        geo = Geometry(D, T, L, 3)
        a = 1.0 / T
        g = torch.Generator().manual_seed(seed)
        n = torch.meshgrid(*[torch.arange(e, dtype=torch.float64) for e in geo.lattice], indexing="ij")
        x = [-a * nn for nn in n]
        dirs = range(D) if D == 2 else (0,)  # coordinates the fields depend on
        gens = algebra_basis(3)[-1:] if abelian else algebra_basis(3)
        ng = gens.shape[0]
        amp = torch.randn(D, ng, D, generator=g, dtype=torch.float64)
        ph = 2 * math.pi * torch.rand(D, ng, D, generator=g, dtype=torch.float64)

        def h(mu, k, xx):
            return sum(amp[mu, k, r] * torch.sin(2 * math.pi * xx[r] + ph[mu, k, r]) for r in dirs)

        def dh(mu, k, r, xx):
            if r not in dirs:
                return torch.zeros_like(xx[0])
            return 2 * math.pi * amp[mu, k, r] * torch.cos(2 * math.pi * xx[r] + ph[mu, k, r])

        def herm(mu, xx):
            return sum(h(mu, k, xx)[..., None, None] * gens[k] for k in range(ng))

        links = []
        for mu in range(D):
            xm = [xx + (0.5 * a if r == mu else 0.0) for r, xx in enumerate(x)]
            links.append(torch.linalg.matrix_exp(1j * a * herm(mu, xm)))
        U = torch.stack(links)[None]
        Fc = []
        for mu, nu in planes(D):
            Am, An = 1j * herm(mu, x), 1j * herm(nu, x)
            d = 1j * sum((dh(nu, k, mu, x) - dh(mu, k, nu, x))[..., None, None] * gens[k] for k in range(ng))
            Fc.append(d + Am @ An - An @ Am)
        g2 = torch.Generator().manual_seed(seed + 1)
        u = torch.randn(geo.spin, 3, generator=g2, dtype=C)
        v = torch.randn(geo.spin, 3, generator=g2, dtype=C)
        last = x[-1] if D == 2 else 2.0 * x[0]  # smooth in the coordinates the fields depend on
        psi = (torch.cos(2 * math.pi * x[0] + 0.3)[..., None, None] * u
               + torch.sin(2 * math.pi * last - 0.7)[..., None, None] * v)[None]
        return geo, U, a * a * torch.stack(Fc, dim=0)[None], psi

    @staticmethod
    def residuals(geo, U, Fa2, psi):
        F = clover_field_strength(U, geo)
        f_err = ((F - Fa2).abs().max() / Fa2.abs().max()).item()
        gam, _ = gamma_matrices(geo.D)
        fw = lambda p, mu: torch.matmul(shift(p, mu, 1), U[:, mu].transpose(-1, -2))  # U_mu(x) p(x+mu)
        bw = lambda p, mu: shift(torch.matmul(p, U[:, mu].conj()), mu, -1)  # U_mu(x-mu)^+ p(x-mu)
        Dn = lambda p: sum(0.5 * torch.matmul(gam[mu], fw(p, mu) - bw(p, mu)) for mu in range(geo.D))
        lap = sum(fw(psi, mu) + bw(psi, mu) - 2 * psi for mu in range(geo.D))
        kappa = 0.1
        Tm = clover_term(U, kappa, 1.0, geo, time_sign=1.0) / (2 * kappa)
        Tpsi = torch.matmul(Tm, psi.reshape(psi.shape[:-2] + (-1, 1))).reshape(psi.shape)
        pauli = 0.5 * (Dn(Dn(psi)) - lap)
        return f_err, ((pauli + Tpsi).norm() / Tpsi.norm()).item(), ((pauli - Tpsi).norm() / Tpsi.norm()).item()

    def test_continuum_limit(self):
        for D, Ts in ((2, (16, 32, 64)), (4, (32, 64, 128))):
            for abelian in (True, False):
                with self.subTest(D=D, abelian=abelian):
                    res = [self.residuals(*self.background(D, T, abelian, 7 + D)) for T in Ts]
                    say(f"D={D} {'abelian' if abelian else 'non-abelian'} T={Ts}: Fhat rel. err "
                        f"{['%.2e' % r[0] for r in res]}, improvement residual {['%.2e' % r[1] for r in res]}, "
                        f"opposite sign {['%.3f' % r[2] for r in res]}")
                    for coarse, fine in zip(res[:-1], res[1:]):
                        self.assertGreater(coarse[0] / fine[0], 3.5)
                        self.assertLess(coarse[0] / fine[0], 4.5)
                        self.assertGreater(coarse[1] / fine[1], 3.5)
                        self.assertLess(coarse[1] / fine[1], 4.5)
                    self.assertLess(res[-1][0], 0.01)
                    self.assertLess(res[-1][1], 0.06)
                    self.assertGreater(res[-1][2], 1.9)


# ---- 2. even-odd ----------------------------------------------------------------------------
class EvenOddTest(unittest.TestCase):

    def test_determinant_factorisation(self):
        """det D = det A_oo det D_hat (slogdet of the full dense D against the even-odd pieces)."""
        for geo, seed in ((G4, 21), (G16, 22), (G2, 23)):
            U, _ = haar_links(geo, 2, seed)
            k = torch.tensor([0.12, 0.10], dtype=torch.float64)
            csw = torch.tensor([1.0, 1.5], dtype=torch.float64)
            ref = log_det_clover_two_flavour(U, k, csw, geo, method="dense_matrix")
            for method in ("even_odd", "assembled"):
                ld = log_det_clover_two_flavour(U, k, csw, geo, method=method)
                self.assertLess((ld - ref).abs().max().item(), 1e-10, msg=f"{geo} {method}")
            # the clover term matters: the Wilson log det differs
            w0 = log_det_clover_two_flavour(U, k, 0.0, geo, method="assembled")
            self.assertGreater((w0 - ref).abs().min().item(), 1e-2)
            # the sign of det D: A_oo is positive and det D = det A_oo det D_hat as complex numbers
            w = CloverWilsonDirac(U, k, csw, geo)
            for b in range(2):
                s_full = torch.linalg.slogdet(dense_matrix(w, geo, batch_index=b))[0]
                s_hat = torch.linalg.slogdet(dense_matrix(w.even_odd(), geo, batch_index=b))[0]
                self.assertLess(abs((s_full - s_hat).item()), 1e-10)

    def test_schur_solve(self):
        for geo, seed in ((G4, 24), (G16, 25)):
            U, gen = haar_links(geo, 2, seed)
            k = torch.tensor([0.12, 0.10], dtype=torch.float64)
            w = CloverWilsonDirac(U, k, 1.2, geo)
            eo = w.even_odd()
            b = torch.randn(geo.wilson_shape(2), dtype=C, generator=gen)
            n_e = geo.volume // 2 * geo.spin * geo.N
            sites = eo.support.reshape(-1).nonzero().squeeze(-1)
            rhs = eo.prepare_rhs(b)
            xe = torch.zeros_like(rhs).reshape(2, geo.volume, -1)
            for i in range(2):  # Schur complement with a dense D_hat solve
                Mh = dense_matrix(eo, geo, batch_index=i)
                r = rhs[i].reshape(geo.volume, -1)[sites].reshape(-1)
                self.assertEqual(r.numel(), n_e)
                xe[i, sites] = torch.linalg.solve(Mh, r).reshape(-1, geo.spin * geo.N)
            x = eo.reconstruct(xe.reshape(rhs.shape), b)
            for i in range(2):
                M = dense_matrix(w, geo, batch_index=i)
                x_ref = torch.linalg.solve(M, b[i].reshape(-1)).reshape(b[i].shape)
                self.assertLess((x[i] - x_ref).abs().max().item() / x_ref.abs().max().item(), 1e-12)
            res = solve(eo, b, tol=1e-12, max_iter=500)
            self.assertLess(res.residual.max().item(), 1e-11)
            self.assertLess((w.apply(res.x) - b).abs().max().item(), 1e-10)


# ---- 3. the clover derivative on its own ----------------------------------------------------
class CloverDerivativeTest(unittest.TestCase):
    """d/dt Re sum tr[X Fhat(e^{itT}U)] against 4th-order finite differences, random weights X."""

    def test_against_finite_differences(self):
        for geo, seed in ((G4, 31), (G16, 32), (G2, 33)):
            with self.subTest(geometry=geo):
                U, gen = haar_links(geo, 2, seed)
                P = len(planes(geo.D))
                X = torch.randn((2, P) + geo.lattice + (geo.N, geo.N), dtype=C, generator=gen)
                f = lambda V: (X * clover_field_strength(V, geo).transpose(-1, -2)).real.reshape(2, -1).sum(1)
                with torch.no_grad():  # as inside the HMC
                    G = clover_derivative(U, X, geo)
                self.assertFalse(G.requires_grad)
                self.assertLess((G - SUN_dagger(G)).abs().max().item(), 1e-14)
                self.assertLess(SUN_trace(G).abs().max().item(), 1e-14)
                for _ in range(2):
                    Y = random_algebra(U.shape, generator=gen)
                    pred = link_sum(trace_product(G, Y))
                    fd = fd_derivative(f, U, Y)
                    self.assertLess(((pred - fd).abs() / fd.abs()).max().item(), 1e-8)
                worst = single_link_check(self, G, f, U, LINKS[geo], 1e-8, geo.N)
                say(f"clover derivative {geo}: worst single-link deviation {worst:.1e}")

    def test_cost(self):
        """Timing of the autograd route against one Wilson-clover application (printed only)."""
        geo = Geometry(4, 4, 4, 3)
        U, gen = haar_links(geo, 1, 34)
        X = torch.randn((1, 6) + geo.lattice + (3, 3), dtype=C, generator=gen)
        w = CloverWilsonDirac(U, 0.12, 1.0, geo)
        psi = torch.randn(geo.wilson_shape(1), dtype=C, generator=gen)
        t0 = time.perf_counter()
        for _ in range(5):
            clover_field_strength(U, geo)
        t1 = time.perf_counter()
        for _ in range(5):
            clover_derivative(U, X, geo)
        t2 = time.perf_counter()
        for _ in range(5):
            w.apply(psi)
        t3 = time.perf_counter()
        say(f"4^4 SU(3): Fhat {1e3 * (t1 - t0) / 5:.2f} ms, clover_derivative {1e3 * (t2 - t1) / 5:.2f} ms, "
            f"one D application {1e3 * (t3 - t2) / 5:.2f} ms")


# ---- 4. forces ------------------------------------------------------------------------------
def dense_pf_action(U, phi, kappa, csw, geo, time_sign, even_odd):
    """S_pf = phi^+ (D_hat^+ D_hat)^-1 phi (or full D) from dense matrices, differentiable in U."""
    w = CloverWilsonDirac(U, kappa, csw, geo, time_sign)
    op = w.even_odd() if even_odd else w
    out = []
    for b in range(U.shape[0]):
        M = dense_matrix(op, geo, batch_index=b)
        v = phi[b].reshape(geo.volume, -1)
        if even_odd:
            v = v[op.support.reshape(-1)]
        v = v.reshape(-1)
        out.append((v.conj() @ torch.linalg.solve(M.conj().T @ M, v)).real)
    return torch.stack(out)


def autograd_force(f, U):
    V = U.detach().clone().requires_grad_(True)
    (G,) = torch.autograd.grad(f(V).sum(), V)
    return project_hermitian_traceless(-1j * SUN_mul(G, SUN_dagger(U)))


class ForceTest(unittest.TestCase):

    def cases(self):
        k = torch.tensor([0.12, 0.10], dtype=torch.float64)
        csw = torch.tensor([1.0, 1.6], dtype=torch.float64)
        yield "4D eo", G4, k, csw, True, -1.0
        yield "4D full", G4, k, csw, False, -1.0
        yield "2^4 eo", G16, k, csw, True, -1.0
        yield "2^4 eo periodic", G16, k, 1.0, True, 1.0
        yield "2D eo", G2, torch.tensor([0.20, 0.17], dtype=torch.float64), csw, True, -1.0

    def test_pseudofermion_force_finite_differences(self):
        for i, (name, geo, k, csw, even_odd, ts) in enumerate(self.cases()):
            with self.subTest(case=name):
                U, gen = haar_links(geo, 2, 40 + i)
                S = TwoFlavourClover(k, csw, geo, time_sign=ts, even_odd=even_odd,
                                     solver_tol=1e-13, action_tol=1e-13, reuse_guess=False)
                S.refresh(U, gen)
                F = S.force(U)
                self.assertLess((F - SUN_dagger(F)).abs().max().item(), 1e-13)
                self.assertLess(SUN_trace(F).abs().max().item(), 1e-13)
                for _ in range(2):
                    X = random_algebra(U.shape, generator=gen)
                    pred = link_sum(trace_product(F, X))
                    fd = fd_derivative(S.action, U, X)
                    self.assertLess(((pred - fd).abs() / fd.abs()).max().item(), 1e-6, msg=f"{pred} {fd}")
                # two links per case (the first is a t = 0 time link): each costs 4 solves per generator
                worst = single_link_check(self, F, S.action, U, LINKS[geo][:2], 1e-6, geo.N)
                say(f"pseudofermion force {name}: worst single-link deviation {worst:.1e}")

    def test_determinant_force_finite_differences(self):
        for i, (name, geo, k, csw, even_odd, ts) in enumerate(self.cases()):
            if not even_odd:
                continue
            with self.subTest(case=name):
                U, gen = haar_links(geo, 2, 50 + i)
                S = CloverDeterminantTerm(k, csw, geo, ts)
                F = S.force(U)
                self.assertLess((F - SUN_dagger(F)).abs().max().item(), 1e-13)
                for _ in range(2):
                    X = random_algebra(U.shape, generator=gen)
                    pred = link_sum(trace_product(F, X))
                    fd = fd_derivative(S.action, U, X)
                    self.assertLess(((pred - fd).abs() / fd.abs()).max().item(), 1e-6)
                worst = single_link_check(self, F, S.action, U, LINKS[geo], 1e-6, geo.N)
                say(f"determinant force {name}: worst single-link deviation {worst:.1e}")
                V = SUN_mul(expi(0.3 * random_algebra(U.shape, generator=gen)), U)
                self.assertLess((S.action_difference(U, V) - (S.action(V) - S.action(U))).abs().max().item(), 1e-12)

    def test_forces_against_autograd_through_dense(self):
        """Both forces against autograd of dense-matrix actions (S_pf via a dense solve; S_det via
        slogdet of the dense block-diagonal A_oo), 2^4 and 2^3 x 4, per-batch kappa and c_sw."""
        k = torch.tensor([0.12, 0.10], dtype=torch.float64)
        csw = torch.tensor([1.0, 1.6], dtype=torch.float64)
        for geo, even_odd, seed in ((G16, True, 61), (G16, False, 62), (G4, True, 63)):
            with self.subTest(geometry=geo, even_odd=even_odd):
                U, gen = haar_links(geo, 2, seed)
                S = TwoFlavourClover(k, csw, geo, even_odd=even_odd, solver_tol=1e-14, action_tol=1e-14,
                                     max_iter=5000)
                S.refresh(U, gen)
                F = S.force(U)
                F_ad = autograd_force(lambda V: dense_pf_action(V, S.phi, k, csw, geo, -1.0, even_odd), U)
                err = ((F - F_ad).abs().max() / F_ad.abs().max()).item()
                self.assertLess(err, 1e-9)
                msg = f"autograd {geo} eo={even_odd}: pf force rel. dev. {err:.1e}"
                if even_odd:
                    Sd = CloverDeterminantTerm(k, csw, geo)
                    odd = Sd.operator(U).odd_mask.reshape(-1) > 0

                    def s_det(V):
                        w = CloverWilsonDirac(V, k, csw, geo)
                        A = (w.T + torch.eye(w.T.shape[-1], dtype=C)).reshape(2, geo.volume, *w.T.shape[-2:])
                        return torch.stack([-2.0 * torch.linalg.slogdet(torch.block_diag(*A[b, odd]))[1]
                                            for b in range(2)])

                    F_ad = autograd_force(s_det, U)
                    err_d = ((Sd.force(U) - F_ad).abs().max() / F_ad.abs().max()).item()
                    self.assertLess(err_d, 1e-9)
                    self.assertLess((s_det(U) - Sd.action(U)).abs().max().item(), 1e-12)
                    msg += f", det force rel. dev. {err_d:.1e}"
                say(msg)


# ---- 5. stochastic estimator of the det ratio ------------------------------------------------
class StochasticEstimatorTest(unittest.TestCase):
    """<exp(-[S_pf(U') - S_pf(U)])>_phi exp(-[S_det(U') - S_det(U)]) = det(D^+ D)(U') / det(D^+ D)(U),
    exact side from dense determinants of the FULL clover operator (no even-odd). U' = exp(i c X) U
    with X of unit norm along (a) the average total force (16 phi + determinant term), c = 0.2,
    and (b) the determinant force, c = 0.5, where det(A_oo)^2 alone changes by ~20%: there the
    pseudofermion estimate without S_det must miss the exact ratio (pins the small determinant
    and its power 2)."""

    def test_determinant_ratio(self):
        geo, kappa, csw, K = G4, 0.12, 1.0, 400
        U1, gen = haar_links(geo, 1, 71)
        probe = TwoFlavourClover(kappa, csw, geo)
        U16 = replicate(U1, 16)
        probe.refresh(U16, gen)
        det_term = CloverDeterminantTerm(kappa, csw, geo)
        F_det = det_term.force(U1)
        U = replicate(U1, K)
        unit = lambda Y: Y / math.sqrt(link_sum(trace_product(Y, Y)).item())
        directions = (("total force", unit(probe.force(U16).mean(0, keepdim=True) + F_det), (0.2,)),
                      ("determinant force", unit(F_det), (0.5,)))
        for name, X1, cs in directions:
            for c in cs:
                with self.subTest(direction=name, c=c):
                    V1 = SUN_mul(expi(c * X1), U1)
                    ld = log_det_clover_two_flavour(torch.cat((U1, V1)), kappa, csw, geo, method="dense_matrix")
                    ratio = math.exp((ld[1] - ld[0]).item())
                    V = replicate(V1, K)
                    S = TwoFlavourClover(kappa, csw, geo)
                    S.refresh(U, gen)
                    w_pf = torch.exp(-S.action_difference(U, V))
                    w = w_pf * torch.exp(-det_term.action_difference(U1, V1))
                    est, err = w.mean().item(), w.std().item() / math.sqrt(K)
                    est_pf, err_pf = w_pf.mean().item(), w_pf.std().item() / math.sqrt(K)
                    say(f"{name} c={c:+.1f}: det ratio exact {ratio:.4f}, stochastic {est:.4f} +- {err:.4f} "
                        f"({abs(ratio - 1) / err:.0f} sigma from 1; pseudofermion only {est_pf:.4f} +- {err_pf:.4f})")
                    self.assertGreater(abs(ratio - 1.0), 5 * err)
                    self.assertLess(abs(est - ratio), 4 * err)
                    if name == "determinant force":
                        self.assertGreater(abs(est_pf - ratio), 5 * err_pf)


# ---- 6. HMC ---------------------------------------------------------------------------------
BETA, KAPPA, CSW = 5.5, 0.12, 1.0


class HMCTest(unittest.TestCase):

    def test_reversibility(self):
        U, gen = haar_links(G16, 3, 81)
        P = H.random_momenta(U.shape, generator=gen)
        ham = CloverHamiltonian(BETA, KAPPA, CSW, G16, n_gauge_substeps=2, reuse_guess=False, solver_tol=1e-12)
        ham.refresh(U, gen)
        self.assertEqual(len(ham.levels), 2)
        self.assertIs(ham.levels[1][0][1], ham.determinant)
        for integrator in H.INTEGRATORS:
            with self.subTest(integrator=integrator):
                hmc = H.HMC(G16, ham, tau=0.5, n_steps=4, integrator=integrator)
                U1, P1, _ = hmc.integrate(U, P)
                U2, P2, _ = hmc.integrate(U1, -P1)
                dU, dP = (U2 - U).abs().max().item(), (P2 + P).abs().max().item()
                say(f"reversibility {integrator}: |U2 - U| = {dU:.1e}, |P2 + P| = {dP:.1e}")
                self.assertGreater((U1 - U).abs().max().item(), 0.1)
                self.assertLess(dU, 1e-11)
                self.assertLess(dP, 1e-11)

    def test_dH_scales_as_eps_squared(self):
        U, gen = haar_links(G16, 4, 82)
        P = H.random_momenta(U.shape, generator=gen)
        ham = CloverHamiltonian(BETA, KAPPA, CSW, G16, n_gauge_substeps=2, solver_tol=1e-12)
        ham.refresh(U, gen)
        steps = (5, 10, 20)
        for integrator in H.INTEGRATORS:
            with self.subTest(integrator=integrator):
                hmc = H.HMC(G16, ham, tau=0.5, n_steps=1, integrator=integrator)
                dH = []
                for n in steps:
                    U1, P1, _ = hmc.integrate(U, P, n_steps=n)
                    dH.append(ham.energy_difference(U, P, U1, P1).abs().mean().item())
                x = torch.tensor([math.log(1.0 / n) for n in steps], dtype=torch.double)
                y = torch.tensor([math.log(d) for d in dH], dtype=torch.double)
                slope = (((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()).item()
                say(f"{integrator}: <|dH|> = {['%.2e' % d for d in dH]}, slope {slope:.3f}")
                self.assertGreater(slope, 1.85)
                self.assertLess(slope, 2.15)

    def test_exp_minus_dH(self):
        """<exp(-dH)> = 1 after thermalisation, 32 chains on 2^4, errors from the chain means."""
        B = 32
        gen = torch.Generator().manual_seed(83)
        U = SUN_identity(G16.link_shape(B)[:-1], dtype=C)
        ham = CloverHamiltonian(BETA, KAPPA, CSW, G16, n_gauge_substeps=2)
        hmc = H.HMC(G16, ham, tau=1.0, n_steps=3, integrator="omelyan")
        x, acc = [], []
        for i in range(18):
            U, accepted, dH, stats = hmc.trajectory(U, gen)
            if i >= 6:
                x.append(stats["exp_minus_dH"])
                acc.append(stats["acceptance"])
        m = torch.stack(x).mean(0)
        mean, err = m.mean().item(), (m.std() / math.sqrt(B)).item()
        st = ham.fermion.stats
        say(f"<exp(-dH)> = {mean:.4f} +- {err:.4f}, acceptance {sum(acc) / len(acc):.3f}, "
            f"CG iterations per solve {st['cg_iterations'] / st['solves']:.1f}")
        self.assertLess(abs(mean - 1.0), 4 * err)
        self.assertGreater(sum(acc) / len(acc), 0.7)


@unittest.skipUnless(SLOW, "set NEOQCD_SLOW=1: HMC against the exact clover Metropolis (~6 min)")
class ExactComparisonTest(unittest.TestCase):
    """Plaquette and log det(D^+ D) of the clover HMC against `ExactCloverMetropolis` on 2^4 SU(3),
    beta = 5.5, kappa = 0.12, c_sw = 1, B = 16 chains; errors from the chain means.

    The HMC starts cold (20 thermalisation trajectories, then 150 measured). The single-link
    Metropolis thermalises far too slowly from a cold start (acceptance ~0.1-0.2: after 60 sweeps
    its <log det> is still at -7.5 against ~15 in equilibrium), so it starts from the thermalised
    HMC configurations and runs 50 + 200 sweeps. If the HMC distribution were wrong, the exact
    chain would relax away from it over these 250 sweeps (its log det moved by ~12 in 60 sweeps
    from the cold start). A quenched HMC run is the reference that makes the log det comparison
    discriminating (asserted: exact and quenched <log det> differ by > 4 sigma).
    """

    B = 16

    def chain_stats(self, x):
        m = torch.stack(x).mean(0)
        return m.mean().item(), (m.std() / math.sqrt(self.B)).item()

    def observables(self, U):
        from neoqcd.fermions._gauge import mean_plaquette
        return (mean_plaquette(plaquette_fields(U, G16.D), G16.N),
                log_det_clover_two_flavour(U, KAPPA, CSW, G16, method="assembled"))

    def run_hmc(self, ham, gen, n_therm, n_meas):
        hmc = H.HMC(G16, ham, tau=1.0, n_steps=3, integrator="omelyan")
        U = SUN_identity(G16.link_shape(self.B)[:-1], dtype=C)
        plaq, logdet, U_therm = [], [], None
        for i in range(n_therm + n_meas):
            U, _, _, _ = hmc.trajectory(U, gen)
            if i == n_therm - 1:
                U_therm = U.clone()
            if i >= n_therm:
                p, l = self.observables(U)
                plaq.append(p)
                logdet.append(l)
        return self.chain_stats(plaq), self.chain_stats(logdet), U_therm

    def test_plaquette_and_log_det(self):
        gen = torch.Generator().manual_seed(91)
        ham = CloverHamiltonian(BETA, KAPPA, CSW, G16, n_gauge_substeps=2)
        (p_h, e_ph), (l_h, e_lh), U_therm = self.run_hmc(ham, gen, 20, 150)
        (p_q, e_pq), (l_q, e_lq), _ = self.run_hmc(H.WilsonGaugeHamiltonian(BETA, G16), gen, 20, 150)
        m = ExactCloverMetropolis(BETA, KAPPA, CSW, G16, eps=0.35)
        m.reset(U_therm)
        pm, lm = [], []
        n_links = G16.D * G16.volume
        for sweep in range(50 + 200):
            for _ in range(n_links):
                m.step(gen)
            if sweep >= 50:
                p, l = self.observables(m.U)
                self.assertLess((m.log_det_value - l).abs().max().item(), 1e-10)
                pm.append(p)
                lm.append(l)
        (p_m, e_pm), (l_m, e_lm) = self.chain_stats(pm), self.chain_stats(lm)
        msg = (f"2^4 beta={BETA} kappa={KAPPA} csw={CSW}:\n"
               f"  <P>       HMC {p_h:.4f}({e_ph:.4f})  exact {p_m:.4f}({e_pm:.4f})  quenched {p_q:.4f}({e_pq:.4f})"
               f"  [HMC-exact {abs(p_h - p_m) / math.hypot(e_ph, e_pm):.2f} sigma]\n"
               f"  <log det> HMC {l_h:.4f}({e_lh:.4f})  exact {l_m:.4f}({e_lm:.4f})  quenched {l_q:.4f}({e_lq:.4f})"
               f"  [HMC-exact {abs(l_h - l_m) / math.hypot(e_lh, e_lm):.2f} sigma, exact-quenched "
               f"{abs(l_m - l_q) / math.hypot(e_lm, e_lq):.2f} sigma]\n"
               f"  Metropolis acceptance {m.acceptance:.3f}")
        say(msg)
        self.assertLess(abs(p_h - p_m), 4 * math.hypot(e_ph, e_pm), msg=msg)
        self.assertLess(abs(l_h - l_m), 4 * math.hypot(e_lh, e_lm), msg=msg)
        self.assertGreater(abs(l_m - l_q), 4 * math.hypot(e_lm, e_lq), msg=msg)


if __name__ == "__main__":
    unittest.main()
