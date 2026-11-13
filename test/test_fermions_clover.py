"""Tests of the Sheikholeslami-Wohlert (clover) N_f = 2 Wilson fermions (neoqcd.fermions.clover).

CPU, complex128, fixed seeds. Lattices: 2^3 x 4 and 2^4 SU(3) (T = 2: the antiperiodic sign and
the clover leaves wrap the time direction), 2D 4 x 4 SU(3) (one plane), 4^4 SU(3) against an
independent operator built from explicit site tables (with extent 2, x + mu = x - mu and a
reversed shift goes unnoticed), and smooth backgrounds on 2D L x L and 4D T x 2^3 for the
continuum checks. The convention-independent checks run in both conventions of Fhat
(`traceless` False = openQCD, the default, and True). NEOQCD_TEST_VERBOSE=1 prints the numbers;
NEOQCD_SLOW=1 enables the HMC against an independent exact sampler.
"""

import itertools
import math
import os
import time
import unittest

import torch

from neoqcd.fermions import hmc as H
from neoqcd.fermions._gauge import SUN_dagger, SUN_haar, SUN_identity, SUN_mul, SUN_trace, plaquette_fields
from neoqcd.fermions.algebra import algebra_basis, expi, project_hermitian_traceless, random_algebra, trace_product
from neoqcd.fermions.clover import (
    SINGULAR_THRESHOLD,
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
CONVENTIONS = (False, True)  # traceless: openQCD u(N) Fhat (default), su(N) projection


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
        for (geo, seed), tl in itertools.product(((G4, 1), (G16, 2), (G2, 3)), CONVENTIONS):
            with self.subTest(geometry=geo, traceless=tl):
                U, _ = haar_links(geo, 2, seed)
                F = clover_field_strength(U, geo, traceless=tl)
                self.assertLess((F + SUN_dagger(F)).abs().max().item(), 1e-15)
                if tl:
                    self.assertLess(SUN_trace(F).abs().max().item(), 1e-15)
                else:  # openQCD: (Q - Q^+) / 8 as it is, tr = (i/4) Im tr Q, nonzero for SU(3)
                    Q = clover_leaves(U, geo)
                    trF = SUN_trace(F)
                    self.assertLess((trF - 0.25j * SUN_trace(Q).imag).abs().max().item(), 1e-15)
                    self.assertGreater(trF.abs().max().item(), 1e-2)
                    F_tl = clover_field_strength(U, geo, traceless=True)
                    eye = torch.eye(geo.N, dtype=C)
                    self.assertLess((F - F_tl - trF[..., None, None] * eye / geo.N).abs().max().item(), 1e-15)
                kappa = torch.tensor([0.12, 0.10], dtype=torch.float64)
                T = clover_term(U, kappa, 1.3, geo, traceless=tl)
                SN = geo.spin * geo.N
                h = SN // 2
                self.assertEqual(T.shape, (2,) + geo.lattice + (SN, SN))
                self.assertLess((T - SUN_dagger(T)).abs().max().item(), 1e-15)
                # block diagonal in chirality (gamma_5 = diag(1, 1, -1, -1) or sigma_3): exactly 0
                self.assertEqual(T[..., :h, h:].abs().max().item(), 0.0)
                self.assertEqual(T[..., h:, :h].abs().max().item(), 0.0)
                # 4D: each chiral block is traceless (the chiral blocks of sigma_{mu nu} are
                # traceless 2 x 2 matrices, so also with tr Fhat != 0); 2D: the blocks are
                # -/+ i kappa c_sw Fhat_01 (sigma_01 = -sigma_3), traceless only for traceless=True;
                # T itself is traceless in both
                tr_blk = [SUN_trace(blk) for blk in (T[..., :h, :h], T[..., h:, h:])]
                self.assertLess((tr_blk[0] + tr_blk[1]).abs().max().item(), 1e-14)
                if geo.D == 4 or tl:
                    self.assertLess(tr_blk[0].abs().max().item(), 1e-14)
                else:
                    k_view = (1.3 * kappa).view(2, 1, 1)
                    self.assertLess((tr_blk[0] + 1j * k_view * SUN_trace(F[:, 0])).abs().max().item(), 1e-14)
                    self.assertGreater(tr_blk[0].abs().max().item(), 1e-3)
                # [T, gamma_5] = 0
                _, g5 = gamma_matrices(geo.D)
                G5 = torch.kron(g5, torch.eye(geo.N, dtype=C))
                self.assertLess((T @ G5 - G5 @ T).abs().max().item(), 1e-15)
                # nonzero, and proportional to kappa c_sw per batch entry
                T1 = clover_term(U, 1.0, 1.0, geo, traceless=tl)
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
        for (geo, seed), tl in itertools.product(((G4, 6), (G16, 7)), CONVENTIONS):
            with self.subTest(geometry=geo, traceless=tl):
                U, gen = haar_links(geo, 2, seed)
                omega = SUN_haar((2,) + geo.lattice + (geo.N,), generator=gen)
                Ug = gauge_transform_links(U, omega)
                F, Fg = clover_field_strength(U, geo, traceless=tl), clover_field_strength(Ug, geo, traceless=tl)
                om = omega[:, None]
                self.assertLess((Fg - SUN_mul(SUN_mul(om, F), SUN_dagger(om))).abs().max().item(), 1e-13)
                psi = torch.randn(geo.wilson_shape(2), dtype=C, generator=gen)
                for time_sign in (-1.0, 1.0):
                    D, Dg = (CloverWilsonDirac(V, 0.12, 1.0, geo, time_sign, tl) for V in (U, Ug))
                    lhs = Dg.apply(gauge_transform_field(psi, omega))
                    rhs = gauge_transform_field(D.apply(psi), omega)
                    self.assertLess((lhs - rhs).abs().max().item(), 1e-13)

    def test_time_sign_does_not_enter(self):
        """Every leaf has 0 or 2 time links at the same time slice: no boundary sign, also at T = 2."""
        for geo, seed in ((G16, 8), (G4, 9), (G2, 10)):
            U, _ = haar_links(geo, 2, seed)
            for tl in CONVENTIONS:
                T_ap = clover_term(U, 0.12, 1.0, geo, time_sign=-1.0, traceless=tl)
                T_p = clover_term(U, 0.12, 1.0, geo, time_sign=1.0, traceless=tl)
                self.assertLess((T_ap - T_p).abs().max().item(), 1e-15)
            # the leaves themselves are unchanged by the sign of the whole t = 0 slice of time
            # links (default time_sign = -1 against +1) ...
            Q_ap, Q_p = clover_leaves(U, geo), clover_leaves(U, geo, 1.0)
            self.assertLess((Q_ap - Q_p).abs().max().item(), 1e-15)
            # ... although they do contain those links: flipping the sign of ONE t = 0 time link
            # changes the leaves of the time planes (the sign cancels in pairs inside each leaf,
            # not because the leaves miss the boundary links)
            V = U.clone()
            V[:, 0, 0, 0] = -V[:, 0, 0, 0]
            Q_one = clover_leaves(V, geo, 1.0)
            time_planes = [p for p, (mu, nu) in enumerate(planes(geo.D)) if mu == 0]
            self.assertGreater((Q_one - Q_p)[:, time_planes].abs().max().item(), 0.1)

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
            for time_sign, tl in itertools.product((-1.0, 1.0), CONVENTIONS):
                with self.subTest(geometry=geo, time_sign=time_sign, traceless=tl):
                    w = CloverWilsonDirac(U, k, csw, geo, time_sign, tl)
                    Hd = assemble_hopping(U, geo, time_sign=time_sign)
                    G5 = gamma5_dense(geo)
                    D_hat, ld_odd = assemble_clover_even_odd(U, k, csw, geo, time_sign, tl)
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
    def residuals(geo, U, Fa2, psi, traceless):
        F = clover_field_strength(U, geo, traceless=traceless)
        f_err = ((F - Fa2).abs().max() / Fa2.abs().max()).item()
        gam, _ = gamma_matrices(geo.D)
        fw = lambda p, mu: torch.matmul(shift(p, mu, 1), U[:, mu].transpose(-1, -2))  # U_mu(x) p(x+mu)
        bw = lambda p, mu: shift(torch.matmul(p, U[:, mu].conj()), mu, -1)  # U_mu(x-mu)^+ p(x-mu)
        Dn = lambda p: sum(0.5 * torch.matmul(gam[mu], fw(p, mu) - bw(p, mu)) for mu in range(geo.D))
        lap = sum(fw(psi, mu) + bw(psi, mu) - 2 * psi for mu in range(geo.D))
        kappa = 0.1
        Tm = clover_term(U, kappa, 1.0, geo, time_sign=1.0, traceless=traceless) / (2 * kappa)
        Tpsi = torch.matmul(Tm, psi.reshape(psi.shape[:-2] + (-1, 1))).reshape(psi.shape)
        pauli = 0.5 * (Dn(Dn(psi)) - lap)
        return f_err, ((pauli + Tpsi).norm() / Tpsi.norm()).item(), ((pauli - Tpsi).norm() / Tpsi.norm()).item()

    def test_continuum_limit(self):
        for D, Ts in ((2, (16, 32, 64)), (4, (32, 64, 128))):
            for abelian, tl in itertools.product((True, False), CONVENTIONS):
                with self.subTest(D=D, abelian=abelian, traceless=tl):
                    res = [self.residuals(*self.background(D, T, abelian, 7 + D), tl) for T in Ts]
                    say(f"D={D} {'abelian' if abelian else 'non-abelian'} traceless={tl} T={Ts}: Fhat rel. err "
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
        for (geo, seed), tl in itertools.product(((G4, 21), (G16, 22), (G2, 23)), CONVENTIONS):
            U, _ = haar_links(geo, 2, seed)
            k = torch.tensor([0.12, 0.10], dtype=torch.float64)
            csw = torch.tensor([1.0, 1.5], dtype=torch.float64)
            ref = log_det_clover_two_flavour(U, k, csw, geo, method="dense_matrix", traceless=tl)
            for method in ("even_odd", "assembled"):
                ld = log_det_clover_two_flavour(U, k, csw, geo, method=method, traceless=tl)
                self.assertLess((ld - ref).abs().max().item(), 1e-10, msg=f"{geo} {method} {tl}")
            # the clover term matters: the Wilson log det differs
            w0 = log_det_clover_two_flavour(U, k, 0.0, geo, method="assembled")
            self.assertGreater((w0 - ref).abs().min().item(), 1e-2)
            if geo.N == 3 and not tl:  # and so does the convention for SU(3)
                ref_tl = log_det_clover_two_flavour(U, k, csw, geo, method="assembled", traceless=True)
                self.assertGreater((ref_tl - ref).abs().min().item(), 1e-6)
            # the sign of det D: A_oo is positive and det D = det A_oo det D_hat as complex numbers
            w = CloverWilsonDirac(U, k, csw, geo, traceless=tl)
            for b in range(2):
                s_full = torch.linalg.slogdet(dense_matrix(w, geo, batch_index=b))[0]
                s_hat = torch.linalg.slogdet(dense_matrix(w.even_odd(), geo, batch_index=b))[0]
                self.assertLess(abs((s_full - s_hat).item()), 1e-10)

    def test_schur_solve(self):
        for (geo, seed), tl in itertools.product(((G4, 24), (G16, 25)), CONVENTIONS):
            U, gen = haar_links(geo, 2, seed)
            k = torch.tensor([0.12, 0.10], dtype=torch.float64)
            w = CloverWilsonDirac(U, k, 1.2, geo, traceless=tl)
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
        for (geo, seed), tl in itertools.product(((G4, 31), (G16, 32), (G2, 33)), CONVENTIONS):
            with self.subTest(geometry=geo, traceless=tl):
                U, gen = haar_links(geo, 2, seed)
                P = len(planes(geo.D))
                X = torch.randn((2, P) + geo.lattice + (geo.N, geo.N), dtype=C, generator=gen)
                f = lambda V: (X * clover_field_strength(V, geo, traceless=tl).transpose(-1, -2)).real.reshape(2, -1).sum(1)
                with torch.no_grad():  # as inside the HMC
                    G = clover_derivative(U, X, geo, traceless=tl)
                self.assertFalse(G.requires_grad)
                self.assertLess((G - SUN_dagger(G)).abs().max().item(), 1e-14)
                self.assertLess(SUN_trace(G).abs().max().item(), 1e-14)
                for _ in range(2):
                    Y = random_algebra(U.shape, generator=gen)
                    pred = link_sum(trace_product(G, Y))
                    fd = fd_derivative(f, U, Y)
                    self.assertLess(((pred - fd).abs() / fd.abs()).max().item(), 1e-8)
                worst = single_link_check(self, G, f, U, LINKS[geo], 1e-8, geo.N)
                say(f"clover derivative {geo} traceless={tl}: worst single-link deviation {worst:.1e}")

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
def dense_pf_action(U, phi, kappa, csw, geo, time_sign, even_odd, traceless=False):
    """S_pf = phi^+ (D_hat^+ D_hat)^-1 phi (or full D) from dense matrices, differentiable in U."""
    w = CloverWilsonDirac(U, kappa, csw, geo, time_sign, traceless)
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
        # name, geometry, kappa, c_sw, even_odd, time_sign, traceless
        yield "4D eo", G4, k, csw, True, -1.0, False
        yield "4D full", G4, k, csw, False, -1.0, False
        yield "2^4 eo", G16, k, csw, True, -1.0, False
        yield "2^4 eo periodic", G16, k, 1.0, True, 1.0, False
        yield "2D eo", G2, torch.tensor([0.20, 0.17], dtype=torch.float64), csw, True, -1.0, False
        yield "4D eo traceless", G4, k, csw, True, -1.0, True
        yield "4D full traceless", G4, k, csw, False, -1.0, True
        yield "2^4 eo traceless", G16, k, csw, True, -1.0, True

    def test_pseudofermion_force_finite_differences(self):
        for i, (name, geo, k, csw, even_odd, ts, tl) in enumerate(self.cases()):
            with self.subTest(case=name):
                U, gen = haar_links(geo, 2, 40 + i)
                S = TwoFlavourClover(k, csw, geo, time_sign=ts, even_odd=even_odd, traceless=tl,
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
        for i, (name, geo, k, csw, even_odd, ts, tl) in enumerate(self.cases()):
            if not even_odd:
                continue
            with self.subTest(case=name):
                U, gen = haar_links(geo, 2, 50 + i)
                S = CloverDeterminantTerm(k, csw, geo, ts, tl)
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
        cases = ((G16, True, 61), (G16, False, 62), (G4, True, 63))
        for (geo, even_odd, seed), tl in itertools.product(cases, CONVENTIONS):
            with self.subTest(geometry=geo, even_odd=even_odd, traceless=tl):
                U, gen = haar_links(geo, 2, seed)
                S = TwoFlavourClover(k, csw, geo, even_odd=even_odd, solver_tol=1e-14, action_tol=1e-14,
                                     max_iter=5000, traceless=tl)
                S.refresh(U, gen)
                F = S.force(U)
                F_ad = autograd_force(lambda V: dense_pf_action(V, S.phi, k, csw, geo, -1.0, even_odd, tl), U)
                err = ((F - F_ad).abs().max() / F_ad.abs().max()).item()
                self.assertLess(err, 1e-9)
                msg = f"autograd {geo} eo={even_odd} traceless={tl}: pf force rel. dev. {err:.1e}"
                if even_odd:
                    Sd = CloverDeterminantTerm(k, csw, geo, traceless=tl)
                    odd = Sd.operator(U).odd_mask.reshape(-1) > 0

                    def s_det(V):
                        w = CloverWilsonDirac(V, k, csw, geo, traceless=tl)
                        A = (w.T + torch.eye(w.T.shape[-1], dtype=C)).reshape(2, geo.volume, *w.T.shape[-2:])
                        return torch.stack([-2.0 * torch.linalg.slogdet(torch.block_diag(*A[b, odd]))[1]
                                            for b in range(2)])

                    F_ad = autograd_force(s_det, U)
                    err_d = ((Sd.force(U) - F_ad).abs().max() / F_ad.abs().max()).item()
                    self.assertLess(err_d, 1e-9)
                    self.assertLess((s_det(U) - Sd.action(U)).abs().max().item(), 1e-12)
                    msg += f", det force rel. dev. {err_d:.1e}"
                say(msg)


# ---- 5. independent operator on 4^4 ---------------------------------------------------------
def _indep_gammas(D):
    """Chiral gamma matrices of openQCD, typed here (no neoqcd code)."""
    s1 = torch.tensor([[0, 1], [1, 0]], dtype=C)
    s2 = torch.tensor([[0, -1j], [1j, 0]], dtype=C)
    s3 = torch.tensor([[1, 0], [0, -1]], dtype=C)
    if D == 2:
        return [s1, s2]
    Z, I = torch.zeros(2, 2, dtype=C), torch.eye(2, dtype=C)
    g0 = torch.cat((torch.cat((Z, -I), 1), torch.cat((-I, Z), 1)), 0)
    gk = [torch.cat((torch.cat((Z, -1j * s), 1), torch.cat((1j * s, Z), 1)), 0) for s in (s1, s2, s3)]
    return [g0] + gk


def _dag(M):
    return M.conj().transpose(-1, -2)


class IndependentClover:
    """The clover-Wilson operator of the module docstring of neoqcd.fermions.clover, rebuilt from
    site tables made by explicit loops over the coordinates ("x + k mu" = array index n - k mod L
    along mu) and gathers; no neoqcd shift, clover or Dirac code. Differentiable in U.
    Fields: links (B, D, V, N, N), Wilson fields (B, V, S, N), V lexicographic."""

    def __init__(self, geo, time_sign):
        lat, D = geo.lattice, geo.D
        sites = list(itertools.product(*[range(e) for e in lat]))
        index = {n: i for i, n in enumerate(sites)}

        def step(n, mu, k):  # x + k mu
            m = list(n)
            m[mu] = (m[mu] - k) % lat[mu]
            return index[tuple(m)]

        self.up = [torch.tensor([step(n, mu, 1) for n in sites]) for mu in range(D)]
        self.dn = [torch.tensor([step(n, mu, -1) for n in sites]) for mu in range(D)]
        self.par = torch.tensor([sum(n) % 2 for n in sites])
        self.sign = torch.ones(D, len(sites), dtype=torch.float64)
        self.sign[0, [i for i, n in enumerate(sites) if n[0] == 0]] = time_sign
        self.geo, self.V = geo, len(sites)
        self.g = _indep_gammas(D)
        self.planes = [(mu, nu) for mu in range(D) for nu in range(mu + 1, D)]

    def links(self, U):
        return U.reshape(U.shape[0], self.geo.D, self.V, self.geo.N, self.geo.N)

    def field_strength(self, U, traceless):
        """Fhat per plane, dict (mu, nu) -> (B, V, N, N)."""
        Uf = self.links(U)
        L = lambda mu, idx: Uf[:, mu][:, idx]
        x = torch.arange(self.V)
        out = {}
        for mu, nu in self.planes:
            up, dn = self.up, self.dn
            xpm, xpn, xmm, xmn = up[mu], up[nu], dn[mu], dn[nu]
            Q = (L(mu, x) @ L(nu, xpm) @ _dag(L(mu, xpn)) @ _dag(L(nu, x))
                 + L(nu, x) @ _dag(L(mu, up[nu][xmm])) @ _dag(L(nu, xmm)) @ L(mu, xmm)
                 + _dag(L(mu, xmm)) @ _dag(L(nu, dn[nu][xmm])) @ L(mu, dn[nu][xmm]) @ L(nu, xmn)
                 + _dag(L(nu, xmn)) @ L(mu, xmn) @ L(nu, dn[nu][xpm]) @ _dag(L(mu, x)))
            F = (Q - _dag(Q)) / 8.0
            if traceless:
                tr = F.diagonal(dim1=-2, dim2=-1).sum(-1)
                F = F - tr[..., None, None] * torch.eye(self.geo.N, dtype=C) / self.geo.N
            out[(mu, nu)] = F
        return out

    def A(self, U, kappa, csw, traceless):
        """1 + T per site, (B, V, S N, S N), index s * N + c."""
        F = self.field_strength(U, traceless)
        S, N = len(self.g[0]), self.geo.N
        T = 0
        for mu, nu in self.planes:
            sig = 0.5j * (self.g[mu] @ self.g[nu] - self.g[nu] @ self.g[mu])
            T = T + torch.einsum("st,bvcd->bvsctd", sig, F[(mu, nu)]).reshape(F[(mu, nu)].shape[:2] + (S * N, S * N))
        c = (torch.as_tensor(kappa, dtype=torch.float64) * torch.as_tensor(csw, dtype=torch.float64)).reshape(-1, 1, 1, 1)
        return torch.eye(S * N, dtype=C) + 1j * c * T

    def hop(self, U, psi):
        Uf = self.links(U)
        one = torch.eye(len(self.g[0]), dtype=C)
        out = 0
        for mu in range(self.geo.D):
            L = Uf[:, mu] * self.sign[mu][None, :, None, None]
            fwd = torch.einsum("bvcd,bvsd->bvsc", L, psi[:, self.up[mu]])                      # L(x) psi(x+mu)
            bwd = torch.einsum("bvdc,bvsd->bvsc", L[:, self.dn[mu]].conj(), psi[:, self.dn[mu]])  # L(x-mu)^+ psi(x-mu)
            out = out + torch.einsum("st,bvtc->bvsc", one - self.g[mu], fwd) \
                      + torch.einsum("st,bvtc->bvsc", one + self.g[mu], bwd)
        return out

    @staticmethod
    def site(M, psi):
        return torch.matmul(M, psi.reshape(psi.shape[:2] + (-1, 1))).reshape(psi.shape)

    def apply(self, U, kappa, csw, traceless, psi):
        """D psi = (1 + T) psi - kappa H psi, psi (B, V, S, N)."""
        k = torch.as_tensor(kappa, dtype=torch.float64).reshape(-1, 1, 1, 1)
        return self.site(self.A(U, kappa, csw, traceless), psi) - k * self.hop(U, psi)

    def apply_eo(self, U, kappa, csw, traceless, psi):
        """D_hat psi_e = A_ee psi_e - kappa^2 H_eo A_oo^-1 H_oe psi_e (even sites, odd entries 0)."""
        k = torch.as_tensor(kappa, dtype=torch.float64).reshape(-1, 1, 1, 1)
        ev = (self.par == 0).to(torch.float64)[None, :, None, None]
        od = 1.0 - ev
        A = self.A(U, kappa, csw, traceless)
        pe = ev * psi
        inner = od * self.site(torch.linalg.inv(A), od * self.hop(U, pe))
        return ev * (self.site(A, pe) - k ** 2 * self.hop(U, inner))

    def det_action(self, U, kappa, csw, traceless):
        """-2 sum_{x odd} log|det A(x)|, shape (B,)."""
        ld = torch.linalg.slogdet(self.A(U, kappa, csw, traceless))[1]
        return -2.0 * ld[:, self.par == 1].sum(1)


class Independent4x4Test(unittest.TestCase):
    """4^4 SU(3) (every extent > 2, so x + mu != x - mu in all directions) with fields that depend
    on all coordinates: Fhat, D, D_hat and both forces against `IndependentClover`. On 2^4 or
    2^3 x 4 a clover leaf built with the spatial shifts reversed coincides with the right one; here
    it does not."""

    geo = Geometry(4, 4, 4, 3)

    def field(self, seed):
        gen = torch.Generator().manual_seed(seed)
        U = expi(0.6 * random_algebra(self.geo.link_shape(2), generator=gen))
        return U, gen

    def test_operator(self):
        geo = self.geo
        U, gen = self.field(101)
        k = torch.tensor([0.12, 0.11], dtype=torch.float64)
        csw = torch.tensor([1.3, 1.0], dtype=torch.float64)
        psi = torch.randn(geo.wilson_shape(2), dtype=C, generator=gen)
        pf = psi.reshape(2, geo.volume, geo.spin, geo.N)
        for ts, tl in itertools.product((-1.0, 1.0), CONVENTIONS):
            with self.subTest(time_sign=ts, traceless=tl):
                ind = IndependentClover(geo, ts)
                F = clover_field_strength(U, geo, ts, tl).reshape(2, len(planes(4)), geo.volume, 3, 3)
                Fi = ind.field_strength(U, tl)
                for p, pl in enumerate(planes(4)):
                    self.assertLess((F[:, p] - Fi[pl]).abs().max().item(), 1e-14, msg=f"plane {pl}")
                w = CloverWilsonDirac(U, k, csw, geo, ts, tl)
                d1 = (w.apply(psi).reshape(pf.shape) - ind.apply(U, k, csw, tl, pf)).abs().max().item()
                d2 = (w.even_odd().apply(psi).reshape(pf.shape) - ind.apply_eo(U, k, csw, tl, pf)).abs().max().item()
                say(f"4^4 independent operator ts={ts} traceless={tl}: |D - D_ind| {d1:.1e}, |D_hat - D_hat_ind| {d2:.1e}")
                self.assertLess(d1, 1e-13)
                self.assertLess(d2, 1e-13)

    def test_forces(self):
        """Determinant force against autograd of the independent S_det; pseudofermion force against
        autograd of -2 Re[Y^+ M_ind(U) X] at fixed X = (M^+ M)^-1 phi, Y = M X (M = D_hat or D),
        which is dS_pf = -X^+ d(M^+ M) X with all link dependence (A_oo^-1 included) in M_ind."""
        geo = self.geo
        U, gen = self.field(102)
        k = torch.tensor([0.12, 0.11], dtype=torch.float64)
        csw = torch.tensor([1.3, 1.0], dtype=torch.float64)
        ind = IndependentClover(geo, -1.0)
        shape = (2, geo.volume, geo.spin, geo.N)
        for tl in CONVENTIONS:
            with self.subTest(term="det", traceless=tl):
                F = CloverDeterminantTerm(k, csw, geo, traceless=tl).force(U)
                F_ad = autograd_force(lambda V: ind.det_action(V, k, csw, tl), U)
                err = ((F - F_ad).abs().max() / F_ad.abs().max()).item()
                say(f"4^4 det force traceless={tl}: rel. dev. from independent autograd {err:.1e}")
                self.assertLess(err, 1e-11)
            for even_odd in (True, False):
                with self.subTest(term="pf", even_odd=even_odd, traceless=tl):
                    S = TwoFlavourClover(k, csw, geo, even_odd=even_odd, traceless=tl, solver_tol=1e-13,
                                         max_iter=5000)
                    S.refresh(U, gen)
                    F = S.force(U)
                    M = ind.apply_eo if even_odd else ind.apply
                    X = S.X.reshape(shape).detach()
                    Y = M(U, k, csw, tl, X).detach()
                    f = lambda V: -2.0 * (Y.conj() * M(V, k, csw, tl, X)).real.reshape(2, -1).sum(1)
                    F_ad = autograd_force(f, U)
                    err = ((F - F_ad).abs().max() / F_ad.abs().max()).item()
                    say(f"4^4 pf force eo={even_odd} traceless={tl}: rel. dev. from independent autograd {err:.1e}")
                    self.assertLess(err, 1e-9)


# ---- 6. stochastic estimator of the det ratio, through the Hamiltonian -----------------------
def plaquette_sum(U):
    """sum_x sum_{mu<nu} Re tr P_{mu nu}(x), shape (B,), from torch.roll (no neoqcd code)."""
    D = U.shape[1]
    tot = 0
    for mu in range(D):
        for nu in range(mu + 1, D):
            Um, Un = U[:, mu], U[:, nu]
            P = Um @ torch.roll(Un, 1, dims=1 + mu) @ _dag(torch.roll(Um, 1, dims=1 + nu)) @ _dag(Un)
            tot = tot + P.diagonal(dim1=-2, dim2=-1).sum(-1).real.reshape(U.shape[0], -1).sum(1)
    return tot


BETA, KAPPA, CSW = 5.5, 0.12, 1.0


class StochasticEstimatorTest(unittest.TestCase):
    """<exp(-[S(U') - S(U)])>_phi = exp(-[S_g(U') - S_g(U)]) det(D^+ D)(U') / det(D^+ D)(U), with
    S(U') - S(U) = `CloverHamiltonian.action_difference` (all its terms, as the HMC accept/reject
    uses them). Exact side: S_g from plaquettes computed here, the determinants from dense
    matrices of the FULL clover operator (no even-odd). Hamiltonians: even-odd (pseudofermion
    plus small-determinant term) and even_odd=False (pseudofermion only). U' = exp(i c X) U with X
    of unit norm along (a) the average total force (16 phi + determinant term), c = 0.2, and
    (b) the determinant force, c = 0.5, where det(A_oo)^2 alone changes by ~20%: there the
    even-odd Hamiltonian without its determinant term must miss the exact ratio (pins the small
    determinant, its power 2 and its presence in the Hamiltonian). The estimates are divided by
    the exact gauge factor, so a missing gauge term fails as well."""

    def test_determinant_ratio(self):
        geo, kappa, csw, K = G4, 0.12, 1.0, 400
        U1, gen = haar_links(geo, 1, 71)
        probe = TwoFlavourClover(kappa, csw, geo)
        U16 = replicate(U1, 16)
        probe.refresh(U16, gen)
        F_det = CloverDeterminantTerm(kappa, csw, geo).force(U1)
        U = replicate(U1, K)
        unit = lambda Y: Y / math.sqrt(link_sum(trace_product(Y, Y)).item())
        directions = (("total force", unit(probe.force(U16).mean(0, keepdim=True) + F_det), 0.2),
                      ("determinant force", unit(F_det), 0.5))
        variants = (("even-odd", {}), ("full", {"even_odd": False}))
        for name, X1, c in directions:
            V1 = SUN_mul(expi(c * X1), U1)
            ld = log_det_clover_two_flavour(torch.cat((U1, V1)), kappa, csw, geo, method="dense_matrix")
            ratio = math.exp((ld[1] - ld[0]).item())
            dSg = BETA * (plaquette_sum(U1) - plaquette_sum(V1)).item() / geo.N
            V = replicate(V1, K)
            for vname, kw in variants:
                with self.subTest(direction=name, hamiltonian=vname):
                    ham = CloverHamiltonian(BETA, kappa, csw, geo, **kw)
                    ham.refresh(U, gen)
                    dS = ham.action_difference(U, V)
                    w = torch.exp(-(dS - dSg))
                    est, err = w.mean().item(), w.std().item() / math.sqrt(K)
                    msg = (f"{name} c={c:+.1f} {vname}: det ratio exact {ratio:.4f}, stochastic {est:.4f} +- {err:.4f} "
                           f"({abs(ratio - 1) / err:.0f} sigma from 1)")
                    if ham.determinant is not None:
                        w_pf = torch.exp(-(dS - ham.determinant.action_difference(U, V) - dSg))
                        est_pf, err_pf = w_pf.mean().item(), w_pf.std().item() / math.sqrt(K)
                        msg += f"; without the determinant term {est_pf:.4f} +- {err_pf:.4f}"
                        self.assertGreater(abs(ratio - 1.0), 5 * err)
                        if name == "determinant force":
                            self.assertGreater(abs(est_pf - ratio), 5 * err_pf)
                    say(msg)
                    self.assertLess(abs(est - ratio), 4 * err, msg=msg)


# ---- 7. HMC ---------------------------------------------------------------------------------
# (name, CloverHamiltonian keyword arguments, integrators)
HMC_VARIANTS = (("eo, det at gauge level", {}, H.INTEGRATORS),
                ("eo, det at fermion level", {"det_level": "fermion"}, ("omelyan",)),
                ("full, no det term", {"even_odd": False}, ("omelyan",)),
                ("eo traceless", {"traceless": True}, ("omelyan",)))


class HMCTest(unittest.TestCase):

    def check_levels(self, ham, kw):
        self.assertEqual(len(ham.levels), 2)
        outer, inner = ham.levels[0][0], ham.levels[1][0]
        self.assertIs(outer[0], ham.fermion)
        self.assertIs(inner[0], ham.gauge)
        if kw.get("even_odd", True):
            self.assertIs((outer if kw.get("det_level") == "fermion" else inner)[1], ham.determinant)
            self.assertEqual(len(outer) + len(inner), 3)
        else:
            self.assertIsNone(ham.determinant)
            self.assertEqual(len(outer) + len(inner), 2)

    def test_reversibility(self):
        U, gen = haar_links(G16, 3, 81)
        P = H.random_momenta(U.shape, generator=gen)
        for vname, kw, integrators in HMC_VARIANTS:
            ham = CloverHamiltonian(BETA, KAPPA, CSW, G16, n_gauge_substeps=2, reuse_guess=False,
                                    solver_tol=1e-12, **kw)
            self.check_levels(ham, kw)
            ham.refresh(U, gen)
            for integrator in integrators:
                with self.subTest(variant=vname, integrator=integrator):
                    hmc = H.HMC(G16, ham, tau=0.5, n_steps=4, integrator=integrator)
                    U1, P1, _ = hmc.integrate(U, P)
                    U2, P2, _ = hmc.integrate(U1, -P1)
                    dU, dP = (U2 - U).abs().max().item(), (P2 + P).abs().max().item()
                    say(f"reversibility {vname} {integrator}: |U2 - U| = {dU:.1e}, |P2 + P| = {dP:.1e}")
                    self.assertGreater((U1 - U).abs().max().item(), 0.1)
                    self.assertLess(dU, 1e-11)
                    self.assertLess(dP, 1e-11)

    def test_dH_scales_as_eps_squared(self):
        U, gen = haar_links(G16, 4, 82)
        P = H.random_momenta(U.shape, generator=gen)
        steps = (5, 10, 20)
        for i, (vname, kw, integrators) in enumerate(HMC_VARIANTS):
            ham = CloverHamiltonian(BETA, KAPPA, CSW, G16, n_gauge_substeps=2, solver_tol=1e-12, **kw)
            Uv, Pv = (U, P) if i == 0 else (U[:2], P[:2])  # the other variants on 2 of the 4 fields
            ham.refresh(Uv, gen)
            for integrator in integrators:
                with self.subTest(variant=vname, integrator=integrator):
                    hmc = H.HMC(G16, ham, tau=0.5, n_steps=1, integrator=integrator)
                    dH = []
                    for n in steps:
                        U1, P1, _ = hmc.integrate(Uv, Pv, n_steps=n)
                        dH.append(ham.energy_difference(Uv, Pv, U1, P1).abs().mean().item())
                    x = torch.tensor([math.log(1.0 / n) for n in steps], dtype=torch.double)
                    y = torch.tensor([math.log(d) for d in dH], dtype=torch.double)
                    slope = (((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()).item()
                    say(f"{vname} {integrator}: <|dH|> = {['%.2e' % d for d in dH]}, slope {slope:.3f}")
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


# ---- 8. validation --------------------------------------------------------------------------
class ValidationTest(unittest.TestCase):

    def test_det_level(self):
        for even_odd in (True, False):
            with self.subTest(even_odd=even_odd):
                with self.assertRaisesRegex(ValueError, "det_level"):
                    CloverHamiltonian(BETA, KAPPA, CSW, G16, det_level="outer", even_odd=even_odd)

    def test_singular_A(self):
        """T is linear in c_sw: at c_sw = -1 / t, t an eigenvalue of T(c_sw = 1), A = 1 + T has an
        exact zero eigenvalue and the operator refuses; slightly below, it does not."""
        U, _ = haar_links(G16, 2, 111)
        T1 = clover_term(U, KAPPA, 1.0, G16)
        t = torch.linalg.eigvalsh(T1).min().item()  # < 0
        csw = -1.0 / t
        with self.assertRaisesRegex(ValueError, "singular"):
            CloverWilsonDirac(U, KAPPA, csw, G16)
        with self.assertRaisesRegex(ValueError, "singular"):
            log_det_clover_two_flavour(U, KAPPA, csw, G16, method="assembled")
        w = CloverWilsonDirac(U, KAPPA, 0.9 * csw, G16)
        self.assertGreater(w.min_abs_eig_A.min().item(), 0.05)
        self.assertGreater(w.min_abs_eig_A.min().item(), SINGULAR_THRESHOLD)
        # A is allowed to be indefinite (|det A| enters): just past the zero crossing it works
        w = CloverWilsonDirac(U, KAPPA, 1.1 * csw, G16)
        self.assertLess(torch.linalg.eigvalsh(w.T + torch.eye(12, dtype=C)).min().item(), 0.0)

    def test_private_couplings(self):
        U, gen = haar_links(G16, 2, 112)
        k = torch.tensor([0.12, 0.10], dtype=torch.float64)
        csw = torch.tensor([1.0, 1.5], dtype=torch.float64)
        terms = (TwoFlavourClover(k, csw, G16), CloverDeterminantTerm(k, csw, G16))
        S_det = terms[1].action(U)
        terms[0].refresh(U, gen)
        S_pf = terms[0].action(U)
        k.fill_(0.2)
        csw.fill_(3.0)
        for term in terms:
            self.assertEqual(term.kappa.tolist(), [0.12, 0.10])
            self.assertEqual(term.csw.tolist(), [1.0, 1.5])
        self.assertEqual((terms[1].action(U) - S_det).abs().max().item(), 0.0)
        self.assertLess((terms[0].action(U) - S_pf).abs().max().item(), 1e-9)

    def test_time_sign_defaults(self):
        import inspect
        for f in (clover_leaves, clover_field_strength, clover_term):
            self.assertEqual(inspect.signature(f).parameters["time_sign"].default, -1.0, msg=f.__name__)
        for f in (clover_field_strength, clover_term, clover_derivative):
            self.assertIs(inspect.signature(f).parameters["traceless"].default, False, msg=f.__name__)


# ---- 9. slow: HMC against an independent exact sampler --------------------------------------
@unittest.skipUnless(SLOW, "set NEOQCD_SLOW=1: HMC against an independent exact sampler (~6 min)")
class ExactComparisonTest(unittest.TestCase):
    """Plaquette and log det(D^+ D) of the clover HMC against an independent exact sampler on 2^4
    SU(3), beta = 5.5, kappa = 0.12, c_sw = 1 (default convention, traceless=False).

    Exact sampler (written here, not `ExactCloverMetropolis`): each proposal is N_HIT = 64
    single-link Metropolis hits of the PURE GAUGE theory (random link, U -> exp(i eps P) U, P
    Gaussian, eps = 0.35, accepted with min(1, exp(-dS_g)), S_g from `plaquette_sum`). One hit is
    a kernel reversible w.r.t. exp(-S_g), so is its N_HIT-th power, and the Metropolis-Hastings
    step with that proposal accepts with min(1, det(D^+ D)(U') / det(D^+ D)(U)) exactly (the
    determinant from `log_det_clover_two_flavour`, validated against dense matrices in the fast
    tests). B = 16 chains start cold and 16 hot; N_BURN proposals are discarded, N_MEAS measured.

    Why not single-link Metropolis on the exact action from a cold start: on 2^4 the spatial
    Polyakov loops (extent 2) sit in Z3 centre sectors, and a cold start begins in the trivial
    sector of all of them, which is not the dominant one with dynamical fermions (the fermion
    determinant favours the non-trivial sectors); single-link moves on the full action leave it
    very slowly. The pure-gauge sub-chain moves between sectors freely enough. The test asserts
    that the cold and hot exact chains agree, that the HMC (cold start, 20 + 150 trajectories)
    agrees with the exact sampler, and that the comparison discriminates: the quenched HMC
    <log det> differs from the exact one by > 4 sigma. Errors from the chain means (4 sigma)."""

    B, N_HIT, EPS, N_BURN, N_MEAS = 16, 64, 0.35, 800, 2000

    @staticmethod
    def log_det(U, traceless=False):
        return log_det_clover_two_flavour(U, KAPPA, CSW, G16, method="assembled", traceless=traceless)

    @staticmethod
    def gauge_action(U):
        n_planes = G16.D * (G16.D - 1) // 2
        return BETA * (n_planes * G16.volume - plaquette_sum(U) / G16.N)

    @staticmethod
    def plaquette(U):
        n_planes = G16.D * (G16.D - 1) // 2
        return plaquette_sum(U) / (G16.N * n_planes * G16.volume)

    def chain_stats(self, x):
        m = torch.stack(x).mean(0)
        return m.mean().item(), (m.std() / math.sqrt(m.shape[0])).item()

    def run_hmc(self, ham, gen, n_therm, n_meas):
        hmc = H.HMC(G16, ham, tau=1.0, n_steps=3, integrator="omelyan")
        U = SUN_identity(G16.link_shape(self.B)[:-1], dtype=C)
        plaq, logdet = [], []
        for i in range(n_therm + n_meas):
            U, _, _, _ = hmc.trajectory(U, gen)
            if i >= n_therm:
                plaq.append(self.plaquette(U))
                logdet.append(self.log_det(U))
        return self.chain_stats(plaq), self.chain_stats(logdet)

    def run_exact(self, gen):
        B2, n_links = 2 * self.B, G16.D * G16.volume
        cold = SUN_identity(G16.link_shape(self.B)[:-1], dtype=C)
        hot = SUN_haar(G16.link_shape(self.B)[:-1], generator=gen)
        U = torch.cat((cold, hot))
        self.assertLess((self.gauge_action(hot) - H.WilsonGaugeAction(BETA, G16).action(hot)).abs().max().item(), 1e-10)
        Sg, ld = self.gauge_action(U), self.log_det(U)
        flat = (B2, n_links, G16.N, G16.N)
        bshape = (B2,) + (1,) * (U.dim() - 1)
        plaq, logdet, d_conv = [], [], []
        acc_g = acc_d = 0.0
        for it in range(self.N_BURN + self.N_MEAS):
            V, SgV = U, Sg
            for _ in range(self.N_HIT):
                link = int(torch.randint(n_links, (1,), generator=gen))
                W = V.reshape(flat).clone()
                W[:, link] = expi(self.EPS * random_algebra((B2, G16.N, G16.N), generator=gen)) @ W[:, link]
                W = W.reshape(U.shape)
                SgW = self.gauge_action(W)
                a = torch.log(torch.rand(B2, generator=gen, dtype=torch.float64)) < -(SgW - SgV)
                V, SgV = torch.where(a.view(bshape), W, V), torch.where(a, SgW, SgV)
                acc_g += a.double().mean().item()
            ldV = self.log_det(V)
            a = torch.log(torch.rand(B2, generator=gen, dtype=torch.float64)) < (ldV - ld)
            U, Sg, ld = torch.where(a.view(bshape), V, U), torch.where(a, SgV, Sg), torch.where(a, ldV, ld)
            acc_d += a.double().mean().item()
            if it >= self.N_BURN:
                plaq.append(self.plaquette(U))
                logdet.append(ld)
                if it % 50 == 0:  # the two conventions of Fhat on the same configurations
                    d_conv.append(ld - self.log_det(U, traceless=True))
        self.assertLess((Sg - self.gauge_action(U)).abs().max().item(), 1e-8)
        self.assertLess((ld - self.log_det(U)).abs().max().item(), 1e-8)
        n = self.N_BURN + self.N_MEAS
        return plaq, logdet, d_conv, acc_g / (n * self.N_HIT), acc_d / n

    def test_plaquette_and_log_det(self):
        t0 = time.perf_counter()
        gen = torch.Generator().manual_seed(91)
        plaq, logdet, d_conv, acc_g, acc_d = self.run_exact(gen)
        t1 = time.perf_counter()
        ham = CloverHamiltonian(BETA, KAPPA, CSW, G16, n_gauge_substeps=2)
        (p_h, e_ph), (l_h, e_lh) = self.run_hmc(ham, gen, 20, 150)
        (p_q, e_pq), (l_q, e_lq) = self.run_hmc(H.WilsonGaugeHamiltonian(BETA, G16), gen, 20, 150)
        t2 = time.perf_counter()
        half = slice(0, self.B), slice(self.B, 2 * self.B)
        (p_c, e_pc), (p_t, e_pt) = (self.chain_stats([x[h] for x in plaq]) for h in half)
        (l_c, e_lc), (l_t, e_lt) = (self.chain_stats([x[h] for x in logdet]) for h in half)
        (p_m, e_pm), (l_m, e_lm) = self.chain_stats(plaq), self.chain_stats(logdet)
        d_m, e_dm = self.chain_stats(d_conv)
        sig = lambda a, ea, b, eb: abs(a - b) / math.hypot(ea, eb)
        msg = (f"2^4 beta={BETA} kappa={KAPPA} csw={CSW} (openQCD Fhat):\n"
               f"  <P>       exact cold {p_c:.4f}({e_pc:.4f}) hot {p_t:.4f}({e_pt:.4f}) [{sig(p_c, e_pc, p_t, e_pt):.2f} sigma]"
               f"  all {p_m:.4f}({e_pm:.4f})  HMC {p_h:.4f}({e_ph:.4f}) [{sig(p_h, e_ph, p_m, e_pm):.2f} sigma]"
               f"  quenched {p_q:.4f}({e_pq:.4f})\n"
               f"  <log det> exact cold {l_c:.3f}({e_lc:.3f}) hot {l_t:.3f}({e_lt:.3f}) [{sig(l_c, e_lc, l_t, e_lt):.2f} sigma]"
               f"  all {l_m:.3f}({e_lm:.3f})  HMC {l_h:.3f}({e_lh:.3f}) [{sig(l_h, e_lh, l_m, e_lm):.2f} sigma]"
               f"  quenched {l_q:.3f}({e_lq:.3f}) [exact-quenched {sig(l_m, e_lm, l_q, e_lq):.1f} sigma]\n"
               f"  <log det(openQCD) - log det(traceless)> on the exact configurations {d_m:.4f}({e_dm:.4f})\n"
               f"  exact sampler: gauge-hit acceptance {acc_g:.3f}, determinant acceptance {acc_d:.3f}; "
               f"time exact {t1 - t0:.0f} s, HMC {t2 - t1:.0f} s")
        say(msg)
        self.assertLess(abs(p_c - p_t), 4 * math.hypot(e_pc, e_pt), msg=msg)
        self.assertLess(abs(l_c - l_t), 4 * math.hypot(e_lc, e_lt), msg=msg)
        self.assertLess(abs(p_h - p_m), 4 * math.hypot(e_ph, e_pm), msg=msg)
        self.assertLess(abs(l_h - l_m), 4 * math.hypot(e_lh, e_lm), msg=msg)
        self.assertGreater(abs(l_m - l_q), 4 * math.hypot(e_lm, e_lq), msg=msg)


if __name__ == "__main__":
    unittest.main()
