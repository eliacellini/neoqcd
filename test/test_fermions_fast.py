"""Tests of the opt-in fast path (neoqcd.fermions.fast) against the reference implementation.

Every kernel is compared with `EvenOddWilsonDirac`, `solver.cg`, `pseudofermion.hopping_derivative`
and `TwoFlavourWilson` on random Haar links (CPU, tiny lattices). 4^3 x 6 is included because
there x + mu and x - mu are different sites in every direction (on extent 2 they coincide, which
would hide a swapped forward/backward table).
"""

import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

from neoqcd.fermions import hmc as H
from neoqcd.fermions._gauge import SUN_haar
from neoqcd.fermions.dirac import WilsonDirac, gamma_matrices
from neoqcd.fermions.fast import (
    FastTwoFlavourHamiltonian,
    FastTwoFlavourWilson,
    HalfLatticeWilson,
    colour_mul,
    lean_cg,
    mixed_precision_cg,
    resolve_colour_product,
    site_tables,
    spin_projection_tables,
    to_full,
    to_half,
)
from neoqcd.fermions.lattice import Geometry, even_odd_masks
from neoqcd.fermions.pseudofermion import (
    TwoFlavourHamiltonian,
    TwoFlavourWilson,
    UnconvergedSolveWarning,
    hopping_derivative,
)
from neoqcd.fermions.solver import _norm2, cg

ROOT = Path(__file__).resolve().parents[1]

G4s = Geometry(4, 4, 2, 3)  # 2^3 x 4
G44 = Geometry(4, 4, 4, 3)  # 4^4
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4
G46 = Geometry(4, 6, 4, 3)  # 4^3 x 6: +mu and -mu neighbours differ everywhere
C, C64 = torch.complex128, torch.complex64


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


def clustered_system(levels, n=48, batch=2, seed=5):
    """Dense hermitian positive-definite A = Q diag(levels, each repeated) Q^dagger and a rhs:
    CG converges in ~len(levels) iterations, but with levels spanning 1e8 the true residual
    |b - A x| / |b| has a floor ~1e-8 while the recursive one keeps falling."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=C, generator=g))
    d = torch.tensor(levels, dtype=torch.float64).repeat_interleave(n // len(levels))
    A = (Q * d.to(C)) @ Q.conj().T
    A = 0.5 * (A + A.conj().T)
    b = torch.randn(batch, n, dtype=C, generator=g)
    A64 = A.to(C64)
    return (lambda x: x @ A.T), (lambda x: x @ A64.T), b


class HalfLatticeOperatorTest(unittest.TestCase):
    def test_matches_even_odd_reference(self):
        """D_hat, D_hat^dagger, normal against EvenOddWilsonDirac: 1e-13 (complex128), 1e-5 (complex64)."""
        k = kappas(0.11, 0.13)
        for geo in (G4s, G44, G2, G46):
            U = links(geo, 2, 1)
            psi = field(geo, 2, 2)
            x = to_half(psi, geo)
            for ts in (-1.0, 1.0):
                eo = WilsonDirac(U, k, geo, time_sign=ts).even_odd()
                op = HalfLatticeWilson(U, k, geo, time_sign=ts)
                op64 = HalfLatticeWilson(U.to(C64), k, geo, time_sign=ts)
                for name in ("apply", "apply_dagger", "normal"):
                    with self.subTest(lattice=geo.lattice, time_sign=ts, op=name):
                        ref_full = getattr(eo, name)(psi)
                        self.assertEqual(to_half(ref_full, geo, 1).abs().max().item(), 0.0)
                        ref = to_half(ref_full, geo)
                        out = getattr(op, name)(x)
                        self.assertEqual(out.dtype, C)
                        self.assertEqual(tuple(out.shape), op.field_shape)
                        self.assertLess(rel(out, ref), 1e-13)
                        out64 = getattr(op64, name)(x.to(C64))
                        self.assertEqual(out64.dtype, C64)
                        self.assertLess(rel(out64, ref), 1e-5)
                        # to() from the complex128 operator gives the same complex64 operator
                        self.assertLess(rel(getattr(op.to(C64), name)(x.to(C64)), ref), 1e-5)

    def test_hops_rhs_and_reconstruction(self):
        """H_eo, H_oe, their daggers (gamma_5 H gamma_5), prepare_rhs and reconstruct on 4^3 x 6."""
        geo = G46
        U = links(geo, 2, 3)
        k = kappas(0.12, 0.14)
        w = WilsonDirac(U, k, geo)
        eo = w.even_odd()
        op = HalfLatticeWilson(U, k, geo)
        psi = field(geo, 2, 4)
        e, o = to_half(psi, geo, 0), to_half(psi, geo, 1)
        self.assertLess(rel(op.hop_eo(o), to_half(w.hopping(eo.odd(psi)), geo, 0)), 1e-13)
        self.assertLess(rel(op.hop_oe(e), to_half(w.hopping(eo.even(psi)), geo, 1)), 1e-13)
        self.assertLess(rel(op.hop_eo(o, dagger=True),
                            to_half(w.gamma5(w.hopping(w.gamma5(eo.odd(psi)))), geo, 0)), 1e-13)
        self.assertLess(rel(op.hop_oe(e, dagger=True),
                            to_half(w.gamma5(w.hopping(w.gamma5(eo.even(psi)))), geo, 1)), 1e-13)
        self.assertLess(rel(op.gamma5(e), to_half(w.gamma5(psi), geo)), 1e-15)
        self.assertLess(rel(op.prepare_rhs(psi), to_half(eo.prepare_rhs(psi), geo)), 1e-13)
        x_e = to_half(field(geo, 2, 5), geo)
        self.assertLess(rel(op.reconstruct(x_e, psi), eo.reconstruct(to_full(x_e, geo), psi)), 1e-13)

    def test_upcast_recovers_double_precision_links(self):
        """to(C64).to(C128) and dtype=C64 then to(C128) re-gather the complex128 links (item 2)."""
        geo = G46
        U = links(geo, 2, 60)
        k = kappas(0.12, 0.13)
        psi = field(geo, 2, 61)
        x = to_half(psi, geo)
        ref = to_half(WilsonDirac(U, k, geo).even_odd().normal(psi), geo)
        op = HalfLatticeWilson(U, k, geo)
        for name, up in (("to(C64).to(C128)", op.to(C64).to(C)),
                         ("dtype=C64 .to(C128)", HalfLatticeWilson(U, k, geo, dtype=C64).to(C))):
            with self.subTest(name):
                self.assertEqual(up.dtype, C)
                self.assertEqual(up._links[0].dtype, C)
                self.assertLess(rel(up.normal(x), ref), 1e-13)
                self.assertTrue(torch.equal(up._links[1], op._links[1]))
        # the complex64 operator itself is untouched by the upcast copy
        op64 = op.to(C64)
        op64.to(C)
        self.assertEqual(op64._links[0].dtype, C64)

    def test_site_order_both_parities_lexicographic(self):
        """sites[0] and sites[1] are the even / odd sites in lexicographic order (t, x_1, ...)."""
        for geo in (G4s, G2, G46):
            even, odd = even_odd_masks(geo, dtype=torch.float64)
            tab = site_tables(geo)
            for p, mask in ((0, even), (1, odd)):
                sites = mask.reshape(-1).nonzero().squeeze(-1)
                self.assertTrue(torch.equal(tab["sites"][p], sites), (geo.lattice, p))
                self.assertTrue(bool((sites[1:] > sites[:-1]).all()))
                # and to_half picks exactly those sites in that order
                psi = field(geo, 1, 62)
                flat = psi.reshape(1, geo.volume, geo.spin, geo.N)
                self.assertTrue(torch.equal(to_half(psi, geo, p), flat[:, sites]))

    def test_conversion_round_trip(self):
        for geo in (G4s, G2, G46):
            psi = field(geo, 3, 6)
            even, odd = even_odd_masks(geo, dtype=torch.float64)
            e, o = to_half(psi, geo, 0), to_half(psi, geo, 1)
            self.assertEqual(tuple(e.shape), (3, geo.volume // 2, geo.spin, geo.N))
            self.assertTrue(torch.equal(to_full(e, geo), even * psi))
            self.assertTrue(torch.equal(to_full(o, geo, parity=1), odd * psi))
            self.assertTrue(torch.equal(to_full(e, geo, odd=o), psi))
            self.assertTrue(torch.equal(to_half(to_full(e, geo), geo), e))
            # lexicographic order of the even sites, as in dense_matrix / EvenOddWilsonDirac
            sites = even.reshape(-1).nonzero().squeeze(-1)
            self.assertTrue(torch.equal(site_tables(geo)["sites"][0], sites))

    def test_neighbour_tables_differ(self):
        """On 4^3 x 6 the forward and backward neighbours are different sites in every direction."""
        tab = site_tables(G46)
        for mu in range(G46.D):
            self.assertTrue(bool((tab["fwd"][mu] != tab["bwd"][mu]).all()))
            fwd = tab["hop_index"][0].view(-1, 2 * G46.D)[:, mu] // (2 * G46.D)
            bwd = tab["hop_index"][0].view(-1, 2 * G46.D)[:, G46.D + mu] // (2 * G46.D)
            self.assertTrue(bool((fwd != bwd).all()))

    def test_dtype_and_batch_handling(self):
        geo = G4s
        U = links(geo, 2, 7)
        op = HalfLatticeWilson(U, 0.12, geo)
        x = to_half(field(geo, 2, 8), geo)
        with self.assertRaises(TypeError):
            op.normal(x.to(C64))
        with self.assertRaises(TypeError):
            op.to(C64).normal(x)
        # float32 kappa tensor: cast, no promotion of complex64 fields
        op64 = HalfLatticeWilson(U.to(C64), torch.tensor([0.12, 0.13], dtype=torch.float32), geo)
        self.assertEqual(op64.normal(x.to(C64)).dtype, C64)
        # links of batch 1 applied to a batch of fields
        op1 = HalfLatticeWilson(U[:1], 0.12, geo)
        ref = WilsonDirac(U[:1], 0.12, geo).even_odd().normal(to_full(x, geo))
        self.assertLess(rel(op1.normal(x), to_half(ref, geo)), 1e-13)

    def test_hopping_derivative_matches_reference(self):
        k = kappas(0.12, 0.13)
        for geo in (G46, G2):
            for ts in (-1.0, 1.0):
                with self.subTest(lattice=geo.lattice, time_sign=ts):
                    U = links(geo, 2, 9)
                    w = WilsonDirac(U, k, geo, time_sign=ts)
                    op = HalfLatticeWilson(U, k, geo, time_sign=ts)
                    a, b = field(geo, 2, 10), field(geo, 2, 11)
                    ref = hopping_derivative(w, a, b)
                    out = op.hopping_derivative(to_half(a, geo), to_half(a, geo, 1), to_half(b, geo),
                                                to_half(b, geo, 1))
                    self.assertEqual(out.shape, ref.shape)
                    self.assertLess(rel(out, ref), 1e-13)


class ColourProductTest(unittest.TestCase):
    """The broadcast colour products equal torch.matmul (CPU, complex128 to 1e-14)."""

    MODES = ("broadcast", "broadcast_sum")

    def test_colour_mul(self):
        g = torch.Generator().manual_seed(40)
        a = torch.randn(3, 5, 4, 2, 3, dtype=C, generator=g)
        b = torch.randn(3, 5, 4, 3, 3, dtype=C, generator=g)
        ref = torch.matmul(a, b)
        for mode in self.MODES:
            self.assertLess(rel(colour_mul(a, b, mode), ref), 1e-14)
            # strided, conjugated and broadcast operands, as in the force
            self.assertLess(rel(colour_mul(b.transpose(-1, -2), b[:1].conj(), mode),
                                torch.matmul(b.transpose(-1, -2), b[:1].conj())), 1e-14)
        self.assertEqual(resolve_colour_product("auto", "cpu"), "matmul")
        self.assertEqual(resolve_colour_product("auto", "cuda"), "broadcast_sum")
        with self.assertRaises(ValueError):
            resolve_colour_product("einsum", "cpu")

    def test_modes_agree(self):
        k = kappas(0.11, 0.13)
        for geo in (G46, G2):
            U = links(geo, 2, 41)
            psi = field(geo, 2, 42)
            x, xo = to_half(psi, geo), to_half(psi, geo, 1)
            ref = HalfLatticeWilson(U, k, geo, colour_product="matmul")
            self.assertEqual(HalfLatticeWilson(U, k, geo).colour_product, "matmul")  # auto on CPU
            for mode in self.MODES:
                with self.subTest(lattice=geo.lattice, mode=mode):
                    op = HalfLatticeWilson(U, k, geo, colour_product=mode)
                    for name in ("apply", "apply_dagger", "normal"):
                        self.assertLess(rel(getattr(op, name)(x), getattr(ref, name)(x)), 1e-14)
                    self.assertLess(rel(op.hop_eo(xo, dagger=True), ref.hop_eo(xo, dagger=True)), 1e-14)
                    self.assertLess(rel(op.hopping_derivative(x, xo, xo.flip(1), x.flip(1)),
                                        ref.hopping_derivative(x, xo, xo.flip(1), x.flip(1))), 1e-14)
                    op64, ref64 = op.to(C64), ref.to(C64)
                    self.assertEqual(op64.colour_product, mode)
                    self.assertLess(rel(op64.normal(x.to(C64)), ref64.normal(x.to(C64)).to(C)), 1e-6)

    def test_term_modes_agree(self):
        geo = G4s
        U, U2 = links(geo, 2, 43), links(geo, 2, 44)
        out = {}
        for mode in ("matmul", "broadcast"):
            term = FastTwoFlavourWilson(0.12, geo, solver_tol=1e-12, colour_product=mode, reuse_guess=False)
            term.refresh(U, torch.Generator().manual_seed(45))
            out[mode] = (term.action(U2), term.force(U2))
            self.assertEqual(term.operator(U2).colour_product, mode)
        self.assertLess(rel(out["broadcast"][0], out["matmul"][0]), 1e-12)
        self.assertLess(rel(out["broadcast"][1], out["matmul"][1]), 1e-10)
        with self.assertRaises(ValueError):
            FastTwoFlavourWilson(0.12, geo, colour_product="einsum")


class SpinProjectionTest(unittest.TestCase):
    """h = u + s g d and (1 + s gamma_mu) psi = (h, s g^dagger h) for every direction and sign."""

    def test_tables_against_projectors(self):
        g = torch.Generator().manual_seed(12)
        for D in (2, 4):
            gammas, gamma5 = gamma_matrices(D)
            S = gammas.shape[-1]
            Hh = S // 2
            one = torch.eye(S, dtype=C)
            tab = spin_projection_tables(D)
            psi = torch.randn(S, 3, dtype=C, generator=g)
            for k in range(2 * D):
                mu, s = k % D, (-1.0 if k < D else 1.0)
                for dagger in (False, True):
                    with self.subTest(D=D, mu=mu, s=s, dagger=dagger):
                        sd = -s if dagger else s
                        sgn = -1.0 if dagger else 1.0
                        P = one + sd * gammas[mu]
                        perm = tab["proj_perm"][k * Hh:(k + 1) * Hh]
                        h = psi[:Hh] + sgn * tab["proj_coef"][k][:, None] * psi[Hh + perm]
                        # reconstruction of the lower half from the flat (term, j) tables
                        chi = torch.zeros(2 * D * Hh, 3, dtype=C)
                        chi[k * Hh:(k + 1) * Hh] = h
                        rows = slice(k * Hh, (k + 1) * Hh)
                        lower = sgn * tab["rec_coef"][rows][:, None] * chi[tab["rec_perm"][rows]]
                        self.assertLess((torch.cat((h, lower)) - P @ psi).abs().max().item(), 1e-14)
                        # P = R R^dagger with R = [1; s g^dagger], R^dagger R = 2
                        gmu = tab["g"][mu]
                        R = torch.cat((torch.eye(Hh, dtype=C), sd * gmu.conj().T))
                        self.assertLess((R @ R.conj().T - P).abs().max().item(), 1e-14)
                        self.assertLess((R.conj().T @ R - 2 * torch.eye(Hh, dtype=C)).abs().max().item(), 1e-14)
                        self.assertLess((R.conj().T @ psi - h).abs().max().item(), 1e-14)
                        # H^dagger = gamma_5 H gamma_5: the dagger tables are the flipped projector
                        self.assertLess((gamma5 @ (one + s * gammas[mu]) @ gamma5
                                         - (one - s * gammas[mu])).abs().max().item(), 1e-14)

    def test_written_out_formulas_4d(self):
        """The D = 4 half spinors of the module docstring, literally."""
        gammas, _ = gamma_matrices(4)
        one = torch.eye(4, dtype=C)
        p = torch.randn(4, dtype=C, generator=torch.Generator().manual_seed(13))
        i = 1j
        for s in (-1.0, 1.0):
            formulas = [
                ((p[0] - s * p[2], p[1] - s * p[3]), lambda h: (-s * h[0], -s * h[1])),
                ((p[0] - i * s * p[3], p[1] - i * s * p[2]), lambda h: (i * s * h[1], i * s * h[0])),
                ((p[0] - s * p[3], p[1] + s * p[2]), lambda h: (s * h[1], -s * h[0])),
                ((p[0] - i * s * p[2], p[1] + i * s * p[3]), lambda h: (i * s * h[0], -i * s * h[1])),
            ]
            for mu, (h, lower) in enumerate(formulas):
                full = torch.stack(list(h) + list(lower(h)))
                self.assertLess((full - (one + s * gammas[mu]) @ p).abs().max().item(), 1e-14, (mu, s))

    def test_operator_projection(self):
        """HalfLatticeWilson.project slice k equals R_k^dagger psi site by site."""
        for geo in (G4s, G2):
            op = HalfLatticeWilson(links(geo, 1, 14), 0.1, geo)
            x = to_half(field(geo, 1, 15), geo)
            proj = op.project(x)
            gammas, _ = gamma_matrices(geo.D)
            Hh = geo.spin // 2
            for k in range(2 * geo.D):
                mu, s = k % geo.D, (-1.0 if k < geo.D else 1.0)
                Rd = torch.cat((torch.eye(Hh, dtype=C), s * gammas[mu][:Hh, Hh:]), dim=1)  # R^dagger
                self.assertLess((proj[:, :, k] - torch.matmul(Rd, x)).abs().max().item(), 1e-14)


class LeanCGTest(unittest.TestCase):
    def test_matches_solver_cg(self):
        geo = G44
        U = links(geo, 3, 16)
        k = kappas(0.05, 0.10, 0.13)
        eo = WilsonDirac(U, k, geo).even_odd()
        op = HalfLatticeWilson(U, k, geo)
        b = to_half(field(geo, 3, 17), geo)
        ref = cg(eo.normal, to_full(b, geo), tol=1e-12)
        its = ref.iterations.tolist()
        self.assertTrue(its[0] < its[1] < its[2], its)
        for check_every in (1, 3, 10):
            with self.subTest(check_every=check_every):
                info = {}
                res = lean_cg(op.normal, b, tol=1e-12, max_iter=500, check_every=check_every, info=info)
                self.assertEqual(res.iterations.tolist(), its)
                self.assertLess(rel(res.x, to_half(ref.x, geo)), 1e-10)
                self.assertTrue(bool(torch.isfinite(torch.view_as_real(res.x)).all()))
                self.assertLess(res.residual.max().item(), 1e-12)
                self.assertGreaterEqual(info["loop_iterations"], max(its))
                self.assertLess(info["loop_iterations"], max(its) + check_every)
                self.assertEqual(info["loop_iterations"], -(-max(its) // check_every) * check_every)
                # true residual recomputed independently
                r = torch.sqrt(_norm2(b - op.normal(res.x)) / _norm2(b))
                self.assertLess((r - res.residual).abs().max().item(), 1e-15)

    def test_zero_rhs_and_warm_start(self):
        geo = G4s
        U = links(geo, 3, 18)
        op = HalfLatticeWilson(U, kappas(0.1, 0.12, 0.13), geo)
        b = to_half(field(geo, 3, 19), geo)
        b[1] = 0.0
        info = {}
        res = lean_cg(op.normal, b, tol=1e-10, check_every=4, info=info)
        self.assertEqual(res.iterations[1].item(), 0)
        self.assertEqual(res.x[1].abs().max().item(), 0.0)
        self.assertTrue(bool(torch.isfinite(torch.view_as_real(res.x)).all()))
        self.assertLess(res.residual.max().item(), 1e-10)
        info = {}
        zero = lean_cg(op.normal, torch.zeros_like(b), tol=1e-10, info=info)
        self.assertEqual(info["loop_iterations"], 0)
        self.assertEqual(zero.x.abs().max().item(), 0.0)
        warm = lean_cg(op.normal, b, x0=res.x, tol=1e-6)
        self.assertEqual(warm.iterations.tolist(), [0, 0, 0])

    def test_conjugate_view_rhs(self):
        geo = G2
        op = HalfLatticeWilson(links(geo, 2, 33), 0.2, geo)
        b = to_half(field(geo, 2, 34), geo)
        a = lean_cg(op.normal, b.conj(), tol=1e-12)
        c = lean_cg(op.normal, b.conj().resolve_conj(), tol=1e-12)
        self.assertTrue(torch.equal(a.x, c.x))

    def test_max_iter_honoured(self):
        """max_iter caps the loop; non-convergence is reported (item 6c)."""
        geo = G44
        op = HalfLatticeWilson(links(geo, 2, 63), kappas(0.12, 0.13), geo)
        b = to_half(field(geo, 2, 64), geo)
        for check_every in (1, 3, 10):
            info = {}
            res = lean_cg(op.normal, b, tol=1e-12, max_iter=5, check_every=check_every, info=info)
            self.assertEqual(info["loop_iterations"], 5)
            self.assertEqual(res.iterations.tolist(), [5, 5])
            self.assertFalse(bool(info["converged"].any()))
            self.assertGreater(res.residual.min().item(), 1e-6)
            self.assertEqual(info["restarts"], 0)

    def test_returned_residual_is_true_residual(self):
        """On an ill-conditioned system the recursive residual drops below tol while the true
        one stays ~1e-8: the returned residual is the true one (item 6d)."""
        A, _, b = clustered_system([1.0, 1e4, 1e8])
        ref = cg(A, b, tol=1e-10, max_iter=200)  # recursive convergence in ~7 iterations
        self.assertLess(ref.iterations.max().item(), 20)
        for max_restarts in (0, None):
            with self.subTest(max_restarts=max_restarts):
                info = {}
                res = lean_cg(A, b, tol=1e-10, max_iter=200, check_every=1, info=info, max_restarts=max_restarts)
                true = torch.sqrt(_norm2(b - A(res.x)) / _norm2(b))
                self.assertLess((res.residual - true).abs().max().item(), 1e-20)
                self.assertGreater(res.residual.min().item(), 1e-9)  # far above tol: not the recursive one
                self.assertFalse(bool(info["converged"].any()))

    def test_restart_when_true_residual_above_target(self):
        """Item 5. A sample flagged converged on its recursive residual but with a true residual
        above 10 tol is restarted from r = b - A x instead of returned. Forced with an operator
        that is perturbed (A + delta) for the first loop applications only: CG converges for
        A + delta, the exit check with the exact A finds the true residual ~delta, and the
        restart (exact A from then on) converges."""
        geo = G4s
        op = HalfLatticeWilson(links(geo, 2, 65), kappas(0.12, 0.13), geo)
        b = to_half(field(geo, 2, 66), geo)
        delta = 1e-6

        class Perturbed:
            def __init__(self, n_perturbed):
                self.calls, self.n = 0, n_perturbed

            def __call__(self, x):
                y = op.normal(x)
                if self.calls < self.n:
                    y = y + delta * x
                self.calls += 1
                return y

        first = {}
        stage1 = lean_cg(Perturbed(10 ** 9), b, tol=1e-12, max_iter=500, check_every=1, info=first)
        n1 = first["loop_iterations"]
        true1 = torch.sqrt(_norm2(b - op.normal(stage1.x)) / _norm2(b))
        self.assertGreater(true1.min().item(), 1e-9)  # the forced case: converged for A + delta only
        info = {}
        apply = Perturbed(n1)
        res = lean_cg(apply, b, tol=1e-12, max_iter=500, check_every=1, info=info)
        self.assertEqual(info["restarts"], 1)
        self.assertGreater(info["loop_iterations"], n1)
        self.assertTrue(bool(info["converged"].all()))
        self.assertLess(res.residual.max().item(), 1e-11)
        true = torch.sqrt(_norm2(b - op.normal(res.x)) / _norm2(b))
        self.assertLess((true - res.residual).abs().max().item(), 1e-20)
        # without restarts the same forced case returns the residual ~delta, reported as such
        info0 = {}
        res0 = lean_cg(Perturbed(n1), b, tol=1e-12, max_iter=500, check_every=1, info=info0, max_restarts=0)
        self.assertEqual(info0["loop_iterations"], n1)
        self.assertGreater(res0.residual.min().item(), 1e-9)
        self.assertFalse(bool(info0["converged"].any()))

    def test_restarts_stop_at_true_residual_floor(self):
        """Below the attainable residual the restarts stop (floor rule) well before max_iter."""
        A, _, b = clustered_system([1.0, 1e4, 1e8])
        info = {}
        res = lean_cg(A, b, tol=1e-12, max_iter=2000, check_every=1, info=info)
        self.assertGreaterEqual(info["restarts"], 1)
        self.assertTrue(info["stagnated"])
        self.assertLess(info["loop_iterations"], 100)
        self.assertFalse(bool(info["converged"].any()))
        self.assertGreater(res.residual.min().item(), 1e-10)

    def test_tol_zero_stops_at_roundoff(self):
        geo = G2
        op = HalfLatticeWilson(links(geo, 2, 20), 0.2, geo)
        b = to_half(field(geo, 2, 21), geo)
        info = {}
        res = lean_cg(op.normal, b, tol=0.0, max_iter=400, check_every=5, info=info)
        self.assertLess(info["loop_iterations"], 400)
        self.assertLess(res.residual.max().item(), 1e-13)


class MixedPrecisionTest(unittest.TestCase):
    def test_reaches_double_tolerance(self):
        geo = G44
        U = links(geo, 3, 22)
        k = kappas(0.10, 0.12, 0.13)
        op = HalfLatticeWilson(U, k, geo)
        op64 = op.to(C64)
        b = to_half(field(geo, 3, 23), geo)
        double = lean_cg(op.normal, b, tol=1e-12, max_iter=500)
        for tol in (1e-10, 1e-12):
            with self.subTest(tol=tol):
                info = {}
                res = mixed_precision_cg(op, op64, b, tol=tol, inner_tol=1e-5, info=info)
                self.assertEqual(res.x.dtype, C)
                r = torch.sqrt(_norm2(b - op.normal(res.x)) / _norm2(b))
                self.assertLess(r.max().item(), tol)
                self.assertLess((r - res.residual).abs().max().item(), 1e-16)
                self.assertGreaterEqual(info["outer"], 2)
                self.assertEqual(len(info["inner_iterations"]), info["outer"])
                self.assertLess(rel(res.x, double.x), 50 * tol)
        # a converged warm start costs nothing; a zero right-hand side sample stays zero
        info = {}
        warm = mixed_precision_cg(op, op64, b, x0=double.x, tol=1e-10, info=info)
        self.assertEqual(info["outer"], 0)
        b[0] = 0.0
        res = mixed_precision_cg(op, op64, b, tol=1e-10)
        self.assertEqual(res.x[0].abs().max().item(), 0.0)
        self.assertLess(res.residual.max().item(), 1e-10)


    def test_stagnation_exit(self):
        """Tolerance below the attainable double residual: the outer loop stops on stagnation
        (true residual not halved in an outer step), long before max_outer (item 3)."""
        A, A64, b = clustered_system([1.0, 1e2, 1e4])
        info = {}
        res = mixed_precision_cg(A, A64, b, tol=1e-15, max_outer=30, max_iter=5000, info=info)
        self.assertLess(info["outer"], 10)
        self.assertTrue(bool(info["stagnated"].all()))
        self.assertFalse(bool(info["converged"].any()))
        self.assertFalse(info["budget_exhausted"])
        self.assertEqual(info["double_applications"], info["outer"])
        self.assertLess(res.residual.max().item(), 1e-11)  # it got to the floor (~1e-12)
        true = torch.sqrt(_norm2(b - A(res.x)) / _norm2(b))
        self.assertLess((true - res.residual).abs().max().item(), 1e-20)
        # an attainable tolerance converges, not stagnated
        info = {}
        res = mixed_precision_cg(A, A64, b, tol=1e-10, info=info)
        self.assertTrue(bool(info["converged"].all()))
        self.assertFalse(bool(info["stagnated"].any()))
        self.assertLess(res.residual.max().item(), 1e-10)

    def test_worsening_step_is_undone(self):
        """An outer step that increases the true residual is undone per sample. Sample 1's complex64
        operator is 10x too small, so its correction overshoots: r -> r - 10 r = -9 r. Its x stays
        bitwise the start x0, its residual the initial one, and it is marked stagnated; sample 0
        (exact complex64 operator) converges as on its own (to round-off: dense BLAS products
        of another batch size)."""
        A, A64, b = clustered_system([1.0, 3.0], n=48, batch=2, seed=7)
        w = torch.tensor([[1.0], [0.1]], dtype=torch.float32).to(C64)
        A64_bad = lambda x: A64(x) * w  # noqa: E731
        x0 = 0.1 * b.flip(-1)
        r0 = torch.sqrt(_norm2(b - A(x0)) / _norm2(b))
        info = {}
        res = mixed_precision_cg(A, A64_bad, b, x0=x0, tol=1e-12, inner_tol=1e-6, info=info)
        self.assertTrue(torch.equal(res.x[1], x0[1]))
        self.assertEqual(res.residual[1].item(), r0[1].item())
        self.assertEqual(info["stagnated"].tolist(), [False, True])
        self.assertEqual(info["converged"].tolist(), [True, False])
        alone = mixed_precision_cg(A, A64, b[:1], x0=x0[:1], tol=1e-12, inner_tol=1e-6)
        self.assertLess(rel(res.x[:1], alone.x), 1e-12)
        self.assertLess(res.residual[0].item(), 1e-12)

    def test_non_finite_sample_is_isolated(self):
        """A sample with a NaN (or inf) right-hand side does not hold up the batch, in
        mixed_precision_cg as in lean_cg: the healthy samples end bitwise as in a batch without it
        (same x, residuals, per-sample iterations and loop cost, inner budget not spent), the bad
        sample ends with a non-finite residual, not converged; the fast term's convergence error
        names only that sample (double and mixed)."""
        geo = G44
        op = HalfLatticeWilson(links(geo, 1, 101), 0.13, geo)  # links of batch 1: fields of any batch
        op64 = op.to(C64)
        b = to_half(field(geo, 3, 102), geo)
        healthy = b[[0, 2]].clone()
        solvers = {
            "mixed": lambda rhs, info: mixed_precision_cg(op, op64, rhs, tol=1e-12, inner_tol=1e-6, max_iter=2000,
                                                          info=info),
            "double": lambda rhs, info: lean_cg(op.normal, rhs, tol=1e-12, max_iter=2000, info=info),
        }
        for value in (float("nan"), float("inf")):
            bad = b.clone()
            bad[1, 3, 1, 2] = value
            for name, solve in solvers.items():
                with self.subTest(value=value, solver=name):
                    want_info, info = {}, {}
                    want, got = solve(healthy, want_info), solve(bad, info)
                    self.assertTrue(torch.equal(got.x[[0, 2]], want.x))
                    self.assertTrue(torch.equal(got.residual[[0, 2]], want.residual))
                    self.assertEqual(got.iterations[[0, 2]].tolist(), want.iterations.tolist())
                    self.assertEqual(info["loop_iterations"], want_info["loop_iterations"])
                    self.assertLess(got.residual[[0, 2]].max().item(), 1e-12)
                    self.assertFalse(bool(torch.isfinite(got.residual[1])))
                    self.assertEqual(info["converged"].tolist(), [True, False, True])
                    if name == "mixed":
                        self.assertFalse(info["budget_exhausted"])
                        self.assertEqual(info["outer"], want_info["outer"])
        U = links(geo, 3, 103)
        for precision in ("double", "mixed"):
            with self.subTest(term=precision):
                term = FastTwoFlavourWilson(0.13, geo, precision=precision, solver_tol=1e-12)
                term.refresh(U, torch.Generator().manual_seed(104))
                term._phi_h[1, 3, 1, 2] = float("nan")
                with self.assertRaisesRegex(RuntimeError, r"action: CG not converged, relative residual inf .* "
                                                          r"for sample\(s\) \[1\],"):
                    term.action(U)
                self.assertLess(term.stats["last_residual"][[0, 2]].max().item(), 1e-11)

    def test_max_iter_is_total_inner_budget(self):
        geo = G44
        op = HalfLatticeWilson(links(geo, 2, 67), kappas(0.12, 0.13), geo)
        b = to_half(field(geo, 2, 68), geo)
        free = {}
        mixed_precision_cg(op, op.to(C64), b, tol=1e-12, inner_tol=1e-3, max_iter=10000, info=free)
        self.assertGreaterEqual(free["outer"], 3)
        budget = free["loop_iterations"] // 2
        info = {}
        res = mixed_precision_cg(op, op.to(C64), b, tol=1e-12, inner_tol=1e-3, max_iter=budget, info=info)
        self.assertLessEqual(info["loop_iterations"], budget)
        self.assertTrue(info["budget_exhausted"])
        self.assertFalse(bool(info["converged"].all()))
        self.assertGreater(res.residual.max().item(), 1e-12)


class FastTermTest(unittest.TestCase):
    def _pair(self, geo, k, precision="double", **kw):
        ref = TwoFlavourWilson(k, geo, **kw)
        fast = FastTwoFlavourWilson(k, geo, precision=precision, **kw)
        return ref, fast

    def test_refresh_action_force(self):
        geo = G46
        k = kappas(0.11, 0.13)
        U = links(geo, 2, 24)
        U2 = links(geo, 2, 25)
        for precision, tol, force_tol in (("double", 1e-12, 1e-9), ("mixed", 1e-12, 1e-9), ("mixed", 1e-10, 1e-7)):
            with self.subTest(precision=precision, tol=tol):
                ref, fast = self._pair(geo, k, precision, solver_tol=tol)
                ref.refresh(U, torch.Generator().manual_seed(26))
                fast.refresh(U, torch.Generator().manual_seed(26))
                self.assertLess(rel(fast.phi, ref.phi), 1e-14)
                self.assertLess(rel(fast._refresh_action, ref._refresh_action), 1e-14)
                S_ref, S_fast = ref.action(U2), fast.action(U2)
                self.assertLess(((S_fast - S_ref).abs() / S_ref.abs()).max().item(), 1e-10)
                # at the refresh links S = eta^dagger eta
                self.assertLess(((fast.action(U) - fast._refresh_action).abs()
                                 / fast._refresh_action).max().item(), 1e-10)
                F_ref, F_fast = ref.force(U2), fast.force(U2)
                self.assertEqual(F_fast.shape, U2.shape)
                self.assertLess(rel(F_fast, F_ref), force_tol)
                self.assertEqual(fast.stats["solves"], 3)
                self.assertGreater(fast.stats["cg_loop_iterations"], 0)
                if precision == "mixed":
                    self.assertGreater(fast.stats["outer_iterations"], 0)

    def test_periodic_time_sign_matches_reference(self):
        """time_sign = +1: refresh, action and force against the reference (item 6a)."""
        geo = G46
        k = kappas(0.11, 0.13)
        U, U2 = links(geo, 2, 70), links(geo, 2, 71)
        ref, fast = self._pair(geo, k, time_sign=1.0, solver_tol=1e-12)
        anti = FastTwoFlavourWilson(k, geo, time_sign=-1.0, solver_tol=1e-12)
        for term in (ref, fast, anti):
            term.refresh(U, torch.Generator().manual_seed(72))
        self.assertLess(rel(fast.phi, ref.phi), 1e-14)
        S_ref, S_fast = ref.action(U2), fast.action(U2)
        self.assertLess(((S_fast - S_ref).abs() / S_ref.abs()).max().item(), 1e-10)
        self.assertGreater(((anti.action(U2) - S_ref).abs() / S_ref.abs()).max().item(), 1e-4)
        self.assertLess(rel(fast.force(U2), ref.force(U2)), 1e-9)

    def test_action_solved_at_action_tol(self):
        """Loose solver_tol, tight action_tol: the action solve reaches action_tol, the force
        solve only solver_tol (item 6b)."""
        geo = G46
        k = kappas(0.12, 0.13)
        U, U2 = links(geo, 2, 73), links(geo, 2, 74)
        for precision in ("double", "mixed"):
            with self.subTest(precision=precision):
                ref, fast = self._pair(geo, k, precision, solver_tol=1e-4, action_tol=1e-12, reuse_guess=False)
                for term in (ref, fast):
                    term.refresh(U, torch.Generator().manual_seed(75))
                S_ref, S_fast = ref.action(U2), fast.action(U2)
                self.assertLess(fast.stats["last_residual"].max().item(), 1e-11)
                self.assertLess(((S_fast - S_ref).abs() / S_ref.abs()).max().item(), 1e-10)
                fast.force(U2)
                self.assertGreater(fast.stats["last_residual"].max().item(), 1e-9)
                self.assertLess(fast.stats["last_residual"].max().item(), 1e-3)

    def test_refresh_resets_warm_start(self):
        """After refresh the next solve starts from zero: bitwise the result of a fresh term (6f)."""
        geo = G4s
        U, U2 = links(geo, 2, 76), links(geo, 2, 77)
        used = FastTwoFlavourWilson(0.12, geo, solver_tol=1e-8, action_tol=1e-8)
        used.refresh(U2, torch.Generator().manual_seed(78))
        used.action(U)  # leaves a guess X for an unrelated phi
        self.assertIsNotNone(used._X_h)
        used.refresh(U, torch.Generator().manual_seed(79))
        self.assertIsNone(used._X_h)
        fresh = FastTwoFlavourWilson(0.12, geo, solver_tol=1e-8, action_tol=1e-8)
        fresh.refresh(U, torch.Generator().manual_seed(79))
        self.assertTrue(torch.equal(used.action(U2), fresh.action(U2)))
        self.assertTrue(torch.equal(used.force(U2), fresh.force(U2)))

    def test_operator_cache_follows_parameters(self):
        """Changing kappa (float, tensor, in place), time_sign or colour_product after an operator
        was cached for the same links takes effect (item 1)."""
        geo = G4s
        U, U2 = links(geo, 2, 80), links(geo, 2, 81)
        fast = FastTwoFlavourWilson(0.12, geo, solver_tol=1e-12)
        fast.refresh(U, torch.Generator().manual_seed(82))
        phi = fast.phi
        fast.action(U2)
        op_before = fast.operator(U2)
        self.assertIs(fast.operator(U2), op_before)

        def reference(kappa, time_sign=-1.0):
            ref = TwoFlavourWilson(kappa, geo, time_sign=time_sign, solver_tol=1e-12)
            ref.phi = phi
            return ref.action(U2)

        def close(a, b):
            return ((a - b).abs() / b.abs()).max().item() < 1e-10

        fast.kappa = 0.13
        self.assertTrue(close(fast.action(U2), reference(0.13)))
        fast.kappa = kappas(0.12, 0.14)
        self.assertTrue(close(fast.action(U2), reference(kappas(0.12, 0.14))))
        fast.kappa[1] = 0.11  # in place: compared by value
        self.assertTrue(close(fast.action(U2), reference(kappas(0.12, 0.11))))
        fast.time_sign = 1.0
        self.assertTrue(close(fast.action(U2), reference(kappas(0.12, 0.11), 1.0)))
        fast.colour_product = "broadcast"
        self.assertEqual(fast.operator(U2).colour_product, "broadcast")
        op = fast.operator(U2)
        self.assertIs(fast.operator(U2), op)  # unchanged parameters: still cached

    def test_convergence_policy(self):
        """on_unconverged: raise (default) names kind, residual and samples; warn once per kind;
        ignore counts (item 4, fast term, double and mixed)."""
        geo = G4s
        U, U2 = links(geo, 2, 83), links(geo, 2, 84)
        with self.assertRaises(ValueError):
            FastTwoFlavourWilson(0.12, geo, on_unconverged="maybe")
        for precision in ("double", "mixed"):
            with self.subTest(precision=precision):
                term = FastTwoFlavourWilson(0.12, geo, precision=precision, max_iter=2, max_outer=2)
                self.assertEqual(term.on_unconverged, "raise")
                term.refresh(U, torch.Generator().manual_seed(85))
                with self.assertRaisesRegex(RuntimeError, r"action: CG not converged, relative residual .* sample\(s\) \[0, 1\]"):
                    term.action(U2)
                with self.assertRaisesRegex(RuntimeError, "force: CG not converged"):
                    term.force(U2)
                self.assertEqual(term.stats["unconverged_solves"], 2)
                warn = FastTwoFlavourWilson(0.12, geo, precision=precision, max_iter=2, max_outer=2,
                                            on_unconverged="warn")
                warn.refresh(U, torch.Generator().manual_seed(85))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    warn.action(U2)
                    warn.action(U2)
                    warn.force(U2)
                kinds = [str(w.message).split(":")[0] for w in caught if issubclass(w.category, UnconvergedSolveWarning)]
                self.assertEqual(kinds, ["FastTwoFlavourWilson action", "FastTwoFlavourWilson force"])
                self.assertEqual(warn.stats["unconverged_solves"], 3)
                quiet = FastTwoFlavourWilson(0.12, geo, precision=precision, max_iter=2, max_outer=2,
                                             on_unconverged="ignore")
                quiet.refresh(U, torch.Generator().manual_seed(85))
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    quiet.action(U2)
                self.assertEqual(quiet.stats["unconverged_solves"], 1)
                # converged solves pass silently under the default
                ok = FastTwoFlavourWilson(0.12, geo, precision=precision, solver_tol=1e-12)
                ok.refresh(U, torch.Generator().manual_seed(85))
                ok.action(U2)
                ok.force(U2)
                self.assertEqual(ok.stats["unconverged_solves"], 0)

    def test_cache_and_full_lattice_views(self):
        geo = G4s
        U = links(geo, 2, 27)
        U_new = links(geo, 2, 28)
        ref, fast = self._pair(geo, 0.12, solver_tol=1e-12)
        for term in (ref, fast):
            term.refresh(U, torch.Generator().manual_seed(29))
        d_ref, d_fast = ref.action_difference(U, U_new), fast.action_difference(U, U_new)
        self.assertLess((d_ref - d_fast).abs().max().item(), 1e-9)
        self.assertEqual(fast.stats["solves"], 1)  # S(U) from the eta^dagger eta cache
        self.assertIs(fast.operator(U_new), fast.operator(U_new))  # cached per links tensor
        op_before = fast.operator(U)
        U.mul_(1.0)  # in-place change: _version bumps, the cache must not be used
        self.assertIsNot(fast.operator(U), op_before)
        fast.action_difference(U, U_new)
        ref.action_difference(U, U_new)
        self.assertEqual(fast.stats["solves"], 3)
        self.assertEqual(ref.stats["solves"], 3)
        # phi and X read and write full-lattice tensors
        self.assertEqual(tuple(fast.phi.shape), geo.wilson_shape(2))
        self.assertEqual(to_half(fast.phi, geo, 1).abs().max().item(), 0.0)
        fresh = FastTwoFlavourWilson(0.12, geo, solver_tol=1e-12)
        fresh.phi = ref.phi
        self.assertLess(((fresh.action(U_new) - ref.action(U_new)).abs()
                         / ref.action(U_new).abs()).max().item(), 1e-10)
        self.assertLess(rel(fresh.X, ref.X), 1e-9)
        with self.assertRaises(ValueError):
            FastTwoFlavourWilson(0.12, geo, even_odd=False)
        with self.assertRaises(ValueError):
            FastTwoFlavourWilson(0.12, geo, precision="half")

    def test_hmc_trajectory_matches_reference_and_is_reversible(self):
        geo = G2
        kw = dict(n_gauge_substeps=2, solver_tol=1e-12, action_tol=1e-13)
        U0 = links(geo, 3, 30)
        out = {}
        for name, cls in (("ref", TwoFlavourHamiltonian), ("fast", FastTwoFlavourHamiltonian)):
            ham = cls(2.0, 0.2, geo, **kw)
            hmc = H.HMC(geo, ham, tau=0.5, n_steps=4)
            gen = torch.Generator().manual_seed(31)
            U = U0.clone()
            dHs = []
            for _ in range(2):
                U, acc, dH, _ = hmc.trajectory(U, gen)
                dHs.append(dH)
            out[name] = (U, torch.stack(dHs))
        self.assertLess((out["fast"][1] - out["ref"][1]).abs().max().item(), 1e-8)
        self.assertLess((out["fast"][0] - out["ref"][0]).abs().max().item(), 1e-10)
        # reversibility of the fast integrator (CG from zero)
        gen = torch.Generator().manual_seed(32)
        ham = FastTwoFlavourHamiltonian(2.0, 0.2, geo, n_gauge_substeps=2, solver_tol=1e-12, reuse_guess=False)
        ham.refresh(U0, gen)
        P = H.random_momenta(U0.shape, generator=gen)
        hmc = H.HMC(geo, ham, tau=0.5, n_steps=4)
        U1, P1, _ = hmc.integrate(U0, P)
        U2, P2, _ = hmc.integrate(U1, -P1)
        self.assertGreater((U1 - U0).abs().max().item(), 0.1)
        self.assertLess((U2 - U0).abs().max().item(), 1e-10)
        self.assertLess((P2 + P).abs().max().item(), 1e-10)

    def test_reuse_guess_flag_honoured(self):
        """At a loose solver tolerance (1e-4) the warm start breaks reversibility at O(tol)
        (measured ~5e-6) while the cold start is reversible to rounding (~3e-15) (item 6e)."""
        geo = G2
        U0 = links(geo, 3, 30)
        err = {}
        for reuse in (False, True):
            gen = torch.Generator().manual_seed(32)
            ham = FastTwoFlavourHamiltonian(2.0, 0.2, geo, n_gauge_substeps=2, solver_tol=1e-4, reuse_guess=reuse)
            self.assertEqual(ham.fermion.reuse_guess, reuse)
            ham.refresh(U0, gen)
            P = H.random_momenta(U0.shape, generator=gen)
            hmc = H.HMC(geo, ham, tau=0.5, n_steps=4)
            U1, P1, _ = hmc.integrate(U0, P)
            U2, P2, _ = hmc.integrate(U1, -P1)
            err[reuse] = max((U2 - U0).abs().max().item(), (P2 + P).abs().max().item())
        self.assertLess(err[False], 1e-12)
        self.assertGreater(err[True], 1e-8)

    def test_hmc_with_rejections_matches_reference(self):
        """Large step size (acceptance 0.3-0.8): same accept/reject pattern as the reference and
        the same links, rejected replicas restored bitwise, in 2D and on 2^3 x 4 (item 6h)."""
        for geo, beta, kappa, B in ((G2, 2.0, 0.2, 8), (G4s, 5.0, 0.12, 4)):
            with self.subTest(lattice=geo.lattice):
                U0 = links(geo, B, 50)
                out = {}
                for name, cls in (("ref", TwoFlavourHamiltonian), ("fast", FastTwoFlavourHamiltonian)):
                    ham = cls(beta, kappa, geo, n_gauge_substeps=1, solver_tol=1e-12, action_tol=1e-13)
                    hmc = H.HMC(geo, ham, tau=0.8, n_steps=1)
                    gen = torch.Generator().manual_seed(51)
                    U = U0.clone()
                    accs, dHs = [], []
                    for _ in range(3):
                        U_prev = U.clone()
                        U, acc, dH, _ = hmc.trajectory(U, gen)
                        for b in (~acc).nonzero().flatten().tolist():
                            self.assertTrue(torch.equal(U[b], U_prev[b]))
                        accs.append(acc)
                        dHs.append(dH)
                    out[name] = (U, torch.stack(accs), torch.stack(dHs))
                rate = out["ref"][1].double().mean().item()
                self.assertGreaterEqual(rate, 0.3)
                self.assertLessEqual(rate, 0.8)
                self.assertTrue(torch.equal(out["fast"][1], out["ref"][1]))
                self.assertLess((out["fast"][2] - out["ref"][2]).abs().max().item(), 1e-9)
                self.assertLess((out["fast"][0] - out["ref"][0]).abs().max().item(), 1e-10)


class BenchRowsTest(unittest.TestCase):
    def test_bench_fast_rows(self):
        spec = importlib.util.spec_from_file_location("bench_fermions", ROOT / "main" / "bench_fermions.py")
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        with tempfile.TemporaryDirectory() as tmp:
            res = bench.main(["--device", "cpu", "--lattices", "2^3x4", "--batches", "2", "--dtypes", "complex128",
                              "--target-time", "0", "--trials", "1", "--warmup", "1", "--min-repeats", "1",
                              "--max-repeats", "1", "--cg-iters", "3", "--cg-min-iters", "1", "--no-compile",
                              "--no-profile", "--breakdown-config", "", "--quiet", "--out", str(Path(tmp) / "b")])
        rows = {(r["measurement"], r["variant"]): r for r in res["rows"]}
        modes = ("matmul", "broadcast", "broadcast_sum")
        for key in tuple(("eo_normal", f"half_lattice_{m}{c}") for m in modes for c in ("", "_c64")) + (
                    ("cg_iteration", "lean_cg_half"),
                    ("cg_iteration", "lean_cg_reference"), ("solve", "solver_cg_reference"),
                    ("solve", "lean_cg_double"), ("solve", "mixed_precision")):
            self.assertIn(key, rows)
            self.assertEqual(rows[key]["status"], "ok", key)
        for m in modes:
            self.assertEqual(rows[("eo_normal", f"half_lattice_{m}_c64")]["out_dtype"], "complex64")
            self.assertLess(float(rows[("eo_normal", f"half_lattice_{m}")]["max_rel_error"]), 1e-12)
        for key in (("cg_iteration", "lean_cg_half"), ("solve", "lean_cg_double"), ("solve", "mixed_precision")):
            self.assertIn("colour_product=matmul", rows[key]["notes"])
        fast = res["meta_dict"]["fast_path"]
        self.assertLess(fast["2^3x4:2:complex128:half_lattice_broadcast_c64"], 1e-5)
        self.assertLessEqual(fast["2^3x4:2:complex128:mixed_precision"]["residual"], 1e-10)


class MainHMCFastTest(unittest.TestCase):
    def test_fast_flag_reproduces_reference_run(self):
        spec = importlib.util.spec_from_file_location("main_hmc", ROOT / "main" / "main_hmc.py")
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)
        from neoqcd.fermions.io import read_observables

        common = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "2.0", "--batch", "2", "--tau", "0.5",
                  "--n-steps", "3", "--n-gauge-substeps", "2", "--device", "cpu", "--seed", "7", "--start", "hot",
                  "--quiet", "--fermions", "wilson", "--kappa", "0.12", "--n-traj", "3", "--solver-tol", "1e-12"]
        with tempfile.TemporaryDirectory() as tmp:
            cols = {}
            runs = (("ref", []), ("fast", ["--fast"]), ("mixed", ["--fast", "--precision", "mixed"]),
                    ("broadcast", ["--fast", "--colour-product", "broadcast"]))
            for name, extra in runs:
                out = Path(tmp) / name
                main.main(common + extra + ["--out", str(out)])
                cols[name] = read_observables(out / "observables.csv")
                meta = json.loads((out / "run_meta.json").read_text())
                self.assertEqual(meta["parameters"]["fast"], name != "ref")
                self.assertEqual(meta["parameters"]["precision"], "mixed" if name == "mixed" else "double")
                self.assertEqual(meta["run"]["fermion_operator"], "reference" if name == "ref" else "half_lattice_fast")
                self.assertEqual(meta["run"]["colour_product"],
                                 None if name == "ref" else ("broadcast" if name == "broadcast" else "matmul"))
            for name in ("fast", "mixed", "broadcast"):
                np.testing.assert_allclose(cols[name]["dH"], cols["ref"]["dH"], rtol=0, atol=1e-8)
                np.testing.assert_allclose(cols[name]["plaquette"], cols["ref"]["plaquette"], rtol=0, atol=1e-10)
                np.testing.assert_array_equal(cols[name]["accepted"], cols["ref"]["accepted"])
        with self.assertRaises(SystemExit):
            main.parse_args(common + ["--precision", "mixed"])
        # --fast with staggered fermions is accepted (fast_terms.FastStaggeredPseudofermion)
        args = main.parse_args(common[:-8] + ["--fermions", "staggered", "--mass", "0.3", "--fast"])
        self.assertTrue(args.fast)
        self.assertEqual(args.fermions, "staggered")

    def test_fast_flag_periodic_time(self):
        """--time-sign 1: the --fast run reproduces the reference run (item 6a)."""
        spec = importlib.util.spec_from_file_location("main_hmc", ROOT / "main" / "main_hmc.py")
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)
        from neoqcd.fermions.io import read_observables

        common = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "2.0", "--batch", "2", "--tau", "0.5",
                  "--n-steps", "3", "--n-gauge-substeps", "2", "--device", "cpu", "--seed", "8", "--start", "hot",
                  "--quiet", "--fermions", "wilson", "--kappa", "0.2", "--n-traj", "3", "--solver-tol", "1e-12",
                  "--time-sign", "1"]
        with tempfile.TemporaryDirectory() as tmp:
            cols = {}
            for name, extra in (("ref", []), ("fast", ["--fast"]), ("anti", ["--fast", "--time-sign", "-1"])):
                out = Path(tmp) / name
                main.main(common + extra + ["--out", str(out)])
                cols[name] = read_observables(out / "observables.csv")
                meta = json.loads((out / "run_meta.json").read_text())
                self.assertEqual(meta["parameters"]["time_sign"], -1.0 if name == "anti" else 1.0)
            np.testing.assert_allclose(cols["fast"]["dH"], cols["ref"]["dH"], rtol=0, atol=1e-8)
            np.testing.assert_allclose(cols["fast"]["plaquette"], cols["ref"]["plaquette"], rtol=0, atol=1e-10)
            np.testing.assert_array_equal(cols["fast"]["accepted"], cols["ref"]["accepted"])
            self.assertGreater(np.abs(cols["anti"]["dH"] - cols["ref"]["dH"]).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
