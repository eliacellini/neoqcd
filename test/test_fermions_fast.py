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
from neoqcd.fermions.pseudofermion import TwoFlavourHamiltonian, TwoFlavourWilson, hopping_derivative
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
        self.assertEqual(resolve_colour_product("auto", "cuda"), "broadcast")
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
        with self.assertRaises(SystemExit):
            main.parse_args(common[:-8] + ["--fermions", "staggered", "--mass", "0.3", "--fast"])


if __name__ == "__main__":
    unittest.main()
