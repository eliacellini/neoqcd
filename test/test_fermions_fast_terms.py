"""Tests of neoqcd.fermions.fast_terms: the half-lattice staggered operator and pseudofermion term,
and the fast Hasenbusch terms, each against its reference class (the ground truth).

CPU, tiny lattices, fixed seeds. Lattices: 2^3 x 4, 4^4, 4^3 x 6 SU(3) and 2D 4 x 4 SU(3).
Tolerances: operator 1e-13 (complex128) / 1e-5 (complex64) relative, action 1e-10, force 1e-9
relative, HMC dH 1e-8. Timings behind NEOQCD_TEST_VERBOSE.
"""

import os
import time
import unittest
import warnings

import torch

from neoqcd.fermions import hmc as H
from neoqcd.fermions._gauge import SUN_haar, SUN_mul
from neoqcd.fermions.algebra import expi, random_algebra
from neoqcd.fermions.fast import COLOUR_PRODUCTS, to_full, to_half
from neoqcd.fermions.fast_terms import (
    FastHasenbuschHamiltonian,
    FastHeavyWilsonTerm,
    FastRatioWilsonTerm,
    FastStaggeredHamiltonian,
    FastStaggeredPseudofermion,
    HalfLatticeStaggered,
    to_full_staggered,
    to_half_staggered,
)
from neoqcd.fermions.hasenbusch import HasenbuschHamiltonian, HeavyWilsonTerm, RatioWilsonTerm
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.pseudofermion import UnconvergedSolveWarning
from neoqcd.fermions.staggered import StaggeredDirac
from neoqcd.fermions.staggered_pseudofermion import (
    StaggeredHamiltonian,
    StaggeredPseudofermion,
    staggered_hopping_derivative,
)

G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4
G44 = Geometry(4, 4, 4, 3)  # 4^4
G46 = Geometry(4, 6, 4, 3)  # 4^3 x 6
G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4
C = torch.complex128
MODES = tuple(m for m in COLOUR_PRODUCTS if m != "auto")
VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))


def haar_links(geometry, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(geometry.link_shape(batch)[:-1], generator=gen), gen


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def masses(B):
    return torch.linspace(0.3, 0.6, B, dtype=torch.float64)


# ---- stage 1: staggered ------------------------------------------------------------------------
class HalfLatticeStaggeredTest(unittest.TestCase):
    def test_operator_against_reference(self):
        for geo in (G4, G44, G2, G46):
            U, gen = haar_links(geo, 2, 11)
            for ts in (-1.0, 1.0):
                ref = StaggeredDirac(U, masses(2), geo, ts).even_odd()
                chi = torch.randn(geo.staggered_shape(2), generator=gen, dtype=C)
                want = to_half_staggered(ref.apply(chi), geo)
                rhs_ref = to_half_staggered(ref.prepare_rhs(chi), geo)
                hop_ref = to_half_staggered(ref.staggered.hopping(ref.even(chi)), geo, 1)
                for mode in MODES:
                    with self.subTest(geo=geo.lattice, time_sign=ts, mode=mode):
                        op = HalfLatticeStaggered(U, masses(2), geo, ts, colour_product=mode)
                        x = to_half_staggered(chi, geo)
                        self.assertLess(rel(op.apply(x), want), 1e-13)
                        self.assertLess(rel(op.prepare_rhs(chi), rhs_ref), 1e-13)
                        self.assertLess(rel(op.hop(x, 1), hop_ref), 1e-13)
                        full = op.reconstruct(x, chi)
                        self.assertLess(rel(full, ref.reconstruct(chi, chi)), 1e-13)
                        op32 = op.to(torch.complex64)
                        y = op32.apply(x.to(torch.complex64))
                        self.assertEqual(y.dtype, torch.complex64)
                        self.assertLess(rel(y.to(C), want), 1e-5)

    def test_dtype_conversions_broadcast(self):
        U, gen = haar_links(G2, 1, 12)
        op = HalfLatticeStaggered(U, 0.4, G2)
        x = torch.randn((3, G2.volume // 2, 3), generator=gen, dtype=C)
        with self.assertRaises(TypeError):
            op.apply(x.to(torch.complex64))
        with self.assertRaises(TypeError):
            HalfLatticeStaggered(U.real, 0.4, G2)
        ref = StaggeredDirac(U, 0.4, G2).even_odd()
        want = to_half_staggered(ref.apply(to_full_staggered(x, G2)), G2)
        self.assertLess(rel(op.apply(x), want), 1e-13)  # links B = 1, fields B' = 3
        chi = torch.randn(G2.staggered_shape(2), generator=gen, dtype=C)
        e, o = to_half_staggered(chi, G2, 0), to_half_staggered(chi, G2, 1)
        self.assertTrue(torch.equal(to_full_staggered(e, G2, 0, other=o), chi))
        up = op.to(torch.complex64).to(C)  # the upcast re-gathers from the complex128 links
        self.assertTrue(torch.equal(up._links[0], op._links[0]))

    def test_hopping_derivative_against_reference(self):
        for geo in (G4, G2):
            U, gen = haar_links(geo, 2, 13)
            for ts in (-1.0, 1.0):
                ref = StaggeredDirac(U, masses(2), geo, ts)
                a = torch.randn(geo.staggered_shape(2), generator=gen, dtype=C)
                b = torch.randn(geo.staggered_shape(2), generator=gen, dtype=C)
                want = staggered_hopping_derivative(ref, a, b)
                for mode in MODES:
                    with self.subTest(geo=geo.lattice, time_sign=ts, mode=mode):
                        op = HalfLatticeStaggered(U, masses(2), geo, ts, colour_product=mode)
                        h = lambda f, p: to_half_staggered(f, geo, p)
                        got = op.hopping_derivative(h(a, 0), h(a, 1), h(b, 0), h(b, 1))
                        self.assertLess(rel(got, want), 1e-13)
                        # the pseudofermion pattern: a even only, b odd only (None halves)
                        ae, bo = ref.even_odd().even(a), ref.even_odd().odd(b)
                        got = op.hopping_derivative(h(a, 0), None, None, h(b, 1))
                        self.assertLess(rel(got, staggered_hopping_derivative(ref, ae, bo)), 1e-13)


class FastStaggeredPseudofermionTest(unittest.TestCase):
    def _pair(self, geo, ts=-1.0, **kw):
        kw.setdefault("solver_tol", 1e-12)
        return (StaggeredPseudofermion(masses(2), geo, time_sign=ts, **kw),
                FastStaggeredPseudofermion(masses(2), geo, time_sign=ts, **kw))

    def test_refresh_action_force(self):
        for geo in (G4, G2, G44):
            U, _ = haar_links(geo, 2, 21)
            for ts in (-1.0, 1.0):
                with self.subTest(geo=geo.lattice, time_sign=ts):
                    ref, fast = self._pair(geo, ts)
                    ref.refresh(U, torch.Generator().manual_seed(5))
                    fast.refresh(U, torch.Generator().manual_seed(5))
                    self.assertLess((ref.phi - fast.phi).abs().max().item(), 1e-14)
                    self.assertLess(rel(fast.action(U), ref.action(U)), 1e-10)
                    self.assertLess(rel(fast.force(U), ref.force(U)), 1e-9)
                    U2 = SUN_mul(expi(0.2 * random_algebra(U.shape, generator=torch.Generator().manual_seed(1))), U)
                    self.assertLess(rel(fast.action_difference(U, U2), ref.action_difference(U, U2)), 1e-9)
                    self.assertEqual(ref.stats["solves"], fast.stats["solves"])

    def test_mixed_precision(self):
        U, _ = haar_links(G2, 2, 22)
        ref, _ = self._pair(G2)
        fast = FastStaggeredPseudofermion(masses(2), G2, solver_tol=1e-12, precision="mixed")
        ref.refresh(U, torch.Generator().manual_seed(5))
        fast.refresh(U, torch.Generator().manual_seed(5))
        self.assertLess(rel(fast.action(U), ref.action(U)), 1e-10)
        self.assertLess(rel(fast.force(U), ref.force(U)), 1e-9)
        self.assertGreater(fast.stats["outer_iterations"], 0)

    def test_operator_cache(self):
        U, gen = haar_links(G2, 2, 23)
        fast = FastStaggeredPseudofermion(masses(2), G2, solver_tol=1e-12)
        fast.refresh(U, gen)
        op = fast.operator(U)
        self.assertIs(fast.operator(U), op)
        S0 = fast.action(U)
        U.mul_(1.0)  # in-place change: _version bumps
        self.assertIsNot(fast.operator(U), op)
        with torch.no_grad():
            U.copy_(SUN_mul(expi(0.3 * random_algebra(U.shape, generator=gen)), U.clone()))
        ref = StaggeredPseudofermion(masses(2), G2, solver_tol=1e-12)
        ref.phi = fast.phi
        self.assertLess(rel(fast.action(U), ref.action(U)), 1e-10)
        self.assertGreater((fast.action(U) - S0).abs().max().item(), 1e-3)
        op = fast.operator(U)
        fast.mass = masses(2) + 0.1  # mass by value
        self.assertIsNot(fast.operator(U), op)
        ref.mass = masses(2) + 0.1
        self.assertLess(rel(fast.action(U), ref.action(U)), 1e-10)
        op = fast.operator(U)
        fast.time_sign = 1.0
        self.assertIsNot(fast.operator(U), op)
        # refresh-action cache: an in-place change of U_old forces a new solve
        fast.refresh(U, gen)
        U2 = U.clone()
        d1 = fast.action_difference(U, U2)  # caches S at U
        n = fast.stats["solves"]
        fast.action_difference(U, U2)
        self.assertEqual(fast.stats["solves"] - n, 1)
        U.mul_(1.0)
        fast.action_difference(U, U2)
        self.assertEqual(fast.stats["solves"] - n, 3)
        self.assertLess(d1.abs().max().item(), 1e-9)

    def test_unconverged_policy(self):
        U, gen = haar_links(G2, 2, 24)
        fast = FastStaggeredPseudofermion(0.05, G2, max_iter=3)
        fast.refresh(U, gen)
        with self.assertRaisesRegex(RuntimeError, "action: CG not converged"):
            fast.action(U)
        fast.on_unconverged = "warn"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fast.force(U)
        self.assertTrue(any(issubclass(x.category, UnconvergedSolveWarning) for x in w))
        self.assertEqual(fast.stats["unconverged_solves"], 2)
        with self.assertRaises(ValueError):
            FastStaggeredPseudofermion(0.3, G2, on_unconverged="maybe")

    def test_hmc_matches_reference(self):
        """Same seeds, same trajectories: dH and links agree to 1e-8 including rejections."""
        for tau, n_steps in ((1.0, 6), (1.0, 2)):  # the second: large step, about half rejected
            U0, _ = haar_links(G2, 4, 25)
            kw = dict(n_gauge_substeps=2, solver_tol=1e-13, action_tol=1e-13, reuse_guess=False)
            out = {}
            for name, cls in (("ref", StaggeredHamiltonian), ("fast", FastStaggeredHamiltonian)):
                ham = cls(2.0, masses(4), G2, **kw)
                hmc = H.HMC(G2, ham, tau=tau, n_steps=n_steps)
                gen = torch.Generator().manual_seed(77)
                U, dHs, acc = U0.clone(), [], []
                for _ in range(4):
                    U, a, dH, _ = hmc.trajectory(U, gen)
                    dHs.append(dH)
                    acc.append(a)
                out[name] = (U, torch.stack(dHs), torch.stack(acc))
            with self.subTest(tau=tau, n_steps=n_steps):
                dd = (out["ref"][1] - out["fast"][1]).abs() / out["ref"][1].abs().clamp(min=1.0)
                self.assertLess(dd.max().item(), 1e-8)
                self.assertTrue(torch.equal(out["ref"][2], out["fast"][2]))
                self.assertLess((out["ref"][0] - out["fast"][0]).abs().max().item(), 1e-8)
                self.assertFalse(bool(out["ref"][2].all()), "expected rejections")
                if n_steps == 2:
                    self.assertTrue(bool(out["ref"][2].any()), "expected acceptances")
                if VERBOSE:
                    print(f"\nstaggered HMC tau {tau} n_steps {n_steps}: dH {out['fast'][1].tolist()}, "
                          f"acc {out['fast'][2].float().mean().item():.2f}")

    def test_timings(self):
        if not VERBOSE:
            self.skipTest("timings only with NEOQCD_TEST_VERBOSE")
        for geo in (G44, Geometry(4, 8, 8, 3)):
            U, gen = haar_links(geo, 4, 26)
            ref = StaggeredDirac(U, 0.1, geo).even_odd()
            op = HalfLatticeStaggered(U, 0.1, geo)
            chi = torch.randn(geo.staggered_shape(4), generator=gen, dtype=C)
            x = to_half_staggered(chi, geo)
            for name, f, v in (("reference", ref.apply, chi), ("fast", op.apply, x)):
                f(v)
                t0 = time.perf_counter()
                for _ in range(10):
                    f(v)
                print(f"\n{geo.lattice} B=4 staggered M_ee {name}: {(time.perf_counter() - t0) / 10 * 1e3:.2f} ms")


# ---- stage 2: Hasenbusch ------------------------------------------------------------------------
class FastHasenbuschTest(unittest.TestCase):
    def test_ratio_and_heavy_against_reference(self):
        for geo in (G4, G2):
            U, _ = haar_links(geo, 2, 31)
            kappa = torch.tensor([0.12, 0.13], dtype=torch.float64) if geo.D == 4 else 0.20
            kh = 0.09 if geo.D == 4 else torch.tensor([0.15, 0.12], dtype=torch.float64)
            U2 = SUN_mul(expi(0.2 * random_algebra(U.shape, generator=torch.Generator().manual_seed(2))), U)
            for ts in (-1.0, 1.0):
                for mode in MODES:
                    with self.subTest(geo=geo.lattice, time_sign=ts, mode=mode):
                        kw = dict(time_sign=ts, solver_tol=1e-12)
                        pairs = ((RatioWilsonTerm(kappa, kh, geo, **kw),
                                  FastRatioWilsonTerm(kappa, kh, geo, colour_product=mode, **kw)),
                                 (HeavyWilsonTerm(kh, geo, **kw), FastHeavyWilsonTerm(kh, geo, colour_product=mode, **kw)))
                        for ref, fast in pairs:
                            ref.refresh(U, torch.Generator().manual_seed(5))
                            fast.refresh(U, torch.Generator().manual_seed(5))
                            self.assertLess(rel(fast.phi, ref.phi), 1e-11)
                            self.assertLess(rel(fast._refresh_action, ref._refresh_action), 1e-14)
                            self.assertLess(rel(fast.action(U2), ref.action(U2)), 1e-10)
                            self.assertLess(rel(fast.force(U2), ref.force(U2)), 1e-9)
                            self.assertLess(rel(fast.action_difference(U, U2), ref.action_difference(U, U2)), 1e-8)
                        self.assertEqual(pairs[0][1].stats["refresh_solves"], 1)

    def test_heavy_operator_shares_links(self):
        U, gen = haar_links(G2, 2, 32)
        ham = FastHasenbuschHamiltonian(2.0, 0.2, 0.15, G2)
        light = ham.ratio.operator(U)
        heavy = ham.ratio.heavy_operator(U)
        self.assertIs(heavy._links[0], light._links[0])
        self.assertIs(ham.heavy.operator(U), heavy)
        self.assertAlmostEqual(heavy.kappa.item(), 0.15)
        self.assertAlmostEqual(light.kappa.item(), 0.2)
        ham.ratio.kappa_heavy = 0.1  # by value: a new heavy operator; the heavy term then builds its own
        self.assertAlmostEqual(ham.ratio.heavy_operator(U).kappa.item(), 0.1)
        self.assertIsNot(ham.heavy.operator(U), ham.ratio.heavy_operator(U))
        self.assertAlmostEqual(ham.heavy.operator(U).kappa.item(), 0.15)

    def test_refresh_guard(self):
        U, gen = haar_links(G2, 2, 33)
        S = FastRatioWilsonTerm(0.20, 0.15, G2, max_iter=3)
        with self.assertRaisesRegex(RuntimeError, "not converged, relative residual"):
            S.refresh(U, gen)
        S = FastRatioWilsonTerm(0.20, 0.15, G2, max_iter=3, on_unconverged="ignore")
        with self.assertRaisesRegex(RuntimeError, "FastRatioWilsonTerm.refresh: heavy CG not converged"):
            S.refresh(U, gen)
        FastRatioWilsonTerm(0.20, 0.15, G2).refresh(U, gen)

    def test_argument_validation(self):
        args = (5.0, 0.12, 0.09, G4)
        for kwargs, pattern in (
                (dict(time_sign=-1.0, heavy_kwargs={"time_sign": 1.0}), "time_sign"),
                (dict(heavy_kwargs={"geometry": G2}), "geometry"),
                (dict(layout="shared", n_heavy_substeps=2), "n_heavy_substeps"),
                (dict(layout=((("ratio",), 1), (("heavy", "gauge"), 2)), n_gauge_substeps=2), "explicit levels"),
                (dict(layout=(("ratio", 1), (("heavy", "gauge"), 2))), "bare string")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, pattern):
                FastHasenbuschHamiltonian(*args, **kwargs)
        ham = FastHasenbuschHamiltonian(*args, time_sign=1.0, precision="mixed",
                                        heavy_kwargs={"time_sign": 1.0, "solver_tol": 1e-8, "precision": "double"},
                                        layout="shared", n_ratio_substeps=2, n_gauge_substeps=3)
        self.assertEqual(ham.heavy.solver_tol, 1e-8)
        self.assertEqual((ham.ratio.precision, ham.heavy.precision), ("mixed", "double"))
        self.assertEqual([s for _, s in ham.layout], [2, 3])
        self.assertEqual([[type(t).__name__ for t in terms] for terms, _ in ham.levels],
                         [["FastRatioWilsonTerm", "FastHeavyWilsonTerm"], ["WilsonGaugeAction"]])

    def test_hmc_matches_reference(self):
        for layout, tau, n_steps in (("nested", 1.0, 4), ("shared", 1.0, 1)):
            U0, _ = haar_links(G2, 4, 34)
            nh = {"n_heavy_substeps": 2} if layout == "nested" else {}
            kw = dict(n_gauge_substeps=2, layout=layout, solver_tol=1e-13, action_tol=1e-13, reuse_guess=False, **nh)
            out = {}
            for name, cls in (("ref", HasenbuschHamiltonian), ("fast", FastHasenbuschHamiltonian)):
                ham = cls(2.0, 0.22, 0.15, G2, **kw)
                hmc = H.HMC(G2, ham, tau=tau, n_steps=n_steps)
                gen = torch.Generator().manual_seed(78)
                U, dHs, acc = U0.clone(), [], []
                for _ in range(4):
                    U, a, dH, _ = hmc.trajectory(U, gen)
                    dHs.append(dH)
                    acc.append(a)
                out[name] = (U, torch.stack(dHs), torch.stack(acc))
            with self.subTest(layout=layout):
                dd = (out["ref"][1] - out["fast"][1]).abs() / out["ref"][1].abs().clamp(min=1.0)
                self.assertLess(dd.max().item(), 1e-8)
                self.assertTrue(torch.equal(out["ref"][2], out["fast"][2]))
                self.assertLess((out["ref"][0] - out["fast"][0]).abs().max().item(), 1e-8)
                if VERBOSE:
                    print(f"\nHasenbusch HMC {layout}: max dH diff {dd.max().item():.1e}, "
                          f"acc {out['fast'][2].float().mean().item():.2f}")

    def test_timings(self):
        if not VERBOSE:
            self.skipTest("timings only with NEOQCD_TEST_VERBOSE")
        for geo in (G44, Geometry(4, 8, 8, 3)):
            U0, _ = haar_links(geo, 4, 35)
            for name, cls in (("reference", HasenbuschHamiltonian), ("fast", FastHasenbuschHamiltonian)):
                ham = cls(5.6, 0.125, 0.10, geo, n_heavy_substeps=2, n_gauge_substeps=2)
                hmc = H.HMC(geo, ham, tau=0.5, n_steps=2)
                gen = torch.Generator().manual_seed(1)
                t0 = time.perf_counter()
                hmc.trajectory(U0.clone(), gen)
                print(f"\n{geo.lattice} B=4 Hasenbusch trajectory {name}: {time.perf_counter() - t0:.2f} s, "
                      f"CG it ratio {ham.ratio.stats['cg_iterations']} heavy {ham.heavy.stats['cg_iterations']}")
        for geo in (G44, Geometry(4, 8, 8, 3)):
            U, gen = haar_links(geo, 4, 36)
            for name, cls in (("reference", StaggeredPseudofermion), ("fast", FastStaggeredPseudofermion)):
                t = cls(0.1, geo, solver_tol=1e-10)
                t.refresh(U, gen)
                t.action(U)
                n0 = t.stats["cg_iterations"]
                t0 = time.perf_counter()
                t.X = None
                t.action(U)
                dt = time.perf_counter() - t0
                print(f"\n{geo.lattice} B=4 staggered CG {name}: {dt / max(1, t.stats['cg_iterations'] - n0) * 1e3:.2f} "
                      f"ms per iteration ({t.stats['cg_iterations'] - n0} iterations)")


if __name__ == "__main__":
    unittest.main()
