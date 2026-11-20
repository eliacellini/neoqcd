"""`snf_layers.PseudofermionAction` beyond test_fermions_snf: the exact dispatch on the fermion term
type (review of 20 November), the fast staggered term, first derivatives only (a second
derivative raises), and the solver tolerance of the forward solve (action_tol, not solver_tol).

CPU, complex128; 2D 4 x 4 SU(3), rough random links (so that a loose tolerance leaves a visible
error), B = 3.
"""

import unittest

import torch

from neoqcd.fermions.action import (ParametricFastTwoFlavourWilson, ParametricStaggeredPseudofermion,
                                    ParametricTwoFlavourWilson, QCDAction)
from neoqcd.fermions.algebra import expi, random_algebra
from neoqcd.fermions.fast import FastTwoFlavourWilson
from neoqcd.fermions.fast_terms import FastHeavyWilsonTerm, FastRatioWilsonTerm, FastStaggeredPseudofermion
from neoqcd.fermions.hasenbusch import HeavyWilsonTerm, RatioWilsonTerm
from neoqcd.fermions.hmc import WilsonGaugeAction
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.pseudofermion import TwoFlavourWilson
from neoqcd.fermions.staggered_pseudofermion import StaggeredPseudofermion
from neoqcd.fermions._gauge import FlowPars, create_mask
from neoqcd.fermions.snf_layers import PseudofermionAction, _fermion_kind, make_gauge_layers, pseudofermion_action

G2 = Geometry(2, 4, 4, 3)
B = 3
KAPPA, MASS = 0.2, 0.1


def rough_links(seed, batch=B, scale=1.0):
    return expi(scale * random_algebra(G2.link_shape(batch), torch.Generator().manual_seed(seed)))


def wilson_terms(**kw):
    """One instance of every supported Wilson term type (fast ones with a check every iteration,
    so that a loose tolerance really stops the CG early)."""
    return {"TwoFlavourWilson": TwoFlavourWilson(KAPPA, G2, **kw),
            "TwoFlavourWilson(full)": TwoFlavourWilson(KAPPA, G2, even_odd=False, **kw),
            "ParametricTwoFlavourWilson": ParametricTwoFlavourWilson(KAPPA, G2, **kw),
            "FastTwoFlavourWilson": FastTwoFlavourWilson(KAPPA, G2, check_every=1, **kw),
            "ParametricFastTwoFlavourWilson": ParametricFastTwoFlavourWilson(KAPPA, G2, check_every=1, **kw)}


def staggered_terms(**kw):
    return {"StaggeredPseudofermion": StaggeredPseudofermion(MASS, G2, **kw),
            "ParametricStaggeredPseudofermion": ParametricStaggeredPseudofermion(MASS, G2, **kw),
            "FastStaggeredPseudofermion": FastStaggeredPseudofermion(MASS, G2, check_every=1, **kw)}


def refreshed(terms, U, seed):
    """Refresh the first term and give every other term of the family the same phi."""
    first = next(iter(terms.values()))
    first.refresh(U, torch.Generator().manual_seed(seed))
    for name, t in terms.items():
        if t is not first:
            if name == "TwoFlavourWilson(full)":
                t.refresh(U, torch.Generator().manual_seed(seed))
            else:
                t.phi = first.phi.clone()
    return terms


class DispatchTest(unittest.TestCase):
    """Every term type is either handled (by exact type) or refused with a clear TypeError: the
    Hasenbusch heavy/ratio terms (reference AND fast: FastRatioWilsonTerm derives from
    FastTwoFlavourWilson and used to be taken for it, returning Re phi^dagger X instead of
    chi^dagger X), clover, DD-HMC blocks, the fused-kernel term, a user subclass, the rooted
    staggered term."""

    SUPPORTED = {TwoFlavourWilson: "wilson", ParametricTwoFlavourWilson: "wilson",
                 FastTwoFlavourWilson: "wilson_fast", ParametricFastTwoFlavourWilson: "wilson_fast",
                 StaggeredPseudofermion: "staggered", ParametricStaggeredPseudofermion: "staggered",
                 FastStaggeredPseudofermion: "staggered_fast"}

    def test_supported(self):
        for cls, kind in self.SUPPORTED.items():
            with self.subTest(cls=cls.__name__):
                self.assertEqual(_fermion_kind(object.__new__(cls)), kind)
        # the terms QCDAction builds
        self.assertEqual(_fermion_kind(QCDAction(G2, "kappa", beta=5.0, kappa=KAPPA).fermion), "wilson")
        self.assertEqual(_fermion_kind(QCDAction(G2, "kappa", beta=5.0, kappa=KAPPA, fast=True).fermion), "wilson_fast")
        self.assertEqual(_fermion_kind(QCDAction(G2, "mass", beta=5.0, fermions="staggered", mass=MASS).fermion),
                         "staggered")

    def test_rejected(self):
        from neoqcd.fermions.clover import TwoFlavourClover  # here: modules outside the SNF path
        from neoqcd.fermions.ddhmc import BlockTwoFlavour
        from neoqcd.fermions.fused import FusedTwoFlavourWilson
        from neoqcd.fermions.rhmc import RootedStaggeredPseudofermion

        class UserWilson(TwoFlavourWilson):
            pass

        class UserFastStaggered(FastStaggeredPseudofermion):
            pass

        class Other:
            pass
        hasenbusch = (HeavyWilsonTerm, RatioWilsonTerm, FastHeavyWilsonTerm, FastRatioWilsonTerm)
        others = (TwoFlavourClover, BlockTwoFlavour, FusedTwoFlavourWilson, UserWilson, UserFastStaggered,
                  RootedStaggeredPseudofermion, Other)
        for cls in hasenbusch + others:
            with self.subTest(cls=cls.__name__):
                pattern = "Hasenbusch" if cls in hasenbusch else "unsupported fermion term"
                with self.assertRaisesRegex(TypeError, pattern):
                    _fermion_kind(object.__new__(cls))

    def test_fast_ratio_term_refused_by_the_function(self):
        """The reviewer's case: a real FastRatioWilsonTerm with phi set must not give an S_pf."""
        U = rough_links(1, scale=0.3)
        ratio = FastRatioWilsonTerm(KAPPA, 0.12, G2)
        ratio.refresh(U, torch.Generator().manual_seed(2))
        with self.assertRaisesRegex(TypeError, "Hasenbusch"):
            pseudofermion_action(U, ratio)
        with self.assertRaisesRegex(TypeError, "Hasenbusch"):
            PseudofermionAction.apply(U.clone().requires_grad_(), ratio)


class FastStaggeredTest(unittest.TestCase):
    """The fast staggered term gives the reference staggered S_pf and gradient (same phi), also
    through a stout layer, with a (B,) mass."""

    def test_value_and_gradient(self):
        U = rough_links(3, scale=0.5)
        mass = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        ref = StaggeredPseudofermion(mass.clone(), G2, solver_tol=1e-12, action_tol=1e-13, reuse_guess=False)
        fast = FastStaggeredPseudofermion(mass.clone(), G2, solver_tol=1e-12, action_tol=1e-13, reuse_guess=False)
        ref.refresh(U, torch.Generator().manual_seed(4))
        fast.phi = ref.phi.clone()
        Ug = U.clone().requires_grad_()
        S_r, S_f = pseudofermion_action(Ug, ref), pseudofermion_action(Ug, fast)
        w = torch.tensor([0.7, -1.3, 2.0], dtype=torch.float64)  # non-unit grad_output weights
        g_r = torch.autograd.grad((w * S_r).sum(), Ug)[0]
        g_f = torch.autograd.grad((w * S_f).sum(), Ug)[0]
        scale = S_r.abs().max().item()
        self.assertLess((S_f - S_r).abs().max().item(), 1e-10 * scale)
        self.assertLess((S_f - ref.action(U)).abs().max().item(), 1e-10 * scale)
        self.assertGreater(g_r.abs().max().item(), 1e-2)
        self.assertLess((g_f - g_r).abs().max().item(), 1e-9 * g_r.abs().max().item())
        # through a stout layer
        fp = FlowPars(D=G2.D, T=G2.T, L=G2.L, N=G2.N, mask=create_mask(G2.D, G2.T, G2.L), protocol=[0.0],
                      batch_size=B, device="cpu")
        layer = make_gauge_layers(fp, "smearing_signed", rho=0.0)[0]
        with torch.no_grad():
            layer.rho_layer.copy_(torch.tensor([0.05, -0.08, 0.03, -0.06], dtype=torch.float64).view(fp.rho_shape))
        grads = []
        for term in (ref, fast):
            U1, _, _ = layer(U, fp, None, beta=0.0, delta_beta=0.0)
            grads.append(torch.autograd.grad((w * pseudofermion_action(U1, term)).sum(), layer.rho_layer)[0])
        self.assertGreater(grads[0].abs().max().item(), 1e-3)
        self.assertLess((grads[1] - grads[0]).abs().max().item(), 1e-9 * grads[0].abs().max().item())


class SecondDerivativeTest(unittest.TestCase):
    """The Function is differentiable once: create_graph=True raises (before 20 November the
    second derivative came out as ~1e-15 against a finite difference of ~450, silently), also when
    the gradient flows through another term as well; first derivatives are unaffected."""

    def test_create_graph_raises(self):
        U = rough_links(5, scale=0.5)
        gauge = WilsonGaugeAction(5.0, G2)
        terms = dict(wilson_terms(solver_tol=1e-12, action_tol=1e-13), **staggered_terms(solver_tol=1e-12,
                                                                                         action_tol=1e-13))
        for name, term in terms.items():
            with self.subTest(term=name):
                term.refresh(U, torch.Generator().manual_seed(6))
                Ug = U.clone().requires_grad_()
                g = torch.autograd.grad(pseudofermion_action(Ug, term).sum(), Ug)[0]  # first order: fine
                self.assertGreater(g.abs().max().item(), 1e-3)
                with self.assertRaisesRegex(RuntimeError, "differentiable once"):
                    torch.autograd.grad(pseudofermion_action(Ug, term).sum(), Ug, create_graph=True)
                with self.assertRaisesRegex(RuntimeError, "differentiable once"):
                    torch.autograd.grad((gauge.action(Ug) + pseudofermion_action(Ug, term)).sum(), Ug,
                                        create_graph=True)


class ActionToleranceTest(unittest.TestCase):
    """The forward solve runs at action_tol (the accuracy of the SNF work), not at solver_tol: with
    solver_tol = 1e-2 and action_tol = 1e-13 the value equals a tightly solved action to 1e-10
    relative, and every solve of the Function is requested at action_tol (fast terms checked
    every iteration, so a loose tolerance would really stop their CG early)."""

    def test_tolerance(self):
        U = rough_links(7)
        loose = dict(solver_tol=1e-2, action_tol=1e-13, reuse_guess=False)
        tight = dict(solver_tol=1e-13, action_tol=1e-13, reuse_guess=False)
        for family in (wilson_terms, staggered_terms):
            terms = refreshed(family(**loose), U, 8)
            refs = refreshed(family(**tight), U, 8)
            for name, term in terms.items():
                with self.subTest(term=name):
                    refs[name].phi = term.phi.clone()
                    S_ref = refs[name].action(U)
                    tols = []
                    solve = term._solve

                    def recording(op, tol, solve=solve, tols=tols):
                        tols.append(tol)
                        return solve(op, tol)
                    term._solve = recording
                    S = pseudofermion_action(U.clone().requires_grad_(), term)
                    del term._solve
                    self.assertEqual(tols, [term.action_tol])
                    err = (S - S_ref).abs().max().item() / S_ref.abs().max().item()
                    self.assertLess(err, 1e-10, f"{name}: relative error {err:.1e}")
                    # the loose tolerance would be visible: a solve at solver_tol is far off
                    X_loose = term._solve(term.operator(U.detach() if "Fast" in name else U), term.solver_tol)
                    phi = term._phi_h if "Fast" in name else term.phi
                    S_loose = (phi.conj() * X_loose).real.reshape(B, -1).sum(1)
                    self.assertGreater((S_loose - S_ref).abs().max().item() / S_ref.abs().max().item(), 1e-8, name)


if __name__ == "__main__":
    unittest.main()
