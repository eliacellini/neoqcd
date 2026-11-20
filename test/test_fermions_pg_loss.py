"""Tests of the joint PG loss (neoqcd.fermions.pg.fermion_path_gradient_loss).

- per chain, the fermionic loss is d/d eps xi_f(exp(i eps F) U, phi) (finite differences), and its
  gradient in the field parameters equals autograd of Re <grad xi_f, i F_theta U> with grad xi_f
  from dense double autograd at frozen parameters (the analogue of test_cnf's frozen-residual test);
- kappa = 0: the loss and the parameter gradients are bitwise those of the pure-gauge
  `cnf.path_gradient_loss`; kappa > 0: the gauge part is bitwise the pure-gauge loss on the same U;
- E_phi[xi_f | U] = -F . grad log det M / V against the dense determinant (statistical, 2D 4 x 4);
- d_c S: the ladder action equals QCDAction's gauge action and its c-derivative the exact
  five-point `rex.defect_action`, while a jvp through QCDAction is silently zero (the pitfall);
- the defect-mask agreement check.
CPU, complex128, 2D 4 x 4 SU(3) (and 2^3 x 4 for the action equality).
"""

import os
import unittest

import torch

from neoqcd.fermions._gauge import SUN_haar
from neoqcd.fermions._gauge_cnf import CloverChairLinkDrift, apply_velocity, lie_tangent, path_gradient_loss
from neoqcd.fermions.action import QCDAction
from neoqcd.fermions.algebra import random_algebra
from neoqcd.fermions.exact import assemble_even_odd
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.pg import (
    check_same_defect,
    defect_ladder_action,
    fermion_path_gradient_loss,
    frozen_parameters,
    hermitian_tangent,
    pseudofermion_pg_term,
)
from neoqcd.fermions.pseudofermion import TwoFlavourWilson
from neoqcd.fermions.rex import defect_action
from test.test_fermions_pg import dense_xi_and_gradient

G2 = Geometry(2, 4, 4, 3)
G4 = Geometry(4, 4, 2, 3)
VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))


def line_defect(geometry, size=2, time_slice=1, space_slice=1):
    m = torch.zeros((geometry.D,) + geometry.lattice, dtype=torch.float64)
    m[(0, time_slice, slice(space_slice, space_slice + size)) + (space_slice,) * (geometry.D - 2)] = 1.0
    return torch.stack((1.0 - m, m))


def small_field(seed=7, scale=0.01, D=2):
    torch.manual_seed(seed)
    field = CloverChairLinkDrift(D, hidden_dim=4, depth=1, loop_max=1, zero_init=False).to(dtype=torch.float64)
    with torch.no_grad():
        field.output_layer.weight.mul_(scale)
    return field


def setup(kappa=0.2, seed=3, batch=2, time_sign=-1.0):
    gen = torch.Generator().manual_seed(seed)
    U = SUN_haar(G2.link_shape(batch)[:-1], generator=gen)
    term = TwoFlavourWilson(kappa, G2, time_sign=time_sign, solver_tol=1e-13)
    term.refresh(U, gen)
    return U, term, gen


class JointLossTest(unittest.TestCase):
    def setUp(self):
        self.mask = line_defect(G2)
        self.action = defect_ladder_action(G2, 5.0, self.mask)
        self.t = torch.tensor([0.3, 0.7], dtype=torch.float64)

    def test_fermion_loss_is_the_flow_derivative_and_its_parameter_gradient(self):
        for time_sign in (-1.0, 1.0):
            with self.subTest(time_sign=time_sign):
                U, term, _ = setup(time_sign=time_sign)
                field = small_field()
                out = fermion_path_gradient_loss(field, self.action, term, U, self.t)
                self.assertEqual(out["loss"].shape, (2,))
                self.assertEqual(out["solves"]["solves"], 2)
                torch.testing.assert_close(out["loss"].detach(), out["loss_gauge"] + out["loss_fermion"])

                # loss_f = d/d eps xi_f(exp(i eps F) U) with F = drift(U) held fixed along the path
                coeff = field.drift(U, self.t).detach()

                def xi_f(eps):
                    moved = apply_velocity(U, coeff, eps)
                    with frozen_parameters(field), torch.no_grad():
                        tangent = lambda value: lie_tangent(value, field.drift(value, self.t))
                        return pseudofermion_pg_term(moved, term, tangent, tol=1e-14)["xi"]
                h = 1e-3
                fd = (-xi_f(2 * h) + 8 * xi_f(h) - 8 * xi_f(-h) + xi_f(-2 * h)) / (12 * h)
                self.assertGreater(float(out["loss_fermion"].abs().max()), 1e-6)
                torch.testing.assert_close(out["loss_fermion"], fd, rtol=1e-6, atol=1e-12)

                # parameter gradient: autograd of Re <grad xi_f (dense, frozen field), i F_theta U>
                with frozen_parameters(field):
                    tangent = lambda value: lie_tangent(value, field.drift(value, self.t))
                    xi_dense, g_dense = dense_xi_and_gradient(U, term.phi, term.kappa, G2, time_sign, tangent)
                torch.testing.assert_close(out["xi_fermion"], xi_dense, rtol=1e-9, atol=1e-13)
                expected_f = (g_dense.conj() * lie_tangent(U, field.drift(U, self.t))).real.flatten(1).sum(1)
                torch.testing.assert_close(out["loss_fermion"], expected_f.detach(), rtol=1e-8, atol=1e-13)
                gauge = path_gradient_loss(field, self.action, U, self.t, volume=G2.volume)["loss"]
                params = list(field.parameters())
                actual = torch.autograd.grad(out["loss"].mean(), params, allow_unused=True)
                expected = torch.autograd.grad((gauge + expected_f).mean(), params, allow_unused=True)
                nonzero = 0
                for a, e in zip(actual, expected):
                    if a is None or e is None:
                        self.assertIsNone(a)
                        self.assertIsNone(e)
                        continue
                    torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-12)
                    nonzero += int(bool((a != 0).any()))
                self.assertGreater(nonzero, 0)

    def test_kappa_zero_is_bitwise_the_pure_gauge_loss(self):
        U, term, _ = setup(kappa=0.0)
        field = small_field()
        out = fermion_path_gradient_loss(field, self.action, term, U, self.t)
        params = list(field.parameters())
        grads = torch.autograd.grad(out["loss"].mean(), params, allow_unused=True)
        ref = path_gradient_loss(field, self.action, U, self.t, volume=G2.volume)
        ref_grads = torch.autograd.grad(ref["loss"].mean(), params, allow_unused=True)
        self.assertTrue(torch.equal(out["loss"].detach(), ref["loss"].detach()))
        self.assertTrue(torch.equal(out["xi"], ref["xi"]))
        self.assertEqual(float(out["xi_fermion"].abs().max()), 0.0)
        for a, b in zip(grads, ref_grads):
            self.assertEqual(a is None, b is None)
            if a is not None:
                self.assertTrue(torch.equal(a, b))

    def test_gauge_part_is_the_pure_gauge_loss(self):
        U, term, _ = setup(kappa=0.2)
        field = small_field(scale=0.05)
        out = fermion_path_gradient_loss(field, self.action, term, U, self.t, chunk_size=1)
        ref = path_gradient_loss(field, self.action, U, self.t, volume=G2.volume, chunk_size=1)
        self.assertTrue(torch.equal(out["loss_gauge"], ref["loss"].detach()))
        self.assertTrue(torch.equal(out["xi_gauge"], ref["xi"]))
        self.assertGreater(float(out["xi_fermion"].abs().max()), 0.0)
        # chunking the chains changes nothing but the memory
        whole = fermion_path_gradient_loss(field, self.action, term, U, self.t)
        torch.testing.assert_close(out["loss"], whole["loss"], rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(out["xi_fermion"], whole["xi_fermion"], rtol=1e-12, atol=1e-15)

    def test_expectation_over_phi_is_the_log_det_derivative(self):
        # E_phi[X^dagger (d_F M) X] = tr(M^-1 d_F M): xi_f averages to -F . grad log det M / V
        gen = torch.Generator().manual_seed(17)
        U1 = SUN_haar(G2.link_shape(1)[:-1], generator=gen)
        H1 = 0.5 * random_algebra(G2.link_shape(1), generator=gen)
        kappa, K = 0.2, 4000
        U = U1.expand(G2.link_shape(K)).contiguous()
        H = H1.expand(G2.link_shape(K)).contiguous()
        term = TwoFlavourWilson(kappa, G2, solver_tol=1e-12)
        term.refresh(U, gen)
        xi = pseudofermion_pg_term(U, term, H)["xi"]
        Ug = U1.clone().requires_grad_(True)
        logdet = 2.0 * torch.linalg.slogdet(assemble_even_odd(Ug, kappa, G2))[1]
        (g,) = torch.autograd.grad(logdet.sum(), Ug)
        expected = -float((g.conj() * hermitian_tangent(H1)(U1)).real.sum()) / G2.volume
        mean, err = float(xi.mean()), float(xi.std() / K ** 0.5)
        if VERBOSE:
            print(f"<xi_f> = {mean:.5f} +- {err:.5f}, -F.grad log det M / V = {expected:.5f}")
        self.assertLess(abs(mean - expected), 4.0 * err)
        self.assertGreater(abs(expected), 10.0 * err)  # the check resolves N_f = 2 against N_f = 1
        self.assertGreater(abs(mean - 0.5 * expected), 4.0 * err)


class LadderActionTest(unittest.TestCase):
    """d_c S comes from the pure-gauge ladder action; QCDAction's parameters are constants."""

    def test_ladder_action_equals_qcd_gauge_action_and_its_c_derivative(self):
        for geometry, size in ((G2, 2), (G4, 1)):
            with self.subTest(D=geometry.D):
                gen = torch.Generator().manual_seed(5)
                U = SUN_haar(geometry.link_shape(3)[:-1], generator=gen)
                mask = line_defect(geometry, size=size, time_slice=1, space_slice=1)
                beta = 5.3
                action = defect_ladder_action(geometry, beta, mask)
                qcd = QCDAction(geometry, "bc", beta=beta, kappa=0.1, bc=0.5, defect_mask=mask)
                for c in (0.0, 0.37, 1.0, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)):
                    torch.testing.assert_close(action(U, c), qcd.gauge_action(U, c), rtol=1e-13, atol=1e-10)
                c = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
                _, dS = torch.func.jvp(lambda x: action(U, x), (c,), (torch.ones_like(c),))
                torch.testing.assert_close(dS, defect_action(qcd, U, c), rtol=1e-10, atol=1e-10)
                self.assertGreater(float(dS.abs().min()), 1e-3)
                # the pitfall: a jvp through QCDAction gives zero without an error
                _, zero = torch.func.jvp(lambda x: qcd.gauge_action(U, x), (c,), (torch.ones_like(c),))
                self.assertEqual(float(zero.abs().max()), 0.0)

    def test_defect_mask_check(self):
        mask = line_defect(G2)
        check_same_defect(mask.to(torch.int64), mask)
        with self.assertRaisesRegex(ValueError, "differ on 4 links"):
            check_same_defect(line_defect(G2, time_slice=2), mask)
        with self.assertRaisesRegex(ValueError, "shape"):
            check_same_defect(line_defect(G4, size=1), mask)


if __name__ == "__main__":
    unittest.main()
