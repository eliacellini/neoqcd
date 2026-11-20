"""The fast path of the PG pseudofermion term (`fast.FastTwoFlavourWilson`, `QCDAction(fast=True)`)
against the reference term of step 1 (itself checked against dense double autograd).

Half-lattice operator with every colour product, lean CG (double) and mixed-precision CG, both
time signs, per-sample kappa, 2D 4 x 4 and 2^3 x 4 SU(3): xi_f, its gradient and S_pf to the solver
tolerance; the joint loss and its parameter gradients; the convergence policy; exact-type refusal of
fast subclasses (e.g. the fused-kernel term); the single-process trainer with fast=True against the
reference trainer (same swap decisions, losses to the solver tolerance); the driver with --fast.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from neoqcd.fermions._gauge import SUN_haar
from neoqcd.fermions.action import QCDAction
from neoqcd.fermions.algebra import random_algebra
from neoqcd.fermions.fast import FastTwoFlavourWilson
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.pg import (
    FermionCNFREXTrainer,
    defect_ladder_action,
    fermion_path_gradient_loss,
    pseudofermion_pg_term,
)
from neoqcd.fermions.pseudofermion import TwoFlavourWilson
from test.test_fermions_pg import u_dependent_tangent
from test.test_fermions_pg_loss import line_defect, small_field
from test.test_fermions_pg_training import make_field, PHYSICS

ROOT = Path(__file__).resolve().parents[1]
G2 = Geometry(2, 4, 4, 3)
G4 = Geometry(4, 4, 2, 3)
CASES = ((G2, (0.20, 0.17)), (G4, (0.12, 0.10)))
VARIANTS = (dict(precision="double", colour_product="matmul"), dict(precision="double", colour_product="broadcast"),
            dict(precision="double", colour_product="broadcast_sum"), dict(precision="mixed", colour_product="matmul"))


def pair(geometry, kappa, time_sign, seed, **fast):
    gen = torch.Generator().manual_seed(seed)
    U = SUN_haar(geometry.link_shape(2)[:-1], generator=gen)
    k = torch.tensor(kappa, dtype=torch.float64)
    ref = TwoFlavourWilson(k, geometry, time_sign=time_sign, solver_tol=1e-12)
    ref.refresh(U, gen)
    quick = FastTwoFlavourWilson(k, geometry, time_sign=time_sign, solver_tol=1e-12, **fast)
    quick.phi = ref.phi
    return U, ref, quick


class FastTermTest(unittest.TestCase):
    def test_fast_equals_reference(self):
        for geometry, kappa in CASES:
            for time_sign in (-1.0, 1.0):
                for variant in VARIANTS:
                    with self.subTest(D=geometry.D, time_sign=time_sign, **variant):
                        U, ref, quick = pair(geometry, kappa, time_sign, 13, **variant)
                        velocity = u_dependent_tangent(geometry, 2, 4)
                        a = pseudofermion_pg_term(U, ref, velocity)
                        b = pseudofermion_pg_term(U, quick, velocity)
                        torch.testing.assert_close(b["xi"], a["xi"], rtol=1e-9, atol=1e-12)
                        torch.testing.assert_close(b["S_pf"], a["S_pf"], rtol=1e-10, atol=0)
                        scale = float(a["gradient"].abs().max())
                        torch.testing.assert_close(b["gradient"], a["gradient"], rtol=0, atol=1e-9 * scale)
                        self.assertEqual(b["counter"].solves, 2)
                        self.assertGreater(b["counter"].cg_loop_iterations, 0)
                        self.assertLess(b["counter"].max_residual, 1e-10)
                        self.assertEqual(quick.stats["solves"], 0)

    def test_parametric_fast_term_and_policy(self):
        gen = torch.Generator().manual_seed(3)
        U = SUN_haar(G2.link_shape(2)[:-1], generator=gen)
        H = 0.3 * random_algebra(G2.link_shape(2), generator=gen)
        fast = QCDAction(G2, "kappa", beta=5.0, kappa=0.2, fast=True, solver_tol=1e-12)
        ref = QCDAction(G2, "kappa", beta=5.0, kappa=0.2, solver_tol=1e-12)
        fast.refresh(U, torch.Generator().manual_seed(9))
        ref.refresh(U, torch.Generator().manual_seed(9))  # the same draws
        torch.testing.assert_close(fast.phi, ref.phi, rtol=1e-14, atol=1e-14)
        a = pseudofermion_pg_term(U, ref.fermion, H)
        b = pseudofermion_pg_term(U, fast.fermion, H)
        torch.testing.assert_close(b["xi"], a["xi"], rtol=1e-9, atol=1e-12)
        with self.assertRaisesRegex(RuntimeError, "pg_X: CG not converged"):
            pseudofermion_pg_term(U, fast.fermion, H, max_iter=2)

        class Fused(FastTwoFlavourWilson):  # e.g. the fused-kernel term: not validated here
            pass
        other = Fused(0.2, G2)
        other.phi = ref.phi
        with self.assertRaises(TypeError):
            pseudofermion_pg_term(U, other, H)

    def test_joint_loss_with_the_fast_term(self):
        U, ref, quick = pair(G2, (0.2, 0.2), -1.0, 21)
        field = small_field(scale=0.05)
        action = defect_ladder_action(G2, 5.0, line_defect(G2))
        t = torch.tensor([0.3, 0.8], dtype=torch.float64)
        params = list(field.parameters())
        out = []
        for term in (ref, quick):
            result = fermion_path_gradient_loss(field, action, term, U, t)
            grads = torch.autograd.grad(result["loss"].mean(), params, allow_unused=True)
            out.append((result, grads))
        (a, ga), (b, gb) = out
        torch.testing.assert_close(b["loss"].detach(), a["loss"].detach(), rtol=1e-8, atol=1e-12)
        self.assertTrue(torch.equal(b["loss_gauge"], a["loss_gauge"]))
        for x, y in zip(ga, gb):
            if x is not None:
                torch.testing.assert_close(y, x, rtol=1e-8, atol=1e-12)


class FastTrainerTest(unittest.TestCase):
    def test_fast_trainer_follows_the_reference_one(self):
        runs = []
        for fast in (False, True):
            field = make_field(4)
            optimizer = torch.optim.Adam(field.parameters(), lr=1e-2)
            extra = dict(fast=True, precision="double") if fast else {}
            trainer = FermionCNFREXTrainer(field, optimizer, defect_mask=line_defect(G2), ladder=[0.0, 0.5, 1.0],
                                           batch_size=2, seed=6, **dict(PHYSICS, **extra))
            trainer.thermalize(1)
            runs.append([trainer.step() for _ in range(3)])
        for a, b in zip(*runs):
            self.assertEqual(a["swap_acceptance_by_edge"], b["swap_acceptance_by_edge"])
            self.assertEqual(a["hmc_acceptance"], b["hmc_acceptance"])
            for key in ("losses_by_rung", "pinn_fermion_by_rung", "spf_per_component_by_rung", "plaquette_by_rung"):
                torch.testing.assert_close(torch.tensor(b[key]), torch.tensor(a[key]), rtol=1e-6, atol=1e-10, msg=key)
            self.assertEqual(b["loss_solves"], 2)


class FastDriverTest(unittest.TestCase):
    def test_driver_with_fast(self):
        spec = importlib.util.spec_from_file_location("main_pg_fermions", ROOT / "main" / "main_pg_fermions.py")
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        from test.test_fermions_pg_driver import BASE
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            driver.main([*BASE, "--replicas", "2", "--epochs", "2", "--fast", "--precision", "mixed",
                         "--output-dir", str(out)])
            lines = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[-1]["loss_solves"], 2)
            config = json.loads((out / "config.json").read_text())
            self.assertTrue(config["fermions"]["solver"]["fast"])
            self.assertEqual(config["fermions"]["solver"]["precision"], "mixed")
            self.assertGreater(lines[-1]["loss_cg_applications"], 0)
            with self.assertRaisesRegex(SystemExit, "fast"):  # a resume on the other path is refused
                driver.main([*BASE, "--replicas", "2", "--epochs", "3", "--output-dir", str(out), "--resume"])


if __name__ == "__main__":
    unittest.main()
