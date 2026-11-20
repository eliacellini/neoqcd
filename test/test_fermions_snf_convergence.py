"""Solver-convergence record of main/main_snf_fermions.py (review of 20 November): --on-unconverged
reaches the fermion term (through `fermion_kwargs`, which also feeds both REX drivers), and every
output row records the largest true relative residual and the number of unconverged solves, so
that --on-unconverged warn|ignore leaves a trace (samples.csv, training.csv, run_meta.json);
`SolveMonitor` and the CSV summaries are NaN-sticky.

2D 4 x 4 SU(3), a few samples; an unconverged run is made with --max-iter 3.
"""

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "main" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SNF = _load("main_snf_fermions")

from neoqcd.fermions.action import QCDAction  # noqa: E402
from neoqcd.fermions.io import read_observables  # noqa: E402
from neoqcd.fermions.lattice import Geometry  # noqa: E402
from neoqcd.fermions.rex import FermionREX  # noqa: E402
from neoqcd.fermions.solver import SolverResult  # noqa: E402

LATTICE = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "5.0", "--kappa", "0.18",
           "--tau", "0.5", "--n-steps", "2", "--n-gauge-substeps", "2", "--device", "cpu", "--quiet"]
STAGGERED = [a for a in LATTICE if a not in ("--kappa", "0.18")] + ["--fermions", "staggered", "--mass", "0.3"]
NEMCMC = ["--protocol", "kappa", "--final", "0.19", "--n-step", "2", "--batch", "2", "--eval-samples", "4",
          "--seed", "5", "--therm", "1"]


def quiet_main(argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return SNF.main(argv)


class ForwardingTest(unittest.TestCase):
    """--on-unconverged arrives at the fermion term of QCDAction and FermionREX (reference and fast)."""

    def test_fermion_kwargs(self):
        g = Geometry(2, 4, 4, 3)
        for policy in ("warn", "ignore", "raise"):
            for fast in ([], ["--fast"]):
                with self.subTest(policy=policy, fast=fast):
                    args = SNF.parse_args(LATTICE + NEMCMC + ["--on-unconverged", policy] + fast)
                    kw = SNF.fermion_kwargs(args)
                    self.assertEqual(kw["on_unconverged"], policy)
                    act = QCDAction(g, "kappa", beta=5.0, **SNF.action_kwargs(args), **kw)
                    self.assertEqual(act.fermion.on_unconverged, policy)
                    akw = SNF.action_kwargs(args)
                    rex = FermionREX(g, 5.0, akw.pop("kappa"), torch.stack((torch.ones((2, 4, 4)),
                                                                            torch.zeros((2, 4, 4)))),
                                     [0.0, 1.0], **akw, **kw)
                    self.assertEqual(rex.qcd.fermion.on_unconverged, policy)


class SolveMonitorTest(unittest.TestCase):
    """The record is NaN-sticky whatever the order, counts a solve once even with several bad samples,
    and records a solve whose policy then raises."""

    class Term:
        def __init__(self, residuals, raise_on=()):
            self.stats = {"solves": 0, "last_residual": None}
            self.residuals, self.raise_on = list(residuals), set(raise_on)

        def _solve(self, op, tol):
            k = self.stats["solves"]
            self.stats["solves"] += 1
            self.stats["last_residual"] = torch.tensor(self.residuals[k], dtype=torch.float64)
            if k in self.raise_on:
                raise RuntimeError("not converged")
            return k

    def test_nan_sticky(self):
        for order in ([[1e-12, float("nan")], [1e-3, 1e-3]], [[1e-3, 1e-3], [float("nan"), 1e-12]]):
            term = self.Term(order + [[1e-14, 1e-14]])
            mon = SNF.SolveMonitor(term)
            for _ in range(3):
                term._solve(None, 1e-10)
            m, n = mon.interval()
            self.assertTrue(math.isnan(m), order)
            self.assertEqual(n, 2)  # the NaN solve and the 1e-3 solve (twice 1e-3 in one solve: once)
            self.assertEqual(mon.interval(), (None, 0))
            self.assertTrue(math.isnan(mon.total_max))
        for a, b in ((1.0, float("nan")), (float("nan"), 1.0), (None, float("nan"))):
            self.assertTrue(math.isnan(SNF.nan_max(a, b)))
        self.assertEqual(SNF.nan_max(None, None), None)
        term = self.Term([[1e-2]], raise_on={0})
        mon = SNF.SolveMonitor(term)
        with self.assertRaises(RuntimeError):
            term._solve(None, 1e-10)
        self.assertEqual(mon.interval(), (1e-2, 1))
        # with the real terms' SolverResult the residual is per sample
        self.assertIn("residual", SolverResult._fields)

    def test_csv_summary_nan_sticky(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("step,rung,max_residual,unconverged_solves\n0,0,,\n1,0,1e-12,0\n1,1,1e-12,0\n"
                            "2,0,nan,1\n2,1,nan,1\n3,0,1e-3,2\n3,1,1e-3,2\n")
            c = SNF.convergence_from_csv(path, "step")
            self.assertTrue(math.isnan(c["max_residual"]) and c["max_residual_nonfinite"])
            self.assertEqual(c["unconverged_solves"], 3)
            self.assertEqual(c["rows_without_record"], 1)


class DriverConvergenceTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_converged_run_records_residuals(self):
        out = self.tmp / "ok"
        quiet_main(LATTICE + NEMCMC + ["--mode", "snf", "--train-steps", "1", "--out", str(out)])
        for name in ("samples.csv", "training.csv"):
            cols = read_observables(out / name)
            self.assertTrue((cols["max_residual"] > 0).all() and (cols["max_residual"] < 1e-9).all(), name)
            self.assertTrue((cols["unconverged_solves"] == 0).all(), name)
        conv = json.loads((out / "run_meta.json").read_text())["summary"]["convergence"]
        self.assertEqual(conv["unconverged_solves"], 0)
        self.assertLess(conv["max_residual"], 1e-9)
        self.assertFalse(conv["max_residual_nonfinite"])

    def test_unconverged_solves_leave_a_trace(self):
        """--max-iter 3: with ignore (or warn) the run finishes and every row counts its unconverged
        solves; with the default raise it stops. If --on-unconverged did not reach the term (the
        surviving mutation of the review), the ignore run would raise."""
        base = LATTICE + NEMCMC + ["--max-iter", "3"]
        with self.assertRaisesRegex(RuntimeError, "not converged"):
            quiet_main(base + ["--out", str(self.tmp / "raise")])
        for policy in ("ignore", "warn"):
            with self.subTest(policy=policy):
                out = self.tmp / policy
                quiet_main(base + ["--mode", "snf", "--train-steps", "1", "--on-unconverged", policy,
                                   "--out", str(out)])
                for name in ("samples.csv", "training.csv"):
                    cols = read_observables(out / name)
                    self.assertTrue((cols["unconverged_solves"] > 0).all(), name)
                    self.assertTrue((cols["max_residual"] > 1e-9).all(), name)
                meta = json.loads((out / "run_meta.json").read_text())
                conv = meta["summary"]["convergence"]
                self.assertGreater(conv["evaluation"]["unconverged_solves"], 0)
                self.assertGreater(conv["training"]["unconverged_solves"], 0)
                self.assertEqual(conv["unconverged_solves"], conv["evaluation"]["unconverged_solves"]
                                 + conv["training"]["unconverged_solves"])
                self.assertGreater(meta["run"]["therm_unconverged_solves"], 0)

    def test_staggered_without_policy_is_recorded(self):
        """The reference staggered term has no convergence policy: its unconverged solves are counted
        by the monitor with the same rule."""
        out = self.tmp / "stag"
        quiet_main(STAGGERED + ["--protocol", "mass", "--final", "0.25", "--n-step", "2", "--batch", "2",
                                "--eval-samples", "4", "--seed", "5", "--max-iter", "3", "--out", str(out)])
        cols = read_observables(out / "samples.csv")
        self.assertTrue((cols["unconverged_solves"] > 0).all())


if __name__ == "__main__":
    unittest.main()
