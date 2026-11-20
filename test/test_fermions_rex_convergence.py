"""Solver-convergence record of the two REX drivers (review of 20 November): with --on-unconverged
ignore an unconverged run finishes and observables.csv / run_meta.json record the largest true
relative residual and the number of unconverged solves of every step (all ranks for the
distributed driver); with the default raise it stops (so --on-unconverged reaches the replicas'
fermion term). 2D 4 x 4 SU(3), --max-iter 3 for the unconverged runs.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "main" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REX = _load("main_rex_fermions")

from neoqcd.fermions.io import read_observables  # noqa: E402

LATTICE = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "5.0", "--kappa", "0.18",
           "--tau", "0.5", "--n-steps", "2", "--n-gauge-substeps", "2", "--device", "cpu", "--quiet"]
COMMON = LATTICE + ["--ladder", "0,0.5,1", "--defect-shape", "line", "--defect-size", "2", "--n-chains", "2",
                    "--therm", "1", "--steps", "3", "--seed", "11"]


def quiet(fn, argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(argv)


class REXConvergenceTest(unittest.TestCase):

    def test_single_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            quiet(REX.main, COMMON + ["--out", str(tmp / "ok")])
            cols = read_observables(tmp / "ok" / "observables.csv")
            self.assertTrue((cols["max_residual"] > 0).all() and (cols["max_residual"] < 1e-9).all())
            self.assertTrue((cols["unconverged_solves"] == 0).all())
            with self.assertRaisesRegex(RuntimeError, "not converged"):
                quiet(REX.main, COMMON + ["--max-iter", "3", "--out", str(tmp / "raise")])
            quiet(REX.main, COMMON + ["--max-iter", "3", "--on-unconverged", "ignore", "--out", str(tmp / "ignore")])
            cols = read_observables(tmp / "ignore" / "observables.csv")
            self.assertTrue((cols["unconverged_solves"] > 0).all())
            self.assertTrue((cols["max_residual"] > 1e-9).all())
            conv = json.loads((tmp / "ignore" / "run_meta.json").read_text())["summary"]["convergence"]
            per_step = cols["unconverged_solves"][cols["rung"] == 0]
            self.assertEqual(conv["unconverged_solves"], int(per_step.sum()))
            self.assertEqual(per_step.size, 4)  # thermalisation step included


@unittest.skipIf(os.environ.get("NEOQCD_SKIP_GLOO"), "NEOQCD_SKIP_GLOO set")
class DistributedREXConvergenceTest(unittest.TestCase):
    """Three Gloo ranks: the per-rank records are gathered (max over ranks, sum of the counts)."""

    def test_distributed(self):
        from test.test_fermions_rex_distributed import WILSON, DriverExit, run_distributed  # noqa: F401
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            common = WILSON + ["--n-chains", "1", "--therm", "1", "--seed", "11", "--steps", "2",
                               "--checkpoint-every", "0", "--max-iter", "3"]
            run_distributed(common + ["--on-unconverged", "ignore", "--out", str(tmp / "ignore")], tmp / "res")
            cols = read_observables(tmp / "ignore" / "observables.csv")
            self.assertTrue((cols["unconverged_solves"] >= 3).all())  # at least one per rank and step
            self.assertTrue((cols["max_residual"] > 1e-9).all())
            meta = json.loads((tmp / "ignore" / "run_meta.json").read_text())
            self.assertEqual(meta["summary"]["convergence"]["unconverged_solves"],
                             int(cols["unconverged_solves"][cols["rung"] == 0].sum()))
            with self.assertRaises(Exception):  # the default raise stops every rank
                run_distributed(common + ["--out", str(tmp / "raise")], tmp / "res_raise")


if __name__ == "__main__":
    unittest.main()
