"""Smoke test of main/bench_fermions.py on CPU (2^4 and 2^3 x 4 SU(3), a few seconds, no CUDA)."""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("bench_fermions", ROOT / "main" / "bench_fermions.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)

from neoqcd.fermions._gauge import SUN_haar  # noqa: E402
from neoqcd.fermions.dirac import WilsonDirac  # noqa: E402
from neoqcd.fermions.lattice import Geometry  # noqa: E402

FAST = ["--device", "cpu", "--batches", "1", "2", "--dtypes", "complex128", "--target-time", "0",
        "--trials", "1", "--warmup", "1", "--min-repeats", "1", "--max-repeats", "1", "--cg-iters", "3",
        "--cg-min-iters", "1", "--no-compile", "--quiet"]
VARIANTS = ("gather", "gather_spinproj", "roll_spinproj")


class BenchFermionsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_main_cpu(self):
        out = self.tmp / "bench"
        result = BENCH.main(FAST + ["--lattices", "2^4", "2^3x4", "--breakdown-config", "2^3x4:2",
                                    "--out", str(out)])
        with open(out / "bench.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(result["rows"]))
        self.assertEqual(set(rows[0]), set(BENCH.CSV_FIELDS))
        bad = [r for r in rows if r["status"] not in ("ok", "kernel_count")]
        self.assertEqual(bad, [])
        timed = [r for r in rows if r["status"] == "ok"]
        for r in timed:
            self.assertGreater(float(r["seconds_per_call"]), 0.0)
            self.assertAlmostEqual(float(r["site_updates_per_s"]) * float(r["seconds_per_call"]),
                                   int(r["volume"]) * int(r["batch"]), delta=1e-6 * int(r["volume"]) * int(r["batch"]))
        expected = {("hopping", "reference"), ("wilson_apply", "reference"), ("eo_apply", "reference"),
                    ("eo_normal", "reference"), ("eo_normal", "gather_spinproj"),
                    ("cg_iteration", "solver_cg"), ("cg_iteration", "bare"),
                    ("cg_iteration", "bare_gather_spinproj"), ("pf_force", "given_solution"),
                    ("pf_force", "hopping_derivative"), ("gauge_force", "reference"),
                    ("expi_update", "cayley_hamilton"), ("staggered_mee", "reference")}
        expected |= {("hopping", v) for v in VARIANTS}
        for lattice in ("2^4", "2^3x4"):
            for batch in ("1", "2"):
                got = {(r["measurement"], r["variant"]) for r in timed
                       if r["lattice"] == lattice and r["batch"] == batch}
                self.assertTrue(expected <= got, expected - got)
        for r in timed:
            if r["max_rel_error"]:
                self.assertLess(float(r["max_rel_error"]), 1e-12)
        breakdown = [r for r in timed if r["measurement"] == "breakdown"]
        self.assertEqual(len(breakdown), 6)
        self.assertTrue(all(r["lattice"] == "2^3x4" and r["batch"] == "2" for r in breakdown))
        counts = [r for r in rows if r["status"] == "kernel_count"]
        self.assertTrue(counts and all(float(r["kernels_per_call"]) > 0 for r in counts))
        meta = json.loads((out / "run_meta.json").read_text())
        for key in ("device", "torch", "commit", "complex64_supported", "variant_errors", "kernel_counts"):
            self.assertIn(key, meta)
        self.assertEqual(meta["device"], "cpu")
        self.assertEqual(len(meta["variant_errors"]), 2 * 2 * len(VARIANTS))
        self.assertTrue(all(e < 1e-12 for e in meta["variant_errors"].values()))
        if not meta["complex64_supported"]:
            self.assertNotIn("complex64", meta["dtypes_benchmarked"])

    def test_variants_match_reference(self):
        """The variants agree with WilsonDirac.hopping where +mu and -mu differ (4^3 x 6, two kappas)."""
        geo = Geometry(4, 6, 4, 3)
        g = torch.Generator().manual_seed(3)
        U = SUN_haar(geo.link_shape(2)[:-1], generator=g)
        psi = torch.randn(geo.wilson_shape(2), dtype=torch.complex128, generator=g)
        for time_sign in (-1.0, 1.0):
            w = WilsonDirac(U, torch.tensor([0.12, 0.15]), geo, time_sign=time_sign)
            var = BENCH.HoppingVariants(w)
            ref = w.hopping(psi)
            for name in VARIANTS:
                self.assertLess(BENCH.rel_error(getattr(var, name)(psi), ref), 1e-12, name)

    def test_bare_cg_matches_solver(self):
        geo = Geometry(4, 4, 2, 3)
        g = torch.Generator().manual_seed(5)
        U = SUN_haar(geo.link_shape(2)[:-1], generator=g)
        eo = WilsonDirac(U, 0.12, geo).even_odd()
        b = eo.even(torch.randn(geo.wilson_shape(2), dtype=torch.complex128, generator=g))
        x_bare = BENCH.bare_cg(eo.normal, b, 6)
        x_ref = BENCH.cg(eo.normal, b, tol=0.0, max_iter=6).x
        self.assertLess(BENCH.rel_error(x_bare, x_ref), 1e-10)

    def test_parse_lattice(self):
        self.assertEqual(BENCH.parse_lattice("16^4"), Geometry(4, 16, 16, 3))
        self.assertEqual(BENCH.parse_lattice("12^3x24"), Geometry(4, 24, 12, 3))
        self.assertEqual(BENCH.parse_config("12^3x24:16"), ("12^3x24", 16))


if __name__ == "__main__":
    unittest.main()
