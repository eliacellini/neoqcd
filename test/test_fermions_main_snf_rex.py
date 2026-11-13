"""End-to-end tests of main/main_snf_fermions.py and main/main_rex_fermions.py on 2D 4 x 4 SU(3)
(a handful of samples/steps; checks that the outputs exist and parse, not physics)."""

import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
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
REX = _load("main_rex_fermions")

from neoqcd.fermions.io import read_observables  # noqa: E402

LATTICE = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "5.0", "--kappa", "0.18",
           "--tau", "0.5", "--n-steps", "2", "--n-gauge-substeps", "2", "--device", "cpu", "--quiet"]

STAGGERED = [a for a in LATTICE if a not in ("--kappa", "0.18")] + ["--fermions", "staggered", "--mass", "0.3"]


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class MainSNFFermionsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _check_eval(self, out, n_samples, n_step):
        rows = _rows(out / "samples.csv")
        self.assertEqual(len(rows), n_samples)
        self.assertEqual(list(rows[0].keys()), list(SNF.SAMPLE_FIELDS))
        self.assertEqual(len(_rows(out / "blocks.csv")), (n_samples // 4) * n_step)
        meta = json.loads((out / "run_meta.json").read_text())
        s = meta["summary"]
        self.assertTrue(meta["finished"])
        self.assertTrue(math.isfinite(s["dF"]) and math.isfinite(s["dF_error"]))
        self.assertGreater(s["ess"], 0.0)
        self.assertLessEqual(s["ess"], 1.0 + 1e-12)
        self.assertEqual(s["n_samples"], n_samples)
        self.assertGreater(s["solves_per_sample"], 0)
        self.assertEqual(len(s["blocks"]["acceptance"]), n_step)
        cols = read_observables(out / "samples.csv")
        # W + Q = S_final - S_0 - logJ per sample (identity of the bookkeeping)
        lhs = cols["work"] + cols["heat"]
        rhs = cols["S_final"] - cols["S0"] - cols["logJ"]
        np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-8 * np.abs(cols["S0"]).max())
        return s

    def test_nemcmc_kappa(self):
        out = self.tmp / "nemcmc"
        res = SNF.main(LATTICE + ["--protocol", "kappa", "--final", "0.2", "--n-step", "2", "--mode", "nemcmc",
                                  "--batch", "4", "--eval-samples", "8", "--therm", "3", "--seed", "3",
                                  "--out", str(out)])
        s = self._check_eval(out, 8, 2)
        self.assertEqual(s["blocks"]["parameter"], [0.19, 0.2])
        self.assertTrue((out / "pool.pt").exists() and (out / "pool.json").exists())
        self.assertFalse((out / "flow.pt").exists())
        self.assertEqual(res["summary"]["dF"], s["dF"])

    def test_snf_bc_training_and_resume(self):
        out = self.tmp / "snf"
        base = LATTICE + ["--protocol", "bc", "--final", "0.0", "--n-step", "2", "--mode", "snf",
                          "--layers", "smearing", "--rho-init", "0.01", "--defect-shape", "line",
                          "--defect-size", "2", "--batch", "4", "--therm", "2", "--seed", "5",
                          "--lr", "1e-3", "--checkpoint-every", "1", "--out", str(out)]
        SNF.main(base + ["--train-steps", "2", "--eval-samples", "8"])
        train = _rows(out / "training.csv")
        self.assertEqual(len(train), 2)
        self.assertTrue(all(math.isfinite(float(r["loss"])) and float(r["grad_norm"]) > 0 for r in train))
        s = self._check_eval(out, 8, 2)
        self.assertEqual(s["training"]["steps"], 2)
        payload = torch.load(out / "flow.pt")
        self.assertEqual(payload["train_step"], 2)
        self.assertTrue(any("rho" in k for k in payload["state_dict"]))
        json.loads((out / "flow.json").read_text())
        # a finished run resumes to a no-op; --load-flow evaluates the trained layers elsewhere
        SNF.main(base + ["--train-steps", "2", "--eval-samples", "8", "--resume"])
        self.assertEqual(len(_rows(out / "samples.csv")), 8)
        out2 = self.tmp / "snf_eval"
        args2 = [a if a != str(out) else str(out2) for a in base]
        SNF.main(args2 + ["--eval-samples", "8", "--load-flow", str(out / "flow.pt")])
        self._check_eval(out2, 8, 2)

    def test_interrupted_snf_resumes_identically(self):
        base = LATTICE + ["--protocol", "kappa", "--final", "0.19", "--n-step", "2", "--mode", "snf",
                          "--rho-init", "0.01", "--batch", "4", "--therm", "1", "--seed", "21",
                          "--train-steps", "3", "--eval-samples", "12", "--checkpoint-every", "1"]
        full, part = self.tmp / "full", self.tmp / "part"
        SNF.main(base + ["--out", str(full)])
        for stop in (2, 5):  # inside training (2nd step), inside evaluation (2nd batch)
            calls = {"n": 0}
            original = SNF.effective_sample_size

            def failing(W, original=original, stop=stop):
                calls["n"] += 1
                if calls["n"] == stop:
                    raise RuntimeError("simulated wall time")
                return original(W)
            SNF.effective_sample_size = failing
            try:
                with self.assertRaises(RuntimeError):
                    SNF.main(base + ["--out", str(part)] + (["--resume"] if stop == 5 else []))
            finally:
                SNF.effective_sample_size = original
        SNF.main(base + ["--out", str(part), "--resume"])
        for name, field in (("training.csv", "loss"), ("samples.csv", "work"), ("samples.csv", "S_final")):
            a, b = read_observables(full / name)[field], read_observables(part / name)[field]
            np.testing.assert_array_equal(a, b)

    def test_snf_residual_one_step(self):
        out = self.tmp / "res"
        SNF.main(LATTICE + ["--protocol", "beta", "--final", "5.2", "--n-step", "2", "--mode", "snf",
                            "--layers", "residual", "--batch", "4", "--train-steps", "1", "--eval-samples", "4",
                            "--seed", "9", "--out", str(out)])
        self._check_eval(out, 4, 2)

    def test_nemcmc_staggered_mass(self):
        out = self.tmp / "stag"
        SNF.main(STAGGERED + ["--protocol", "mass", "--final", "0.25", "--n-step", "2", "--mode", "nemcmc",
                              "--batch", "4", "--eval-samples", "8", "--therm", "2", "--seed", "5",
                              "--out", str(out)])
        s = self._check_eval(out, 8, 2)
        np.testing.assert_allclose(s["blocks"]["parameter"], [0.275, 0.25])
        meta = json.loads((out / "run_meta.json").read_text())
        self.assertEqual(meta["parameters"]["fermions"], "staggered")
        self.assertEqual(meta["parameters"]["mass"], 0.3)
        self.assertIsNone(meta["parameters"]["kappa"])
        side = json.loads((out / "pool.json").read_text())
        self.assertEqual(side["fermions"], "staggered")
        self.assertEqual(side["mass"], 0.3)

    def test_wrong_fermion_combinations(self):
        for argv in (STAGGERED + ["--kappa", "0.1", "--protocol", "mass", "--final", "0.2", "--n-step", "1"],
                     STAGGERED + ["--protocol", "kappa", "--final", "0.2", "--n-step", "1"],
                     LATTICE + ["--protocol", "mass", "--final", "0.2", "--n-step", "1"],
                     LATTICE + ["--mass", "0.3", "--protocol", "beta", "--final", "5.2", "--n-step", "1"]):
            with self.assertRaises(SystemExit):
                SNF.parse_args(argv)


class MainREXFermionsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_outputs_and_resume(self):
        common = LATTICE + ["--ladder", "0,0.5,1", "--defect-shape", "line", "--defect-size", "2",
                            "--n-chains", "2", "--therm", "2", "--seed", "11", "--periodic-translation"]
        full = self.tmp / "full"
        REX.main(common + ["--steps", "8", "--checkpoint-every", "4", "--out", str(full)])
        for name in ("observables.csv", "replica_history.npz", "run_meta.json", "checkpoint.pt",
                     "checkpoint.json", "rex_state.pt"):
            self.assertTrue((full / name).exists(), name)
        rows = _rows(full / "observables.csv")
        self.assertEqual(len(rows), 10 * 3)
        self.assertEqual(list(rows[0].keys()), list(REX.OBS_FIELDS))
        with np.load(full / "replica_history.npz") as h:
            self.assertEqual(h["labels"].shape, (10, 3, 2))
            self.assertEqual(h["plaquette"].shape, (10, 3, 2))
            for t in range(10):  # every chain carries each rung once
                np.testing.assert_array_equal(np.sort(h["labels"][t], axis=0), np.arange(3)[:, None].repeat(2, 1))
            labels_full, plaq_full = h["labels"].copy(), h["plaquette"].copy()
        meta = json.loads((full / "run_meta.json").read_text())
        s = meta["summary"]
        self.assertTrue(meta["finished"])
        self.assertEqual(s["steps_measured"], 8)
        self.assertEqual(len(s["swap_acceptance"]), 2)
        self.assertIn("round_trips", s["history"])
        self.assertIn("plaquette_bottom", s["history"])
        self.assertGreater(s["solves_per_step"], 0)

        # interrupted run: 6 steps (checkpoint at 4, rows up to 6), then resume to 10
        part = self.tmp / "part"
        REX.main(common + ["--steps", "4", "--checkpoint-every", "4", "--out", str(part)])
        with open(part / "observables.csv", "a") as handle:  # a row past the checkpoint, dropped on resume
            handle.write("6,0,0,0.0,0.5,1.0,0,1,1.0,1,1,0.1\n")
        REX.main(common + ["--steps", "8", "--checkpoint-every", "4", "--out", str(part), "--resume"])
        with np.load(part / "replica_history.npz") as h:
            np.testing.assert_array_equal(h["labels"], labels_full)
            np.testing.assert_allclose(h["plaquette"], plaq_full, rtol=0, atol=1e-12)
        a, b = read_observables(full / "observables.csv"), read_observables(part / "observables.csv")
        np.testing.assert_allclose(a["plaquette"], b["plaquette"], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(a["solves"], b["solves"])

    def test_staggered(self):
        out = self.tmp / "stag"
        REX.main(STAGGERED + ["--ladder", "0,0.5,1", "--defect-shape", "line", "--defect-size", "2",
                              "--n-chains", "2", "--therm", "1", "--seed", "7", "--steps", "3", "--out", str(out)])
        for name in ("observables.csv", "replica_history.npz", "run_meta.json", "checkpoint.pt",
                     "checkpoint.json", "rex_state.pt"):
            self.assertTrue((out / name).exists(), name)
        rows = _rows(out / "observables.csv")
        self.assertEqual(len(rows), 4 * 3)
        cols = read_observables(out / "observables.csv")
        self.assertTrue(np.all(np.isfinite(cols["plaquette"])))
        with np.load(out / "replica_history.npz") as h:
            self.assertEqual(h["labels"].shape, (4, 3, 2))
        meta = json.loads((out / "run_meta.json").read_text())
        self.assertTrue(meta["finished"])
        self.assertEqual(meta["parameters"]["fermions"], "staggered")
        self.assertEqual(meta["run"]["mass"], 0.3)
        side = json.loads((out / "checkpoint.json").read_text())
        self.assertEqual((side["fermions"], side["mass"]), ("staggered", 0.3))
        self.assertIsNone(side.get("kappa"))


class REXDefectAndLadderTest(unittest.TestCase):
    """The drivers build the defect of size L_d exactly as main_pt_obc.py (`utils.Defect`), and a
    fermionic REX run feeds the optimal-ladder machinery of the pure-gauge code unchanged."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_defect_mask_equals_pure_gauge(self):
        from neoqcd.fermions._gauge import Defect
        from neoqcd.fermions.lattice import Geometry
        DIST = _load("main_rex_fermions_distributed")
        for D in (4, 2):
            lattice = ["--D", str(D), "--T", "4", "--L", "4", "--N", "3", "--kappa", "0.18", "--device", "cpu"]
            for spelling in (["--time-slice", "1", "--space-slice", "2"],
                             ["--defect-time-slice", "1", "--defect-space-slice", "2"]):
                reference = Defect(D, 4, 4, 2, time_slice=1, space_slice=2).create_defect_mask()
                for module, extra in ((REX, ["--ladder", "0,1"]), (DIST, [])):
                    with self.subTest(D=D, spelling=spelling[0], driver=module.__name__):
                        args = module.parse_args(lattice + ["--defect-size", "2"] + spelling + extra)
                        self.assertEqual(args.defect_shape, "cube")
                        mask = SNF.build_defect_mask(args, Geometry(D, 4, 4, 3))
                        self.assertEqual(mask.dtype, torch.float64)
                        self.assertTrue(torch.equal(mask, reference.to(torch.float64)))

    def test_optimal_ladder_from_fermionic_run(self):
        from neoqcd.fermions._gauge import (optimal_ladder, parse_ladder, rung_moments, thermodynamic_length,
                                            validate_ladder)
        run = self.tmp / "run"
        therm, steps = 2, 16
        REX.main(LATTICE + ["--ladder", "0,0.5,1", "--defect-size", "2", "--time-slice", "1", "--space-slice", "1",
                            "--n-chains", "3", "--therm", str(therm), "--steps", str(steps), "--seed", "4",
                            "--periodic-translation", "--out", str(run)])
        with np.load(run / "replica_history.npz") as h:
            self.assertEqual(h["defect_action"].shape, h["labels"].shape)
            np.testing.assert_allclose(h["parameter_ladder"], [0.0, 0.5, 1.0])
            labels, sd, c = h["labels"].astype(np.int64), h["defect_action"], h["parameter_ladder"]
        meta = json.loads((run / "run_meta.json").read_text())
        self.assertEqual(meta["swap"]["link_parameter_left"], [0.0, 0.5])
        self.assertEqual(sum(meta["swap"]["link_proposed"]), steps * 3)
        self.assertEqual(len(meta["replica_history"]["defect_action"]["variance"]), 3)
        # the functions of neoqcd.ladder, through the _gauge re-exports
        moments = rung_moments(labels, sd, 3, discard_fraction=therm / (therm + steps))
        var = np.asarray(moments["variance"])
        self.assertTrue(np.all(var > 0))
        lam = thermodynamic_length(c, var)
        new = optimal_ladder(c, lam, 4)
        self.assertEqual(validate_ladder(new, 4, 0.0, 1.0), [float(v) for v in new])
        # main/optimal_ladder.py on the directory, unchanged, and the result back through --ladder-file
        target = self.tmp / "ladder.json"
        _load("optimal_ladder").main([str(run), "--n-rep", "4", "--output", str(target)])
        ladder = parse_ladder(str(target))
        self.assertEqual(len(ladder), 4)
        again = self.tmp / "again"
        REX.main(LATTICE + ["--ladder-file", str(target), "--n-chains", "1", "--steps", "2", "--out", str(again)])
        meta = json.loads((again / "run_meta.json").read_text())
        self.assertEqual(meta["parameters"]["ladder_values"], ladder)
        with self.assertRaises(SystemExit):  # --ladder and --n-rep exclude each other
            REX.parse_args(LATTICE + ["--ladder", "0,0.5", "--n-rep", "3"])


if __name__ == "__main__":
    unittest.main()
