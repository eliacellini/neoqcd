"""End-to-end tests of main/main_hmc.py on 2D 4 x 4 SU(3) (a few seconds on a laptop CPU)."""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("main_hmc", ROOT / "main" / "main_hmc.py")
MAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAIN)

from neoqcd.fermions.io import load_state, read_observables  # noqa: E402

COMMON = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "2.0", "--batch", "2",
          "--tau", "0.5", "--n-steps", "3", "--n-gauge-substeps", "2", "--device", "cpu",
          "--seed", "7", "--start", "hot", "--quiet"]
WILSON = COMMON + ["--fermions", "wilson", "--kappa", "0.12"]
DETERMINISTIC = ("traj", "replica", "therm", "plaquette", "dH", "accepted", "exp_mdH", "logdet",
                 "solves", "cg_iter", "cg_iter_replica")


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class MainHMCTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_wilson_outputs_and_bitwise_resume(self):
        full = self.tmp / "full"
        res = MAIN.main(WILSON + ["--therm", "2", "--n-traj", "8", "--checkpoint-every", "4",
                                  "--exact-logdet", "--meas-every", "2", "--out", str(full)])
        for name in ("observables.csv", "observables.npz", "run_meta.json", "checkpoint.pt", "checkpoint.json"):
            self.assertTrue((full / name).exists(), name)
        rows = _rows(full / "observables.csv")
        self.assertEqual(len(rows), 10 * 2)
        self.assertEqual(list(rows[0].keys()), list(MAIN.OBS_FIELDS))
        self.assertEqual(sum(int(r["therm"]) for r in rows), 2 * 2)
        cols = read_observables(full / "observables.csv")
        self.assertTrue(np.all(np.isfinite(cols["plaquette"])))
        self.assertTrue(np.all((cols["plaquette"] > -1) & (cols["plaquette"] < 1)))
        np.testing.assert_allclose(cols["exp_mdH"], np.exp(-cols["dH"]), rtol=1e-12)
        self.assertTrue(np.all(cols["solves"] > 0) and np.all(cols["cg_iter"] > 0))
        # logdet every second trajectory only
        has_logdet = ~np.isnan(cols["logdet"])
        np.testing.assert_array_equal(has_logdet, cols["traj"] % 2 == 0)
        npz = np.load(full / "observables.npz")
        np.testing.assert_array_equal(npz["plaquette"], cols["plaquette"])

        meta = json.loads((full / "run_meta.json").read_text())
        self.assertTrue(meta["finished"])
        self.assertEqual(meta["trajectories_done"], 10)
        s = meta["summary"]
        self.assertEqual(s["n_traj_measured"], 8)
        self.assertEqual(s["plaquette"]["status"], "gamma")
        self.assertGreater(s["plaquette"]["error"], 0.0)
        measured = cols["therm"] == 0
        self.assertAlmostEqual(s["plaquette"]["value"], cols["plaquette"][measured].mean(), places=12)
        self.assertAlmostEqual(s["acceptance"]["value"], cols["accepted"][measured].mean(), places=12)
        self.assertGreater(s["acceptance"]["value"], 0.5)
        self.assertLess(np.abs(cols["dH"]).max(), 1.0)
        for key in ("seconds_per_traj", "seconds_per_solve", "cg_iter_per_solve", "solves_per_traj"):
            self.assertGreater(s[key], 0.0, key)
        self.assertIn("logdet", s)
        self.assertEqual(res["summary"]["n_traj_measured"], 8)
        self.assertEqual(meta["parameters"]["kappa"], 0.12)

        U_full, state_meta, _ = load_state(full / "checkpoint")
        self.assertEqual(state_meta["trajectory"], 10)
        self.assertEqual(state_meta["n_flavours"], 2)
        self.assertEqual(state_meta["geometry"], {"D": 2, "T": 4, "L": 4, "N": 3})

        # interrupted run: stop after 5 trajectories, then resume to 10
        part = self.tmp / "part"
        MAIN.main(WILSON + ["--therm", "2", "--n-traj", "3", "--checkpoint-every", "4",
                            "--exact-logdet", "--meas-every", "2", "--out", str(part)])
        self.assertEqual(load_state(part / "checkpoint")[1]["trajectory"], 5)
        with self.assertRaises(SystemExit):  # a new run would clobber it
            MAIN.main(WILSON + ["--n-traj", "3", "--out", str(part)])
        with self.assertRaises(SystemExit):  # different physics
            MAIN.main(COMMON + ["--fermions", "wilson", "--kappa", "0.11", "--therm", "2", "--n-traj", "8",
                                "--out", str(part), "--resume"])
        MAIN.main(WILSON + ["--therm", "2", "--n-traj", "8", "--checkpoint-every", "4",
                            "--exact-logdet", "--meas-every", "2", "--out", str(part), "--resume"])
        a, b = _rows(full / "observables.csv"), _rows(part / "observables.csv")
        self.assertEqual(len(a), len(b))
        for ra, rb in zip(a, b):
            for key in DETERMINISTIC:
                self.assertEqual(ra[key], rb[key], (ra["traj"], key))  # text of repr: bitwise
        U_part = load_state(part / "checkpoint")[0]
        self.assertTrue(torch.equal(U_full, U_part))

    def test_resume_drops_rows_past_checkpoint(self):
        out = self.tmp / "trunc"
        MAIN.main(WILSON + ["--n-traj", "4", "--checkpoint-every", "2", "--out", str(out)])
        # simulate a crash after trajectory 4 with the checkpoint at 2: rewind the checkpoint
        ref = _rows(out / "observables.csv")
        mid = self.tmp / "mid"
        MAIN.main(WILSON + ["--n-traj", "2", "--out", str(mid)])
        for suffix in (".pt", ".json"):
            (out / ("checkpoint" + suffix)).write_bytes((mid / ("checkpoint" + suffix)).read_bytes())
        MAIN.main(WILSON + ["--n-traj", "4", "--checkpoint-every", "2", "--out", str(out), "--resume"])
        rows = _rows(out / "observables.csv")
        self.assertEqual([r["traj"] for r in rows], [r["traj"] for r in ref])
        for ra, rb in zip(ref, rows):
            self.assertEqual(ra["plaquette"], rb["plaquette"])

    def test_quenched_and_staggered_run(self):
        for extra, nf in ((["--fermions", "none"], 0), (["--fermions", "staggered", "--mass", "0.3", "--exact-logdet"], 2)):
            out = self.tmp / extra[1]
            MAIN.main(COMMON + extra + ["--n-traj", "5", "--out", str(out)])
            meta = json.loads((out / "run_meta.json").read_text())
            self.assertEqual(meta["summary"]["n_traj_measured"], 5)
            self.assertEqual(meta["run"]["n_flavours"], nf)
            cols = read_observables(out / "observables.csv")
            self.assertEqual(cols["traj"].size, 5 * 2)
            self.assertTrue(np.all(np.isfinite(cols["dH"])))
            if nf == 0:
                self.assertTrue(np.all(cols["solves"] == 0))
                self.assertIsNone(meta["summary"]["seconds_per_solve"])
            else:
                self.assertTrue(np.all(cols["solves"] > 0))
                self.assertTrue(np.all(np.isfinite(cols["logdet"])))

    def test_argument_checks(self):
        with self.assertRaises(SystemExit):
            MAIN.parse_args(COMMON + ["--fermions", "wilson"])  # no kappa
        with self.assertRaises(SystemExit):
            MAIN.parse_args(COMMON + ["--fermions", "staggered", "--kappa", "0.1"])
        with self.assertRaises(SystemExit):  # 8^4 Wilson SU(3) is far too large for the dense determinant
            MAIN.main(["--D", "4", "--T", "8", "--L", "8", "--kappa", "0.153", "--exact-logdet",
                       "--device", "cpu", "--quiet", "--out", str(self.tmp / "big")])


if __name__ == "__main__":
    unittest.main()
