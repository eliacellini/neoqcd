"""End-to-end tests of main/main_hmc.py --dd-blocks (DD-HMC, neoqcd.fermions.ddhmc) on 2D 8x8 SU(3)
with 4x4 blocks (the smallest 2D lattice with active links; a few seconds per test)."""

import contextlib
import csv
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("main_hmc_dd", ROOT / "main" / "main_hmc.py")
MAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAIN)

from neoqcd.fermions.ddhmc import DDHMC, BlockDecomposition, DDHMCHamiltonian  # noqa: E402
from neoqcd.fermions.io import load_state, read_observables  # noqa: E402
from neoqcd.fermions.lattice import Geometry  # noqa: E402

LATTICE = ["--D", "2", "--T", "8", "--L", "8", "--N", "3"]
RUN = ["--beta", "5.0", "--batch", "2", "--tau", "1.0", "--n-steps", "2", "--n-gauge-substeps", "2",
       "--device", "cpu", "--seed", "11", "--start", "hot", "--quiet"]
COMMON = LATTICE + RUN + ["--fermions", "wilson", "--kappa", "0.2"]  # plain HMC
DD = COMMON + ["--n-block-substeps", "2", "--dd-blocks", "4,4"]
DETERMINISTIC = ("traj", "replica", "therm", "plaquette", "dH", "accepted", "exp_mdH", "logdet",
                 "solves", "cg_iter", "cg_iter_replica", "max_residual")


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class MainHMCDDTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def assert_same_rows(self, a, b, keys=DETERMINISTIC):
        self.assertEqual(len(a), len(b))
        for ra, rb in zip(a, b):
            for key in keys:
                self.assertEqual(ra[key], rb[key], (ra["traj"], ra["replica"], key))  # text of repr: bitwise

    def test_outputs_and_bitwise_resume(self):
        """Columns and run_meta of a DD run; an interrupted run resumed mid-period of the "luscher"
        and "odometer" sequences is bitwise the uninterrupted one; without the restored sequence
        state it is not (the test discriminates)."""
        for translation in ("luscher", "odometer"):
            with self.subTest(translation=translation):
                base = DD + ["--dd-translation", translation, "--therm", "1", "--checkpoint-every", "3"]
                full = self.tmp / f"full_{translation}"
                MAIN.main(base + ["--n-traj", "6", "--out", str(full)])
                rows = _rows(full / "observables.csv")
                self.assertEqual(len(rows), 7 * 2)
                self.assertEqual(list(rows[0].keys()), list(MAIN.OBS_FIELDS))
                cols = read_observables(full / "observables.csv")
                np.testing.assert_allclose(cols["exp_mdH"], np.exp(-cols["dH"]), rtol=1e-12)
                self.assertTrue(np.all(cols["solves"] > 0) and np.all(cols["cg_iter"] > 0))
                self.assertTrue(np.all(cols["cg_iter_replica"] > 0))
                self.assertTrue(np.all((cols["max_residual"] > 0) & (cols["max_residual"] < 1e-9)))
                self.assertTrue(np.all(np.isin(cols["accepted"], (0, 1))))
                self.assertLess(np.abs(cols["dH"]).max(), 1.0)
                meta = json.loads((full / "run_meta.json").read_text())
                dd = meta["run"]["dd"]
                self.assertEqual((dd["blocks"], dd["n_blocks"], dd["translation"]), ([4, 4], [2, 2], translation))
                self.assertAlmostEqual(dd["active_fraction"], 0.375, places=14)  # 2 * 3 * 2 / (2 * 16)
                self.assertEqual(dd["active_fraction"], dd["expected_active_fraction"])
                self.assertEqual([lvl[0][0] for lvl in dd["levels"]], ["SchurTwoFlavour", "BlockTwoFlavour",
                                                                      "MaskedActionTerm"])
                self.assertEqual([lvl[1] for lvl in dd["levels"]], [1, 2, 2])
                self.assertEqual(dd["n_regular"], 1 if translation == "luscher" else 15)
                cost = dd["cost_this_process"]
                self.assertEqual(cost["schur"]["refresh_calls"], 7)
                # cost columns = Schur + block solves
                self.assertEqual(int(cols["solves"][::2].sum()), cost["schur"]["solves"] + cost["block"]["solves"])
                self.assertEqual(int(cols["cg_iter"][::2].sum()),
                                 cost["schur"]["cg_iterations"] + cost["block"]["cg_iterations"])
                self.assertEqual(meta["parameters"]["dd_blocks"], [4, 4])
                self.assertEqual(meta["summary"]["n_traj_measured"], 6)
                state = load_state(full / "checkpoint")[1]["extra"]
                self.assertEqual(state["dd"], {"blocks": [4, 4], "translation": translation})
                self.assertEqual(set(state["dd_state"]), {"k", "offset"})

                part = self.tmp / f"part_{translation}"
                MAIN.main(base + ["--n-traj", "2", "--out", str(part)])  # stops after 3 cycles: mid-period
                self.assertNotEqual(load_state(part / "checkpoint")[1]["extra"]["dd_state"]["k"], 0)
                lost = self.tmp / f"lost_{translation}"
                lost.mkdir()
                for name in ("observables.csv", "checkpoint.pt", "checkpoint.json", "run_meta.json"):
                    (lost / name).write_bytes((part / name).read_bytes())
                MAIN.main(base + ["--n-traj", "6", "--out", str(part), "--resume"])
                self.assert_same_rows(_rows(full / "observables.csv"), _rows(part / "observables.csv"))
                self.assertTrue(torch.equal(load_state(full / "checkpoint")[0], load_state(part / "checkpoint")[0]))
                # the same resume without the sequence state starts a new period: another chain
                with mock.patch.object(MAIN, "restore_dd_state", lambda hmc, state, device: None):
                    MAIN.main(base + ["--n-traj", "6", "--out", str(lost), "--resume"])
                a, b = _rows(full / "observables.csv"), _rows(lost / "observables.csv")
                self.assertNotEqual([r["plaquette"] for r in a[6:]], [r["plaquette"] for r in b[6:]])

    def test_matches_the_library_loop(self):
        """The driver's chain is DDHMC.cycle on DDHMCHamiltonian(Schur, block x n_block_substeps,
        gauge x n_gauge_substeps) with the run generator: bitwise the same dH and plaquettes."""
        out = self.tmp / "lib"
        MAIN.main(DD + ["--dd-translation", "luscher", "--n-traj", "3", "--integrator", "omelyan,leapfrog,omelyan",
                        "--out", str(out)])
        got = read_observables(out / "observables.csv")
        G = Geometry(2, 8, 8, 3)
        gen = torch.Generator().manual_seed(11)
        args = MAIN.parse_args(DD + ["--n-traj", "3"])
        U = MAIN.initial_links(args, G, torch.device("cpu"), gen)
        bd = BlockDecomposition(G, 4)
        kw = dict(solver_tol=1e-10, action_tol=None, max_iter=5000, on_unconverged="raise")
        ham = DDHMCHamiltonian(5.0, 0.2, bd, n_block_substeps=2, n_gauge_substeps=2,
                               block_kwargs=dict(kw, reuse_guess=True), schur_kwargs=kw)
        hmc = DDHMC(G, ham, tau=1.0, n_steps=2, integrator=["omelyan", "leapfrog", "omelyan"], translation="luscher")
        dH, plaq = [], []
        for _ in range(3):
            U, _, d, st = hmc.cycle(U, gen)
            dH += d.tolist()
            plaq += st["plaquette"].tolist()
        self.assertEqual(got["dH"].tolist(), dH)
        self.assertEqual(got["plaquette"].tolist(), plaq)

    def test_resume_key(self):
        out = self.tmp / "dd"
        MAIN.main(DD + ["--n-traj", "2", "--out", str(out)])
        for other in (COMMON, DD + ["--dd-translation", "luscher"], DD[:-1] + ["4"] + ["--batch", "3"]):
            with self.subTest(other=other[-2:]), self.assertRaisesRegex(SystemExit, "differ"):
                MAIN.main(other + ["--n-traj", "3", "--resume", "--out", str(out)])
        # other blocks (no other valid blocks on 8x8: the stored entry is edited)
        path = out / "checkpoint.json"
        saved = path.read_text()
        meta = json.loads(saved)
        meta["extra"]["dd"]["blocks"] = [8, 4]
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(SystemExit, "differ"):
            MAIN.main(DD + ["--n-traj", "3", "--resume", "--out", str(out)])
        path.write_text(saved)
        MAIN.main(DD[:-1] + ["4", "--n-block-substeps", "3", "--n-traj", "3", "--resume", "--out", str(out)])
        self.assertEqual(json.loads((out / "run_meta.json").read_text())["trajectories_done"], 3)
        # a plain checkpoint does not resume as DD-HMC (and has no dd entries)
        plain = self.tmp / "plain"
        MAIN.main(COMMON + ["--n-traj", "2", "--out", str(plain)])
        extra = load_state(plain / "checkpoint")[1]["extra"]
        self.assertNotIn("dd", extra)
        self.assertNotIn("dd_state", extra)
        with self.assertRaisesRegex(SystemExit, "differ"):
            MAIN.main(DD + ["--n-traj", "3", "--resume", "--out", str(plain)])
        # a DD checkpoint without the sequence state is refused
        path = out / "checkpoint.json"
        meta = json.loads(path.read_text())
        del meta["extra"]["dd_state"]
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(SystemExit, "translation state"):
            MAIN.main(DD + ["--n-traj", "4", "--resume", "--out", str(out)])

    def test_argument_checks(self):
        refused = {
            "--dd-translation needs --dd-blocks": COMMON + ["--dd-translation", "luscher"],
            "--n-block-substeps needs --dd-blocks": COMMON + ["--n-block-substeps", "2"],
            "N_f = 2 Wilson only": LATTICE + RUN + ["--fermions", "staggered", "--mass", "0.3", "--dd-blocks", "4"],
            "Hasenbusch": DD + ["--kappa-heavy", "0.1"],
            "even and >= 4": DD[:-1] + ["2,4"],
            "even and >= 4 ": DD[:-1] + ["3,4"],
            "even multiple": DD[:-1] + ["4", "--T", "12"],
            "even multiple ": DD[:-1] + ["6,4"],
            "one extent per direction": DD[:-1] + ["4,4,4"],
            "integers": DD[:-1] + ["4x4"],
            "--n-block-substeps must be >= 1": DD + ["--n-block-substeps", "0"],
            "3 level": DD + ["--integrator", "omelyan,leapfrog"],
        }
        for pattern, argv in refused.items():
            with self.subTest(pattern=pattern), self.assertRaisesRegex(SystemExit, pattern.strip()):
                MAIN.parse_args(argv)
        with self.assertRaisesRegex(SystemExit, "wilson"):
            MAIN.parse_args(LATTICE + RUN + ["--fermions", "none", "--dd-blocks", "4"])
        args = MAIN.parse_args(DD[:-1] + ["4"])
        self.assertEqual((args.dd_blocks, args.dd_translation, args.n_block_substeps, args.integrator_levels),
                         ((4, 4), "random", 2, ["omelyan"] * 3))
        args = MAIN.parse_args(DD[:-1] + ["8,4", "--dd-translation", "none"] + ["--T", "16"])
        self.assertEqual((args.dd_blocks, MAIN.dd_translation(args)), ((8, 4), None))
        self.assertEqual(MAIN.dd_key(args), {"blocks": [8, 4], "translation": "none"})
        self.assertIsNone(MAIN.dd_key(MAIN.parse_args(COMMON)))
        self.assertIn("_dd8x4_", MAIN.default_name(args))
        self.assertNotIn("_dd", MAIN.default_name(MAIN.parse_args(COMMON)))

    def test_fast_and_fused_match_reference(self):
        """--fast (and --kernel loop / cpu, the host build of the CUDA source when a C++ compiler
        is available) reproduce the reference DD run: dH to 1e-8, the same accept pattern."""
        from neoqcd.fermions.fused import backend_status
        tight = ["--n-traj", "3", "--therm", "1", "--no-reuse-guess", "--solver-tol", "1e-13", "--action-tol", "1e-13",
                 "--dd-translation", "luscher"]
        ref = self.tmp / "reference"
        MAIN.main(DD + tight + ["--out", str(ref)])
        want = read_observables(ref / "observables.csv")
        cases = {"fast": ["--fast"], "loop": ["--fast", "--kernel", "loop"],
                 "mixed": ["--fast", "--precision", "mixed", "--inner-tol", "1e-5"]}
        if backend_status("cpu")[0]:
            cases["cpu"] = ["--fast", "--kernel", "cpu"]
        for name, extra in cases.items():
            with self.subTest(case=name):
                out = self.tmp / name
                MAIN.main(DD + tight + extra + ["--out", str(out)])
                got = read_observables(out / "observables.csv")
                self.assertLess(np.abs(got["dH"] - want["dH"]).max(), 1e-8)
                self.assertTrue(np.array_equal(got["accepted"], want["accepted"]))
                self.assertLess(np.abs(got["plaquette"] - want["plaquette"]).max(), 1e-10)
                self.assertTrue(np.all(got["max_residual"] < 1e-11))
                meta = json.loads((out / "run_meta.json").read_text())
                fused = name in ("loop", "cpu")
                self.assertEqual(meta["run"]["fermion_operator"], "half_lattice_fused" if fused else "half_lattice_fast")
                levels = [lvl[0][0] for lvl in meta["run"]["dd"]["levels"]]
                self.assertEqual(levels, ["FastSchurTwoFlavour", "FusedBlockTwoFlavour" if fused else "FastBlockTwoFlavour",
                                          "MaskedActionTerm"])
                if name == "mixed":  # the real cost: complex64 loop + complex128 outer applications of both terms
                    self.assertIn("complex64", meta["run"]["cg_iter_definition"])
                    cost = meta["run"]["dd"]["cost_this_process"]
                    self.assertEqual(int(got["cg_iter"][::2].sum()),
                                     sum(cost[t]["cg_loop_iterations"] + cost[t]["double_applications"]
                                         for t in ("schur", "block")))
                    self.assertTrue(all(cost[t]["double_applications"] > 0 for t in ("schur", "block")))
        # the solver path is in the resume key
        with self.assertRaisesRegex(SystemExit, "differ"):
            MAIN.main(DD + tight[:1] + ["5"] + tight[2:] + ["--out", str(ref), "--resume", "--fast"])

    def test_translation_none_warns_and_keeps_the_grid(self):
        out = self.tmp / "none"
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            MAIN.main(DD + ["--dd-translation", "none", "--n-traj", "2", "--out", str(out)])
        self.assertIn("not ergodic", err.getvalue())
        self.assertEqual(load_state(out / "checkpoint")[1]["extra"]["dd_state"], {"k": 0, "offset": None})
        self.assertTrue(math.isfinite(json.loads((out / "run_meta.json").read_text())["summary"]["plaquette"]["value"]))


if __name__ == "__main__":
    unittest.main()
