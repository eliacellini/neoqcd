"""End-to-end tests of main/main_rex_fermions_distributed.py: three Gloo ranks on the CPU (one rung per
rank), 2D 4 x 4 SU(3), N_f = 2 Wilson and staggered, a handful of steps.

Each run spawns three processes that call the driver's `main(argv)` with a static env:// rendezvous
(MASTER_ADDR = 127.0.0.1, a free port, GLOO_SOCKET_IFNAME = lo0 on the Mac). Checked: the outputs
exist and parse, the labels of every step are a permutation of the rungs in every chain, every rank's
bc equals its column of rank 0's parameter matrix, a resumed run repeats the uninterrupted one, and
main/optimal_ladder.py turns the output into a valid ladder that the driver accepts back through
--ladder-file. Set NEOQCD_SKIP_GLOO=1 to skip.
"""

import csv
import importlib.util
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from neoqcd.fermions._gauge import parse_ladder, validate_ladder  # noqa: E402
from neoqcd.fermions.io import load_state, read_observables  # noqa: E402

WORLD = 3
LATTICE = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "5.0",
           "--tau", "0.5", "--n-steps", "2", "--n-gauge-substeps", "2", "--device", "cpu", "--quiet",
           "--defect-size", "2", "--time-slice", "1", "--space-slice", "1"]
WILSON = LATTICE + ["--kappa", "0.18"]
STAGGERED = LATTICE + ["--fermions", "staggered", "--mass", "0.3"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "main" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, argv, result_dir):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
    torch.set_num_threads(1)
    for p in (ROOT / "src", ROOT / "main"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    driver = _load("main_rex_fermions_distributed")
    result = driver.main(list(argv))
    result.pop("meta", None)
    with open(Path(result_dir) / f"rank{rank}.json", "w") as handle:
        json.dump(result, handle)


def run_distributed(argv, result_dir):
    """Spawn WORLD ranks running main(argv); returns the per-rank results (list, rank order)."""
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    mp.spawn(_worker, args=(WORLD, _free_port(), list(argv), str(result_dir)), nprocs=WORLD, join=True)
    return [json.loads((Path(result_dir) / f"rank{r}.json").read_text()) for r in range(WORLD)]


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@unittest.skipIf(os.environ.get("NEOQCD_SKIP_GLOO"), "NEOQCD_SKIP_GLOO set")
class DistributedREXDriverTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _check_ranks_agree(self, results, out):
        """Every rank's bc = its column of rank 0's matrix, in memory and in the rank checkpoints."""
        matrix = np.asarray(results[0]["parameter_matrix"])  # (n_chains, world)
        for r in range(WORLD):
            np.testing.assert_array_equal(np.asarray(results[r]["local_parameters"]), matrix[:, r])
        states = [torch.load(out / f"rank{r:03d}" / "rex_state.pt", map_location="cpu") for r in range(WORLD)]
        np.testing.assert_array_equal(states[0]["parameter_matrix"].numpy(), matrix)
        for r in range(WORLD):
            np.testing.assert_array_equal(states[r]["local_parameters"].numpy(), matrix[:, r])
            self.assertEqual(states[r]["n_steps_done"], states[0]["n_steps_done"])

    def test_wilson_outputs_resume_and_optimal_ladder(self):
        common = WILSON + ["--n-chains", "2", "--therm", "2", "--seed", "11", "--periodic-translation",
                           "--checkpoint-every", "4", "--log-every", "3"]
        full = self.tmp / "full"
        results = run_distributed(common + ["--steps", "6", "--out", str(full)], self.tmp / "res_full")

        # outputs exist and parse
        for name in ("observables.csv", "replica_history.npz", "run_meta.json"):
            self.assertTrue((full / name).exists(), name)
        for r in range(WORLD):
            for name in ("checkpoint.pt", "checkpoint.json", "rex_state.pt"):
                self.assertTrue((full / f"rank{r:03d}" / name).exists(), (r, name))
        driver = _load("main_rex_fermions_distributed")
        rows = _rows(full / "observables.csv")
        self.assertEqual(len(rows), 8 * WORLD)
        self.assertEqual(list(rows[0].keys()), list(driver.OBS_FIELDS))
        cols = read_observables(full / "observables.csv")
        self.assertTrue(np.all(np.isfinite(cols["plaquette"])))
        self.assertEqual(int(np.sum(cols["therm"])), 2 * WORLD)
        with np.load(full / "replica_history.npz") as h:
            for name in ("labels", "plaquette", "defect_action", "trajectory_labels", "exp_minus_dH"):
                self.assertEqual(h[name].shape, (6, WORLD, 2), name)  # measured steps only
            np.testing.assert_allclose(h["parameter_ladder"], [0.0, 0.5, 1.0])
            for t in range(6):  # each chain carries every rung once, before and after the swaps
                for name in ("labels", "trajectory_labels"):
                    np.testing.assert_array_equal(np.sort(h[name][t], axis=0),
                                                  np.arange(WORLD)[:, None].repeat(2, 1))
            self.assertTrue(np.all(np.isfinite(h["defect_action"])))
            full_hist = {k: h[k].copy() for k in h.files}
        meta = json.loads((full / "run_meta.json").read_text())
        self.assertTrue(meta["finished"])
        self.assertEqual(meta["world_size"], WORLD)
        self.assertEqual(meta["steps_completed"], 6)
        swap = meta["swap"]
        self.assertEqual(swap["link_parameter_left"], [0.0, 0.5])
        self.assertEqual(swap["link_parameter_right"], [0.5, 1.0])
        self.assertEqual(sum(swap["link_proposed"]), 6 * 2)  # alternating: one pair per chain and step
        self.assertGreater(swap["total_accepted"], 0)
        self.assertIn("round_trips", meta["replica_history"])
        self.assertEqual(len(meta["replica_history"]["defect_action"]["variance"]), WORLD)
        self.assertEqual(meta["parameters"]["defect_time_slice"], 1)
        self._check_ranks_agree(results, full)

        # interrupted run (4 steps, checkpoint at the end), then resumed to 8: same history and links
        part = self.tmp / "part"
        run_distributed(common + ["--steps", "2", "--out", str(part)], self.tmp / "res_part1")
        with open(part / "observables.csv", "a") as handle:  # a row past the checkpoint, dropped on resume
            handle.write("4,0,0,0.0,0.5,1.0,0,1,1.0,1,1,0.1\n")
        resumed = run_distributed(common + ["--steps", "6", "--out", str(part), "--resume"], self.tmp / "res_part2")
        with np.load(part / "replica_history.npz") as h:
            np.testing.assert_array_equal(h["labels"], full_hist["labels"])
            np.testing.assert_array_equal(h["trajectory_labels"], full_hist["trajectory_labels"])
            for name in ("plaquette", "defect_action", "exp_minus_dH"):
                np.testing.assert_allclose(h[name], full_hist[name], rtol=0, atol=1e-10, err_msg=name)
        a, b = read_observables(full / "observables.csv"), read_observables(part / "observables.csv")
        self.assertEqual(a["step"].size, b["step"].size)
        np.testing.assert_allclose(a["plaquette"], b["plaquette"], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(a["swap_accepted"], b["swap_accepted"])
        np.testing.assert_array_equal(a["solves"], b["solves"])
        for r in range(WORLD):
            Ua, _, _ = load_state(full / f"rank{r:03d}" / "checkpoint")
            Ub, _, _ = load_state(part / f"rank{r:03d}" / "checkpoint")
            self.assertLess((Ua - Ub).abs().max().item(), 1e-10)
        self.assertEqual(resumed[0]["parameter_matrix"], results[0]["parameter_matrix"])
        self._check_ranks_agree(resumed, part)
        self.assertEqual(json.loads((part / "run_meta.json").read_text())["swap"]["link_accepted"],
                         swap["link_accepted"])

        # the existing optimal-ladder tool reads the directory unchanged
        optimal = _load("optimal_ladder")
        target = self.tmp / "ladder.json"
        optimal.main([str(full), "--output", str(target)])
        ladder = parse_ladder(str(target))
        self.assertEqual(validate_ladder(ladder, WORLD, 0.0, 1.0), ladder)
        if all(0 < v for v in swap["link_acceptance"]):
            optimal.main([str(full), "--metric", "acceptance", "--output", str(self.tmp / "ladder_acc.json")])
            validate_ladder(parse_ladder(str(self.tmp / "ladder_acc.json")), WORLD, 0.0, 1.0)

        # ... and the driver takes that ladder back (staggered fermions this time)
        stag = self.tmp / "stag"
        res = run_distributed(STAGGERED + ["--n-chains", "1", "--therm", "1", "--steps", "2", "--seed", "5",
                                           "--ladder-file", str(target), "--out", str(stag)], self.tmp / "res_stag")
        meta = json.loads((stag / "run_meta.json").read_text())
        np.testing.assert_allclose(meta["parameter_ladder"], ladder, rtol=0, atol=0)
        self.assertEqual(meta["parameters"]["fermions"], "staggered")
        side = json.loads((stag / "rank000" / "checkpoint.json").read_text())
        self.assertEqual((side["fermions"], side["mass"]), ("staggered", 0.3))
        self.assertTrue(np.all(np.isfinite(read_observables(stag / "observables.csv")["plaquette"])))
        self._check_ranks_agree(res, stag)

    def test_refuses_previous_run(self):
        out = self.tmp / "run"
        argv = WILSON + ["--n-chains", "1", "--steps", "1", "--out", str(out)]
        run_distributed(argv, self.tmp / "res1")
        with self.assertRaises(Exception):  # SystemExit on every rank -> spawn raises
            run_distributed(argv, self.tmp / "res2")


class DistributedREXArgsTest(unittest.TestCase):
    """Argument handling that needs no process group."""

    def test_parse(self):
        driver = _load("main_rex_fermions_distributed")
        args = driver.parse_args(WILSON + ["--thermal-steps", "3", "--batch-size", "4", "--output-dir", "x",
                                           "--parameter-ladder", "0,0.3,1"])
        self.assertEqual((args.therm, args.n_chains, args.out, args.ladder), (3, 4, "x", "0,0.3,1"))
        self.assertEqual((args.defect_time_slice, args.defect_space_slice), (1, 1))
        self.assertEqual(driver.parse_args(WILSON + ["--device", "cuda:1"]).device, "cuda:1")  # last one wins
        for bad in (["--device", "cuda:x"], ["--device", "gpu"], ["--device", "cuda:"], ["--log-every", "0"]):
            with self.assertRaises(SystemExit):
                driver.parse_args(WILSON + bad)


if __name__ == "__main__":
    unittest.main()
