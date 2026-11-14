"""End-to-end tests of main/main_rex_fermions_distributed.py: three Gloo ranks on the CPU (one rung per
rank), 2D 4 x 4 SU(3), N_f = 2 Wilson and staggered, a handful of steps.

Each run spawns three processes that call the driver's `main(argv)` with a static env:// rendezvous
(MASTER_ADDR = 127.0.0.1, a free port, GLOO_SOCKET_IFNAME = lo0 on the Mac), with a deadline (a rank
left blocked in a collective fails the test instead of hanging it). Faults are injected in the
workers (`mode`). Checked: the outputs exist and parse, the labels of every step are a permutation
of the rungs in every chain, every rank's bc equals its column of rank 0's parameter matrix, resumed
runs (interrupted at an odd and at an even measured step, after a crash between checkpoints, after a
checkpoint that failed on one rank only) repeat the uninterrupted one, rank-0 failures stop every
rank, foreign files in --out are neither blocking nor deleted, a start killed before its first
checkpoint restarts, the summary fallback keeps the swap block, and main/optimal_ladder.py turns the
output into a valid ladder that the driver accepts back through --ladder-file. Set
NEOQCD_SKIP_GLOO=1 to skip.
"""

import csv
import importlib.util
import json
import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from neoqcd.fermions._gauge import parse_ladder, validate_ladder  # noqa: E402
from neoqcd.fermions.io import load_state, read_observables  # noqa: E402

WORLD = 3
DEADLINE_S = 600.0
LATTICE = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "5.0",
           "--tau", "0.5", "--n-steps", "2", "--n-gauge-substeps", "2", "--device", "cpu", "--quiet",
           "--defect-size", "2", "--time-slice", "1", "--space-slice", "1"]
WILSON = LATTICE + ["--kappa", "0.18"]
STAGGERED = LATTICE + ["--fermions", "staggered", "--mass", "0.3"]


class DriverExit(Exception):
    """The driver left with SystemExit; `messages` maps rank -> message."""

    def __init__(self, messages):
        super().__init__(str(messages))
        self.messages = messages


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "main" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _inject(driver, rank, mode):
    """Faults of the worker (in this process only)."""
    if mode is None:
        return
    kind, _, arg = mode.partition(":")
    if kind == "crash_step":  # every rank raises at the arg-th measured step (a kill between checkpoints)
        inner, calls = driver.distributed_rex_step, [0]

        def crashing(*a, **kw):
            calls[0] += 1
            if calls[0] == int(arg):
                raise RuntimeError("injected crash")
            return inner(*a, **kw)
        driver.distributed_rex_step = crashing
    elif kind == "rank0_rows":  # rank 0's CSV write fails at its arg-th call (rank-0-only section)
        if rank == 0:
            inner, calls = driver.ObservableWriter.write_rows, [0]

            def write_rows(self, rows):
                calls[0] += 1
                if calls[0] == int(arg):
                    raise OSError("injected: disk full")
                return inner(self, rows)
            driver.ObservableWriter.write_rows = write_rows
    elif kind == "save_fails_on":  # save_state fails on rank arg only (ranks end at different steps)
        if rank == int(arg):
            def save_state(*a, **kw):
                raise OSError("injected: quota exceeded")
            driver.save_state = save_state
    elif kind == "summary_fails":  # the Gamma analysis of the history fails
        def summarize_history(*a, **kw):
            raise FloatingPointError("injected")
        driver.summarize_history = summarize_history
    else:
        raise ValueError(mode)


def _worker(rank, world_size, port, argv, result_dir, mode):
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
    _inject(driver, rank, mode)
    try:
        result = driver.main(list(argv))
    except SystemExit as exc:
        (Path(result_dir) / f"rank{rank}.exit").write_text(str(exc))
        return
    result.pop("meta", None)
    with open(Path(result_dir) / f"rank{rank}.json", "w") as handle:
        json.dump(result, handle)


def run_distributed(argv, result_dir, mode=None):
    """Spawn WORLD ranks running main(argv); returns the per-rank results (list, rank order).
    DriverExit if the driver exited with SystemExit (on any rank), ProcessRaisedException if a rank
    raised; a run still alive after DEADLINE_S (a rank blocked in a collective) fails."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    for old in list(result_dir.glob("rank*.json")) + list(result_dir.glob("rank*.exit")):
        old.unlink()
    context = mp.start_processes(_worker, args=(WORLD, _free_port(), list(argv), str(result_dir), mode),
                                 nprocs=WORLD, join=False, start_method="spawn")
    deadline = time.time() + DEADLINE_S
    while not context.join(timeout=1.0):
        if time.time() > deadline:
            for proc in context.processes:
                proc.kill()
            raise AssertionError(f"distributed run still alive after {DEADLINE_S} s (rank blocked?)")
    exits = {r: (result_dir / f"rank{r}.exit").read_text() for r in range(WORLD) if (result_dir / f"rank{r}.exit").exists()}
    if exits:
        raise DriverExit(exits)
    return [json.loads((result_dir / f"rank{r}.json").read_text()) for r in range(WORLD)]


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _marker(out):
    return json.loads((Path(out) / "checkpoint_step.json").read_text())


def _rank_state(out, rank, slot=None):
    slot = _marker(out)["slot"] if slot is None else slot
    return torch.load(Path(out) / f"rank{rank:03d}" / f"rex_state_{slot}.pt", map_location="cpu")


@unittest.skipIf(os.environ.get("NEOQCD_SKIP_GLOO"), "NEOQCD_SKIP_GLOO set")
class DistributedREXDriverTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _check_ranks_agree(self, results, out):
        """Every rank's bc = its column of rank 0's matrix, in memory and in the committed checkpoint."""
        matrix = np.asarray(results[0]["parameter_matrix"])  # (n_chains, world)
        for r in range(WORLD):
            np.testing.assert_array_equal(np.asarray(results[r]["local_parameters"]), matrix[:, r])
        states = [_rank_state(out, r) for r in range(WORLD)]
        np.testing.assert_array_equal(states[0]["parameter_matrix"].numpy(), matrix)
        for r in range(WORLD):
            np.testing.assert_array_equal(states[r]["local_parameters"].numpy(), matrix[:, r])
            self.assertEqual(states[r]["n_steps_done"], _marker(out)["step"])

    def _check_same_run(self, a, b, results_a=None, results_b=None):
        """Directory b repeats directory a: history, CSV, links, swap counters, PT state."""
        with np.load(a / "replica_history.npz") as ha, np.load(b / "replica_history.npz") as hb:
            for name in ("labels", "trajectory_labels"):
                np.testing.assert_array_equal(hb[name], ha[name], err_msg=name)
            for name in ("plaquette", "defect_action", "exp_minus_dH"):
                np.testing.assert_allclose(hb[name], ha[name], rtol=0, atol=1e-10, err_msg=name)
        ca, cb = read_observables(a / "observables.csv"), read_observables(b / "observables.csv")
        self.assertEqual(ca["step"].size, cb["step"].size)
        np.testing.assert_allclose(cb["plaquette"], ca["plaquette"], rtol=0, atol=1e-10)
        for name in ("step", "swap_accepted", "swap_proposed", "solves"):
            np.testing.assert_array_equal(cb[name], ca[name], err_msg=name)
        self.assertEqual(_marker(a)["step"], _marker(b)["step"])
        for r in range(WORLD):
            Ua, _, _ = load_state(a / f"rank{r:03d}" / f"checkpoint_{_marker(a)['slot']}")
            Ub, _, _ = load_state(b / f"rank{r:03d}" / f"checkpoint_{_marker(b)['slot']}")
            self.assertLess((Ua - Ub).abs().max().item(), 1e-10)
            sa, sb = _rank_state(a, r), _rank_state(b, r)
            self.assertEqual(sa["pt_step"], sb["pt_step"])
            self.assertTrue(torch.equal(sa["swap_rng"], sb["swap_rng"]))
        ma, mb = json.loads((a / "run_meta.json").read_text()), json.loads((b / "run_meta.json").read_text())
        for name in ("link_accepted", "link_proposed"):
            self.assertEqual(mb["swap"][name], ma["swap"][name], name)
        if results_a is not None:
            self.assertEqual(results_b[0]["parameter_matrix"], results_a[0]["parameter_matrix"])
            self._check_ranks_agree(results_b, b)

    def test_wilson_outputs_resume_and_optimal_ladder(self):
        common = WILSON + ["--n-chains", "2", "--therm", "2", "--seed", "11", "--periodic-translation",
                           "--checkpoint-every", "4", "--log-every", "3"]
        full = self.tmp / "full"
        results = run_distributed(common + ["--steps", "6", "--out", str(full)], self.tmp / "res_full")

        # outputs exist and parse
        for name in ("observables.csv", "replica_history.npz", "run_meta.json", "checkpoint_step.json"):
            self.assertTrue((full / name).exists(), name)
        self.assertEqual(_marker(full), {"step": 8, "slot": 1, "world_size": WORLD})  # commits at 4 (slot 0), 8
        for r in range(WORLD):
            for slot in (0, 1):
                for name in (f"checkpoint_{slot}.pt", f"checkpoint_{slot}.json", f"rex_state_{slot}.pt"):
                    self.assertTrue((full / f"rank{r:03d}" / name).exists(), (r, name))
            self.assertEqual(_rank_state(full, r, slot=0)["n_steps_done"], 4)  # the previous generation
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
            self.assertEqual(int(h["thermal_steps"]), 2)
            for t in range(6):  # each chain carries every rung once, before and after the swaps
                for name in ("labels", "trajectory_labels"):
                    np.testing.assert_array_equal(np.sort(h[name][t], axis=0),
                                                  np.arange(WORLD)[:, None].repeat(2, 1))
            self.assertTrue(np.all(np.isfinite(h["defect_action"])))
        meta = json.loads((full / "run_meta.json").read_text())
        self.assertTrue(meta["finished"])
        self.assertEqual(meta["world_size"], WORLD)
        self.assertEqual(meta["steps_completed"], 6)
        swap = meta["swap"]
        self.assertEqual(swap["link_parameter_left"], [0.0, 0.5])
        self.assertEqual(swap["link_parameter_right"], [0.5, 1.0])
        self.assertEqual(sum(swap["link_proposed"]), 6 * 2)  # alternating: one pair per chain and step
        self.assertEqual(swap["link_proposed"], [3 * 2, 3 * 2])  # even pairs at even steps, odd at odd
        self.assertGreater(swap["total_accepted"], 0)
        self.assertIn("round_trips", meta["replica_history"])
        self.assertEqual(len(meta["replica_history"]["defect_action"]["variance"]), WORLD)
        self.assertEqual(meta["parameters"]["defect_time_slice"], 1)
        self._check_ranks_agree(results, full)

        # interrupted at an ODD measured step (5 steps = 2 therm + 3 measured, pt._pt_step = 3), then
        # resumed to 8: a lost parity counter would swap the wrong pairs (link_proposed differs)
        odd = self.tmp / "odd"
        run_distributed(common + ["--steps", "3", "--out", str(odd)], self.tmp / "res_odd1")
        self.assertEqual(_rank_state(odd, 0)["pt_step"], 3)
        resumed = run_distributed(common + ["--steps", "6", "--out", str(odd), "--resume"], self.tmp / "res_odd2")
        self._check_same_run(full, odd, results, resumed)

        # crash between checkpoints, in a rank-0-only section: rank 0's CSV write of step 6 fails ->
        # every rank exits (none left blocked); on disk rows of steps 0..5 and a history of 4 measured
        # steps (log at step 5), committed checkpoint at step 4 (2 measured: EVEN parity)
        crash = self.tmp / "crash"
        crash.mkdir()
        (crash / "notes.txt").write_text("mine")
        with self.assertRaises(DriverExit) as ctx:
            run_distributed(common + ["--steps", "6", "--out", str(crash)], self.tmp / "res_c1", mode="rank0_rows:7")
        self.assertEqual(sorted(ctx.exception.messages), list(range(WORLD)))
        self.assertIn("output of step 6 failed on rank 0", ctx.exception.messages[0])
        self.assertEqual(_marker(crash)["step"], 4)
        with np.load(crash / "replica_history.npz") as h:
            self.assertEqual(h["labels"].shape[0], 4)  # past the checkpoint: truncated on resume
        self.assertEqual(int(read_observables(crash / "observables.csv")["step"].max()), 5)
        # resume; the final checkpoint (step 8) fails on rank 1 only: ranks 0 and 2 hold step 8 in
        # slot 1, rank 1 does not, the marker stays at step 4 (slot 0)
        with self.assertRaises(DriverExit) as ctx:
            run_distributed(common + ["--steps", "6", "--out", str(crash), "--resume"], self.tmp / "res_c2",
                            mode="save_fails_on:1")
        self.assertIn("rank 1 failed", ctx.exception.messages[1])
        self.assertEqual(_marker(crash), {"step": 4, "slot": 0, "world_size": WORLD})
        self.assertEqual(_rank_state(crash, 0, slot=1)["n_steps_done"], 8)
        self.assertEqual(_rank_state(crash, 1, slot=0)["n_steps_done"], 4)
        resumed = run_distributed(common + ["--steps", "6", "--out", str(crash), "--resume"], self.tmp / "res_c3")
        self._check_same_run(full, crash, results, resumed)
        self.assertEqual((crash / "notes.txt").read_text(), "mine")

        # the existing optimal-ladder tool reads the directory unchanged
        optimal = _load("optimal_ladder")
        target = self.tmp / "ladder.json"
        optimal.main([str(full), "--output", str(target)])
        ladder = parse_ladder(str(target))
        self.assertEqual(validate_ladder(ladder, WORLD, 0.0, 1.0), ladder)
        self.assertTrue(all(v > 0 for v in swap["link_acceptance"]),
                        f"--metric acceptance needs every link accepted at least once: {swap['link_acceptance']}")
        optimal.main([str(full), "--metric", "acceptance", "--output", str(self.tmp / "ladder_acc.json")])
        validate_ladder(parse_ladder(str(self.tmp / "ladder_acc.json")), WORLD, 0.0, 1.0)

        # ... and the driver takes that ladder back (staggered fermions this time), with a failing
        # Gamma analysis: run_meta keeps the swap block and records the error
        stag = self.tmp / "stag"
        res = run_distributed(STAGGERED + ["--n-chains", "1", "--therm", "1", "--steps", "2", "--seed", "5",
                                           "--ladder-file", str(target), "--out", str(stag)], self.tmp / "res_stag",
                              mode="summary_fails")
        meta = json.loads((stag / "run_meta.json").read_text())
        np.testing.assert_allclose(meta["parameter_ladder"], ladder, rtol=0, atol=0)
        self.assertEqual(meta["parameters"]["fermions"], "staggered")
        self.assertTrue(meta["finished"])
        self.assertIn("FloatingPointError", meta["replica_history"]["error"])
        self.assertEqual(sum(meta["swap"]["link_proposed"]), 2)
        self.assertGreater(meta["summary"]["solves_per_step"], 0)
        side = json.loads((stag / "rank000" / f"checkpoint_{_marker(stag)['slot']}.json").read_text())
        self.assertEqual((side["fermions"], side["mass"]), ("staggered", 0.3))
        self.assertTrue(np.all(np.isfinite(read_observables(stag / "observables.csv")["plaquette"])))
        self._check_ranks_agree(res, stag)

    def test_own_files_only_and_restart_before_first_checkpoint(self):
        argv = WILSON + ["--n-chains", "1", "--therm", "1", "--steps", "2", "--seed", "3", "--checkpoint-every", "0"]
        foreign = ("notes.txt", "sub/data.txt", "rank000/mine.txt", "rank007/other.txt")

        def plant(out):
            for name in foreign:
                (out / name).parent.mkdir(parents=True, exist_ok=True)
                (out / name).write_text(name)

        def check_foreign(out):
            for name in foreign:
                self.assertEqual((out / name).read_text(), name, name)

        # foreign files neither block a start nor get deleted
        a = self.tmp / "a"
        plant(a)
        run_distributed(argv + ["--out", str(a)], self.tmp / "r1")
        check_foreign(a)
        ref = read_observables(a / "observables.csv")
        # a finished run is refused without --resume / --overwrite (SystemExit on every rank)
        with self.assertRaises(DriverExit) as ctx:
            run_distributed(argv + ["--out", str(a)], self.tmp / "r2")
        self.assertEqual(sorted(ctx.exception.messages), list(range(WORLD)))
        self.assertIn("holds a previous run", ctx.exception.messages[0])
        # --overwrite deletes the run's own files only
        run_distributed(argv + ["--out", str(a), "--overwrite"], self.tmp / "r3")
        check_foreign(a)
        np.testing.assert_array_equal(read_observables(a / "observables.csv")["plaquette"], ref["plaquette"])

        # killed in the first measured step, before any checkpoint: rows on disk, no marker
        b = self.tmp / "b"
        plant(b)
        with self.assertRaises(ProcessRaisedException):
            run_distributed(argv + ["--out", str(b)], self.tmp / "r4", mode="crash_step:1")
        self.assertFalse((b / "checkpoint_step.json").exists())
        self.assertTrue((b / "observables.csv").exists())
        # the next start (as the job script, without --resume) deletes the partial files and restarts
        run_distributed(argv + ["--out", str(b)], self.tmp / "r5")
        check_foreign(b)
        got = read_observables(b / "observables.csv")
        np.testing.assert_array_equal(got["step"], ref["step"])
        np.testing.assert_array_equal(got["plaquette"], ref["plaquette"])


class DistributedREXArgsTest(unittest.TestCase):
    """Argument handling, seeds and devices: no process group needed."""

    def test_parse(self):
        driver = _load("main_rex_fermions_distributed")
        args = driver.parse_args(WILSON + ["--thermal-steps", "3", "--batch-size", "4", "--output-dir", "x",
                                           "--parameter-ladder", "0,0.3,1"])
        self.assertEqual((args.therm, args.n_chains, args.out, args.ladder), (3, 4, "x", "0,0.3,1"))
        self.assertEqual((args.defect_time_slice, args.defect_space_slice), (1, 1))
        self.assertFalse(args.allow_shared_gpu)
        self.assertTrue(driver.parse_args(WILSON + ["--allow-shared-gpu"]).allow_shared_gpu)
        self.assertEqual(driver.parse_args(WILSON + ["--device", "cuda:1"]).device, "cuda:1")  # last one wins
        for bad in (["--device", "cuda:x"], ["--device", "gpu"], ["--device", "cuda:"], ["--log-every", "0"]):
            with self.assertRaises(SystemExit):
                driver.parse_args(WILSON + bad)

    def test_seeds_collision_free(self):
        """Seeds s and s + 1 share no stream on any pair of ranks (seed + rank would: s, 1 == s + 1, 0)."""
        driver = _load("main_rex_fermions_distributed")
        for world in (2, 3, 4, 8):
            for s in (0, 11, 1234):
                streams = {}
                for seed in (s, s + 1):
                    for r in range(world):
                        for name, value in driver.stream_seeds(seed, r).items():
                            streams[(seed, r, name)] = value
                self.assertEqual(len(set(streams.values())), len(streams), (world, s))
                draws = {k: torch.rand(4, generator=torch.Generator().manual_seed(v), dtype=torch.float64)
                         for k, v in streams.items() if k[2] == "run"}
                for r in range(world):
                    for q in range(world):
                        self.assertFalse(torch.equal(draws[(s, r, "run")], draws[(s + 1, q, "run")]), (world, s, r, q))
        self.assertEqual(driver.stream_seeds(7, 2), driver.stream_seeds(7, 2))  # deterministic

    def test_rank_device(self):
        driver = _load("main_rex_fermions_distributed")

        def choose(spec, count, local_rank, local_world, allow=False):
            env = {"LOCAL_RANK": str(local_rank), "LOCAL_WORLD_SIZE": str(local_world)}
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(torch.cuda, "is_available", return_value=True), \
                    mock.patch.object(torch.cuda, "device_count", return_value=count), \
                    mock.patch.object(torch.cuda, "set_device"):
                return driver.choose_rank_device(spec, allow)

        self.assertEqual(choose("cuda", 4, 2, 4), (torch.device("cuda", 2), "nccl"))
        self.assertEqual(choose("auto", 4, 0, 4), (torch.device("cuda", 0), "nccl"))
        with self.assertRaises(SystemExit):  # rank 2 of 4 on a node with one visible GPU
            choose("cuda", 1, 2, 4)
        with self.assertRaises(SystemExit):
            choose("cuda", 2, 3, 4)
        self.assertEqual(choose("cuda", 2, 3, 4, allow=True)[0], torch.device("cuda", 1))
        with self.assertRaises(SystemExit):  # every rank on cuda:1
            choose("cuda:1", 4, 0, 4)
        self.assertEqual(choose("cuda:1", 4, 0, 4, allow=True)[0], torch.device("cuda", 1))
        self.assertEqual(choose("cuda:1", 4, 0, 1)[0], torch.device("cuda", 1))  # one local rank: fine
        with self.assertRaises(SystemExit):  # index beyond the visible GPUs
            choose("cuda:4", 4, 0, 1)
        self.assertEqual(driver.choose_rank_device("cpu"), (torch.device("cpu"), "gloo"))

    def test_own_files(self):
        driver = _load("main_rex_fermions_distributed")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ("observables.csv", "checkpoint_step.json", ".run_meta.json.abc.tmp", "notes.txt",
                         "rank000/checkpoint_0.pt", "rank000/rex_state_1.pt", "rank000/mine.txt",
                         "rank001/checkpoint.json", "rankX/checkpoint_0.pt", "sub/observables.csv"):
                (out / name).parent.mkdir(parents=True, exist_ok=True)
                (out / name).write_text("x")
            own = driver.own_files(out)
            self.assertEqual(own, [".run_meta.json.abc.tmp", "checkpoint_step.json", "observables.csv",
                                   "rank000/checkpoint_0.pt", "rank000/rex_state_1.pt", "rank001/checkpoint.json"])
            driver.remove_own_files(out, own)
            left = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
            self.assertEqual(left, ["notes.txt", "rank000/mine.txt", "rankX/checkpoint_0.pt", "sub/observables.csv"])
            self.assertFalse((out / "rank001").exists())  # emptied rank directory removed


if __name__ == "__main__":
    unittest.main()
