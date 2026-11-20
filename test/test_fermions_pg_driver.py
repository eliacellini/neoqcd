"""Smoke tests of main/main_pg_fermions.py (PG training with fermions), 2D 4 x 4 and 4D 4 x 6^3 SU(3), CPU.

- single process: outputs (metrics.jsonl keys per rung, config.json with the fermions block, the
  checkpoint pair), a resumed run equal to the uninterrupted one bitwise (metrics and field), metrics
  past the checkpoint dropped, refusals (a previous run without --resume/--overwrite, a resume with
  other physics, --defect-shape line, staggered fermions);
- the checkpoint read by the unchanged pure-gauge loader (`cnf_io.load_cnf_checkpoint`) and, for a
  defect-hyper-smearing field in 4D, by main/convert_cnf_to_smearing_flow.py (NEO-REX target);
- two Gloo ranks (one rung each; NEOQCD_SKIP_GLOO=1 skips): an interrupted and resumed run equals
  the uninterrupted one.
"""

import importlib.util
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.multiprocessing as mp

from neoqcd.fermions._gauge_cnf import load_cnf_checkpoint

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TIMING = ("sweep_s", "loss_s", "gauge_loss_s", "fermion_loss_s", "peak_memory_gb", "epoch_s")
BASE = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "5.0", "--kappa", "0.18", "--tau", "0.5",
        "--n-steps", "2", "--n-gauge-substeps", "2", "--device", "cpu", "--quiet", "--defect-size", "2",
        "--time-slice", "1", "--space-slice", "1", "--architecture", "clover-chair", "--hidden-dim", "4",
        "--depth", "1", "--loop-max", "1", "--batch-size", "2", "--warmup", "1", "--therm", "1",
        "--save-every", "1", "--lr", "1e-2", "--seed", "3"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "main" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(out):
    lines = [json.loads(line) for line in (Path(out) / "metrics.jsonl").read_text().splitlines()]
    return [{k: v for k, v in line.items() if k not in TIMING} for line in lines]


class SingleProcessDriverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = _load("main_pg_fermions")

    def run_driver(self, *argv):
        return self.driver.main(list(argv))

    def test_outputs_resume_and_pure_gauge_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            self.run_driver(*BASE, "--replicas", "3", "--epochs", "2", "--output-dir", str(a))
            first = metrics(a)
            self.assertEqual([m["epoch"] for m in first], [1, 2])
            for key in ("losses_by_rung", "pinn_by_rung", "pinn_gauge_by_rung", "pinn_fermion_by_rung",
                        "pinn_cov_by_rung", "spf_per_component_by_rung", "plaquette_by_rung"):
                self.assertEqual(len(first[-1][key]), 3, key)
            for key in ("trajectory_solves", "trajectory_cg_iter", "loss_solves", "loss_cg_iter", "max_residual",
                        "unconverged_solves", "swap_acceptance_by_edge", "grad_norm", "lr"):
                self.assertIn(key, first[-1])
            self.assertEqual(first[-1]["loss_solves"], 2)
            self.assertGreater(first[-1]["pinn_fermion_by_rung"][0], 0.0)
            config = json.loads((a / "config.json").read_text())
            self.assertEqual(config["fermions"]["fermion_term"], "joint_pseudofermion")
            self.assertEqual(config["fermions"]["kappa"], 0.18)
            self.assertEqual(config["domain"], "defect")
            self.assertTrue((a / "cnf_checkpoint.rank0.slot0.pt").exists())

            # refusals
            with self.assertRaisesRegex(SystemExit, "previous run"):
                self.run_driver(*BASE, "--replicas", "3", "--epochs", "4", "--output-dir", str(a))
            with self.assertRaisesRegex(SystemExit, "kappa"):
                self.run_driver(*[v if v != "0.18" else "0.17" for v in BASE], "--replicas", "3", "--epochs", "4",
                                "--output-dir", str(a), "--resume")
            for extra, message in ((["--defect-shape", "line"], "cube"), (["--fermions", "staggered", "--mass", "0.3"], "wilson")):
                with self.assertRaisesRegex(SystemExit, message):
                    self.run_driver(*BASE, *extra, "--replicas", "3", "--epochs", "1", "--output-dir", str(b))

            # a line past the checkpoint (a crash between checkpoints) is dropped on resume
            with open(a / "metrics.jsonl", "a") as handle:
                handle.write(json.dumps({"epoch": 3, "stale": True}) + "\n")
            self.run_driver(*BASE, "--replicas", "3", "--epochs", "4", "--output-dir", str(a), "--resume")
            self.run_driver(*BASE, "--replicas", "3", "--epochs", "4", "--output-dir", str(b))
            resumed, whole = metrics(a), metrics(b)
            self.assertEqual([m["epoch"] for m in resumed], [1, 2, 3, 4])
            self.assertEqual(resumed, whole)

            field_a, _, mask, payload = load_cnf_checkpoint(a / "cnf_checkpoint.pt")
            field_b, _, _, _ = load_cnf_checkpoint(b / "cnf_checkpoint.pt")
            self.assertEqual((payload["format"], payload["version"], payload["epoch"]), ("neoqcd_cnf", 1, 4))
            self.assertEqual(int(mask[1].sum()), 2)
            for (name, x), (_, y) in zip(field_a.state_dict().items(), field_b.state_dict().items()):
                self.assertTrue(torch.equal(x, y), name)
            self.assertGreater(sum(float(p.detach().abs().sum()) for p in field_a.output_layer.parameters()), 0.0)

    def test_defect_field_converts_to_the_neorex_layer(self):
        flags = ["--D", "4", "--T", "4", "--L", "6", "--N", "3", "--beta", "6.0", "--defect-size", "1",
                 "--time-slice", "2", "--space-slice", "2"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            self.run_driver(*flags, "--kappa", "0.12", "--tau", "0.25", "--n-steps", "1", "--n-gauge-substeps", "1",
                            "--device", "cpu", "--quiet", "--architecture", "defect-hyper-smearing",
                            "--hyper-smearing-mode", "per_link", "--hidden-dim", "4", "--depth", "1",
                            "--replicas", "2", "--batch-size", "2", "--warmup", "0", "--therm", "1", "--epochs", "1",
                            "--output-dir", str(out))
            converter = _load("convert_cnf_to_smearing_flow")
            target = converter.convert("neorex", out / "cnf_checkpoint.pt", Path(tmp) / "neorex_flow.pt", [
                "--cpu", "--algorithm", "neorex", *flags, "--parameter-min", "0", "--parameter-max", "1",
                "--nf-layer-type", "smearing", "--use-hyper-smearing", "--hyper-smearing-mode", "per_link",
                "--hyper-hidden-dim", "4", "--hyper-depth", "1", "--hyper-scale-by-delta",
                "--hyper-no-normalize-by-nstep", "--neorex-protocol-steps", "2",
                "--neorex-heatbath-steps-per-step", "1", "--batch-size", "2"])
            converted = torch.load(target, weights_only=False)
            self.assertEqual(converted["source_cnf_epoch"], 1)
            self.assertEqual(converted["checkpoint_format"], "neoqcd_neorex_flow")


# ---- two Gloo ranks ------------------------------------------------------------------------------
def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, argv):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank),
                      WORLD_SIZE=str(world_size), LOCAL_WORLD_SIZE=str(world_size), OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
    torch.set_num_threads(1)
    _load("main_pg_fermions").main(argv)


@unittest.skipIf(os.environ.get("NEOQCD_SKIP_GLOO"), "NEOQCD_SKIP_GLOO set")
class DistributedDriverTest(unittest.TestCase):
    def test_resumed_run_equals_the_uninterrupted_one_on_two_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            common = [*BASE, "--replicas", "2", "--parameter-min", "0.5"]
            mp.spawn(_worker, args=(2, _free_port(), [*common, "--epochs", "2", "--output-dir", str(a)]), nprocs=2)
            self.assertTrue(all((a / f"cnf_checkpoint.rank{r}.slot0.pt").exists() for r in range(2)))
            mp.spawn(_worker, args=(2, _free_port(), [*common, "--epochs", "3", "--output-dir", str(a), "--resume"]),
                     nprocs=2)
            mp.spawn(_worker, args=(2, _free_port(), [*common, "--epochs", "3", "--output-dir", str(b)]), nprocs=2)
            resumed, whole = metrics(a), metrics(b)
            self.assertEqual([m["epoch"] for m in resumed], [1, 2, 3])
            self.assertEqual(resumed, whole)
            self.assertEqual(len(whole[-1]["losses_by_rung"]), 2)
            self.assertEqual(whole[-1]["loss_solves"], 4)  # two per rank
            field_a = load_cnf_checkpoint(a / "cnf_checkpoint.pt")[0]
            field_b = load_cnf_checkpoint(b / "cnf_checkpoint.pt")[0]
            for x, y in zip(field_a.parameters(), field_b.parameters()):
                self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()
