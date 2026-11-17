"""Path Gradient training with one replica per rank, on three Gloo ranks."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neoqcd.cnf import path_gradient_loss
from neoqcd.cnf_io import build_cnf
from neoqcd.cnf_training import DistributedCNFREXTrainer
from test.test_cnf_neorex import _free_port, _multirank_worker as _pt_obc_worker

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(
    architecture="hyper-residual", domain="beta", D=2, T=4, L=4, beta=6.0, parameter_min=0.5,
    parameter_max=1.0, batch_size=2, hidden_dim=4, depth=1, rho_init=1e-2, residual_n_pt=1,
    residual_mask_mod=4, orsteps=1,
)


def _environment(rank, world_size, port):
    # Same environment torchrun provides.
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
    torch.set_num_threads(1)


def _gradient_worker(rank, world_size, port):
    _environment(rank, world_size, port)
    dist.init_process_group("gloo")
    try:
        torch.manual_seed(7 + rank)  # different fields per rank: the trainer must broadcast rank 0's
        field, pars, mask = build_cnf(CONFIG)
        with torch.no_grad():
            for parameter in field.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        ladder = torch.linspace(0.5, 1.0, world_size, dtype=torch.float64)
        trainer = DistributedCNFREXTrainer(
            field, pars, ladder, torch.optim.Adam(field.parameters(), lr=1e-3), rank=rank,
            world_size=world_size, batch_size=2,
        )
        trainer.warmup(3)
        labels = trainer._gather(trainer._labels())
        own = torch.arange(world_size).repeat_interleave(2)
        assert bool((labels != own).any()), "no swap was accepted: the check would not mix labels"

        trainer.optimizer.zero_grad(set_to_none=True)
        trainer._backward()
        distributed = [p.grad.clone() for p in field.parameters()]

        # Reference: one process, every chain of every rank at its own parameter.
        configs = trainer._gather(trainer.x)
        times = trainer._gather(trainer.pt.local_parameters)
        reference = copy.deepcopy(field)
        reference.zero_grad(set_to_none=True)
        path_gradient_loss(reference, trainer.action, configs, times, volume=trainer.volume)["loss"].mean().backward()
        for grad, expected in zip(distributed, (p.grad for p in reference.parameters())):
            torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-13)
        assert any(float(g.abs().max()) > 0 for g in distributed)
    finally:
        dist.destroy_process_group()


def _main_worker(rank, world_size, port, argv):
    _environment(rank, world_size, port)
    spec = importlib.util.spec_from_file_location("main_cnf_rex", ROOT / "main" / "main_cnf_rex.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(argv)


def _fused_worker(rank, world_size, port, argv, emulate, count_file):
    # main_cnf_rex.py with the kernel of the HB+OR sweeps in its CPU emulation (--fused-kernels turns it on;
    # HBOR.fused_emulate lets it run on CPU tensors), counting the sweeps the kernel ran
    _environment(rank, world_size, port)
    import neoqcd.fused as fused
    from neoqcd.mcmc import HBOR
    HBOR.fused_emulate = bool(emulate)
    original, calls = fused.hbor_call, [0]

    def counted(*args, **kwargs):
        out = original(*args, **kwargs)
        calls[0] += out is not None
        return out

    fused.hbor_call = counted
    spec = importlib.util.spec_from_file_location("main_cnf_rex", ROOT / "main" / "main_cnf_rex.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(argv)
    Path(f"{count_file}.rank{rank}").write_text(str(calls[0]))


class DistributedPathGradientTest(unittest.TestCase):
    def test_training_with_the_fused_sweeps_equals_the_torch_one(self):
        # The defect smearing class of the 16^4 trainings (SU(3), D = 4: the kernel applies), two ranks,
        # same seed: the kernel changes only the sampling sweeps, which agree with the torch ones to
        # rounding, so the losses and the swap decisions of every epoch agree too.
        with tempfile.TemporaryDirectory() as directory:
            base = ["--architecture", "defect-hyper-smearing", "--hyper-smearing-mode", "class", "--domain", "defect",
                    "--D", "4", "--T", "6", "--L", "6", "--beta", "6", "--defect-size", "1", "--time-slice", "3",
                    "--space-slice", "2", "--parameter-min", "0.9", "--parameter-max", "1", "--replicas", "2",
                    "--batch-size", "2", "--orsteps", "1", "--mcmc-steps", "1", "--warmup", "2", "--epochs", "3",
                    "--save-every", "3", "--swap-parity", "even_odd", "--lr", "1e-2", "--seed", "11"]
            runs = {}
            for name, extra in (("torch", []), ("fused", ["--fused-kernels"])):
                out = Path(directory) / name
                count = Path(directory) / f"{name}.count"
                mp.spawn(_fused_worker, args=(2, _free_port(), [*base, *extra, "--output-dir", str(out)],
                                              name == "fused", str(count)), nprocs=2, join=True)
                lines = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
                calls = sum(int(Path(f"{count}.rank{r}").read_text()) for r in range(2))
                runs[name] = (lines, calls)
            (torch_lines, torch_calls), (fused_lines, fused_calls) = runs["torch"], runs["fused"]
            self.assertEqual(torch_calls, 0)
            self.assertGreater(fused_calls, 0)
            self.assertEqual(len(torch_lines), 3)
            for a, b in zip(torch_lines, fused_lines):
                self.assertEqual(a["epoch"], b["epoch"])
                self.assertEqual(a["swap_acceptance_by_edge"], b["swap_acceptance_by_edge"])
                for key in ("losses_by_rung", "pinn_by_rung"):
                    x, y = torch.tensor(a[key], dtype=torch.float64), torch.tensor(b[key], dtype=torch.float64)
                    self.assertTrue(bool(torch.isfinite(x).all()), key)
                    torch.testing.assert_close(y, x, rtol=1e-8, atol=1e-12, msg=key)
            self.assertGreater(max(abs(v) for v in torch_lines[-1]["pinn_by_rung"]), 0.0)
            self.assertGreater(max(line["swap_acceptance"] for line in torch_lines), 0.0)  # swaps were decided

    def test_gradient_equals_the_single_process_one(self):
        mp.spawn(_gradient_worker, args=(3, _free_port()), nprocs=3, join=True)

    def test_runner_trains_saves_and_resumes_on_three_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            base = ["--architecture", "hyper-residual", "--domain", "beta", "--D", "2", "--T", "4", "--L", "4",
                    "--replicas", "3", "--batch-size", "2", "--residual-n-pt", "1", "--residual-mask-mod", "4",
                    "--hidden-dim", "4", "--depth", "1", "--warmup", "1", "--save-every", "1",
                    "--output-dir", str(out)]
            mp.spawn(_main_worker, args=(3, _free_port(), [*base, "--epochs", "1", "--profile-epoch", "1",
                                                           "--pg-direction-batch", "8"]), nprocs=3, join=True)
            self.assertTrue(all((out / f"cnf_checkpoint.rank{r}.pt").exists() for r in range(3)))
            self.assertIn("operators by self", (out / "profile_epoch1.txt").read_text())
            mp.spawn(_main_worker, args=(3, _free_port(), [*base, "--epochs", "2",
                                                           "--checkpoint", str(out / "cnf_checkpoint.pt")]),
                     nprocs=3, join=True)
            lines = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([line["epoch"] for line in lines], [1, 2])
            for line in lines:
                self.assertEqual(len(line["pinn_by_rung"]), 3)
                self.assertEqual(len(line["swap_acceptance_by_edge"]), 2)
            run_info = json.loads((out / "run_info.json").read_text())
            self.assertEqual(run_info["world_size"], 3)
            self.assertIn("--checkpoint", run_info["argv"])
            checkpoint = torch.load(out / "cnf_checkpoint.pt", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertTrue(checkpoint["trainer"]["distributed"])

    def test_cosine_learning_rate_decays_with_the_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            base = ["--architecture", "hyper-residual", "--domain", "beta", "--D", "2", "--T", "4", "--L", "4",
                    "--replicas", "3", "--batch-size", "2", "--hidden-dim", "4", "--depth", "1", "--warmup", "0",
                    "--save-every", "1", "--lr", "1e-2", "--lr-schedule", "cosine", "--lr-min", "1e-4",
                    "--output-dir", str(out)]
            mp.spawn(_main_worker, args=(3, _free_port(), [*base, "--epochs", "4"]), nprocs=3, join=True)
            first = [json.loads(line)["lr"] for line in (out / "metrics.jsonl").read_text().splitlines()]
            # The rate depends on the epoch number only, so a resumed run continues the same curve.
            spec = importlib.util.spec_from_file_location("main_cnf_rex", ROOT / "main" / "main_cnf_rex.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            args = module.parse_args([*base, "--epochs", "4"])
            expected = [module._learning_rate(1e-2, args, epoch) for epoch in range(4)]
            self.assertEqual(first, expected)
            self.assertEqual(expected[0], 1e-2)
            self.assertTrue(all(a > b > 1e-4 for a, b in zip(expected, expected[1:])))
            self.assertAlmostEqual(module._learning_rate(1e-2, args, 4), 1e-4)

    def test_training_starts_from_chains_thermalized_by_pt_obc(self):
        geometry = ["--D", "4", "--T", "4", "--L", "6", "--defect-size", "1", "--time-slice", "2",
                    "--space-slice", "2", "--beta", "6", "--parameter-min", "0", "--parameter-max", "1",
                    "--batch-size", "2", "--orsteps", "1"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp.spawn(_pt_obc_worker, args=(3, _free_port(), [
                "--cpu", "--algorithm", "pt", "--N", "3", *geometry, "--thermal-steps", "2", "--steps", "0",
                "--main-dir", str(root), "--output-dir", str(root / "pt"), "--cfg-cache-dir", str(root / "cfgs"),
            ]), nprocs=3, join=True)
            (cache,) = (root / "cfgs").iterdir()

            out = root / "run"
            train = ["--architecture", "defect-hyper-smearing", "--domain", "defect", *geometry, "--replicas", "3",
                     "--hidden-dim", "4", "--depth", "1", "--warmup", "0", "--epochs", "1", "--output-dir", str(out)]
            mp.spawn(_main_worker, args=(3, _free_port(), [*train, "--init-cfgs", str(cache)]), nprocs=3, join=True)
            (line,) = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(line["plaquette_by_rung"]), 3)
            self.assertTrue(all(0.0 < p < 1.0 for p in line["plaquette_by_rung"]))

            spec = importlib.util.spec_from_file_location("main_cnf_rex", ROOT / "main" / "main_cnf_rex.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config = json.loads((out / "config.json").read_text())
            ladder = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
            cached = torch.load(cache / "rank_001.pt", weights_only=True)["cfgs"]
            torch.testing.assert_close(module._thermalized_cfgs(cache, config, ladder, 1, 3), cached, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "orsteps"):
                module._thermalized_cfgs(cache, dict(config, orsteps=4), ladder, 1, 3)
            with self.assertRaisesRegex(ValueError, "ladder"):
                module._thermalized_cfgs(cache, config, torch.linspace(0.0, 0.5, 3, dtype=torch.float64), 1, 3)


if __name__ == "__main__":
    unittest.main()
