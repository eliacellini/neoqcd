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
from test.test_cnf_neorex import _free_port

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


class DistributedPathGradientTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
