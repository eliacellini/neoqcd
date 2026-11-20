"""Tests of the fermionic PG trainers (neoqcd.fermions.pg): single process on `FermionREX`, distributed
on `ParallelTempering` + `FermionREXTarget` (Gloo ranks on the CPU).

- single process: an epoch's outputs, a checkpoint round trip through torch.save / weights_only
  load, and a bitwise resume (losses, residuals, swaps, field parameters);
- <S_pf>/n = 1 on every rung after the swaps (the phi of a chain's last trajectory stays with its
  links; statistical, 2D 4 x 4);
- distributed (3 Gloo ranks, NEOQCD_SKIP_GLOO=1 skips): the field is broadcast, the all-reduced
  gradient equals the single-process gradient of the gathered chains, phi and labels, and a run
  resumed from the rank shards repeats the uninterrupted one bitwise.
CPU, complex128.
"""

import copy
import io
import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neoqcd.fermions._gauge_cnf import CloverChairLinkDrift
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.pg import DistributedFermionCNFREXTrainer, FermionCNFREXTrainer, fermion_path_gradient_loss
from neoqcd.fermions.pseudofermion import TwoFlavourWilson

G2 = Geometry(2, 4, 4, 3)
VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))
PHYSICS = dict(geometry=G2, beta=5.0, kappa=0.18, tau=0.5, n_steps=2, n_gauge_substeps=2, solver_tol=1e-12)
TIMING = ("sweep_s", "loss_s", "gauge_loss_s", "fermion_loss_s", "peak_memory_gb")


def line_defect(geometry, size=2, time_slice=1, space_slice=1):
    m = torch.zeros((geometry.D,) + geometry.lattice, dtype=torch.float64)
    m[(0, time_slice, slice(space_slice, space_slice + size)) + (space_slice,) * (geometry.D - 2)] = 1.0
    return torch.stack((1.0 - m, m))


def make_field(seed, scale=0.05):
    torch.manual_seed(seed)
    field = CloverChairLinkDrift(2, hidden_dim=4, depth=1, loop_max=1, zero_init=False, time_range=(0.0, 1.0))
    field = field.to(dtype=torch.float64)
    with torch.no_grad():
        field.output_layer.weight.mul_(scale)
    return field


def roundtrip(obj):
    """torch.save / torch.load(weights_only=True): what a checkpoint keeps."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def comparable(metrics):
    return {k: v for k, v in metrics.items() if k not in TIMING}


def single_trainer(field, seed=5, batch_size=2, ladder=(0.0, 0.5, 1.0)):
    optimizer = torch.optim.Adam(field.parameters(), lr=1e-2)
    return FermionCNFREXTrainer(field, optimizer, defect_mask=line_defect(G2), ladder=list(ladder), batch_size=batch_size,
                                seed=seed, field_defect_mask=line_defect(G2), **PHYSICS)


class SingleProcessTrainerTest(unittest.TestCase):
    def test_epoch_outputs_and_bitwise_resume(self):
        field = make_field(1)
        trainer = single_trainer(field)
        trainer.thermalize(2)
        trainer.warmup(1)
        first = trainer.step()
        for key in ("losses_by_rung", "pinn_by_rung", "pinn_gauge_by_rung", "pinn_fermion_by_rung", "pinn_cov_by_rung",
                    "spf_per_component_by_rung", "plaquette_by_rung"):
            self.assertEqual(len(first[key]), 3, key)
            self.assertTrue(all(v == v for v in first[key]), key)
        self.assertEqual(len(first["swap_acceptance_by_edge"]), 2)
        self.assertEqual(first["loss_solves"], 2)
        self.assertGreater(first["trajectory_solves"], 2)
        self.assertTrue(first["optimizer_updated"])
        self.assertEqual(first["unconverged_solves"], 0)
        self.assertLess(first["max_residual"], 1e-10)

        snapshot = roundtrip({"field": field.state_dict(), "state": trainer.state_dict(),
                              "shard": trainer.shard_state_dict()})
        rest = [comparable(trainer.step()) for _ in range(2)]
        final = copy.deepcopy(field.state_dict())

        other = make_field(99)
        resumed = single_trainer(other)
        other.load_state_dict(snapshot["field"])
        resumed.load_state_dict(snapshot["state"], snapshot["shard"])
        again = [comparable(resumed.step()) for _ in range(2)]
        self.assertEqual(rest, again)
        for name, value in other.state_dict().items():
            self.assertTrue(torch.equal(value, final[name]), name)
        self.assertTrue(torch.equal(resumed.x, trainer.x))

    def test_defect_masks_must_agree(self):
        with self.assertRaisesRegex(ValueError, "differ"):
            FermionCNFREXTrainer(make_field(1), None, defect_mask=line_defect(G2), ladder=[0.0, 1.0],
                                 field_defect_mask=line_defect(G2, time_slice=2), **PHYSICS)

    def test_pseudofermion_action_per_component_is_one_after_the_swaps(self):
        # the field does not touch the sampling: sweep, then the loss's X solve gives S_pf(U, phi)
        trainer = single_trainer(make_field(2), seed=8, batch_size=4)
        trainer.thermalize(10)
        trainer.warmup(5)
        values = [[] for _ in range(3)]
        for _ in range(25):
            trainer._sweep()
            t = trainer.local_parameters().to(torch.float64)
            S = trainer._loss(trainer.x, t)["S_pf"].detach() / trainer.fermion.n_components
            labels = trainer.labels()
            for k in range(3):
                values[k].append(S[labels == k])
        for k in range(3):
            v = torch.cat(values[k])
            mean, err = float(v.mean()), float(v.std() / v.numel() ** 0.5)
            if VERBOSE:
                print(f"rung {k}: <S_pf>/n = {mean:.4f} +- {err:.4f} ({v.numel()} values)")
            self.assertLess(abs(mean - 1.0), 4.0 * err)
            self.assertLess(err, 0.03)


# ---- distributed --------------------------------------------------------------------------------
def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _environment(rank, world_size, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank),
                      WORLD_SIZE=str(world_size), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
    torch.set_num_threads(1)


def _distributed_trainer(field, rank, world_size):
    optimizer = torch.optim.Adam(field.parameters(), lr=1e-2)
    return DistributedFermionCNFREXTrainer(
        field, optimizer, rank=rank, world_size=world_size, defect_mask=line_defect(G2),
        ladder=torch.linspace(0.0, 1.0, world_size, dtype=torch.float64), batch_size=2, seed=11,
        field_defect_mask=line_defect(G2), **PHYSICS)


def _worker(rank, world_size, port):
    _environment(rank, world_size, port)
    dist.init_process_group("gloo")
    try:
        field = make_field(7 + rank)  # different fields per rank: the trainer broadcasts rank 0's
        trainer = _distributed_trainer(field, rank, world_size)
        reference_field = make_field(7)
        for a, b in zip(field.parameters(), reference_field.parameters()):
            assert torch.equal(a, b), "the field was not broadcast from rank 0"
        trainer.thermalize(2)
        trainer.warmup(3)
        labels = trainer._gather(trainer._labels())
        assert bool((labels != torch.arange(world_size).repeat_interleave(2)).any()), "no swap was accepted"

        # all-reduced gradient = single-process gradient of the gathered chains, phi and labels
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer._backward()
        distributed = [p.grad.clone() for p in field.parameters()]
        configs = trainer._gather(trainer.x)
        times = trainer._gather(trainer.pt.local_parameters)
        phi = trainer._gather(trainer.fermion.phi)
        term = TwoFlavourWilson(PHYSICS["kappa"], G2, solver_tol=PHYSICS["solver_tol"])
        term.phi = phi
        reference = copy.deepcopy(field)
        reference.zero_grad(set_to_none=True)
        fermion_path_gradient_loss(reference, trainer.action, term, configs, times)["loss"].mean().backward()
        for grad, expected in zip(distributed, (p.grad for p in reference.parameters())):
            torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-13)
        assert any(float(g.abs().max()) > 0 for g in distributed)

        # bitwise resume from the shards
        trainer.step()
        snapshot = roundtrip({"field": field.state_dict(), "state": trainer.state_dict(),
                              "shard": trainer.shard_state_dict()})
        rest = [comparable(trainer.step()) for _ in range(2)]
        final = copy.deepcopy(field.state_dict())
        other = make_field(50 + rank)
        resumed = _distributed_trainer(other, rank, world_size)
        other.load_state_dict(snapshot["field"])
        resumed.load_state_dict(snapshot["state"], snapshot["shard"])
        again = [comparable(resumed.step()) for _ in range(2)]
        assert rest == again, (rest, again)
        for name, value in other.state_dict().items():
            assert torch.equal(value, final[name]), name
        assert torch.equal(resumed.x, trainer.x)
        assert len(rest[0]["losses_by_rung"]) == world_size and len(rest[0]["swap_acceptance_by_edge"]) == world_size - 1
    finally:
        dist.destroy_process_group()


@unittest.skipIf(os.environ.get("NEOQCD_SKIP_GLOO"), "NEOQCD_SKIP_GLOO set")
class DistributedTrainerTest(unittest.TestCase):
    def test_gradient_and_bitwise_resume_on_three_ranks(self):
        mp.spawn(_worker, args=(3, _free_port()), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
