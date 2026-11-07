"""Tests of the fermionic REX on the boundary-condition defect (phase 5 of fermions_plan.md):
N_f = 2 Wilson fermions, defect on the gauge action only, replicas updated with the fermionic HMC,
swaps from the gauge-only local action difference (neoqcd.fermions.rex).

CPU, complex128, fixed seeds. 2D 4 x 4 SU(3) (and 2^3 x 4 SU(3) for the action identities), a
defect of two time links at t = 0. Set NEOQCD_TEST_VERBOSE=1 to print the numbers, and
NEOQCD_SKIP_GLOO=1 to skip the three-rank Gloo test.
"""

import json
import math
import os
import socket
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neoqcd.distributedPT import ParallelTempering, PTConfig
from neoqcd.fermions._gauge import SUN_haar, rung_indices, translate_periodic_replica
from neoqcd.fermions.action import QCDAction
from neoqcd.fermions.exact import log_det_two_flavour
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.rex import FermionREX, FermionREXTarget, distributed_rex_step, swap_action_difference

G2 = Geometry(2, 4, 4, 3)  # 2D 4 x 4, SU(3)
G4 = Geometry(4, 4, 2, 3)  # 2^3 x 4, SU(3)
BETA, KAPPA = 5.0, 0.18
LADDER = (0.0, 0.5, 1.0)  # rung 0 = open, top rung = periodic
VERBOSE = bool(os.environ.get("NEOQCD_TEST_VERBOSE"))


def defect_mask(g):
    """Two time links at t = 0 (x = 0, 1, other coordinates 0), layout of utils.Defect."""
    d = torch.zeros((g.D,) + g.lattice, dtype=torch.double)
    d[(0, 0, slice(0, 2)) + (0,) * (g.D - 2)] = 1.0
    return torch.stack((1.0 - d, d), dim=0)


def haar_links(g, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    return SUN_haar(g.link_shape(batch)[:-1], generator=gen), gen


def chain_mean_error(series):
    """Mean and error of a (steps, chains) series from the spread of the chain means."""
    m = np.asarray(series).mean(axis=0)
    return float(m.mean()), float(m.std(ddof=1) / math.sqrt(m.size))


class SwapActionDifferenceTest(unittest.TestCase):
    """1. The gauge-only local difference used by the swap equals S(U; bc_new) - S(U; bc_old) from
    two full QCDAction.action calls (S_pf cancels exactly), and costs no solve."""

    def test_local_difference_equals_full_action_difference(self):
        for g in (G2, G4):
            with self.subTest(geometry=g):
                B = 3
                U, gen = haar_links(g, B, 31)
                qcd = QCDAction(g, "bc", beta=BETA, kappa=KAPPA, bc=1.0, defect_mask=defect_mask(g))
                target = FermionREXTarget(qcd)
                qcd.refresh(U, gen)
                ladder = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
                current = ladder.clone()  # one configuration per bc
                target.set_parameter(current)
                solves = qcd.fermion.stats["solves"]
                table = target.action_table(U, ladder)
                pairs = [(1.0, 0.5), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0)]
                local = [target.action_difference(U, a, b) for a, b in pairs]
                self.assertEqual(qcd.fermion.stats["solves"], solves)  # no solve for the table/differences
                for (a, b), dS in zip(pairs, local):
                    full = target.action(U, b) - target.action(U, a)
                    err = (dS - full).abs().max().item()
                    if VERBOSE:
                        print(f"\n{g}: bc {a} -> {b}: dS = {dS.tolist()}, |local - full| = {err:.1e}")
                    self.assertLess(err, 1e-10)
                    self.assertGreater(dS.abs().max().item(), 1e-3)  # the defect changes the action
                for k, v in enumerate(ladder.tolist()):
                    full = target.action(U, torch.full((B,), v, dtype=torch.float64)) - target.action(U, current)
                    self.assertLess((table[:, k] - full).abs().max().item(), 1e-10)
                # column k at the configuration's own bc is exactly zero
                self.assertTrue(bool((table[torch.arange(B), torch.arange(B)] == 0).all()))
                # swap dS of configurations 0 (bc 1) and 1 (bc 0.5) from the table and from full actions
                dS_swap = (table[0, 1] - table[0, 0]) + (table[1, 0] - table[1, 1])
                S = lambda i, bc: target.action(U[i:i + 1], bc)[0]
                full_swap = (S(0, 0.5) + S(1, 1.0)) - (S(0, 1.0) + S(1, 0.5))
                self.assertLess(abs(float(dS_swap - full_swap)), 1e-10)

    def test_periodic_translation_keeps_the_weight(self):
        """The translation of the bc = 1 replica is exact: S_g(bc = 1) and log det(D^dagger D)
        (antiperiodic in time) are invariant under a one-site shift in every direction."""
        U, _ = haar_links(G2, 2, 41)
        qcd = QCDAction(G2, "bc", beta=BETA, kappa=KAPPA, bc=1.0, defect_mask=defect_mask(G2))
        S0 = qcd.gauge_action(U, 1.0)
        L0 = log_det_two_flavour(U, KAPPA, G2, method="assembled")
        for mu in range(G2.D):
            V = torch.roll(U, shifts=1, dims=2 + mu)
            self.assertLess((qcd.gauge_action(V, 1.0) - S0).abs().max().item(), 1e-10)
            self.assertLess((log_det_two_flavour(V, KAPPA, G2, method="assembled") - L0).abs().max().item(), 1e-9)
        # only bc = 1 configurations move
        V = translate_periodic_replica(U, torch.tensor([1.0, 0.5], dtype=torch.float64), 1.0, G2.D)
        self.assertTrue(torch.equal(V[1], U[1]))
        self.assertFalse(torch.equal(V[0], U[0]))


_RUN = {}


def rex_run():
    """The shared REX run of tests 2 and 3: 16 ladders of 3 rungs, 15 + 80 steps (~12 s)."""
    if "rex" not in _RUN:
        rex = FermionREX(G2, BETA, KAPPA, defect_mask(G2), LADDER, n_chains=16, tau=1.0, n_steps=2,
                         n_gauge_substeps=3, swap_parity="alternating", periodic_translation=True)
        gen = torch.Generator().manual_seed(2026)
        stats = rex.run(95, gen)
        _RUN["rex"] = (rex, stats)
    return _RUN["rex"]


def plain_hmc(bc, n_chains, n_therm, n_meas, seed):
    """Plain fermionic HMC at fixed bc: (steps, chains) plaquette and exp(-dH)."""
    qcd = QCDAction(G2, "bc", beta=BETA, kappa=KAPPA, bc=bc, defect_mask=defect_mask(G2), n_gauge_substeps=3)
    target = FermionREXTarget(qcd, tau=1.0, n_steps=2)
    gen = torch.Generator().manual_seed(seed)
    U = torch.eye(G2.N, dtype=torch.cdouble).expand(G2.link_shape(n_chains)).clone()
    plaq, x = [], []
    for i in range(n_therm + n_meas):
        U, _, _, st = target.trajectory(U, gen)
        if i >= n_therm:
            plaq.append(st["plaquette"].numpy())
            x.append(st["exp_minus_dH"].numpy())
    return np.stack(plaq), np.stack(x)


class REXDetailedBalanceTest(unittest.TestCase):
    """2. REX with fermionic HMC against plain fermionic HMC at fixed bc: the plaquette on the
    periodic (bc = 1) and open (bc = 0) rungs within 4 sigma, <exp(-dH)> = 1 on every rung, swap
    acceptances in (0.1, 0.9). 16 independent ladders; errors from the spread of the chain means."""

    THERM = 15

    def test_rung_plaquette_matches_plain_hmc(self):
        rex, _ = rex_run()
        labels, plaq, traj_labels, x = rex.history_arrays(start=self.THERM)
        lines = []
        for rung, bc in ((2, 1.0), (0, 0.0)):
            p_rex, e_rex = chain_mean_error(rex.rung_series(plaq, labels, rung))
            ref_plaq, ref_x = plain_hmc(bc, 16, self.THERM, 80, seed=7 + rung)
            p_ref, e_ref = chain_mean_error(ref_plaq)
            sigma = abs(p_rex - p_ref) / math.hypot(e_rex, e_ref)
            lines.append(f"bc={bc}: <P> REX {p_rex:.4f}({e_rex:.4f})  plain HMC {p_ref:.4f}({e_ref:.4f})  "
                         f"{sigma:.2f} sigma; plain <exp(-dH)> {ref_x.mean():.4f}")
            self.assertLess(sigma, 4.0, msg="\n".join(lines))
        # the open and periodic rungs differ, so the comparison does resolve the labels
        p_top, e_top = chain_mean_error(rex.rung_series(plaq, labels, 2))
        p_bot, e_bot = chain_mean_error(rex.rung_series(plaq, labels, 0))
        self.assertGreater(abs(p_top - p_bot), 6 * math.hypot(e_top, e_bot))
        for rung in range(rex.n_rungs):
            m, e = chain_mean_error(rex.rung_series(x, traj_labels, rung))
            lines.append(f"rung {rung} (bc={LADDER[rung]}): <P> {chain_mean_error(rex.rung_series(plaq, labels, rung))[0]:.4f}"
                         f"  <exp(-dH)> {m:.4f}({e:.4f})")
            self.assertLess(abs(m - 1.0), 4 * e, msg="\n".join(lines))
        acc = rex.swap_acceptance
        lines.append(f"swap acceptance per pair {['%.3f' % a for a in acc]}")
        if VERBOSE:
            print("\n" + "\n".join(lines))
        for a in acc:
            self.assertGreater(a, 0.1, msg="\n".join(lines))
            self.assertLess(a, 0.9, msg="\n".join(lines))


class ReplicaHistoryTest(unittest.TestCase):
    """3. Labels recorded as a permutation of the rungs in every ladder at every step; round
    trips and flux counted by replica_history.summarize_history."""

    def test_history_and_round_trips(self):
        rex, stats = rex_run()
        labels, plaq, traj_labels, _ = rex.history_arrays()
        self.assertEqual(labels.shape, (95, 3, 16))
        self.assertTrue(np.all(np.sort(labels, axis=1) == np.arange(3).reshape(1, 3, 1)))
        # labels after step t are the trajectory labels of step t + 1
        self.assertTrue(np.array_equal(labels[:-1], traj_labels[1:]))
        # the per-step counts add up to the totals
        self.assertEqual(sum(s["link_accepted"][0] for s in stats), int(rex.link_accepted[0]))
        summary = rex.summarize()
        if VERBOSE:
            print(f"\nsummary: round trips {summary['round_trips']}, mean {summary['round_trip_steps']:.2f} "
                  f"({summary['round_trip_steps_sem']:.2f}) steps, flux {summary['flux_fraction']}, "
                  f"tau_int(P) bottom {summary['plaquette_bottom']['tau_int']:.2f}, "
                  f"top {summary['plaquette_top']['tau_int']:.2f}")
        self.assertGreaterEqual(summary["round_trips"], 1)
        self.assertEqual(summary["walkers"], 48)
        self.assertEqual(summary["flux_fraction"][0], 1.0)
        self.assertEqual(summary["flux_fraction"][-1], 0.0)
        self.assertEqual(summary["plaquette_top"]["status"], "ok")


# ---- 4. three Gloo ranks ------------------------------------------------------------------------

def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _distributed_worker(rank, world_size, port, out_path):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        torch.manual_seed(100 + rank)  # rank 0's swap uniforms come from the global generator
        B = 2
        cfg = PTConfig(batch_size=B, parameter_min=0.0, parameter_max=1.0, parameter_name="bc",
                       acceptance_rule="exact", swap_parity="alternating")
        pt = ParallelTempering(cfg, rank, world_size, torch.device("cpu"))
        qcd = QCDAction(G2, "bc", beta=BETA, kappa=KAPPA, bc=pt.local_parameters, defect_mask=defect_mask(G2),
                        n_gauge_substeps=3)
        target = FermionREXTarget(qcd, tau=1.0, n_steps=2)
        gen = torch.Generator().manual_seed(10 + rank)
        U = torch.eye(G2.N, dtype=torch.cdouble).expand(G2.link_shape(B)).clone()
        history, accepted, proposed, max_err = [], 0, 0, 0.0
        initial = None if rank != 0 else rung_indices(pt.parameter_matrix, pt.ladder).clone()
        for step in range(8):
            U, stats, hmc_stats = distributed_rex_step(pt, target, U, gen)
            if step in (0, 7):
                # relative table against two full actions at fixed (U, phi of the last trajectory)
                table = target.action_table(U, pt.ladder)
                full0 = target.action(U, target.parameter)
                for k, v in enumerate(pt.ladder.tolist()):
                    full = target.action(U, torch.full((B,), v, dtype=torch.float64)) - full0
                    max_err = max(max_err, (table[:, k] - full).abs().max().item())
            if rank == 0:
                history.append(rung_indices(pt.parameter_matrix, pt.ladder).tolist())
                accepted += stats["n_accepted"]
                proposed += stats["n_proposed"]
        # every rank's labels agree with rank 0's matrix
        mine = target.parameter.to(torch.float64).reshape(-1)
        gathered = [torch.zeros_like(mine) for _ in range(world_size)] if rank == 0 else None
        dist.gather(mine, gather_list=gathered, dst=0)
        err = torch.tensor([max_err], dtype=torch.float64)
        dist.all_reduce(err, op=dist.ReduceOp.MAX)
        if rank == 0:
            matrix_ok = bool(torch.equal(torch.stack(gathered, dim=1), pt.parameter_matrix))
            with open(out_path, "w") as f:
                json.dump({"history": history, "initial": initial.tolist(), "accepted": accepted,
                           "proposed": proposed, "max_table_error": float(err.item()),
                           "matrix_ok": matrix_ok}, f)
    finally:
        dist.destroy_process_group()


@unittest.skipIf(os.environ.get("NEOQCD_SKIP_GLOO"), "NEOQCD_SKIP_GLOO set")
class DistributedREXTest(unittest.TestCase):
    """4. distributedPT.ParallelTempering on three Gloo ranks with FermionREXTarget and fermionic HMC:
    it runs, the bc labels get swapped, the relative action table equals full action differences."""

    def test_three_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "rex.json"
            mp.spawn(_distributed_worker, args=(3, _free_port(), str(out)), nprocs=3, join=True)
            result = json.loads(out.read_text())
        if VERBOSE:
            print(f"\nGloo REX: swaps {result['accepted']}/{result['proposed']}, "
                  f"max |table - full| = {result['max_table_error']:.1e}, labels {result['history'][-1]}")
        self.assertTrue(result["matrix_ok"])
        self.assertLess(result["max_table_error"], 1e-10)
        self.assertGreater(result["accepted"], 0)
        self.assertTrue(any(h != result["initial"] for h in result["history"]))
        for h in result["history"]:  # each chain slot b carries a permutation of the rungs
            self.assertTrue(np.all(np.sort(np.asarray(h), axis=0) == np.arange(3).reshape(3, 1)))


if __name__ == "__main__":
    unittest.main()
