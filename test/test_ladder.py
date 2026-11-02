"""Optimal replica ladder: thermodynamic length, inversion, explicit ladders in the PT, defect action."""

import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import neoqcd.sun_utils as sun
from neoqcd.distributedPT import ParallelTempering, PTConfig
from neoqcd.ladder import (
    cumulative_length,
    is_uniform,
    lengths_on_ladder,
    link_lengths_from_acceptance,
    optimal_ladder,
    parse_ladder,
    predicted_acceptance,
    rex_acceptance,
    thermodynamic_length,
    validate_ladder,
)
from neoqcd.mcmc import DefectMCMCAdapter
from neoqcd.replica_history import rung_moments, summarize_history
from neoqcd.utils import Defect, create_mask
from test.test_cnf_neorex import _free_port, _multirank_worker


class LadderMathTest(unittest.TestCase):
    def test_constant_metric_gives_the_linear_ladder(self):
        c = np.linspace(0.0, 1.0, 14)
        lam = thermodynamic_length(c, np.full(c.size, 4.0))
        np.testing.assert_allclose(lam, 2.0 * c, atol=1e-14)
        np.testing.assert_allclose(optimal_ladder(c, lam, 8), np.linspace(0.0, 1.0, 8), atol=1e-12)

    def test_inversion_of_a_known_length(self):
        # sqrt(Var) = 1 + 3c: Lambda(c) = c + 1.5 c^2 (the trapezoid is exact), Lambda = 2.5.
        c = np.linspace(0.0, 1.0, 201)
        lam = thermodynamic_length(c, (1.0 + 3.0 * c) ** 2)
        np.testing.assert_allclose(lam, c + 1.5 * c ** 2, atol=1e-12)
        n_rep = 7
        ladder = optimal_ladder(c, lam, n_rep)
        t = 2.5 * np.arange(n_rep) / (n_rep - 1)
        np.testing.assert_allclose(ladder, (-1.0 + np.sqrt(1.0 + 6.0 * t)) / 3.0, atol=1e-6)
        self.assertEqual((ladder[0], ladder[-1]), (0.0, 1.0))
        # Every link of the new ladder has the same length, so the same predicted REX acceptance.
        links = lengths_on_ladder(c, lam, ladder)
        np.testing.assert_allclose(links, 2.5 / (n_rep - 1), atol=1e-6)
        np.testing.assert_allclose(rex_acceptance(links), predicted_acceptance(2.5, n_rep), atol=1e-6)

    def test_acceptance_metric_inverts_erfc(self):
        lengths = np.array([0.3, 0.8, 1.4, 0.5])
        acceptance = np.array([math.erfc(l / 2) for l in lengths])
        np.testing.assert_allclose(link_lengths_from_acceptance(acceptance), lengths, atol=1e-12)
        np.testing.assert_allclose(cumulative_length(lengths), [0.0, 0.3, 1.1, 2.5, 3.0], atol=1e-15)
        # The new ladder is evenly spaced in the same interpolated Lambda(c).
        c = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        ladder = optimal_ladder(c, cumulative_length(lengths), 5)
        np.testing.assert_allclose(lengths_on_ladder(c, cumulative_length(lengths), ladder), 0.75, atol=1e-10)
        with self.assertRaises(ValueError):
            link_lengths_from_acceptance([0.5, 0.0])

    def test_predicted_acceptance_matches_the_paper_table(self):
        # L=114: Lambda = 12.391, 14 replicas, 50.0 %.
        self.assertAlmostEqual(predicted_acceptance(12.391, 14), 0.500, places=3)

    def test_zero_length_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            optimal_ladder([0.0, 0.5, 1.0], [0.0, 0.0, 1.0], 4)

    def test_parse_and_validate(self):
        self.assertEqual(parse_ladder("0, 0.2,0.7, 1"), [0.0, 0.2, 0.7, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ladder.json")
            with open(path, "w") as f:
                json.dump({"ladder": [0.0, 0.3, 1.0], "Lambda": 3.0}, f)
            self.assertEqual(parse_ladder(path), [0.0, 0.3, 1.0])
        self.assertEqual(validate_ladder([0.0, 0.3, 1.0], 3, 0.0, 1.0), [0.0, 0.3, 1.0])
        for bad, kwargs in (([0.0, 0.3, 0.3], {}), ([0.0, 1.0], {"n_rep": 3}), ([0.1, 1.0], {"parameter_min": 0.0}),
                            ([0.0, 0.9], {"parameter_max": 1.0}), ([0.0, float("nan"), 1.0], {})):
            with self.assertRaises(ValueError):
                validate_ladder(bad, **kwargs)
        self.assertTrue(is_uniform(np.linspace(0, 1, 8)))
        self.assertFalse(is_uniform([0.0, 0.2, 1.0]))


class RungMomentsTest(unittest.TestCase):
    def test_moments_follow_the_rung_not_the_slot(self):
        rng = np.random.default_rng(0)
        steps, world, batch = 400, 3, 16
        sigma = np.array([1.0, 2.0, 3.0])  # rung r carries values of standard deviation sigma[r]
        labels = np.stack([np.stack([rng.permutation(world) for _ in range(batch)], axis=1) for _ in range(steps)])
        values = rng.normal(size=labels.shape) * sigma[labels] + 10.0 * labels
        moments = rung_moments(labels, values, world, discard_fraction=0.25)
        self.assertEqual(moments["steps_used"], 300)
        for r in range(world):
            self.assertAlmostEqual(moments["mean"][r], 10.0 * r, delta=5 * moments["mean_error"][r])
            self.assertAlmostEqual(moments["variance"][r], sigma[r] ** 2, delta=5 * moments["variance_error"][r])
            # Jackknife over 16 independent slots of 300 independent samples: sigma^2 sqrt(2 / 4800).
            self.assertAlmostEqual(moments["variance_error"][r] / (sigma[r] ** 2 * math.sqrt(2 / 4800)), 1.0, delta=0.5)
        summary = summarize_history(labels, values, world, defect_action=values)
        self.assertEqual(summary["defect_action"]["variance"], rung_moments(labels, values, world)["variance"])


class DefectActionTest(unittest.TestCase):
    def test_defect_action_is_the_exact_derivative(self):
        torch.manual_seed(5)
        D, T, L = 4, 6, 6
        pars = SimpleNamespace(N=3, D=D, orsteps=1, device=torch.device("cpu"), mask=create_mask(D, T, L), defect=None)
        for dsize, time_slice, space_slice in ((1, 2, 2), (2, 0, 3)):
            defect = Defect(D, T, L, dsize=dsize, time_slice=time_slice, space_slice=space_slice)
            adapter = DefectMCMCAdapter(pars, beta=6.0, defect_mask=defect.create_defect_mask(), batch_size=3)
            x = sun.SUN_haar((3, D, T, L, L, L, 3), dtype=torch.cdouble)
            c = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)
            s_d = adapter.defect_action(x, c)
            # Exact for any step: the plaquette weights are polynomials of degree <= 4 in c.
            torch.testing.assert_close(adapter.defect_action(x, c, step=0.01), s_d, rtol=1e-10, atol=1e-10)
            # Same as a tiny central difference of the whole-lattice action difference.
            h = 1e-5
            reference = adapter.action_difference(x, c - h, c + h, full_lattice=True) / (2 * h)
            torch.testing.assert_close(s_d, reference, rtol=1e-7, atol=1e-7)
            # And of the full action itself, up to its configuration-independent part (same coupling
            # for every chain, inside (0, 1) where the Wilson action is defined).
            mid = torch.full((3,), 0.37, dtype=torch.float64)
            # (h = 1e-2: the whole action is ~5e4, so a smaller h only amplifies its rounding.)
            s_mid = adapter.defect_action(x, mid)
            full = (adapter.action(x, mid + 1e-2) - adapter.action(x, mid - 1e-2)) / 2e-2
            torch.testing.assert_close(s_mid - s_mid[0], full - full[0], rtol=0, atol=1e-4)
            self.assertGreater(float(s_d.abs().min()), 0.0)


class PeriodicTranslationTest(unittest.TestCase):
    def test_only_periodic_chains_move_by_one_site(self):
        from neoqcd.utils import translate_periodic_replica

        torch.manual_seed(11)
        D, T, L = 4, 4, 4
        cfgs = sun.SUN_haar((6, D, T, L, L, L, 3), dtype=torch.cdouble)
        c = torch.tensor([1.0, 0.5, 1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
        out = translate_periodic_replica(cfgs, c, 1.0, D, generator=torch.Generator().manual_seed(3))
        for b in range(6):
            if c[b] != 1.0:
                self.assertTrue(torch.equal(out[b], cfgs[b]), b)
                continue
            # The whole field (every link direction) moved by +1 along exactly one lattice axis.
            hits = [mu for mu in range(D) if torch.equal(out[b], torch.roll(cfgs[b], 1, dims=1 + mu))]
            self.assertEqual(len(hits), 1, b)
        # A translation leaves translation-invariant observables unchanged.
        torch.testing.assert_close(sun.plaquette_SUN(out, D, 3), sun.plaquette_SUN(cfgs, D, 3), rtol=0, atol=1e-12)

    def test_every_direction_is_drawn(self):
        from neoqcd.utils import translate_periodic_replica

        D = 4
        cfgs = torch.arange(200 * D * 2 * 2 * 2 * 2, dtype=torch.float64).reshape(200, D, 2, 2, 2, 2, 1, 1)
        out = translate_periodic_replica(cfgs, torch.ones(200, dtype=torch.float64), 1.0, D,
                                         generator=torch.Generator().manual_seed(5))
        seen = {mu for b in range(200) for mu in range(D) if torch.equal(out[b], torch.roll(cfgs[b], 1, dims=1 + mu))}
        self.assertEqual(seen, set(range(D)))


def _explicit_ladder_worker(rank, world_size, port, ladder):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        device = torch.device("cpu")
        config = PTConfig(batch_size=4, parameter_min=0.0, parameter_max=1.0, parameter_name="bc",
                          parameter_dtype=torch.float64, parameter_ladder=ladder)
        pt = ParallelTempering(config, rank, world_size, device)
        assert torch.equal(pt.local_parameters, torch.full((4,), ladder[rank], dtype=torch.float64)), pt.local_parameters
        assert not pt.uniform_ladder and math.isnan(float(pt.delta_parameter))
        torch.manual_seed(rank)
        energy = torch.randn(4, dtype=torch.float64)
        for _ in range(20):
            # S(x; c) = c E(x): exact-action swaps between the explicit couplings.
            actions = torch.tensor(ladder, dtype=torch.float64).view(1, -1) * energy.view(-1, 1)
            pt.step(local_actions=actions)
        gathered = [torch.zeros(4, dtype=torch.float64) for _ in range(world_size)]
        dist.all_gather(gathered, pt.local_parameters)
        labels = torch.stack(gathered, dim=1)  # (B, world_size)
        assert torch.equal(labels.sort(dim=1).values, torch.tensor(ladder, dtype=torch.float64).expand(4, -1)), labels
        # Deterministic even-odd: every pair is proposed on every chain at every step.
        deo = ParallelTempering(PTConfig(batch_size=4, parameter_min=0.0, parameter_max=1.0, parameter_name="bc",
                                         parameter_dtype=torch.float64, parameter_ladder=ladder, swap_parity="even_odd"),
                                rank, world_size, device)
        actions = torch.tensor(ladder, dtype=torch.float64).view(1, -1) * energy.view(-1, 1)
        stats = deo.step(local_actions=actions)
        if rank == 0:
            assert stats["link_proposed"] == [4] * (world_size - 1), stats
            assert stats["n_proposed"] == 4 * (world_size - 1), stats
        uniform = ParallelTempering(PTConfig(batch_size=2, parameter_min=0.0, parameter_max=1.0, parameter_name="bc"),
                                    rank, world_size, device)
        assert uniform.uniform_ladder
        assert torch.equal(uniform.ladder, torch.linspace(0.0, 1.0, world_size, dtype=torch.float64))
    finally:
        dist.destroy_process_group()


class MainLadderTest(unittest.TestCase):
    def test_orex_run_on_an_explicit_ladder_with_translation_records_the_defect_action(self):
        world_size, batch_size, steps = 3, 2, 4
        with tempfile.TemporaryDirectory() as directory:
            output_dir = os.path.join(directory, "run")
            argv = [
                "--cpu", "--algorithm", "orex", "--orex-protocol-steps", "1",
                "--D", "4", "--T", "4", "--L", "4", "--N", "3",
                "--defect-size", "1", "--time-slice", "1", "--space-slice", "1", "--beta", "6",
                "--parameter-ladder", "0,0.3,1", "--batch-size", str(batch_size),
                "--thermal-steps", "1", "--steps", str(steps), "--orsteps", "0", "--periodic-translation",
                "--no-cfg-cache", "--output-dir", output_dir, "--main-dir", directory,
            ]
            mp.spawn(_multirank_worker, args=(world_size, _free_port(), argv), nprocs=world_size, join=True)
            with open(os.path.join(output_dir, "run_meta.json")) as f:
                meta = json.load(f)
            self.assertEqual(meta["swap"]["link_parameter_left"], [0.0, 0.3])
            self.assertEqual(meta["swap"]["link_parameter_right"], [0.3, 1.0])
            history = np.load(os.path.join(output_dir, "replica_history.npz"))
            self.assertEqual(history["defect_action"].shape, (steps, world_size, batch_size))
            self.assertTrue(np.all(np.isfinite(history["defect_action"])))
            np.testing.assert_array_equal(history["parameter_ladder"], [0.0, 0.3, 1.0])
            moments = meta["replica_history"]["defect_action"]
            self.assertEqual(moments["parameter_ladder"], [0.0, 0.3, 1.0])
            self.assertTrue(all(v >= 0 for v in moments["variance"]))


class TrainerDEOTest(unittest.TestCase):
    def test_pg_training_with_even_odd_swaps_on_an_explicit_ladder(self):
        from test.test_cnf_distributed import _main_worker

        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "run")
            argv = ["--architecture", "hyper-residual", "--domain", "beta", "--D", "2", "--T", "4", "--L", "4",
                    "--replicas", "3", "--batch-size", "2", "--residual-n-pt", "1", "--residual-mask-mod", "4",
                    "--hidden-dim", "4", "--depth", "1", "--warmup", "1", "--save-every", "1", "--epochs", "2",
                    "--parameter-min", "0.5", "--parameter-max", "1.0", "--parameter-ladder", "0.5,0.6,1.0",
                    "--swap-parity", "even_odd", "--output-dir", out]
            mp.spawn(_main_worker, args=(3, _free_port(), argv), nprocs=3, join=True)
            with open(os.path.join(out, "metrics.jsonl")) as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(lines[0]["swap_acceptance_by_edge"]), 2)


class ExplicitLadderTest(unittest.TestCase):
    def test_parallel_tempering_runs_on_an_explicit_ladder(self):
        mp.spawn(_explicit_ladder_worker, args=(3, _free_port(), [0.0, 0.2, 1.0]), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
