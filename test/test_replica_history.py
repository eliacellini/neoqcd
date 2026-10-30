"""Round trips, flux and tau_int from a replica label history."""

import math
import unittest

import numpy as np
import torch

from neoqcd.replica_history import (
    replica_observable_series,
    round_trips,
    rung_indices,
    summarize_history,
    tau_int,
)


class ReplicaHistoryTest(unittest.TestCase):
    def test_rung_indices_follow_the_parameter_matrix(self):
        ladder = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
        # Chain 0 holds the identity labelling; chain 1 has rank 0 and rank 3 swapped.
        matrix = torch.stack([ladder, ladder[[3, 1, 2, 0]]])  # (B=2, world_size=4)
        idx = rung_indices(matrix, ladder)
        self.assertEqual(tuple(idx.shape), (4, 2))
        self.assertEqual(idx[:, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(idx[:, 1].tolist(), [3, 1, 2, 0])

    def test_round_trips_of_a_scripted_walker(self):
        # One walker on a 3-rung ladder: first at 0 (t=0), reaches the top at t=2, back at 0 at t=4
        # (round trip of 4 steps), top again at t=6, back at t=7 (round trip of 3 steps), then stays up.
        path = [0, 1, 2, 1, 0, 1, 2, 0, 1, 1]
        labels = np.asarray(path, dtype=np.int64).reshape(-1, 1, 1)
        trips = round_trips(labels, n_rungs=3)
        self.assertEqual(trips["durations"].tolist(), [4, 3])
        self.assertEqual(trips["count"], 2)
        self.assertAlmostEqual(trips["mean_steps"], 3.5)
        self.assertAlmostEqual(trips["rate_per_walker_step"], 2 / 10)
        self.assertEqual(trips["completed_per_walker"].tolist(), [[2]])
        # Flux: rung 0 is always "up"; rung 2 always "down"; rung 1 is visited up at t=1,5,8,9 and down at t=3.
        flux = trips["flux_fraction"]
        self.assertEqual(flux[0], 1.0)
        self.assertEqual(flux[2], 0.0)
        self.assertAlmostEqual(flux[1], 4 / 5)

    def test_round_trips_need_a_visit_to_the_top(self):
        # Bouncing between the two lowest rungs never completes a round trip.
        path = [0, 1, 0, 1, 0]
        trips = round_trips(np.asarray(path).reshape(-1, 1, 1), n_rungs=4)
        self.assertEqual(trips["count"], 0)
        self.assertTrue(math.isnan(trips["mean_steps"]))
        self.assertTrue(math.isnan(trips["flux_fraction"][3]))

    def test_round_trips_pool_walkers_over_ranks_and_chains(self):
        # Two ranks, two chains: labels are a permutation across ranks for every (step, chain).
        a = [0, 1, 0, 1, 0]
        b = [1, 0, 1, 0, 1]
        labels = np.zeros((5, 2, 2), dtype=np.int64)
        labels[:, 0, 0], labels[:, 1, 0] = a, b
        labels[:, 0, 1], labels[:, 1, 1] = b, a
        trips = round_trips(labels, n_rungs=2)
        # Each walker alternates every step; the ones starting at 0 complete trips at t=2 and t=4.
        self.assertEqual(trips["count"], 2 + 2 + 1 + 1)
        self.assertEqual(sorted(trips["durations"].tolist()), [2, 2, 2, 2, 2, 2])
        self.assertEqual(trips["completed_per_walker"].tolist(), [[2, 1], [1, 2]])

    def test_replica_observable_series_picks_the_carrier_of_the_rung(self):
        labels = np.asarray([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])  # (steps=2, ranks=2, B=2)
        values = np.asarray([[[10.0, 11.0], [20.0, 21.0]], [[30.0, 31.0], [40.0, 41.0]]])
        top = replica_observable_series(labels, values, rung=1)
        self.assertEqual(top.tolist(), [[20.0, 11.0], [30.0, 41.0]])
        with self.assertRaises(ValueError):
            replica_observable_series(labels, values, rung=2)

    def test_tau_int_of_an_ar1_series(self):
        rng = np.random.default_rng(1)
        phi = 0.8  # exact tau_int = (1 + phi) / (2 (1 - phi)) = 4.5
        steps, chains = 20000, 4
        noise = rng.standard_normal((steps, chains))
        series = np.zeros_like(noise)
        for t in range(1, steps):
            series[t] = phi * series[t - 1] + noise[t]
        result = tau_int(series)
        self.assertEqual(result["status"], "ok")
        self.assertLess(abs(result["tau_int"] - 4.5), 3 * result["tau_int_error"] + 0.3)
        self.assertEqual(tau_int(series[:4])["status"], "insufficient_statistics")
        self.assertEqual(tau_int(np.ones((100, 2)))["status"], "insufficient_statistics")

    def test_summary_is_json_ready(self):
        rng = np.random.default_rng(2)
        steps, ranks, chains = 64, 3, 2
        labels = np.stack([rng.permutation(ranks) for _ in range(steps * chains)]).reshape(steps, chains, ranks)
        labels = np.transpose(labels, (0, 2, 1))
        plaquette = 0.5 + 0.01 * rng.standard_normal((steps, ranks, chains))
        summary = summarize_history(labels, plaquette, n_rungs=ranks)
        self.assertEqual(summary["walkers"], ranks * chains)
        self.assertEqual(len(summary["flux_fraction"]), ranks)
        self.assertIn("tau_int", summary["plaquette_top"])
        self.assertEqual(summary["plaquette_top"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
