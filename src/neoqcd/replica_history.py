"""Replica label history of a REX run: round-trip times, rung flux and tau_int of a replica observable.

Configurations never leave their rank in the distributed PT: a chain slot (rank, b) is a walker whose
ladder label (rung index) moves through accepted swaps. `rung_indices` reads the labels from rank 0's
parameter matrix after each step; `round_trips` turns the history into the Katzgraber et al.
(J. Stat. Mech. 2006, P03018) diagnostics:

- a walker is "up" after its last visit to rung 0 and "down" after its last visit to the top rung;
- a round trip 0 -> top -> 0 is completed whenever a walker that has already visited rung 0 turns from
  down to up, and its duration is the time since the walker's previous arrival at rung 0;
- the flux fraction f(r) is the share of up walkers among the walkers sitting on rung r; the ideal
  ladder has f linear in r.

`replica_observable_series` picks, for each chain slot b and step, the value of the walker that carries
a given rung, so `tau_int` of that series (Gamma method, `neoqcd.obs.Obs`) is the integrated
autocorrelation time of the observable on that rung in REX steps, the quantity that enters the cost.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from neoqcd.obs import Obs


def rung_indices(parameter_matrix: torch.Tensor, ladder: torch.Tensor) -> torch.Tensor:
    """(world_size, B) int64 rung index of each chain slot from rank 0's (B, world_size) parameter matrix."""
    distances = (parameter_matrix.unsqueeze(-1) - ladder.to(parameter_matrix).view(1, 1, -1)).abs()
    return distances.argmin(dim=-1).T.contiguous()


def round_trips(labels: np.ndarray, n_rungs: int) -> dict:
    """Round-trip statistics of a (steps, world_size, B) rung-index history.

    Returns the completed round-trip durations (steps), their count, mean and standard error, the
    rate per walker and step, the per-rung flux fraction f(r) (nan where no labelled walker sat),
    and the per-walker count of completed round trips.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"labels must have shape (steps, world_size, B), got {labels.shape}")
    steps = labels.shape[0]
    walkers = labels.reshape(steps, -1)  # (steps, W)
    n_walkers = walkers.shape[1]
    top = n_rungs - 1
    direction = np.zeros(n_walkers, dtype=np.int8)  # 0 unlabelled, +1 up, -1 down
    start = np.full(n_walkers, -1, dtype=np.int64)  # step of the last arrival at rung 0
    durations = []
    completed = np.zeros(n_walkers, dtype=np.int64)
    up_count = np.zeros(n_rungs, dtype=np.int64)
    labelled_count = np.zeros(n_rungs, dtype=np.int64)
    for t in range(steps):
        rung = walkers[t]
        at_bottom = rung == 0
        at_top = rung == top
        finished = at_bottom & (direction == -1) & (start >= 0)
        for w in np.flatnonzero(finished):
            durations.append(int(t - start[w]))
            completed[w] += 1
        direction = np.where(at_bottom, 1, np.where(at_top, -1, direction)).astype(np.int8)
        start = np.where(at_bottom, t, start)
        labelled = direction != 0
        np.add.at(labelled_count, rung[labelled], 1)
        np.add.at(up_count, rung[direction == 1], 1)
    durations = np.asarray(durations, dtype=np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        flux = np.where(labelled_count > 0, up_count / np.maximum(labelled_count, 1), np.nan)
    n = int(durations.size)
    mean = float(durations.mean()) if n else float("nan")
    sem = float(durations.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {
        "durations": durations,
        "count": n,
        "mean_steps": mean,
        "sem_steps": sem,
        "rate_per_walker_step": n / float(n_walkers * steps) if steps else float("nan"),
        "flux_fraction": flux,
        "completed_per_walker": completed.reshape(labels.shape[1:]),
    }


def replica_observable_series(labels: np.ndarray, values: np.ndarray, rung: int) -> np.ndarray:
    """(steps, B) values of the walker carrying `rung` for each chain slot b; labels/values (steps, W, B)."""
    labels = np.asarray(labels)
    values = np.asarray(values)
    if labels.shape != values.shape:
        raise ValueError(f"labels {labels.shape} and values {values.shape} must have the same shape")
    hits = labels == rung
    if not np.all(hits.sum(axis=1) == 1):
        raise ValueError(f"rung {rung} must sit on exactly one rank per step and chain slot")
    return np.sum(np.where(hits, values, 0.0), axis=1)


def tau_int(series: np.ndarray, S: float = 2.0, min_steps: int = 8) -> dict:
    """Gamma-method mean, error, tau_int (steps) and its error of a (steps, B) series; B chains, one ensemble."""
    series = np.asarray(series, dtype=np.float64)
    empty = {"value": float("nan"), "error": float("nan"), "tau_int": float("nan"),
             "tau_int_error": float("nan"), "window": float("nan"), "status": "insufficient_statistics"}
    if series.ndim != 2 or series.shape[0] < min_steps or not np.all(np.isfinite(series)):
        return empty
    if np.ptp(series) == 0.0:
        return empty
    obs = Obs([series[:, b] for b in range(series.shape[1])], [f"rex|chain_{b}" for b in range(series.shape[1])])
    obs.gamma_method(S=S)
    ensemble = next(iter(obs.e_tauint))
    return {
        "value": float(obs.value),
        "error": float(obs.dvalue),
        "tau_int": float(obs.e_tauint[ensemble]),
        "tau_int_error": float(obs.e_dtauint[ensemble]),
        "window": float(obs.e_windowsize[ensemble]),
        "status": "ok",
    }


def summarize_history(labels: np.ndarray, plaquette: np.ndarray | None, n_rungs: int) -> dict:
    """JSON-ready summary: round trips, flux and plaquette tau_int on the bottom and top rungs."""
    trips = round_trips(labels, n_rungs)
    summary = {
        "steps": int(labels.shape[0]),
        "walkers": int(labels.shape[1] * labels.shape[2]),
        "round_trips": int(trips["count"]),
        "round_trip_steps": trips["mean_steps"],
        "round_trip_steps_sem": trips["sem_steps"],
        "round_trips_per_walker_step": trips["rate_per_walker_step"],
        "flux_fraction": [float(f) for f in trips["flux_fraction"]],
    }
    if plaquette is not None:
        for name, rung in (("bottom", 0), ("top", n_rungs - 1)):
            summary[f"plaquette_{name}"] = tau_int(replica_observable_series(labels, plaquette, rung))
    return summary
