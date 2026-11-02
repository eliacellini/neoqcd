"""Optimal replica ladder (Syed et al. 2021, 2024), as in the appendix "Optimal ladder" of the O-REX paper.

The local metric of the ladder is the variance of the defect action S_D = dS/dc, Var_c(S_D). Its square
root integrated over the coupling is the thermodynamic length

    Lambda(c) = int_{c_min}^{c} dc' sqrt(Var_c'(S_D)),    Lambda = Lambda(c_max).

The optimal ladder with n_rep replicas puts the couplings evenly in Lambda(c),
Lambda(c_r) = Lambda r / (n_rep - 1), r = 0, ..., n_rep - 1 (increasing c, the code's convention). On it
every link of standard REX has the same acceptance erfc[Lambda / (2 (n_rep - 1))] (Gaussian work,
<Delta> = Lambda^2 / (n_rep - 1)^2).

Lambda(c) is the cumulative trapezoid of sqrt(Var) on the measured couplings, interpolated with a
monotone cubic (PCHIP) and inverted by root finding: the ladder is increasing for any data, its links
have exactly equal lengths in the interpolated Lambda(c) used for the predictions, and on smooth data it
agrees with the paper's cubic spline.

The same inversion works with the length of each link of a measured ladder, 2 erfc^-1(A_link), from the
link acceptances A_link of any run (O-REX, NEO-REX): this equalizes the acceptance of that algorithm
rather than the one of standard REX (the "O-REX-aware" ladder of the paper).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _increasing(name: str, values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
        raise ValueError(f"{name} must hold at least two finite, strictly increasing values, got {values.tolist()}")
    return values


def thermodynamic_length(parameters: Sequence[float], variances: Sequence[float]) -> np.ndarray:
    """Lambda(c) at each measured coupling: cumulative trapezoid of sqrt(Var_c(S_D)), 0 at the first one."""
    c = _increasing("parameters", parameters)
    var = np.asarray(variances, dtype=np.float64).reshape(-1)
    if var.shape != c.shape or not np.all(np.isfinite(var)) or np.any(var < 0):
        raise ValueError(f"variances must be finite, non-negative and one per parameter, got {var.tolist()}")
    root = np.sqrt(var)
    return np.concatenate(([0.0], np.cumsum(0.5 * (root[1:] + root[:-1]) * np.diff(c))))


def link_lengths_from_acceptance(acceptance: Sequence[float]) -> np.ndarray:
    """Length of each link from its acceptance, 2 erfc^-1(A), exact for Gaussian work."""
    from scipy.special import erfcinv

    acceptance = np.asarray(acceptance, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(acceptance)) or np.any(acceptance <= 0) or np.any(acceptance > 1):
        raise ValueError(f"link acceptances must lie in (0, 1], got {acceptance.tolist()}")
    return 2.0 * erfcinv(acceptance)


def cumulative_length(link_lengths: Sequence[float]) -> np.ndarray:
    """Lambda at the couplings of a ladder from the lengths of its links, 0 at the first coupling."""
    lengths = np.asarray(link_lengths, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths < 0):
        raise ValueError(f"link lengths must be finite and non-negative, got {lengths.tolist()}")
    return np.concatenate(([0.0], np.cumsum(lengths)))


def optimal_ladder(parameters: Sequence[float], lengths: Sequence[float], n_rep: int) -> np.ndarray:
    """n_rep couplings evenly spaced in Lambda between the first and the last measured coupling.

    `lengths` is Lambda(c) at `parameters` (from thermodynamic_length or cumulative_length); it must be
    strictly increasing, i.e. every interval between measured couplings must have a non-zero length.
    """
    from scipy.optimize import brentq

    c = _increasing("parameters", parameters)
    lam = _increasing("thermodynamic lengths", lengths)
    if lam.shape != c.shape:
        raise ValueError(f"one length per parameter: {lam.size} lengths for {c.size} parameters")
    n_rep = int(n_rep)
    if n_rep < 2:
        raise ValueError(f"n_rep must be >= 2, got {n_rep}")
    curve = _length_curve(c, lam)
    targets = lam[0] + (lam[-1] - lam[0]) * np.arange(n_rep) / (n_rep - 1)
    ladder = [c[0]]
    for target in targets[1:-1]:
        # Lambda(c) is increasing: bracket the root between the measured couplings around it.
        k = int(np.searchsorted(lam, target))
        lo, hi = c[max(k - 1, 0)], c[min(k, c.size - 1)]
        ladder.append(lo if lo == hi else brentq(lambda x: float(curve(x)) - target, lo, hi, xtol=1e-14, rtol=1e-14))
    ladder.append(c[-1])
    return _increasing("ladder", ladder)


def _length_curve(c: np.ndarray, lam: np.ndarray):
    """Monotone cubic (PCHIP) interpolant of Lambda(c) through the measured points."""
    from scipy.interpolate import PchipInterpolator

    return PchipInterpolator(c, lam)


def lengths_on_ladder(parameters: Sequence[float], lengths: Sequence[float], ladder: Sequence[float]) -> np.ndarray:
    """Length of each link of `ladder`, from Lambda(c) interpolated (PCHIP) on the measured couplings."""
    c = _increasing("parameters", parameters)
    lam = _increasing("thermodynamic lengths", lengths)
    ladder = _increasing("ladder", ladder)
    if ladder[0] < c[0] - 1e-12 or ladder[-1] > c[-1] + 1e-12:
        raise ValueError(f"ladder {ladder.tolist()} leaves the measured range [{c[0]}, {c[-1]}]")
    return np.diff(_length_curve(c, lam)(np.clip(ladder, c[0], c[-1])))


def rex_acceptance(link_length) -> np.ndarray:
    """Standard REX acceptance of a link of thermodynamic length l: erfc(l / 2)."""
    return np.vectorize(lambda l: math.erfc(0.5 * l), otypes=[np.float64])(np.asarray(link_length, dtype=np.float64))


def predicted_acceptance(total_length: float, n_rep: int) -> float:
    """Standard REX acceptance of every link of the optimal ladder, erfc[Lambda / (2 (n_rep - 1))]."""
    return math.erfc(float(total_length) / (2.0 * (int(n_rep) - 1)))


def validate_ladder(
    ladder: Sequence[float],
    n_rep: Optional[int] = None,
    parameter_min: Optional[float] = None,
    parameter_max: Optional[float] = None,
    atol: float = 1e-12,
) -> list[float]:
    """Check an explicit ladder (length, strictly increasing, end points) and return it as floats."""
    values = _increasing("ladder", ladder)
    if n_rep is not None and values.size != int(n_rep):
        raise ValueError(f"the ladder has {values.size} couplings for {int(n_rep)} replicas")
    if parameter_min is not None and abs(values[0] - float(parameter_min)) > atol:
        raise ValueError(f"the ladder starts at {values[0]}, not at parameter_min={parameter_min}")
    if parameter_max is not None and abs(values[-1] - float(parameter_max)) > atol:
        raise ValueError(f"the ladder ends at {values[-1]}, not at parameter_max={parameter_max}")
    return [float(v) for v in values]


def parse_ladder(spec: str) -> list[float]:
    """Couplings from a comma-separated list or from a JSON file (a list, or a dict with key "ladder")."""
    spec = str(spec).strip()
    path = Path(spec)
    if path.is_file():
        data = json.loads(path.read_text())
        values = data["ladder"] if isinstance(data, dict) else data
    else:
        values = [v for v in spec.split(",") if v.strip()]
    return validate_ladder([float(v) for v in values])


def is_uniform(ladder: Sequence[float], atol: float = 1e-12) -> bool:
    values = np.asarray(ladder, dtype=np.float64).reshape(-1)
    return bool(np.allclose(values, np.linspace(values[0], values[-1], values.size), rtol=0, atol=atol))
