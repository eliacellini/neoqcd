"""Gamma-method analysis for standalone evaluation observable CSV files."""

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from neoqcd.obs import Obs


OBSERVABLES_HEADER = [
    "trajectory", "measurement", "sample", "coupling", "work", "heat", "logJ",
    "action", "plaquette", "plaquette2",
]
REPORT_HEADER = [
    "measurement", "coupling", "S", "observable", "value", "error", "error_of_error",
    "tau_int", "tau_int_error", "window", "status",
]
_BASE_FIELDS = ("work", "heat", "logJ", "action", "plaquette", "plaquette2")


def _read_observables(path):
    by_key = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != OBSERVABLES_HEADER:
            raise ValueError(f"CSV header does not match standalone observables schema: {path}")
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"too many columns at {path}:{line}")
            try:
                trajectory = int(row["trajectory"])
                measurement = int(row["measurement"])
                sample = int(row["sample"])
                coupling = float(row["coupling"])
                values = {name: float(row[name]) for name in _BASE_FIELDS}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric value at {path}:{line}") from exc
            if trajectory < 1 or measurement < 1 or sample < 0:
                raise ValueError(f"invalid trajectory/measurement/sample index at {path}:{line}")
            if not math.isfinite(coupling) or not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"non-finite observable at {path}:{line}")
            key = (trajectory, measurement, sample)
            if key in by_key:
                raise ValueError(f"duplicate trajectory/measurement/sample row at {path}:{line}: {key}")
            by_key[key] = (coupling, values)

    if not by_key:
        raise ValueError(f"no observable rows found in {path}")
    trajectories = sorted({key[0] for key in by_key})
    measurements = sorted({key[1] for key in by_key})
    samples = sorted({key[2] for key in by_key})

    def contiguous(indices, first):
        return indices[0] == first and all(right == left + 1 for left, right in zip(indices, indices[1:]))

    if not contiguous(trajectories, 1):
        raise ValueError("trajectory indices must be contiguous from 1")
    if not contiguous(measurements, 1):
        raise ValueError("measurement indices must be contiguous from 1")
    if not contiguous(samples, 0):
        raise ValueError("sample indices must be contiguous from 0")
    expected_rows = len(trajectories) * len(measurements) * len(samples)
    if len(by_key) != expected_rows:
        raise ValueError("incomplete observable grid")

    meta_path = Path(path).with_name("run_meta.json")
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot validate run metadata {meta_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"run metadata must be a JSON object: {meta_path}")
        if metadata.get("phase") == "evaluate":
            expected_count = metadata.get("evaluation_trajectory_count")
            if expected_count is not None and int(expected_count) != len(trajectories):
                raise ValueError(
                    f"observables contain {len(trajectories)} trajectories; run metadata expects {expected_count}"
                )
            args = metadata.get("args", {})
            if not isinstance(args, dict):
                raise ValueError(f"run metadata args must be an object: {meta_path}")
            if args.get("batch_size") is not None and int(args["batch_size"]) != len(samples):
                raise ValueError("observables sample count does not match run metadata batch_size")
            if args.get("int_meas") is not None and int(args["int_meas"]) != len(measurements):
                raise ValueError("observables measurement count does not match run metadata int_meas")
    return by_key, trajectories, measurements, samples


def _measurement_metrics(by_key, trajectories, measurement, samples, enough_data, S):
    raw = {
        name: {(trajectory, sample): by_key[(trajectory, measurement, sample)][1][name]
               for trajectory in trajectories for sample in samples}
        for name in _BASE_FIELDS
    }
    work_all = np.asarray([raw["work"][key] for key in raw["work"]], dtype=np.float64)
    work_reference = float(np.min(work_all))
    weights_raw = {key: math.exp(-(value - work_reference)) for key, value in raw["work"].items()}
    weights_mean = float(np.mean(list(weights_raw.values())))
    weights2_mean = float(np.mean([value * value for value in weights_raw.values()]))
    means = {name: float(np.mean(list(values.values()))) for name, values in raw.items()}
    centered_work = work_all - work_reference
    means["work_variance"] = float(np.mean(centered_work * centered_work) - np.mean(centered_work) ** 2)
    means["deltaF"] = work_reference - math.log(weights_mean)
    means["ess_fraction"] = weights_mean * weights_mean / weights2_mean
    for name in ("heat", "logJ", "action", "plaquette", "plaquette2"):
        means[f"{name}_reweighted"] = float(np.mean([
            raw[name][key] * weights_raw[key] for key in weights_raw
        ]) / weights_mean)

    if not enough_data:
        return means, {name: None for name in means}, "insufficient_statistics"

    observables = {name: _obs(values, trajectories, samples) for name, values in raw.items()}
    work = observables["work"]
    centered_work = _obs(
        {key: value - work_reference for key, value in raw["work"].items()}, trajectories, samples
    )
    weights = _obs(weights_raw, trajectories, samples)
    metrics = {
        "work": work,
        "work_variance": _obs(
            {key: (value - work_reference) ** 2 for key, value in raw["work"].items()}, trajectories, samples
        ) - centered_work ** 2,
        "deltaF": work_reference - weights.log(),
        "ess_fraction": weights ** 2 / _obs({key: value * value for key, value in weights_raw.items()}, trajectories, samples),
        **{name: observables[name] for name in ("heat", "logJ", "action", "plaquette", "plaquette2")},
    }
    for name in ("heat", "logJ", "action", "plaquette", "plaquette2"):
        metrics[f"{name}_reweighted"] = _obs(
            {key: raw[name][key] * weights_raw[key] for key in weights_raw}, trajectories, samples
        ) / weights

    stats = {}
    for name, observable in metrics.items():
        observable.gamma_method(S=S)
        if not all(math.isfinite(float(value)) for value in (observable.value, observable.dvalue, observable.ddvalue)):
            raise ValueError(f"non-finite Gamma result for {name} at measurement {measurement}")
        if len(observable.e_tauint) != 1:
            raise ValueError(f"expected one ensemble in Gamma result for {name}")
        ensemble = next(iter(observable.e_tauint))
        tau = float(observable.e_tauint[ensemble])
        dtau = float(observable.e_dtauint[ensemble])
        window = float(observable.e_windowsize[ensemble])
        if not all(math.isfinite(value) for value in (tau, dtau, window)):
            raise ValueError(f"non-finite Gamma window for {name} at measurement {measurement}")
        stats[name] = {
            "value": float(observable.value), "error": float(observable.dvalue),
            "error_of_error": float(observable.ddvalue), "tau_int": tau,
            "tau_int_error": dtau, "window": window,
        }
    return means, stats, "ok"


def _obs(values, trajectories, samples):
    arrays = [np.asarray([values[(trajectory, sample)] for trajectory in trajectories], dtype=np.float64)
              for sample in samples]
    names = [f"standalone|sample_{sample}" for sample in samples]
    return Obs(arrays, names)


def _atomic_report(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_HEADER)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def analyze_observables_csv(path, output_path=None, skip=0, nmax=None, S=2.0):
    """Analyze each intermediate measurement and sample replica with the Gamma method."""
    path = Path(path)
    try:
        S = float(S)
    except (TypeError, ValueError) as exc:
        raise ValueError("S must be finite and nonnegative") from exc
    if not isinstance(skip, int) or skip < 0 or (nmax is not None and (not isinstance(nmax, int) or nmax < 1)) or not math.isfinite(S) or S < 0:
        raise ValueError("skip must be nonnegative, nmax positive, and S finite/nonnegative")
    by_key, all_trajectories, measurements, samples = _read_observables(path)
    selected = all_trajectories[skip:]
    if nmax is not None:
        selected = selected[:nmax]
    if not selected:
        raise ValueError("skip/nmax selects no trajectories")
    # Obs currently needs five observations in every independent sample stream.
    enough_data = len(selected) >= 5
    rows = []
    final_work_error = None
    statuses = []
    for measurement in measurements:
        selected_couplings = [by_key[(trajectory, measurement, sample)][0]
                              for trajectory in selected for sample in samples]
        if not np.allclose(selected_couplings, selected_couplings[0], rtol=1e-12, atol=1e-14):
            raise ValueError(f"coupling changes within measurement {measurement}")
        means, stats, status = _measurement_metrics(by_key, selected, measurement, samples, enough_data, float(S))
        statuses.append(status)
        for name, mean in means.items():
            record = stats.get(name)
            row = {
                "measurement": measurement, "coupling": format(float(selected_couplings[-1]), ".17g"),
                "S": format(float(S), ".17g"),
                "observable": name, "value": format(float(record["value"] if record else mean), ".17g"),
                "error": "" if record is None else format(record["error"], ".17g"),
                "error_of_error": "" if record is None else format(record["error_of_error"], ".17g"),
                "tau_int": "" if record is None else format(record["tau_int"], ".17g"),
                "tau_int_error": "" if record is None else format(record["tau_int_error"], ".17g"),
                "window": "" if record is None else format(record["window"], ".17g"),
                "status": status,
            }
            rows.append(row)
            if measurement == measurements[-1] and name == "work" and record is not None:
                final_work_error = record["error"]

    report_path = Path(output_path) if output_path is not None else path.with_name("gamma_analysis.csv")
    if report_path.resolve() == path.resolve():
        raise ValueError("analysis output cannot overwrite observables.csv")
    _atomic_report(report_path, rows)
    return {
        "path": str(report_path.resolve()),
        "status": "ok" if all(status == "ok" for status in statuses) else "insufficient_statistics",
        "trajectory_count": len(all_trajectories), "analyzed_trajectory_count": len(selected),
        "measurement_count": len(measurements), "stream_count": len(samples),
        "skip": int(skip), "nmax": nmax, "S": float(S), "work_sem_gamma_final": final_work_error,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gamma-method analysis of standalone observables.csv")
    parser.add_argument("observables", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--S", type=float, default=2.0)
    args = parser.parse_args(argv)
    print(json.dumps(analyze_observables_csv(args.observables, args.output, args.skip, args.nmax, args.S), indent=2))


if __name__ == "__main__":
    main()
