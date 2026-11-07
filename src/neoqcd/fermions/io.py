"""Run-state and observable I/O for the fermionic HMC driver (main/main_hmc.py).

Run state = two files with a common prefix:
- `<prefix>.pt`:   a torch file {"U": links (B, D, lattice, N, N), complex, on CPU,
                   "rng_state": ByteTensor of the run generator};
- `<prefix>.json`: a JSON sidecar with the format version, the geometry, the couplings (beta,
                   kappa or mass, time_sign, fermions, N_f), the trajectory count, the RNG
                   device and state (base64 of the same ByteTensor), and any `extra` metadata.
The RNG state sits in both files on purpose: the torch file is what `load_state` uses, the
JSON copy keeps the sidecar self-describing. `load_state` checks that they agree.

Per-trajectory observables: `ObservableWriter` appends rows to a CSV with a fixed header (one
row per trajectory and replica), floats written with 17 significant digits (`repr`-exact
round trip), so resumed and uninterrupted runs can be compared bitwise from the text.
`read_observables` returns the columns as numpy arrays; `write_npz` stores them as `.npz`.

No neoqcd.* imports outside the fermion package are needed here.
"""

import base64
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from neoqcd.fermions.lattice import Geometry

__all__ = [
    "STATE_FORMAT_VERSION",
    "n_flavours",
    "save_state",
    "load_state",
    "restore_generator",
    "ObservableWriter",
    "read_observables",
    "write_npz",
    "atomic_json_save",
]

STATE_FORMAT_VERSION = 1


def n_flavours(fermions, D):
    """Continuum flavours of a run: 2 (Wilson pseudofermion), 2^(D/2) (one staggered field), 0."""
    if fermions == "wilson":
        return 2
    if fermions == "staggered":
        return 2 ** (int(D) // 2)
    if fermions == "none":
        return 0
    raise ValueError(f"unknown fermions {fermions!r}")


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json_save(payload, path):
    def write(tmp):
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
    _atomic_write(path, write)


def _prefix(path):
    """`run/checkpoint`, `run/checkpoint.pt` and `run/checkpoint.json` all name the same state."""
    path = Path(path)
    if path.suffix in (".pt", ".json"):
        path = path.with_suffix("")
    return path


def save_state(path, U, *, geometry, beta, fermions, trajectory, time_sign=-1.0, kappa=None,
               mass=None, generator=None, extra=None):
    """Write `<prefix>.pt` and `<prefix>.json` (module doc); returns (pt_path, json_path).

    `trajectory` is the number of trajectories completed when U was produced. The generator's
    state is saved if a generator is given (torch.Generator, CPU or CUDA).
    """
    prefix = _prefix(path)
    pt_path, json_path = prefix.with_suffix(".pt"), prefix.with_suffix(".json")
    geometry = Geometry(int(geometry.D), int(geometry.T), int(geometry.L), int(geometry.N))
    expected = geometry.link_shape(U.shape[0])
    if tuple(U.shape) != expected:
        raise ValueError(f"links of shape {tuple(U.shape)} do not match the geometry {expected}")
    rng_state = None if generator is None else generator.get_state().cpu().clone()
    payload = {"U": U.detach().cpu().clone(), "rng_state": rng_state}
    meta = {
        "format_version": STATE_FORMAT_VERSION,
        "geometry": {"D": geometry.D, "T": geometry.T, "L": geometry.L, "N": geometry.N},
        "lattice": list(geometry.lattice),
        "batch": int(U.shape[0]),
        "dtype": str(U.dtype).replace("torch.", ""),
        "beta": float(beta),
        "fermions": str(fermions),
        "kappa": None if kappa is None else float(kappa),
        "mass": None if mass is None else float(mass),
        "time_sign": float(time_sign),
        "n_flavours": n_flavours(fermions, geometry.D),
        "trajectory": int(trajectory),
        "rng": None if rng_state is None else {
            "device": str(generator.device),
            "state_base64": base64.b64encode(rng_state.numpy().tobytes()).decode("ascii"),
        },
        "extra": {} if extra is None else extra,
    }
    _atomic_write(pt_path, lambda tmp: torch.save(payload, tmp))
    atomic_json_save(meta, json_path)
    return pt_path, json_path


def load_state(path, device=None):
    """Read a state written by `save_state`: returns (U on `device`, meta dict, rng_state or None)."""
    prefix = _prefix(path)
    with open(prefix.with_suffix(".json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("format_version") != STATE_FORMAT_VERSION:
        raise ValueError(f"unsupported state format {meta.get('format_version')!r}")
    payload = torch.load(prefix.with_suffix(".pt"), map_location="cpu")
    U = payload["U"]
    g = meta["geometry"]
    expected = Geometry(g["D"], g["T"], g["L"], g["N"]).link_shape(meta["batch"])
    if tuple(U.shape) != expected:
        raise ValueError(f"stored links {tuple(U.shape)} do not match the sidecar geometry {expected}")
    rng_state = payload.get("rng_state")
    if meta.get("rng") is not None:
        from_json = torch.from_numpy(np.frombuffer(
            base64.b64decode(meta["rng"]["state_base64"]), dtype=np.uint8).copy())
        if rng_state is None or not torch.equal(rng_state, from_json):
            raise ValueError("RNG state in the torch file and in the JSON sidecar differ")
    if device is not None:
        U = U.to(device)
    return U, meta, rng_state


def restore_generator(generator, rng_state):
    """Set a generator from a saved ByteTensor state (the device types must match)."""
    generator.set_state(rng_state.cpu())
    return generator


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


class ObservableWriter:
    """Append-only CSV of per-trajectory rows with a fixed header.

    `ObservableWriter(path, fields)` creates the file with its header if missing (or checks the
    header of an existing one). `truncate(n)` drops rows with trajectory >= n, used on resume
    when the run went past its last checkpoint before stopping. Values: ints as ints, floats
    with `repr` (exact round trip), None as an empty field.
    """

    def __init__(self, path, fields, trajectory_field="traj"):
        self.path = Path(path)
        self.fields = list(fields)
        self.trajectory_field = trajectory_field
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            if header != self.fields:
                raise ValueError(f"{self.path} has header {header}, expected {self.fields}")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(self.fields)

    def truncate(self, n_trajectories):
        with open(self.path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            index = header.index(self.trajectory_field)
            rows = [row for row in reader if int(row[index]) < int(n_trajectories)]

        def write(tmp):
            with open(tmp, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)
        _atomic_write(self.path, write)

    def write_rows(self, rows):
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for row in rows:
                missing = set(self.fields) - set(row)
                if missing:
                    raise KeyError(f"row misses fields {sorted(missing)}")
                writer.writerow([_fmt(row[f]) for f in self.fields])
            handle.flush()


def read_observables(path):
    """Columns of an observables CSV as numpy float64 arrays (empty fields -> nan)."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    out = {}
    for j, name in enumerate(header):
        out[name] = np.asarray([float(r[j]) if r[j] != "" else np.nan for r in rows], dtype=np.float64)
    return out


def write_npz(path, columns):
    """Store a dict of columns (e.g. from `read_observables`) as an uncompressed .npz."""
    def write(tmp):
        with open(tmp, "wb") as handle:
            np.savez(handle, **{k: np.asarray(v) for k, v in columns.items()})
    _atomic_write(path, write)
