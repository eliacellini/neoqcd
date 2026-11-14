"""Cost of the fermion kernels of neoqcd.fermions on one device (GPU target, runs on CPU too).

Why: on an A100 (complex128) one CG iteration on D_hat^dagger D_hat costs 2.8 ms at 4^4 and
5.8 ms at 8^4 (batch 4): 16x the volume for 2x the time, so the kernels are dominated by launch
and Python overhead. This script measures where the time goes, to decide how to optimise.

For every lattice x batch x dtype of the grid (random Haar SU(3) links) it times, with warm-up
and a device synchronisation around each timed block:

    hopping            WilsonDirac.hopping (reference) and experimental variants defined here:
                         gather           neighbour index tables + one index_select of the 2D
                                          neighbours, one batched link matmul, one spin einsum;
                         gather_spinproj  as gather, with the rank-2 spin projection
                                          (1 -+ gamma_mu) = 2 V V^dagger: half spinors V^dagger psi
                                          are multiplied by the links and reconstructed with 2 V;
                         roll_spinproj    torch.roll as in the package, with the spin projection;
                         compiled_*       torch.compile of reference / gather_spinproj (a subset of
                                          the configurations, --compile-configs; failures reported);
                         cudagraph        CUDA-graph replay of the reference (CUDA only).
                       Every variant is checked against WilsonDirac.hopping before timing (relative
                       max-norm error, 1e-12 for complex128) and skipped if it disagrees.
    wilson_apply       WilsonDirac.apply
    eo_apply           EvenOddWilsonDirac.apply
    eo_normal          EvenOddWilsonDirac.normal (reference; gather_spinproj hopping; cudagraph)
    cg_iteration       per CG iteration on eo.normal: solver.cg with tol = 0 (per-iteration host
                       sync and per-sample masking; the time includes the final true-residual
                       normal() and is divided by the iterations actually performed), a bare CG
                       loop without syncs or masks (package operator, gather_spinproj operator)
                       and the bare loop captured in one CUDA graph (CUDA only)
    half_lattice       neoqcd.fermions.fast (opt-in package path), checked against the reference:
                         eo_normal    half_lattice_<mode>      HalfLatticeWilson.normal in the config
                                                        dtype, error vs EvenOddWilsonDirac, for every
                                                        colour product mode (matmul, broadcast,
                                                        broadcast_sum; `auto` = broadcast on CUDA,
                                                        matmul on CPU);
                                      half_lattice_<mode>_c64  the same operators cast to complex64
                                                        (complex128 configurations only; error vs
                                                        complex128 in `notes`, not in max_rel_error);
                         (lean_cg_half and the solve rows use colour_product="auto"; the
                         resolved mode is in `notes`)
                         cg_iteration lean_cg_half      lean_cg (host check every --check-every
                                                        iterations, no masking) on the half-lattice
                                                        normal operator, tol = 0, per loop iteration;
                                      lean_cg_reference lean_cg on EvenOddWilsonDirac.normal;
                         solve        solver_cg_reference / lean_cg_double / mixed_precision: total
                                      time of one solve of the same right-hand side to --solve-tol
                                      (complex128 configurations; "call" = one solve; iterations and
                                      outer steps in `iterations` / `notes`)
    half_lattice_fused neoqcd.fermions.fused (opt-in fused hop kernels, --fused-kernels):
                         eo_normal    half_lattice_fused_<k>[_c64]  FusedHalfLatticeWilson.normal, error
                                                        vs EvenOddWilsonDirac (c64: vs complex128, in
                                                        `notes`); <k> = loop (pure torch), cuda (the
                                                        CUDA kernel, CUDA devices) or cpu (the same
                                                        source built for the host, CPU devices), and
                                                        cuda_soa / cpu_soa (site-fastest links); an
                                                        unavailable backend gives one row with status
                                                        "skipped" and the reason in `notes`;
                         cg_iteration lean_cg_half_fused_<k>    lean_cg per loop iteration, tol = 0;
                         solve        lean_cg_double_fused_<k> / mixed_precision_fused_<k>: one solve
                                      to --solve-tol (compiled, site layout, complex128 configurations;
                                      rel diff vs lean_cg_double in `notes`).
    pf_force           pseudofermion force given the solution X (the post-solve part of
                       TwoFlavourWilson.force, even-odd), and hopping_derivative alone
    gauge_force        WilsonGaugeAction.force
    expi_update        U <- SUN_mul(expi(h P), U) (one link update of the integrator)
    staggered_mee      StaggeredEvenOdd.apply (M_ee = m^2 - K_eo K_oe)

Columns: seconds per call (median over trials, and minimum), seconds per call per site per batch
entry, and site-updates per second = V * B / seconds (V = full-lattice volume for every kernel,
even-odd ones included; for cg_iteration "call" = one iteration). For one configuration
(--breakdown-config) the hopping term is split into its parts (rolls, link products with and
without the lazy conj view, spin combination, one gather), and the CUDA kernel launches (CPU: top
level aten ops) per call are counted with torch.profiler.

complex64: the operators keep kappa as a float64 tensor, so WilsonDirac.apply on complex64 fields
returns complex128 (type promotion). The script checks this at start; if any operator does not
preserve complex64 it is reported in run_meta.json and only complex128 is benchmarked, unless
--force-complex64 is given.

Outputs in --out: bench.csv (one row per measurement, written as it goes), run_meta.json
(device, torch/python/CUDA versions, commit, arguments, complex64 check, compile status,
variant errors, kernel counts) and a readable table on stdout.

--skip-legacy drops the rows that earlier runs settled (experimental hopping variants, the
solver.cg / bare / reference-operator CG iterations and the reference solve); --half-modes selects
the colour product modes of the half_lattice rows (the lean_cg / solve rows need the "auto" mode).

Usage:
    python main/bench_fermions.py --device cuda --out output/bench_fermions/run1
    python main/bench_fermions.py --lattices 8^4 16^4 --batches 4 16 --dtypes complex128
Lattice specs: "L^4" (T = L), "L^3xT"; generally "L^kxT" (D = k + 1) or "L^k" (D = k).
"""

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neoqcd.fermions._gauge import SUN_haar, SUN_mul  # noqa: E402
from neoqcd.fermions.algebra import expi, random_algebra  # noqa: E402
from neoqcd.fermions.dirac import WilsonDirac  # noqa: E402
from neoqcd.fermions.fast import (  # noqa: E402
    HalfLatticeWilson,
    lean_cg,
    mixed_precision_cg,
    resolve_colour_product,
    to_half,
)
from neoqcd.fermions.fused import BackendUnavailable, FusedHalfLatticeWilson  # noqa: E402

COLOUR_MODES = ("matmul", "broadcast", "broadcast_sum")
from neoqcd.fermions.hmc import WilsonGaugeAction  # noqa: E402
from neoqcd.fermions.lattice import Geometry, shift  # noqa: E402
from neoqcd.fermions.pseudofermion import hopping_derivative  # noqa: E402
from neoqcd.fermions.solver import cg  # noqa: E402
from neoqcd.fermions.staggered import StaggeredDirac  # noqa: E402

DTYPES = {"complex128": torch.complex128, "complex64": torch.complex64}
TOLERANCE = {torch.complex128: 1e-12, torch.complex64: 1e-5}
CSV_FIELDS = ["lattice", "D", "T", "L", "volume", "batch", "dtype", "measurement", "variant",
              "status", "seconds_per_call", "seconds_min", "seconds_per_site_batch",
              "site_updates_per_s", "repeats", "trials", "iterations", "max_rel_error", "out_dtype",
              "kernels_per_call", "notes"]


# ---- configuration ------------------------------------------------------------------------------
def parse_lattice(spec, N=3):
    """'L^4' -> 4D L^4, 'L^3xT' -> 4D L^3 x T, generally 'L^k' (D = k) or 'L^kxT' (D = k + 1)."""
    s = spec.strip().lower()
    try:
        if "x" in s:
            space, T = s.split("x")
            L, k = (int(v) for v in space.split("^"))
            return Geometry(k + 1, int(T), L, N)
        L, k = (int(v) for v in s.split("^"))
        return Geometry(k, L, L, N)
    except ValueError:
        raise ValueError(f"bad lattice spec {spec!r}: use 'L^4' or 'L^3xT'")


def parse_config(spec):
    """'8^4:4' -> ('8^4', 4)."""
    lat, batch = spec.rsplit(":", 1)
    return lat.strip(), int(batch)


def estimate_bytes(geo, batch, dtype):
    """Rough peak memory of one configuration: link copies (U, boundary links, gathered link
    stacks, half-lattice link tables, staggered, gauge-force temporaries) and fields (CG vectors,
    2D gathered neighbours)."""
    elt = 16 if dtype == torch.complex128 else 8
    links = batch * geo.D * geo.volume * geo.N * geo.N
    field = batch * geo.volume * geo.spin * geo.N
    return (8 * links + (8 * geo.D + 8) * field) * elt


def git_info():
    def git(*command):
        try:
            return subprocess.run(["git", *command], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": git("status", "--porcelain", "--untracked-files=no")}


# ---- timing --------------------------------------------------------------------------------------
def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_calls(fn, device, args, warmup=None, min_repeats=None):
    """(median, min) seconds per call of fn() over `args.trials` trials, and the repeats per trial.

    The repeats are chosen so that a trial lasts about `args.target_time` seconds, within
    [min_repeats, args.max_repeats]; each trial is one synchronised block of back-to-back calls,
    so the figure includes launch and Python overhead, as in production.
    """
    warmup = args.warmup if warmup is None else warmup
    min_repeats = args.min_repeats if min_repeats is None else min_repeats
    for _ in range(warmup):
        fn()
    synchronize(device)
    t0 = time.perf_counter()
    fn()
    synchronize(device)
    single = time.perf_counter() - t0
    repeats = int(args.target_time / max(single, 1e-9))
    repeats = max(min_repeats, min(args.max_repeats, repeats))
    per_call = []
    for _ in range(args.trials):
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        synchronize(device)
        per_call.append((time.perf_counter() - t0) / repeats)
    return statistics.median(per_call), min(per_call), repeats


def rel_error(x, ref):
    den = ref.abs().max().item()
    return (x.to(ref.dtype) - ref).abs().max().item() / (den if den > 0 else 1.0)


def count_kernels(fn, device, calls=3):
    """CUDA kernel launches per call of fn (CPU: top-level aten ops per call), via torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    fn()
    synchronize(device)
    with profile(activities=activities) as prof:
        for _ in range(calls):
            fn()
        synchronize(device)
    events = prof.events()
    if device.type == "cuda":
        cuda_type = torch.autograd.DeviceType.CUDA
        direct = sum(1 for e in events if getattr(e, "device_type", None) == cuda_type)
        attached = sum(len(getattr(e, "kernels", []) or []) for e in events)
        n = max(direct, attached)
    else:
        n = sum(1 for e in events if e.name.startswith("aten::") and getattr(e, "cpu_parent", None) is None)
    return n / calls


# ---- experimental hopping variants (benchmark only, not part of the package) ----------------------
class HoppingVariants:
    """Alternative implementations of WilsonDirac.hopping for the same links and conventions.

    Reference: (H psi)(x) = sum_mu [(1 - g_mu) L_mu(x) psi(x + mu) + (1 + g_mu) L_mu(x - mu)^dag psi(x - mu)],
    with shift(f, mu, +1)[n] = f[n - e_mu] (torch.roll by +1). In flattened site order the forward
    neighbour of site n along mu is idx_fwd[mu][n] = flat(n - e_mu) and the backward one
    idx_bwd[mu][n] = flat(n + e_mu). The 2D hops k = 0..D-1 (forward, projector 1 - g_mu, link
    L_mu(x)) and k = D..2D-1 (backward, projector 1 + g_mu, link L_mu(x - mu)^dag gathered at
    idx_bwd) are stacked so that one index_select, one batched matmul and one einsum do the
    whole hopping term. Colour products follow the package: psi (S, N) @ L^T is L psi.

    Spin projection: P = 1 -+ g_mu is twice an orthogonal projector of rank S / 2, P = 2 V V^dag
    with V (S, S/2) its orthonormal eigenvectors of eigenvalue 2; the half spinor V^dag psi is
    multiplied by the link (half the colour work) and reconstructed with 2 V.
    """

    def __init__(self, wilson):
        self.w = wilson
        geo = wilson.geometry
        self.geo = geo
        self.D, self.V, self.S, self.N = geo.D, geo.volume, geo.spin, geo.N
        dev = wilson.device
        sites = torch.arange(self.V, device=dev).reshape(geo.lattice)
        idx_fwd = [torch.roll(sites, 1, dims=mu).reshape(-1) for mu in range(self.D)]
        idx_bwd = [torch.roll(sites, -1, dims=mu).reshape(-1) for mu in range(self.D)]
        self.idx_fwd, self.idx_bwd = idx_fwd, idx_bwd
        self.idx_all = torch.cat(idx_fwd + idx_bwd)
        B = wilson.U.shape[0]
        flat = (B, self.V, self.N, self.N)
        fwd = [wilson._fwd[mu].resolve_conj().reshape(flat) for mu in range(self.D)]
        bwd = [wilson._bwd[mu].resolve_conj().reshape(flat).index_select(1, idx_bwd[mu])
               for mu in range(self.D)]
        self.links = torch.stack(fwd + bwd, dim=1).contiguous()  # (B, 2D, V, N, N)
        self.fwd_links = [l.resolve_conj().contiguous() for l in wilson._fwd]
        self.bwd_links = [l.resolve_conj().contiguous() for l in wilson._bwd]
        g = wilson.gammas
        one = torch.eye(self.S, dtype=g.dtype, device=dev)
        proj = [one - g[mu] for mu in range(self.D)] + [one + g[mu] for mu in range(self.D)]
        self.proj = torch.stack(proj)  # (2D, S, S)
        vh, rc = [], []
        for P in proj:
            evals, evecs = torch.linalg.eigh(P)  # ascending: eigenvalues 0 then 2
            Vk = evecs[:, self.S // 2:]
            if not torch.allclose(evals[self.S // 2:], torch.full((self.S // 2,), 2.0, dtype=evals.dtype,
                                                                       device=dev)):
                raise RuntimeError("1 -+ gamma_mu is not 2 x a rank-S/2 projector")
            vh.append(Vk.conj().transpose(0, 1))
            rc.append(2.0 * Vk)
        self.vh = torch.stack(vh).contiguous()  # (2D, S/2, S)
        self.rc = torch.stack(rc).contiguous()  # (2D, S, S/2)

    def _neighbours(self, psi):
        B = psi.shape[0]
        flat = psi.reshape(B, self.V, self.S, self.N)
        return flat.index_select(1, self.idx_all).view(B, 2 * self.D, self.V, self.S, self.N)

    def gather(self, psi):
        prod = torch.matmul(self._neighbours(psi), self.links)
        return torch.einsum("ksr,bkvrn->bvsn", self.proj, prod).reshape(psi.shape)

    def gather_spinproj(self, psi):
        half = torch.einsum("khs,bkvsn->bkvhn", self.vh, self._neighbours(psi))
        prod = torch.matmul(half, self.links)
        return torch.einsum("ksh,bkvhn->bvsn", self.rc, prod).reshape(psi.shape)

    def roll_spinproj(self, psi):
        D = self.D
        out = None
        for mu in range(D):
            f = torch.matmul(torch.matmul(self.vh[mu], shift(psi, mu, 1)), self.fwd_links[mu])
            b = shift(torch.matmul(torch.matmul(self.vh[D + mu], psi), self.bwd_links[mu]), mu, -1)
            term = torch.matmul(self.rc[mu], f) + torch.matmul(self.rc[D + mu], b)
            out = term if out is None else out + term
        return out


def eo_normal_from(hopping, eo):
    """EvenOddWilsonDirac.normal with another hopping implementation (same masks and algebra)."""
    w = eo.wilson

    def apply(psi):
        psi_e = eo.even(psi)
        k2 = w._kappa_view(psi) ** 2
        return eo.even(psi_e - k2 * hopping(hopping(psi_e)))

    def normal(psi):
        return w.gamma5(apply(w.gamma5(apply(psi))))

    return normal


def _ndot(a, b):
    return (a.conj() * b).reshape(a.shape[0], -1).sum(dim=1)


def bare_cg(apply_normal, b, n_iter):
    """CG without host syncs, convergence checks or per-sample masks: n_iter iterations exactly."""
    view = (-1,) + (1,) * (b.dim() - 1)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = _ndot(r, r).real
    for _ in range(n_iter):
        Ap = apply_normal(p)
        alpha = (rr / _ndot(p, Ap).real).view(view).to(b.dtype)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = _ndot(r, r).real
        p = r + (rr_new / rr).view(view).to(b.dtype) * p
        rr = rr_new
    return x


def pf_force_given_solution(eo, X):
    """Post-solve part of TwoFlavourWilson.force (even-odd), copied from pseudofermion.py."""
    w = eo.wilson
    Y = eo.apply(X)
    W = eo.odd(w.hopping(X))
    Z = eo.odd(w.gamma5(w.hopping(w.gamma5(Y))))
    G = hopping_derivative(w, Y + Z, X + W)
    return (2.0 * w.kappa ** 2).view((-1,) + (1,) * (G.dim() - 1)).to(G.dtype) * G


def cuda_graph(fn, static_input, warmup=3):
    """Capture fn(static_input) in a CUDA graph; returns (replay, static_output)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn(static_input)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = fn(static_input)
    return graph.replay, static_output


# ---- complex64 support ----------------------------------------------------------------------------
def check_complex64(device):
    """Output dtypes of the operators on complex64 inputs (2^4 lattice); supported if all complex64."""
    geo = Geometry(4, 2, 2, 3)
    g = torch.Generator().manual_seed(0)
    U = SUN_haar(geo.link_shape(1)[:-1], generator=g).to(device=device, dtype=torch.complex64)
    psi = torch.randn(geo.wilson_shape(1), dtype=torch.complex64, generator=g).to(device)
    out = {}
    try:
        w = WilsonDirac(U, 0.12, geo)
        eo = w.even_odd()
        out["hopping"] = str(w.hopping(psi).dtype)
        out["wilson_apply"] = str(w.apply(psi).dtype)
        out["eo_apply"] = str(eo.apply(psi).dtype)
        out["eo_normal"] = str(eo.normal(psi).dtype)
        out["solver_cg"] = str(cg(eo.normal, eo.even(psi), tol=1e-4, max_iter=3).x.dtype)
        chi = torch.randn(geo.staggered_shape(1), dtype=torch.complex64, generator=g).to(device)
        out["staggered_mee"] = str(StaggeredDirac(U, 0.1, geo).even_odd().apply(chi).dtype)
        out["gauge_force"] = str(WilsonGaugeAction(5.6, geo).force(U).dtype)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    supported = "error" not in out and all(v == str(torch.complex64) for v in out.values())
    return supported, out


# ---- benchmark driver -------------------------------------------------------------------------------
class Bench:
    def __init__(self, args, device, writer, handle):
        self.args = args
        self.device = device
        self.writer = writer
        self.handle = handle
        self.rows = []
        self.meta_variants = {}
        self.meta_compile = {}
        self.meta_kernels = {}
        self.meta_fast = {}

    def emit(self, ctx, measurement, variant, status="ok", t=None, t_min=None, repeats=None,
             iterations=None, error=None, out_dtype=None, kernels=None, notes=""):
        geo, batch, dtype_name = ctx
        sites = geo.volume * batch
        row = {
            "lattice": self.lattice_name(geo), "D": geo.D, "T": geo.T, "L": geo.L, "volume": geo.volume,
            "batch": batch, "dtype": dtype_name, "measurement": measurement, "variant": variant,
            "status": status,
            "seconds_per_call": "" if t is None else f"{t:.6e}",
            "seconds_min": "" if t_min is None else f"{t_min:.6e}",
            "seconds_per_site_batch": "" if t is None else f"{t / sites:.6e}",
            "site_updates_per_s": "" if t is None else f"{sites / t:.6e}",
            "repeats": "" if repeats is None else repeats,
            "trials": "" if t is None else self.args.trials,
            "iterations": "" if iterations is None else iterations,
            "max_rel_error": "" if error is None else f"{error:.3e}",
            "out_dtype": "" if out_dtype is None else str(out_dtype).replace("torch.", ""),
            "kernels_per_call": "" if kernels is None else f"{kernels:g}",
            "notes": notes,
        }
        self.writer.writerow(row)
        self.handle.flush()
        self.rows.append(row)
        if not self.args.quiet:
            tt = "" if t is None else f"{t * 1e3:10.4f} ms  {sites / t:10.3e} site/s"
            extra = []
            if iterations is not None:
                extra.append(f"it={iterations}")
            if error is not None:
                extra.append(f"err={error:.1e}")
            if kernels is not None:
                extra.append(f"kern={kernels:g}")
            if status != "ok":
                extra.append(status)
            if notes:
                extra.append(notes)
            print(f"  {measurement:<14s} {variant:<26s} {tt:<36s} {' '.join(extra)}", flush=True)
        return row

    @staticmethod
    def lattice_name(geo):
        if geo.T == geo.L:
            return f"{geo.L}^{geo.D}"
        return f"{geo.L}^{geo.D - 1}x{geo.T}"

    def timed(self, ctx, measurement, variant, fn, out=None, divide=1, iterations=None, error=None,
              notes="", warmup=None, min_repeats=None, kernels=None):
        """Time fn and emit a row; errors (OOM included) become a row with the status."""
        try:
            t, t_min, reps = time_calls(fn, self.device, self.args, warmup=warmup, min_repeats=min_repeats)
        except RuntimeError as exc:
            status = "oom" if "out of memory" in str(exc).lower() else "error"
            self._free()
            return self.emit(ctx, measurement, variant, status=status, notes=_short(exc))
        return self.emit(ctx, measurement, variant, t=t / divide, t_min=t_min / divide, repeats=reps,
                         iterations=iterations, error=error, kernels=kernels,
                         out_dtype=None if out is None else out.dtype, notes=notes)

    def _free(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- one configuration ----
    def run_config(self, geo, batch, dtype_name, compile_here, breakdown_here):
        args, device = self.args, self.device
        dtype = DTYPES[dtype_name]
        ctx = (geo, batch, dtype_name)
        key = f"{self.lattice_name(geo)}:{batch}:{dtype_name}"
        tol = TOLERANCE[dtype]
        g = torch.Generator().manual_seed(args.seed)
        U = SUN_haar(geo.link_shape(batch)[:-1], generator=g).to(device=device, dtype=dtype)
        psi = torch.randn(geo.wilson_shape(batch), dtype=torch.complex128, generator=g).to(device=device,
                                                                                            dtype=dtype)
        w = WilsonDirac(U, args.kappa, geo)
        eo = w.even_odd()
        psi_e = eo.even(psi)
        ref = w.hopping(psi)

        # (a) hopping: reference and variants
        self.timed(ctx, "hopping", "reference", lambda: w.hopping(psi), out=ref)
        var = HoppingVariants(w)
        candidates = [] if args.skip_legacy else [("gather", var.gather), ("gather_spinproj", var.gather_spinproj),
                                                  ("roll_spinproj", var.roll_spinproj)]
        good = {}
        for name, fn in candidates:
            out = fn(psi)
            err = rel_error(out, ref)
            self.meta_variants[f"{key}:{name}"] = err
            if not err <= tol:
                self.emit(ctx, "hopping", name, status="mismatch", error=err)
                continue
            good[name] = fn
            self.timed(ctx, "hopping", name, lambda fn=fn: fn(psi), out=out, error=err)
        if compile_here:
            self._compiled(ctx, key, "hopping", [("compiled_reference", w.hopping)]
                           + ([("compiled_gather_spinproj", var.gather_spinproj)]
                              if "gather_spinproj" in good else []), psi, ref, tol)
        if device.type == "cuda" and not args.no_cuda_graphs:
            self._graph(ctx, "hopping", "cudagraph", w.hopping, psi, ref, tol)

        # (b), (c) operators
        self.timed(ctx, "wilson_apply", "reference", lambda: w.apply(psi), out=w.apply(psi))
        self.timed(ctx, "eo_apply", "reference", lambda: eo.apply(psi_e), out=eo.apply(psi_e))
        ref_normal = eo.normal(psi_e)
        row = self.timed(ctx, "eo_normal", "reference", lambda: eo.normal(psi_e), out=ref_normal)
        t_normal = float(row["seconds_per_call"]) if row["seconds_per_call"] else None
        fast_normal = None
        if "gather_spinproj" in good:
            fast_normal = eo_normal_from(var.gather_spinproj, eo)
            out = fast_normal(psi_e)
            err = rel_error(out, ref_normal)
            if err <= tol:
                self.timed(ctx, "eo_normal", "gather_spinproj", lambda: fast_normal(psi_e), out=out, error=err)
            else:
                self.emit(ctx, "eo_normal", "gather_spinproj", status="mismatch", error=err)
                fast_normal = None
        if device.type == "cuda" and not args.no_cuda_graphs:
            self._graph(ctx, "eo_normal", "cudagraph", eo.normal, psi_e, ref_normal, tol)

        # (d) CG iterations
        n_iter = args.cg_iters
        if t_normal and args.target_time > 0:
            n_iter = max(args.cg_min_iters, min(args.cg_iters, int(args.target_time / t_normal)))
        b = eo.even(torch.randn(geo.wilson_shape(batch), dtype=torch.complex128, generator=g).to(device=device,
                                                                                                  dtype=dtype))
        if not args.skip_legacy:
            try:
                res = cg(eo.normal, b, tol=0.0, max_iter=n_iter)
                it = int(res.iterations.max().item())
                self.timed(ctx, "cg_iteration", "solver_cg",
                           lambda: cg(eo.normal, b, tol=0.0, max_iter=n_iter), out=res.x, divide=max(it, 1),
                           iterations=it, warmup=1, min_repeats=1,
                           notes="tol=0; includes the final true-residual normal()")
            except RuntimeError as exc:
                self.emit(ctx, "cg_iteration", "solver_cg", status="error", notes=_short(exc))
            self.timed(ctx, "cg_iteration", "bare", lambda: bare_cg(eo.normal, b, n_iter), divide=n_iter,
                       iterations=n_iter, warmup=1, min_repeats=1, notes="no host syncs, no masking")
        if fast_normal is not None:
            self.timed(ctx, "cg_iteration", "bare_gather_spinproj", lambda: bare_cg(fast_normal, b, n_iter),
                       divide=n_iter, iterations=n_iter, warmup=1, min_repeats=1)
        if device.type == "cuda" and not args.no_cuda_graphs:
            try:
                replay, _ = cuda_graph(lambda v: bare_cg(eo.normal, v, n_iter), b, warmup=1)
                self.timed(ctx, "cg_iteration", "bare_cudagraph", replay, divide=n_iter, iterations=n_iter,
                           warmup=1, min_repeats=1, notes="whole loop in one CUDA graph")
            except RuntimeError as exc:
                self.emit(ctx, "cg_iteration", "bare_cudagraph", status="error", notes=_short(exc))
            torch.cuda.synchronize()

        # (d') half-lattice operator and solvers of neoqcd.fermions.fast
        self._half_lattice(ctx, geo, eo, U, psi_e, ref_normal, b, n_iter, tol)

        # (e) pseudofermion force given the solution
        X = eo.even(torch.randn(geo.wilson_shape(batch), dtype=torch.complex128, generator=g).to(device=device,
                                                                                                  dtype=dtype))
        a2 = torch.randn(geo.wilson_shape(batch), dtype=torch.complex128, generator=g).to(device=device, dtype=dtype)
        self.timed(ctx, "pf_force", "given_solution", lambda: pf_force_given_solution(eo, X))
        self.timed(ctx, "pf_force", "hopping_derivative", lambda: hopping_derivative(w, a2, psi))

        # (f) gauge force and link update
        gauge = WilsonGaugeAction(args.beta, geo)
        self.timed(ctx, "gauge_force", "reference", lambda: gauge.force(U))
        try:
            P = random_algebra(U.shape, generator=g, dtype=torch.complex128).to(device=device, dtype=dtype)
        except RuntimeError as exc:  # OOM while building the momenta: a row, not a crash
            P = None
            self._free()
            self.emit(ctx, "expi_update", "cayley_hamilton",
                      status="oom" if "out of memory" in str(exc).lower() else "error", notes=_short(exc))
        if P is not None:
            self.timed(ctx, "expi_update", "cayley_hamilton", lambda: SUN_mul(expi(0.05 * P), U))
        del P, gauge

        # (g) staggered M_ee
        st = StaggeredDirac(U, args.mass, geo).even_odd()
        chi = st.even(torch.randn(geo.staggered_shape(batch), dtype=torch.complex128, generator=g).to(
            device=device, dtype=dtype))
        self.timed(ctx, "staggered_mee", "reference", lambda: st.apply(chi), out=st.apply(chi))
        del st, chi

        if breakdown_here:
            self._breakdown(ctx, key, w, var, eo, psi, psi_e, good, fast_normal)
        del U, psi, psi_e, w, eo, var, ref, ref_normal, b, X, a2
        self._free()

    def _half_lattice(self, ctx, geo, eo, U, psi_e, ref_normal, b_full, n_iter, tol):
        """Rows of the half-lattice operator, lean_cg and the mixed-precision solve (module doc)."""
        args = self.args
        dtype = U.dtype
        key = f"{self.lattice_name(geo)}:{ctx[1]}:{ctx[2]}"
        x = to_half(psi_e, geo)
        ref_half = to_half(ref_normal, geo)
        auto = resolve_colour_product("auto", U.device)
        half = half64 = None
        modes = list(dict.fromkeys(list(args.half_modes) + [auto]))  # the auto mode feeds the CG rows
        for mode in modes:
            try:
                op = HalfLatticeWilson(U, args.kappa, geo, colour_product=mode)
            except RuntimeError as exc:
                self._free()
                self.emit(ctx, "eo_normal", f"half_lattice_{mode}", status="error", notes=_short(exc))
                continue
            out = op.normal(x)
            err = rel_error(out, ref_half)
            self.meta_fast[f"{key}:half_lattice_{mode}"] = err
            if not err <= tol:
                self.emit(ctx, "eo_normal", f"half_lattice_{mode}", status="mismatch", error=err)
                continue
            self.timed(ctx, "eo_normal", f"half_lattice_{mode}", lambda op=op: op.normal(x), out=out, error=err,
                       notes=f"colour_product={mode}")
            op64 = None
            if dtype == torch.complex128:
                op64 = op.to(torch.complex64)
                x64 = x.to(torch.complex64)
                out64 = op64.normal(x64)
                err64 = rel_error(out64, ref_half)
                self.meta_fast[f"{key}:half_lattice_{mode}_c64"] = err64
                if err64 <= TOLERANCE[torch.complex64] and out64.dtype == torch.complex64:
                    self.timed(ctx, "eo_normal", f"half_lattice_{mode}_c64", lambda op64=op64: op64.normal(x64),
                               out=out64, notes=f"colour_product={mode}; err vs complex128 {err64:.1e}")
                else:
                    self.emit(ctx, "eo_normal", f"half_lattice_{mode}_c64", status="mismatch",
                              notes=f"err vs complex128 {err64:.1e}; out {out64.dtype}")
                    op64 = None
            if mode == auto:
                half, half64 = op, op64
            del op, op64
        bh = to_half(b_full, geo)
        self._fused(ctx, geo, U, x, ref_half, bh, n_iter, tol)
        if half is None:
            return
        cg_rows = [("lean_cg_half", half.normal)] + ([] if args.skip_legacy else [("lean_cg_reference", eo.normal)])
        for variant, normal in cg_rows:
            rhs = bh if variant == "lean_cg_half" else b_full
            info = {}
            res = lean_cg(normal, rhs, tol=0.0, max_iter=n_iter, check_every=args.check_every, info=info)
            loops = max(info["loop_iterations"], 1)
            self.timed(ctx, "cg_iteration", variant,
                       lambda normal=normal, rhs=rhs: lean_cg(normal, rhs, tol=0.0, max_iter=n_iter,
                                                             check_every=args.check_every),
                       out=res.x, divide=loops, iterations=loops, warmup=1, min_repeats=1,
                       notes=f"tol=0; check_every={args.check_every}; colour_product={auto}; "
                             "includes the final true-residual normal()")
        if dtype != torch.complex128:
            return
        # one solve of the same right-hand side to --solve-tol: reference, lean double, mixed
        st, mi = args.solve_tol, args.solve_max_iter
        if not args.skip_legacy:
            ref = cg(eo.normal, b_full, tol=st, max_iter=mi)
            it = int(ref.iterations.max().item())
            self.timed(ctx, "solve", "solver_cg_reference", lambda: cg(eo.normal, b_full, tol=st, max_iter=mi),
                       out=ref.x, iterations=it, warmup=1, min_repeats=1,
                       notes=f"tol={st:g}; residual {ref.residual.max().item():.1e}")
        info = {}
        dbl = lean_cg(half.normal, bh, tol=st, max_iter=mi, check_every=args.check_every, info=info)
        self.timed(ctx, "solve", "lean_cg_double",
                   lambda: lean_cg(half.normal, bh, tol=st, max_iter=mi, check_every=args.check_every),
                   out=dbl.x, iterations=info["loop_iterations"], warmup=1, min_repeats=1,
                   notes=f"tol={st:g}; colour_product={auto}; residual {dbl.residual.max().item():.1e}")
        if half64 is None:
            return
        info = {}
        mix = mixed_precision_cg(half, half64, bh, tol=st, inner_tol=args.inner_tol, max_iter=mi,
                                 check_every=args.check_every, info=info)
        diff = rel_error(mix.x, dbl.x)
        status_note = (f"tol={st:g}; colour_product={auto}; inner_tol={args.inner_tol:g}; outer={info['outer']}; "
                       f"inner max per outer={info['inner_iterations']}; residual {mix.residual.max().item():.1e}; "
                       f"rel diff vs double {diff:.1e}").replace(",", ";")
        self.meta_fast[f"{key}:mixed_precision"] = {"outer": info["outer"], "inner": info["inner_iterations"],
                                                    "loops": info["loop_iterations"],
                                                    "residual": mix.residual.max().item(), "diff": diff}
        if not mix.residual.max().item() <= st:
            self.emit(ctx, "solve", "mixed_precision", status="not_converged", notes=status_note)
            return
        self.timed(ctx, "solve", "mixed_precision",
                   lambda: mixed_precision_cg(half, half64, bh, tol=st, inner_tol=args.inner_tol, max_iter=mi,
                                              check_every=args.check_every),
                   out=mix.x, iterations=info["loop_iterations"], warmup=1, min_repeats=1, notes=status_note)

    def _fused_kernels(self):
        """(label, kernel) pairs of --fused-kernels on this device ("compiled": cuda or cpu)."""
        out = []
        k = "cuda" if self.device.type == "cuda" else "cpu"
        for name in self.args.fused_kernels:
            if name == "loop":
                out.append(("loop", "loop", "site"))
            elif name == "compiled":
                out.append((k, k, "site"))
            elif name == "compiled_soa":
                out.append((f"{k}_soa", k, "soa"))
        return out

    def _fused(self, ctx, geo, U, x, ref_half, bh, n_iter, tol):
        """Rows of neoqcd.fermions.fused: normal (complex128 and complex64), lean_cg per iteration,
        solves for the compiled kernels (module docstring)."""
        key = f"{self.lattice_name(geo)}:{ctx[1]}:{ctx[2]}"
        for label, kernel, layout in self._fused_kernels():
            variant = f"half_lattice_fused_{label}"
            try:
                op = FusedHalfLatticeWilson(U, self.args.kappa, geo, kernel=kernel, strict=True, link_layout=layout)
            except BackendUnavailable as exc:
                self.meta_fast[f"{key}:{variant}"] = f"skipped: {exc}"
                self.emit(ctx, "eo_normal", variant, status="skipped", notes=f"skipped: {_short(exc, 400)}")
                continue
            try:
                self._fused_one(ctx, key, label, layout, variant, op, U.dtype, x, ref_half, bh, n_iter, tol)
            except RuntimeError as exc:  # OOM or a kernel error: a row, not a lost configuration
                status = "oom" if "out of memory" in str(exc).lower() else "error"
                self.emit(ctx, "eo_normal", variant, status=status, notes=_short(exc, 400))
            del op
            self._free()

    def _fused_one(self, ctx, key, label, layout, variant, op, dtype, x, ref_half, bh, n_iter, tol):
        """Rows of one fused operator (called by _fused, which turns a RuntimeError into a row)."""
        args = self.args
        out = op.normal(x)
        err = rel_error(out, ref_half)
        self.meta_fast[f"{key}:{variant}"] = err
        if not err <= tol:
            self.emit(ctx, "eo_normal", variant, status="mismatch", error=err, notes=f"kernel={op.kernel}")
            return
        self.timed(ctx, "eo_normal", variant, lambda op=op: op.normal(x), out=out, error=err,
                   notes=f"kernel={op.kernel}")
        op64 = None
        if dtype == torch.complex128:
            op64 = op.to(torch.complex64)
            x64 = x.to(torch.complex64)
            out64 = op64.normal(x64)
            err64 = rel_error(out64, ref_half)
            self.meta_fast[f"{key}:{variant}_c64"] = err64
            if err64 <= TOLERANCE[torch.complex64] and out64.dtype == torch.complex64:
                self.timed(ctx, "eo_normal", f"{variant}_c64", lambda op64=op64: op64.normal(x64), out=out64,
                           notes=f"kernel={op.kernel}; err vs complex128 {err64:.1e}")
            else:
                self.emit(ctx, "eo_normal", f"{variant}_c64", status="mismatch",
                          notes=f"err vs complex128 {err64:.1e}; out {out64.dtype}")
                op64 = None
            del out64
        info = {}
        res = lean_cg(op.normal, bh, tol=0.0, max_iter=n_iter, check_every=args.check_every, info=info)
        loops = max(info["loop_iterations"], 1)
        self.timed(ctx, "cg_iteration", f"lean_cg_half_fused_{label}",
                   lambda op=op: lean_cg(op.normal, bh, tol=0.0, max_iter=n_iter, check_every=args.check_every),
                   out=res.x, divide=loops, iterations=loops, warmup=1, min_repeats=1,
                   notes=f"tol=0; check_every={args.check_every}; kernel={op.kernel}; "
                         "includes the final true-residual normal()")
        del res
        if label == "loop" or layout != "site" or dtype != torch.complex128:
            return
        st, mi = args.solve_tol, args.solve_max_iter
        info = {}
        dbl = lean_cg(op.normal, bh, tol=st, max_iter=mi, check_every=args.check_every, info=info)
        self.timed(ctx, "solve", f"lean_cg_double_fused_{label}",
                   lambda op=op: lean_cg(op.normal, bh, tol=st, max_iter=mi, check_every=args.check_every),
                   out=dbl.x, iterations=info["loop_iterations"], warmup=1, min_repeats=1,
                   notes=f"tol={st:g}; kernel={op.kernel}; residual {dbl.residual.max().item():.1e}")
        if op64 is not None:
            info = {}
            mix = mixed_precision_cg(op, op64, bh, tol=st, inner_tol=args.inner_tol, max_iter=mi,
                                     check_every=args.check_every, info=info)
            note = (f"tol={st:g}; kernel={op.kernel}; inner_tol={args.inner_tol:g}; outer={info['outer']}; "
                    f"residual {mix.residual.max().item():.1e}; rel diff vs double {rel_error(mix.x, dbl.x):.1e}")
            if mix.residual.max().item() <= st:
                self.timed(ctx, "solve", f"mixed_precision_fused_{label}",
                           lambda op=op, op64=op64: mixed_precision_cg(op, op64, bh, tol=st,
                                                                       inner_tol=args.inner_tol, max_iter=mi,
                                                                       check_every=args.check_every),
                           out=mix.x, iterations=info["loop_iterations"], warmup=1, min_repeats=1, notes=note)
            else:
                self.emit(ctx, "solve", f"mixed_precision_fused_{label}", status="not_converged", notes=note)

    def _compiled(self, ctx, key, measurement, fns, x, ref, tol):
        if not hasattr(torch, "compile"):
            self.meta_compile[key] = "torch.compile not available"
            self.emit(ctx, measurement, "compiled", status="unavailable")
            return
        for name, fn in fns:
            t0 = time.perf_counter()
            try:
                compiled = torch.compile(fn, dynamic=False)
                out = compiled(x)
                synchronize(self.device)
                t_compile = time.perf_counter() - t0
                err = rel_error(out, ref)
            except Exception as exc:  # noqa: BLE001 - any compile failure is a result, not a crash
                self.meta_compile[f"{key}:{name}"] = f"failed: {_short(exc, 400)}"
                self.emit(ctx, measurement, name, status="compile_failed", notes=_short(exc))
                continue
            self.meta_compile[f"{key}:{name}"] = f"ok, compile {t_compile:.1f} s, err {err:.1e}"
            if not err <= tol:
                self.emit(ctx, measurement, name, status="mismatch", error=err)
                continue
            self.timed(ctx, measurement, name, lambda c=compiled: c(x), out=out, error=err,
                       notes=f"compile {t_compile:.1f} s")

    def _graph(self, ctx, measurement, variant, fn, x, ref, tol):
        try:
            static_x = x.clone()
            replay, static_out = cuda_graph(fn, static_x)
            replay()
            torch.cuda.synchronize()
            err = rel_error(static_out, ref)
        except RuntimeError as exc:
            self.emit(ctx, measurement, variant, status="error", notes=_short(exc))
            torch.cuda.synchronize()
            return
        if not err <= tol:
            self.emit(ctx, measurement, variant, status="mismatch", error=err)
            return
        self.timed(ctx, measurement, variant, replay, out=static_out, error=err)

    def _breakdown(self, ctx, key, w, var, eo, psi, psi_e, good, fast_normal):
        """Parts of the reference hopping term, and kernel launches per call."""
        D = w.geometry.D
        fwd = [torch.matmul(shift(psi, mu, 1), w._fwd[mu]) for mu in range(D)]
        bwd = [shift(torch.matmul(psi, w._bwd[mu]), mu, -1) for mu in range(D)]
        resolved_bwd = [l.resolve_conj().contiguous() for l in w._bwd]
        flat = psi.reshape(psi.shape[0], var.V, var.S, var.N)

        def rolls():
            for mu in range(D):
                shift(psi, mu, 1)
                shift(psi, mu, -1)

        def link_fwd():
            for mu in range(D):
                torch.matmul(psi, w._fwd[mu])

        def link_bwd_conjview():
            for mu in range(D):
                torch.matmul(psi, w._bwd[mu])

        def link_bwd_resolved():
            for mu in range(D):
                torch.matmul(psi, resolved_bwd[mu])

        def spin_combine():
            out = None
            for mu in range(D):
                term = (fwd[mu] + bwd[mu]) - torch.matmul(w.gammas[mu], fwd[mu] - bwd[mu])
                out = term if out is None else out + term
            return out

        def gather_all():
            return flat.index_select(1, var.idx_all)

        parts = [("rolls_2D", rolls), ("link_fwd_D", link_fwd), ("link_bwd_conjview_D", link_bwd_conjview),
                 ("link_bwd_resolved_D", link_bwd_resolved), ("spin_combine", spin_combine),
                 ("gather_2D_index_select", gather_all)]
        for name, fn in parts:
            kern = self._kernels(f"{key}:breakdown:{name}", fn)
            self.timed(ctx, "breakdown", name, fn, kernels=kern)
        # kernel launches per call of whole kernels (rows without timing, status kernel_count)
        whole = [("hopping:reference", lambda: w.hopping(psi)), ("eo_normal:reference", lambda: eo.normal(psi_e))]
        whole += [(f"hopping:{name}", (lambda fn=fn: fn(psi))) for name, fn in good.items()]
        if fast_normal is not None:
            whole.append(("eo_normal:gather_spinproj", lambda: fast_normal(psi_e)))
        b = psi_e.clone()
        whole.append(("cg_iteration:solver_cg(1 iter)", lambda: cg(eo.normal, b, tol=0.0, max_iter=1)))
        whole.append(("cg_iteration:bare(1 iter)", lambda: bare_cg(eo.normal, b, 1)))
        for name, fn in whole:
            kern = self._kernels(f"{key}:{name}", fn)
            if kern is None:
                continue
            measurement, variant = name.split(":", 1)
            self.emit(ctx, measurement, variant, status="kernel_count", kernels=kern,
                      notes="CUDA kernels per call" if self.device.type == "cuda" else "top-level aten ops per call")

    def _kernels(self, label, fn):
        if self.args.no_profile:
            return None
        try:
            n = count_kernels(fn, self.device)
        except Exception as exc:  # noqa: BLE001
            self.meta_kernels[label] = f"failed: {_short(exc)}"
            return None
        self.meta_kernels[label] = n
        return n


def _short(exc, n=160):
    s = f"{type(exc).__name__}: {exc}".replace("\n", " ").replace(",", ";")
    return s[:n]


def print_table(rows):
    print("\n=== summary (ms per call; cg_iteration: per iteration) ===")
    header = f"{'lattice':<10s} {'B':>3s} {'dtype':<10s} {'measurement':<14s} {'variant':<26s} " \
             f"{'ms':>10s} {'site/s':>10s} {'kern':>6s} status"
    print(header)
    print("-" * len(header))
    for r in rows:
        ms = f"{float(r['seconds_per_call']) * 1e3:10.4f}" if r["seconds_per_call"] else f"{'':10s}"
        su = f"{float(r['site_updates_per_s']):10.3e}" if r["site_updates_per_s"] else f"{'':10s}"
        print(f"{r['lattice']:<10s} {r['batch']:>3d} {r['dtype']:<10s} {r['measurement']:<14s} "
              f"{r['variant']:<26s} {ms} {su} {r['kernels_per_call']:>6s} {r['status']}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lattices", nargs="+", default=["4^4", "8^4", "12^3x24", "16^4"])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--dtypes", nargs="+", default=["complex128", "complex64"], choices=sorted(DTYPES))
    p.add_argument("--force-complex64", action="store_true",
                   help="benchmark complex64 even if the operators promote it to complex128")
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--kappa", type=float, default=0.153)
    p.add_argument("--beta", type=float, default=5.6)
    p.add_argument("--mass", type=float, default=0.1, help="staggered mass")
    p.add_argument("--device", default="auto", help="auto, cuda, cpu")
    p.add_argument("--mem-limit-gb", type=float, default=None,
                   help="skip configurations whose estimated peak exceeds this (default: 0.8 x GPU memory, "
                        "4 GB on CPU)")
    p.add_argument("--target-time", type=float, default=0.2, help="seconds per timing trial")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--min-repeats", type=int, default=3)
    p.add_argument("--max-repeats", type=int, default=200)
    p.add_argument("--cg-iters", type=int, default=50, help="CG iterations per timed solve (maximum)")
    p.add_argument("--cg-min-iters", type=int, default=5,
                   help="minimum CG iterations when --target-time shortens the solve on large lattices")
    p.add_argument("--check-every", type=int, default=10, help="lean_cg host convergence check interval")
    p.add_argument("--solve-tol", type=float, default=1e-10, help="tolerance of the 'solve' rows")
    p.add_argument("--solve-max-iter", type=int, default=5000, help="iteration cap of the 'solve' rows")
    p.add_argument("--inner-tol", type=float, default=1e-6, help="complex64 inner tolerance of mixed_precision_cg")
    p.add_argument("--half-modes", nargs="+", default=list(COLOUR_MODES), choices=COLOUR_MODES,
                   help="colour product modes of the half_lattice rows (the auto mode always runs)")
    p.add_argument("--fused-kernels", nargs="*", default=["loop", "compiled", "compiled_soa"],
                   choices=["loop", "compiled", "compiled_soa"],
                   help="neoqcd.fermions.fused kernels to benchmark: loop (pure torch), compiled (cuda on a "
                        "CUDA device, the host build on the CPU), compiled_soa (the same with the site-fastest "
                        "link layout; no solve rows); none given: no fused rows")
    p.add_argument("--skip-legacy", action="store_true",
                   help="drop the experimental hopping variants, solver_cg / bare / reference CG iterations "
                        "and the reference solve")
    p.add_argument("--compile-configs", nargs="*", default=["8^4:4", "16^4:4"],
                   help="lattice:batch pairs (complex128) where torch.compile is tried; none to skip")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-cuda-graphs", action="store_true")
    p.add_argument("--breakdown-config", default="8^4:4", help="lattice:batch of the hopping breakdown")
    p.add_argument("--no-profile", action="store_true", help="skip torch.profiler kernel counts")
    p.add_argument("--time-budget", type=float, default=None,
                   help="seconds; configurations not started within it are skipped")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="output/bench_fermions")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    start = time.perf_counter()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        # torch 2.0: set_device(torch.device("cuda")) raises without an index, so resolve one
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mem_limit_gb is None:
        if device.type == "cuda":
            args.mem_limit_gb = 0.8 * torch.cuda.get_device_properties(device).total_memory / 2**30
        else:
            args.mem_limit_gb = 4.0

    c64_ok, c64_detail = check_complex64(device)
    dtypes = list(args.dtypes)
    c64_note = "supported" if c64_ok else "not preserved by the operators"
    if "complex64" in dtypes and not c64_ok and not args.force_complex64:
        dtypes.remove("complex64")
        c64_note += "; complex64 skipped (use --force-complex64 to benchmark it anyway)"
    print(f"=== bench_fermions on {device} torch {torch.__version__}; complex64: {c64_note}", flush=True)
    print(f"    complex64 output dtypes: {c64_detail}", flush=True)

    compile_set = set() if args.no_compile else {parse_config(c) for c in (args.compile_configs or [])}
    breakdown = parse_config(args.breakdown_config) if args.breakdown_config else None

    meta = {
        "script": "main/bench_fermions.py",
        "argv": list(sys.argv[1:] if argv is None else argv),
        "args": vars(args),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "complex64_supported": c64_ok,
        "complex64_output_dtypes": c64_detail,
        "dtypes_benchmarked": dtypes,
        "torch_compile_available": hasattr(torch, "compile"),
    }
    meta.update(git_info())
    csv_path, meta_path = out_dir / "bench.csv", out_dir / "run_meta.json"
    skipped = []
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        bench = Bench(args, device, writer, handle)
        for dtype_name in dtypes:
            for lat in args.lattices:
                geo = parse_lattice(lat, args.N)
                for batch in args.batches:
                    label = f"{lat}:{batch}:{dtype_name}"
                    need = estimate_bytes(geo, batch, DTYPES[dtype_name]) / 2**30
                    if need > args.mem_limit_gb:
                        skipped.append(f"{label} (est. {need:.1f} GB > {args.mem_limit_gb:.1f} GB)")
                        print(f"--- skip {label}: estimated {need:.1f} GB", flush=True)
                        continue
                    if args.time_budget is not None and time.perf_counter() - start > args.time_budget:
                        skipped.append(f"{label} (time budget)")
                        print(f"--- skip {label}: time budget", flush=True)
                        continue
                    print(f"--- {label} (V = {geo.volume}, est. {need:.2f} GB) "
                          f"{time.perf_counter() - start:.0f} s", flush=True)
                    here = (lat, batch)
                    try:
                        bench.run_config(geo, batch, dtype_name,
                                         compile_here=(here in compile_set and dtype_name == "complex128"),
                                         breakdown_here=(here == breakdown and dtype_name == "complex128"))
                    except RuntimeError as exc:
                        skipped.append(f"{label} ({_short(exc)})")
                        print(f"--- {label} failed: {_short(exc)}", flush=True)
                        bench._free()
    print_table(bench.rows)
    meta.update({
        "skipped": skipped,
        "variant_errors": bench.meta_variants,
        "compile": bench.meta_compile,
        "kernel_counts": bench.meta_kernels,
        "fast_path": bench.meta_fast,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv": str(csv_path),
    })
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2, default=str)
    print(f"=== wrote {csv_path} and {meta_path} in {meta['wall_seconds']} s", flush=True)
    return {"csv": csv_path, "meta": meta_path, "rows": bench.rows, "meta_dict": meta}


if __name__ == "__main__":
    main()
