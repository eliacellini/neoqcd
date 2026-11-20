"""Path Gradient training of a flow field on a fermionic REX ladder (N_f = 2 Wilson), one rung per rank.

The fermionic port of main/main_cnf_rex.py (training phase only; doc/cnf.md, doc/fermions.md "PG
training with fermions"). The field (--architecture and the field flags of main_cnf_rex.py) is
trained with the unchanged local Path Gradient loss on a persistent REX ensemble of the joint
action S_g(U; beta, bc) + S_pf(U, phi) (`neoqcd.fermions.pg`): one epoch = --mcmc-steps fermionic
HMC trajectories of every replica at its bc label (fresh phi per trajectory), solve-free gauge-only
swaps, the PG loss of every chain at its post-swap label with the phi of its last trajectory (the
pure-gauge loss on the ladder's gauge action plus the pseudofermion term, two CG solves per chain),
one Adam step. The defect is the gauge-only bc defect, the time variable of the field is bc.

Launch: under torchrun (WORLD_SIZE > 1) one rung per rank (`pg.DistributedFermionCNFREXTrainer`:
`ParallelTempering` + `FermionREXTarget`, field broadcast, gradients all-reduced), --replicas must
equal the number of ranks, --batch-size chains per rank; --device cpu (Gloo) or cuda/auto
(cuda:LOCAL_RANK, NCCL; --allow-shared-gpu for tests). Without torchrun one process holds the
--replicas x --batch-size replicas (`pg.FermionCNFREXTrainer` on `rex.FermionREX`).

Flags: the field and optimiser flags of main_cnf_rex.py (--architecture, --replicas, --batch-size,
--hidden-dim, --depth, --rho-init, --rho-functional, --hyper-smearing-mode, --residual-n-pt,
--loop-max, --residual-mask-mod, --residual-mode, --mcmc-steps, --warmup, --epochs, --lr,
--lr-schedule, --lr-min, --grad-clip-norm, --pg-chunk-size, --pg-direction-batch, --save-every,
--swap-parity, --output-dir, --checkpoint; the ladder --parameter-min/--parameter-max (aliases
--bc-min/--bc-max) or --parameter-ladder (aliases --ladder, --ladder-file)), and the physics, HMC,
solver and defect flags of main_snf_fermions.py (--D --T --L --N --beta --kappa --time-sign
--solver-tol --action-tol --max-iter --no-reuse-guess --on-unconverged --fast --precision
--colour-product --check-every --inner-tol --max-outer --tau --n-steps --n-gauge-substeps
--integrator --defect-size --defect-time-slice (alias --time-slice) --defect-space-slice (alias
--space-slice) --start-config --therm --seed --device --resume --overwrite --quiet), plus
--loss-tol (the two loss solves; default --solver-tol). The domain is always the bc defect
(--defect-shape cube, `utils.Defect`, the defect of main_pt_obc.py and of the CNF checkpoint); N = 3,
Wilson fermions. --therm trajectories at fixed labels and --warmup REX steps (with swaps) precede a
fresh run. The field's defect mask (`cnf_io.build_cnf`) and the action's
(`main_snf_fermions.build_defect_mask`) are built separately and checked to agree.

Outputs in --output-dir:
    metrics.jsonl   (rank 0) one line per epoch: per rung (lists over the ladder) losses_by_rung,
                    pinn_by_rung = V Var(xi_g + xi_f), pinn_gauge_by_rung, pinn_fermion_by_rung,
                    pinn_cov_by_rung = V Cov(xi_g, xi_f), residuals[_gauge|_fermion]_by_rung,
                    spf_per_component_by_rung (<S_pf>/n, 1 in equilibrium), plaquette_by_rung;
                    swap_acceptance_by_edge, swap_acceptance, hmc_acceptance; trajectory_* and loss_*
                    solves, cg_iter (sum of the per-solve maximum) and cg_applications (operator
                    applications, the real cost with --fast), max_residual (largest true relative CG
                    residual, NaN-sticky), unconverged_solves (all summed over the ranks);
                    sweep_s, loss_s, gauge_loss_s, fermion_loss_s (slowest rank), peak_memory_gb;
                    grad_norm, optimizer_updated, lr, epoch_s;
    config.json     the checkpoint's config: the keys of main_cnf_rex.py (domain "defect") and a
                    "fermions" block (type, kappa, time_sign, HMC, solver and defect settings,
                    fermion_term "joint_pseudofermion");
    run_info.json   commit, uncommitted changes, command line, world size of each launch;
    cnf_checkpoint.pt  the unchanged `neoqcd_cnf` version-1 checkpoint (`cnf_io.save_cnf_checkpoint`:
                    config, field, epoch, trainer state with the shard slot), so `cnf_io.load_cnf_checkpoint`
                    and main/convert_cnf_to_smearing_flow.py read it as a pure-gauge field;
    cnf_checkpoint.rank<r>.slot<s>.pt  rank r's chains, labels and generators (two slots: the shards
                    of a checkpoint are written first, atomically, into the slot the main checkpoint
                    does not reference, then the main checkpoint commits the slot; a kill at any point
                    leaves a consistent pair).
Resume: --resume (or --checkpoint PATH) continues from cnf_checkpoint.pt; the saved config must equal
the one of the command line except the run-control flags (--epochs, --save-every, --pg-chunk-size,
--pg-direction-batch, --device, --quiet); metrics.jsonl lines past the checkpoint are dropped. A
previous run in --output-dir is refused without --resume or --overwrite. The continuation repeats an
uninterrupted run bitwise (same device and thread count).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from main_hmc import choose_device  # noqa: E402
from main_rex_fermions_distributed import choose_rank_device, init_process_group  # noqa: E402
from main_snf_fermions import (  # noqa: E402
    add_physics_args,
    build_defect_mask,
    clean_json,
    fermion_kwargs,
    start_links,
    validate_common,
)

from neoqcd.fermions._gauge import parse_ladder, validate_ladder  # noqa: E402
from neoqcd.fermions._gauge_cnf import (  # noqa: E402
    DEFECT_PATCH_ARCHITECTURES,
    build_cnf,
    defect_patch_buffer,
    load_cnf_checkpoint,
    save_cnf_checkpoint,
)
from neoqcd.fermions.io import _atomic_write  # noqa: E402
from neoqcd.fermions.lattice import Geometry  # noqa: E402
from neoqcd.fermions.pg import FERMION_TERM, DistributedFermionCNFREXTrainer, FermionCNFREXTrainer  # noqa: E402

ARCHITECTURES = ("clover-chair", "smearing", "hyper-smearing", "defect-hyper-smearing", "residual", "hyper-residual",
                 "defect-hyper-residual", "defect-clover-chair")
CHECKPOINT = "cnf_checkpoint.pt"
# flags that may change between a run and its continuation (not compared on resume)
RUN_CONTROL = ("epochs", "save_every", "pg_chunk_size", "pg_direction_batch", "device")


# ---- arguments ----------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PG training of a flow field on a fermionic REX ladder.",
                                allow_abbrev=False)
    add_physics_args(p)
    p.add_argument("--time-slice", dest="defect_time_slice", type=int, default=argparse.SUPPRESS,
                   help="Alias of --defect-time-slice (main_cnf_rex.py / main_pt_obc.py name).")
    p.add_argument("--space-slice", dest="defect_space_slice", type=int, default=argparse.SUPPRESS,
                   help="Alias of --defect-space-slice (main_cnf_rex.py / main_pt_obc.py name).")
    p.add_argument("--architecture", choices=ARCHITECTURES, default="clover-chair")
    p.add_argument("--parameter-min", "--bc-min", dest="parameter_min", type=float, default=0.0)
    p.add_argument("--parameter-max", "--bc-max", dest="parameter_max", type=float, default=1.0)
    p.add_argument("--parameter-ladder", "--ladder", "--ladder-file", dest="parameter_ladder", default=None,
                   help="Explicit bc values, comma-separated or a JSON file of main/optimal_ladder.py; one per "
                        "replica, from --parameter-min to --parameter-max. Default: evenly spaced.")
    p.add_argument("--swap-parity", choices=("alternating", "even_odd"), default="alternating")
    p.add_argument("--replicas", type=int, default=4, help="Rungs; = the number of ranks under torchrun.")
    p.add_argument("--batch-size", type=int, default=4, help="Chains per rung (per rank); >= 2 for PINN.")
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--rho-init", type=float, default=1e-3)
    p.add_argument("--rho-functional", action="store_true")
    p.add_argument("--hyper-smearing-mode", choices=("shared", "per_link", "class", "per_site"), default="shared")
    p.add_argument("--residual-n-pt", type=int, default=0)
    p.add_argument("--loop-max", type=int, default=2)
    p.add_argument("--residual-mask-mod", type=int, choices=(2, 4), default=2)
    p.add_argument("--residual-mode", choices=("shared", "class"), default="shared")
    p.add_argument("--mcmc-steps", type=int, default=1, help="HMC trajectories per replica per epoch (before the swaps).")
    p.add_argument("--warmup", type=int, default=20, help="REX steps (with swaps) before the first epoch.")
    p.add_argument("--epochs", type=int, default=100, help="Total epochs (also on resume).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--lr-min", type=float, default=0.0)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--pg-chunk-size", type=int, default=0, help="Chains per PG chunk (memory only; 0: all).")
    p.add_argument("--pg-direction-batch", type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument("--loss-tol", type=float, default=None,
                   help="CG tolerance of the two loss solves (default --solver-tol).")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="Resume from this cnf_checkpoint.pt.")
    p.add_argument("--allow-shared-gpu", action="store_true", help="Ranks may share a GPU (tests and debugging).")
    args = p.parse_args(argv)
    validate(args)
    return args


def validate(args):
    validate_common(args)
    if args.N != 3:
        raise SystemExit("the flow fields are SU(3): --N 3")
    if args.fermions != "wilson":
        raise SystemExit("PG training with fermions is N_f = 2 Wilson only (--fermions wilson)")
    if args.defect_shape != "cube":
        raise SystemExit("--defect-shape cube only: the CNF checkpoint records the defect of utils.Defect "
                         "(main_pt_obc.py), which the field and the evaluation runners rebuild")
    if args.fast:
        raise SystemExit("--fast is not yet supported by the PG loss")
    for key in ("replicas", "batch_size", "hidden_dim", "depth", "mcmc_steps", "epochs", "save_every"):
        if getattr(args, key) < 1:
            raise SystemExit(f"--{key.replace('_', '-')} must be positive")
    if args.batch_size < 2 or args.replicas < 2:
        raise SystemExit("--batch-size and --replicas must be >= 2")
    if args.loop_max < 1 or args.pg_chunk_size < 0 or args.warmup < 0 or args.residual_n_pt < 0:
        raise SystemExit("--loop-max >= 1, --pg-chunk-size, --warmup, --residual-n-pt >= 0")
    values = (args.parameter_min, args.parameter_max, args.lr, args.beta, args.rho_init, args.grad_clip_norm)
    if not all(math.isfinite(v) for v in values):
        raise SystemExit("numeric parameters must be finite")
    if not 0 <= args.parameter_min < args.parameter_max <= 1:
        raise SystemExit("require 0 <= --parameter-min < --parameter-max <= 1 (bc ladder)")
    if args.lr <= 0 or args.grad_clip_norm < 0 or not 0 <= args.lr_min <= args.lr:
        raise SystemExit("require --lr > 0, --grad-clip-norm >= 0 and 0 <= --lr-min <= --lr")
    if args.loss_tol is not None and not args.loss_tol > 0:
        raise SystemExit("--loss-tol must be positive")
    if args.parameter_ladder is not None:
        try:
            args.parameter_ladder = validate_ladder(parse_ladder(args.parameter_ladder), args.replicas,
                                                    args.parameter_min, args.parameter_max)
        except (ValueError, KeyError, OSError) as err:
            raise SystemExit(f"--parameter-ladder: {err}")
    if args.rho_functional and args.architecture != "smearing":
        raise SystemExit("--rho-functional requires --architecture smearing")
    if args.hyper_smearing_mode != "shared" and args.architecture not in {"hyper-smearing", "defect-hyper-smearing"}:
        raise SystemExit("--hyper-smearing-mode requires a hyper-smearing architecture")
    if args.residual_mode == "class" and args.architecture != "defect-hyper-residual":
        raise SystemExit("--residual-mode class requires --architecture defect-hyper-residual")
    if args.hyper_smearing_mode in {"class", "per_site"} and args.architecture != "defect-hyper-smearing":
        raise SystemExit(f"--hyper-smearing-mode {args.hyper_smearing_mode} requires defect-hyper-smearing")
    ts, ss, size = args.defect_time_slice, args.defect_space_slice, args.defect_size
    if not (0 <= ts < args.T and 0 <= ss <= args.L - size):
        raise SystemExit("the defect must fit inside the lattice")
    if args.architecture in DEFECT_PATCH_ARCHITECTURES:
        if args.D != 4:
            raise SystemExit(f"{args.architecture} requires --D 4")
        b = defect_patch_buffer(args.architecture, args.residual_n_pt, args.loop_max)
        if args.T < 2 * b or args.L < size + 2 * b:
            raise SystemExit(f"{args.architecture} requires T >= {2 * b} and L >= defect-size + {2 * b}")
        if not (b <= ts <= args.T - b and b <= ss <= args.L - size - b):
            raise SystemExit(f"{args.architecture} requires a {b}-site buffer around the defect")


def ladder_values(args):
    if args.parameter_ladder is not None:
        return [float(v) for v in args.parameter_ladder]
    return torch.linspace(args.parameter_min, args.parameter_max, args.replicas, dtype=torch.float64).tolist()


def build_config(args):
    """The checkpoint config: the keys of main_cnf_rex.py (domain "defect") plus the "fermions" block."""
    return {
        "architecture": args.architecture, "domain": "defect", "D": args.D, "T": args.T, "L": args.L, "N": args.N,
        "parameter_min": float(args.parameter_min), "parameter_max": float(args.parameter_max),
        "parameter_ladder": args.parameter_ladder, "swap_parity": args.swap_parity, "beta": float(args.beta),
        "defect_size": args.defect_size, "time_slice": args.defect_time_slice, "space_slice": args.defect_space_slice,
        "replicas": args.replicas, "batch_size": args.batch_size, "hidden_dim": args.hidden_dim, "depth": args.depth,
        "rho_init": float(args.rho_init), "rho_functional": bool(args.rho_functional),
        "hyper_smearing_mode": args.hyper_smearing_mode, "residual_n_pt": args.residual_n_pt,
        "loop_max": args.loop_max, "residual_mask_mod": args.residual_mask_mod, "residual_mode": args.residual_mode,
        "orsteps": 0,  # no HB/OR here: the replicas are updated by the fermionic HMC
        "mcmc_steps": args.mcmc_steps, "warmup": args.warmup, "epochs": args.epochs, "lr": float(args.lr),
        "lr_schedule": args.lr_schedule, "lr_min": float(args.lr_min), "grad_clip_norm": float(args.grad_clip_norm),
        "pg_chunk_size": args.pg_chunk_size, "pg_direction_batch": args.pg_direction_batch, "seed": args.seed,
        "device": args.device, "save_every": args.save_every, "time_normalization": "unit_interval",
        "fermions": {
            "type": "wilson", "n_flavours": 2, "kappa": float(args.kappa), "time_sign": float(args.time_sign),
            "fermion_term": FERMION_TERM,
            "hmc": {"tau": float(args.tau), "n_steps": args.n_steps, "n_gauge_substeps": args.n_gauge_substeps,
                    "integrator": list(args.integrator_levels), "trajectories_per_epoch": args.mcmc_steps,
                    "therm": args.therm, "start_config": args.start_config},
            "solver": {"solver_tol": float(args.solver_tol), "action_tol": args.action_tol, "max_iter": args.max_iter,
                       "reuse_guess": bool(args.reuse_guess), "on_unconverged": args.on_unconverged,
                       "loss_tol": float(args.solver_tol if args.loss_tol is None else args.loss_tol),
                       "fast": bool(args.fast), "precision": args.precision if args.fast else None,
                       "colour_product": args.colour_product if args.fast else None,
                       "check_every": args.check_every if args.fast else None,
                       "inner_tol": args.inner_tol if args.fast and args.precision == "mixed" else None,
                       "max_outer": args.max_outer if args.fast and args.precision == "mixed" else None},
            "defect": {"shape": args.defect_shape, "size": args.defect_size, "time_slice": args.defect_time_slice,
                       "space_slice": args.defect_space_slice},
        },
    }


def config_differences(saved, current):
    """'key: saved -> current' for every key that is not run control."""
    keys = sorted((set(saved) | set(current)) - set(RUN_CONTROL))
    return [f"{k}: {saved.get(k, '<absent>')!r} -> {current.get(k, '<absent>')!r}"
            for k in keys if saved.get(k, "<absent>") != current.get(k, "<absent>")]


def shard_path(checkpoint, rank, slot):
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(f"{checkpoint.stem}.rank{rank}.slot{slot}{checkpoint.suffix}")


def learning_rate(config, epochs, epoch):
    """Learning rate of a 0-based epoch (main_cnf_rex.py `_learning_rate`)."""
    if config["lr_schedule"] == "constant":
        return config["lr"]
    return config["lr_min"] + 0.5 * (config["lr"] - config["lr_min"]) * (1 + math.cos(math.pi * epoch / epochs))


def run_info(argv, world_size, device):
    def git(*command):
        try:
            return subprocess.run(["git", *command], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"script": "main/main_pg_fermions.py", "commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": git("status", "--porcelain", "--untracked-files=no"),
            "argv": list(sys.argv[1:] if argv is None else argv), "world_size": world_size,
            "device": str(device), "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


class _CommittedState:
    """The trainer's state_dict plus the shard slot and epoch it commits (for save_cnf_checkpoint)."""

    def __init__(self, trainer, slot, epoch):
        self.trainer, self.slot, self.epoch = trainer, slot, epoch

    def state_dict(self):
        return dict(self.trainer.state_dict(), shard_slot=int(self.slot), shard_epoch=int(self.epoch))


# ---- the run ------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    own_group = False
    if world > 1:
        device, backend = choose_rank_device(args.device, args.allow_shared_gpu)
        own_group = init_process_group(backend, device)
    else:
        device = choose_device(args.device)
    try:
        return run(args, argv, device, world)
    finally:
        if own_group and dist.is_initialized():
            dist.destroy_process_group()


def _start(args, out):
    """The checkpoint to resume, or None (fresh start): the same read-only decision on every rank.

    --checkpoint PATH resumes PATH; --resume resumes --output-dir/cnf_checkpoint.pt if it exists and
    otherwise starts afresh (a job killed before its first checkpoint holds nothing resumable);
    without either, a previous run in --output-dir (checkpoint or metrics) is refused unless
    --overwrite."""
    checkpoint = out / CHECKPOINT
    metrics = out / "metrics.jsonl"
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise SystemExit(f"--checkpoint {args.checkpoint} does not exist")
        return args.checkpoint
    if args.resume:
        return checkpoint if checkpoint.exists() else None
    previous = checkpoint.exists() or (metrics.exists() and metrics.stat().st_size > 0)
    if previous and not args.overwrite:
        raise SystemExit(f"{out} holds a previous run: use --resume or --overwrite")
    return None


def run(args, argv, device, world):
    distributed = world > 1
    rank = dist.get_rank() if distributed else 0
    out = args.output_dir
    resume_from = _start(args, out)
    geometry = Geometry(args.D, args.T, args.L, args.N)
    config = build_config(args)
    log = (lambda *m: print(*m, flush=True)) if rank == 0 and not args.quiet else (lambda *m: None)

    torch.manual_seed(args.seed)  # the field's initialisation (rank 0's is broadcast)
    payload = None
    if resume_from is not None:
        field, _, field_mask, payload = load_cnf_checkpoint(resume_from, device)
        saved = payload["config"]
        if not isinstance(saved.get("fermions"), dict) or saved["fermions"].get("fermion_term") != FERMION_TERM:
            raise SystemExit(f"{resume_from} is not a checkpoint of this driver (no fermions block)")
        differences = config_differences(saved, config)
        if differences:
            raise SystemExit(f"--resume: the checkpoint's settings differ from this run's: {'; '.join(differences)}")
        config = dict(saved, **{k: config[k] for k in RUN_CONTROL})
    else:
        field, _, field_mask = build_cnf(config, device)
    if distributed and config["replicas"] != world:
        raise SystemExit(f"one rung per rank: --replicas {config['replicas']} on {world} ranks")
    ladder = ladder_values(args)
    defect_mask = build_defect_mask(args, geometry)
    C = config["batch_size"]
    total = C * (world if distributed else config["replicas"])
    U0 = None
    if payload is None:
        U0, source = start_links(args, geometry, device, total, args.beta, args.kappa, 1.0)
        if distributed:
            U0 = U0[rank * C:(rank + 1) * C].clone()
    optimizer = torch.optim.Adam(field.parameters(), lr=config["lr"])
    kw = dict(geometry=geometry, beta=args.beta, kappa=args.kappa, defect_mask=defect_mask, ladder=ladder,
              batch_size=C, tau=args.tau, n_steps=args.n_steps, integrator=args.integrator_levels,
              n_gauge_substeps=args.n_gauge_substeps, updates_per_replica=config["mcmc_steps"],
              swap_parity=config["swap_parity"], grad_clip_norm=config["grad_clip_norm"],
              chunk_size=args.pg_chunk_size or None, directions_per_term=args.pg_direction_batch,
              loss_tol=config["fermions"]["solver"]["loss_tol"], seed=config["seed"], device=device, U=U0,
              field_defect_mask=field_mask, **fermion_kwargs(args))
    if distributed:
        trainer = DistributedFermionCNFREXTrainer(field, optimizer, rank=rank, world_size=world, **kw)
    else:
        trainer = FermionCNFREXTrainer(field, optimizer, **kw)

    checkpoint = out / CHECKPOINT
    start, slot = 0, 1
    if payload is not None:
        state = payload.get("trainer")
        if state is None:
            raise SystemExit(f"{resume_from} holds no trainer state")
        slot, start = int(state["shard_slot"]), int(payload["epoch"])
        shard_file = shard_path(resume_from, rank, slot)
        shard = torch.load(shard_file, map_location="cpu", weights_only=True)
        if int(shard.get("epoch", -1)) != start or int(state.get("shard_epoch", -1)) != start:
            raise SystemExit(f"{shard_file} holds epoch {shard.get('epoch')}, the checkpoint epoch {start}")
        trainer.load_state_dict(state, shard)
        log(f"resumed from {resume_from} at epoch {start}")
    else:
        log(f"{'distributed ' + str(world) + ' ranks' if distributed else 'single process'}, device {device}, "
            f"ladder {ladder} x {C} chains, {geometry.lattice} SU(3), beta {args.beta}, kappa {args.kappa}, "
            f"defect L_d={args.defect_size} ({int(defect_mask[1].sum())} links), start {source} -> {out}")
        trainer.thermalize(args.therm)
        trainer.warmup(config["warmup"])

    metrics_path = out / "metrics.jsonl"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if payload is None and checkpoint.exists():
            # a fresh start over a previous run (--overwrite, or --resume after a kill before the
            # first checkpoint of an earlier --overwrite): its main checkpoint must not be resumed
            log(f"{checkpoint}: a previous run's checkpoint, removed (fresh start)")
            checkpoint.unlink()
        kept = []
        if payload is not None and metrics_path.exists():
            kept = [line for line in metrics_path.read_text().splitlines()
                    if line.strip() and json.loads(line)["epoch"] <= start]
        metrics_path.write_text("".join(line + "\n" for line in kept))
        (out / "config.json").write_text(json.dumps(clean_json(config), indent=2) + "\n")
        (out / "run_info.json").write_text(json.dumps(run_info(argv, world, device), indent=2) + "\n")
    if distributed:
        dist.barrier()

    def save(epoch):
        nonlocal slot
        new = 1 - slot
        shard = dict(trainer.shard_state_dict(), epoch=int(epoch))
        _atomic_write(shard_path(checkpoint, rank, new), lambda tmp: torch.save(shard, tmp))
        if distributed:
            dist.barrier()
        if rank == 0:
            save_cnf_checkpoint(checkpoint, field, config, epoch=epoch, trainer=_CommittedState(trainer, new, epoch))
        if distributed:
            dist.barrier()
        slot = new

    output = metrics_path.open("a") if rank == 0 else None
    try:
        for epoch in range(start, args.epochs):
            t0 = time.perf_counter()
            lr = learning_rate(config, args.epochs, epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            metrics = trainer.step()
            metrics = {"epoch": epoch + 1, **metrics, "lr": lr, "epoch_s": round(time.perf_counter() - t0, 4)}
            if rank == 0:
                line = json.dumps(clean_json(metrics))
                output.write(line + "\n")
                output.flush()
                if not args.quiet:
                    print(line, flush=True)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                save(epoch + 1)
    finally:
        if output is not None:
            output.close()
    return {"out": str(out), "rank": rank, "epochs": max(start, args.epochs)}


if __name__ == "__main__":
    main()
