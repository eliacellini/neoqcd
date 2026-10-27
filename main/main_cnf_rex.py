"""Train a flow field with local Path Gradients on a persistent REX ladder.

Training never integrates an SNF. Evaluation reconstructs one with CF4 and
reports numerical integration diagnostics; its Haar logdet is approximate.
"""

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

from neoqcd.cnf import CNFProtocolLayer, integrate_flow
from neoqcd.cnf_io import DEFECT_PATCH_ARCHITECTURES, build_cnf, defect_patch_buffer, load_cnf_checkpoint, save_cnf_checkpoint
from neoqcd.cnf_training import CNFREXTrainer, DistributedCNFREXTrainer
from neoqcd.distributedPT import NEOREXProtocolWrapper
from neoqcd.mcmc import HBOR


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--phase", choices=("train", "evaluate"), default="train")
    p.add_argument("--architecture", choices=("clover-chair", "smearing", "hyper-smearing", "defect-hyper-smearing", "residual", "hyper-residual", "defect-hyper-residual", "defect-clover-chair"), default="clover-chair")
    p.add_argument("--domain", choices=("beta", "defect"), default="beta")
    p.add_argument("--D", type=int, default=2)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--parameter-min", type=float, default=0.5)
    p.add_argument("--parameter-max", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=6.0, help="Fixed beta for a defect ladder")
    p.add_argument("--defect-size", type=int, default=1)
    p.add_argument("--time-slice", type=int, default=1)
    p.add_argument("--space-slice", type=int, default=1)
    p.add_argument("--replicas", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4, help="Independent chains per rung; >=2 for PINN")
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--rho-init", type=float, default=1e-3)
    p.add_argument("--rho-functional", action="store_true")
    p.add_argument("--hyper-smearing-mode", choices=("shared", "per_link", "class", "per_site"), default="shared")
    p.add_argument("--residual-n-pt", type=int, default=0)
    p.add_argument("--loop-max", type=int, default=2,
                   help="Largest rectangle side of the clover-chair loops (1: plaquettes and chairs only)")
    p.add_argument("--residual-mask-mod", type=int, choices=(2, 4), default=2)
    p.add_argument("--orsteps", type=int, default=1)
    p.add_argument("--mcmc-steps", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                   help="cosine: decay from the run's lr to --lr-min over epochs 0..--epochs, set from the "
                        "epoch number, so a resumed run follows the same curve")
    p.add_argument("--lr-min", type=float, default=0.0)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--pg-chunk-size", type=int, default=0,
                   help="Chains per path-gradient chunk to bound memory (0: all chains of a rung at once)")
    p.add_argument("--pg-direction-batch", type=int, choices=(1, 2, 4, 8), default=1,
                   help="Generator directions per divergence term: more is faster and uses more memory")
    p.add_argument("--profile-epoch", type=int, default=0,
                   help="Profile this epoch on rank 0 into OUTPUT_DIR/profile_epoch<N>.txt (0: never)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, help="Resume training or load a field for evaluation")
    p.add_argument("--init-cfgs", type=Path,
                   help="Start the chains from a main_pt_obc.py configuration cache (data/cfgs/<run>): "
                        "rank r loads the chains it thermalized at the r-th rung of the same ladder")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--protocol-steps", type=int, default=4)
    p.add_argument("--integration-steps", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=4)
    args = p.parse_args(argv)
    for key in ("replicas", "batch_size", "hidden_dim", "depth", "mcmc_steps", "epochs", "save_every", "protocol_steps", "integration_steps", "eval_steps"):
        if getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.batch_size < 2 or args.replicas < 2:
        p.error("batch-size and replicas must be >=2")
    if args.D not in (2, 3, 4) or min(args.T, args.L) < 2 or args.T % 2 or args.L % 2:
        p.error("HBOR requires 2 <= D <= 4 and even T,L >=2")
    if args.loop_max < 1:
        p.error("--loop-max must be at least 1")
    if args.pg_chunk_size < 0:
        p.error("--pg-chunk-size must be non-negative")
    if args.warmup < 0 or args.orsteps < 0 or args.residual_n_pt < 0:
        p.error("warmup, orsteps and residual-n-pt must be nonnegative")
    if not all(math.isfinite(v) for v in (args.parameter_min, args.parameter_max, args.lr, args.beta, args.rho_init, args.grad_clip_norm)):
        p.error("numeric parameters must be finite")
    if not 0 <= args.parameter_min < args.parameter_max or args.lr <= 0 or args.beta <= 0 or args.grad_clip_norm < 0:
        p.error("require 0 <= parameter-min < parameter-max, lr,beta >0 and grad-clip-norm >=0")
    if args.domain == "defect":
        if args.D != 4 or args.parameter_max > 1:
            p.error("defect requires D=4 and parameters in [0,1]")
        if not (0 <= args.time_slice < args.T and args.defect_size > 0 and 0 <= args.space_slice <= args.L - args.defect_size):
            p.error("defect must fit inside the lattice")
    if args.phase == "evaluate" and args.checkpoint is None:
        p.error("evaluation requires --checkpoint")
    if not 0 <= args.lr_min <= args.lr:
        p.error("require 0 <= lr-min <= lr")
    if args.init_cfgs is not None and (args.checkpoint is not None or args.domain != "defect"):
        p.error("--init-cfgs starts a new defect run: it excludes --checkpoint and needs --domain defect")
    if args.rho_functional and args.architecture != "smearing":
        p.error("rho-functional requires architecture=smearing")
    if args.hyper_smearing_mode != "shared" and args.architecture not in {"hyper-smearing", "defect-hyper-smearing"}:
        p.error("hyper-smearing-mode requires a hyper-smearing architecture")
    if args.hyper_smearing_mode in {"class", "per_site"} and args.architecture != "defect-hyper-smearing":
        p.error(f"hyper-smearing-mode={args.hyper_smearing_mode} requires architecture=defect-hyper-smearing")
    if args.architecture in DEFECT_PATCH_ARCHITECTURES:
        if args.domain != "defect":
            p.error(f"{args.architecture} requires --domain defect")
        # Longer loops (residual convolutions, CloverChair) need a wider patch buffer.
        b = defect_patch_buffer(args.architecture, args.residual_n_pt, args.loop_max)
        if args.T < 2 * b or args.L < args.defect_size + 2 * b:
            p.error(f"{args.architecture} requires T >= {2 * b} and L >= defect-size + {2 * b}")
        if not (b <= args.time_slice <= args.T - b and b <= args.space_slice <= args.L - args.defect_size - b):
            p.error(f"{args.architecture} requires a {b}-site buffer around the defect")
    return args


@torch.no_grad()
def evaluate(field, trainer, args):
    """Evaluate using the same protocol wrapper as NEO-REX, with full actions."""
    c = trainer.parameters
    lo, hi = c[0], c[-1]
    layer = CNFProtocolLayer(field, n_steps=args.integration_steps)
    # A plain callable deliberately uses full actions, not the defect halo cache.
    action = lambda x, beta: trainer.action(x, beta)
    protocol = NEOREXProtocolWrapper(layer, action, args.protocol_steps)
    x0 = trainer.x[0].clone()
    _, coarse_j = integrate_flow(x0, lo, hi, field, n_steps=args.integration_steps)
    fine_x, fine_j = integrate_flow(x0, lo, hi, field, n_steps=2 * args.integration_steps)
    reverse_x, reverse_j = integrate_flow(fine_x, hi, lo, field, n_steps=2 * args.integration_steps)
    works = []
    for _ in range(args.eval_steps):
        trainer.warmup(1)
        x = trainer.x[0].clone()
        work = torch.zeros(x.shape[0], dtype=x.real.dtype, device=x.device)
        # NF + HBOR at each target rung; fixed-parameter MCMC adds no work.
        points = torch.linspace(float(lo), float(hi), args.protocol_steps + 1, device=x.device, dtype=x.real.dtype)
        for a, b in zip(points[:-1], points[1:]):
            x, dw = protocol(x, a.expand(x.shape[0]), b.expand(x.shape[0]), nf_only=True)
            work += dw
            update = HBOR(trainer.flow_pars, beta=float(b) if trainer.domain == "beta" else trainer.beta,
                          defect_par=1.0 if trainer.domain == "beta" else float(b))
            for _ in range(trainer.updates_per_replica):
                x = update(x, trainer.flow_pars.mask, trainer.defect_mask)
        works.append(work.cpu())
    w = torch.cat(works)
    lw = -w
    return {
        "samples": w.numel(), "work_mean": w.mean().item(), "work_variance": w.var(unbiased=False).item(),
        "ess_fraction": (torch.exp(2 * torch.logsumexp(lw, 0) - torch.logsumexp(2 * lw, 0)) / w.numel()).item(),
        "delta_f": (-torch.logsumexp(lw, 0) + math.log(w.numel())).item(),
        "logdet_refinement_max_abs": (coarse_j - fine_j).abs().max().item(),
        "reverse_configuration_max_abs": (reverse_x - x0).abs().max().item(),
        "reverse_logdet_max_abs": (fine_j + reverse_j).abs().max().item(),
        "jacobian": "CF4 integrated divergence; finite-step approximation",
    }


def _distributed(args):
    """(rank, world_size, device) under torchrun, with one replica per rank; None otherwise."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return None
    if args.phase != "train":
        raise SystemExit("distributed runs train only; evaluate with a single process")
    if args.device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    return dist.get_rank(), dist.get_world_size(), device


def _run_info(argv, world_size):
    """Code version and launch of this run, rewritten at every (re)start."""
    def git(*command):
        try:
            return subprocess.run(["git", *command], cwd=Path(__file__).resolve().parents[1],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {
        "commit": git("rev-parse", "HEAD"),
        "uncommitted_changes": git("status", "--porcelain", "--untracked-files=no"),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "world_size": world_size,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _profiled_step(trainer, path):
    """One training step under the profiler: operators by device time.

    Memory is not profiled: recording it for the thousands of small operators of the
    loss exhausts the GPU. The per-epoch metrics already report the peak memory.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    on_gpu = torch.cuda.is_available() and str(trainer.flow_pars.device).startswith("cuda")
    if on_gpu:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profiler:
        metrics = trainer.step()
    averages = profiler.key_averages()
    device = "cuda" if on_gpu else "cpu"
    with open(path, "w") as handle:
        handle.write(f"# operators by self {device} time\n")
        handle.write(averages.table(sort_by=f"self_{device}_time_total", row_limit=40))
        handle.write(f"\n# operators by total {device} time, including their children\n")
        handle.write(averages.table(sort_by=f"{device}_time_total", row_limit=40))
        regions = [event for event in averages if event.key.startswith(("pg.", "rex.", "residual."))]
        handle.write(f"\n# annotated regions: calls, cpu total ms, {device} total ms\n")
        for event in sorted(regions, key=lambda e: -e.cpu_time_total):
            gpu_ms = getattr(event, "cuda_time_total", 0) / 1e3 if on_gpu else 0.0
            handle.write(f"{event.key:40s} {event.count:8d} {event.cpu_time_total / 1e3:12.1f} {gpu_ms:12.1f}\n")
    return metrics


def _thermalized_cfgs(directory, config, ladder, rank, world_size):
    """This rank's chains from a main_pt_obc.py cache, after checking it describes this run."""
    if not (directory / "DONE").exists():
        raise ValueError(f"{directory} is not a completed configuration cache")
    key = json.loads((directory / "meta.json").read_text())["key"]
    expected = {"D": config["D"], "T": config["T"], "L": config["L"], "N": 3, "beta": config["beta"],
                "defect_size": config["defect_size"], "time_slice": config["time_slice"],
                "space_slice": config["space_slice"], "batch_size": config["batch_size"],
                "world_size": world_size, "parameter_name": "bc", "orsteps": config["orsteps"]}
    for name, value in expected.items():
        if key.get(name) != value:
            raise ValueError(f"configuration cache {name}: cached {key.get(name)!r}, this run {value!r}")
    if not torch.allclose(torch.tensor(key["parameter_ladder"], dtype=torch.float64), ladder.cpu(), rtol=0, atol=1e-12):
        raise ValueError("configuration cache was thermalized on another ladder")
    payload = torch.load(directory / f"rank_{rank:03d}.pt", map_location="cpu", weights_only=True)
    if int(payload["rank"]) != rank:
        raise ValueError(f"configuration cache file of rank {payload['rank']} read by rank {rank}")
    return payload["cfgs"]


def _learning_rate(lr, args, epoch):
    """Learning rate of a 0-based epoch: lr, or its cosine decay to lr_min at the last epoch."""
    if args.lr_schedule == "constant":
        return lr
    return args.lr_min + 0.5 * (lr - args.lr_min) * (1 + math.cos(math.pi * epoch / args.epochs))


def _shard_path(checkpoint, rank):
    return checkpoint.with_name(f"{checkpoint.stem}.rank{rank}{checkpoint.suffix}")


def main(argv=None):
    args = parse_args(argv)
    distributed = _distributed(args)
    rank, world_size, device = distributed if distributed else (0, 1, args.device)
    torch.manual_seed(args.seed)
    if args.checkpoint:
        field, pars, mask, payload = load_cnf_checkpoint(args.checkpoint, device)
        config = payload["config"]
    else:
        config = {key: value for key, value in vars(args).items() if not isinstance(value, Path)}
        config["time_normalization"] = "unit_interval"
        field, pars, mask = build_cnf(config, device)
        payload = None
    if distributed and config["replicas"] != world_size:
        raise SystemExit(f"one replica per rank: --replicas {config['replicas']} on {world_size} ranks")
    # Independent Markov chains on every rank; the field itself is broadcast from rank 0.
    torch.manual_seed(args.seed + rank)
    ladder = torch.linspace(config["parameter_min"], config["parameter_max"], config["replicas"], dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(field.parameters(), lr=config["lr"])
    options = dict(domain=config["domain"], beta=config["beta"], defect_mask=mask, batch_size=config["batch_size"],
                   updates_per_replica=config["mcmc_steps"], grad_clip_norm=config["grad_clip_norm"],
                   chunk_size=args.pg_chunk_size or None, directions_per_term=args.pg_direction_batch)
    if distributed:
        trainer = DistributedCNFREXTrainer(field, pars, ladder, optimizer, rank=rank, world_size=world_size, **options)
    else:
        trainer = CNFREXTrainer(field, pars, ladder, optimizer, **options)
    start = 0
    if payload and "trainer" in payload:
        if distributed:
            trainer.load_state_dict(payload["trainer"], torch.load(_shard_path(args.checkpoint, rank), weights_only=False))
        else:
            trainer.load_state_dict(payload["trainer"])
            torch.set_rng_state(payload["rng"])
            if args.device == "cuda" and "cuda_rng" in payload:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        start = payload["epoch"]
    else:
        if args.init_cfgs is not None:
            if not distributed:
                raise SystemExit("--init-cfgs needs one replica per rank under torchrun")
            # A new trainer has rank r's chains at ladder[r], as in the thermalization run.
            cfgs = _thermalized_cfgs(args.init_cfgs, config, ladder, rank, world_size)
            if tuple(cfgs.shape) != tuple(trainer.x.shape):
                raise ValueError(f"cached chains {tuple(cfgs.shape)}, trainer chains {tuple(trainer.x.shape)}")
            trainer.x.copy_(cfgs.to(device=trainer.x.device, dtype=trainer.x.dtype))
        trainer.warmup(config["warmup"])
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "evaluate":
        result = evaluate(field, trainer, args)
        (args.output_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    metrics_path = args.output_dir / "metrics.jsonl"
    if rank == 0 and metrics_path.exists() and metrics_path.stat().st_size:
        if not args.checkpoint or args.checkpoint.resolve().parent != args.output_dir.resolve():
            raise FileExistsError(f"Refusing to append to another training run: {args.output_dir}")
        last = json.loads(metrics_path.read_text().splitlines()[-1])
        if last["epoch"] != start:
            raise ValueError("Metrics and checkpoint progress differ; resume into a new output directory")
    if rank == 0:
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (args.output_dir / "run_info.json").write_text(json.dumps(_run_info(argv, world_size), indent=2) + "\n")
    output = metrics_path.open("a") if rank == 0 else None
    try:
        for epoch in range(start, args.epochs):
            start_time = time.perf_counter()
            lr = _learning_rate(config["lr"], args, epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if rank == 0 and epoch + 1 == args.profile_epoch:
                metrics = _profiled_step(trainer, args.output_dir / f"profile_epoch{epoch + 1}.txt")
            else:
                metrics = trainer.step()
            metrics = {"epoch": epoch + 1, **metrics, "lr": lr, "epoch_s": round(time.perf_counter() - start_time, 3)}
            if rank == 0:
                line = json.dumps(metrics)
                output.write(line + "\n")
                output.flush()
                print(line, flush=True)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                checkpoint = args.output_dir / "cnf_checkpoint.pt"
                if distributed:
                    # Shards first, then the main checkpoint that refers to them.
                    torch.save(trainer.shard_state_dict(), _shard_path(checkpoint, rank))
                    dist.barrier()
                if rank == 0:
                    save_cnf_checkpoint(checkpoint, field, config, epoch=epoch + 1, trainer=trainer)
    finally:
        if output is not None:
            output.close()
        if distributed:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
