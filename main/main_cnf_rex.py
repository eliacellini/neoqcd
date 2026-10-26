"""Train a flow field with local Path Gradients on a persistent REX ladder.

Training never integrates an SNF. Evaluation reconstructs one with CF4 and
reports numerical integration diagnostics; its Haar logdet is approximate.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from neoqcd.cnf import CNFProtocolLayer, integrate_flow
from neoqcd.cnf_io import build_cnf, load_cnf_checkpoint, save_cnf_checkpoint
from neoqcd.cnf_training import CNFREXTrainer
from neoqcd.distributedPT import NEOREXProtocolWrapper
from neoqcd.mcmc import HBOR


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--phase", choices=("train", "evaluate"), default="train")
    p.add_argument("--architecture", choices=("clover-chair", "smearing", "hyper-smearing", "residual", "hyper-residual"), default="clover-chair")
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
    p.add_argument("--hyper-smearing-mode", choices=("shared", "per_link"), default="shared")
    p.add_argument("--residual-n-pt", type=int, default=0)
    p.add_argument("--residual-mask-mod", type=int, choices=(2, 4), default=2)
    p.add_argument("--orsteps", type=int, default=1)
    p.add_argument("--mcmc-steps", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, help="Resume training or load a field for evaluation")
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
    if args.rho_functional and args.architecture != "smearing":
        p.error("rho-functional requires architecture=smearing")
    if args.hyper_smearing_mode != "shared" and args.architecture != "hyper-smearing":
        p.error("hyper-smearing-mode requires architecture=hyper-smearing")
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


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if args.checkpoint:
        field, pars, mask, payload = load_cnf_checkpoint(args.checkpoint, args.device)
        config = payload["config"]
    else:
        config = {key: value for key, value in vars(args).items() if not isinstance(value, Path)}
        config["time_normalization"] = "unit_interval"
        field, pars, mask = build_cnf(config, args.device)
        payload = None
    ladder = torch.linspace(config["parameter_min"], config["parameter_max"], config["replicas"], dtype=torch.float64, device=args.device)
    optimizer = torch.optim.Adam(field.parameters(), lr=config["lr"])
    trainer = CNFREXTrainer(field, pars, ladder, optimizer, domain=config["domain"], beta=config["beta"],
                           defect_mask=mask, batch_size=config["batch_size"], updates_per_replica=config["mcmc_steps"],
                           grad_clip_norm=config["grad_clip_norm"])
    start = 0
    if payload and "trainer" in payload:
        trainer.load_state_dict(payload["trainer"])
        torch.set_rng_state(payload["rng"])
        if args.device == "cuda" and "cuda_rng" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        start = payload["epoch"]
    else:
        trainer.warmup(config["warmup"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "evaluate":
        result = evaluate(field, trainer, args)
        (args.output_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists() and metrics_path.stat().st_size:
        if not args.checkpoint or args.checkpoint.resolve().parent != args.output_dir.resolve():
            raise FileExistsError(f"Refusing to append to another training run: {args.output_dir}")
        last = json.loads(metrics_path.read_text().splitlines()[-1])
        if last["epoch"] != start:
            raise ValueError("Metrics and checkpoint progress differ; resume into a new output directory")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with metrics_path.open("a") as output:
        for epoch in range(start, args.epochs):
            metrics = {"epoch": epoch + 1, **trainer.step()}
            line = json.dumps(metrics)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                save_cnf_checkpoint(args.output_dir / "cnf_checkpoint.pt", field, config, epoch=epoch + 1, trainer=trainer)


if __name__ == "__main__":
    main()
