"""Time and profile the finite defect smearing layer of NEO-REX on one GPU.

Loads a PG-trained defect-hyper-smearing field, copies its network into the finite layer
and times, per batch size: the layer forward, its inverse and one HBOR sweep of the whole
lattice (the MCMC update the layer is paired with in a protocol step). Thermalization is not
needed for timing, so the links are Haar random.
"""

import argparse
import time

import torch

import neoqcd.sun_utils as sun
from neoqcd.cnf_io import load_cnf_checkpoint
from neoqcd.mcmc import HBOR
from neoqcd.smearing import DefectCouplingLayer


def clock():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def timed(fn, repeats):
    fn()
    start = clock()
    for _ in range(repeats):
        fn()
    return (clock() - start) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--delta", type=float, default=1 / 28)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", default="", help="Write a profiler table of forward + inverse here")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    field, pars, defect_mask, payload = load_cnf_checkpoint(args.checkpoint, device)
    layer = DefectCouplingLayer(pars, torch.zeros(pars.rho_shape, dtype=torch.double, device=device), time=1.0)
    layer.smearing.hyper_smearing.load_state_dict(field.smearing.hyper_smearing.state_dict())
    D, T, L = int(pars.D), int(pars.T), int(pars.L)
    for batch in args.batch:
        U = sun.SUN_haar((batch, D, T, *([L] * (D - 1)), 3), dtype=torch.cdouble, device=device)
        bc = torch.full((batch,), 0.5, dtype=torch.double, device=device)
        step = torch.full_like(bc, args.delta)
        update = HBOR(pars, beta=6.0, defect_par=0.5)
        with torch.no_grad():
            out = layer(U, pars, None, beta=bc, delta_beta=step)[0]
            forward = timed(lambda: layer(U, pars, None, beta=bc, delta_beta=step), args.repeats)
            inverse = timed(lambda: layer.inverse(out, pars, None, beta=bc, delta_beta=step), args.repeats)
            sweep = timed(lambda: update(U, pars.mask, defect_mask), args.repeats)
        print(f"batch {batch}: forward {forward * 1e3:.1f} ms  inverse {inverse * 1e3:.1f} ms  "
              f"HBOR sweep {sweep * 1e3:.1f} ms", flush=True)
        if args.profile and batch == args.batch[-1]:
            with torch.no_grad(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else [])
            ) as profiler:
                layer(U, pars, None, beta=bc, delta_beta=step)
                layer.inverse(out, pars, None, beta=bc, delta_beta=step)
                clock()
            averages = profiler.key_averages()
            with open(args.profile, "w") as handle:
                handle.write(f"# batch {batch}: operators by self {device.type} time\n")
                handle.write(averages.table(sort_by=f"self_{device.type}_time_total", row_limit=30))
                handle.write("\n# operators by self cpu time\n")
                handle.write(averages.table(sort_by="self_cpu_time_total", row_limit=30))
                handle.write("\n# operators by call count\n")
                for event in sorted(averages, key=lambda e: -e.count)[:30]:
                    handle.write(f"{event.key:60s} {event.count:8d}\n")


if __name__ == "__main__":
    main()
