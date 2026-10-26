"""Copy a defect-hyper-smearing CNF field into a smearing-flow checkpoint.

The field trained by main_cnf_rex.py (--architecture defect-hyper-smearing) holds the
same HyperSmearing network as the defect coupling layer of both runners. Copying it
lets them evaluate the trained rho(bc) with the finite smearing layer and its exact
logdet.

usage:
  main/convert_cnf_to_smearing_flow.py {neorex,snf} CNF_CHECKPOINT OUT_PATH -- <runner flags>

neorex: flags of the main_pt_obc.py NEO-REX evaluation; run it with
        --neorex-load-flow OUT_PATH --neorex-eval-flow.
snf:    flags of the main_nemcmc_snf.py SNF evaluation; run it with
        --phase evaluate --checkpoint OUT_PATH.
"""

import importlib.util
from pathlib import Path
import sys
from unittest import mock

import torch

from neoqcd.cnf_io import load_cnf_checkpoint
from neoqcd.distributedPT import NEOREXProtocolWrapper
from neoqcd.utils import Defect, create_mask


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ("D", "T", "L", "defect_size", "time_slice", "space_slice", "hyper_smearing_mode")


def _load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "main" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trained_network(cnf_checkpoint, args, beta):
    if not args.hyper_scale_by_delta or args.hyper_normalize_by_nstep:
        raise ValueError(
            "the CNF rho is per unit parameter: use --hyper-scale-by-delta --hyper-no-normalize-by-nstep"
        )
    field, _, _, payload = load_cnf_checkpoint(cnf_checkpoint)
    config = payload["config"]
    if config["architecture"] != "defect-hyper-smearing":
        raise ValueError(f"only defect-hyper-smearing fields have a smearing twin, got {config['architecture']!r}")
    for key, value in [(key, getattr(args, key)) for key in GEOMETRY] + [("beta", beta)]:
        if config[key] != value:
            raise ValueError(f"{key}: CNF field has {config[key]!r}, evaluation flags have {value!r}")
    return field.smearing.hyper_smearing.state_dict(), payload


def _neorex_payload(cnf_checkpoint, flags):
    runner = _load_runner("main_pt_obc")
    with mock.patch.object(sys, "argv", ["main_pt_obc", *flags]):
        args = runner.parse_args()
    if str(args.nf_layer_type or "smearing") != "smearing" or not args.use_hyper_smearing:
        raise ValueError("evaluation flags must select --nf-layer-type smearing --use-hyper-smearing")
    network, source = _trained_network(cnf_checkpoint, args, args.beta)

    defect = Defect(args.D, args.T, args.L, args.defect_size, args.time_slice, args.space_slice)
    pars = runner._build_neorex_flow_pars(args, torch.device("cpu"), create_mask(args.D, args.T, args.L), defect)
    layer = runner.DefectSNFProtocolLayer(pars)
    # Strict: the evaluation hypernetwork must have the trained layout.
    layer.layer.smearing.hyper_smearing.load_state_dict(network)
    wrapper = NEOREXProtocolWrapper(layer, None, args.neorex_protocol_steps)
    return {
        "checkpoint_format": "neoqcd_neorex_flow",
        "format_version": 2,
        "smearing_jacobian_version": layer.smearing_jacobian_version,
        "state_dict": wrapper.state_dict(),
        "semantic_config": layer.checkpoint_semantics,
    }, source


def _snf_payload(cnf_checkpoint, flags):
    runner = _load_runner("main_nemcmc_snf")
    args = runner.parse_args(flags)
    if args.algorithm != "snf" or args.domain != "defect" or args.nf_architecture != "hyper-smearing":
        raise ValueError("evaluation flags must select --algorithm snf --domain defect --nf-architecture hyper-smearing")
    if args.rho_functional:
        raise ValueError("the CNF network is a hyper-smearing network, not a functional rho")
    start, end, beta, _ = runner._protocol_endpoints(args)
    network, source = _trained_network(cnf_checkpoint, args, beta)

    protocol = runner._protocol_values(args, start, end)
    pars, defect_mask = runner._build_flow_pars(args, torch.device("cpu"), protocol[:-1], args.batch_size)
    pars._standalone_beta = beta
    model = runner.StandaloneProtocol(args, pars, defect_mask, start, end, protocol)
    if model.nf_layer is None:
        raise ValueError("hyper-smearing SNF must share one layer across the protocol")
    model.nf_layer.layer.smearing.hyper_smearing.load_state_dict(network)
    return {
        "checkpoint_semantics": runner._checkpoint_semantics(args, protocol),
        "model_state_dict": model.state_dict(),
    }, source


def convert(target, cnf_checkpoint, out_path, flags):
    builders = {"neorex": _neorex_payload, "snf": _snf_payload}
    if target not in builders:
        raise ValueError(f"target must be one of {sorted(builders)}, got {target!r}")
    payload, source = builders[target](cnf_checkpoint, list(flags))
    payload.update(
        source_cnf_checkpoint=str(Path(cnf_checkpoint).resolve()),
        source_cnf_epoch=int(source["epoch"]),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return out_path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) < 4 or argv[3] != "--":
        raise SystemExit(__doc__)
    out_path = convert(argv[0], argv[1], argv[2], argv[4:])
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
