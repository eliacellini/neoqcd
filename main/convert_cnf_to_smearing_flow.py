"""Copy a PG-trained CNF field into the checkpoint of its finite flow layer.

The fields trained by main_cnf_rex.py with --architecture defect-hyper-smearing,
defect-hyper-residual, hyper-smearing or hyper-residual hold the same network as the
finite layer of the runners: they are its infinitesimal generator per unit parameter.
Copying the network lets the runners evaluate it with the finite layer and its exact
logdet.

usage:
  main/convert_cnf_to_smearing_flow.py {neorex,snf} CNF_CHECKPOINT OUT_PATH -- <runner flags>

neorex: flags of the main_pt_obc.py NEO-REX evaluation (defect fields only);
        run it with --neorex-load-flow OUT_PATH --neorex-eval-flow.
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
DEFECT_GEOMETRY = ("defect_size", "time_slice", "space_slice")
# CNF architecture -> (domain, standalone --nf-architecture, settings that must match the field)
RESIDUAL_SETTINGS = ("residual_n_pt", "residual_mask_mod", "residual_mode")
# Settings added after some fields were trained, with the value those fields used.
SETTING_DEFAULTS = {"residual_mode": "shared"}
SNF_TWINS = {
    "defect-hyper-smearing": ("defect", "hyper-smearing", ("hyper_smearing_mode",)),
    "defect-hyper-residual": ("defect", "hyper-residual", RESIDUAL_SETTINGS),
    "hyper-smearing": ("beta", "hyper-smearing", ("hyper_smearing_mode",)),
    "hyper-residual": ("beta", "hyper-residual", RESIDUAL_SETTINGS),
}


def _load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "main" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_map_settings(architecture, args):
    """The finite layer must apply rho (or residual coefficients) per unit parameter, as the field."""
    if architecture.endswith("hyper-smearing"):
        if not args.hyper_scale_by_delta or args.hyper_normalize_by_nstep:
            raise ValueError(
                "the CNF rho is per unit parameter: use --hyper-scale-by-delta --hyper-no-normalize-by-nstep"
            )
        if getattr(args, "hyper_rho_eps", 0.0) or getattr(args, "hyper_rho_max", 0.0):
            raise ValueError("the CNF field has no rho floor or clamp: use --hyper-rho-eps 0 --hyper-rho-max 0")
    elif args.residual_coeff_max or not args.residual_quadratic or not args.residual_include_imag:
        raise ValueError(
            "the CNF residual field has quadratic and imaginary features and no coefficient clamp: "
            "use --residual-quadratic --residual-include-imag --residual-coeff-max 0"
        )


def _trained_network(cnf_checkpoint, args, architectures, settings, beta=None, interval=None):
    """The trained network of a CNF field whose architecture and geometry match the flags."""
    field, _, _, payload = load_cnf_checkpoint(cnf_checkpoint)
    config = payload["config"]
    architecture = config["architecture"]
    if architecture not in architectures:
        raise ValueError(f"the evaluation flags need a {' or '.join(architectures)} field, got {architecture!r}")
    _check_map_settings(architecture, args)
    keys = ("D", "T", "L") + (DEFECT_GEOMETRY if config["domain"] == "defect" else ()) + settings
    expected = [(key, getattr(args, key)) for key in keys]
    if beta is not None:
        expected.append(("beta", beta))
    for key, value in expected:
        saved = config.get(key, SETTING_DEFAULTS.get(key))
        if saved != value:
            raise ValueError(f"{key}: CNF field has {saved!r}, evaluation flags have {value!r}")
    if interval is not None:
        low, high = sorted(interval)
        if low < config["parameter_min"] or high > config["parameter_max"]:
            raise ValueError(
                f"protocol [{low}, {high}] leaves the trained interval "
                f"[{config['parameter_min']}, {config['parameter_max']}]"
            )
    module = field.residual if architecture.endswith("hyper-residual") else field.smearing.hyper_smearing
    return module.state_dict(), payload


def _neorex_payload(cnf_checkpoint, flags):
    runner = _load_runner("main_pt_obc")
    with mock.patch.object(sys, "argv", ["main_pt_obc", *flags]):
        args = runner.parse_args()
    kind = str(args.nf_layer_type or "smearing")
    if kind == "residual":
        architecture, settings, build = "defect-hyper-residual", RESIDUAL_SETTINGS, runner.ResidualDefectSNFProtocolLayer
    elif kind == "smearing" and args.use_hyper_smearing:
        architecture, settings, build = "defect-hyper-smearing", ("hyper_smearing_mode",), runner.DefectSNFProtocolLayer
    else:
        raise ValueError("evaluation flags must select --nf-layer-type residual, or smearing with --use-hyper-smearing")
    network, source = _trained_network(
        cnf_checkpoint, args, (architecture,), settings, beta=args.beta,
        interval=(args.parameter_min, args.parameter_max),
    )

    defect = Defect(args.D, args.T, args.L, args.defect_size, args.time_slice, args.space_slice)
    pars = runner._build_neorex_flow_pars(args, torch.device("cpu"), create_mask(args.D, args.T, args.L), defect)
    layer = build(pars)
    # Strict: the evaluation network must have the trained layout.
    (layer.layer if kind == "residual" else layer.layer.smearing.hyper_smearing).load_state_dict(network)
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
    twins = [name for name, (domain, nf, _) in SNF_TWINS.items()
             if domain == args.domain and nf == args.nf_architecture]
    if args.algorithm != "snf" or not twins:
        raise ValueError(
            "evaluation flags must select --algorithm snf and --nf-architecture hyper-smearing or hyper-residual"
        )
    if args.rho_functional:
        raise ValueError("the CNF network is a hyper-smearing network, not a functional rho")
    start, end, beta, _ = runner._protocol_endpoints(args)
    network, source = _trained_network(
        cnf_checkpoint, args, twins, SNF_TWINS[twins[0]][2],
        beta=beta if args.domain == "defect" else None, interval=(start, end),
    )

    protocol = runner._protocol_values(args, start, end)
    pars, defect_mask = runner._build_flow_pars(args, torch.device("cpu"), protocol[:-1], args.batch_size)
    pars._standalone_beta = beta
    model = runner.StandaloneProtocol(args, pars, defect_mask, start, end, protocol)
    if model.nf_layer is None:
        raise ValueError(f"{args.nf_architecture} SNF must share one layer across the protocol")
    layer = model.nf_layer.layer
    if args.nf_architecture == "hyper-residual":
        # The defect patch adapter holds the residual flow itself, the lattice one wraps it.
        target = getattr(layer, "residual_flow", layer)
    else:
        target = layer.smearing.hyper_smearing
    # Strict: the evaluation network must have the trained layout.
    target.load_state_dict(network)
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
