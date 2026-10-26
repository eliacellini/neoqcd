"""Copy a defect-hyper-smearing CNF field into a NEO-REX smearing-flow checkpoint.

The field trained by main_cnf_rex.py (--architecture defect-hyper-smearing) holds the
same HyperSmearing network as DefectSNFProtocolLayer. Copying it lets main_pt_obc.py
evaluate the trained rho(bc) with the finite smearing layer and its exact logdet.

usage:
  main/convert_cnf_to_neorex_flow.py CNF_CHECKPOINT OUT_PATH -- <main_pt_obc.py NEO-REX flags>

The flags after "--" must be those of the evaluation run; it then needs
--neorex-load-flow OUT_PATH --neorex-eval-flow.
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
GEOMETRY = ("D", "T", "L", "beta", "defect_size", "time_slice", "space_slice", "hyper_smearing_mode")


def _load_pt_runner():
    spec = importlib.util.spec_from_file_location("main_pt_obc", ROOT / "main" / "main_pt_obc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert(cnf_checkpoint, out_path, pt_flags):
    runner = _load_pt_runner()
    with mock.patch.object(sys, "argv", ["main_pt_obc", *pt_flags]):
        args = runner.parse_args()
    if str(args.nf_layer_type or "smearing") != "smearing" or not args.use_hyper_smearing:
        raise ValueError("evaluation flags must select --nf-layer-type smearing --use-hyper-smearing")
    if not args.hyper_scale_by_delta or args.hyper_normalize_by_nstep:
        raise ValueError(
            "the CNF rho is per unit parameter: use --hyper-scale-by-delta --hyper-no-normalize-by-nstep"
        )

    field, _, _, payload = load_cnf_checkpoint(cnf_checkpoint)
    config = payload["config"]
    if config["architecture"] != "defect-hyper-smearing":
        raise ValueError(f"only defect-hyper-smearing fields have a smearing twin, got {config['architecture']!r}")
    for key in GEOMETRY:
        if config[key] != getattr(args, key):
            raise ValueError(f"{key}: CNF field has {config[key]!r}, evaluation flags have {getattr(args, key)!r}")

    device = torch.device("cpu")
    defect = Defect(args.D, args.T, args.L, args.defect_size, args.time_slice, args.space_slice)
    pars = runner._build_neorex_flow_pars(args, device, create_mask(args.D, args.T, args.L), defect)
    layer = runner.DefectSNFProtocolLayer(pars)
    # Strict: the evaluation hypernetwork must have the trained layout.
    layer.layer.smearing.hyper_smearing.load_state_dict(field.smearing.hyper_smearing.state_dict())
    wrapper = NEOREXProtocolWrapper(layer, None, args.neorex_protocol_steps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_format": "neoqcd_neorex_flow",
        "format_version": 2,
        "smearing_jacobian_version": layer.smearing_jacobian_version,
        "state_dict": wrapper.state_dict(),
        "semantic_config": layer.checkpoint_semantics,
        "source_cnf_checkpoint": str(Path(cnf_checkpoint).resolve()),
        "source_cnf_epoch": int(payload["epoch"]),
    }, out_path)
    return out_path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) < 3 or argv[2] != "--":
        raise SystemExit(__doc__)
    out_path = convert(argv[0], argv[1], argv[3:])
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
