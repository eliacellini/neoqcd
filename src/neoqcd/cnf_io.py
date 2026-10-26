"""Construction and portable checkpoints for unconditional SU(3) flow fields."""

import os
import tempfile
from pathlib import Path

import torch

from neoqcd.flow import FlowPars
from neoqcd.utils import Defect, create_mask


def build_cnf(config, device="cpu"):
    from neoqcd.cnf import CloverChairLinkDrift
    from neoqcd.cnf_adapters import make_existing_drift

    c = config
    D, T, L = (int(c[key]) for key in ("D", "T", "L"))
    defect = None
    defect_mask = None
    if c["domain"] == "defect":
        defect = Defect(D, T, L, c["defect_size"], c["time_slice"], c["space_slice"])
        defect_mask = defect.create_defect_mask().to(device)
    architecture = c["architecture"]
    pars = FlowPars(
        D=D, T=T, L=L, N=3, mask=create_mask(D, T, L).to(device),
        protocol=[c["parameter_max"]], batch_size=c["batch_size"], device=device,
        defect=defect, orsteps=c["orsteps"],
        use_hyper_smearing=architecture == "hyper-smearing",
        hyper_smearing_mode=c.get("hyper_smearing_mode", "shared"),
        rho_shape_type=1 if c.get("hyper_smearing_mode") == "per_link" else 0,
        use_rho_functional=c.get("rho_functional", False), rho_scale_fac=c["rho_init"],
        hyper_hidden_dim=c["hidden_dim"], hyper_depth=c["depth"],
        hyper_normalize_by_nstep=False, hyper_scale_by_delta=True,
        hyper_rho_init=c["rho_init"],
        residual_parameterization="static" if architecture == "residual" else "hyper",
        residual_n_pt=c["residual_n_pt"], residual_mask_mod=c["residual_mask_mod"],
        residual_coeff_init=c["rho_init"],
    )
    if architecture == "clover-chair":
        # Checkpoints without this key predate the normalization and keep the raw parameter.
        time_range = None
        if c.get("time_normalization", "none") == "unit_interval":
            time_range = (c["parameter_min"], c["parameter_max"])
        elif c.get("time_normalization", "none") != "none":
            raise ValueError(f"unknown time_normalization {c['time_normalization']!r}")
        field = CloverChairLinkDrift(D, hidden_dim=c["hidden_dim"], depth=c["depth"], time_range=time_range)
    else:
        field = make_existing_drift(architecture, pars, rho_init=c["rho_init"])
    field.cnf_config = dict(config)
    return field.to(device=device, dtype=torch.float64), pars, defect_mask


def save_cnf_checkpoint(path, field, config, *, epoch=0, trainer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "neoqcd_cnf", "version": 1, "config": dict(config),
        "field": field.state_dict(), "epoch": int(epoch),
        "training_objective": "infinitesimal_path_gradient_frozen_residual_v1",
        "rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    if trainer is not None:
        payload["trainer"] = trainer.state_dict()
    fd, temporary = tempfile.mkstemp(prefix=".cnf-", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_cnf_checkpoint(path, device="cpu", expected=None, parameter_range=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or (payload.get("format"), payload.get("version")) != ("neoqcd_cnf", 1):
        raise ValueError("Expected a version 1 neoqcd CNF checkpoint")
    config = payload["config"]
    for key, value in (expected or {}).items():
        if config.get(key) != value:
            raise ValueError(f"CNF checkpoint {key}: saved {config.get(key)!r}, requested {value!r}")
    if parameter_range is not None:
        if min(parameter_range) < config["parameter_min"] or max(parameter_range) > config["parameter_max"]:
            raise ValueError("Requested CNF evaluation lies outside the trained parameter interval")
    field, pars, mask = build_cnf(config, device)
    field.load_state_dict(payload["field"], strict=True)
    return field, pars, mask, payload
