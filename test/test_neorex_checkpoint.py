import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock
import unittest

import torch
import torch.distributed as dist
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("main_pt_obc", ROOT / "main" / "main_pt_obc.py")
MAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAIN)


class _VersionedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.25]))
        self.smearing_jacobian_version = MAIN.LOGDET_VERSION


class _VersionedFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.nf_layer = _VersionedLayer()


def _fake_pt():
    flow = _VersionedFlow()
    pt = object.__new__(MAIN.NEOREX)
    pt.rank = 0
    pt.device = torch.device("cpu")
    pt.flow = flow
    pt.ddp_flow = SimpleNamespace(module=flow, parameters=flow.parameters)
    return pt, flow


def _metadata_args():
    return SimpleNamespace(
        nf_layer_type="smearing",
        explicit_backward=False,
        neorex_protocol_steps=1,
        neorex_heatbath_steps_per_step=0,
        neorex_work_mode="logdet",
        train_steps=0,
        neorex_eval_flow=False,
        hyper_smearing_mode="per_link",
        hyper_time_embedding_dim=8,
        hyper_hidden_dim=16,
        hyper_depth=2,
        hyper_activation="silu",
        hyper_rho_init=1e-3,
        hyper_scale_by_delta=False,
        hyper_normalize_by_nstep=False,
        D=4,
        T=6,
        L=6,
        N=3,
        defect_size=2,
        time_slice=2,
        space_slice=2,
        batch_size=1,
        bc_min=5.8,
        bc_max=6.0,
    )


def _real_neorex_pt(
    *,
    nsteps=1,
    layer_type="smearing",
    activation="silu",
    scale_by_delta=True,
    normalize_by_nstep=False,
    rho_max=0.0,
    coeff_max=0.0,
    residual_parameterization="hyper",
    use_hyper_smearing=True,
):
    args_list = [
        "--cpu", "--algorithm", "neorex",
        "--D", "4", "--T", "6", "--L", "6", "--N", "3",
        "--defect-size", "2", "--time-slice", "2", "--space-slice", "2",
        "--batch-size", "1", "--neorex-protocol-steps", str(nsteps),
        "--nf-layer-type", layer_type, "--hyper-activation", activation,
        "--hyper-rho-max", str(rho_max), "--residual-coeff-max", str(coeff_max),
    ]
    if not use_hyper_smearing:
        args_list.append("--no-hyper-smearing")
    args_list.append("--hyper-scale-by-delta" if scale_by_delta else "--hyper-no-scale-by-delta")
    args_list.append("--hyper-normalize-by-nstep" if normalize_by_nstep else "--hyper-no-normalize-by-nstep")
    with mock.patch.object(sys, "argv", ["main_pt_obc", *args_list]):
        args = MAIN.parse_args()

    device = torch.device("cpu")
    mask = MAIN.create_mask(args.D, args.T, args.L).to(device)
    defect = MAIN.Defect(
        D=args.D,
        T=args.T,
        L=args.L,
        dsize=args.defect_size,
        time_slice=args.time_slice,
        space_slice=args.space_slice,
    )
    flow_pars = MAIN._build_neorex_flow_pars(args, device, mask, defect)
    flow_pars.residual_parameterization = residual_parameterization
    nf_layer = (
        MAIN.ResidualDefectSNFProtocolLayer(flow_pars)
        if layer_type == "residual"
        else MAIN.DefectSNFProtocolLayer(flow_pars)
    )
    flow = MAIN.NEOREXProtocolWrapper(
        nf_layer=nf_layer,
        theory=None,
        n_protocol_steps=nsteps,
        parameter_name="bc",
    )
    pt = object.__new__(MAIN.NEOREX)
    pt.rank = 0
    pt.device = device
    pt.flow = flow
    pt.ddp_flow = SimpleNamespace(module=flow, parameters=flow.parameters)
    return pt


class NEOREXCheckpointTest(unittest.TestCase):
    def test_api_checkpoint_is_self_contained_and_legacy_smearing_is_rejected(self):
        pt, flow = _fake_pt()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "api_flow.pt"
            with mock.patch.object(MAIN.dist, "broadcast_object_list"), mock.patch.object(MAIN.dist, "broadcast"):
                pt.save_flow(str(checkpoint))
                self.assertTrue(checkpoint.exists())
                self.assertFalse(Path(f"{checkpoint}.meta.json").exists())

                with torch.no_grad():
                    flow.nf_layer.weight.zero_()
                pt.load_flow(str(checkpoint))
                self.assertTrue(torch.equal(flow.nf_layer.weight, torch.tensor([1.25])))

                legacy = Path(directory) / "legacy_state_dict.pt"
                torch.save(flow.state_dict(), legacy)
                with self.assertRaisesRegex(RuntimeError, "unversioned state_dict"):
                    pt.load_flow(str(legacy))

    def test_cli_loads_api_checkpoint_without_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            init_file = Path(directory) / "gloo_init"
            init_file.touch()
            output_dir = Path(directory) / "output"
            checkpoint = Path(directory) / "api_flow.pt"
            args_list = [
                "--cpu",
                "--algorithm", "neorex",
                "--D", "4", "--T", "6", "--L", "6", "--N", "3",
                "--defect-size", "2", "--time-slice", "2", "--space-slice", "2",
                "--batch-size", "1", "--neorex-protocol-steps", "1",
                "--thermal-steps", "0", "--steps", "0", "--orsteps", "0",
                "--no-cfg-cache", "--output-dir", str(output_dir),
                "--neorex-load-flow", str(checkpoint),
            ]
            with mock.patch.object(sys, "argv", ["main_pt_obc", *args_list]):
                args = MAIN.parse_args()

            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{init_file}",
                rank=0,
                world_size=1,
            )
            try:
                device = torch.device("cpu")
                mask = MAIN.create_mask(args.D, args.T, args.L).to(device)
                defect = MAIN.Defect(
                    D=args.D,
                    T=args.T,
                    L=args.L,
                    dsize=args.defect_size,
                    time_slice=args.time_slice,
                    space_slice=args.space_slice,
                )
                flow_pars = MAIN._build_neorex_flow_pars(args, device, mask, defect)
                nf_layer = MAIN.DefectSNFProtocolLayer(flow_pars)
                flow = MAIN.NEOREXProtocolWrapper(
                    nf_layer=nf_layer,
                    theory=None,
                    n_protocol_steps=args.neorex_protocol_steps,
                    parameter_name="bc",
                )
                pt = object.__new__(MAIN.NEOREX)
                pt.rank = 0
                pt.flow = flow
                pt.ddp_flow = SimpleNamespace(module=flow)
                pt.save_flow(str(checkpoint))
                self.assertFalse(Path(f"{checkpoint}.meta.json").exists())

                state = MAIN.RunState(
                    rank=0,
                    world_size=1,
                    local_rank=0,
                    device=device,
                    backend="gloo",
                )
                with mock.patch.object(MAIN, "setup_runtime", return_value=state):
                    MAIN.main(args)

                run_meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
                self.assertEqual(run_meta["neorex_flow_checkpoint"]["load_mode"], "load")
                self.assertEqual(run_meta["neorex_flow_checkpoint"]["load_path"], str(checkpoint))
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()

    def test_semantic_mismatches_fail_before_loading_weights(self):
        mismatches = [
            ({"activation": "gelu"}, "mathematical semantics do not match"),
            ({"scale_by_delta": False}, "mathematical semantics do not match"),
            ({"rho_max": 0.5}, "mathematical semantics do not match"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "semantic_flow.pt"
            source = _real_neorex_pt()
            with mock.patch.object(MAIN.dist, "broadcast_object_list"):
                source.save_flow(str(checkpoint))

            for options, message in mismatches:
                with self.subTest(options=options):
                    target = _real_neorex_pt(**options)
                    before = {key: value.clone() for key, value in target.flow.state_dict().items()}
                    with mock.patch.object(MAIN.dist, "broadcast_object_list"):
                        with self.assertRaisesRegex(RuntimeError, message):
                            target.load_flow(str(checkpoint))
                    after = target.flow.state_dict()
                    self.assertEqual(before.keys(), after.keys())
                    self.assertTrue(all(torch.equal(before[key], after[key]) for key in before))

    def test_nstep_transfer_depends_on_effective_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "semantic_flow.pt"
            source = _real_neorex_pt(nsteps=1, normalize_by_nstep=False)
            with mock.patch.object(MAIN.dist, "broadcast_object_list"):
                source.save_flow(str(checkpoint))

            target = _real_neorex_pt(nsteps=2, normalize_by_nstep=False)
            with mock.patch.object(MAIN.dist, "broadcast_object_list"), mock.patch.object(MAIN.dist, "broadcast"):
                target.load_flow(str(checkpoint))
            for key, value in source.flow.state_dict().items():
                self.assertTrue(torch.equal(value, target.flow.state_dict()[key]))

            normalized_source = _real_neorex_pt(nsteps=1, normalize_by_nstep=True)
            normalized_target = _real_neorex_pt(nsteps=2, normalize_by_nstep=True)
            with mock.patch.object(MAIN.dist, "broadcast_object_list"):
                normalized_source.save_flow(str(checkpoint))
                with self.assertRaisesRegex(RuntimeError, "mathematical semantics do not match"):
                    normalized_target.load_flow(str(checkpoint))

    def test_residual_coefficient_bound_is_checked_for_hyper_and_static_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "residual_flow.pt"
            for parameterization in ("hyper", "static"):
                with self.subTest(parameterization=parameterization):
                    source = _real_neorex_pt(
                        layer_type="residual", coeff_max=0.2,
                        residual_parameterization=parameterization,
                    )
                    target = _real_neorex_pt(
                        layer_type="residual", coeff_max=0.3,
                        residual_parameterization=parameterization,
                    )
                    with mock.patch.object(MAIN.dist, "broadcast_object_list"):
                        source.save_flow(str(checkpoint))
                        with self.assertRaisesRegex(RuntimeError, "mathematical semantics do not match"):
                            target.load_flow(str(checkpoint))

    def test_legacy_hyper_checkpoint_without_semantics_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy_hyper_flow.pt"
            for layer_type in ("smearing", "residual"):
                with self.subTest(layer_type=layer_type):
                    pt = _real_neorex_pt(layer_type=layer_type)
                    payload = {
                        "checkpoint_format": "neoqcd_neorex_flow",
                        "format_version": 1,
                        "smearing_jacobian_version": (
                            MAIN.LOGDET_VERSION if layer_type == "smearing" else None
                        ),
                        "state_dict": pt.flow.state_dict(),
                    }
                    torch.save(payload, checkpoint)
                    with mock.patch.object(MAIN.dist, "broadcast_object_list"):
                        with self.assertRaisesRegex(RuntimeError, "lacks verifiable map semantics"):
                            pt.load_flow(str(checkpoint))

    def test_nonhyper_versioned_classic_checkpoint_remains_compatible(self):
        pt = _real_neorex_pt(use_hyper_smearing=False)
        payload = {
            "checkpoint_format": "neoqcd_neorex_flow",
            "format_version": 1,
            "smearing_jacobian_version": MAIN.LOGDET_VERSION,
            "state_dict": pt.flow.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy_static_flow.pt"
            torch.save(payload, checkpoint)
            with mock.patch.object(MAIN.dist, "broadcast_object_list"), mock.patch.object(MAIN.dist, "broadcast"):
                pt.load_flow(str(checkpoint))

    def test_metadata_write_error_is_shared_with_all_ranks(self):
        args = _metadata_args()
        broadcast_error = []

        def rank0_broadcast(error, src):
            self.assertEqual(src, 0)
            broadcast_error.append(error[0])

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "metadata_failure.pt")
            with mock.patch.object(MAIN.dist, "broadcast_object_list", side_effect=rank0_broadcast):
                with mock.patch.object(MAIN.os, "replace", side_effect=OSError("simulated metadata disk failure")):
                    with self.assertRaisesRegex(RuntimeError, "simulated metadata disk failure") as rank0_exc:
                        MAIN._write_neorex_flow_metadata(args, path, rank=0)

            self.assertEqual(len(broadcast_error), 1)
            self.assertIn("simulated metadata disk failure", broadcast_error[0])

            def rank1_broadcast(error, src):
                self.assertEqual(src, 0)
                error[0] = broadcast_error[0]

            with mock.patch.object(MAIN.dist, "broadcast_object_list", side_effect=rank1_broadcast):
                with self.assertRaisesRegex(RuntimeError, "simulated metadata disk failure") as rank1_exc:
                    MAIN._write_neorex_flow_metadata(args, path, rank=1)
            self.assertEqual(str(rank0_exc.exception), str(rank1_exc.exception))


if __name__ == "__main__":
    unittest.main()
