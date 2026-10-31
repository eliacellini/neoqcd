import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neoqcd.cnf_io import build_cnf, save_cnf_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("main_pt_obc_cnf", ROOT / "main" / "main_pt_obc.py")
MAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAIN)


CONFIG = {
    "architecture": "clover-chair",
    "domain": "defect",
    "D": 4,
    "T": 2,
    "L": 2,
    "beta": 6.0,
    "defect_size": 1,
    "time_slice": 1,
    "space_slice": 1,
    "parameter_min": 0.0,
    "parameter_max": 1.0,
    "batch_size": 2,
    "hidden_dim": 4,
    "depth": 1,
    "rho_init": 1e-3,
    "residual_n_pt": 0,
    "residual_mask_mod": 2,
    "orsteps": 0,
}


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _multirank_worker(rank, world_size, port, argv):
    # Same environment torchrun provides; main() initializes and destroys the group.
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
        LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
    torch.set_num_threads(1)
    with mock.patch.object(sys, "argv", ["main_pt_obc", *argv]):
        args = MAIN.parse_args()
    MAIN.main(args)


class CNFNEOREXTest(unittest.TestCase):
    def test_cli_loads_full_lattice_cnf_and_saves_neorex_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(CONFIG)
            field, _, _ = build_cnf(config)
            cnf_checkpoint = root / "cnf.pt"
            save_cnf_checkpoint(cnf_checkpoint, field, config)
            output_dir = root / "run"
            argv = [
                "--cpu", "--algorithm", "neorex", "--nf-layer-type", "cnf",
                "--cnf-checkpoint", str(cnf_checkpoint), "--cnf-integration-steps", "1",
                "--D", "4", "--T", "2", "--L", "2", "--N", "3",
                "--defect-size", "1", "--time-slice", "1", "--space-slice", "1",
                "--beta", "6", "--parameter-min", "0", "--parameter-max", "1",
                "--batch-size", "2", "--neorex-protocol-steps", "1",
                "--thermal-steps", "0", "--steps", "0", "--orsteps", "0",
                "--no-cfg-cache", "--output-dir", str(output_dir), "--main-dir", str(root),
                "--neorex-save-flow", "auto",
            ]
            with mock.patch.object(sys, "argv", ["main_pt_obc", *argv]):
                args = MAIN.parse_args()
            self.assertFalse(args.neorex_eval_flow)

            init_file = root / "gloo_init"
            init_file.touch()
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
            dist.init_process_group(
                backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1
            )
            try:
                MAIN.main(args)
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()

            self.assertTrue(args.neorex_eval_flow)
            saved_path = root / "data" / "neorex_flows" / "global.pt"
            self.assertTrue(saved_path.exists())
            payload = torch.load(saved_path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["checkpoint_format"], "neoqcd_neorex_flow")
            self.assertEqual(payload["semantic_config"]["layer"], "cnf")
            self.assertEqual(payload["semantic_config"]["training_config"], config)

            run_meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertIn(str(saved_path), run_meta["neorex_flow_checkpoint"]["saved_paths"])

    def test_replica_history_failure_is_recorded_not_raised(self):
        # A broken analysis must leave run_meta.json complete; the error lands in replica_history.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(CONFIG)
            field, _, _ = build_cnf(config)
            cnf_checkpoint = root / "cnf.pt"
            save_cnf_checkpoint(cnf_checkpoint, field, config)
            output_dir = root / "run"
            argv = [
                "--cpu", "--algorithm", "neorex", "--nf-layer-type", "cnf",
                "--cnf-checkpoint", str(cnf_checkpoint), "--cnf-integration-steps", "1",
                "--D", "4", "--T", "2", "--L", "2", "--N", "3",
                "--defect-size", "1", "--time-slice", "1", "--space-slice", "1",
                "--beta", "6", "--parameter-min", "0", "--parameter-max", "1",
                "--batch-size", "2", "--neorex-protocol-steps", "1",
                "--thermal-steps", "0", "--steps", "2", "--orsteps", "0",
                "--no-cfg-cache", "--output-dir", str(output_dir), "--main-dir", str(root),
                "--profile-steps", "1",
            ]
            with mock.patch.object(sys, "argv", ["main_pt_obc", *argv]):
                args = MAIN.parse_args()
            init_file = root / "gloo_init"
            init_file.touch()
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
            dist.init_process_group(
                backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1
            )
            try:
                with mock.patch.object(MAIN, "summarize_history", side_effect=RuntimeError("gamma broke")):
                    MAIN.main(args)
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()
            meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertIn("swap", meta)
            self.assertIn("work", meta)
            self.assertEqual(meta["replica_history"]["error"], "RuntimeError: gamma broke")
            self.assertTrue((output_dir / "replica_history.npz").exists())
            profile = (output_dir / "step_profile.txt").read_text(encoding="utf-8")
            self.assertIn("equilibrium_sweep", profile)
            self.assertIn("swap_step", profile)
            self.assertIn("operators by self cpu time", profile)

    def test_multirank_gloo_swaps_every_link_without_training_the_field(self):
        self._run_multirank(dict(CONFIG), world_size=3)

    def test_defect_clover_chair_runs_neorex_on_its_patch(self):
        config = dict(CONFIG, architecture="defect-clover-chair", T=6, L=6, time_slice=3, space_slice=2,
                      loop_max=1)
        self._run_multirank(config, world_size=2)

    def _run_multirank(self, config, world_size):
        steps, batch_size = 2, 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(config, batch_size=batch_size)
            torch.manual_seed(3)
            field, _, _ = build_cnf(config)
            inner = getattr(field, "field", field)
            with torch.no_grad():
                # A non-trivial but small drift, so the CNF actually moves the links.
                for parameter in inner.output_layer.parameters():
                    parameter.normal_(std=1e-3)
            cnf_checkpoint = root / "cnf.pt"
            save_cnf_checkpoint(cnf_checkpoint, field, config)
            output_dir = root / "run"
            argv = [
                "--cpu", "--algorithm", "neorex", "--nf-layer-type", "cnf",
                "--cnf-checkpoint", str(cnf_checkpoint), "--cnf-integration-steps", "1",
                "--D", "4", "--T", str(config["T"]), "--L", str(config["L"]), "--N", "3",
                "--defect-size", str(config["defect_size"]), "--time-slice", str(config["time_slice"]),
                "--space-slice", str(config["space_slice"]),
                "--beta", "6", "--parameter-min", "0", "--parameter-max", "1",
                "--batch-size", str(batch_size), "--neorex-protocol-steps", "1",
                "--thermal-steps", "1", "--steps", str(steps), "--orsteps", "0",
                "--no-cfg-cache", "--output-dir", str(output_dir), "--main-dir", str(root),
                "--neorex-save-flow", "auto",
            ]
            mp.spawn(
                _multirank_worker, args=(world_size, _free_port(), argv),
                nprocs=world_size, join=True,
            )

            meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
            swap = meta["swap"]
            self.assertEqual(swap["parameter_name"], "bc")
            self.assertEqual(len(swap["link_proposed"]), world_size - 1)
            # Both parities ran, so every adjacent pair was proposed on every chain.
            self.assertTrue(all(count > 0 and count % batch_size == 0 for count in swap["link_proposed"]))
            self.assertEqual(swap["total_proposed"], sum(swap["link_proposed"]))
            self.assertLessEqual(swap["total_accepted"], swap["total_proposed"])
            self.assertTrue(all(0.0 <= rate <= 1.0 for rate in swap["link_acceptance"]))

            # Replica history: one rung per rank for every step and chain, plaquettes finite, summary present.
            history = np.load(output_dir / "replica_history.npz")
            self.assertEqual(history["labels"].shape, (steps, world_size, batch_size))
            self.assertEqual(history["plaquette"].shape, (steps, world_size, batch_size))
            for step in range(steps):
                for chain in range(batch_size):
                    self.assertEqual(sorted(history["labels"][step, :, chain].tolist()), list(range(world_size)))
            self.assertTrue(np.all(np.isfinite(history["plaquette"])))
            self.assertEqual(meta["replica_history"]["steps"], steps)
            self.assertEqual(len(meta["replica_history"]["flux_fraction"]), world_size)
            self.assertEqual(meta["replica_history"]["plaquette_top"]["status"], "insufficient_statistics")

            work = meta["work"]
            self.assertEqual(work["count"], swap["total_proposed"])
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in work["link_mean"]))
            self.assertTrue(any(abs(v) > 0 for v in work["link_mean"]))

            # Evaluation-only: the field saved by NEO-REX equals the loaded checkpoint.
            saved = torch.load(root / "data" / "neorex_flows" / "global.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["semantic_config"]["layer"], "cnf")
            reference = field.state_dict()
            flow_state = saved["state_dict"]
            field_state = {k.removeprefix("nf_layer.field."): v for k, v in flow_state.items() if k.startswith("nf_layer.field.")}
            self.assertEqual(field_state.keys(), reference.keys())
            for key, value in reference.items():
                torch.testing.assert_close(field_state[key], value, rtol=0, atol=0)

    def test_defect_field_converts_to_the_neorex_smearing_layer(self):
        spec = importlib.util.spec_from_file_location(
            "convert_cnf_to_smearing_flow", ROOT / "main" / "convert_cnf_to_smearing_flow.py"
        )
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(
                CONFIG, architecture="defect-hyper-smearing", T=4, L=6, time_slice=2, space_slice=2,
                hidden_dim=8, depth=1, hyper_smearing_mode="per_link",
            )
            torch.manual_seed(11)
            field, _, _ = build_cnf(config)
            with torch.no_grad():
                for parameter in field.smearing.hyper_smearing.mlp.parameters():
                    parameter.add_(0.05 * torch.randn_like(parameter))
            cnf_checkpoint = root / "cnf.pt"
            save_cnf_checkpoint(cnf_checkpoint, field, config, epoch=7)
            flags = [
                "--cpu", "--algorithm", "neorex", "--D", "4", "--T", "4", "--L", "6", "--N", "3",
                "--defect-size", "1", "--time-slice", "2", "--space-slice", "2",
                "--beta", "6", "--parameter-min", "0", "--parameter-max", "1",
                "--nf-layer-type", "smearing", "--use-hyper-smearing", "--hyper-smearing-mode", "per_link",
                "--hyper-hidden-dim", "8", "--hyper-depth", "1",
                "--hyper-scale-by-delta", "--hyper-no-normalize-by-nstep",
                "--batch-size", "2", "--neorex-protocol-steps", "1",
            ]
            converted = converter.convert("neorex", cnf_checkpoint, root / "flow.pt", flags)
            self.assertEqual(torch.load(converted, weights_only=True)["source_cnf_epoch"], 7)

            mismatched = [*flags[:-1], "2"]
            mismatched[mismatched.index("--hyper-hidden-dim") + 1] = "16"
            with self.assertRaises(RuntimeError):
                converter.convert("neorex", cnf_checkpoint, root / "bad.pt", mismatched)

            argv = [
                *flags, "--neorex-load-flow", str(converted), "--neorex-eval-flow",
                "--thermal-steps", "0", "--steps", "1", "--orsteps", "0", "--no-cfg-cache",
                "--output-dir", str(root / "run"), "--main-dir", str(root), "--neorex-save-flow", "auto",
            ]
            with mock.patch.object(sys, "argv", ["main_pt_obc", *argv]):
                args = MAIN.parse_args()
            init_file = root / "gloo_init"
            init_file.touch()
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if os.uname().sysname == "Darwin" else "lo")
            dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
            try:
                MAIN.main(args)
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()

            # The evaluated (and re-saved) smearing layer carries the trained hypernetwork.
            saved = torch.load(root / "data" / "neorex_flows" / "global.pt", map_location="cpu", weights_only=True)
            prefix = "nf_layer.layer.smearing.hyper_smearing."
            used = {k.removeprefix(prefix): v for k, v in saved["state_dict"].items() if k.startswith(prefix)}
            reference = field.smearing.hyper_smearing.state_dict()
            self.assertEqual(used.keys(), reference.keys())
            for key, value in reference.items():
                torch.testing.assert_close(used[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
