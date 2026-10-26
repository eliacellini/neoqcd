import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from neoqcd.cnf_io import build_cnf, save_cnf_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("main_pt_obc_cnf", ROOT / "main" / "main_pt_obc.py")
MAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAIN)


class CNFNEOREXTest(unittest.TestCase):
    def test_cli_loads_full_lattice_cnf_and_saves_neorex_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
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


if __name__ == "__main__":
    unittest.main()
