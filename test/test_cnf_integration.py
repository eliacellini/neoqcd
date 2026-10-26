import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from neoqcd.cnf_io import build_cnf, load_cnf_checkpoint, save_cnf_checkpoint
from neoqcd.distributedPT import NEOREXProtocolWrapper


ROOT = Path(__file__).resolve().parents[1]


def load_main(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "main" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CNFIntegrationTest(unittest.TestCase):
    def test_full_lattice_protocol_never_uses_defect_halo(self):
        class Layer(torch.nn.Module):
            full_lattice = True

            def forward(self, x, parameter, delta_parameter):
                return x + delta_parameter[:, None], torch.zeros_like(parameter)

        class Theory:
            def local_nf_action_change(self, *args, **kwargs):
                raise AssertionError("A full-lattice field cannot use the defect halo")

            def action(self, x, parameter, **kwargs):
                return parameter * x.square().sum(1)

        x = torch.ones(2, 3)
        a, b = torch.ones(2), torch.full((2,), 2.)
        protocol = NEOREXProtocolWrapper(Layer(), Theory(), 2)
        y, w = protocol(x, a, b)
        torch.testing.assert_close(y, 2 * x)
        torch.testing.assert_close(w, b * y.square().sum(1) - a * x.square().sum(1))

    def test_checkpoint_and_standalone_snf_factory(self):
        runner = load_main("main_cnf_rex")
        snf = load_main("main_nemcmc_snf")
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parse_args(["--output-dir", directory, "--D", "2", "--T", "2", "--L", "2",
                                      "--hidden-dim", "4", "--depth", "1", "--batch-size", "2"])
            config = {key: value for key, value in vars(args).items() if not isinstance(value, Path)}
            field, pars, _ = build_cnf(config)
            checkpoint = Path(directory) / "field.pt"
            save_cnf_checkpoint(checkpoint, field, config)
            restored, _, _, payload = load_cnf_checkpoint(checkpoint, expected={"domain": "beta", "D": 2})
            self.assertEqual(payload["epoch"], 0)
            for a, b in zip(field.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b)
            with self.assertRaisesRegex(ValueError, "domain"):
                load_cnf_checkpoint(checkpoint, expected={"domain": "defect"})
            self.assertIsNone(restored.time_range)

            # Runner-created configs normalize the encoder input over the trained interval.
            runner.main(["--output-dir", directory, "--D", "2", "--T", "2", "--L", "2",
                         "--parameter-min", ".5", "--parameter-max", "1", "--replicas", "2",
                         "--hidden-dim", "4", "--depth", "1", "--batch-size", "2",
                         "--warmup", "0", "--epochs", "1"])
            trained, _, _, payload = load_cnf_checkpoint(Path(directory) / "cnf_checkpoint.pt")
            self.assertEqual(payload["config"]["time_normalization"], "unit_interval")
            self.assertEqual(trained.time_range, (0.5, 1.0))
            with self.assertRaisesRegex(ValueError, "interval"):
                load_cnf_checkpoint(checkpoint, parameter_range=(.5, 6.))
            opts = snf.parse_args(["--algorithm", "snf", "--phase", "evaluate", "--domain", "beta",
                                  "--nf-architecture", "cnf", "--cnf-checkpoint", str(checkpoint),
                                  "--D", "2", "--T", "2", "--L", "2", "--batch-size", "2", "--cpu",
                                  "--beta-initial", ".5", "--beta-final", "1.0"])
            layer = snf._make_snf_layer(opts, pars)
            self.assertTrue(layer.full_lattice)
            self.assertEqual(layer.checkpoint_semantics["training_config"], config)
            self.assertEqual(snf._nf_layer_sharing(opts), "shared")
            self.assertEqual(snf._checkpoint_semantics(opts, [.5, 1.])["cnf_config"], config)
            opts.output_dir = str(Path(directory) / "snf_eval")
            opts.cfg_cache = "off"
            opts.thermal_steps = 1
            opts.prior_mcmc_steps = 1
            opts.eval_steps = 1
            opts.protocol_steps = 1
            opts.int_meas = 1
            opts.cnf_integration_steps = 1
            opts.orsteps = 0
            snf.main(opts)
            summary = json.loads((Path(opts.output_dir) / "summary.json").read_text())
            self.assertIsNotNone(summary["work_mean"])

    def test_defect_field_converts_to_the_standalone_snf_smearing_layer(self):
        converter = load_main("convert_cnf_to_smearing_flow")
        snf = load_main("main_nemcmc_snf")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(
                architecture="defect-hyper-smearing", domain="defect", D=4, T=4, L=6, beta=6.0,
                defect_size=1, time_slice=2, space_slice=2, parameter_min=0.0, parameter_max=1.0,
                batch_size=2, hidden_dim=8, depth=1, rho_init=1e-3, residual_n_pt=0,
                residual_mask_mod=2, orsteps=0, hyper_smearing_mode="per_link",
            )
            torch.manual_seed(13)
            field, _, _ = build_cnf(config)
            with torch.no_grad():
                for parameter in field.smearing.hyper_smearing.mlp.parameters():
                    parameter.add_(0.05 * torch.randn_like(parameter))
            save_cnf_checkpoint(root / "cnf.pt", field, config, epoch=3)
            flags = [
                "--algorithm", "snf", "--domain", "defect", "--cpu", "--D", "4", "--T", "4", "--L", "6",
                "--beta", "6", "--bc-initial", "0", "--bc-final", "1",
                "--defect-size", "1", "--time-slice", "2", "--space-slice", "2",
                "--nf-architecture", "hyper-smearing", "--hyper-smearing-mode", "per_link",
                "--hyper-hidden-dim", "8", "--hyper-depth", "1", "--hyper-no-normalize-by-nstep",
                "--batch-size", "2", "--protocol-steps", "2", "--prior-mcmc-steps", "0",
                "--updates-per-layer", "1", "--orsteps", "0", "--cfg-cache", "off",
            ]
            converted = converter.convert("snf", root / "cnf.pt", root / "snf.pt", flags)
            with self.assertRaisesRegex(ValueError, "per unit parameter"):
                converter.convert("snf", root / "cnf.pt", root / "bad.pt",
                                  [f for f in flags if f != "--hyper-no-normalize-by-nstep"])

            args = snf.parse_args([*flags, "--phase", "evaluate", "--thermal-steps", "0", "--eval-steps", "1",
                                   "--checkpoint", str(converted), "--output-dir", str(root / "run")])
            snf.main(args)
            evaluated = snf._safe_torch_load(root / "run" / "snf_checkpoint.pt")["model_state_dict"]
            prefix = "nf_layer.layer.smearing.hyper_smearing."
            used = {k.removeprefix(prefix): v for k, v in evaluated.items() if k.startswith(prefix)}
            reference = field.smearing.hyper_smearing.state_dict()
            self.assertEqual(used.keys(), reference.keys())
            for key, value in reference.items():
                torch.testing.assert_close(used[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
