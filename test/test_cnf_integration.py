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


if __name__ == "__main__":
    unittest.main()
