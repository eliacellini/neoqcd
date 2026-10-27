import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

from neoqcd.cnf_io import build_cnf, load_cnf_checkpoint, save_cnf_checkpoint
from neoqcd.distributedPT import NEOREXProtocolWrapper
import neoqcd.sun_utils as sun
from neoqcd.utils import Defect, create_mask


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

    def test_full_lattice_fields_convert_to_the_standalone_snf_layers(self):
        converter = load_main("convert_cnf_to_smearing_flow")
        snf = load_main("main_nemcmc_snf")
        cases = {
            "hyper-smearing": (["--hyper-smearing-mode", "per_link", "--hyper-no-normalize-by-nstep"],
                               "nf_layer.layer.smearing.hyper_smearing.", lambda f: f.smearing.hyper_smearing),
            "hyper-residual": (["--residual-n-pt", "1", "--residual-mask-mod", "4"],
                               "nf_layer.layer.residual_flow.", lambda f: f.residual),
        }
        for architecture, (extra, prefix, network) in cases.items():
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = dict(
                    architecture=architecture, domain="beta", D=4, T=4, L=4, beta=6.0,
                    parameter_min=1.0, parameter_max=2.0, batch_size=2, hidden_dim=8, depth=1,
                    rho_init=1e-3, residual_n_pt=1 if architecture == "hyper-residual" else 0,
                    residual_mask_mod=4 if architecture == "hyper-residual" else 2, orsteps=0,
                    hyper_smearing_mode="per_link" if architecture == "hyper-smearing" else "shared",
                )
                torch.manual_seed(14)
                field, _, _ = build_cnf(config)
                with torch.no_grad():
                    for parameter in network(field).parameters():
                        parameter.add_(0.05 * torch.randn_like(parameter))
                save_cnf_checkpoint(root / "cnf.pt", field, config, epoch=2)
                flags = [
                    "--algorithm", "snf", "--domain", "beta", "--cpu", "--D", "4", "--T", "4", "--L", "4",
                    "--beta-initial", "1.2", "--beta-final", "1.8", "--nf-architecture", architecture,
                    "--hyper-hidden-dim", "8", "--hyper-depth", "1", *extra,
                    "--batch-size", "2", "--protocol-steps", "2", "--prior-mcmc-steps", "0",
                    "--updates-per-layer", "1", "--orsteps", "0", "--cfg-cache", "off",
                ]
                converted = converter.convert("snf", root / "cnf.pt", root / "snf.pt", flags)
                with self.assertRaisesRegex(ValueError, "trained interval"):
                    converter.convert("snf", root / "cnf.pt", root / "bad.pt",
                                      [*flags[:flags.index("--beta-final") + 1], "2.5",
                                       *flags[flags.index("--beta-final") + 2:]])

                args = snf.parse_args([*flags, "--phase", "evaluate", "--thermal-steps", "0", "--eval-steps", "1",
                                       "--checkpoint", str(converted), "--output-dir", str(root / "run")])
                snf.main(args)
                evaluated = snf._safe_torch_load(root / "run" / "snf_checkpoint.pt")["model_state_dict"]
                used = {k.removeprefix(prefix): v for k, v in evaluated.items() if k.startswith(prefix)}
                reference = network(field).state_dict()
                self.assertEqual(used.keys(), reference.keys())
                for key, value in reference.items():
                    torch.testing.assert_close(used[key], value, rtol=0, atol=0)

    def test_full_lattice_fields_are_the_generators_of_their_finite_layers(self):
        from neoqcd.cnf import apply_velocity
        from neoqcd.utils import create_mask
        import neoqcd.sun_utils as sun

        mask = create_mask(4, 4, 4)
        for architecture in ("hyper-smearing", "hyper-residual"):
            with self.subTest(architecture=architecture):
                config = dict(
                    architecture=architecture, domain="beta", D=4, T=4, L=4, beta=6.0,
                    parameter_min=1.0, parameter_max=2.0, batch_size=1, hidden_dim=8, depth=1,
                    rho_init=1e-2, residual_n_pt=1 if architecture == "hyper-residual" else 0,
                    residual_mask_mod=4 if architecture == "hyper-residual" else 2, orsteps=0,
                    hyper_smearing_mode="per_link" if architecture == "hyper-smearing" else "shared",
                )
                torch.manual_seed(15)
                field, _, _ = build_cnf(config)
                network = field.residual if architecture == "hyper-residual" else field.smearing.hyper_smearing
                with torch.no_grad():
                    for parameter in network.parameters():
                        parameter.add_(0.05 * torch.randn_like(parameter))
                cfgs = sun.SUN_haar((1, 4, 4, 4, 4, 4, 3), dtype=torch.cdouble)
                t = torch.tensor([1.4], dtype=torch.double)
                F, div = field(cfgs, t)
                errors = []
                for h in (1e-3, 2e-3):
                    step = torch.tensor([h], dtype=torch.double)
                    with torch.no_grad():
                        if architecture == "hyper-residual":
                            finite, _, logdet = field.residual(cfgs, mask, beta=t, delta_beta=step)
                        else:
                            finite, _, logdet = field.smearing(cfgs, mask, None, beta=t, delta_beta=step)
                    continuous = apply_velocity(cfgs, F, h)
                    self.assertGreater((continuous - cfgs).abs().max().item(), 100 * (finite - continuous).abs().max().item())
                    errors.append(((finite - continuous).abs().max().item(), (logdet - h * div).abs().max().item()))
                # both differences are second order in the step
                for coarse, fine in zip(errors[1], errors[0]):
                    self.assertAlmostEqual(coarse / fine, 4.0, delta=0.5)

    def test_defect_residual_field_converts_to_both_runners(self):
        converter = load_main("convert_cnf_to_smearing_flow")
        snf = load_main("main_nemcmc_snf")
        pt_obc = load_main("main_pt_obc")
        geometry = ["--D", "4", "--T", "6", "--L", "8", "--defect-size", "1", "--time-slice", "3", "--space-slice", "3"]
        network = ["--hyper-hidden-dim", "8", "--hyper-depth", "1", "--residual-n-pt", "1", "--residual-mask-mod", "4"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(
                architecture="defect-hyper-residual", domain="defect", D=4, T=6, L=8, beta=6.0,
                defect_size=1, time_slice=3, space_slice=3, parameter_min=0.0, parameter_max=1.0,
                batch_size=2, hidden_dim=8, depth=1, rho_init=1e-3, residual_n_pt=1,
                residual_mask_mod=4, orsteps=0,
            )
            torch.manual_seed(16)
            field, _, _ = build_cnf(config)
            with torch.no_grad():
                for parameter in field.residual.parameters():
                    parameter.add_(0.05 * torch.randn_like(parameter))
            save_cnf_checkpoint(root / "cnf.pt", field, config, epoch=4)
            reference = field.residual.state_dict()

            snf_flags = ["--algorithm", "snf", "--domain", "defect", "--cpu", *geometry, "--beta", "6",
                         "--bc-initial", "0", "--bc-final", "1", "--nf-architecture", "hyper-residual", *network,
                         "--batch-size", "2", "--protocol-steps", "2", "--prior-mcmc-steps", "0",
                         "--updates-per-layer", "1", "--orsteps", "0", "--cfg-cache", "off"]
            converted = converter.convert("snf", root / "cnf.pt", root / "snf.pt", snf_flags)
            args = snf.parse_args([*snf_flags, "--phase", "evaluate", "--thermal-steps", "0", "--eval-steps", "1",
                                   "--checkpoint", str(converted), "--output-dir", str(root / "run")])
            snf.main(args)
            evaluated = snf._safe_torch_load(root / "run" / "snf_checkpoint.pt")["model_state_dict"]
            prefix = "nf_layer.layer."
            used = {k.removeprefix(prefix): v for k, v in evaluated.items() if k.startswith(prefix)}
            self.assertEqual(used.keys(), reference.keys())
            for key, value in reference.items():
                torch.testing.assert_close(used[key], value, rtol=0, atol=0)

            neorex_flags = ["--cpu", "--algorithm", "neorex", *geometry, "--N", "3", "--beta", "6",
                            "--parameter-min", "0", "--parameter-max", "1", "--nf-layer-type", "residual", *network,
                            "--neorex-protocol-steps", "2", "--batch-size", "2"]
            payload = torch.load(converter.convert("neorex", root / "cnf.pt", root / "neorex.pt", neorex_flags),
                                 weights_only=False)
            prefix = "nf_layer.layer."
            used = {k.removeprefix(prefix): v for k, v in payload["state_dict"].items() if k.startswith(prefix)}
            self.assertEqual(used.keys(), reference.keys())
            for key, value in reference.items():
                torch.testing.assert_close(used[key], value, rtol=0, atol=0)
            self.assertEqual(payload["semantic_config"]["geometry"]["crop_buffer"], 3)

            # Both runners apply the same map on the same patch.
            with mock.patch.object(sys, "argv", ["main_pt_obc", *neorex_flags]):
                pt_args = pt_obc.parse_args()
            defect = Defect(4, 6, 8, 1, 3, 3)
            neorex_layer = pt_obc.ResidualDefectSNFProtocolLayer(
                pt_obc._build_neorex_flow_pars(pt_args, torch.device("cpu"), create_mask(4, 6, 8), defect))
            neorex_layer.layer.load_state_dict(reference)
            flow, _ = snf._build_flow_pars(snf.parse_args(snf_flags), torch.device("cpu"), [0.0, 0.5], 2)
            snf_layer = snf._make_snf_layer(snf.parse_args(snf_flags), flow)
            snf_layer.layer.load_state_dict(reference)
            x = sun.SUN_haar((2, 4, 6, 8, 8, 8, 3), dtype=torch.cdouble)
            bc, delta = torch.tensor([0.2, 0.6], dtype=torch.double), torch.tensor([0.5, 0.5], dtype=torch.double)
            with torch.no_grad():
                y_neorex, logdet_neorex = neorex_layer(x, bc, delta)
                y_snf, logdet_snf = snf_layer(x, bc, delta)
            self.assertGreater((y_neorex - x).abs().max().item(), 1e-4)
            torch.testing.assert_close(y_neorex, y_snf, rtol=0, atol=1e-13)
            torch.testing.assert_close(logdet_neorex, logdet_snf, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
