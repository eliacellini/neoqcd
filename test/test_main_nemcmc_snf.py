import importlib.util
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from neoqcd.flow import Flow


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MAIN_PATH = ROOT / "main" / "main_nemcmc_snf.py"
try:
    SPEC = importlib.util.spec_from_file_location("main_nemcmc_snf", MAIN_PATH)
    MAIN = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MAIN)
except ModuleNotFoundError as exc:
    # The repository's Obs helper imports optional autograd/numdifftools.  Do
    # not install or fake them in a test: the real smoke tests must exercise
    # the same dependency set as production.
    MAIN = None
    IMPORT_ERROR = exc


@unittest.skipUnless(MAIN is not None, "optional runtime dependency missing: autograd/numdifftools")
class StandaloneMainTest(unittest.TestCase):
    @staticmethod
    def _random_su3(shape):
        z = torch.randn(shape + (3, 3), dtype=torch.cdouble)
        q, r = torch.linalg.qr(z)
        diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
        phase = diagonal / torch.clamp(diagonal.abs(), min=1e-14)
        q = q * phase.conj().unsqueeze(-2)
        determinant = torch.linalg.det(q)
        return q / determinant.pow(1.0 / 3.0).unsqueeze(-1).unsqueeze(-1)

    def test_global_batch_and_beta_protocol_are_parsed(self):
        args = MAIN.parse_args(
            [
                "--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta",
                "--cpu", "--batch-size", "6", "--protocol-steps", "2",
                "--beta-initial", "5.8", "--beta-final", "6.2", "--cfg-cache", "off",
            ]
        )
        start, end, _, name = MAIN._protocol_endpoints(args)
        self.assertEqual(args.batch_size, 6)
        self.assertEqual((start, end, name), (5.8, 6.2, "beta"))

    def test_gradient_safety_limit_cli_and_legacy_resume_semantics(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu", "--cfg-cache", "off"]
        )
        self.assertEqual(args.max_consecutive_nan_grad_updates, 5)
        semantics = MAIN._resume_semantics(args, {"schema": 1}, 1)
        legacy = dict(semantics)
        legacy.pop("max_consecutive_nan_grad_updates")
        self.assertTrue(MAIN._run_semantics_match(legacy, semantics))
        args.max_consecutive_nan_grad_updates = 4
        self.assertFalse(MAIN._run_semantics_match(legacy, MAIN._resume_semantics(args, {"schema": 1}, 1)))

        eval_args = MAIN.parse_args(
            ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta", "--cpu", "--cfg-cache", "off"]
        )
        eval_semantics = MAIN._resume_semantics(eval_args, {"schema": 1}, 1)
        self.assertNotIn("max_consecutive_nan_grad_updates", eval_semantics)
        self.assertTrue(MAIN._run_semantics_match(dict(eval_semantics), eval_semantics))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            MAIN.parse_args(
                ["--algorithm", "snf", "--phase", "train", "--cpu", "--max-consecutive-nan-grad-updates", "0"]
            )

    def test_optimizer_gradient_skips_threshold_recovery_and_inactive_streaks(self):
        args = type("Args", (), {"grad_clip_norm": 0.0, "max_consecutive_nan_grad_updates": 2})()

        def finish(optimizers, schedulers, losses, streaks, active=None, world_size=1, all_reduce=None):
            with mock.patch.object(MAIN.dist, "get_world_size", return_value=world_size), mock.patch.object(
                MAIN.dist, "all_reduce", side_effect=all_reduce or (lambda tensor, op=None: None)
            ):
                return MAIN._finish_optimizer_step(
                    optimizers, schedulers, None, losses, args, active, streaks
                )

        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(2.0))
        shared = torch.optim.Adam([first, second], lr=0.1)
        shared_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(shared)
        original = (first.detach().clone(), second.detach().clone())
        first.grad = torch.tensor(float("inf"))
        second.grad = torch.tensor(1.0)
        failed = finish([shared], [shared_scheduler], [1.0], [0])
        self.assertEqual(failed["status"], ["nonfinite_grad"])
        self.assertEqual(failed["skipped"], [1])
        self.assertEqual(failed["consecutive"], [1])
        self.assertIsNone(failed["stop_error"])
        self.assertFalse(shared.state)
        self.assertEqual(shared_scheduler.last_epoch, 0)
        self.assertTrue(torch.equal(first, original[0]))
        self.assertTrue(torch.equal(second, original[1]))

        first.grad = torch.tensor(float("inf"))
        at_limit = finish([shared], [shared_scheduler], [1.0], failed["consecutive"])
        self.assertIn("optimizer 0", at_limit["stop_error"])
        self.assertIn("2/2", at_limit["stop_error"])
        self.assertEqual(at_limit["consecutive"], [2])
        self.assertFalse(shared.state)
        self.assertEqual(shared_scheduler.last_epoch, 0)

        first.grad = torch.tensor(1.0)
        second.grad = torch.tensor(1.0)
        recovered = finish([shared], [shared_scheduler], [1.0], [1])
        self.assertEqual(recovered["status"], ["updated"])
        self.assertEqual(recovered["consecutive"], [0])
        self.assertTrue(shared.state)

        p0 = torch.nn.Parameter(torch.tensor(1.0))
        p1 = torch.nn.Parameter(torch.tensor(2.0))
        optimizers = [torch.optim.Adam([p0]), torch.optim.Adam([p1])]
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt) for opt in optimizers]
        p1.grad = torch.tensor(1.0)
        progressive = finish(optimizers, schedulers, [None, 1.0], [4, 1], [1])
        self.assertEqual(progressive["status"], ["inactive", "updated"])
        self.assertEqual(progressive["skipped"], [0, 0])
        self.assertEqual(progressive["consecutive"], [4, 0])
        no_loss = finish(optimizers, schedulers, [None, None], [4, 3], [1])
        self.assertEqual(no_loss["status"], ["inactive", "no_loss"])
        self.assertEqual(no_loss["skipped"], [0, 1])
        self.assertEqual(no_loss["consecutive"], [4, 3])
        self.assertIsNone(no_loss["stop_error"])

        p1.grad = torch.tensor(1.0)
        recovered_after_no_loss = finish(optimizers, schedulers, [None, 1.0], [4, 3], [1])
        self.assertEqual(recovered_after_no_loss["status"], ["inactive", "updated"])
        self.assertEqual(recovered_after_no_loss["consecutive"], [4, 0])

        missing = torch.nn.Parameter(torch.tensor(1.0))
        missing_optimizer = torch.optim.Adam([missing])
        missing_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(missing_optimizer)
        missing.grad = None

        def other_rank_has_grad(packed, op=None):
            packed[-1] += 1

        missing_result = finish(
            [missing_optimizer], [missing_scheduler], [1.0], [0], world_size=2,
            all_reduce=other_rank_has_grad,
        )
        self.assertEqual(missing_result["status"], ["missing_grad"])
        self.assertEqual(missing_result["consecutive"], [1])
        self.assertFalse(missing_optimizer.state)

    def test_resume_preserves_bad_gradient_streak_and_resets_after_update(self):
        def make_args(output, steps, resume=False):
            argv = [
                "--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
                "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
                "--protocol-steps", "1", "--thermal-steps", "0", "--prior-mcmc-steps", "0",
                "--updates-per-layer", "1", "--orsteps", "0", "--nf-architecture", "residual",
                "--training-mode", "blockwise", "--grad-acc", "2", "--cfg-cache", "off",
                "--seed", "731", "--flow-lr", "0.0001", "--checkpoint-every", "1",
                "--max-consecutive-nan-grad-updates", "3", "--steps", str(steps),
                "--output-dir", str(output),
            ]
            if resume:
                argv.append("--resume")
            return MAIN.parse_args(argv)

        def inject_first_bad_gradient():
            average = MAIN._average_gradients
            injected = {"done": False}

            def wrapped(parameters):
                states = average(parameters)
                if not injected["done"]:
                    parameter = next((parameter for parameter in parameters if parameter.grad is not None), None)
                    if parameter is not None:
                        parameter.grad.reshape(-1)[0] = float("inf")
                        injected["done"] = True
                return states

            return wrapped

        with tempfile.TemporaryDirectory() as directory:
            full_dir = Path(directory) / "full"
            split_dir = Path(directory) / "split"
            with mock.patch.object(MAIN, "_average_gradients", side_effect=inject_first_bad_gradient()):
                MAIN.main(make_args(full_dir, 2))
            with mock.patch.object(MAIN, "_average_gradients", side_effect=inject_first_bad_gradient()):
                MAIN.main(make_args(split_dir, 1))
            first_checkpoint = MAIN._safe_torch_load(split_dir / "snf_checkpoint.pt")
            self.assertEqual(
                first_checkpoint["run_state"]["gradient_safety"]["consecutive_bad_grad_updates"], [1]
            )
            self.assertFalse(first_checkpoint["optimizer_state_dicts"][0]["state"])
            with mock.patch.object(MAIN, "_average_gradients", wraps=MAIN._average_gradients):
                MAIN.main(make_args(split_dir, 2, resume=True))

            full = MAIN._safe_torch_load(full_dir / "snf_checkpoint.pt")
            split = MAIN._safe_torch_load(split_dir / "snf_checkpoint.pt")
            self.assertEqual(split["run_state"]["gradient_safety"]["consecutive_bad_grad_updates"], [0])
            self.assertEqual(split["run_state"]["gradient_safety"]["total_skipped_updates"], 1)
            for name, expected in full["model_state_dict"].items():
                self.assertTrue(torch.equal(split["model_state_dict"][name], expected), name)
            for full_opt, split_opt in zip(full["optimizer_state_dicts"], split["optimizer_state_dicts"]):
                self.assertEqual(full_opt["param_groups"], split_opt["param_groups"])
                self.assertEqual(full_opt["state"].keys(), split_opt["state"].keys())
                for parameter_id, state in full_opt["state"].items():
                    for key, value in state.items():
                        other = split_opt["state"][parameter_id][key]
                        self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)
            self.assertEqual(full["scheduler_state_dicts"], split["scheduler_state_dicts"])
            self.assertTrue(torch.equal(full["rank_states"][0]["prior_state"], split["rank_states"][0]["prior_state"]))
            self.assertEqual(
                (full_dir / "metrics.csv").read_text(encoding="utf-8"),
                (split_dir / "metrics.csv").read_text(encoding="utf-8"),
            )
            with (full_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["skipped_update"] for row in rows], ["1", "0"])
            self.assertEqual(json.loads(rows[0]["optimizer_status_json"]), ["nonfinite_grad"])
            self.assertEqual(json.loads(rows[1]["optimizer_status_json"]), ["updated"])

    def test_hierarchical_cli_defaults_and_geometry_validation(self):
        args = MAIN.parse_args(
            ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "defect",
             "--cpu", "--D", "4", "--T", "12", "--L", "12", "--defect-size", "2",
             "--time-slice", "6", "--space-slice", "5", "--hierarchical-levels", "2",
             "--hierarchical-updates", "1", "2", "--cfg-cache", "off"]
        )
        self.assertEqual(args.hierarchical_buffers, [3, 2])
        flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [0.0, 1.0], 1)
        self.assertEqual(flow.hierarchical_levels, 2)
        self.assertEqual(flow.hierarchical_updates, [1, 2])
        self.assertEqual(flow.hierarchical_rectangles, [3, 2])
        with self.assertRaisesRegex(ValueError, "one count per"):
            MAIN.parse_args(
                ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "defect",
                 "--cpu", "--hierarchical-levels", "2", "--hierarchical-updates", "1"]
            )
        with self.assertRaisesRegex(ValueError, "require --domain defect"):
            MAIN.parse_args(
                ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta",
                 "--cpu", "--hierarchical-levels", "1", "--hierarchical-updates", "1"]
            )

    def test_full_training_keeps_gradients_through_nonzero_mcmc(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
             "--protocol-steps", "1", "--updates-per-layer", "1", "--orsteps", "0",
             "--nf-architecture", "residual", "--training-mode", "full", "--cfg-cache", "off"]
        )
        flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [5.8], 1)
        actual = MAIN.StandaloneProtocol(args, flow, None, 5.8, 6.0)
        reference_flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [5.8], 1)
        reference = MAIN.StandaloneProtocol(args, reference_flow, None, 5.8, 6.0)
        reference.load_state_dict(actual.state_dict())
        x = self._random_su3(flow.init_shape[:-1])

        _, actual_work, _ = actual.full_training_forward(x)
        actual_work.mean().backward()
        actual_grads = [p.grad.detach().clone() if p.grad is not None else None for p in actual.parameters()]

        x_nf, logdet = reference._run_nf_block(x, 0)
        expected_work = MAIN._action(reference_flow, x_nf, "beta", 6.0, None) - logdet
        expected_work.mean().backward()
        expected_grads = [p.grad for p in reference.parameters()]
        self.assertTrue(torch.isfinite(actual_work).all().item())
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            if expected_grad is None:
                self.assertIsNone(actual_grad)
            else:
                self.assertIsNotNone(actual_grad)
                self.assertTrue(torch.allclose(actual_grad, expected_grad, atol=1e-9, rtol=1e-8))

    def test_full_multistep_runner_matches_generic_flow_gradients(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
             "--protocol-steps", "2", "--updates-per-layer", "1", "--orsteps", "0",
             "--nf-architecture", "residual", "--training-mode", "full", "--cfg-cache", "off"]
        )
        targets = [5.8, 5.9, 6.0]
        flow_pars, _ = MAIN._build_flow_pars(args, torch.device("cpu"), targets[:-1], 1)
        standalone = MAIN.StandaloneProtocol(args, flow_pars, None, targets[0], targets[-1], targets)

        reference_pars, _ = MAIN._build_flow_pars(args, torch.device("cpu"), targets[1:], 1)
        reference_pars.nf_layer_type = "residual"
        reference = Flow(reference_pars, flow_type="snf", protocol_type="beta")
        for step in range(args.protocol_steps):
            reference.layers[2 * step].load_state_dict(
                standalone.nf_layers[step].layer.state_dict()
            )
        x = self._random_su3(flow_pars.init_shape[:-1])

        torch.manual_seed(314)
        _, standalone_work, _ = standalone.full_training_forward(x)
        standalone_work.mean().backward()
        torch.manual_seed(314)
        initial_action = MAIN._action(reference_pars, x, "beta", targets[0], None)
        _, reference_work, _, _, _, _ = reference(x, initial_action, reference_pars)
        reference_work.mean().backward()

        self.assertTrue(torch.allclose(standalone_work, reference_work, atol=1e-10, rtol=1e-9))
        for step in range(args.protocol_steps):
            actual_parameters = list(standalone.nf_layers[step].layer.parameters())
            expected_parameters = list(reference.layers[2 * step].parameters())
            self.assertEqual(len(actual_parameters), len(expected_parameters))
            for actual_parameter, expected_parameter in zip(actual_parameters, expected_parameters):
                self.assertIsNotNone(actual_parameter.grad)
                self.assertIsNotNone(expected_parameter.grad)
                self.assertTrue(torch.allclose(actual_parameter.grad, expected_parameter.grad, atol=1e-9, rtol=1e-8))

    def test_custom_protocol_file_validates_targets_and_endpoint(self):
        args = MAIN.parse_args(
            ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta",
             "--cpu", "--protocol-steps", "2", "--beta-initial", "5.8",
             "--beta-final", "6.0", "--protocol-file", "placeholder", "--cfg-cache", "off"]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.txt"
            args.protocol_file = str(path)
            path.write_text("5.9\n6.0\n", encoding="utf-8")
            self.assertEqual(MAIN._protocol_values(args, 5.8, 6.0), [5.8, 5.9, 6.0])
            path.write_text("5.9\n5.99\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must equal target"):
                MAIN._protocol_values(args, 5.8, 6.0)
            path.write_text("5.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 2"):
                MAIN._protocol_values(args, 5.8, 6.0)

    def test_resume_csv_truncates_uncheckpointed_rows_and_checks_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            MAIN._prepare_csv(path, MAIN.METRICS_HEADER, False)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("1,one\n2,two\n3,three\n")
            self.assertEqual(MAIN._prepare_csv(path, MAIN.METRICS_HEADER, True, keep_rows=1), 1)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
            path.write_text("wrong,header\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "header does not match"):
                MAIN._validate_resume_csv(path, MAIN.METRICS_HEADER, 0)

    def test_resume_migrates_legacy_metrics_without_claiming_optimizer_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(MAIN.LEGACY_METRICS_HEADER)
                writer.writerow([1, 1.0, 0.1, 0.0, 1.0, 1, 0.5, "[0.5]", "[0]", 0, "off"])
            MAIN._validate_resume_csv(path, MAIN.METRICS_HEADER, 1, MAIN.LEGACY_METRICS_HEADER)
            self.assertEqual(
                MAIN._prepare_csv(
                    path, MAIN.METRICS_HEADER, True, 1,
                    legacy_header=MAIN.LEGACY_METRICS_HEADER,
                ),
                1,
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0]["block_skipped_json"]), [0])
            self.assertEqual(json.loads(rows[0]["optimizer_status_json"]), [])
            self.assertEqual(rows[0]["skipped_update"], "")

    def test_checkpoint_rng_round_trip_restores_python_numpy_and_torch(self):
        import random

        random.seed(123)
        np_state = MAIN.np.random.get_state()
        MAIN.np.random.seed(456)
        torch.manual_seed(789)
        state = MAIN._checkpoint_rng_state(torch.device("cpu"))
        expected = (random.random(), MAIN.np.random.rand(), torch.rand(3))
        random.seed(0)
        MAIN.np.random.seed(0)
        torch.manual_seed(0)
        MAIN._restore_checkpoint_rng_state(state, torch.device("cpu"))
        actual = (random.random(), MAIN.np.random.rand(), torch.rand(3))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        MAIN.np.random.set_state(np_state)

    def test_runtime_seed_reproduces_python_numpy_and_torch_streams(self):
        args = MAIN.parse_args(
            ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta",
             "--cpu", "--seed", "731", "--batch-size", "1", "--cfg-cache", "off"]
        )
        with mock.patch.object(MAIN, "_init_process_group", return_value=(0, 1, False)):
            samples = []
            for _ in range(2):
                MAIN.setup_runtime(args)
                samples.append((MAIN.random.random(), MAIN.np.random.random(), torch.rand(()).item()))
        self.assertEqual(samples[0], samples[1])

    def test_resume_preflight_rejects_progress_and_missing_optimizer_states(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
             "--protocol-steps", "1", "--nf-architecture", "residual", "--cfg-cache", "off"]
        )
        rank_state = {
            "prior_state": torch.view_as_real(
                torch.eye(3, dtype=torch.cdouble).view(1, 1, 1, 1, 3, 3).expand(
                    MAIN._expected_cfg_shape(args, 1)
                ).contiguous()
            ),
            "rng_state": MAIN._checkpoint_rng_state(torch.device("cpu")),
            "evaluation_work": [],
        }
        malformed = {
            "progress": 2, "metric_rows": 2, "observable_rows": 0,
            "rank_states": [rank_state],
        }
        with self.assertRaisesRegex(ValueError, "requested total"):
            MAIN._validate_resume_checkpoint(malformed, args, 1, 1, torch.device("cpu"), 1)

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        with self.assertRaisesRegex(ValueError, "optimizer count"):
            MAIN._restore_optimizer_states(
                {"optimizer_state_dicts": [], "scheduler_state_dicts": []},
                [optimizer], [scheduler],
            )

    def test_eval_samples_is_exclusive_and_requests_global_samples(self):
        args = MAIN.parse_args(
            ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta",
             "--cpu", "--batch-size", "16", "--eval-samples", "100", "--cfg-cache", "off"]
        )
        self.assertEqual(args.eval_samples, 100)
        with self.assertRaises(ValueError):
            MAIN.parse_args(
                ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta",
                 "--cpu", "--batch-size", "16", "--eval-samples", "100", "--eval-steps", "2",
                 "--cfg-cache", "off"]
            )

    def test_default_builder_and_beta_per_link_builder_are_compatible(self):
        default = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "4", "--T", "4", "--L", "4", "--batch-size", "1", "--cfg-cache", "off"]
        )
        flow, _ = MAIN._build_flow_pars(default, torch.device("cpu"), [5.8,], 1)
        self.assertEqual(flow.rho_shape_type, 0)  # default shared hyper-smearing

        per_link = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "4", "--T", "4", "--L", "4", "--batch-size", "1", "--cfg-cache", "off",
             "--hyper-smearing-mode", "per_link"]
        )
        flow, _ = MAIN._build_flow_pars(per_link, torch.device("cpu"), [5.8,], 1)
        self.assertEqual(flow.rho_shape_type, 1)

    def test_functional_rho_is_shared_across_steps_and_trainable(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "4", "--T", "4", "--L", "4", "--batch-size", "1",
             "--protocol-steps", "3", "--rho-functional", "--training-mode", "full", "--cfg-cache", "off"]
        )
        flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [5.8, 5.9, 6.0], 1)
        model = MAIN.StandaloneProtocol(args, flow, None, 5.8, 6.0)
        self.assertEqual(model.nf_layer_sharing, "shared")
        self.assertIsNotNone(model.rho_functional)
        self.assertEqual(tuple(model.rho_functional(1 / 3).shape), tuple(flow.rho_shape))
        (model.rho_functional(1 / 3).square().sum()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.rho_functional.parameters()))
        layer = model.nf_layer.layer
        self.assertFalse(isinstance(layer.rho_layer, torch.nn.Parameter))

    def test_smearing_dimension_validation_and_positive_trainable_rho(self):
        with self.assertRaisesRegex(ValueError, "D=4"):
            MAIN.parse_args(
                ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
                 "--D", "2", "--T", "2", "--L", "2", "--batch-size", "2",
                 "--nf-architecture", "hyper-smearing", "--hyper-smearing-mode", "per_link"]
            )

        residual = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "2", "--L", "2", "--batch-size", "2",
             "--nf-architecture", "residual", "--cfg-cache", "off"]
        )
        self.assertEqual(residual.D, 2)

        standard = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "4", "--T", "4", "--L", "4", "--batch-size", "1",
             "--nf-architecture", "smearing", "--smearing-rho-init", "0.002", "--cfg-cache", "off"]
        )
        flow, _ = MAIN._build_flow_pars(standard, torch.device("cpu"), [5.8,], 1)
        layer = MAIN._make_snf_layer(standard, flow).layer
        self.assertIsInstance(layer.rho_layer, torch.nn.Parameter)
        self.assertTrue(torch.all(layer.rho_layer == 0.002).item())
        self.assertTrue(layer.rho_layer.requires_grad)

        with self.assertRaises(ValueError):
            MAIN.parse_args(
                ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
                 "--D", "4", "--T", "4", "--L", "4", "--batch-size", "1",
                 "--nf-architecture", "smearing", "--smearing-rho-init", "0"]
            )

    def test_hyper_smearing_batched_rho_broadcasts_on_d4(self):
        for mode in ("shared", "per_link"):
            args = MAIN.parse_args(
                ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
                 "--D", "4", "--T", "4", "--L", "4", "--batch-size", "2",
                 "--protocol-steps", "1", "--nf-architecture", "hyper-smearing",
                 "--hyper-smearing-mode", mode, "--cfg-cache", "off"]
            )
            flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [5.8], 2)
            model = MAIN.StandaloneProtocol(args, flow, None, 5.8, 6.0)
            x = self._random_su3(flow.init_shape[:-1])
            with torch.no_grad():
                s0 = MAIN._action(flow, x, "beta", 5.8, None)
            _, work, _, _, _ = model.up_to_block_nograd(x, s0, 0)
            self.assertTrue(torch.isfinite(work).all().item())
            work.mean().backward()
            self.assertTrue(
                all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                )
            )

    def test_defect_class_has_nine_trainable_classes_and_fixed_inactive_baseline(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "defect", "--cpu",
             "--D", "4", "--T", "4", "--L", "5", "--defect-size", "1",
             "--time-slice", "2", "--space-slice", "2", "--batch-size", "1",
             "--protocol-steps", "1", "--nf-architecture", "hyper-smearing",
             "--hyper-smearing-mode", "class", "--cfg-cache", "off"]
        )
        flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [5.8], 1)
        layer = MAIN._make_snf_layer(args, flow).layer
        hyper = layer.smearing.hyper_smearing
        self.assertEqual(tuple(hyper.class_masks.shape[:1]), (9,))
        self.assertEqual(hyper.mlp[-1].bias.numel(), 9)
        self.assertTrue(hyper.mlp[-1].bias.requires_grad)
        rho = hyper(beta=torch.tensor([5.9]), delta_beta=torch.tensor([0.0]))
        active = hyper.class_masks.sum(dim=0) > 0
        self.assertTrue(torch.isfinite(rho).all().item())
        self.assertTrue(torch.all(rho[0][~active] == MAIN.HYPER_CLASS_INACTIVE_RHO).item())
        self.assertTrue(torch.all(rho[0][active] == 0.0).item())

    def test_runner_rejects_a_run_with_no_finite_work(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
             "--protocol-steps", "1", "--thermal-steps", "1", "--steps", "1",
             "--updates-per-layer", "0", "--nf-architecture", "residual",
             "--cfg-cache", "off"]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            args.output_dir = output_dir
            def nonfinite_blockwise(_model, x, *_args):
                bad = torch.full((x.shape[0],), float("inf"), dtype=x.real.dtype, device=x.device)
                return x, bad, [None], [1]

            with mock.patch.object(MAIN.StandaloneProtocol, "blockwise_training_forward", nonfinite_blockwise):
                with self.assertRaisesRegex(RuntimeError, "No finite work values"):
                    MAIN.main(args)
            self.assertFalse((Path(output_dir) / "snf_checkpoint.pt").exists())

    def test_interrupted_training_resume_matches_uninterrupted_weights_and_rows(self):
        def make_args(output, steps, resume=False):
            argv = [
                "--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
                "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
                "--protocol-steps", "2", "--thermal-steps", "0", "--prior-mcmc-steps", "0",
                "--updates-per-layer", "0", "--nf-architecture", "residual", "--cfg-cache", "off",
                "--seed", "223", "--flow-lr", "0.0001", "--checkpoint-every", "1",
                "--steps", str(steps), "--output-dir", str(output),
            ]
            if resume:
                argv.append("--resume")
            return MAIN.parse_args(argv)

        with tempfile.TemporaryDirectory() as directory:
            full_dir = Path(directory) / "full"
            resumed_dir = Path(directory) / "resumed"
            MAIN.main(make_args(full_dir, 2))
            MAIN.main(make_args(resumed_dir, 1))
            MAIN.main(make_args(resumed_dir, 2, resume=True))
            full = MAIN._safe_torch_load(full_dir / "snf_checkpoint.pt")
            resumed = MAIN._safe_torch_load(resumed_dir / "snf_checkpoint.pt")
            self.assertEqual(full["progress"], 2)
            self.assertEqual(resumed["progress"], 2)
            for name, expected in full["model_state_dict"].items():
                self.assertTrue(torch.equal(resumed["model_state_dict"][name], expected), name)
            self.assertEqual(
                (full_dir / "metrics.csv").read_text(encoding="utf-8"),
                (resumed_dir / "metrics.csv").read_text(encoding="utf-8"),
            )
            eval_dir = Path(directory) / "continued-evaluation"
            eval_args = MAIN.parse_args(
                ["--algorithm", "snf", "--phase", "evaluate", "--domain", "beta", "--cpu",
                 "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
                 "--protocol-steps", "2", "--prior-mcmc-steps", "0", "--updates-per-layer", "0",
                 "--nf-architecture", "residual", "--cfg-cache", "off", "--seed", "999",
                 "--eval-steps", "1", "--checkpoint", str(resumed_dir / "snf_checkpoint.pt"),
                 "--load-prior", "--output-dir", str(eval_dir)]
            )
            with mock.patch.object(MAIN, "prepare_prior", side_effect=AssertionError("must use checkpoint chain")):
                MAIN.main(eval_args)
            continued = MAIN._safe_torch_load(eval_dir / "snf_checkpoint.pt")
            self.assertEqual(continued["progress"], 1)
            self.assertTrue(torch.equal(
                continued["rank_states"][0]["prior_state"],
                resumed["rank_states"][0]["prior_state"],
            ))

    def test_nemcmc_checkpoint_can_continue_prior_without_model_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory) / "source"
            source_args = MAIN.parse_args(
                ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta", "--cpu",
                 "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
                 "--protocol-steps", "1", "--thermal-steps", "0", "--prior-mcmc-steps", "0",
                 "--updates-per-layer", "0", "--orsteps", "0", "--cfg-cache", "off",
                 "--eval-steps", "1", "--checkpoint-every", "1", "--output-dir", str(source_dir)]
            )
            MAIN.main(source_args)
            source_checkpoint = MAIN._safe_torch_load(source_dir / "snf_checkpoint.pt")
            self.assertIsNone(source_checkpoint["model_state_dict"])

            continued_dir = Path(directory) / "continued"
            continued_args = MAIN.parse_args(
                ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta", "--cpu",
                 "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
                 "--protocol-steps", "1", "--thermal-steps", "0", "--prior-mcmc-steps", "0",
                 "--updates-per-layer", "0", "--orsteps", "0", "--cfg-cache", "off",
                 "--eval-steps", "1", "--checkpoint-every", "1", "--load-prior",
                 "--checkpoint", str(source_dir / "snf_checkpoint.pt"), "--output-dir", str(continued_dir)]
            )
            with mock.patch.object(MAIN, "prepare_prior", side_effect=AssertionError("must use checkpoint chain")):
                MAIN.main(continued_args)
            continued_checkpoint = MAIN._safe_torch_load(continued_dir / "snf_checkpoint.pt")
            self.assertTrue(torch.equal(
                continued_checkpoint["rank_states"][0]["prior_state"],
                source_checkpoint["rank_states"][0]["prior_state"],
            ))

    def test_evaluation_gamma_report_is_rebuilt_after_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"

            def args(steps, resume=False):
                argv = [
                    "--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta", "--cpu",
                    "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
                    "--protocol-steps", "2", "--int-meas", "2", "--thermal-steps", "0",
                    "--prior-mcmc-steps", "0", "--updates-per-layer", "0", "--orsteps", "0",
                    "--cfg-cache", "off", "--eval-steps", str(steps), "--checkpoint-every", "1",
                    "--seed", "193", "--output-dir", str(output),
                ]
                if resume:
                    argv.append("--resume")
                return MAIN.parse_args(argv)

            MAIN.main(args(3))
            first = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(first["gamma_analysis_status"], "insufficient_statistics")
            MAIN.main(args(8, resume=True))

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            metadata = json.loads((output / "run_meta.json").read_text(encoding="utf-8"))
            with (output / "observables.csv").open(newline="", encoding="utf-8") as handle:
                observations = list(csv.DictReader(handle))
            with (output / "gamma_analysis.csv").open(newline="", encoding="utf-8") as handle:
                report = list(csv.DictReader(handle))
            self.assertEqual(len(observations), 16)  # eight trajectories × two intermediate measurements
            self.assertEqual(len(report), 28)  # fourteen observables × two measurements
            self.assertEqual(len({(row["measurement"], row["observable"]) for row in report}), len(report))
            self.assertEqual(summary["gamma_analysis_status"], "ok")
            self.assertTrue(math.isfinite(summary["work_sem_gamma_final"]))
            self.assertEqual(summary["work_sem_method"], "pooled_iid")
            self.assertEqual(metadata["gamma_analysis"]["analyzed_trajectory_count"], 8)
            self.assertEqual(metadata["work_sem_method"], "pooled_iid")

    def test_shared_hyper_blockwise_training_and_resume(self):
        def make_args(output, architecture, steps, resume=False):
            argv = [
                "--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
                "--D", "4", "--T", "4", "--L", "4", "--batch-size", "1",
                "--protocol-steps", "2", "--thermal-steps", "0", "--prior-mcmc-steps", "0",
                "--updates-per-layer", "1", "--orsteps", "0", "--nf-architecture", architecture,
                "--training-mode", "blockwise", "--grad-acc", "2", "--cfg-cache", "off",
                "--seed", "563", "--flow-lr", "0.0001", "--checkpoint-every", "1",
                "--steps", str(steps), "--output-dir", str(output),
            ]
            if resume:
                argv.append("--resume")
            return MAIN.parse_args(argv)

        for architecture in ("hyper-smearing", "hyper-residual"):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                full_dir = Path(directory) / "full"
                resumed_dir = Path(directory) / "resumed"
                full_args = make_args(full_dir, architecture, 2)

                # Construct the same seeded model state as the runner, before its first optimizer step.
                MAIN.random.seed(full_args.seed)
                MAIN.np.random.seed(full_args.seed)
                torch.manual_seed(full_args.seed)
                start, end, fixed_beta, _ = MAIN._protocol_endpoints(full_args)
                protocol = MAIN._protocol_values(full_args, start, end)
                flow_pars, defect_mask = MAIN._build_flow_pars(
                    full_args, torch.device("cpu"), protocol[:-1], full_args.batch_size
                )
                flow_pars._standalone_beta = fixed_beta
                initial_model = MAIN.StandaloneProtocol(
                    full_args, flow_pars, defect_mask, start, end, protocol
                )
                initial_parameters = {
                    name: parameter.detach().clone()
                    for name, parameter in initial_model.named_parameters()
                }

                MAIN.main(full_args)
                MAIN.main(make_args(resumed_dir, architecture, 1))
                MAIN.main(make_args(resumed_dir, architecture, 2, resume=True))

                full = MAIN._safe_torch_load(full_dir / "snf_checkpoint.pt")
                resumed = MAIN._safe_torch_load(resumed_dir / "snf_checkpoint.pt")
                self.assertEqual(full["progress"], 2)
                self.assertEqual(resumed["progress"], 2)
                changed = False
                for name, expected in full["model_state_dict"].items():
                    self.assertTrue(torch.equal(resumed["model_state_dict"][name], expected), name)
                    if name in initial_parameters and not torch.equal(expected, initial_parameters[name]):
                        changed = True
                self.assertTrue(changed, "shared hypernetwork parameters should be updated")
                self.assertEqual(len(full["optimizer_state_dicts"]), 1)
                optimizer_state = full["optimizer_state_dicts"][0]["state"]
                self.assertTrue(optimizer_state, "the shared optimizer should take optimizer steps")
                self.assertTrue(any(
                    int(state["step"].item()) == 2 for state in optimizer_state.values()
                ))
                for full_state, resumed_state in zip(
                    full["optimizer_state_dicts"], resumed["optimizer_state_dicts"]
                ):
                    self.assertEqual(full_state["param_groups"], resumed_state["param_groups"])
                    self.assertEqual(full_state["state"].keys(), resumed_state["state"].keys())
                    for parameter_id, state in full_state["state"].items():
                        for key, value in state.items():
                            if torch.is_tensor(value):
                                self.assertTrue(torch.equal(value, resumed_state["state"][parameter_id][key]))
                            else:
                                self.assertEqual(value, resumed_state["state"][parameter_id][key])
                self.assertEqual(full["scheduler_state_dicts"], resumed["scheduler_state_dicts"])
                self.assertTrue(torch.equal(
                    full["rank_states"][0]["prior_state"], resumed["rank_states"][0]["prior_state"]
                ))
                self.assertTrue(torch.equal(
                    full["rank_states"][0]["rng_state"]["cpu"], resumed["rank_states"][0]["rng_state"]["cpu"]
                ))
                full_rng, resumed_rng = full["rank_states"][0]["rng_state"], resumed["rank_states"][0]["rng_state"]
                self.assertEqual(full_rng["python"], resumed_rng["python"])
                self.assertEqual(full_rng["numpy"][0], resumed_rng["numpy"][0])
                self.assertTrue(torch.equal(full_rng["numpy"][1], resumed_rng["numpy"][1]))
                self.assertEqual(full_rng["numpy"][2:], resumed_rng["numpy"][2:])

                with (full_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 2)
                for row in rows:
                    self.assertTrue(math.isfinite(float(row["work_mean"])))
                    self.assertTrue(math.isfinite(float(row["loss_epoch"])))
                    block_losses = json.loads(row["block_loss_json"])
                    block_skips = json.loads(row["block_skipped_json"])
                    self.assertEqual(len(block_losses), 2)
                    self.assertTrue(all(math.isfinite(value) for value in block_losses))
                    self.assertEqual(len(block_skips), 2)
                    self.assertEqual(block_skips, [0, 0])
                    self.assertEqual(json.loads(row["optimizer_status_json"]), ["updated"])
                    self.assertEqual(json.loads(row["optimizer_skipped_json"]), [0])
                    self.assertEqual(json.loads(row["optimizer_grad_streak_json"]), [0])
                    self.assertEqual(row["skipped_update"], "0")
                self.assertEqual(
                    (full_dir / "metrics.csv").read_text(encoding="utf-8"),
                    (resumed_dir / "metrics.csv").read_text(encoding="utf-8"),
                )

    def test_updates_zero_keeps_two_step_work_graph_and_telescopes(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
             "--protocol-steps", "2", "--updates-per-layer", "0",
             "--nf-architecture", "residual", "--cfg-cache", "off"]
        )
        flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [5.8, 5.9], 1)
        model = MAIN.StandaloneProtocol(args, flow, None, 5.8, 6.0)
        x = self._random_su3(flow.init_shape[:-1])
        with torch.no_grad():
            s0 = MAIN._action(flow, x, "beta", 5.8, None)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer_steps = 0
        for block in range(2):
            optimizer.zero_grad(set_to_none=True)
            _, block_work, _, _, _ = model.up_to_block_nograd(x, s0, block)
            self.assertTrue(block_work.requires_grad)
            block_work.mean().backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            self.assertTrue(any(gradient is not None and torch.isfinite(gradient).all().item() and gradient.abs().max().item() > 0.0 for gradient in gradients))
            optimizer.step()
            optimizer_steps += 1
        self.assertEqual(optimizer_steps, 2)

    def test_blockwise_gradient_accumulation_averages_microbatches(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
             "--protocol-steps", "2", "--updates-per-layer", "0",
             "--nf-architecture", "residual", "--cfg-cache", "off"]
        )
        flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [5.8, 5.9], 1)
        accumulated = MAIN.StandaloneProtocol(args, flow, None, 5.8, 6.0)
        reference = MAIN.StandaloneProtocol(args, flow, None, 5.8, 6.0)
        reference.load_state_dict(accumulated.state_dict())
        x = self._random_su3(flow.init_shape[:-1])
        s0 = MAIN._action(flow, x, "beta", 5.8, None)
        with mock.patch.object(MAIN.dist, "all_reduce", side_effect=lambda *_args, **_kwargs: None), \
             mock.patch.object(MAIN.dist, "get_world_size", return_value=1):
            for _ in range(2):
                accumulated.blockwise_training_forward(x, s0, scale=2, world_size=1, device=torch.device("cpu"))
            reference.blockwise_training_forward(x, s0, scale=1, world_size=1, device=torch.device("cpu"))
        accumulated_grads = [p.grad for p in accumulated.parameters() if p.requires_grad]
        reference_grads = [p.grad for p in reference.parameters() if p.requires_grad]
        self.assertEqual(len(accumulated_grads), len(reference_grads))
        for actual, expected in zip(accumulated_grads, reference_grads):
            self.assertIsNotNone(actual)
            self.assertIsNotNone(expected)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=1e-8))

    def test_shared_layer_and_residual_parameterizations(self):
        for architecture, parameterization, has_mlp in (
            ("residual", "static", False),
            ("hyper-residual", "hyper", True),
        ):
            args = MAIN.parse_args(
                ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
                 "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
                 "--protocol-steps", "3", "--nf-architecture", architecture, "--cfg-cache", "off"]
            )
            flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [5.8, 5.9, 6.0], 1)
            model = MAIN.StandaloneProtocol(args, flow, None, 5.8, 6.0)
            if parameterization == "hyper":
                self.assertEqual(model.nf_layer_sharing, "shared")
                self.assertIsNotNone(model.nf_layer)
                self.assertIsNone(model.nf_layers)
                layer_objects = [model.nf_layer]
            else:
                self.assertEqual(model.nf_layer_sharing, "per_block")
                self.assertIsNone(model.nf_layer)
                self.assertEqual(len(model.nf_layers), 3)
                self.assertEqual(len({id(layer) for layer in model.nf_layers}), 3)
                layer_objects = list(model.nf_layers)
            residual_flow = layer_objects[0].layer.residual_flow
            self.assertEqual(residual_flow.parameterization, parameterization)
            self.assertEqual(hasattr(residual_flow, "mlp"), has_mlp)
            if not has_mlp:
                self.assertIsInstance(residual_flow.static_linear, torch.nn.Parameter)
                self.assertGreater(residual_flow.static_linear.abs().max().item(), 0.0)
            x = self._random_su3(flow.init_shape[:-1])
            with torch.no_grad():
                s0 = MAIN._action(flow, x, "beta", 5.8, None)
            _, work, _, _, _ = model.up_to_block_nograd(x, s0, 0)
            self.assertTrue(torch.isfinite(work).all().item())
            work.mean().backward()
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
            )
            if parameterization == "static":
                model.zero_grad(set_to_none=True)
                _, work_block_1, _, _, _ = model.up_to_block_nograd(x, s0, 1)
                work_block_1.mean().backward()
                self.assertTrue(all(parameter.grad is None for parameter in model.nf_layers[0].parameters()))
                self.assertTrue(any(parameter.grad is not None for parameter in model.nf_layers[1].parameters()))

    def test_static_residual_quadratic_is_symmetrized_with_finite_backward(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
             "--nf-architecture", "residual", "--cfg-cache", "off"]
        )
        flow, _ = MAIN._build_flow_pars(args, torch.device("cpu"), [5.8], 1)
        residual = MAIN.ResidualNormalizingFlows(flow)
        with torch.no_grad():
            raw = torch.arange(
                residual.n_features * residual.n_features,
                dtype=torch.float64,
            ).view(residual.n_features, residual.n_features)
            residual.static_quadratic[0].copy_(raw)
        _, quad, _ = residual._coefficients(
            beta=torch.tensor([5.8]),
            delta_beta=torch.tensor([0.1]),
            batch_size=1,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        expected = 0.5 * (residual.static_quadratic[0] + residual.static_quadratic[0].transpose(-1, -2))
        self.assertTrue(torch.allclose(quad[0, 0], expected))
        (quad.square().sum()).backward()
        self.assertTrue(torch.isfinite(residual.static_quadratic.grad).all().item())
        self.assertLess(residual.static_quadratic.grad.sub(residual.static_quadratic.grad.transpose(-1, -2)).abs().max().item(), 1e-12)

    def test_checkpoint_semantics_schema_and_nonweight_forward_parameters(self):
        args = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "4", "--T", "4", "--L", "4", "--batch-size", "1",
             "--nf-architecture", "residual", "--hyper-rho-eps", "1e-5",
             "--hyper-rho-max", "0.2", "--residual-coeff-max", "0.3"]
        )
        semantics = MAIN._checkpoint_semantics(args, [5.8, 6.0])
        self.assertEqual(semantics["schema"], 6)
        self.assertEqual(semantics["hyper_rho_eps"], 1e-5)
        self.assertEqual(semantics["hyper_rho_max"], 0.2)
        self.assertEqual(semantics["residual_coeff_max"], 0.3)
        self.assertEqual(semantics["nf_layer_sharing"], "per_block")

    def test_cache_contract_has_no_world_size(self):
        args = MAIN.parse_args(
            [
                "--algorithm", "snf", "--phase", "train", "--domain", "defect",
                "--cpu", "--D", "4", "--T", "8", "--L", "8", "--batch-size", "4",
                "--cfg-cache", "off",
            ]
        )
        key = MAIN.cache_key(args, 0.0, 1.0, "bc")
        self.assertEqual(key["global_batch"], 4)
        self.assertNotIn("world_size", key)
        self.assertEqual(key["domain"], "defect")

    def test_cache_key_ignores_target_and_protocol_update_count(self):
        args = MAIN.parse_args(
            ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta",
             "--cpu", "--batch-size", "2", "--cfg-cache", "off", "--thermal-steps", "3"]
        )
        first = MAIN.cache_key(args, 5.8, 6.0, "beta")
        args.thermal_steps = 20
        args.updates_per_layer = 7
        second = MAIN.cache_key(args, 5.8, 6.7, "beta")
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], 6)

    def test_cache_requires_current_done_marker(self):
        args = MAIN.parse_args(
            ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta",
             "--cpu", "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
             "--cfg-cache", "off"]
        )
        key = MAIN.cache_key(args, 5.8, 6.0, "beta")
        cfgs = torch.eye(3, dtype=torch.cdouble).view((1, 1, 1, 1, 3, 3)).expand(
            MAIN._expected_cfg_shape(args, 1)
        ).clone()
        with tempfile.TemporaryDirectory() as cache_dir:
            path = Path(cache_dir)
            payload_name = "prior_cfgs.pt"
            torch.save({"cfgs": cfgs}, path / payload_name)

            hit, _, _ = MAIN._validated_cache_payload(path, key, args)
            self.assertFalse(hit)

            with open(path / "DONE", "w", encoding="utf-8") as handle:
                json.dump({"status": "ok", "schema": 6, "key": {"stale": True},
                           "payload": payload_name, "thermal_steps": 10}, handle)
            hit, _, _ = MAIN._validated_cache_payload(path, key, args)
            self.assertFalse(hit)

            with open(path / "DONE", "w", encoding="utf-8") as handle:
                json.dump({"status": "ok", "schema": 6, "key": key,
                           "payload": payload_name, "thermal_steps": 10}, handle)
            # meta.json is advisory: a concurrent publisher may overwrite it.
            with open(path / "meta.json", "w", encoding="utf-8") as handle:
                json.dump({"schema": 1, "key": {"stale": True}}, handle)
            hit, cached, _ = MAIN._validated_cache_payload(path, key, args)
            self.assertTrue(hit)
            self.assertTrue(torch.equal(cached["cfgs"], cfgs))

    def test_thermal_seed_stream_is_separate_from_model_architecture(self):
        residual = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "2", "--T", "2", "--L", "2", "--batch-size", "1",
             "--nf-architecture", "residual", "--cfg-cache", "off"]
        )
        hyper = MAIN.parse_args(
            ["--algorithm", "snf", "--phase", "train", "--domain", "beta", "--cpu",
             "--D", "4", "--T", "4", "--L", "4", "--batch-size", "1",
             "--nf-architecture", "hyper-smearing", "--cfg-cache", "off"]
        )
        self.assertEqual(MAIN._thermal_seed(residual, 0), MAIN._thermal_seed(hyper, 0))
        self.assertNotEqual(MAIN._thermal_seed(residual, 0), MAIN._thermal_seed(residual, 1))
        self.assertEqual(residual.seed, hyper.seed)

    def test_persistent_prior_chain_is_not_reset_or_mutated_by_target_clone(self):
        class DummyPriorModel:
            def __init__(self):
                self.inputs = []

            def advance_prior(self, state, steps):
                self.inputs.append(state.clone())
                return state + int(steps)

        model = DummyPriorModel()
        prior_state = torch.tensor([0.0])
        prior_state = MAIN._advance_persistent_prior(model, prior_state, 1)
        target = prior_state.detach().clone()
        target.add_(100.0)
        prior_state = MAIN._advance_persistent_prior(model, prior_state, 1)
        self.assertEqual([float(value.item()) for value in model.inputs], [0.0, 1.0])
        self.assertEqual(float(prior_state.item()), 2.0)

    def test_global_work_metrics_are_finite_and_include_free_energy_observables(self):
        init_file = tempfile.NamedTemporaryFile(delete=False)
        init_file.close()
        try:
            dist = torch.distributed
            dist.init_process_group(
                backend="gloo", init_method=f"file://{init_file.name}", rank=0, world_size=1
            )
            try:
                metrics = MAIN._global_work_metrics(torch.tensor([1000.0, 1001.0], dtype=torch.float64))
            finally:
                dist.destroy_process_group()
        finally:
            Path(init_file.name).unlink(missing_ok=True)
        self.assertTrue(all(value is None or torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertIsNotNone(metrics["deltaF"])
        self.assertIsNotNone(metrics["ess_fraction"])

    def test_pooled_metrics_are_not_an_average_of_batch_delta_f(self):
        init_file = tempfile.NamedTemporaryFile(delete=False)
        init_file.close()
        try:
            dist = torch.distributed
            dist.init_process_group(
                backend="gloo", init_method=f"file://{init_file.name}", rank=0, world_size=1
            )
            try:
                pooled = MAIN._global_work_metrics(torch.tensor([0.0, 2.0], dtype=torch.float64))
            finally:
                dist.destroy_process_group()
        finally:
            Path(init_file.name).unlink(missing_ok=True)
        expected_delta_f = -torch.log((1.0 + torch.exp(torch.tensor(-2.0))) / 2.0).item()
        self.assertAlmostEqual(pooled["deltaF"], expected_delta_f, places=6)
        self.assertNotAlmostEqual(pooled["deltaF"], 1.0, places=3)
        self.assertEqual(pooled["finite_count"], 2)

    def test_nemcmc_training_is_rejected_and_checkpoint_is_required_for_snf_eval(self):
        with self.assertRaises(ValueError):
            MAIN.parse_args(["--algorithm", "nemcmc", "--phase", "train"])
        with self.assertRaises(ValueError):
            MAIN.parse_args(["--algorithm", "snf", "--phase", "evaluate"])
        with self.assertRaisesRegex(ValueError, "requires --load-prior"):
            MAIN.parse_args(
                ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta",
                 "--cpu", "--checkpoint", "prior.pt", "--cfg-cache", "off"]
            )
        args = MAIN.parse_args(
            ["--algorithm", "nemcmc", "--phase", "evaluate", "--domain", "beta",
             "--cpu", "--checkpoint", "prior.pt", "--load-prior", "--cfg-cache", "off"]
        )
        self.assertTrue(args.load_prior)


if __name__ == "__main__":
    unittest.main()
