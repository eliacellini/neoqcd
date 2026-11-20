"""Argument checks, defaults, checkpoints and flow loading of main/main_snf_fermions.py (review of
20 November): the default layers (smearing_signed from the identity), the |rho| layer refused with
lr >= |rho_init|, residual layers with all conditioning steps zero refused, options a layer kind
ignores warned about; the flow/pool checkpoint pair (a kill between the two writes resumes from
flow_prev.pt, an inconsistent pair is refused), --load-flow refusing another protocol unless
--load-flow-force, the training settings in the resume key (older checkpoints accepted with a
note), skipped optimizer steps on a non-finite gradient reported and counted.

2D 4 x 4 SU(3), a handful of samples; checks the bookkeeping, not physics.
"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "main" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SNF = _load("main_snf_fermions")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from neoqcd.fermions.io import read_observables  # noqa: E402

LATTICE = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "5.0", "--kappa", "0.18",
           "--tau", "0.5", "--n-steps", "2", "--n-gauge-substeps", "2", "--device", "cpu", "--quiet"]
KAPPA = LATTICE + ["--protocol", "kappa", "--final", "0.19", "--n-step", "2"]


def parse(argv):
    """parse_args with the RuntimeWarnings of the options a layer kind ignores recorded."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        args = SNF.parse_args(argv)
    return args, [str(w.message) for w in records if issubclass(w.category, RuntimeWarning) and "ignores" in
                  str(w.message)]


class DefaultsAndChecksTest(unittest.TestCase):

    def test_default_layers_are_a_trainable_identity(self):
        args, _ = parse(KAPPA + ["--mode", "snf", "--train-steps", "2"])
        self.assertEqual(args.layers, "smearing_signed")
        self.assertEqual(args.rho_init, 0.0)
        args, _ = parse(KAPPA)
        self.assertEqual(args.layers, "none")

    def test_smearing_needs_lr_below_rho_init(self):
        snf = KAPPA + ["--mode", "snf", "--layers", "smearing", "--train-steps", "1"]
        for extra in ([], ["--rho-init", "1e-3"], ["--rho-init", "1e-3", "--lr", "2e-3"],
                      ["--rho-init=-5e-4", "--lr", "1e-3"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as cm:
                SNF.parse_args(snf + extra)
            if extra:
                self.assertIn("--lr", str(cm.exception))
        args, _ = parse(snf + ["--rho-init", "1e-2", "--lr", "1e-3"])
        self.assertEqual(args.layers, "smearing")
        # evaluation only (no training): lr is irrelevant
        args, _ = parse(KAPPA + ["--mode", "snf", "--layers", "smearing", "--rho-init", "1e-3"])
        self.assertEqual(args.train_steps, 0)

    def test_residual_zero_conditioning_step_refused(self):
        for protocol in (["--final", "0.19", "--n-step", "1", "--residual-delta", "step"],
                         ["--final", "0.18", "--n-step", "2"],  # constant protocol, fraction
                         ["--final", "0.18", "--n-step", "2", "--residual-delta", "step"]):
            for layers in ("residual", "hyper-residual"):
                argv = LATTICE + ["--protocol", "kappa", "--mode", "snf", "--layers", layers] + protocol
                with self.subTest(protocol=protocol, layers=layers):
                    for train in (["--train-steps", "1"], []):  # also the silent NE-MCMC labelled snf
                        with self.assertRaisesRegex(SystemExit, "identity"):
                            SNF.parse_args(argv + train)
        # a zero step inside a moving protocol is legal (that layer is the identity)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "protocol.txt"
            f.write_text("0.19\n0.19\n0.2\n")
            args, _ = parse([a for a in LATTICE] + ["--protocol", "kappa", "--protocol-file", str(f), "--mode", "snf",
                                                    "--layers", "residual"])
            self.assertEqual(args.n_step, 3)

    def test_ignored_options_warn_once(self):
        _, w = parse(KAPPA + ["--mode", "snf", "--layers", "residual", "--rho-init", "0.1"])
        self.assertEqual(len(w), 1)
        self.assertIn("--rho-init", w[0])
        _, w = parse(KAPPA + ["--mode", "snf", "--layers", "smearing_signed", "--residual-n-pt", "0",
                              "--residual-condition", "time", "--hyper-depth", "2"])
        self.assertEqual(len(w), 1)
        for name in ("--residual-n-pt", "--residual-condition", "--hyper-depth"):
            self.assertIn(name, w[0])
        _, w = parse(KAPPA + ["--rho-init", "0.0"])  # nemcmc: no layers
        self.assertEqual(len(w), 1)
        _, w = parse(KAPPA + ["--mode", "snf", "--rho-init", "0.01"])  # used: no warning
        self.assertEqual(w, [])
        _, w = parse(KAPPA + ["--mode", "snf", "--layers", "residual", "--residual-n-pt", "0"])
        self.assertEqual(w, [])
        with self.assertRaises(SystemExit):  # a non-default residual option with stout layers still exits
            SNF.parse_args(KAPPA + ["--mode", "snf", "--residual-n-pt", "1"])

    def test_default_snf_run_trains_from_the_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                SNF.main(KAPPA + ["--mode", "snf", "--batch", "2", "--train-steps", "2", "--eval-samples", "2",
                                  "--seed", "4", "--out", str(out)])
            t = read_observables(out / "training.csv")
            self.assertEqual(t["param_abs_mean"][0], 0.0)
            self.assertEqual(t["logJ_mean"][0], 0.0)
            self.assertGreater(t["grad_norm"][0], 0.0)
            self.assertGreater(t["param_abs_mean"][1], 0.0)



def quiet_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = SNF.main(argv)
    return result, out.getvalue() + err.getvalue()


class CheckpointPairTest(unittest.TestCase):
    """flow.pt is written (atomically, previous generation kept as flow_prev.pt) before the pool,
    whose sidecar commits the pair; on resume the flow file of the pool's step is used and an
    inconsistent pair is refused."""

    BASE = KAPPA + ["--mode", "snf", "--batch", "2", "--seed", "11", "--train-steps", "3", "--eval-samples", "4",
                    "--checkpoint-every", "1", "--therm", "1"]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _killed_at_pool_write(self, out, n):
        """Run until the n-th pool save_state, which fails (the flow of that checkpoint is written)."""
        calls = {"n": 0}
        original = SNF.save_state

        def failing(*a, **kw):
            calls["n"] += 1
            if calls["n"] == n:
                raise KeyboardInterrupt("killed between the flow and the pool writes")
            return original(*a, **kw)
        SNF.save_state = failing
        try:
            with self.assertRaises(KeyboardInterrupt):
                quiet_main(self.BASE + ["--out", str(out)])
        finally:
            SNF.save_state = original

    def test_kill_between_flow_and_pool_resumes_identically(self):
        full, part = self.tmp / "full", self.tmp / "part"
        quiet_main(self.BASE + ["--out", str(full)])
        self._killed_at_pool_write(part, 3)  # checkpoints: train 0, train 1, (killed) train 2
        self.assertEqual(torch.load(part / "flow.pt")["train_step"], 2)
        self.assertEqual(torch.load(part / "flow_prev.pt")["train_step"], 1)
        self.assertEqual(json.loads((part / "pool.json").read_text())["extra"]["progress"], 1)
        _, log = quiet_main([a for a in self.BASE if a != "--quiet"] + ["--out", str(part), "--resume"])
        self.assertIn("flow_prev.pt", log)
        for name, field in (("training.csv", "loss"), ("training.csv", "param_abs_mean"), ("samples.csv", "work")):
            np.testing.assert_array_equal(read_observables(full / name)[field], read_observables(part / name)[field])

    def test_inconsistent_pair_refused(self):
        part = self.tmp / "part"
        self._killed_at_pool_write(part, 3)
        # the pair the older write order could leave (pool newer than every flow file)
        for name in ("flow.pt", "flow_prev.pt"):
            payload = torch.load(part / name)
            payload["train_step"] = 0
            torch.save(payload, part / name)
        with self.assertRaisesRegex(SystemExit, "matches the pool checkpoint"):
            quiet_main(self.BASE + ["--out", str(part), "--resume"])

    def test_inconsistent_single_flow_refused(self):
        """Without flow_prev.pt (as a run of the older code leaves it): a flow.pt of another step than
        the pool is refused, not loaded silently."""
        part = self.tmp / "part"
        self._killed_at_pool_write(part, 3)
        progress = json.loads((part / "pool.json").read_text())["extra"]["progress"]
        payload = torch.load(part / "flow.pt")
        payload["train_step"] = progress + 1
        torch.save(payload, part / "flow.pt")
        if (part / "flow_prev.pt").exists():
            (part / "flow_prev.pt").unlink()
        with self.assertRaisesRegex(SystemExit, "matches the pool checkpoint"):
            quiet_main(self.BASE + ["--out", str(part), "--resume"])


class LoadFlowProtocolTest(unittest.TestCase):
    """--load-flow compares the protocol parameter, endpoints, values and conditioning."""

    def test_protocol_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            common = LATTICE + ["--mode", "snf", "--batch", "2", "--seed", "3", "--eval-samples", "2"]
            for layers in ("smearing_signed", "residual"):
                trained = tmp / f"trained_{layers}"
                quiet_main(common + ["--layers", layers, "--protocol", "kappa", "--final", "0.19", "--n-step", "2",
                                     "--train-steps", "1", "--out", str(trained)])
                flow = str(trained / "flow.pt")
                same = ["--layers", layers, "--protocol", "kappa", "--final", "0.19", "--n-step", "2"]
                quiet_main(common + same + ["--load-flow", flow, "--out", str(tmp / f"same_{layers}")])
                for k, other in enumerate((["--protocol", "beta", "--final", "5.2", "--n-step", "2"],
                                           ["--protocol", "kappa", "--final", "0.2", "--n-step", "2"],
                                           ["--protocol", "kappa", "--initial", "0.19", "--final", "0.18",
                                            "--n-step", "2"])):
                    argv = common + ["--layers", layers] + other + ["--load-flow", flow]
                    with self.subTest(layers=layers, other=other):
                        with self.assertRaisesRegex(SystemExit, "another protocol") as cm:
                            quiet_main(argv + ["--out", str(tmp / f"bad_{layers}_{k}")])
                        self.assertIn("--load-flow-force", str(cm.exception))
                        if k == 0:
                            self.assertIn("parameter kappa (flow) vs beta (this run)", str(cm.exception))
                        _, log = quiet_main([a for a in argv if a != "--quiet"] + ["--load-flow-force", "--out",
                                                                                  str(tmp / f"force_{layers}_{k}")])
                        self.assertIn("WARNING: --load-flow-force", log)
                        self.assertTrue((tmp / f"force_{layers}_{k}" / "samples.csv").exists())


    def test_shared_flow_at_another_n_step(self):
        """A shared hyper-residual flow (one layer for every step) loads at another --n-step (with
        --load-flow-force: the protocol differs) and gives the same coefficients at the same
        conditioning value; per-step layers at another --n-step exit with a clear message."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            common = LATTICE + ["--mode", "snf", "--batch", "2", "--seed", "3", "--eval-samples", "2",
                                "--protocol", "kappa", "--final", "0.19"]
            for layers in ("hyper-residual", "residual"):
                trained = tmp / layers
                quiet_main(common + ["--layers", layers, "--n-step", "2", "--train-steps", "2", "--out", str(trained)])
                argv = common + ["--layers", layers, "--n-step", "3", "--load-flow", str(trained / "flow.pt"),
                                 "--load-flow-force", "--out", str(tmp / f"{layers}_n3")]
                if layers == "residual":
                    with self.assertRaisesRegex(SystemExit, "does not fit this run's 3 protocol steps"):
                        quiet_main(argv)
                    continue
                quiet_main(argv)
                saved = torch.load(trained / "flow.pt")["state_dict"]
                args = SNF.parse_args(common + ["--layers", layers, "--n-step", "3"])
                flow = SNF.FermionSNF(SNF.QCDAction(SNF.Geometry(2, 4, 4, 3), "kappa", beta=5.0, kappa=0.18),
                                      args.protocol_values,
                                      SNF.make_gauge_layers(SNF.FlowPars(D=2, T=4, L=4, N=3,
                                                                         mask=SNF.create_mask(2, 4, 4),
                                                                         protocol=args.protocol_values, batch_size=2,
                                                                         device="cpu"),
                                                            layers, residual={k: v for k, v in args.residual.items()
                                                                              if k != "parameterization"}),
                                      None, initial_parameter=0.18)
                SNF.load_flow_state(flow, saved, True, "test")
                for k, v in flow.state_dict().items():
                    rest = k.split(".", 2)[2]
                    self.assertTrue(torch.equal(v, saved["blocks.0." + rest]), k)
                self.assertIs(flow.blocks[0].layer, flow.blocks[2].layer)


class ResumeKeyTest(unittest.TestCase):
    """--n-between, --train-steps, --lr, --lr-warmup, --grad-clip, --rho-init are in the resume key;
    a checkpoint written before them is accepted with a note."""

    def test_training_settings_in_key(self):
        base = KAPPA + ["--mode", "snf", "--batch", "2", "--seed", "3", "--train-steps", "2", "--eval-samples", "2",
                        "--rho-init", "0.01", "--checkpoint-every", "1"]
        key = SNF.physics_key(SNF.parse_args(base))
        for name, value in (("n_between", 1), ("train_steps", 2), ("lr", 1e-3), ("lr_warmup", 0), ("grad_clip", 0.0),
                            ("rho_init", 0.01)):
            self.assertEqual(key[name], value, name)
        self.assertNotIn("rho_init", SNF.physics_key(SNF.parse_args(KAPPA + ["--mode", "snf", "--layers", "residual"])))
        self.assertNotIn("lr", SNF.physics_key(SNF.parse_args(KAPPA)))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            quiet_main(base + ["--out", str(out)])
            for changed in (["--lr", "2e-3"], ["--n-between", "2"], ["--rho-init", "0.02"], ["--grad-clip", "1"],
                            ["--lr-warmup", "3"], ["--train-steps", "3"]):
                name = changed[0][2:].replace("-", "_")
                with self.subTest(changed=changed), self.assertRaisesRegex(SystemExit, name):
                    quiet_main(base + changed + ["--out", str(out), "--resume"])
            # a checkpoint of the older code: the new entries absent -> accepted, with a note
            side = json.loads((out / "pool.json").read_text())
            for name in SNF.ADDED_RESUME_KEYS:
                side["extra"]["physics"].pop(name, None)
            (out / "pool.json").write_text(json.dumps(side))
            _, log = quiet_main([a for a in base if a != "--quiet"] + ["--lr", "2e-3", "--out", str(out), "--resume"])
            self.assertIn("predates the resume-key entries", log)


class NonFiniteGradientTest(unittest.TestCase):
    """A non-finite gradient skips the optimizer step with a printed line and is counted in
    run_meta.json (summary.training.nonfinite_grad_steps)."""

    def test_skipped_steps_reported(self):
        original = SNF.FermionSNF.__dict__["loss"]  # the staticmethod object itself

        def nan_loss(W):
            return W.mean() * float("nan")
        SNF.FermionSNF.loss = staticmethod(nan_loss)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "run"
                _, log = quiet_main(KAPPA + ["--mode", "snf", "--batch", "2", "--seed", "3", "--train-steps", "3",
                                             "--rho-init", "0.01", "--eval-samples", "2", "--out", str(out)])
                meta = json.loads((out / "run_meta.json").read_text())
                t = read_observables(out / "training.csv")
        finally:
            SNF.FermionSNF.loss = original
        self.assertEqual(log.count("is not finite, optimizer step skipped"), 3)
        self.assertEqual(meta["summary"]["training"]["nonfinite_grad_steps"], 3)
        np.testing.assert_array_equal(t["param_abs_mean"], [0.01, 0.01, 0.01])  # never stepped


if __name__ == "__main__":
    unittest.main()
