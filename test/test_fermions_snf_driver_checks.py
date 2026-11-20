"""Argument checks, defaults, checkpoints and flow loading of main/main_snf_fermions.py (review of
20 November): the default layers (smearing_signed from the identity), the |rho| layer refused with
lr >= |rho_init|, residual layers with all conditioning steps zero refused, options a layer kind
ignores warned about.

2D 4 x 4 SU(3), a handful of samples; checks the bookkeeping, not physics.
"""

import contextlib
import importlib.util
import io
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


if __name__ == "__main__":
    unittest.main()
