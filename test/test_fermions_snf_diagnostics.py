"""Bijectivity diagnostics of the SNF layers (`snf_layers.layer_diagnostics`, review of 20 November)
and their use in main/main_snf_fermions.py: diagnostics only, nothing clamped.

- The stout bound 1/(2(D - 1)) is real: on a cold 4D configuration (4 x 2^3) two inputs that
  differ on one link have the same image at rho = 0.2 > 1/6, with a finite log J; below the bound
  the image angle is monotone in the input angle.
- `layer_diagnostics` reports max |rho| (either sign) and the ratio to the bound; for the residual
  kinds a contraction estimate that equals the stout ratio of the equivalent stout map (real
  features, no quadratic term); NaN parameters count as exceeded.
- The driver writes max_abs_coeff / injectivity_ratio per training step and the evaluated flow's
  diagnostics in run_meta.json, with one RuntimeWarning in training and one for the evaluated
  (trained or --load-flow) flow when the bound is exceeded, none below it; an older training.csv
  without the columns is upgraded on resume.
"""

import contextlib
import csv
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from neoqcd.fermions._gauge import FlowPars, create_mask
from neoqcd.fermions.action import QCDAction
from neoqcd.fermions.algebra import expi, random_algebra
from neoqcd.fermions.lattice import Geometry
from neoqcd.fermions.snf_layers import (FermionSNF, layer_diagnostics, make_gauge_layers, stout_rho_bound)

ROOT = Path(__file__).resolve().parents[1]


def flow_pars(D, T, L, protocol, B=1):
    return FlowPars(D=D, T=T, L=L, N=3, mask=create_mask(D, T, L), protocol=list(protocol), batch_size=B,
                    device="cpu")


class StoutBoundTest(unittest.TestCase):
    """The map folds above 1/(2(D - 1)) (the reviewer's construction): only the mask block m = 0
    (direction 0, even sites) is active, the input is cold except one even-site time link
    diag(e^{it}, e^{-it}, 1); the image angle of that link as a function of t."""

    D, T, L = 4, 4, 2

    def image(self, rho, t):
        fp = flow_pars(self.D, self.T, self.L, [0.0])
        layer = make_gauge_layers(fp, "smearing_signed", rho=0.0)[0]
        with torch.no_grad():
            layer.rho_layer[0, 0] = rho
        U = torch.eye(3, dtype=torch.cdouble).expand((1, self.D, self.T) + (self.L,) * 3 + (3, 3)).clone()
        U[0, 0, 0, 0, 0, 0] = torch.diag(torch.tensor([1j * t, -1j * t, 0], dtype=torch.cdouble).exp())
        with torch.no_grad():
            Up, _, logJ = layer(U, fp, None)
        return Up, float(torch.as_tensor(logJ).reshape(-1)[0]), float(torch.angle(Up[0, 0, 0, 0, 0, 0, 0, 0]))

    def test_fold_above_the_bound(self):
        self.assertAlmostEqual(stout_rho_bound(4), 1.0 / 6.0)
        self.assertAlmostEqual(stout_rho_bound(2), 0.5)
        ts = [0.05 * k for k in range(1, 61)]
        below = [self.image(0.15, t)[2] for t in ts]
        self.assertTrue(all(b > a for a, b in zip(below, below[1:])), "monotone below the bound")
        # rho = 0.2: the image angle first decreases, then returns to 0 at some t* in (0.3, 1.5): bisection
        f = lambda t: self.image(0.2, t)[2]  # noqa: E731
        lo, hi = 0.3, 1.5
        self.assertLess(f(lo), 0.0)
        self.assertGreater(f(hi), 0.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < 0.0 else (lo, mid)
        # analytically the image angle is t - 6 rho sin t: the root of t = 1.2 sin t
        self.assertAlmostEqual(lo - 1.2 * math.sin(lo), 0.0, places=12)
        Ua, lja, _ = self.image(0.2, 0.0)
        Ub, ljb, _ = self.image(0.2, lo)
        print(f"\nstout rho = 0.2 > 1/6: inputs differing by t* = {lo:.6f} on one link, max |f(U_a) - f(U_b)| = "
              f"{(Ua - Ub).abs().max().item():.1e}, log J = {lja:.4f}, {ljb:.4f}")
        self.assertLess((Ua - Ub).abs().max().item(), 1e-10)
        self.assertTrue(math.isfinite(lja) and math.isfinite(ljb))


class LayerDiagnosticsTest(unittest.TestCase):

    G = Geometry(2, 4, 4, 3)
    PROTOCOL = [0.19, 0.2, 0.23]

    def snf(self, kind, rho=0.0, residual=None, protocol=None, **kw):
        protocol = self.PROTOCOL if protocol is None else protocol
        fp = flow_pars(2, 4, 4, protocol)
        act = QCDAction(self.G, "kappa", beta=5.0, kappa=0.18)
        layers = make_gauge_layers(fp, kind, rho=rho, residual=residual)
        return FermionSNF(act, protocol, layers, fp, initial_parameter=0.18, **kw), fp

    def test_stout(self):
        for kind in ("smearing", "smearing_signed"):
            with self.subTest(kind=kind):
                snf, fp = self.snf(kind, rho=0.01)
                with torch.no_grad():
                    snf.blocks[1].layer.rho_layer.view(-1)[2] = -0.3  # either sign counts as |rho|
                d = snf.layer_diagnostics()
                self.assertEqual(d["kind"], "stout")
                self.assertEqual(d["rho_bound"], 0.5)
                self.assertAlmostEqual(d["max_abs_coeff"], 0.3)
                self.assertAlmostEqual(d["injectivity_ratio"], 0.6)
                self.assertFalse(d["exceeded"])
                self.assertEqual(len(d["per_step"]), 3)
                self.assertAlmostEqual(d["per_step"][0], 0.02)
                with torch.no_grad():
                    snf.blocks[2].layer.rho_layer.view(-1)[0] = 0.5
                self.assertTrue(snf.layer_diagnostics()["exceeded"])  # at the bound: no guarantee
                with torch.no_grad():
                    snf.blocks[2].layer.rho_layer.view(-1)[0] = float("nan")
                d = snf.layer_diagnostics()
                self.assertTrue(d["exceeded"] and math.isnan(d["injectivity_ratio"]))

    def test_residual_matches_stout(self):
        """Real features, no quadratic term: the residual layer IS the stout map with
        rho = d * c / N, and its contraction estimate is that map's stout ratio."""
        c = 0.3
        res, fp = self.snf("residual", residual=dict(include_imag=False, quadratic=False, coeff_init=c))
        d_i = [d for _, d in res.conditioning]
        diag = res.layer_diagnostics()
        self.assertEqual(diag["kind"], "residual")
        self.assertIsNone(diag["rho_bound"])
        self.assertAlmostEqual(diag["max_abs_coeff"], c)
        expected = [abs(d) * c / 3.0 / stout_rho_bound(2) for d in d_i]
        for a, b in zip(diag["per_step"], expected):
            self.assertAlmostEqual(a, b, places=14)
        U = expi(random_algebra(self.G.link_shape(2), torch.Generator().manual_seed(1)))
        blk = res.blocks[0]
        stout = make_gauge_layers(fp, "smearing_signed", rho=d_i[0] * c / 3.0)[0]
        with torch.no_grad():
            Ur, _, lr = blk.layer(U, fp, None, beta=blk.condition, delta_beta=blk.delta_condition)
            Us, _, ls = stout(U, fp, None)
        self.assertLess((Ur - Us).abs().max().item(), 1e-13)
        # quadratic and imaginary features enlarge the estimate; the identity has ratio 0
        full, _ = self.snf("residual", residual=dict(coeff_init=c))
        with torch.no_grad():
            for b in full.blocks:
                b.layer.residual_flow.static_quadratic.fill_(0.01)
        self.assertGreater(full.layer_diagnostics()["injectivity_ratio"], diag["injectivity_ratio"])
        zero, _ = self.snf("residual", residual=dict(coeff_init=0.0))
        self.assertEqual(zero.layer_diagnostics()["injectivity_ratio"], 0.0)

    def test_hyper_residual_per_step(self):
        """Shared hyper layer: the estimate follows each step's conditioning (c_i, d_i)."""
        snf, _ = self.snf("hyper-residual", residual=dict(coeff_init=0.2))
        d = snf.layer_diagnostics()
        self.assertEqual(len(d["per_step"]), 3)
        steps = [abs(x) for _, x in snf.conditioning]  # d_i = 1/6, 1/6, 1/2 of the total variation
        for a, s in zip(d["per_step"], steps):
            self.assertAlmostEqual(a / s, d["per_step"][0] / steps[0], places=12)  # untrained: coefficients const
        self.assertAlmostEqual(d["injectivity_ratio"], max(d["per_step"]))


def _load_main():
    for p in (ROOT / "main",):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("main_snf_fermions", ROOT / "main" / "main_snf_fermions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LATTICE = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "5.0", "--kappa", "0.18",
           "--tau", "0.5", "--n-steps", "2", "--n-gauge-substeps", "2", "--device", "cpu", "--quiet"]


def _csv_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _bijection_warnings(records):
    return [w for w in records if issubclass(w.category, RuntimeWarning) and "bijection" in str(w.message)]


class DriverDiagnosticsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.SNF = _load_main()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, argv):
        with warnings.catch_warnings(record=True) as records, contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("always")
            self.SNF.main(argv)
        return _bijection_warnings(records)

    def test_columns_warnings_and_load(self):
        base = LATTICE + ["--protocol", "kappa", "--final", "0.19", "--n-step", "2", "--mode", "snf",
                          "--layers", "smearing_signed", "--batch", "2", "--seed", "3", "--eval-samples", "2"]
        over, under = self.tmp / "over", self.tmp / "under"
        # 2D bound 1/2: rho = 0.6 exceeds it (two steps: one warning in training, one at evaluation)
        w = self.run_main(base + ["--rho-init", "0.6", "--lr", "1e-3", "--train-steps", "2", "--out", str(over)])
        self.assertEqual(len(w), 2, [str(x.message) for x in w])
        self.assertIn("training", str(w[0].message))
        self.assertIn("evaluated flow", str(w[1].message))
        rows = _csv_rows(over / "training.csv")
        self.assertAlmostEqual(float(rows[0]["max_abs_coeff"]), 0.6)
        self.assertAlmostEqual(float(rows[0]["injectivity_ratio"]), 1.2)
        meta = json.loads((over / "run_meta.json").read_text())
        diag = meta["run"]["layer_diagnostics"]
        self.assertEqual(diag["rho_bound"], 0.5)
        self.assertTrue(diag["exceeded"])
        self.assertEqual(meta["summary"]["training"]["injectivity_exceeded_steps"], 2)
        # an evaluation of the loaded flow warns once
        w = self.run_main(base + ["--rho-init", "0.6", "--load-flow", str(over / "flow.pt"),
                                  "--out", str(self.tmp / "load")])
        self.assertEqual(len(w), 1)
        self.assertIn("evaluated flow", str(w[0].message))
        # below the bound: no warning, the columns are there
        w = self.run_main(base + ["--rho-init", "0.05", "--lr", "1e-3", "--train-steps", "1", "--out", str(under)])
        self.assertEqual(w, [])
        meta = json.loads((under / "run_meta.json").read_text())
        self.assertFalse(meta["run"]["layer_diagnostics"]["exceeded"])
        self.assertAlmostEqual(meta["run"]["layer_diagnostics"]["injectivity_ratio"], 0.1, places=2)

    def test_old_training_csv_is_upgraded_on_resume(self):
        SNF = self.SNF
        out = self.tmp / "old"
        argv = LATTICE + ["--protocol", "kappa", "--final", "0.19", "--n-step", "2", "--mode", "snf",
                          "--layers", "smearing_signed", "--rho-init", "0.01", "--batch", "2", "--seed", "3",
                          "--train-steps", "2", "--eval-samples", "2", "--checkpoint-every", "1", "--out", str(out)]
        calls = {"n": 0}
        original = SNF.effective_sample_size

        def failing(W):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated wall time")
            return original(W)
        SNF.effective_sample_size = failing
        try:
            with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                SNF.main(argv)
        finally:
            SNF.effective_sample_size = original
        # the file as written by the code before the diagnostics columns
        old = [f for f in SNF.TRAIN_FIELDS if f not in ("max_abs_coeff", "injectivity_ratio")]
        rows = _csv_rows(out / "training.csv")
        with open(out / "training.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(old)
            writer.writerows([[r[f] for f in old] for r in rows])
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            SNF.main([a for a in argv if a != "--quiet"] + ["--resume"])
        self.assertIn("header upgraded", log.getvalue())
        rows = _csv_rows(out / "training.csv")
        self.assertEqual(list(rows[0]), list(SNF.TRAIN_FIELDS))
        self.assertEqual([r["max_abs_coeff"] == "" for r in rows], [True, False])


if __name__ == "__main__":
    unittest.main()
