"""End-to-end tests of main/main_hmc.py on 2D 4 x 4 SU(3) (a few seconds on a laptop CPU)."""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("main_hmc", ROOT / "main" / "main_hmc.py")
MAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAIN)

from neoqcd.fermions.io import load_state, read_observables  # noqa: E402

COMMON = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "2.0", "--batch", "2",
          "--tau", "0.5", "--n-steps", "3", "--n-gauge-substeps", "2", "--device", "cpu",
          "--seed", "7", "--start", "hot", "--quiet"]
WILSON = COMMON + ["--fermions", "wilson", "--kappa", "0.12"]
DETERMINISTIC = ("traj", "replica", "therm", "plaquette", "dH", "accepted", "exp_mdH", "logdet",
                 "solves", "cg_iter", "cg_iter_replica")


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class MainHMCTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_wilson_outputs_and_bitwise_resume(self):
        full = self.tmp / "full"
        res = MAIN.main(WILSON + ["--therm", "2", "--n-traj", "8", "--checkpoint-every", "4",
                                  "--exact-logdet", "--meas-every", "2", "--out", str(full)])
        for name in ("observables.csv", "observables.npz", "run_meta.json", "checkpoint.pt", "checkpoint.json"):
            self.assertTrue((full / name).exists(), name)
        rows = _rows(full / "observables.csv")
        self.assertEqual(len(rows), 10 * 2)
        self.assertEqual(list(rows[0].keys()), list(MAIN.OBS_FIELDS))
        self.assertEqual(sum(int(r["therm"]) for r in rows), 2 * 2)
        cols = read_observables(full / "observables.csv")
        self.assertTrue(np.all(np.isfinite(cols["plaquette"])))
        self.assertTrue(np.all((cols["plaquette"] > -1) & (cols["plaquette"] < 1)))
        np.testing.assert_allclose(cols["exp_mdH"], np.exp(-cols["dH"]), rtol=1e-12)
        self.assertTrue(np.all(cols["solves"] > 0) and np.all(cols["cg_iter"] > 0))
        # logdet every second trajectory only
        has_logdet = ~np.isnan(cols["logdet"])
        np.testing.assert_array_equal(has_logdet, cols["traj"] % 2 == 0)
        npz = np.load(full / "observables.npz")
        np.testing.assert_array_equal(npz["plaquette"], cols["plaquette"])

        meta = json.loads((full / "run_meta.json").read_text())
        self.assertTrue(meta["finished"])
        self.assertEqual(meta["trajectories_done"], 10)
        s = meta["summary"]
        self.assertEqual(s["n_traj_measured"], 8)
        self.assertEqual(s["plaquette"]["status"], "gamma")
        self.assertGreater(s["plaquette"]["error"], 0.0)
        measured = cols["therm"] == 0
        self.assertAlmostEqual(s["plaquette"]["value"], cols["plaquette"][measured].mean(), places=12)
        self.assertAlmostEqual(s["acceptance"]["value"], cols["accepted"][measured].mean(), places=12)
        self.assertGreater(s["acceptance"]["value"], 0.5)
        self.assertLess(np.abs(cols["dH"]).max(), 1.0)
        for key in ("seconds_per_traj", "seconds_per_solve", "cg_iter_per_solve", "solves_per_traj"):
            self.assertGreater(s[key], 0.0, key)
        self.assertIn("logdet", s)
        self.assertEqual(res["summary"]["n_traj_measured"], 8)
        self.assertEqual(meta["parameters"]["kappa"], 0.12)

        U_full, state_meta, _ = load_state(full / "checkpoint")
        self.assertEqual(state_meta["trajectory"], 10)
        self.assertEqual(state_meta["n_flavours"], 2)
        self.assertEqual(state_meta["geometry"], {"D": 2, "T": 4, "L": 4, "N": 3})

        # interrupted run: stop after 5 trajectories, then resume to 10
        part = self.tmp / "part"
        MAIN.main(WILSON + ["--therm", "2", "--n-traj", "3", "--checkpoint-every", "4",
                            "--exact-logdet", "--meas-every", "2", "--out", str(part)])
        self.assertEqual(load_state(part / "checkpoint")[1]["trajectory"], 5)
        with self.assertRaises(SystemExit):  # a new run would clobber it
            MAIN.main(WILSON + ["--n-traj", "3", "--out", str(part)])
        with self.assertRaises(SystemExit):  # different physics
            MAIN.main(COMMON + ["--fermions", "wilson", "--kappa", "0.11", "--therm", "2", "--n-traj", "8",
                                "--out", str(part), "--resume"])
        MAIN.main(WILSON + ["--therm", "2", "--n-traj", "8", "--checkpoint-every", "4",
                            "--exact-logdet", "--meas-every", "2", "--out", str(part), "--resume"])
        a, b = _rows(full / "observables.csv"), _rows(part / "observables.csv")
        self.assertEqual(len(a), len(b))
        for ra, rb in zip(a, b):
            for key in DETERMINISTIC:
                self.assertEqual(ra[key], rb[key], (ra["traj"], key))  # text of repr: bitwise
        U_part = load_state(part / "checkpoint")[0]
        self.assertTrue(torch.equal(U_full, U_part))

    def test_resume_drops_rows_past_checkpoint(self):
        out = self.tmp / "trunc"
        MAIN.main(WILSON + ["--n-traj", "4", "--checkpoint-every", "2", "--out", str(out)])
        # simulate a crash after trajectory 4 with the checkpoint at 2: rewind the checkpoint
        ref = _rows(out / "observables.csv")
        mid = self.tmp / "mid"
        MAIN.main(WILSON + ["--n-traj", "2", "--out", str(mid)])
        for suffix in (".pt", ".json"):
            (out / ("checkpoint" + suffix)).write_bytes((mid / ("checkpoint" + suffix)).read_bytes())
        MAIN.main(WILSON + ["--n-traj", "4", "--checkpoint-every", "2", "--out", str(out), "--resume"])
        rows = _rows(out / "observables.csv")
        self.assertEqual([r["traj"] for r in rows], [r["traj"] for r in ref])
        for ra, rb in zip(ref, rows):
            self.assertEqual(ra["plaquette"], rb["plaquette"])

    def test_quenched_and_staggered_run(self):
        for extra, nf in ((["--fermions", "none"], 0), (["--fermions", "staggered", "--mass", "0.3", "--exact-logdet"], 2)):
            out = self.tmp / extra[1]
            MAIN.main(COMMON + extra + ["--n-traj", "5", "--out", str(out)])
            meta = json.loads((out / "run_meta.json").read_text())
            self.assertEqual(meta["summary"]["n_traj_measured"], 5)
            self.assertEqual(meta["run"]["n_flavours"], nf)
            cols = read_observables(out / "observables.csv")
            self.assertEqual(cols["traj"].size, 5 * 2)
            self.assertTrue(np.all(np.isfinite(cols["dH"])))
            if nf == 0:
                self.assertTrue(np.all(cols["solves"] == 0))
                self.assertIsNone(meta["summary"]["seconds_per_solve"])
            else:
                self.assertTrue(np.all(cols["solves"] > 0))
                self.assertTrue(np.all(np.isfinite(cols["logdet"])))

    def test_argument_checks(self):
        for extra in (["--colour-product", "broadcast"], ["--check-every", "5"], ["--fast", "--inner-tol", "1e-5"],
                      ["--fast", "--max-outer", "5"], ["--inner-tol", "1e-5"], ["--precision", "mixed"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                MAIN.parse_args(WILSON + extra)
        with self.assertRaises(SystemExit):  # Wilson only
            MAIN.parse_args(COMMON + ["--fermions", "staggered", "--mass", "0.3", "--on-unconverged", "warn"])
        args = MAIN.parse_args(WILSON + ["--fast", "--precision", "mixed", "--inner-tol", "1e-5", "--max-outer", "7"])
        self.assertEqual((args.inner_tol, args.max_outer, args.check_every, args.colour_product, args.on_unconverged),
                         (1e-5, 7, 10, "auto", "raise"))
        self.assertIsNone(MAIN.parse_args(COMMON + ["--fermions", "staggered", "--mass", "0.3"]).on_unconverged)
        with self.assertRaises(SystemExit):
            MAIN.parse_args(COMMON + ["--fermions", "wilson"])  # no kappa
        with self.assertRaises(SystemExit):
            MAIN.parse_args(COMMON + ["--fermions", "staggered", "--kappa", "0.1"])
        with self.assertRaises(SystemExit):  # 8^4 Wilson SU(3) is far too large for the dense determinant
            MAIN.main(["--D", "4", "--T", "8", "--L", "8", "--kappa", "0.153", "--exact-logdet",
                       "--device", "cpu", "--quiet", "--out", str(self.tmp / "big")])

    def test_max_residual_and_convergence_policy(self):
        """max_residual column and --on-unconverged (default raise) for reference and --fast."""
        for extra in ([], ["--fast"], ["--fast", "--precision", "mixed"]):
            with self.subTest(extra=extra):
                out = self.tmp / ("ok" + "_".join(extra))
                MAIN.main(WILSON + extra + ["--n-traj", "2", "--solver-tol", "1e-10", "--out", str(out)])
                cols = read_observables(out / "observables.csv")
                self.assertTrue(np.all(cols["max_residual"] > 0) and np.all(cols["max_residual"] < 1e-9))
                meta = json.loads((out / "run_meta.json").read_text())
                self.assertEqual(meta["run"]["on_unconverged"], "raise")
                self.assertEqual(meta["parameters"]["on_unconverged"], "raise")
                with self.assertRaisesRegex(RuntimeError, "not converged"):
                    MAIN.main(WILSON + extra + ["--n-traj", "2", "--max-iter", "2", "--out", str(out) + "_raise"])
                out = self.tmp / ("warn" + "_".join(extra))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    MAIN.main(WILSON + extra + ["--n-traj", "2", "--max-iter", "2", "--on-unconverged", "warn",
                                                "--out", str(out)])
                cols = read_observables(out / "observables.csv")
                self.assertTrue(np.all(cols["max_residual"] > 1e-6))
        # quenched: no solves, empty column
        out = self.tmp / "quenched"
        MAIN.main(COMMON + ["--fermions", "none", "--n-traj", "2", "--out", str(out)])
        self.assertTrue(np.all(np.isnan(read_observables(out / "observables.csv")["max_residual"])))

    def test_mixed_cost_columns(self):
        """--precision mixed: cg_iter counts complex64 loop + complex128 outer applications."""
        out = self.tmp / "mixed"
        MAIN.main(WILSON + ["--fast", "--precision", "mixed", "--n-traj", "2", "--inner-tol", "1e-4",
                            "--max-outer", "20", "--out", str(out)])
        meta = json.loads((out / "run_meta.json").read_text())
        self.assertIn("complex64 inner loop applications", meta["run"]["cg_iter_definition"])
        cols = read_observables(out / "observables.csv")
        # the real cost exceeds the per-replica inner counts by at least one outer application per solve
        self.assertTrue(np.all(cols["cg_iter"] >= cols["cg_iter_replica"] + cols["solves"]))
        ref = self.tmp / "ref"
        MAIN.main(WILSON + ["--n-traj", "2", "--out", str(ref)])
        meta = json.loads((ref / "run_meta.json").read_text())
        self.assertEqual(meta["run"]["cg_iter_definition"], "sum over solves of the largest per-replica CG iteration count")

    @staticmethod
    def _ar1(seed=33, phi=0.5, n=50, B=16):
        """AR(1) series (tau_int ~ 1.5-3) that makes Obs.gamma_method raise in _compute_drho
        (odd w_max = 25: the mirrored rho slice is one element short)."""
        rng = np.random.default_rng(seed)
        x = np.zeros((n, B))
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal(size=B)
        return x

    def test_gamma_summary_falls_back_when_gamma_method_raises(self):
        x = self._ar1()
        s = MAIN.gamma_summary(x, S=2.0)
        self.assertTrue(s["status"].startswith("gamma_failed: "), s["status"])
        self.assertAlmostEqual(s["value"], float(x.mean()))
        self.assertEqual(s["error_method"], "replica_means")
        self.assertAlmostEqual(s["error"], float(x.mean(axis=0).std(ddof=1) / 4.0))
        self.assertEqual(sorted(s["binned_errors"], key=int), ["1", "2", "4", "8", "16"])
        # forced failure through a monkeypatched Obs method (independent of the obs.py bug)
        from neoqcd import obs as obs_mod
        orig = obs_mod.Obs.gamma_method
        try:
            obs_mod.Obs.gamma_method = lambda self, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
            s = MAIN.gamma_summary(self._ar1(seed=0), S=2.0)
        finally:
            obs_mod.Obs.gamma_method = orig
        self.assertEqual(s["status"], "gamma_failed: RuntimeError: boom")
        self.assertIsNotNone(s["error"])

    def test_summary_failure_does_not_kill_checkpoint_and_stale_meta_resumes(self):
        out = self.tmp / "run"
        orig = MAIN.summarize
        try:
            MAIN.summarize = lambda *a, **k: (_ for _ in ()).throw(ValueError("broken stats"))
            MAIN.main(WILSON + ["--n-traj", "4", "--checkpoint-every", "2", "--out", str(out)])
        finally:
            MAIN.summarize = orig
        meta = json.loads((out / "run_meta.json").read_text())
        self.assertTrue(meta["summary"]["status"].startswith("summary_failed: ValueError"))
        self.assertEqual(load_state(out / "checkpoint")[1]["trajectory"], 4)
        # stale run_meta.json (older than the checkpoint): resume reads only checkpoint.* + CSV
        meta["trajectories_done"] = 0
        (out / "run_meta.json").write_text(json.dumps(meta))
        MAIN.main(WILSON + ["--n-traj", "6", "--checkpoint-every", "2", "--out", str(out), "--resume"])
        meta = json.loads((out / "run_meta.json").read_text())
        self.assertEqual(meta["trajectories_done"], 6)

    def test_fast_staggered_and_hasenbusch_match_reference(self):
        """--fast staggered and --kappa-heavy (both layouts): --fast gives the reference dH."""
        tight = ["--n-traj", "3", "--no-reuse-guess", "--solver-tol", "1e-13", "--action-tol", "1e-13"]
        cases = {
            "staggered": COMMON + ["--fermions", "staggered", "--mass", "0.3"],
            "hb_nested": WILSON + ["--kappa-heavy", "0.08", "--n-heavy-substeps", "2"],
            "hb_shared": WILSON + ["--kappa-heavy", "0.08", "--hasenbusch-layout", "shared", "--n-ratio-substeps", "2"],
        }
        for name, base in cases.items():
            with self.subTest(case=name):
                cols = {}
                for fast in ([], ["--fast"]):
                    out = self.tmp / (name + "_".join(fast))
                    MAIN.main(base + tight + fast + ["--out", str(out)])
                    cols[bool(fast)] = read_observables(out / "observables.csv")
                    meta = json.loads((out / "run_meta.json").read_text())
                    self.assertEqual(meta["run"]["fermion_operator"], "half_lattice_fast" if fast else "reference")
                    if name.startswith("hb"):
                        hb = meta["run"]["hasenbusch"]
                        self.assertEqual(hb["kappa_heavy"], 0.08)
                        self.assertEqual(hb["layout"], name[3:])
                        self.assertEqual(len(hb["levels"]), 3 if name == "hb_nested" else 2)
                        self.assertEqual(hb["levels"][0][0][0], "FastRatioWilsonTerm" if fast else "RatioWilsonTerm")
                        self.assertEqual(meta["parameters"]["kappa_heavy"], 0.08)
                    self.assertTrue(np.all(cols[bool(fast)]["solves"] > 0))
                    self.assertTrue(np.all(cols[bool(fast)]["max_residual"] < 1e-11))
                self.assertLess(np.abs(cols[True]["dH"] - cols[False]["dH"]).max(), 1e-8)
                self.assertTrue(np.array_equal(cols[True]["accepted"], cols[False]["accepted"]))
                self.assertLess(np.abs(cols[True]["plaquette"] - cols[False]["plaquette"]).max(), 1e-10)

    def test_fused_kernel_matches_fast(self):
        """--kernel cpu (host build of the fused kernel; skipped without a C++ compiler) and
        --kernel loop reproduce the --fast run: dH to 1e-8, identical accept pattern; also with
        --precision mixed."""
        from neoqcd.fermions.fused import backend_status
        cpu_ok, cpu_why = backend_status("cpu")
        kernels = ["loop"] + (["cpu"] if cpu_ok else [])
        tight = ["--n-traj", "4", "--therm", "1", "--no-reuse-guess", "--solver-tol", "1e-13", "--action-tol", "1e-13"]
        for precision in ("double", "mixed"):
            prec = ["--precision", precision]
            ref = self.tmp / f"fast_{precision}"
            MAIN.main(WILSON + tight + ["--fast"] + prec + ["--out", str(ref)])
            want = read_observables(ref / "observables.csv")
            meta = json.loads((ref / "run_meta.json").read_text())
            self.assertIsNone(meta["run"]["fused_kernel"])
            self.assertEqual(meta["run"]["fermion_operator"], "half_lattice_fast")
            for kernel in kernels:
                with self.subTest(kernel=kernel, precision=precision):
                    out = self.tmp / f"{kernel}_{precision}"
                    MAIN.main(WILSON + tight + ["--fast", "--kernel", kernel] + prec + ["--out", str(out)])
                    got = read_observables(out / "observables.csv")
                    self.assertLess(np.abs(got["dH"] - want["dH"]).max(), 1e-8)
                    self.assertTrue(np.array_equal(got["accepted"], want["accepted"]))
                    self.assertLess(np.abs(got["plaquette"] - want["plaquette"]).max(), 1e-10)
                    meta = json.loads((out / "run_meta.json").read_text())
                    info = meta["run"]["fused_kernel"]
                    self.assertEqual(meta["run"]["fermion_operator"], "half_lattice_fused")
                    self.assertEqual((info["requested"], info["resolved"], info["link_layout"]),
                                     (kernel, kernel, "soa"))
                    self.assertEqual(info["compiled"], kernel == "cpu")
                    self.assertIsInstance(info["backend_status"], str)
                    if kernel == "cpu":
                        self.assertIn("cpu library", info["backend_status"])
                    self.assertEqual(meta["parameters"]["kernel"], kernel)
        if not cpu_ok:
            self.skipTest(f"--kernel cpu not tested: host build of the fused kernel unavailable: {cpu_why}")

    def test_fused_kernel_arguments_auto_and_resume_key(self):
        for extra in (["--kernel", "cpu"],  # needs --fast
                      ["--fast", "--link-layout", "site"],  # needs --kernel
                      ["--fast", "--kernel", "cpu", "--kappa-heavy", "0.08"],
                      ["--fast", "--kernel", "triton"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                MAIN.parse_args(WILSON + extra)
        with self.assertRaisesRegex(SystemExit, "wilson"):
            MAIN.parse_args(COMMON + ["--fermions", "staggered", "--mass", "0.3", "--fast", "--kernel", "loop"])
        args = MAIN.parse_args(WILSON + ["--fast", "--kernel", "loop"])
        self.assertEqual(args.link_layout, "soa")
        self.assertEqual(MAIN.parse_args(WILSON + ["--fast", "--kernel", "cpu", "--link-layout", "site"]).link_layout,
                         "site")
        if not torch.cuda.is_available():  # an explicit CUDA kernel on the CPU is an error, no fallback
            with self.assertRaisesRegex(SystemExit, "unavailable"):
                MAIN.main(WILSON + ["--fast", "--kernel", "cuda", "--n-traj", "1", "--out", str(self.tmp / "c")])
            # auto on the CPU: the --fast operator (matmul), recorded
            out = self.tmp / "auto"
            MAIN.main(WILSON + ["--fast", "--kernel", "auto", "--n-traj", "1", "--out", str(out)])
            info = json.loads((out / "run_meta.json").read_text())["run"]["fused_kernel"]
            self.assertEqual((info["requested"], info["resolved"], info["compiled"]), ("auto", "fast", False))
            self.assertEqual(info["colour_product_of_fallback"], "matmul")
            self.assertEqual(info["backend_status"], "no compiled backend requested")
        # resume across kernels (or layouts) is refused; the same kernel resumes
        out = self.tmp / "loop"
        MAIN.main(WILSON + ["--fast", "--kernel", "loop", "--n-traj", "2", "--out", str(out)])
        for extra in ([], ["--kernel", "auto"], ["--kernel", "loop", "--link-layout", "site"]):
            with self.subTest(resume_with=extra), self.assertRaisesRegex(SystemExit, "differ"):
                MAIN.main(WILSON + ["--fast"] + extra + ["--n-traj", "3", "--resume", "--out", str(out)])
        MAIN.main(WILSON + ["--fast", "--kernel", "loop", "--n-traj", "3", "--resume", "--out", str(out)])
        self.assertEqual(json.loads((out / "run_meta.json").read_text())["trajectories_done"], 3)
        # a --kernel off checkpoint does not resume with a fused kernel
        out = self.tmp / "off"
        MAIN.main(WILSON + ["--fast", "--n-traj", "2", "--out", str(out)])
        with self.assertRaisesRegex(SystemExit, "differ"):
            MAIN.main(WILSON + ["--fast", "--kernel", "loop", "--n-traj", "3", "--resume", "--out", str(out)])

    def test_hasenbusch_arguments_and_resume_key(self):
        for extra in (["--hasenbusch-layout", "shared"], ["--n-heavy-substeps", "2"], ["--n-ratio-substeps", "2"],
                      ["--kappa-heavy", "0.2"], ["--kappa-heavy", "0.08", "--hasenbusch-layout", "shared",
                                                 "--n-heavy-substeps", "2"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                MAIN.parse_args(WILSON + extra)
        with self.assertRaises(SystemExit):
            MAIN.parse_args(COMMON + ["--fermions", "staggered", "--mass", "0.3", "--kappa-heavy", "0.1"])
        with self.assertRaises(SystemExit):
            MAIN.parse_args(COMMON + ["--fermions", "none", "--fast"])
        args = MAIN.parse_args(WILSON + ["--kappa-heavy", "0.08"])
        self.assertEqual((args.hasenbusch_layout, args.integrator_levels), ("nested", ["omelyan"] * 3))
        args = MAIN.parse_args(COMMON + ["--fermions", "staggered", "--mass", "0.3", "--fast"])
        self.assertEqual(args.on_unconverged, "raise")
        out = self.tmp / "hb"
        MAIN.main(WILSON + ["--kappa-heavy", "0.08", "--n-traj", "2", "--out", str(out)])
        with self.assertRaisesRegex(SystemExit, "differ"):
            MAIN.main(WILSON + ["--kappa-heavy", "0.09", "--n-traj", "3", "--resume", "--out", str(out)])
        with self.assertRaisesRegex(SystemExit, "differ"):
            MAIN.main(WILSON + ["--n-traj", "3", "--resume", "--out", str(out)])
        MAIN.main(WILSON + ["--kappa-heavy", "0.08", "--n-traj", "3", "--resume", "--out", str(out)])
        self.assertEqual(json.loads((out / "run_meta.json").read_text())["trajectories_done"], 3)


if __name__ == "__main__":
    unittest.main()
