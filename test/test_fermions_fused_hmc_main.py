"""main/main_hmc.py --md-kernels (neoqcd.fermions.fused_hmc) end to end, on the CPU emulation.

The host build of the kernels ("--kernel cpu") must reproduce the run without --md-kernels to
rounding: dH within 1e-9, identical accept decisions and CG counts; --md-kernels none and runs on
"auto" without a compiled target are the old path bitwise; the components are in the resume key.
(The bitwise identity of runs without --md-kernels with the driver before fused_hmc was checked
once against commit 7495125: reference, --fast, --kernel cpu double/mixed, --kernel loop, quenched,
fast staggered; see doc/fermions.md 5b.)
"""

import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("main_hmc_md", ROOT / "main" / "main_hmc.py")
MAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAIN)

from neoqcd.fermions import fused as F  # noqa: E402
from neoqcd.fermions import fused_hmc as M  # noqa: E402

COMMON = ["--D", "2", "--T", "4", "--L", "4", "--N", "3", "--beta", "2.0", "--batch", "2",
          "--tau", "0.5", "--n-steps", "3", "--n-gauge-substeps", "2", "--device", "cpu",
          "--seed", "7", "--start", "hot", "--quiet", "--fermions", "wilson", "--kappa", "0.12",
          "--solver-tol", "1e-12", "--action-tol", "1e-13", "--fast"]
G4 = ["--D", "4", "--T", "4", "--L", "2"]  # 2^3 x 4 (later flags win)
DETERMINISTIC = ("traj", "replica", "therm", "plaquette", "dH", "accepted", "exp_mdH", "solves", "cg_iter",
                 "cg_iter_replica", "max_residual")

_SAVED_CACHE = None
_TMP = None


def setUpModule():
    global _SAVED_CACHE, _TMP
    _SAVED_CACHE = os.environ.get("NEOQCD_KERNEL_CACHE")
    if _SAVED_CACHE is None:
        _TMP = tempfile.TemporaryDirectory(prefix="neoqcd_kernels_")
        os.environ["NEOQCD_KERNEL_CACHE"] = _TMP.name


def tearDownModule():
    if _SAVED_CACHE is None:
        os.environ.pop("NEOQCD_KERNEL_CACHE", None)
        _TMP.cleanup()


def _rows(path):
    with open(Path(path) / "observables.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def _cpu_reason():
    for status in (F.backend_status("cpu"), M.md_backend_status("cpu")):
        if not status[0]:
            return status[1]
    return None


class MDKernelsDriverTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, argv, name):
        out = self.tmp / name
        MAIN.main(argv + ["--out", str(out)])
        return out

    def assert_same_chain(self, a, b, dh_tol=1e-9):
        ra, rb = _rows(a), _rows(b)
        self.assertEqual(len(ra), len(rb))
        for x, y in zip(ra, rb):
            self.assertLess(abs(float(x["dH"]) - float(y["dH"])), dh_tol)
            self.assertLess(abs(float(x["plaquette"]) - float(y["plaquette"])), 1e-12)
            for col in ("accepted", "solves", "cg_iter", "cg_iter_replica"):
                self.assertEqual(x[col], y[col], col)

    def test_md_kernels_reproduce_the_fused_hop_run(self):
        """--md-kernels all and each component alone (2D, 4D; double and mixed precision) give the
        --kernel cpu chain: dH to 1e-9, the same accept decisions and CG counts; run_meta records."""
        reason = _cpu_reason()
        if reason is not None:
            self.skipTest(f"host build unavailable: {reason}")
        for geo, prec in (([], "double"), ([], "mixed"), (G4, "double")):
            base = COMMON + geo + ["--kernel", "cpu", "--precision", prec, "--n-traj", "3"]
            ref = self.run_main(base, f"ref_{len(geo)}_{prec}")
            meta = json.loads((ref / "run_meta.json").read_text())
            self.assertIsNone(meta["run"]["md_kernels"])
            self.assertEqual(meta["parameters"]["md_kernels"], [])
            sets = ["all"] + (list(M.MD_COMPONENTS) if not geo and prec == "double" else [])
            for md in sets:
                with self.subTest(geometry=geo, precision=prec, md=md):
                    out = self.run_main(base + ["--md-kernels", md], f"md_{len(geo)}_{prec}_{md}")
                    self.assert_same_chain(ref, out)
                    info = json.loads((out / "run_meta.json").read_text())["run"]["md_kernels"]
                    want = list(M.MD_COMPONENTS) if md == "all" else [md]
                    self.assertEqual(info["requested"], want)
                    self.assertEqual(info["target"], "cpu")
                    self.assertIsNone(info["fallback_reason"])
                    self.assertIn("neoqcd_md_cpu_", info["backend_status"])

    def test_none_and_auto_without_target_are_the_old_path_bitwise(self):
        """--md-kernels none is the run without the flag; on a CPU device "auto" resolves the hop to the
        --fast operator, and --md-kernels all then runs the torch code: bitwise the same chain, with
        the reason in run_meta.json and one log line."""
        base = COMMON + ["--n-traj", "2"]
        for kernel in (["--kernel", "auto"],) + ((["--kernel", "cpu"],) if _cpu_reason() is None else ()):
            with self.subTest(kernel=kernel[1]):
                plain = self.run_main(base + kernel, f"plain_{kernel[1]}")
                none = self.run_main(base + kernel + ["--md-kernels", "none"], f"none_{kernel[1]}")
                ckpt = json.loads((none / "checkpoint.json").read_text())
                self.assertNotIn("md_kernels", ckpt.get("extra", {}))
                for a, b in zip(_rows(plain), _rows(none)):
                    self.assertEqual([a[c] for c in DETERMINISTIC], [b[c] for c in DETERMINISTIC])
        argv = COMMON + ["--n-traj", "2", "--kernel", "auto", "--md-kernels", "all"]
        argv.remove("--quiet")
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            out = self.run_main(argv, "auto_all")
        info = json.loads((out / "run_meta.json").read_text())["run"]["md_kernels"]
        self.assertIsNone(info["target"])
        self.assertIn("no compiled target", log.getvalue())
        for a, b in zip(_rows(self.tmp / "plain_auto"), _rows(out)):
            self.assertEqual([a[c] for c in DETERMINISTIC], [b[c] for c in DETERMINISTIC])

    def test_argument_checks(self):
        for extra in (["--md-kernels", "all"],  # needs --kernel
                      ["--kernel", "loop", "--md-kernels", "all"],  # needs a compiled kernel
                      ["--kernel", "cpu", "--md-kernels", "gauge"],  # unknown component
                      ["--kernel", "cpu", "--md-kernels", "all,cg"],
                      ["--kernel", "cpu", "--md-kernels", "all", "--dd-blocks", "4", "--T", "8", "--L", "8"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                MAIN.parse_args(COMMON + extra)
        args = MAIN.parse_args(COMMON + ["--kernel", "cpu", "--md-kernels", "cg, gauge_force"])
        self.assertEqual(args.md_kernels, ["gauge_force", "cg"])
        self.assertEqual(MAIN.physics_key(args)["md_kernels"], ["gauge_force", "cg"])
        self.assertEqual(MAIN.parse_args(COMMON + ["--kernel", "cpu"]).md_kernels, [])
        self.assertEqual(MAIN.parse_args(COMMON + ["--kernel", "cpu", "--md-kernels", "all"]).md_kernels,
                         list(M.MD_COMPONENTS))

    def test_resume_key_and_bitwise_resume(self):
        """A run with --md-kernels resumes bitwise with the same components; resuming it with other
        components or without them is refused, and so is adding them to an older run."""
        reason = _cpu_reason()
        if reason is not None:
            self.skipTest(f"host build unavailable: {reason}")
        base = COMMON + ["--kernel", "cpu", "--md-kernels", "all", "--checkpoint-every", "2"]
        full = self.run_main(base + ["--n-traj", "4"], "full")
        part = self.tmp / "part"
        MAIN.main(base + ["--n-traj", "2", "--out", str(part)])
        MAIN.main(base + ["--n-traj", "4", "--out", str(part), "--resume"])
        for a, b in zip(_rows(full), _rows(part)):
            self.assertEqual([a[c] for c in DETERMINISTIC], [b[c] for c in DETERMINISTIC])
        for other in (["--md-kernels", "gauge_force"], []):
            argv = COMMON + ["--kernel", "cpu"] + other + ["--n-traj", "5", "--out", str(part), "--resume"]
            with self.subTest(other=other), self.assertRaises(SystemExit):
                MAIN.main(argv)
        old = self.run_main(COMMON + ["--kernel", "cpu", "--n-traj", "1"], "old")
        with self.assertRaises(SystemExit):
            MAIN.main(COMMON + ["--kernel", "cpu", "--md-kernels", "all", "--n-traj", "2", "--out", str(old),
                                "--resume"])


class CudaDriverTest(unittest.TestCase):
    """CUDA twin (skipped without a CUDA device and nvcc): main_hmc on the GPU with --kernel cuda and
    --md-kernels all reproduces the --kernel cuda chain (4D 4^4, batch 4): dH to 1e-9, the same
    accept decisions and CG counts; each component alone too."""

    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("no CUDA device")
        for ok, why in (F.backend_status("cuda"), M.md_backend_status("cuda")):
            if not ok:
                raise unittest.SkipTest(f"cuda kernels unavailable: {why}")

    def test_cuda_md_kernels_reproduce_the_fused_hop_run(self):
        base = [a for a in COMMON if a != "cpu"]
        base[base.index("--device") + 1:base.index("--device") + 1] = ["cuda"]
        base += ["--D", "4", "--T", "4", "--L", "4", "--beta", "5.6", "--batch", "4", "--kappa", "0.13",
                 "--kernel", "cuda", "--n-traj", "3"]
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref"
            MAIN.main(base + ["--out", str(ref)])
            for md in ["all"] + list(M.MD_COMPONENTS):
                with self.subTest(md=md):
                    out = Path(tmp) / md
                    MAIN.main(base + ["--md-kernels", md, "--out", str(out)])
                    for x, y in zip(_rows(ref), _rows(out)):
                        self.assertLess(abs(float(x["dH"]) - float(y["dH"])), 1e-9)
                        for col in ("accepted", "solves", "cg_iter"):
                            self.assertEqual(x[col], y[col], col)
                    info = json.loads((out / "run_meta.json").read_text())["run"]["md_kernels"]
                    self.assertEqual(info["target"], "cuda")


class BenchTrajectoryTest(unittest.TestCase):
    def test_trajectory_rows(self):
        """main/bench_fermions.py --trajectory-only: one md_<setting> row per setting and its parts;
        with the kernels on, the dH of md_none to rounding (notes)."""
        reason = _cpu_reason()
        if reason is not None:
            self.skipTest(f"host build unavailable: {reason}")
        spec = importlib.util.spec_from_file_location("bench_fermions_md", ROOT / "main" / "bench_fermions.py")
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            bench.main(["--device", "cpu", "--trajectory-only", "--trajectory-configs", "2^3x4:2",
                        "--trajectory-md", "none", "all", "gauge_force", "--trajectory-steps", "2",
                        "--trajectory-gauge-substeps", "2", "--trajectory-repeats", "1", "--trajectory-warmup", "0",
                        "--quiet", "--out", tmp])
            with open(Path(tmp) / "bench.csv", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertTrue(all(r["status"] == "ok" for r in rows), rows)
        totals = {r["variant"]: r for r in rows if r["measurement"] == "trajectory"}
        self.assertEqual(sorted(totals), ["md_all", "md_gauge_force", "md_none"])
        parts = [r for r in rows if r["measurement"] == "trajectory_part"]
        self.assertEqual(len(parts), 3 * len(bench.TRAJECTORY_PARTS))
        self.assertIn("update=cpu", totals["md_all"]["notes"])
        self.assertIn("update=torch", totals["md_gauge_force"]["notes"])
        diff = float(totals["md_all"]["notes"].split("max|dH - dH(md_none)|=")[1])
        self.assertLess(diff, 1e-9)
        for r in parts:
            self.assertGreaterEqual(float(r["seconds_per_call"]), 0.0)


if __name__ == "__main__":
    unittest.main()
