"""main_pt_obc.py --save-resume / --resume: a run stopped and resumed equals the run done in one go.

Two ranks on the CPU: 4 steps in one job against 2 steps with --save-resume followed by --resume to 4
steps in the same output directory. The replica history, the swap works, the totals, the per-step log and
(for NEO-REX, which trains its flow along the run) the final flow must be identical.
"""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

COMMON = ["--cpu", "--D", "4", "--T", "8", "--L", "8", "--N", "3", "--defect-size", "1", "--time-slice", "4",
          "--space-slice", "3", "--beta", "6", "--parameter-min", "0", "--parameter-max", "1", "--batch-size", "2",
          "--thermal-steps", "2", "--orsteps", "1", "--no-cfg-cache", "--log-every", "1",
          "--save-link-work-values", "--protocol-patch-halfsize", "1", "--protocol-symmetric-updates",
          "--periodic-translation"]
METHODS = {
    "orex": ["--algorithm", "orex", "--orex-protocol-steps", "2"],
    "neorex": ["--algorithm", "neorex", "--nf-layer-type", "smearing", "--use-hyper-smearing",
               "--hyper-smearing-mode", "class", "--neorex-protocol-steps", "2",
               "--neorex-heatbath-steps-per-step", "1"],
}
TIMING_COLUMNS = {"elapsed_seconds", "thermalization_seconds", "measurement_elapsed_seconds", "total_elapsed_seconds"}


def _run(root, out, extra):
    from test.test_cnf_neorex import _free_port, _multirank_worker
    argv = [*COMMON, *extra, "--output-dir", str(out), "--main-dir", str(root)]
    mp.spawn(_multirank_worker, args=(2, _free_port(), argv), nprocs=2, join=True)


def _outputs(out):
    history = np.load(out / "replica_history.npz")
    works = np.load(out / "link_work_values.npz")
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    with open(out / "pt_metrics.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    keep = [i for i, name in enumerate(rows[0]) if name not in TIMING_COLUMNS and not name.startswith("gpu_")]
    return ({k: history[k] for k in history.files}, {k: works[k] for k in works.files}, meta,
            [[row[i] for i in keep] for row in rows])


class ResumeTest(unittest.TestCase):
    def assertSameRun(self, method):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _run(root, root / "whole", [*METHODS[method], "--steps", "4"])
            checkpoints = root / "checkpoints"
            _run(root, root / "parts", [*METHODS[method], "--steps", "2", "--save-resume", "--resume-dir", str(checkpoints)])
            self.assertTrue((checkpoints / "latest.json").is_file())
            _run(root, root / "parts", [*METHODS[method], "--steps", "4", "--resume", "--save-resume",
                                        "--resume-dir", str(checkpoints)])
            whole, parts = _outputs(root / "whole"), _outputs(root / "parts")
            for key in whole[0]:
                self.assertTrue(np.array_equal(whole[0][key], parts[0][key]), f"replica history {key}")
            self.assertEqual(whole[0]["labels"].shape[0], 4)
            for key in whole[1]:
                self.assertTrue(np.array_equal(whole[1][key], parts[1][key]), f"work values {key}")
            for section in ("swap", "work"):
                strip = lambda d: {k: v for k, v in d.items() if not k.endswith("_path")}  # noqa: E731
                self.assertEqual(strip(whole[2][section]), strip(parts[2][section]), section)
            self.assertEqual(whole[3], parts[3])  # the per-step log, timings aside
            self.assertEqual(parts[2]["steps_completed"], 4)
            self.assertEqual([r["from_step"] for r in parts[2]["resumes"]], [2])
            latest = json.loads((checkpoints / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["step"], 4)
            self.assertEqual(len(list(checkpoints.glob("step_*"))), 1)  # older ones removed
            if method == "neorex":
                name = "neorex_flow_final_nsteps2.pt"
                a = torch.load(root / "whole" / name, weights_only=False)
                b = torch.load(root / "parts" / name, weights_only=False)
                a = a.get("state_dict", a)
                b = b.get("state_dict", b)
                self.assertEqual(set(a), set(b))
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_orex(self):
        self.assertSameRun("orex")

    def test_neorex_training_its_flow(self):
        self.assertSameRun("neorex")

    def test_default_directory_is_under_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = root / "scratch"
            old = os.environ.get("SCRATCH")
            os.environ["SCRATCH"] = str(scratch)  # inherited by the spawned ranks
            try:
                _run(root, root / "run", [*METHODS["orex"], "--steps", "1", "--save-resume"])
                _run(root, root / "run", [*METHODS["orex"], "--steps", "2", "--resume"])
            finally:
                if old is None:
                    os.environ.pop("SCRATCH")
                else:
                    os.environ["SCRATCH"] = old
            found = list((scratch / "neoqcd_resume").glob("run_*/latest.json"))
            self.assertEqual(len(found), 1)
            self.assertFalse((root / "run" / "resume").exists())
            meta = json.loads((root / "run" / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["steps_completed"], 2)

    def test_changed_run_arguments_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _run(root, root / "run", [*METHODS["orex"], "--steps", "1", "--save-resume"])
            with self.assertRaises(Exception):
                _run(root, root / "run", [*METHODS["orex"], "--steps", "2", "--resume", "--beta", "6.1"])


if __name__ == "__main__":
    unittest.main()
