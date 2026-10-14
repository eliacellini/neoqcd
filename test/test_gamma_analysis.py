import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neoqcd.analysis import OBSERVABLES_HEADER, REPORT_HEADER, analyze_observables_csv


def _write(path, values):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OBSERVABLES_HEADER)
        for trajectory in range(values.shape[0]):
            for measurement in range(values.shape[1]):
                for sample in range(values.shape[2]):
                    writer.writerow([
                        trajectory + 1, measurement + 1, sample, 5.8 + 0.1 * measurement,
                        *values[trajectory, measurement, sample].tolist(),
                    ])


def _streams(length, seed, phi=0.0, samples=3, measurements=1):
    rng = np.random.default_rng(seed)
    work = np.empty((length, samples), dtype=np.float64)
    noise = rng.normal(size=(length, samples))
    work[0] = noise[0]
    for index in range(1, length):
        work[index] = phi * work[index - 1] + noise[index]
    values = np.empty((length, measurements, samples, len(OBSERVABLES_HEADER) - 4), dtype=np.float64)
    for measurement in range(measurements):
        x = work + measurement * 2.0
        values[:, measurement, :, 0] = 3.0 + 0.4 * x
        values[:, measurement, :, 1] = 1.0 + 0.1 * x
        values[:, measurement, :, 2] = -0.2 + 0.05 * x
        values[:, measurement, :, 3] = 0.8 + 0.02 * x
        values[:, measurement, :, 4] = 0.3 + 0.01 * x
        values[:, measurement, :, 5] = 0.09 + 0.006 * x
    return values


def _report(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == REPORT_HEADER
        return {(int(row["measurement"]), row["observable"]): row for row in reader}


class GammaAnalysisTest(unittest.TestCase):
    def test_correlated_streams_have_larger_gamma_tau_than_iid(self):
        with tempfile.TemporaryDirectory() as directory:
            correlated = Path(directory) / "correlated.csv"
            iid = Path(directory) / "iid.csv"
            _write(correlated, _streams(256, 11, phi=0.85))
            _write(iid, _streams(256, 12, phi=0.0))
            analyze_observables_csv(correlated, output_path=Path(directory) / "correlated_gamma.csv")
            correlated_work = float(_report(Path(directory) / "correlated_gamma.csv")[(1, "work")]["tau_int"])
            analyze_observables_csv(iid, output_path=Path(directory) / "iid_gamma.csv")
            iid_work = float(_report(Path(directory) / "iid_gamma.csv")[(1, "work")]["tau_int"])
            self.assertGreater(correlated_work, iid_work + 0.5)

    def test_stable_reweighting_and_centered_variance_under_constant_work_shift(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.csv"
            shifted = Path(directory) / "shifted.csv"
            values = _streams(64, 18)
            _write(original, values)
            _write(shifted, values.copy())
            with shifted.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            for row in rows[1:]:
                row[4] = str(float(row[4]) + 10000.0)
            with shifted.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            analyze_observables_csv(original)
            original_rows = _report(Path(directory) / "gamma_analysis.csv")
            analyze_observables_csv(shifted, output_path=Path(directory) / "shifted_gamma.csv")
            shifted_rows = _report(Path(directory) / "shifted_gamma.csv")
            self.assertAlmostEqual(
                float(shifted_rows[(1, "deltaF")]["value"]) - float(original_rows[(1, "deltaF")]["value"]),
                10000.0, places=9,
            )
            for name in ("deltaF", "work_variance", "ess_fraction", "heat_reweighted",
                         "logJ_reweighted", "action_reweighted", "plaquette_reweighted",
                         "plaquette2_reweighted"):
                for field in ("error", "tau_int"):
                    self.assertAlmostEqual(
                        float(shifted_rows[(1, name)][field]), float(original_rows[(1, name)][field]), places=9,
                        msg=f"{name} {field}",
                    )
                if name != "deltaF":
                    self.assertAlmostEqual(
                        float(shifted_rows[(1, name)]["value"]), float(original_rows[(1, name)]["value"]), places=9,
                        msg=name,
                    )

    def test_intermediate_measurements_are_analyzed_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observables.csv"
            _write(path, _streams(12, 21, measurements=2))
            result = analyze_observables_csv(path)
            rows = _report(Path(directory) / "gamma_analysis.csv")
            self.assertEqual(result["measurement_count"], 2)
            self.assertAlmostEqual(float(rows[(2, "plaquette")]["value"]) - float(rows[(1, "plaquette")]["value"]), 0.02)
            self.assertAlmostEqual(float(rows[(1, "work")]["coupling"]), 5.8)
            self.assertAlmostEqual(float(rows[(2, "work")]["coupling"]), 5.9)

    def test_incomplete_grid_is_rejected_and_short_series_has_no_iid_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            incomplete = Path(directory) / "incomplete.csv"
            values = _streams(8, 22)
            _write(incomplete, values)
            with incomplete.open(newline="", encoding="utf-8") as handle:
                all_rows = list(csv.reader(handle))
            rows = all_rows.copy()
            rows.pop()
            with incomplete.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            with self.assertRaisesRegex(ValueError, "incomplete observable grid"):
                analyze_observables_csv(incomplete)

            truncated_dir = Path(directory) / "truncated"
            truncated_dir.mkdir()
            truncated = truncated_dir / "observables.csv"
            with truncated.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(OBSERVABLES_HEADER)
                for row in all_rows[1:-3]:
                    writer.writerow(row)
            (truncated_dir / "run_meta.json").write_text(
                json.dumps({"phase": "evaluate", "evaluation_trajectory_count": 8,
                            "args": {"batch_size": 3, "int_meas": 1}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "run metadata expects"):
                analyze_observables_csv(truncated)

            short = Path(directory) / "short.csv"
            _write(short, _streams(4, 23))
            result = analyze_observables_csv(short)
            self.assertEqual(result["status"], "insufficient_statistics")
            short_rows = _report(Path(directory) / "gamma_analysis.csv")
            self.assertEqual(short_rows[(1, "work")]["status"], "insufficient_statistics")
            self.assertEqual(short_rows[(1, "work")]["error"], "")
            self.assertEqual(short_rows[(1, "work")]["tau_int"], "")

    def test_constant_series_at_minimum_length_has_zero_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constant.csv"
            values = np.ones((5, 1, 2, len(OBSERVABLES_HEADER) - 4), dtype=np.float64)
            values[..., 0] = 2.5
            _write(path, values)
            result = analyze_observables_csv(path)
            self.assertEqual(result["status"], "ok")
            row = _report(Path(directory) / "gamma_analysis.csv")[(1, "work")]
            self.assertEqual(float(row["error"]), 0.0)
            self.assertEqual(float(row["tau_int"]), 0.5)
            self.assertEqual(float(row["tau_int_error"]), 0.0)


if __name__ == "__main__":
    unittest.main()
