from pathlib import Path
import sys
import unittest
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neoqcd.obs import Obs, _drho_terms


def _ar1(length, replicas, phi, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(length, replicas))
    x = np.empty((length, replicas), dtype=np.float64)
    x[0] = noise[0]
    for t in range(1, length):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def _obs(x):
    return Obs([x[:, b] for b in range(x.shape[1])], [f"hmc|r{b}" for b in range(x.shape[1])])


def _drho_terms_loop(rho, i, w_max):
    # Madras-Sokal: rho(k + i) + rho(|k - i|) - 2 rho(i) rho(k), k = 1 .. w_max - i - 1
    return np.array([rho[k + i] + rho[abs(k - i)] - 2 * rho[i] * rho[k] for k in range(1, w_max - i)])


class ObsDrhoTest(unittest.TestCase):
    def test_terms_match_direct_loop_for_odd_and_even_windows(self):
        rng = np.random.default_rng(5)
        for w_max in (2, 3, 4, 5, 6, 7, 10, 11, 24, 25):
            rho = np.concatenate(([1.0], rng.uniform(-0.5, 1.0, size=w_max - 1)))
            for i in range(1, w_max):
                with self.subTest(w_max=w_max, i=i):
                    np.testing.assert_allclose(_drho_terms(rho, i, w_max), _drho_terms_loop(rho, i, w_max),
                                               rtol=0.0, atol=1e-15)
                    self.assertEqual(_drho_terms(rho, i, w_max).shape, (w_max - i - 1,))

    def test_unclosed_window_with_odd_w_max_gives_finite_errors(self):
        # 50 rows -> w_max = 25: the window does not close and falls back to i >= w_max // 2
        x = _ar1(50, 16, 0.5, 33)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = _obs(x)
            o.gamma_method(S=2.0)
        self.assertEqual(len(o.e_rho["hmc"]), 25)
        self.assertGreaterEqual(o.e_windowsize["hmc"], 25 // 2)
        for value in (o.dvalue, o.ddvalue, o.e_dtauint["hmc"], o.e_drho["hmc"][o.e_windowsize["hmc"]]):
            self.assertTrue(np.isfinite(value))
            self.assertGreater(value, 0.0)

    def test_e_drho_at_window_matches_direct_loop(self):
        cases = 0
        for length in (50, 51, 52, 53):  # w_max = 25, 25, 26, 26
            for seed in range(40):
                x = _ar1(length, 4, 0.8, 100 * length + seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    o = _obs(x)
                    o.gamma_method(S=2.0)
                w_max = len(o.e_rho["hmc"])
                i = o.e_windowsize["hmc"]
                expected = np.sqrt(np.sum(_drho_terms_loop(o.e_rho["hmc"], i, w_max) ** 2) / x.size)
                self.assertAlmostEqual(o.e_drho["hmc"][i], expected, places=14)
                self.assertTrue(np.isfinite(o.dvalue) and o.dvalue > 0.0)
                cases += i >= w_max // 2
        self.assertGreater(cases, 0)

    def test_many_short_correlated_series_do_not_raise(self):
        for phi in (0.5, 0.65, 0.8):
            for seed in range(30):
                x = _ar1(50, 16, phi, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    o = _obs(x)
                    o.gamma_method(S=2.0)
                self.assertTrue(np.isfinite(o.dvalue) and o.dvalue > 0.0)
                self.assertTrue(np.isfinite(o.ddvalue))


if __name__ == "__main__":
    unittest.main()
