"""Unit tests for the traffic-aware aggregation weights.

Run from phase2/:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.windowing import LazyWindows, Scaler  # noqa: E402
from src.federated.traffic_aggregation import (  # noqa: E402
    MU, PHI, PSI, R, SIGMA, V,
    aggregation_weights,
    client_metrics,
    metric_matrix,
)

FLOORS = (0.0, 0.1, 0.5, 1.0)


def random_case(rng, k=4):
    Z = rng.uniform(0.5, 100.0, size=(k, 6))
    sizes = rng.integers(100, 5000, size=k)
    coef = rng.uniform(0.1, 1.5, size=6)
    return Z, sizes, coef


class TestAggregationWeights(unittest.TestCase):

    def test_identical_clients_get_equal_weights(self):
        row = [1200, 210.0, 95.0, 0.0, 14.0, 20.0]
        Z = np.array([row] * 4)
        sizes = [1200] * 4
        rng = np.random.default_rng(0)
        for floor in FLOORS:
            for coef in ([1.0] * 6, rng.uniform(0.1, 1.5, 6)):
                w, parts = aggregation_weights(Z, sizes, coef, floor)
                np.testing.assert_allclose(w, 0.25, rtol=0, atol=1e-15)
                np.testing.assert_allclose(parts["traffic"], 0.25, rtol=0, atol=1e-15)

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            Z, sizes, coef = random_case(rng, k=int(rng.integers(2, 9)))
            floor = float(rng.uniform(0, 1))
            w, parts = aggregation_weights(Z, sizes, coef, floor)
            self.assertAlmostEqual(w.sum(), 1.0, places=12)
            self.assertAlmostEqual(parts["traffic"].sum(), 1.0, places=12)
            self.assertTrue((w >= 0).all())

    def test_floor_one_is_fedavg_exactly(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            Z, sizes, coef = random_case(rng)
            w, _ = aggregation_weights(Z, sizes, coef, floor=1.0)
            s = np.array(sizes, dtype=float)
            fedavg = s / s.sum()              # the runner's FedAvg expression
            self.assertTrue(np.array_equal(w, fedavg), (w, fedavg))

    def test_worst_client_still_gets_weight_with_floor(self):
        # Client 0 is the lowest on every normalised term: fewest windows, lowest
        # mean, std, Phi and Psi, and the stalest data (largest R, which is inverted).
        Z = np.array([
            [100, 50.0, 10.0, 30.0, 1.0, 2.0],
            [900, 150.0, 40.0, 0.0, 6.0, 9.0],
            [500, 300.0, 80.0, 10.0, 12.0, 15.0],
            [700, 220.0, 60.0, 5.0, 9.0, 11.0],
        ])
        sizes = Z[:, V]
        w0, parts = aggregation_weights(Z, sizes, [1.0] * 6, floor=0.0)
        self.assertEqual(parts["score"][0], 0.0)
        self.assertEqual(w0[0], 0.0)          # without a floor the paper's rule zeroes it
        for floor in (0.01, 0.1, 0.5):
            w, _ = aggregation_weights(Z, sizes, [1.0] * 6, floor=floor)
            self.assertGreater(w[0], 0.0)
            self.assertAlmostEqual(w[0], floor * sizes[0] / sizes.sum(), places=15)

    def test_constant_recency_normalises_to_zero(self):
        rng = np.random.default_rng(3)
        Z, sizes, coef = random_case(rng)
        Z[:, R] = 0.0
        _, parts = aggregation_weights(Z, sizes, coef, 0.1)
        np.testing.assert_array_equal(parts["normalized"][:, R], 0.0)

    def test_recency_is_inverted(self):
        Z = np.ones((3, 6))
        Z[:, R] = [0.0, 5.0, 10.0]           # client 0 is freshest
        _, parts = aggregation_weights(Z, [1, 1, 1], [0, 0, 0, 1, 0, 0], 0.0)
        self.assertGreater(parts["traffic"][0], parts["traffic"][1])
        self.assertEqual(parts["traffic"][2], 0.0)

    def test_coef_pairing_follows_eq14(self):
        # beta multiplies sigma, gamma multiplies mu.
        Z = np.ones((2, 6))
        Z[:, SIGMA] = [0.0, 1.0]
        Z[:, MU] = [1.0, 0.0]
        _, beta_only = aggregation_weights(Z, [1, 1], [0, 1, 0, 0, 0, 0], 0.0)
        _, gamma_only = aggregation_weights(Z, [1, 1], [0, 0, 1, 0, 0, 0], 0.0)
        np.testing.assert_allclose(beta_only["traffic"], [0.0, 1.0])
        np.testing.assert_allclose(gamma_only["traffic"], [1.0, 0.0])

    def test_bad_inputs_raise(self):
        Z = np.ones((2, 6))
        with self.assertRaises(ValueError):
            aggregation_weights(Z, [1, 1], [1, 1, 1], 0.1)
        with self.assertRaises(ValueError):
            aggregation_weights(Z, [1, 1], [1, 1, 1, 1, 1, -1], 0.1)
        with self.assertRaises(ValueError):
            aggregation_weights(Z, [1, 1], [1] * 6, 1.5)


class TestClientMetrics(unittest.TestCase):

    def test_metrics_in_vehicle_units(self):
        # one sensor, flow 10, 20, 15, 25, 30 vehicles, stored z-scored
        flow = np.array([10.0, 20.0, 15.0, 25.0, 30.0])
        scaler = Scaler(mean=20.0, std=5.0)
        tensor = np.zeros((5, 1, 2), dtype=np.float32)
        tensor[:, 0, 0] = scaler.transform(flow)
        lw = LazyWindows(tensor, np.arange(2), lookback=2, horizon=2, target_channel=0)

        m = client_metrics(lw, scaler)
        step = np.diff(flow)                  # 10, -5, 10, 5
        self.assertEqual(m["V"], 2.0)
        self.assertAlmostEqual(m["mu"], flow.mean(), places=5)
        self.assertAlmostEqual(m["sigma"], flow.std(), places=5)
        self.assertAlmostEqual(m["phi"], np.abs(step).mean(), places=5)
        self.assertAlmostEqual(m["psi"], step.std(), places=5)
        self.assertEqual(m["t_max"], 4)

    def test_recency_from_t_max(self):
        Z = metric_matrix([
            {"V": 1, "mu": 1, "sigma": 1, "t_max": 99, "phi": 1, "psi": 1},
            {"V": 1, "mu": 1, "sigma": 1, "t_max": 90, "phi": 1, "psi": 1},
        ])
        np.testing.assert_array_equal(Z[:, R], [0, 9])
        np.testing.assert_array_equal(Z[:, [V, MU, PHI, PSI]], 1)


if __name__ == "__main__":
    unittest.main()
