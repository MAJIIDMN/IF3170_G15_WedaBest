import math
import random
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from algoritma import SA
from models.model import Package, Placement, State, Truck


class SimulatedAnnealingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truck = Truck(dimensions=(2, 1, 1), max_capacity=2)
        self.a = Package("a", (1, 1, 1), value=1, weight=1)
        self.b = Package("b", (1, 1, 1), value=2, weight=1)
        self.state = State(
            trucks=[self.truck],
            placements=[
                Placement(self.a, position=(0, 0, 0)),
                Placement(self.b, position=None),
            ],
        )

    def test_acceptance_probability_is_exp_delta_over_temperature(self) -> None:
        self.assertAlmostEqual(SA.acceptance_probability(-10, 100), math.exp(-0.1))
        self.assertAlmostEqual(SA.acceptance_probability(0, 10), 1.0)
        self.assertGreater(SA.acceptance_probability(5, 10), 1.0)

    def test_run_uses_probabilistic_acceptance_without_threshold(self) -> None:
        result = SA.run(
            self.state,
            random.Random(1),
            initial_temperature=1.0,
            cooling_rate=0.9,
            min_temperature=1e-6,
            max_iter=50,
        )

        self.assertNotIn("probability_threshold", result["params"])
        self.assertTrue(result["history"])
        for row in result["history"]:
            self.assertIsNotNone(row["accept_prob"])
            if row["delta_e"] > 0:
                self.assertTrue(row["accepted"])

    def test_stuck_metric_is_window_based_and_has_rate(self) -> None:
        result = SA.run(
            self.state,
            random.Random(1),
            initial_temperature=1.0,
            cooling_rate=0.9,
            min_temperature=1e-6,
            max_iter=50,
            stuck_window=10,
        )

        extra = result["extra"]
        self.assertEqual(extra["stuck_window"], 10)
        self.assertIn("stuck_definition", extra)
        expected_rate = extra["stuck_count"] / result["iterations"] * 1000.0
        self.assertAlmostEqual(extra["stuck_rate"], expected_rate)


if __name__ == "__main__":
    unittest.main()
