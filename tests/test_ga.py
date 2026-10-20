import random
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from algoritma import GA
from models.model import Package, Truck


class GeneticAlgorithmTests(unittest.TestCase):
    def test_run_returns_best_by_fitness_and_reports_final_value(self) -> None:
        trucks = [Truck(dimensions=(2, 1, 1), max_capacity=2)]
        packages = [
            Package("a", (1, 1, 1), value=1, weight=1),
            Package("b", (1, 1, 1), value=2, weight=1),
        ]

        result = GA.run(
            trucks,
            packages,
            random.Random(1),
            population_size=6,
            max_iter=3,
            elitisme=1,
        )

        self.assertEqual(result["iterations"], 3)
        self.assertEqual(result["final_value"], result["final_state"].total_value())
        self.assertIsNotNone(result["best_valid_state"])


if __name__ == "__main__":
    unittest.main()
