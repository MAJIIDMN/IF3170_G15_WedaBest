import random
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from algoritma import HC
from models.geometry import is_feasible
from models.model import Package, Placement, State, Truck
from models.operators import iter_all_feasible_neighbors, state_signature


class HillClimbingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truck = Truck(dimensions=(1, 1, 1), max_capacity=1)
        self.low = Package("low", (1, 1, 1), value=1, weight=1)
        self.high = Package("high", (1, 1, 1), value=2, weight=1)
        self.state = State(
            trucks=[self.truck],
            placements=[
                Placement(self.low, position=(0, 0, 0)),
                Placement(self.high, position=None),
            ],
        )

    def test_exhaustive_neighbors_are_feasible_unique_and_not_current(self) -> None:
        neighbors = list(iter_all_feasible_neighbors(self.state))
        signatures = [state_signature(neighbor) for neighbor in neighbors]

        self.assertTrue(neighbors)
        self.assertEqual(len(signatures), len(set(signatures)))
        self.assertNotIn(state_signature(self.state), signatures)
        self.assertTrue(all(is_feasible(neighbor)[0] for neighbor in neighbors))

    def test_steepest_chooses_highest_valued_successor(self) -> None:
        result = HC.run_steepest(self.state, random.Random(1), max_iter=10)

        self.assertEqual(result["final_state"].total_value(), 2)
        self.assertEqual(result["termination_reason"], "LOCAL_OPTIMUM")
        self.assertTrue(is_feasible(result["final_state"])[0])

    def test_sideways_stops_at_configured_limit(self) -> None:
        only_package_state = State(
            trucks=[Truck(dimensions=(2, 1, 1), max_capacity=1)],
            placements=[Placement(self.low, position=(0, 0, 0))],
        )
        result = HC.run_sideways(
            only_package_state,
            random.Random(2),
            max_iter=10,
            max_sideways=2,
        )

        self.assertEqual(result["termination_reason"], "MAX_SIDEWAYS")
        self.assertEqual(result["extra"]["sideways_moves"], 2)

    def test_stochastic_never_accepts_non_improving_candidate(self) -> None:
        result = HC.run_stochastic(
            self.state,
            random.Random(3),
            max_iter=50,
            max_neighbor_tries=50,
        )

        previous_value = result["history"][0]["value"]
        for row in result["history"][1:]:
            if row["accepted"]:
                self.assertGreater(row["candidate_value"], previous_value)
            previous_value = row["value"]

    def test_random_restart_reports_restart_and_iteration_counts(self) -> None:
        result = HC.run_random_restart(
            self.state,
            random.Random(4),
            max_iter_per_restart=10,
            max_restart=2,
        )

        self.assertEqual(result["termination_reason"], "MAX_RESTARTS")
        self.assertEqual(result["extra"]["restart_count"], 2)
        self.assertEqual(len(result["extra"]["iterations_per_restart"]), 3)
        self.assertEqual(
            result["iterations"],
            sum(result["extra"]["iterations_per_restart"]),
        )


if __name__ == "__main__":
    unittest.main()
