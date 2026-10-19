from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

from models.geometry import is_feasible
from models.model import Package, State, Truck
from models.objective import normalize_objective, state_score
from models.operators import (
    iter_all_feasible_neighbors,
    make_initial_state,
    make_random_neighbor,
    state_signature,
)


VARIANTS = (
    "steepest_ascent",
    "sideways",
    "stochastic",
    "random_restart",
)

CONFIG_KEYS = {
    "steepest_ascent": ("max_iter",),
    "sideways": ("max_iter", "max_sideways"),
    "stochastic": ("max_iter", "max_neighbor_tries"),
    "random_restart": ("max_iter_per_restart", "max_restart"),
}


def normalize_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Varian HC tidak dikenal: {variant}")
    return variant


def _require_feasible_initial_state(initial_state: State) -> None:
    feasible, violations = is_feasible(initial_state)
    if not feasible:
        raise ValueError(f"Initial state HC harus feasible: {violations}")


def _require_nonnegative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} tidak boleh negatif, dapat {value}")


def _best_feasible_neighbor(
    state: State,
    rng: random.Random,
    objective: str = "value",
) -> Tuple[Optional[State], Optional[int], int]:
    best: Optional[State] = None
    best_value: Optional[int] = None
    best_count = 0
    neighbor_count = 0

    for neighbor in iter_all_feasible_neighbors(state):
        neighbor_count += 1
        value = state_score(neighbor, objective)
        if best_value is None or value > best_value:
            best = neighbor
            best_value = value
            best_count = 1
        elif value == best_value:
            best_count += 1
            if rng.randrange(best_count) == 0:
                best = neighbor

    return best, best_value, neighbor_count


def _random_feasible_neighbor(
    state: State,
    rng: random.Random,
    max_tries: int,
) -> Tuple[Optional[State], int]:
    current_signature = state_signature(state)
    for tries in range(1, max_tries + 1):
        candidate = make_random_neighbor(state, rng)
        if state_signature(candidate) == current_signature:
            continue
        if is_feasible(candidate)[0]:
            return candidate, tries
    return None, max_tries


def _result(
    variant: str,
    params: dict,
    initial_state: State,
    final_state: State,
    iterations: int,
    duration_s: float,
    history: List[dict],
    termination_reason: str,
    extra: Optional[dict] = None,
) -> dict:
    return {
        "algorithm": "hill_climbing",
        "variant": variant,
        "params": params,
        "initial_state": initial_state.copy(),
        "final_state": final_state.copy(),
        "best_valid_state": final_state.copy(),
        "final_objective": state_score(final_state, params.get("objective", "value")),
        "best_valid_value": final_state.total_value(),
        "iterations": iterations,
        "duration_s": duration_s,
        "termination_reason": termination_reason,
        "history": history,
        "extra": extra or {},
    }


def run_steepest(
    initial_state: State,
    rng: random.Random,
    max_iter: int = 1000,
    objective: str = "value",
) -> dict:
    normalize_objective(objective)
    _require_nonnegative("max_iter", max_iter)
    _require_feasible_initial_state(initial_state)
    start = time.perf_counter()
    current = initial_state.copy()
    current_value = state_score(current, objective)
    history: List[dict] = [{
        "iter": 0,
        "objective": current_value,
        "value": current.total_value(),
        "candidate_value": None,
        "accepted": False,
        "neighbors_evaluated": 0,
    }]

    iterations = 0
    termination_reason = "MAX_ITERATIONS"
    for iteration in range(1, max_iter + 1):
        neighbor, neighbor_value, neighbor_count = _best_feasible_neighbor(current, rng, objective)
        iterations = iteration
        accepted = neighbor is not None and neighbor_value > current_value

        if accepted:
            current = neighbor
            assert neighbor_value is not None
            current_value = neighbor_value

        history.append({
            "iter": iteration,
            "objective": current_value,
            "value": current.total_value(),
            "candidate_value": neighbor_value,
            "accepted": accepted,
            "neighbors_evaluated": neighbor_count,
        })

        if neighbor is None:
            termination_reason = "NO_FEASIBLE_NEIGHBOR"
            break
        if not accepted:
            termination_reason = "LOCAL_OPTIMUM"
            break

    duration = time.perf_counter() - start
    return _result(
        "steepest_ascent",
        {"max_iter": max_iter, "objective": objective},
        initial_state,
        current,
        iterations,
        duration,
        history,
        termination_reason,
    )


def run_sideways(
    initial_state: State,
    rng: random.Random,
    max_iter: int = 1000,
    max_sideways: int = 50,
    objective: str = "value",
) -> dict:
    normalize_objective(objective)
    _require_nonnegative("max_iter", max_iter)
    _require_nonnegative("max_sideways", max_sideways)
    _require_feasible_initial_state(initial_state)
    start = time.perf_counter()
    current = initial_state.copy()
    current_value = state_score(current, objective)
    consecutive_sideways = 0
    total_sideways = 0
    history: List[dict] = [{
        "iter": 0,
        "objective": current_value,
        "value": current.total_value(),
        "candidate_value": None,
        "accepted": False,
        "neighbors_evaluated": 0,
        "consecutive_sideways": 0,
        "total_sideways": 0,
    }]

    iterations = 0
    termination_reason = "MAX_ITERATIONS"
    for iteration in range(1, max_iter + 1):
        neighbor, neighbor_value, neighbor_count = _best_feasible_neighbor(current, rng, objective)
        iterations = iteration
        accepted = False
        was_sideways = False

        if neighbor is None:
            termination_reason = "NO_FEASIBLE_NEIGHBOR"
        elif neighbor_value > current_value:
            current = neighbor
            current_value = neighbor_value
            consecutive_sideways = 0
            accepted = True
        elif neighbor_value == current_value and consecutive_sideways < max_sideways:
            current = neighbor
            consecutive_sideways += 1
            total_sideways += 1
            accepted = True
            was_sideways = True
        elif neighbor_value == current_value:
            termination_reason = "MAX_SIDEWAYS"
        else:
            termination_reason = "LOCAL_OPTIMUM"

        history.append({
            "iter": iteration,
            "objective": current_value,
            "value": current.total_value(),
            "candidate_value": neighbor_value,
            "accepted": accepted,
            "neighbors_evaluated": neighbor_count,
            "consecutive_sideways": consecutive_sideways,
            "total_sideways": total_sideways,
        })

        if neighbor is None or not accepted:
            break
        if was_sideways and consecutive_sideways >= max_sideways:
            termination_reason = "MAX_SIDEWAYS"
            break

    duration = time.perf_counter() - start
    return _result(
        "sideways",
        {"max_iter": max_iter, "max_sideways": max_sideways, "objective": objective},
        initial_state,
        current,
        iterations,
        duration,
        history,
        termination_reason,
        {
            "sideways_moves": total_sideways,
            "final_consecutive_sideways": consecutive_sideways,
        },
    )


def run_stochastic(
    initial_state: State,
    rng: random.Random,
    max_iter: int = 1000,
    max_neighbor_tries: int = 50,
    objective: str = "value",
) -> dict:
    normalize_objective(objective)
    _require_nonnegative("max_iter", max_iter)
    if max_neighbor_tries <= 0:
        raise ValueError(
            f"max_neighbor_tries harus lebih besar dari 0, dapat {max_neighbor_tries}"
        )
    _require_feasible_initial_state(initial_state)
    start = time.perf_counter()
    current = initial_state.copy()
    current_value = state_score(current, objective)
    accepted_moves = 0
    rejected_moves = 0
    history: List[dict] = [{
        "iter": 0,
        "objective": current_value,
        "value": current.total_value(),
        "candidate_value": None,
        "accepted": False,
        "neighbor_tries": 0,
    }]

    iterations = 0
    termination_reason = "MAX_ITERATIONS"
    for iteration in range(1, max_iter + 1):
        neighbor, tries = _random_feasible_neighbor(current, rng, max_neighbor_tries)
        iterations = iteration

        if neighbor is None:
            history.append({
                "iter": iteration,
                "objective": current_value,
                "value": current.total_value(),
                "candidate_value": None,
                "accepted": False,
                "neighbor_tries": tries,
            })
            termination_reason = "NO_FEASIBLE_NEIGHBOR_FOUND"
            break

        neighbor_value = state_score(neighbor, objective)
        accepted = neighbor_value > current_value
        if accepted:
            current = neighbor
            current_value = neighbor_value
            accepted_moves += 1
        else:
            rejected_moves += 1

        history.append({
            "iter": iteration,
            "objective": current_value,
            "value": current.total_value(),
            "candidate_value": neighbor_value,
            "accepted": accepted,
            "neighbor_tries": tries,
        })

    duration = time.perf_counter() - start
    return _result(
        "stochastic",
        {"max_iter": max_iter, "max_neighbor_tries": max_neighbor_tries, "objective": objective},
        initial_state,
        current,
        iterations,
        duration,
        history,
        termination_reason,
        {"accepted_moves": accepted_moves, "rejected_moves": rejected_moves},
    )


def run_random_restart(
    initial_state: State,
    rng: random.Random,
    max_iter_per_restart: int = 1000,
    max_restart: int = 10,
    objective: str = "value",
) -> dict:
    normalize_objective(objective)
    _require_nonnegative("max_iter_per_restart", max_iter_per_restart)
    _require_nonnegative("max_restart", max_restart)
    _require_feasible_initial_state(initial_state)
    start = time.perf_counter()
    trucks: List[Truck] = list(initial_state.trucks)
    packages: List[Package] = [p.package for p in initial_state.placements]
    upper_bound = sum(package.value for package in packages)

    first_initial = initial_state.copy()
    best_state: Optional[State] = None
    best_restart = 0
    total_iterations = 0
    restart_summaries: List[dict] = []
    history: List[dict] = []
    running_global_best = -1
    termination_reason = "MAX_RESTARTS"

    for search_number in range(1, max_restart + 2):
        search_initial = (
            first_initial.copy()
            if search_number == 1
            else make_initial_state(trucks, packages, rng)
        )
        search_result = run_steepest(search_initial, rng, max_iter_per_restart, objective)
        search_final = search_result["final_state"]
        search_value = search_final.total_value()

        for row in search_result["history"]:
            running_global_best = max(running_global_best, row["value"])
            history.append({
                "global_iter": total_iterations + row["iter"],
                "restart": search_number,
                "iter_in_restart": row["iter"],
                "objective": row["objective"],
                "value": row["value"],
                "candidate_value": row["candidate_value"],
                "accepted": row["accepted"],
                "neighbors_evaluated": row["neighbors_evaluated"],
                "global_best_value": running_global_best,
            })

        total_iterations += search_result["iterations"]
        restart_summaries.append({
            "restart": search_number,
            "initial_value": search_initial.total_value(),
            "final_value": search_value,
            "iterations": search_result["iterations"],
            "termination_reason": search_result["termination_reason"],
            "duration_s": search_result["duration_s"],
        })

        if best_state is None or state_score(search_final, objective) > state_score(best_state, objective):
            best_state = search_final.copy()
            best_restart = search_number

        if best_state.total_value() == upper_bound:
            termination_reason = "ALL_PACKAGES_LOADED"
            break

    assert best_state is not None
    duration = time.perf_counter() - start
    iterations_per_restart = [entry["iterations"] for entry in restart_summaries]
    restart_count = len(restart_summaries) - 1

    return _result(
        "random_restart",
        {
            "max_iter_per_restart": max_iter_per_restart,
            "max_restart": max_restart,
            "objective": objective,
        },
        first_initial,
        best_state,
        total_iterations,
        duration,
        history,
        termination_reason,
        {
            "restart_count": restart_count,
            "iterations_per_restart": iterations_per_restart,
            "best_restart": best_restart,
            "restart_summaries": restart_summaries,
        },
    )


def run(
    initial_state: State,
    rng: random.Random,
    variant: str,
    **params,
) -> dict:
    normalized = normalize_variant(variant)
    if normalized == "steepest_ascent":
        return run_steepest(initial_state, rng, **params)
    if normalized == "sideways":
        return run_sideways(initial_state, rng, **params)
    if normalized == "stochastic":
        return run_stochastic(initial_state, rng, **params)
    return run_random_restart(initial_state, rng, **params)


def params_from_config(config: dict, variant: str) -> dict:
    normalized = normalize_variant(variant)
    section = config["hill_climbing"]
    if normalized not in section:
        raise KeyError(f"Config hill_climbing tidak punya bagian: {normalized}")
    variant_config = section[normalized]
    missing = [key for key in CONFIG_KEYS[normalized] if key not in variant_config]
    if missing:
        raise KeyError(f"Config {normalized} tidak punya key: {missing}")
    params = {key: variant_config[key] for key in CONFIG_KEYS[normalized]}
    params["objective"] = config.get("objective", "value")
    return params


def save_result(result: dict, out_dir: str | Path) -> Path:
    from models.io import save_history_csv, save_state

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_state(out / "initial_state.json", result["initial_state"])
    save_state(out / "final_state.json", result["final_state"])
    save_history_csv(out / "history.csv", result["history"])

    extra = result["extra"]
    if result["variant"] == "random_restart":
        save_history_csv(out / "restart_summary.csv", extra["restart_summaries"])

    summary = {
        "algorithm": result["algorithm"],
        "variant": result["variant"],
        "seed": result.get("seed"),
        "params": result["params"],
        "initial_value": result["initial_state"].total_value(),
        "final_value": result["final_state"].total_value(),
        "final_objective": result["final_objective"],
        "best_valid_value": result["best_valid_value"],
        "iterations": result["iterations"],
        "duration_s": result["duration_s"],
        "termination_reason": result["termination_reason"],
        "extra": extra,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return out


def run_all_experiments(
    config: dict,
    trucks: List[Truck],
    packages: List[Package],
    output_root: str | Path,
    dataset_name: str,
) -> List[dict]:
    records: List[dict] = []
    for variant in VARIANTS:
        params = params_from_config(config, variant)
        for run_number, seed in enumerate(config["seeds"], start=1):
            rng = random.Random(seed)
            initial_state = make_initial_state(trucks, packages, rng)
            result = run(initial_state, rng, variant, **params)
            result["seed"] = seed
            out = save_result(
                result,
                Path(output_root)
                / "hc"
                / dataset_name
                / variant
                / f"run{run_number}_seed{seed}",
            )
            records.append({"result": result, "output": out})
    return records
