from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

from models.model import State
from models.geometry import is_feasible, overweight_score, validate
from models.operators import make_random_neighbor


def penalty(state: State) -> float:
    violations = validate(state)
    return (
        len(violations["bounds"])
        + len(violations["overlap"])
        + len(violations["support"])
        + len(violations["fragile"])
        + overweight_score(state)
    )


def objective_function(state: State) -> float:
    lambda_penalty = sum(p.package.value for p in state.placements) + 1
    return state.total_value() - lambda_penalty * penalty(state)


def schedule(iteration: int, initial_temperature: float, cooling_rate: float) -> float:
    return initial_temperature * (cooling_rate ** iteration)


def acceptance_probability(delta_e: float, temperature: float) -> float:
    return math.exp(delta_e / temperature)


def make_valid_neighbor(state: State, rng: random.Random, max_tries: int) -> Tuple[State, int]:
    # Bangkitkan neighbor acak sampai valid, paling banyak max_tries kali.
    # Kalau tidak ketemu, kembalikan kandidat terakhir; penalti di objective yang akan menolaknya.
    for tries in range(1, max_tries + 1):
        neighbor = make_random_neighbor(state, rng)
        if is_feasible(neighbor)[0]:
            break
    return neighbor, tries


def run(
    initial_state: State,
    rng: random.Random,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.999,
    min_temperature: float = 0.01,
    probability_threshold: float = 0.6,
    max_iter: int = 20000,
    stuck_window: int = 500,
    max_neighbor_tries: int = 50,
) -> dict:
    start = time.perf_counter()

    current = initial_state
    current_obj = objective_function(current)
    current_feasible = is_feasible(current)[0]

    best_valid_state: Optional[State] = current.copy() if current_feasible else None

    # Stuck: +1 setiap stuck_window iterasi berturut-turut tanpa objective terbaik naik
    best_obj = current_obj
    no_improve_local = 0
    stuck_count_global = 0

    history: List[dict] = []
    iteration = 0

    while iteration < max_iter:
        temperature = schedule(iteration, initial_temperature, cooling_rate)
        if temperature < min_temperature:
            break

        neighbor, tries = make_valid_neighbor(current, rng, max_neighbor_tries)
        neighbor_obj = objective_function(neighbor)
        delta_e = neighbor_obj - current_obj

        if delta_e > 0:
            accept_prob = None
            accepted = True
        else:
            accept_prob = acceptance_probability(delta_e, temperature)
            accepted = accept_prob >= probability_threshold

        if accepted:
            current = neighbor
            current_obj = neighbor_obj
            current_feasible = is_feasible(current)[0]

            if current_feasible and (
                best_valid_state is None or current.total_value() > best_valid_state.total_value()
            ):
                best_valid_state = current.copy()

        if current_obj > best_obj:
            best_obj = current_obj
            no_improve_local = 0
        else:
            no_improve_local += 1
            if no_improve_local >= stuck_window:
                stuck_count_global += 1
                no_improve_local = 0

        history.append({
            "iter": iteration,
            "temperature": temperature,
            "objective": current_obj,
            "value": current.total_value(),
            "feasible": current_feasible,
            "delta_e": delta_e,
            "accept_prob": accept_prob,
            "accepted": accepted,
            "neighbor_tries": tries,
        })

        iteration += 1

    duration = time.perf_counter() - start

    return {
        "algorithm": "sa",
        "params": {
            "initial_temperature": initial_temperature,
            "cooling_rate": cooling_rate,
            "min_temperature": min_temperature,
            "probability_threshold": probability_threshold,
            "max_iter": max_iter,
            "stuck_window": stuck_window,
            "max_neighbor_tries": max_neighbor_tries,
        },
        "initial_state": initial_state,
        "final_state": current,
        "best_valid_state": best_valid_state,
        "final_objective": current_obj,
        "best_valid_value": best_valid_state.total_value() if best_valid_state else None,
        "iterations": iteration,
        "duration_s": duration,
        "history": history,
        "extra": {"stuck_count": stuck_count_global},
    }


CONFIG_KEYS = [
    "initial_temperature",
    "cooling_rate",
    "min_temperature",
    "probability_threshold",
    "max_iter",
    "stuck_window",
    "max_neighbor_tries",
]


def params_from_config(config: dict) -> dict:
    section = config["simulated_annealing"]
    missing = [k for k in CONFIG_KEYS if k not in section]
    if missing:
        raise KeyError(f"Config simulated_annealing tidak punya key: {missing}")
    return {k: section[k] for k in CONFIG_KEYS}


def save_result(result: dict, out_dir: str | Path) -> Path:
    """Simpan state awal & akhir, history, dan ringkasan satu run ke out_dir."""
    import json
    from models.io import save_history_csv, save_state

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_state(out / "initial_state.json", result["initial_state"])
    save_state(out / "final_state.json", result["final_state"])
    save_history_csv(out / "history.csv", result["history"])

    summary = {
        "final_objective": result["final_objective"],
        "duration_s": result["duration_s"],
        "stuck_count": result["extra"]["stuck_count"],
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out


if __name__ == "__main__":
    import sys
    from models.io import load_config, load_dataset
    from models.operators import make_initial_state

    config = load_config("data/experiments.json")
    params = params_from_config(config)
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else config["dataset_utama"]
    dataset = Path(dataset_path).stem
    print("dataset   :", dataset_path)
    print("parameter :", params)

    trucks, packages = load_dataset(dataset_path)
    for k, seed in enumerate(config["seeds"], start=1):
        rng = random.Random(seed)
        s0 = make_initial_state(trucks, packages, rng)
        result = run(s0, rng, **params)
        out = save_result(
            result,
            Path("..") / config["output_dir"] / "sa" / dataset / "default" / f"run{k}_seed{seed}",
        )
        print(
            f"run {k} (seed {seed}): value awal {s0.total_value()}"
            f" -> best_valid {result['best_valid_value']},"
            f" final valid? {is_feasible(result['final_state'])[0]},"
            f" iterasi {result['iterations']}, durasi {result['duration_s']:.2f}s,"
            f" stuck {result['extra']['stuck_count']} -> {out}"
        )
