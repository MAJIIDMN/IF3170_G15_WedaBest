from __future__ import annotations

import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

from models.model import State, Truck, Package
from models.geometry import is_feasible, overweight_score, validate
from models.objective import max_score, normalize_objective, state_score
from models.operators import make_initial_state, make_random_neighbor


def penalty(state: State) -> float:
    violations = validate(state)
    return (
        len(violations["bounds"])
        + len(violations["overlap"])
        + len(violations["support"])
        + len(violations["fragile"])
        + overweight_score(state)
    )


def objective_function(state: State, objective: str = "value") -> float:
    lambda_penalty = max_score(state, objective) + 1
    return state_score(state, objective) - lambda_penalty * penalty(state)


def make_population(
    trucks: List[Truck],
    packages: List[Package],
    rng: random.Random,
    size: int,
) -> List[State]:
    return [make_initial_state(trucks, packages, rng) for _ in range(size)]


def roulette(population: List[State], fitnesses: List[float], rng: random.Random) -> State:
    # Fitness bisa negatif, geser dulu supaya semua bobot > 0.
    floor = min(fitnesses)
    weights = [f - floor + 1e-6 for f in fitnesses]
    total = sum(weights)
    threshold = rng.random() * total
    cumulative = 0.0
    for individual, weight in zip(population, weights):
        cumulative += weight
        if cumulative >= threshold:
            return individual
    return population[-1]


def crossover(parent_a: State, parent_b: State, rng: random.Random, crossover_rate: float) -> State:
    # Uniform crossover: tiap paket mewarisi posisi+orientasi dari salah satu induk.
    if rng.random() >= crossover_rate:
        return parent_a.copy()

    donor = {p.package.id: p for p in parent_b.placements}
    child_placements = []
    for placement in parent_a.placements:
        source = placement if rng.random() < 0.5 else donor[placement.package.id]
        child_placements.append(source.copy())
    return State(trucks=list(parent_a.trucks), placements=child_placements)


def mutate(individual: State, rng: random.Random, mutation_rate: float, max_moves: int = 3) -> State:
    child = individual.copy()
    if rng.random() >= mutation_rate:
        return child
    for _ in range(rng.randint(1, max_moves)):
        child = make_random_neighbor(child, rng)
    return child


def best_by_fitness(population: List[State], fitnesses: List[float]) -> State:
    return population[max(range(len(population)), key=lambda i: fitnesses[i])]


def best_valid(population: List[State], objective: str = "value") -> Optional[State]:
    champion: Optional[State] = None
    for individual in population:
        if not is_feasible(individual)[0]:
            continue
        if champion is None or state_score(individual, objective) > state_score(champion, objective):
            champion = individual.copy()
    return champion


def run(
    trucks: List[Truck],
    packages: List[Package],
    rng: random.Random,
    population_size: int = 50,
    max_iter: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elitisme: int = 2,
    mutation_moves: int = 3,
    objective: str = "value",
) -> dict:
    normalize_objective(objective)
    start = time.perf_counter()

    population = make_population(trucks, packages, rng, population_size)

    history: List[dict] = []
    best_individual: Optional[State] = None
    best_valid_state: Optional[State] = None
    initial_state: Optional[State] = None

    for iteration in range(max_iter):
        fitnesses = [objective_function(ind, objective) for ind in population]
        values = [ind.total_value() if is_feasible(ind)[0] else 0 for ind in population]

        generation_best = best_by_fitness(population, fitnesses)
        if best_individual is None or (
            objective_function(generation_best, objective) > objective_function(best_individual, objective)
        ):
            best_individual = generation_best.copy()
        if iteration == 0:
            initial_state = generation_best.copy()

        valid_champion = best_valid(population, objective)
        if valid_champion is not None and (
            best_valid_state is None
            or state_score(valid_champion, objective) > state_score(best_valid_state, objective)
        ):
            best_valid_state = valid_champion.copy()

        history.append({
            "iter": iteration,
            "max_objective": max(fitnesses),
            "avg_objective": sum(fitnesses) / len(fitnesses),
            "best_value": max(values),
            "avg_value": sum(values) / len(values),
            "feasible_count": sum(1 for ind in population if is_feasible(ind)[0]),
        })

        if iteration == max_iter - 1:
            break

        elite_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:elitisme]
        next_population = [population[i].copy() for i in elite_indices]

        while len(next_population) < population_size:
            parent_a = roulette(population, fitnesses, rng)
            parent_b = roulette(population, fitnesses, rng)
            child = crossover(parent_a, parent_b, rng, crossover_rate)
            child = mutate(child, rng, mutation_rate, mutation_moves)
            next_population.append(child)

        population = next_population

    duration = time.perf_counter() - start
    final_fitnesses = [objective_function(ind, objective) for ind in population]
    best_individual = best_by_fitness(population, final_fitnesses).copy()
    if initial_state is None:
        initial_state = best_individual.copy()

    return {
        "algorithm": "ga",
        "params": {
            "population_size": population_size,
            "max_iter": max_iter,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elitisme": elitisme,
            "mutation_moves": mutation_moves,
            "objective": objective,
        },
        "initial_state": initial_state,
        "final_state": best_individual,
        "best_valid_state": best_valid_state,
        "final_objective": objective_function(best_individual, objective),
        "best_valid_value": best_valid_state.total_value() if best_valid_state else None,
        "population_size": population_size,
        "iterations": max_iter,
        "duration_s": duration,
        "history": history,
        "extra": {},
    }


CONFIG_KEYS = [
    "crossover_rate",
    "mutation_rate",
    "elitisme",
]


def params_from_config(config: dict) -> dict:
    section = config["genetic_algorithm"]
    missing = [k for k in CONFIG_KEYS if k not in section]
    if missing:
        raise KeyError(f"Config genetic_algorithm tidak punya key: {missing}")
    params = {k: section[k] for k in CONFIG_KEYS}
    params["objective"] = config.get("objective", "value")
    return params


def experiment_matrix(ga_config: dict) -> List[Tuple[str, int, int]]:
    matrix: List[Tuple[str, int, int]] = []
    for it in ga_config["eksperimen_A"]["iterasi"]:
        matrix.append(("eksperimen_A", ga_config["eksperimen_A"]["populasi"], it))
    for pop in ga_config["eksperimen_B"]["populasi"]:
        matrix.append(("eksperimen_B", pop, ga_config["eksperimen_B"]["iterasi"]))
    return matrix


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
        "objective": result["params"]["objective"],
        "final_objective": result["final_objective"],
        "best_valid_value": result["best_valid_value"],
        "population_size": result["population_size"],
        "iterations": result["iterations"],
        "duration_s": result["duration_s"],
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out
