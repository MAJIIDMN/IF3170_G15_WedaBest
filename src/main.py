from __future__ import annotations

import random
from pathlib import Path

from algoritma import GA, SA
from models.geometry import is_feasible
from models.io import load_config, load_dataset
from models.operators import make_initial_state

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "data" / "experiments.json"


def _ask(prompt: str, default) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or str(default)


def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw else default


def choose_algorithm() -> str:
    print("Pilih algoritma:")
    print("  1. Simulated Annealing")
    print("  2. Genetic Algorithm")
    while True:
        choice = input("Pilihan (1/2): ").strip()
        if choice == "1":
            return "sa"
        if choice == "2":
            return "ga"
        print("Masukkan 1 atau 2.")


def main() -> None:
    config = load_config(CONFIG_PATH)
    dataset_path = Path(_ask("Path dataset", config["dataset_utama"]))
    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path
    dataset = dataset_path.stem
    trucks, packages = load_dataset(dataset_path)
    print(f"Jumlah truk: {len(trucks)} | jumlah paket: {len(packages)}")

    output_root = BASE_DIR.parent / config["output_dir"]

    algorithm = choose_algorithm()
    seed = _ask_int("Seed", config["seeds"][0])
    rng = random.Random(seed)

    if algorithm == "sa":
        params = SA.params_from_config(config)
        initial_state = make_initial_state(trucks, packages, rng)
        result = SA.run(initial_state, rng, **params)
        out = SA.save_result(result, output_root / "sa" / dataset / f"seed{seed}")
    else:
        params = GA.params_from_config(config)
        population_size = _ask_int("Jumlah populasi", 50)
        max_iter = _ask_int("Banyak iterasi", 100)
        result = GA.run(
            trucks,
            packages,
            rng,
            population_size=population_size,
            max_iter=max_iter,
            **params,
        )
        out = GA.save_result(
            result,
            output_root / "ga" / dataset
            / f"pop{population_size}_iter{max_iter}" / f"seed{seed}",
        )

    feasible = is_feasible(result["final_state"])[0]
    print(f"Value awal  : {result['initial_state'].total_value()}")
    print(f"Value akhir : {result['final_state'].total_value()} (feasible: {feasible})")
    print(f"Best valid  : {result['best_valid_value']}")
    print(f"Durasi      : {result['duration_s']:.2f}s")
    print(f"Output      : {out}")


if __name__ == "__main__":
    main()
