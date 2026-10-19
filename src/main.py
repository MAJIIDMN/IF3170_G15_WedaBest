from __future__ import annotations

import random
from pathlib import Path

from algoritma import GA, HC, SA
from models.geometry import is_feasible
from models.io import load_config, load_dataset
from models.objective import OBJECTIVES, normalize_objective, output_root_for
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
    print("  3. Hill Climbing")
    while True:
        choice = input("Pilihan (1/2/3): ").strip()
        if choice == "1":
            return "sa"
        if choice == "2":
            return "ga"
        if choice == "3":
            return "hc"
        print("Masukkan 1, 2, atau 3.")


def choose_hc_variant() -> str:
    print("Pilih varian Hill Climbing:")
    print("  1. Steepest-Ascent")
    print("  2. Sideways Move")
    print("  3. Stochastic")
    print("  4. Random Restart")
    print("  5. Semua varian (seluruh seed eksperimen)")
    choices = {
        "1": "steepest_ascent",
        "2": "sideways",
        "3": "stochastic",
        "4": "random_restart",
        "5": "all",
    }
    while True:
        choice = input("Pilihan (1/2/3/4/5): ").strip()
        if choice in choices:
            return choices[choice]
        print("Masukkan angka 1 sampai 5.")


def choose_objective(default: str) -> str:
    print("Pilih objective function:")
    print("  value : total value paket di dalam truk (OF utama)")
    print("  eta   : value x (1 + 1/eta), paket ber-ETA kecil diutamakan (OF bonus)")
    while True:
        choice = _ask("Objective", default)
        if choice in OBJECTIVES:
            return choice
        print(f"Masukkan salah satu dari: {', '.join(OBJECTIVES)}.")


def main() -> None:
    config = load_config(CONFIG_PATH)
    dataset_path = Path(_ask("Path dataset", config["dataset_utama"]))
    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path
    dataset = dataset_path.stem
    trucks, packages = load_dataset(dataset_path)
    print(f"Jumlah truk: {len(trucks)} | jumlah paket: {len(packages)}")

    config["objective"] = choose_objective(normalize_objective(config.get("objective", "value")))
    output_root = output_root_for(BASE_DIR.parent / config["output_dir"], config["objective"])

    algorithm = choose_algorithm()
    if algorithm == "sa":
        seed = _ask_int("Seed", config["seeds"][0])
        rng = random.Random(seed)
        params = SA.params_from_config(config)
        initial_state = make_initial_state(trucks, packages, rng)
        result = SA.run(initial_state, rng, **params)
        out = SA.save_result(result, output_root / "sa" / dataset / f"seed{seed}")
    elif algorithm == "ga":
        seed = _ask_int("Seed", config["seeds"][0])
        rng = random.Random(seed)
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
    else:
        variant = choose_hc_variant()
        if variant == "all":
            records = HC.run_all_experiments(
                config,
                trucks,
                packages,
                output_root,
                dataset,
            )
            print(f"Selesai menjalankan {len(records)} eksperimen HC.")
            for record in records:
                hc_result = record["result"]
                print(
                    f"{hc_result['variant']} seed={hc_result['seed']}: "
                    f"{hc_result['initial_state'].total_value()} -> "
                    f"{hc_result['final_state'].total_value()}, "
                    f"iterasi={hc_result['iterations']}, "
                    f"durasi={hc_result['duration_s']:.2f}s"
                )
            print(f"Output      : {output_root / 'hc' / dataset}")
            return

        seed = _ask_int("Seed", config["seeds"][0])
        rng = random.Random(seed)
        params = HC.params_from_config(config, variant)
        initial_state = make_initial_state(trucks, packages, rng)
        result = HC.run(initial_state, rng, variant, **params)
        result["seed"] = seed
        out = HC.save_result(
            result,
            output_root / "hc" / dataset / variant / f"seed{seed}",
        )

    feasible = is_feasible(result["final_state"])[0]
    print(f"Value awal  : {result['initial_state'].total_value()}")
    print(f"Value akhir : {result['final_state'].total_value()} (feasible: {feasible})")
    print(f"Objective   : {result['final_objective']:.2f} ({config['objective']})")
    print(f"Best valid  : {result['best_valid_value']}")
    print(f"Durasi      : {result['duration_s']:.2f}s")
    print(f"Output      : {out}")


if __name__ == "__main__":
    main()
