from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

from algoritma import GA, HC, SA
from models.geometry import is_feasible
from models.io import load_config, load_dataset
from models.objective import OBJECTIVES, normalize_objective, output_root_for
from models.operators import make_initial_state
from visualisasi.export_replay import export_replays
from visualisasi.plots import find_runs, plot_run

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = DATA_DIR / "experiments.json"

HC_VARIANTS = {
    "1": "steepest_ascent",
    "2": "sideways",
    "3": "stochastic",
    "4": "random_restart",
}


def _ask(prompt: str, default) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or str(default)


def _ask_int(prompt: str, default: int, minimum: int = 0) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Masukkan bilangan bulat.")
            continue
        if value < minimum:
            print(f"Nilai minimal {minimum}.")
            continue
        return value


def _choose(title: str, options: Dict[str, str]) -> str:
    print(title)
    for key, label in options.items():
        print(f"  {key}. {label}")
    while True:
        choice = input(f"Pilihan ({'/'.join(options)}): ").strip()
        if choice in options:
            return choice
        print(f"Masukkan salah satu dari: {', '.join(options)}.")


def _ask_seeds(config: dict) -> List[int]:
    default = ",".join(str(s) for s in config["seeds"])
    while True:
        raw = _ask("Seed (pisahkan dengan koma)", default)
        try:
            return [int(part) for part in raw.split(",") if part.strip()]
        except ValueError:
            print("Contoh input: 1 atau 1,2,3")


def choose_dataset(config: dict) -> Path:
    files = sorted(p for p in DATA_DIR.glob("*.json") if p.name != CONFIG_PATH.name)
    default = Path(config["dataset_utama"]).name
    print("Dataset di src/data/:")
    for i, path in enumerate(files, start=1):
        print(f"  {i:2d}. {path.name}")
    while True:
        raw = _ask("Nomor dataset atau path file", default)
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            return files[int(raw) - 1]
        for candidate in (Path(raw), BASE_DIR / raw, DATA_DIR / raw):
            if candidate.is_file():
                return candidate.resolve()
        print(f"Dataset '{raw}' tidak ditemukan. Masukkan nomor 1-{len(files)} atau path file .json.")


def choose_objective(default: str) -> str:
    print("Pilih objective function:")
    print("  value : total value paket di dalam truk (OF utama)")
    print("  eta   : value x (1 + 1/eta), paket ber-ETA kecil diutamakan (OF bonus)")
    while True:
        choice = _ask("Objective", default)
        if choice in OBJECTIVES:
            return choice
        print(f"Masukkan salah satu dari: {', '.join(OBJECTIVES)}.")


def choose_hc_variant() -> str:
    options = {
        "1": "Steepest-Ascent",
        "2": "Sideways Move",
        "3": "Stochastic",
        "4": "Random Restart",
    }
    return HC_VARIANTS[_choose("Pilih varian Hill Climbing:", options)]


def print_result(result: dict, out: Path, objective: str, plots: List[Path]) -> None:
    feasible = is_feasible(result["final_state"])[0]
    print(f"Value awal  : {result['initial_state'].total_value()}")
    print(f"Value akhir : {result['final_state'].total_value()} (feasible: {feasible})")
    print(f"Objective   : {result['final_objective']:.2f} ({objective})")
    print(f"Best valid  : {result['best_valid_value']}")
    print(f"Durasi      : {result['duration_s']:.2f}s")
    print(f"Output      : {out}")
    print(f"Plot        : {', '.join(p.name for p in plots)}")


def run_single(ctx: dict, algorithm: str) -> None:
    config, trucks, packages = ctx["config"], ctx["trucks"], ctx["packages"]
    base = ctx["output_root"] / algorithm / ctx["dataset"]
    seed = _ask_int("Seed", config["seeds"][0])
    rng = random.Random(seed)

    if algorithm == "sa":
        initial_state = make_initial_state(trucks, packages, rng)
        result = SA.run(initial_state, rng, **SA.params_from_config(config))
        out = SA.save_result(result, base / f"seed{seed}")
    elif algorithm == "ga":
        population_size = _ask_int("Jumlah populasi", 50, minimum=2)
        max_iter = _ask_int("Banyak iterasi", 100, minimum=1)
        result = GA.run(trucks, packages, rng, population_size=population_size, max_iter=max_iter,
                        **GA.params_from_config(config))
        out = GA.save_result(result, base / f"pop{population_size}_iter{max_iter}" / f"seed{seed}")
    else:
        variant = choose_hc_variant()
        initial_state = make_initial_state(trucks, packages, rng)
        result = HC.run(initial_state, rng, variant, **HC.params_from_config(config, variant))
        result["seed"] = seed
        out = HC.save_result(result, base / variant / f"seed{seed}")

    plots = plot_run(out, trucks, packages)
    print_result(result, out, config["objective"], plots)


def experiment_hc(ctx: dict) -> None:
    trucks, packages = ctx["trucks"], ctx["packages"]
    records = HC.run_all_experiments(ctx["config"], trucks, packages, ctx["output_root"], ctx["dataset"])
    for record in records:
        plot_run(record["output"], trucks, packages)
        hc_result = record["result"]
        print(
            f"HC {hc_result['variant']} seed={hc_result['seed']}: "
            f"{hc_result['initial_state'].total_value()} -> "
            f"{hc_result['final_state'].total_value()}, "
            f"iterasi={hc_result['iterations']}, "
            f"durasi={hc_result['duration_s']:.2f}s"
        )
    print(f"Output HC   : {ctx['output_root'] / 'hc' / ctx['dataset']}")


def experiment_sa(ctx: dict) -> None:
    config, trucks, packages = ctx["config"], ctx["trucks"], ctx["packages"]
    params = SA.params_from_config(config)
    for k, seed in enumerate(config["seeds"], start=1):
        rng = random.Random(seed)
        initial_state = make_initial_state(trucks, packages, rng)
        result = SA.run(initial_state, rng, **params)
        out = SA.save_result(
            result, ctx["output_root"] / "sa" / ctx["dataset"] / "default" / f"run{k}_seed{seed}"
        )
        plot_run(out, trucks, packages)
        print(
            f"SA seed={seed}: {initial_state.total_value()} -> best_valid {result['best_valid_value']}, "
            f"iterasi={result['iterations']}, "
            f"stuck={result['extra']['stuck_count']} "
            f"({result['extra']['stuck_rate']:.2f}/1000 iter), "
            f"durasi={result['duration_s']:.2f}s"
        )
    print(f"Output SA   : {ctx['output_root'] / 'sa' / ctx['dataset'] / 'default'}")


def experiment_ga(ctx: dict) -> None:
    # Eksperimen A: populasi tetap, iterasi divariasikan; eksperimen B: sebaliknya
    config, trucks, packages = ctx["config"], ctx["trucks"], ctx["packages"]
    params = GA.params_from_config(config)
    for name, population_size, max_iter in GA.experiment_matrix(config["genetic_algorithm"]):
        for k, seed in enumerate(config["seeds"], start=1):
            result = GA.run(trucks, packages, random.Random(seed), population_size=population_size,
                            max_iter=max_iter, **params)
            out = GA.save_result(
                result,
                ctx["output_root"] / "ga" / ctx["dataset"] / name
                / f"pop{population_size}_iter{max_iter}" / f"run{k}_seed{seed}",
            )
            plot_run(out, trucks, packages)
            print(
                f"GA {name} pop={population_size} iter={max_iter} seed={seed}: "
                f"best_valid {result['best_valid_value']}, durasi={result['duration_s']:.2f}s"
            )
    print(f"Output GA   : {ctx['output_root'] / 'ga' / ctx['dataset']}")


def run_experiments(ctx: dict) -> None:
    config = ctx["config"]
    options = {
        "1": "Semua (HC 4 varian, SA, GA eksperimen A & B)",
        "2": "Hill Climbing, 4 varian",
        "3": "Simulated Annealing",
        "4": "Genetic Algorithm, eksperimen A & B",
    }
    choice = _choose(f"Eksperimen untuk laporan (seed {config['seeds']}):", options)
    if choice in ("1", "2"):
        experiment_hc(ctx)
    if choice in ("1", "3"):
        experiment_sa(ctx)
    if choice in ("1", "4"):
        experiment_ga(ctx)


def make_replay(ctx: dict) -> None:
    options = {"1": "Simulated Annealing", "2": "Genetic Algorithm", "3": "Hill Climbing"}
    algorithm = {"1": "sa", "2": "ga", "3": "hc"}[_choose("Replay untuk algoritma:", options)]
    variant: Optional[str] = None
    extra: dict = {}
    if algorithm == "hc":
        variant = choose_hc_variant()
    elif algorithm == "ga":
        extra["population_size"] = _ask_int("Jumlah populasi", 50, minimum=2)
        extra["max_iter"] = _ask_int("Banyak iterasi", 100, minimum=1)
    seeds = _ask_seeds(ctx["config"])
    files = export_replays(ctx["config"], ctx["dataset_path"], algorithm, seeds, variant, extra)
    print(f"{len(files)} file replay dibuat. Buka src/visualisasi/viewer3d.html lalu muat file tersebut.")


def replot(ctx: dict) -> None:
    root = ctx["output_root"]
    folders = [p for p in (root / a / ctx["dataset"] for a in ("hc", "sa", "ga")) if p.exists()]
    runs = [run for folder in folders for run in find_runs(folder)]
    if not runs:
        print(f"Belum ada hasil run untuk dataset {ctx['dataset']} di {root}.")
        return
    for run in runs:
        plot_run(run, ctx["trucks"], ctx["packages"])
    print(f"Plot dibuat ulang untuk {len(runs)} run dataset {ctx['dataset']}.")


def setup(config: dict) -> dict:
    dataset_path = choose_dataset(config)
    trucks, packages = load_dataset(dataset_path)
    print(f"Dataset {dataset_path.name}: {len(trucks)} truk, {len(packages)} paket")
    config["objective"] = choose_objective(normalize_objective(config.get("objective", "value")))
    return {
        "config": config,
        "dataset_path": dataset_path,
        "dataset": dataset_path.stem,
        "trucks": trucks,
        "packages": packages,
        "output_root": output_root_for(BASE_DIR.parent / config["output_dir"], config["objective"]),
    }


def main() -> None:
    config = load_config(CONFIG_PATH)
    ctx = setup(config)
    menu = {
        "1": "Simulated Annealing (satu run)",
        "2": "Genetic Algorithm (satu run)",
        "3": "Hill Climbing (satu run)",
        "4": "Eksperimen untuk laporan (semua seed)",
        "5": "Buat file replay untuk viewer 3D",
        "6": "Buat ulang plot untuk hasil dataset ini",
        "7": "Ganti dataset / objective",
        "0": "Keluar",
    }
    while True:
        print()
        choice = _choose(f"Menu [dataset: {ctx['dataset_path'].name}, objective: {config['objective']}]", menu)
        if choice == "0":
            return
        if choice in ("1", "2", "3"):
            run_single(ctx, {"1": "sa", "2": "ga", "3": "hc"}[choice])
        elif choice == "4":
            run_experiments(ctx)
        elif choice == "5":
            make_replay(ctx)
        elif choice == "6":
            replot(ctx)
        else:
            ctx = setup(config)


if __name__ == "__main__":
    main()
