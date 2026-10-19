"""Jalankan SA, GA, atau HC lalu simpan "replay" (state per langkah) untuk visualisasi/viewer3d.html.

Kode algoritma tidak diubah. Snapshot state diambil dengan "spy": fungsi di modul algoritma
diganti sementara dengan pembungkus yang mencatat state, lalu dikembalikan lagi.
    SA -> SA.make_valid_neighbor(current, ...)  dipanggil sekali per iterasi
          frame baru setiap kali current berganti (= move diterima)
    GA -> GA.best_by_fitness(populasi, fitness)  dipanggil sekali per generasi
          frame = individu terbaik tiap generasi
    HC -> iter_all_feasible_neighbors / make_random_neighbor
          frame baru setiap kali state asal neighbor berganti (= current berganti)

Pakai (dari folder src/):
    python -m visualisasi.export_replay                              -> SA, dataset_utama, semua seed
    python -m visualisasi.export_replay --algo ga --pop 20 --iter 200
    python -m visualisasi.export_replay --algo hc --variant sideways
    python -m visualisasi.export_replay multi_truck_case.json --seed 2
    python -m visualisasi.export_replay --algo sa --param cooling_rate=0.995

Hasil: output/replay/<algo>[-<varian>]_<dataset>_seed<s>.json -> buka di visualisasi/viewer3d.html
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import random
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import models.operators as operators
from models.io import load_config, load_dataset
from models.model import State

MAX_FRAMES = 3000


def snapshot(state):
    # id -> [x, y, z, orientasi, truck_idx], atau None kalau di luar truk
    return {
        p.package.id: (list(p.position) + [p.orientation_idx, p.truck_idx]) if p.position is not None else None
        for p in state.placements
    }


def thin(items, limit):
    if len(items) <= limit:
        return items
    step = len(items) / (limit - 1)
    picked = [items[int(k * step)] for k in range(limit - 1)]
    return picked + [items[-1]]


class Spy:
    """Ganti module.name dengan pembungkus selama blok `with`, lalu kembalikan."""

    def __init__(self, targets):
        self.targets = targets   # list (module, nama, pembungkus)
        self.originals = []

    def __enter__(self):
        for module, name, make_wrapper in self.targets:
            original = getattr(module, name)
            self.originals.append((module, name, original))
            setattr(module, name, make_wrapper(original))

    def __exit__(self, *exc):
        for module, name, original in self.originals:
            setattr(module, name, original)


def call_run(algo, trucks, packages, rng, params, variant):
    # GA: run(trucks, packages, rng, ...); SA: run(initial_state, rng, ...);
    # HC: run(initial_state, rng, variant, ...)
    names = list(inspect.signature(algo.run).parameters)
    if names[0] in ("trucks", "truck"):
        return algo.run(trucks, packages, rng, **params)
    s0 = operators.make_initial_state(trucks, packages, rng)
    if "variant" in names:
        return algo.run(s0, rng, variant, **params)
    return algo.run(s0, rng, **params)


def record_sa(algo, trucks, packages, rng, params, variant):
    frames, last, counter = [], [None], [0]

    def wrap(original):
        def spy(state, *args, **kwargs):
            if state is not last[0]:
                frames.append({"iter": counter[0], "placements": snapshot(state)})
                last[0] = state
            counter[0] += 1
            return original(state, *args, **kwargs)
        return spy

    with Spy([(algo, "make_valid_neighbor", wrap)]):
        result = call_run(algo, trucks, packages, rng, params, variant)
    if result["final_state"] is not last[0]:
        frames.append({"iter": result.get("iterations", counter[0]), "placements": snapshot(result["final_state"])})
    return result, frames


def record_ga(algo, trucks, packages, rng, params, variant):
    frames = []

    def wrap(original):
        def spy(population, fitnesses):
            best = original(population, fitnesses)
            frames.append({"iter": len(frames), "placements": snapshot(best)})
            return best
        return spy

    with Spy([(algo, "best_by_fitness", wrap)]):
        result = call_run(algo, trucks, packages, rng, params, variant)
    # Panggilan terakhir (setelah loop) memakai populasi yang sama dengan generasi terakhir
    if len(frames) >= 2 and frames[-1]["placements"] == frames[-2]["placements"]:
        frames.pop()
    return result, frames


def record_hc(algo, trucks, packages, rng, params, variant):
    frames, last = [], [None]

    def wrap(original):
        def spy(state, *args, **kwargs):
            if isinstance(state, State) and state is not last[0]:
                frames.append({"iter": len(frames), "placements": snapshot(state)})
                last[0] = state
            return original(state, *args, **kwargs)
        return spy

    names = ["iter_all_feasible_neighbors", "make_random_neighbor"]
    targets = [(algo, n, wrap) for n in names if hasattr(algo, n)]
    with Spy(targets):
        result = call_run(algo, trucks, packages, rng, params, variant)
    # final_state HC berupa salinan, jadi bandingkan isinya, bukan objeknya
    final = snapshot(result["final_state"])
    if not frames or frames[-1]["placements"] != final:
        frames.append({"iter": len(frames), "placements": final})
    return result, frames


RECORDERS = {"sa": ("SA", record_sa), "ga": ("GA", record_ga), "hc": ("HC", record_hc)}


def chart_history(algorithm, history):
    # Bentuk seragam untuk grafik di viewer: iter, value, feasible, temperature
    rows = []
    for h in history:
        if algorithm == "sa":
            rows.append({"iter": h["iter"], "value": h["value"], "feasible": h["feasible"],
                         "temperature": h["temperature"]})
        elif algorithm == "ga":
            rows.append({"iter": h["iter"], "value": h["best_value"], "feasible": True})
        else:
            rows.append({"iter": h.get("global_iter", h.get("iter")), "value": h["value"], "feasible": True})
    return thin(rows, MAX_FRAMES)


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def export_replays(config, dataset_path, algo_name, seeds, variant=None, extra_params=None):
    """Jalankan algo_name ('sa'/'ga'/'hc') untuk tiap seed dan simpan replay. Kembalikan daftar file."""
    module_name, recorder = RECORDERS[algo_name]
    algo = importlib.import_module(f"algoritma.{module_name}")
    variant = variant if algo_name == "hc" else None
    if algo_name == "hc":
        params = algo.params_from_config(config, variant)
    else:
        params = algo.params_from_config(config)
    params.update(extra_params or {})

    dataset_path = Path(dataset_path)
    trucks, packages = load_dataset(dataset_path)
    dataset_json = json.loads(dataset_path.read_text(encoding="utf-8"))

    out_dir = SRC_DIR.parent / config["output_dir"] / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for seed in seeds:
        rng = random.Random(seed)
        result, frames = recorder(algo, trucks, packages, rng, params, variant)
        best = result.get("best_valid_state")
        replay = {
            "format": "packing-replay",
            "algorithm": algo_name,
            "variant": variant,
            "dataset_name": dataset_path.name,
            "dataset": dataset_json,
            "seed": seed,
            "params": params,
            "iterations": result.get("iterations"),
            "duration_s": result.get("duration_s"),
            "best_valid_value": result.get("best_valid_value"),
            "best_valid": {"placements": snapshot(best)} if best is not None else None,
            "frames": thin(frames, MAX_FRAMES),
            "history": chart_history(algo_name, result["history"]),
        }
        name = algo_name + (f"-{variant}" if variant else "")
        out = out_dir / f"{name}_{dataset_path.stem}_seed{seed}.json"
        out.write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
        written.append(out)
        print(
            f"{name.upper()} seed {seed}: value {result['initial_state'].total_value()} -> "
            f"best_valid {result.get('best_valid_value')}, {len(frames)} frame, "
            f"durasi {result.get('duration_s', 0):.1f}s  |  {out}"
        )
    return written


def main():
    parser = argparse.ArgumentParser(description="Buat file replay untuk visualisasi/viewer3d.html")
    parser.add_argument("dataset", nargs="?", help="nama file di src/data/ atau path lengkap")
    parser.add_argument("--algo", choices=RECORDERS, default="sa")
    parser.add_argument("--variant", default="steepest_ascent",
                        help="HC: steepest_ascent, sideways, stochastic, atau random_restart")
    parser.add_argument("--seed", type=int, help="default: semua seed di experiments.json")
    parser.add_argument("--pop", type=int, default=50, help="GA: jumlah populasi")
    parser.add_argument("--iter", type=int, default=100, help="GA: banyak iterasi/generasi")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="tambahan/ubah parameter run(), boleh berkali-kali")
    args = parser.parse_args()

    config = load_config(SRC_DIR / "data" / "experiments.json")
    extra = {}
    if args.algo == "ga":
        extra.update(population_size=args.pop, max_iter=args.iter)
    for item in args.param:
        key, _, value = item.partition("=")
        extra[key] = parse_value(value)

    dataset_file = args.dataset or Path(config["dataset_utama"]).name
    dataset_path = Path(dataset_file)
    if not dataset_path.exists():
        dataset_path = SRC_DIR / "data" / dataset_file
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    export_replays(config, dataset_path, args.algo, seeds, args.variant, extra)


if __name__ == "__main__":
    main()
