from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.geometry import is_feasible
from models.io import load_dataset, load_state
from models.model import Package, State, Truck

ALGO_NAMES = {"sa": "Simulated Annealing", "ga": "Genetic Algorithm", "hc": "Hill-Climbing"}
CMAP = plt.get_cmap("tab20")


def _parse(value: str):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value else number


def read_history(path: str | Path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def read_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def detect_algorithm(history: List[dict], summary: dict) -> str:
    # HC memakai "hill_climbing"; SA/GA dikenali dari kolom history.
    if summary.get("algorithm") == "hill_climbing":
        return "hc"
    columns = history[0].keys() if history else ()
    if "temperature" in columns:
        return "sa"
    if "max_objective" in columns:
        return "ga"
    return "hc"


def _title(algorithm: str, summary: dict, run_dir: Path) -> str:
    name = ALGO_NAMES[algorithm]
    if summary.get("variant"):
        name += f" ({summary['variant']})"
    info = [run_dir.name]
    if "duration_s" in summary:
        info.append(f"durasi {summary['duration_s']:.2f}s")
    if "final_objective" in summary:
        info.append(f"objective akhir {summary['final_objective']:.2f}")
    return f"{name}\n{' | '.join(info)}"


def plot_objective(history: List[dict], algorithm: str, path: Path, title: str = "") -> None:
    if algorithm == "ga":
        _plot_objective_ga(history, path, title)
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if algorithm == "sa":
        iters = [h["iter"] for h in history]
        ax.plot(iters, [h["objective"] for h in history], linewidth=0.8, label="objective (current)")
        ax.plot(iters, [h["value"] for h in history], linewidth=0.8, alpha=0.7, label="value (current)")
    elif "global_iter" in history[0]:
        # Random restart: satu segmen per restart + nilai terbaik global
        restarts: dict = {}
        for h in history:
            restarts.setdefault(h["restart"], []).append(h)
        for number, rows in restarts.items():
            ax.plot([h["global_iter"] for h in rows], [h["objective"] for h in rows], color="C0",
                    linewidth=1, marker=".", label="objective (current, per restart)" if number == 1 else None)
            if number > 1:
                ax.axvline(rows[0]["global_iter"], color="grey", linewidth=0.6, alpha=0.5)
        ax.plot([h["global_iter"] for h in history], [h["global_best_value"] for h in history],
                color="C1", linestyle="--", label="value terbaik global")
    else:
        iters = [h["iter"] for h in history]
        ax.plot(iters, [h["objective"] for h in history], linewidth=1, marker=".", label="objective (current)")

    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Objective")
    ax.set_title(title or "Objective vs iterasi")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_objective_ga(history: List[dict], path: Path, title: str) -> None:
    iters = [h["iter"] for h in history]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    top.plot(iters, [h["max_objective"] for h in history], label="objective maksimum")
    top.plot(iters, [h["avg_objective"] for h in history], label="objective rata-rata")
    top.set_ylabel("Objective (dengan penalti)")
    top.set_title(title or "GA: objective vs iterasi")
    top.grid(alpha=0.3)
    top.legend()

    bottom.plot(iters, [h["best_value"] for h in history], label="value terbaik (individu valid)")
    bottom.plot(iters, [h["avg_value"] for h in history], label="value rata-rata (tidak valid = 0)")
    bottom.set_xlabel("Iterasi (generasi)")
    bottom.set_ylabel("Value")
    bottom.grid(alpha=0.3)
    bottom.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_accept_prob(history: List[dict], path: Path, title: str = "") -> None:
    # Semua iterasi diplot, termasuk delta_e > 0 yang menghasilkan e^(dE/T) > 1.
    # Nilai > 1 berarti move selalu diterima; garis 1.0 jadi batas referensi.
    points = [h for h in history if h.get("accept_prob") is not None]
    accepted = [h for h in points if h.get("accepted")]
    rejected = [h for h in points if not h.get("accepted")]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter([h["iter"] for h in accepted], [h["accept_prob"] for h in accepted], s=3, label="diterima")
    ax.scatter([h["iter"] for h in rejected], [h["accept_prob"] for h in rejected], s=3, label="ditolak")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="batas always-accept (1.0)")
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("e^(ΔE/T)")
    ax.set_title(title or "SA: e^(ΔE/T) vs iterasi")
    ax.grid(alpha=0.3)
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_state(state: State, path: Path, title: str = "") -> None:
    n_trucks = len(state.trucks)
    color_of = {p.package.id: CMAP(i % 20) for i, p in enumerate(state.placements)}
    fig = plt.figure(figsize=(7 * n_trucks, 6.5))

    for k, truck in enumerate(state.trucks):
        ax = fig.add_subplot(1, n_trucks, k + 1, projection="3d")
        ax.computed_zorder = False  # supaya label tidak tertutup balok
        on_truck = state.placements_on(k)
        for p in on_truck:
            x, y, z = p.position
            w, l, h = p.effective_dims
            edge = "red" if p.package.is_fragile else "black"
            ax.bar3d(x, y, z, w, l, h, color=color_of[p.package.id], edgecolor=edge,
                     linewidth=1.5 if p.package.is_fragile else 0.6, alpha=0.65)
        for p in on_truck:
            x, y, z = p.position
            w, l, h = p.effective_dims
            ax.text(x + w / 2, y + l / 2, z + h / 2, p.package.id, ha="center", va="center", fontsize=8,
                    weight="bold", zorder=100, bbox=dict(facecolor="white", alpha=0.8, pad=1, linewidth=0))

        W, L, H = truck.dimensions
        ax.set_xlim(0, W)
        ax.set_ylim(0, L)
        ax.set_zlim(0, H)
        ax.set_box_aspect((W, L, H))
        ax.set_xlabel("x (w)")
        ax.set_ylabel("y (l)")
        ax.set_zlabel("z (h)")
        value = sum(p.package.value for p in on_truck)
        ax.set_title(f"Truk {k + 1}: value {value}, berat {state.truck_weight(k)}/{truck.max_capacity}", fontsize=10)

    feasible = is_feasible(state)[0]
    outside = state.outside_placements()
    outside_text = ", ".join(f"{p.package.id} (v={p.package.value})" for p in outside) or "-"
    header = (
        f"Total value: {state.total_value()} | Valid: {'ya' if feasible else 'TIDAK'} | "
        f"di dalam {len(state.inside_placements())}, di luar {len(outside)}"
    )
    fig.suptitle(f"{title}\n{header}" if title else header)
    fig.text(0.5, 0.02, f"Di luar truk: {outside_text}   (garis merah = fragile)", ha="center", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def guess_dataset(run_dir: Path) -> Optional[Path]:
    # output[/objective_eta]/<algo>/<dataset>/... -> src/data/<dataset>.json
    parts = run_dir.resolve().parts
    for i, part in enumerate(parts[:-1]):
        if part in ALGO_NAMES:
            candidate = SRC_DIR / "data" / f"{parts[i + 1]}.json"
            if candidate.exists():
                return candidate
    return None


def plot_run(
    run_dir: str | Path,
    trucks: Optional[List[Truck]] = None,
    packages: Optional[List[Package]] = None,
) -> List[Path]:
    run_dir = Path(run_dir)
    history = read_history(run_dir / "history.csv")
    summary = read_summary(run_dir)
    algorithm = detect_algorithm(history, summary)
    title = _title(algorithm, summary, run_dir)
    made: List[Path] = []

    if history:
        plot_objective(history, algorithm, run_dir / "objective.png", title)
        made.append(run_dir / "objective.png")
    if algorithm == "sa" and history:
        stuck = summary.get("stuck_count")
        sa_title = title + (f" | stuck {stuck}x" if stuck is not None else "")
        plot_accept_prob(history, run_dir / "accept_prob.png", sa_title)
        made.append(run_dir / "accept_prob.png")

    if trucks is None or packages is None:
        dataset = guess_dataset(run_dir)
        if dataset is None:
            print(f"  (lewati gambar state {run_dir}: dataset tidak diketahui, pakai --dataset)")
            return made
        trucks, packages = load_dataset(dataset)

    for name, label in (("initial", "state awal"), ("final", "state akhir")):
        state_path = run_dir / f"{name}_state.json"
        if state_path.exists():
            state = load_state(state_path, trucks, packages)
            draw_state(state, run_dir / f"{name}_state.png", f"{ALGO_NAMES[algorithm]}: {label}")
            made.append(run_dir / f"{name}_state.png")
    return made


def find_runs(root: Path) -> List[Path]:
    if (root / "history.csv").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("history.csv"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot hasil run SA/GA/HC")
    parser.add_argument("folder", help="folder satu run, atau folder induk (semua run di bawahnya)")
    parser.add_argument("--dataset", help="path dataset JSON (default: ditebak dari path output)")
    args = parser.parse_args()

    trucks = packages = None
    if args.dataset:
        trucks, packages = load_dataset(args.dataset)
    runs = find_runs(Path(args.folder))
    if not runs:
        sys.exit(f"Tidak ada history.csv di {args.folder}")
    for run in runs:
        files = plot_run(run, trucks, packages)
        print(f"{run}: {', '.join(f.name for f in files)}")
