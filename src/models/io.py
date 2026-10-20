from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import List, Tuple

from .model import Truck, Package, State, Placement


def load_dataset(path: str | Path) -> Tuple[List[Truck], List[Package]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    packages_data = data.get("packages", data.get("Packages", data.get("items", [])))

    trucks_raw = data.get("trucks", data.get("Trucks"))
    if trucks_raw is None:
        single = data.get("truck", data.get("Truck", data.get("truck_dimensions", {})))
        if not single and "dimensions" in data and "max_capacity" in data:
            single = data
        trucks_raw = [single]

    trucks: List[Truck] = [Truck.from_dict(entry) for entry in trucks_raw]

    packages: List[Package] = []
    for entry in packages_data:
        packages.append(Package.from_dict(entry))
    return trucks, packages


def save_state(path: str | Path, state: State) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)


def load_state(path: str | Path, trucks: List[Truck], packages: List[Package]) -> State:
    pkg_map = {pkg.id: pkg for pkg in packages}
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    # data bisa berupa full to_dict output
    placements_raw = data.get("placements", [])

    placements: List[Placement] = []
    for entry in placements_raw:
        pid = entry.get("package_id", entry.get("id"))
        pkg = pkg_map.get(pid)
        if pkg is None:
            continue
        pos = entry.get("position")
        if pos is not None:
            pos = tuple(int(x) for x in pos)
        ori = int(entry.get("orientation_idx", 0))
        truck_idx = int(entry.get("truck_idx", 0))
        placements.append(Placement(package=pkg, position=pos, orientation_idx=ori, truck_idx=truck_idx))
    return State(trucks=list(trucks), placements=placements)


def save_history_csv(path: str | Path, history: List[dict], fieldnames: List[str] | None = None) -> None:

    if not history:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(history[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

