from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .model import Package, State

# "value": total value paket di dalam truk.
# "eta": value x (1 + 1/eta), paket ber-ETA kecil diutamakan.
OBJECTIVES = ("value", "eta")


def normalize_objective(objective: str) -> str:
    if objective not in OBJECTIVES:
        raise ValueError(f"Objective tidak dikenal: {objective} (pilihan: {', '.join(OBJECTIVES)})")
    return objective


def eta_weight(package: Package) -> float:
    # eta kosong atau tidak positif dianggap tidak mendesak (bobot 1)
    if package.eta is None or package.eta <= 0:
        return 1.0
    return 1.0 + 1.0 / package.eta


def package_score(package: Package, objective: str = "value") -> float:
    if normalize_objective(objective) == "eta":
        return package.value * eta_weight(package)
    return package.value


def _total(packages: Iterable[Package], objective: str) -> float:
    return sum(package_score(pkg, objective) for pkg in packages)


def state_score(state: State, objective: str = "value") -> float:
    if objective == "value":
        return state.total_value()
    return _total((p.package for p in state.inside_placements()), objective)


def max_score(state: State, objective: str = "value") -> float:
    return _total((p.package for p in state.placements), objective)


def output_root_for(output_root: Path, objective: str = "value") -> Path:
    if objective == "value":
        return Path(output_root)
    return Path(output_root) / f"objective_{objective}"
