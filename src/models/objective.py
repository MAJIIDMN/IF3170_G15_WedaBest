from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .model import Package, State

# "value": OF utama dari spesifikasi, total value paket di dalam truk.
# "eta"  : OF bonus, value tiap paket diberi bobot urgensi (1 + 1/eta), jadi paket
#          yang harus cepat sampai (eta kecil) lebih diutamakan. Tidak menambah
#          variabel baru, hanya memakai atribut eta yang sudah ada di dataset.
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
    """Nilai yang dimaksimalkan (tanpa penalti) dari paket-paket di dalam truk."""
    if objective == "value":
        return state.total_value()
    return _total((p.package for p in state.inside_placements()), objective)


def max_score(state: State, objective: str = "value") -> float:
    """Batas atas state_score: semua paket masuk. Dipakai untuk lambda penalti."""
    return _total((p.package for p in state.placements), objective)


def output_root_for(output_root: Path, objective: str = "value") -> Path:
    """Hasil run dengan OF bonus disimpan terpisah supaya tidak menimpa hasil OF utama."""
    if objective == "value":
        return Path(output_root)
    return Path(output_root) / f"objective_{objective}"
