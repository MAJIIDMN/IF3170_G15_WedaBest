from __future__ import annotations

import random
from typing import Iterator, List, Tuple, Optional

from .model import State, Placement, Truck, Package, rotate_orientation_idx, get_effective_dimensions
from .geometry import is_feasible

def random_position(
    effective_dims: Tuple[int, int, int],
    truck: Truck,
    rng: random.Random,
) -> Optional[Tuple[int, int, int]]:
    w, l, h = effective_dims
    W, L, H = truck.dimensions
    if w > W or l > L or h > H:
        return None
    x = rng.randint(0, W - w)
    y = rng.randint(0, L - l)
    z = rng.randint(0, H - h)
    return (x, y, z)


def _pick_two_indices(n: int, rng: random.Random) -> Tuple[int, int]:
    i = rng.randrange(n)
    j = rng.randrange(n - 1)
    if j >= i:
        j += 1
    return i, j


def swap_packages(state: State, i: int, j: int) -> State:

    new_state = state.copy()
    pi = new_state.placements[i]
    pj = new_state.placements[j]
    pi.position, pj.position = pj.position, pi.position
    pi.orientation_idx, pj.orientation_idx = pj.orientation_idx, pi.orientation_idx
    pi.truck_idx, pj.truck_idx = pj.truck_idx, pi.truck_idx
    return new_state


def relocate_package(
    state: State,
    idx: int,
    new_position: Optional[Tuple[int, int, int]],
    truck_idx: Optional[int] = None,
) -> State:
    new_state = state.copy()
    placement = new_state.placements[idx]
    placement.position = new_position
    if truck_idx is not None:
        placement.truck_idx = truck_idx
    return new_state


def relocate_random(
    state: State,
    idx: int,
    rng: random.Random,
) -> State:
    new_state = state.copy()
    placement = new_state.placements[idx]
    if rng.random() < 0.15: #placeholder presentase
        placement.position = None
        return new_state
    truck_idx = rng.randrange(len(new_state.trucks))
    placement.truck_idx = truck_idx
    placement.position = random_position(placement.effective_dims, new_state.trucks[truck_idx], rng)
    return new_state


def rotate_package(state: State, idx: int, axis: str) -> State:
    if axis not in ("x", "y", "z"):
        raise ValueError("axis harus x/y/z")
    new_state = state.copy()
    p = new_state.placements[idx]
    new_idx = rotate_orientation_idx(p.orientation_idx, axis)
    p.orientation_idx = new_idx
    return new_state


def rotate_random(state: State, idx: int, rng: random.Random) -> State:
    axis = rng.choice(["x", "y", "z"])
    return rotate_package(state, idx, axis)


def make_random_neighbor(state: State, rng: random.Random) -> State:
    n = len(state.placements)
    if n == 0:
        return state.copy()
    move = rng.choice(["swap", "relocate", "rotate"])
    if move == "swap" and n >= 2:
        i, j = _pick_two_indices(n, rng)
        return swap_packages(state, i, j)
    elif move == "relocate":
        idx = rng.randrange(n)
        return relocate_random(state, idx, rng)
    else:  # rotate
        idx = rng.randrange(n)
        return rotate_random(state, idx, rng)


def make_neighbors(
    state: State,
    rng: random.Random,
    count: int = 20,
    feasible_only: bool = False,
    max_tries: int = 200,
) -> List[State]:
    neighbors: List[State] = []
    tries = 0
    while len(neighbors) < count and tries < max_tries * count:
        cand = make_random_neighbor(state, rng)
        tries += 1
        if feasible_only:
            feas, _ = is_feasible(cand)
            if not feas:
                continue
        neighbors.append(cand)
    return neighbors


def state_signature(state: State) -> Tuple[tuple, ...]:
    return tuple(sorted(
        (
            p.package.id,
            p.position,
            p.orientation_idx,
            p.truck_idx,
        )
        for p in state.placements
    ))


def iter_all_feasible_neighbors(state: State) -> Iterator[State]:
    # Membangkitkan seluruh successor feasible dari satu operasi yang diizinkan.
    # Operasinya adalah menukar dua paket, memindahkan satu paket ke
    # koordinat integer lain (termasuk ke luar truk), dan merotasi satu paket
    # 90 derajat terhadap sumbu x/y/z. State duplikat dan state asal tidak
    # dikembalikan.

    seen = {state_signature(state)}

    def is_new_feasible(candidate: State) -> bool:
        signature = state_signature(candidate)
        if signature in seen:
            return False
        seen.add(signature)
        return is_feasible(candidate)[0]

    n = len(state.placements)

    # Menukar dua paket, baik paket di dalam maupun di luar truk
    for i in range(n):
        for j in range(i + 1, n):
            candidate = swap_packages(state, i, j)
            if is_new_feasible(candidate):
                yield candidate

    # Memindahkan sebuah paket ke koordinat lain atau ke luar truk
    for idx, placement in enumerate(state.placements):
        if placement.is_inside:
            candidate = relocate_package(state, idx, None)
            if is_new_feasible(candidate):
                yield candidate

        w, l, h = placement.effective_dims
        for truck_idx, truck in enumerate(state.trucks):
            if w > truck.w or l > truck.l or h > truck.h:
                continue
            for x in range(truck.w - w + 1):
                for y in range(truck.l - l + 1):
                    for z in range(truck.h - h + 1):
                        position = (x, y, z)
                        if placement.position == position and placement.truck_idx == truck_idx:
                            continue
                        candidate = relocate_package(state, idx, position, truck_idx)
                        if is_new_feasible(candidate):
                            yield candidate

    # Merotasi sebuah paket 90 derajat terhadap salah satu sumbu
    for idx in range(n):
        for axis in ("x", "y", "z"):
            candidate = rotate_package(state, idx, axis)
            if is_new_feasible(candidate):
                yield candidate


def make_all_feasible_neighbors(state: State) -> List[State]:
    """Versi list dari iter_all_feasible_neighbors untuk pemanggil yang membutuhkannya."""
    return list(iter_all_feasible_neighbors(state))


def make_initial_state(
    trucks: List[Truck],
    packages: List[Package],
    rng: Optional[random.Random] = None,
    max_attempts_per_package: int = 100,
    ensure_feasible: bool = True,
) -> State:
    if not isinstance(trucks, list):
        trucks = [trucks]
    if rng is None:
        rng = random.Random()

    order = packages[:]
    rng.shuffle(order)

    state = State(trucks=trucks, placements=[])
    for pkg in order:
        placed: Optional[Placement] = None
        for _ in range(max_attempts_per_package):
            ori = rng.randrange(6)
            dims = get_effective_dimensions(pkg.dimensions, ori)
            truck_idx = rng.randrange(len(trucks))
            pos = random_position(dims, trucks[truck_idx], rng)
            if pos is None:
                continue
            cand_placement = Placement(package=pkg, position=pos, orientation_idx=ori, truck_idx=truck_idx)
            temp_state = State(trucks=trucks, placements=state.placements + [cand_placement])
            feas, violations = is_feasible(temp_state)
            if feas:
                placed = cand_placement
                break
            if not ensure_feasible:
                if not violations["bounds"] and not violations["overlap"] and not violations["weight"]:
                    placed = cand_placement
                    break

        if placed is None:
            # Biarkan di luar
            state.placements.append(Placement(package=pkg, position=None, orientation_idx=rng.randrange(6)))
        else:
            state.placements.append(placed)
    return state


def make_random_infeasible_state(
    trucks: List[Truck],
    packages: List[Package],
    rng: Optional[random.Random] = None,
) -> State:
    if not isinstance(trucks, list):
        trucks = [trucks]
    if rng is None:
        rng = random.Random()
    placements: List[Placement] = []
    for pkg in packages:
        ori = rng.randrange(6)
        truck_idx = rng.randrange(len(trucks))
        dims = get_effective_dimensions(pkg.dimensions, ori)
        if rng.random() < 0.5:
            pos = random_position(dims, trucks[truck_idx], rng)
        else:
            pos = None
        placements.append(Placement(package=pkg, position=pos, orientation_idx=ori, truck_idx=truck_idx))
    return State(trucks=trucks, placements=placements)
