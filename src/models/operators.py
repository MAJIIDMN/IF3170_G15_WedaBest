from __future__ import annotations

import random
from typing import List, Tuple, Optional

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
    return new_state


def relocate_package(
    state: State,
    idx: int,
    new_position: Optional[Tuple[int, int, int]],
) -> State:
    new_state = state.copy()
    new_state.placements[idx].position = new_position
    return new_state


def relocate_random(
    state: State,
    idx: int,
    rng: random.Random,
) -> State:
    dims = state.placements[idx].effective_dims
    if rng.random() < 0.15: #placeholder presentase
        return relocate_package(state, idx, None)
    pos = random_position(dims, state.truck, rng)
    # Jika tidak muat sama sekali, keluarkan
    if pos is None:
        pos = None
    return relocate_package(state, idx, pos)


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


def generate_random_neighbor(state: State, rng: random.Random) -> State:
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


def generate_neighbors(
    state: State,
    rng: random.Random,
    count: int = 20,
    feasible_only: bool = False,
    max_tries: int = 200,
) -> List[State]:
    neighbors: List[State] = []
    tries = 0
    while len(neighbors) < count and tries < max_tries * count:
        cand = generate_random_neighbor(state, rng)
        tries += 1
        if feasible_only:
            feas, _ = is_feasible(cand)
            if not feas:
                continue
        neighbors.append(cand)
    return neighbors


def generate_initial_state(
    truck: Truck,
    packages: List[Package],
    rng: Optional[random.Random] = None,
    max_attempts_per_package: int = 100,
    ensure_feasible: bool = True,
) -> State:
    if rng is None:
        rng = random.Random()

    order = packages[:]
    rng.shuffle(order)

    state = State(truck=truck, placements=[])
    for pkg in order:
        # Coba orientasi random + posisi random
        placed: Optional[Placement] = None
        for _ in range(max_attempts_per_package):
            ori = rng.randrange(6)
            dims = get_effective_dimensions(pkg.dimensions, ori)
            pos = random_position(dims, truck, rng)
            if pos is None:
                continue
            cand_placement = Placement(package=pkg, position=pos, orientation_idx=ori)
            temp_state = State(truck=truck, placements=state.placements + [cand_placement])
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


def generate_random_infeasible_state(
    truck: Truck,
    packages: List[Package],
    rng: Optional[random.Random] = None,
) -> State:
    if rng is None:
        rng = random.Random()
    placements: List[Placement] = []
    for pkg in packages:
        ori = rng.randrange(6)
        dims = get_effective_dimensions(pkg.dimensions, ori)
        if rng.random() < 0.5:
            pos = random_position(dims, truck, rng)
        else:
            pos = None
        placements.append(Placement(package=pkg, position=pos, orientation_idx=ori))
    return State(truck=truck, placements=placements)
