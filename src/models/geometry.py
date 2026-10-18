from __future__ import annotations

from typing import List, Tuple, Dict
from .model import State, Placement


def is_inside_bounds(placement: Placement, truck_dims: Tuple[int, int, int]) -> bool:
    if not placement.is_inside:
        return True
    x, y, z = placement.position
    w, l, h = placement.effective_dims
    W, L, H = truck_dims
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + l > L or z + h > H:
        return False
    return True


def placements_overlap(a: Placement, b: Placement) -> bool:
    if not a.is_inside or not b.is_inside:
        return False
    if a.package.id == b.package.id:
        return False
    ax1, ay1, az1, ax2, ay2, az2 = a.aabb()
    bx1, by1, bz1, bx2, by2, bz2 = b.aabb()

    # Tidak overlap jika terpisah di salah satu sumbu
    if ax2 <= bx1 or bx2 <= ax1:
        return False
    if ay2 <= by1 or by2 <= ay1:
        return False
    if az2 <= bz1 or bz2 <= az1:
        return False
    return True


def check_bounds(state: State) -> List[str]:
    violations = []
    for p in state.inside_placements():
        truck = state.truck_of(p)
        if not is_inside_bounds(p, truck.dimensions):
            violations.append(
                f"Package {p.package.id} menembus dinding: pos={p.position} dims={p.effective_dims} truck={truck.dimensions}"
            )
    return violations


def check_no_overlap(state: State) -> List[str]:
    violations = []
    for truck_idx in range(len(state.trucks)):
        inside = state.placements_on(truck_idx)
        for i in range(len(inside)):
            for j in range(i + 1, len(inside)):
                if placements_overlap(inside[i], inside[j]):
                    violations.append(
                        f"Overlap: {inside[i].package.id} {inside[i].aabb()} <-> {inside[j].package.id} {inside[j].aabb()}"
                    )
    return violations


def overweight_score(state: State) -> float:
    score = 0.0
    for truck_idx, truck in enumerate(state.trucks):
        weight = state.truck_weight(truck_idx)
        if weight > truck.max_capacity:
            score += 1 + (weight - truck.max_capacity) / truck.max_capacity
    return score


def check_weight(state: State) -> List[str]:
    violations = []
    for truck_idx, truck in enumerate(state.trucks):
        total = state.truck_weight(truck_idx)
        if total > truck.max_capacity:
            violations.append(
                f"Overweight truk {truck_idx}: total {total} > capacity {truck.max_capacity}"
            )
    return violations


def check_fragile_support(state: State) -> List[str]:
    violations = []
    for truck_idx in range(len(state.trucks)):
        inside = state.placements_on(truck_idx)
        for j in inside:
            if not j.package.is_fragile:
                continue
            jx1, jy1, jz1, jx2, jy2, jz2 = j.aabb()
            for i in inside:
                if i.package.id == j.package.id:
                    continue
                ix1, iy1, iz1, ix2, iy2, iz2 = i.aabb()
                # i tepat di atas j?
                if iz1 == jz2:
                    overlap_x = not (ix2 <= jx1 or jx2 <= ix1)
                    overlap_y = not (iy2 <= jy1 or jy2 <= iy1)
                    if overlap_x and overlap_y:
                        violations.append(
                            f"Fragile support: {j.package.id} (fragile) menopang {i.package.id}"
                        )
    return violations


def check_support(state: State) -> List[str]:
    violations = []
    for truck_idx in range(len(state.trucks)):
        inside = state.placements_on(truck_idx)
        for p in inside:
            x, y, z = p.position
            w, l, h = p.effective_dims
            if z == 0:
                continue  # di alas truk -> selalu support

            # Cek apakah ada minimal 1 titik (px, py) di alas yang memiliki support
            has_support = False
            for px in range(x, x + w):
                if has_support:
                    break
                for py in range(y, y + l):
                    for other in inside:
                        if other.package.id == p.package.id:
                            continue
                        if other.package.is_fragile:
                            continue  # fragile tidak bisa jadi support
                        ox, oy, oz = other.position  # type: ignore
                        ow, ol, oh = other.effective_dims
                        # atap other tepat di bawah alas p ?
                        if oz + oh == z and ox <= px < ox + ow and oy <= py < oy + ol:
                            has_support = True
                            break
                    if has_support:
                        break

            if not has_support:
                violations.append(
                    f"Floating: {p.package.id} di {p.position} dims={p.effective_dims} tidak memiliki support"
                )
    return violations


def validate(state: State) -> Dict[str, List[str]]:
    return {
        "bounds": check_bounds(state),
        "overlap": check_no_overlap(state),
        "weight": check_weight(state),
        "fragile": check_fragile_support(state),
        "support": check_support(state),
    }


def is_feasible(state: State) -> Tuple[bool, Dict[str, List[str]]]:
    violations = validate(state)
    feasible = all(len(v) == 0 for v in violations.values())
    return feasible, violations


def all_violations(violations: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for cat, msgs in violations.items():
        for m in msgs:
            out.append(f"[{cat}] {m}")
    return out


def objective_value(state: State, feasible_only: bool = True) -> int:
    if feasible_only:
        feasible, _ = is_feasible(state)
        if not feasible:
            return 0
    return state.total_value()


def evaluate(state: State) -> Tuple[int, bool, Dict[str, List[str]]]:
    feasible, violations = is_feasible(state)
    value = state.total_value() if feasible else 0
    return value, feasible, violations


def evaluate_with_penalty(
    state: State,
    overlap_penalty: int = 50, #placeholder doang
    weight_penalty: int = 10, #placeholder doang
    floating_penalty: int = 30, #placeholder doang
    fragile_penalty: int = 100, #placeholder doang
) -> int:
    base = state.total_value()
    violations = validate(state)
    penalty = (
        len(violations["overlap"]) * overlap_penalty
        + len(violations["weight"]) * weight_penalty
        + len(violations["support"]) * floating_penalty
        + len(violations["fragile"]) * fragile_penalty
        + len(violations["bounds"]) * overlap_penalty
    )
    return max(0, base - penalty)
