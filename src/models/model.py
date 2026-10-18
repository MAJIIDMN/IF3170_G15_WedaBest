from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

ORIENTATIONS: List[Tuple[int, int, int]] = [
    (0, 1, 2),  # 0: (w, l, h)
    (0, 2, 1),  # 1: (w, h, l)
    (1, 0, 2),  # 2: (l, w, h)
    (1, 2, 0),  # 3: (l, h, w)
    (2, 0, 1),  # 4: (h, w, l)
    (2, 1, 0),  # 5: (h, l, w)
]

ORIENTATION_LABELS: Dict[int, str] = {
    0: "{w:x, l:y, h:z}",
    1: "{w:x, l:z, h:y}",
    2: "{w:y, l:x, h:z}",
    3: "{w:z, l:x, h:y}",
    4: "{w:y, l:z, h:x}",
    5: "{w:z, l:y, h:x}",
}

_PERM_TO_IDX: Dict[Tuple[int, int, int], int] = {p: i for i, p in enumerate(ORIENTATIONS)}


def get_effective_dimensions(
    dimensions: Tuple[int, int, int], orientation_idx: int
) -> Tuple[int, int, int]:
    perm = ORIENTATIONS[orientation_idx]
    return (dimensions[perm[0]], dimensions[perm[1]], dimensions[perm[2]])


def rotate_orientation_idx(current_idx: int, axis: str) -> int:
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis harus x/y/z, dapat {axis}")
    perm = list(ORIENTATIONS[current_idx])
    if axis == "x":
        perm[1], perm[2] = perm[2], perm[1]
    elif axis == "y":
        perm[0], perm[2] = perm[2], perm[0]
    else:  # z
        perm[0], perm[1] = perm[1], perm[0]
    return _PERM_TO_IDX[tuple(perm)]

@dataclass
class Package:
    id: str
    dimensions: Tuple[int, int, int]  # (w, l, h)
    value: int
    weight: int
    is_fragile: bool = False
    eta: Optional[int] = None

    def effective_dims(self, orientation_idx: int = 0) -> Tuple[int, int, int]:
        return get_effective_dimensions(self.dimensions, orientation_idx)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dimensions": {"w": self.dimensions[0], "l": self.dimensions[1], "h": self.dimensions[2]},
            "value": self.value,
            "weight": self.weight,
            "is_fragile": self.is_fragile,
            "eta": self.eta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Package":
        raw_dim = d.get("dimensions", d.get("Dimensions", [1, 1, 1]))
        if isinstance(raw_dim, dict):
            dims = (int(raw_dim.get("w", raw_dim.get("width", 1))),
                    int(raw_dim.get("l", raw_dim.get("length", raw_dim.get("d", 1)))),
                    int(raw_dim.get("h", raw_dim.get("height", 1))))
        else:
            dims = tuple(int(x) for x in raw_dim)  # type: ignore
            if len(dims) != 3:
                raise ValueError(f"dimensions harus 3 elemen, dapat {dims}")
            dims = (dims[0], dims[1], dims[2])

        is_frag = d.get("is_fragile", d.get("IsFragile", d.get("isFragile", False)))
        return Package(
            id=str(d.get("id", d.get("Id", d.get("ID", "")))),
            dimensions=dims,
            value=int(d.get("value", d.get("Value", 0))),
            weight=int(d.get("weight", d.get("Weight", 0))),
            is_fragile=bool(is_frag),
            eta=d.get("eta", d.get("ETA", None)),
        )


@dataclass
class Truck:

    dimensions: Tuple[int, int, int]  # (w, l, h)
    max_capacity: int

    @property
    def w(self) -> int:
        return self.dimensions[0]

    @property
    def l(self) -> int:
        return self.dimensions[1]

    @property
    def h(self) -> int:
        return self.dimensions[2]

    def to_dict(self) -> dict:
        return {
            "dimensions": {"w": self.w, "l": self.l, "h": self.h},
            "max_capacity": self.max_capacity,
        }

    @staticmethod
    def from_dict(d: dict) -> "Truck":
        raw_dim = d.get("dimensions", d.get("Dimensions", [10, 10, 10]))
        if isinstance(raw_dim, dict):
            dims = (int(raw_dim.get("w", raw_dim.get("width", 10))),
                    int(raw_dim.get("l", raw_dim.get("length", raw_dim.get("d", 10)))),
                    int(raw_dim.get("h", raw_dim.get("height", 10))))
        else:
            dims = tuple(int(x) for x in raw_dim)  # type: ignore
            dims = (dims[0], dims[1], dims[2])
        cap = d.get("max_capacity", d.get("MaxCapacity", d.get("capacity", 0)))
        return Truck(dimensions=dims, max_capacity=int(cap))


@dataclass
class Placement:

    package: Package
    position: Optional[Tuple[int, int, int]] = None
    orientation_idx: int = 0
    truck_idx: int = 0

    @property
    def is_inside(self) -> bool:
        return self.position is not None

    @property
    def effective_dims(self) -> Tuple[int, int, int]:
        return get_effective_dimensions(self.package.dimensions, self.orientation_idx)

    @property
    def w(self) -> int:
        return self.effective_dims[0]

    @property
    def l(self) -> int:
        return self.effective_dims[1]

    @property
    def h(self) -> int:
        return self.effective_dims[2]

    @property
    def x(self) -> Optional[int]:
        return self.position[0] if self.position else None

    @property
    def y(self) -> Optional[int]:
        return self.position[1] if self.position else None

    @property
    def z(self) -> Optional[int]:
        return self.position[2] if self.position else None

    def aabb(self) -> Optional[Tuple[int, int, int, int, int, int]]:
        if self.position is None:
            return None
        x, y, z = self.position
        return (x, y, z, x + self.w, y + self.l, z + self.h)

    def orientation_label(self) -> str:
        return ORIENTATION_LABELS[self.orientation_idx]

    def copy(self) -> "Placement":
        return Placement(
            package=self.package,
            position=self.position,
            orientation_idx=self.orientation_idx,
            truck_idx=self.truck_idx,
        )

    def to_dict(self) -> dict:
        return {
            "package_id": self.package.id,
            "position": list(self.position) if self.position else None,
            "orientation_idx": self.orientation_idx,
            "orientation_label": self.orientation_label(),
            "truck_idx": self.truck_idx,
            "effective_dimensions": {"w": self.w, "l": self.l, "h": self.h},
            "is_inside": self.is_inside,
        }


@dataclass
class State:

    trucks: List[Truck] = field(default_factory=list)
    placements: List[Placement] = field(default_factory=list)

    @property
    def truck(self) -> Truck:
        return self.trucks[0]

    def truck_of(self, placement: Placement) -> Truck:
        return self.trucks[placement.truck_idx]

    def get_placement(self, package_id: str) -> Optional[Placement]:
        for p in self.placements:
            if p.package.id == package_id:
                return p
        return None

    def inside_placements(self) -> List[Placement]:
        return [p for p in self.placements if p.is_inside]

    def outside_placements(self) -> List[Placement]:
        return [p for p in self.placements if not p.is_inside]

    def placements_on(self, truck_idx: int) -> List[Placement]:
        return [p for p in self.placements if p.is_inside and p.truck_idx == truck_idx]

    def total_value(self) -> int:
        return sum(p.package.value for p in self.inside_placements())

    def total_weight(self) -> int:
        return sum(p.package.weight for p in self.inside_placements())

    def truck_weight(self, truck_idx: int) -> int:
        return sum(p.package.weight for p in self.placements_on(truck_idx))

    def copy(self) -> "State":
        """Copy placements (package ref tetap shared) dan salin list truk (truk immutable)."""
        return State(
            trucks=list(self.trucks),
            placements=[p.copy() for p in self.placements],
        )

    def to_dict(self) -> dict:
        return {
            "trucks": [t.to_dict() for t in self.trucks],
            "placements": [p.to_dict() for p in self.placements],
            "total_value": self.total_value(),
            "total_weight": self.total_weight(),
            "inside_count": len(self.inside_placements()),
            "outside_count": len(self.outside_placements()),
        }

    @staticmethod
    def from_packages(trucks: List[Truck], packages: List[Package]) -> "State":
        if not isinstance(trucks, list):
            trucks = [trucks]
        return State(
            trucks=list(trucks),
            placements=[Placement(package=p, position=None, orientation_idx=0) for p in packages],
        )

    def __len__(self) -> int:
        return len(self.placements)
