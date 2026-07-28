"""Typed, serializable molecular-graph records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class Atom:
    atom_id: int
    atomic_number: int
    symbol: str
    formal_charge: int = 0
    radical_electrons: int = 0
    stereo: str | None = None

    def __post_init__(self) -> None:
        if self.atom_id < 0:
            raise ValueError("atom_id must be non-negative")
        if self.atomic_number <= 0:
            raise ValueError("atomic_number must be positive")
        if self.stereo not in (None, "R", "S"):
            raise ValueError(f"unsupported atom stereo: {self.stereo!r}")


@dataclass(frozen=True, order=True)
class Edge:
    atom_i: int
    atom_j: int
    bond_order: float
    stereo: str | None = None

    def __post_init__(self) -> None:
        if self.atom_i < 0 or self.atom_j < 0:
            raise ValueError("edge atom IDs must be non-negative")
        if self.atom_i >= self.atom_j:
            raise ValueError("edges require atom_i < atom_j")
        if self.bond_order <= 0:
            raise ValueError("stored edges require bond_order > 0")
        if self.stereo not in (None, "E", "Z"):
            raise ValueError(f"unsupported bond stereo: {self.stereo!r}")

    @property
    def pair(self) -> tuple[int, int]:
        return (self.atom_i, self.atom_j)


@dataclass(frozen=True)
class MolecularState:
    atoms: tuple[Atom, ...]
    edges: tuple[Edge, ...]
    components: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        ids = tuple(atom.atom_id for atom in self.atoms)
        if ids != tuple(range(len(self.atoms))):
            raise ValueError("atoms must use contiguous zero-based IDs")
        pairs = [edge.pair for edge in self.edges]
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate edge pair")
        valid_ids = set(ids)
        component_ids = [atom_id for comp in self.components for atom_id in comp]
        if set(component_ids) != valid_ids or len(component_ids) != len(valid_ids):
            raise ValueError("components must partition all atoms exactly once")


@dataclass(frozen=True)
class ReactionRecord:
    reaction_id: str
    reactant: MolecularState
    product: MolecularState
    ts_edges: tuple[Edge, ...]
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {self.split!r}")
        if self.reactant.atoms != self.product.atoms:
            # State attributes may change, so compare only the identity columns.
            left = [(a.atom_id, a.atomic_number, a.symbol) for a in self.reactant.atoms]
            right = [(a.atom_id, a.atomic_number, a.symbol) for a in self.product.atoms]
            if left != right:
                raise ValueError("reactant/product atom identity tables differ")

    @property
    def atom_count(self) -> int:
        return len(self.reactant.atoms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_edges(edges: Iterable[Edge]) -> tuple[Edge, ...]:
    """Return edges sorted by atom pair after validating uniqueness."""

    result = tuple(sorted(edges, key=lambda edge: edge.pair))
    pairs = [edge.pair for edge in result]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate edge pair")
    return result
