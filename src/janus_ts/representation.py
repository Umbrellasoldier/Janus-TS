"""Deterministic MoleCode-TS/v1 serialization.

Chemical parsing belongs in preprocessing.  This module intentionally accepts
only validated :mod:`janus_ts.schema` records and performs no RDKit work, so it
is safe to call from data-loader workers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .constants import REPRESENTATION_VERSION, SEED
from .discretization import format_bond_order
from .schema import Atom, Edge, MolecularState, ReactionRecord, canonical_edges


def _atom_identity_line(atom: Atom) -> str:
    return f"a{atom.atom_id} [element={atom.symbol},Z={atom.atomic_number}]"


def _atom_attribute_line(atom: Atom) -> str | None:
    fields: list[str] = []
    if atom.formal_charge:
        fields.append(f"charge={atom.formal_charge:+d}")
    if atom.radical_electrons:
        fields.append(f"radical={atom.radical_electrons}")
    if atom.stereo is not None:
        fields.append(f"stereo={atom.stereo}")
    if not fields:
        return None
    return f"a{atom.atom_id} [{','.join(fields)}]"


def _edge_line(edge: Edge, *, include_stereo: bool) -> str:
    fields = [f"bo={format_bond_order(edge.bond_order)}"]
    if include_stereo and edge.stereo is not None:
        fields.append(f"stereo={edge.stereo}")
    return (
        f"a{edge.atom_i} --[{','.join(fields)}]-- a{edge.atom_j}"
    )


def _block(name: str, lines: Iterable[str]) -> list[str]:
    return [f"<{name}>", *lines, f"</{name}>"]


def _ordered_edges(
    edges: tuple[Edge, ...],
    *,
    reaction_id: str,
    section: str,
    training: bool,
    epoch: int,
    seed: int,
) -> tuple[Edge, ...]:
    canonical = canonical_edges(edges)
    if not training:
        return canonical

    # A stateless permutation makes the same (seed, epoch, reaction, section)
    # reproducible across worker counts, restarts, ranks, and resume points.
    def key(edge: Edge) -> bytes:
        payload = (
            f"{seed}\0{epoch}\0{reaction_id}\0{section}\0"
            f"{edge.atom_i}\0{edge.atom_j}"
        )
        return hashlib.sha256(payload.encode("utf-8")).digest()

    return tuple(sorted(canonical, key=key))


def _component_lines(state: MolecularState) -> list[str]:
    components = sorted(tuple(sorted(comp)) for comp in state.components)
    components.sort(key=lambda comp: (comp[0], comp))
    return [
        f"c{index}: {' '.join(f'a{atom_id}' for atom_id in component)}"
        for index, component in enumerate(components)
    ]


def _atom_change_lines(record: ReactionRecord) -> list[str]:
    lines: list[str] = []
    for reactant, product in zip(
        record.reactant.atoms, record.product.atoms, strict=True
    ):
        fields: list[str] = []
        if reactant.formal_charge != product.formal_charge:
            fields.append(
                f"charge={reactant.formal_charge:+d}->{product.formal_charge:+d}"
            )
        if reactant.radical_electrons != product.radical_electrons:
            fields.append(
                "radical="
                f"{reactant.radical_electrons}->{product.radical_electrons}"
            )
        if fields:
            lines.append(f"a{reactant.atom_id} [{','.join(fields)}]")
    return lines


def _edge_change_lines(record: ReactionRecord) -> list[str]:
    reactant = {edge.pair: edge for edge in record.reactant.edges}
    product = {edge.pair: edge for edge in record.product.edges}
    lines: list[str] = []
    for atom_i, atom_j in sorted(set(reactant) | set(product)):
        left = reactant.get((atom_i, atom_j))
        right = product.get((atom_i, atom_j))
        left_bo = format_bond_order(left.bond_order) if left else "0"
        right_bo = format_bond_order(right.bond_order) if right else "0"
        left_stereo = left.stereo if left and left.stereo else "none"
        right_stereo = right.stereo if right and right.stereo else "none"
        if (left_bo, left_stereo) == (right_bo, right_stereo):
            continue
        lines.append(
            f"a{atom_i} --[R:bo={left_bo},stereo={left_stereo};"
            f"P:bo={right_bo},stereo={right_stereo}]-- a{atom_j}"
        )
    return lines


def serialize_input(
    record: ReactionRecord,
    *,
    training: bool = False,
    epoch: int = 0,
    seed: int = SEED,
) -> str:
    """Serialize one input graph pair.

    Reactant and product edge lines are permuted once per logical epoch during
    training.  Every other section is canonical.  Evaluation is fully
    canonical regardless of ``epoch``.
    """

    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    lines: list[str] = [REPRESENTATION_VERSION]
    lines += _block("ATOMS", (_atom_identity_line(a) for a in record.reactant.atoms))
    lines += _block(
        "REACTANT_ATOM_ATTRIBUTES",
        filter(None, (_atom_attribute_line(a) for a in record.reactant.atoms)),
    )
    lines += _block("REACTANT_COMPONENTS", _component_lines(record.reactant))
    reactant_edges = _ordered_edges(
        record.reactant.edges,
        reaction_id=record.reaction_id,
        section="REACTANT_EDGES",
        training=training,
        epoch=epoch,
        seed=seed,
    )
    lines += _block(
        "REACTANT_EDGES",
        (_edge_line(edge, include_stereo=True) for edge in reactant_edges),
    )
    lines += _block(
        "PRODUCT_ATOM_ATTRIBUTES",
        filter(None, (_atom_attribute_line(a) for a in record.product.atoms)),
    )
    lines += _block("PRODUCT_COMPONENTS", _component_lines(record.product))
    product_edges = _ordered_edges(
        record.product.edges,
        reaction_id=record.reaction_id,
        section="PRODUCT_EDGES",
        training=training,
        epoch=epoch,
        seed=seed,
    )
    lines += _block(
        "PRODUCT_EDGES",
        (_edge_line(edge, include_stereo=True) for edge in product_edges),
    )
    lines += _block("ATOM_CHANGES", _atom_change_lines(record))
    lines += _block("EDGE_CHANGES", _edge_change_lines(record))
    return "\n".join(lines)


def serialize_target(edges: Iterable[Edge]) -> str:
    """Serialize the canonical transition-state target block."""

    canonical = canonical_edges(edges)
    return "\n".join(
        _block(
            "TS_EDGES",
            (_edge_line(edge, include_stereo=False) for edge in canonical),
        )
    )
