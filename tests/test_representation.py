from __future__ import annotations

from janus_ts.representation import serialize_input, serialize_target
from janus_ts.schema import Atom, Edge, MolecularState, ReactionRecord


def _record() -> ReactionRecord:
    reactant_atoms = (
        Atom(0, 6, "C", stereo="R"),
        Atom(1, 8, "O", formal_charge=-1),
        Atom(2, 1, "H"),
    )
    product_atoms = (
        Atom(0, 6, "C", stereo="S"),
        Atom(1, 8, "O"),
        Atom(2, 1, "H"),
    )
    reactant = MolecularState(
        reactant_atoms,
        (Edge(0, 1, 1.0, "E"), Edge(1, 2, 1.0)),
        ((0, 1, 2),),
    )
    product = MolecularState(
        product_atoms,
        (Edge(0, 1, 2.0, "Z"), Edge(0, 2, 1.0)),
        ((0, 1, 2),),
    )
    return ReactionRecord(
        "rxn0001", reactant, product, (Edge(0, 1, 1.5), Edge(0, 2, 0.5)), "train"
    )


def test_serialization_contains_state_and_change_evidence() -> None:
    text = serialize_input(_record())
    assert "a0 [stereo=R]" in text
    assert "a0 [stereo=S]" in text
    assert "a1 [charge=-1]" in text
    assert "a1 [charge=-1->+0]" in text
    assert "a0 --[R:bo=1,stereo=E;P:bo=2,stereo=Z]-- a1" in text
    # R/S is state evidence, deliberately not duplicated in ATOM_CHANGES.
    atom_changes = text.split("<ATOM_CHANGES>\n", 1)[1].split("\n</ATOM_CHANGES>", 1)[0]
    assert "stereo" not in atom_changes


def test_epoch_permutation_is_stateless_and_eval_canonical() -> None:
    record = _record()
    assert serialize_input(record, training=True, epoch=3) == serialize_input(
        record, training=True, epoch=3
    )
    assert serialize_input(record, training=False, epoch=1) == serialize_input(
        record, training=False, epoch=99
    )


def test_target_is_canonical_and_has_no_stereo() -> None:
    assert serialize_target(_record().ts_edges) == (
        "<TS_EDGES>\n"
        "a0 --[bo=1.5]-- a1\n"
        "a0 --[bo=0.5]-- a2\n"
        "</TS_EDGES>"
    )
