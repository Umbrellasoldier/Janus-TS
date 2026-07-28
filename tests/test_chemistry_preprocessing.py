from __future__ import annotations

import pytest

from janus_ts.chemistry import (
    ChemistryError,
    canonical_mapless_endpoint_smiles,
    directional_reaction_signature,
    molecular_state_from_mapped_smiles,
    parse_strict_mapped_smiles,
)
from janus_ts.preprocessing import (
    PreprocessingError,
    build_processed_rows,
    discretize_ts_edges,
    reaction_record_from_row,
)


def test_mapped_smiles_regenerates_shared_order_attributes_components_and_stereo():
    chiral = "[Cl:2][C@@:1]([F:3])([H:4])[Br:5]"
    state = molecular_state_from_mapped_smiles(chiral, [6, 17, 9, 1, 35])

    assert [atom.atomic_number for atom in state.atoms] == [6, 17, 9, 1, 35]
    assert state.atoms[0].stereo == "S"
    assert [edge.pair for edge in state.edges] == [(0, 1), (0, 2), (0, 3), (0, 4)]
    assert state.components == ((0, 1, 2, 3, 4),)

    alkene = "[F:1]/[C:2]([H:5])=[C:3](/[Cl:4])[H:6]"
    alkene_state = molecular_state_from_mapped_smiles(alkene, [9, 6, 6, 17, 1, 1])
    double_bond = next(edge for edge in alkene_state.edges if edge.pair == (1, 2))
    assert double_bond.bond_order == 2.0
    assert double_bond.stereo == "E"

    salt = molecular_state_from_mapped_smiles("[Na+:1].[Cl-:2]", [11, 17])
    assert [atom.formal_charge for atom in salt.atoms] == [1, -1]
    assert salt.components == ((0,), (1,))


def test_strict_mapped_smiles_rejects_missing_duplicate_or_misaligned_maps():
    with pytest.raises(ChemistryError, match="exactly once"):
        parse_strict_mapped_smiles("[CH3:1][OH]", [6, 8])
    with pytest.raises(ChemistryError, match="exactly once"):
        parse_strict_mapped_smiles("[CH3:1][OH:1]", [6, 8])
    with pytest.raises(ChemistryError, match="expected Z"):
        parse_strict_mapped_smiles("[CH3:1][OH:2]", [8, 6])


def test_mapless_signature_is_component_canonical_and_directional():
    atomic_numbers = [6, 8]
    forward = directional_reaction_signature(
        "[CH3:1].[OH:2]", "[CH3:1][OH:2]", atomic_numbers
    )
    reordered = directional_reaction_signature(
        "[OH:2].[CH3:1]", "[OH:2][CH3:1]", atomic_numbers
    )
    reverse = directional_reaction_signature(
        "[CH3:1][OH:2]", "[CH3:1].[OH:2]", atomic_numbers
    )

    assert forward == reordered
    assert forward != reverse
    assert ":1" not in canonical_mapless_endpoint_smiles("[CH3:1][OH:2]", atomic_numbers)


def test_ts_discretization_uses_cutoff_and_canonical_pairs():
    edges = discretize_ts_edges(
        [(1, 0, 0.0999), (2, 1, 0.1), (0, 2, 2.75)], atom_count=3
    )
    assert [(edge.pair, edge.bond_order) for edge in edges] == [
        ((0, 2), 3.0),
        ((1, 2), 0.5),
    ]


def _raw(reaction_id: str, atom_types: list[int], ts_edges):
    # Endpoint edges are intentionally wrong/empty: preprocessing must never
    # retain them, even for a valid recovery SMILES.
    return {
        "rxn": reaction_id,
        "atom_types": atom_types,
        "reactant": {"atom_types": atom_types, "edges": []},
        "transition_state": {"atom_types": atom_types, "edges": ts_edges},
        "product": {"atom_types": atom_types, "edges": []},
    }


def _recovery(reaction_id: str, atomic_nums: list[int], reactant: str, product: str):
    return {
        "rxn": reaction_id,
        "atomic_nums": atomic_nums,
        "R_smi": reactant,
        "P_smi": product,
        "R_source": "fixture",
        "P_source": "fixture",
        "R_method": "fixture",
        "P_method": "fixture",
        "atom_map_aligned_R": True,
        "atom_map_aligned_P": True,
    }


def test_processed_join_regenerates_endpoint_edges_and_target():
    reaction_id = "rxn0001"
    raw = {"train": [_raw(reaction_id, [6, 8], [(0, 1, 0.1)])], "val": [], "test": []}
    recovery = {
        reaction_id: _recovery(reaction_id, [6, 8], "[CH3:1][OH:2]", "[CH3:1][OH:2]")
    }
    rows, audit = build_processed_rows(
        raw,
        recovery,
        expected_raw_counts={"train": 1, "val": 0, "test": 0},
        expected_retained_counts={"train": 1, "val": 0, "test": 0},
        quarantined_reaction_ids=(),
    )

    record = reaction_record_from_row(rows["train"][0])
    assert [(edge.pair, edge.bond_order) for edge in record.reactant.edges] == [
        ((0, 1), 1.0)
    ]
    assert [(edge.pair, edge.bond_order) for edge in record.ts_edges] == [
        ((0, 1), 0.5)
    ]
    assert audit["retained_counts"] == {"train": 1, "val": 0, "test": 0}


def test_quarantine_must_exactly_equal_observed_strict_failures():
    reaction_id = "rxn0002"
    raw = {"train": [_raw(reaction_id, [6], [])], "val": [], "test": []}
    recovery = {
        reaction_id: _recovery(reaction_id, [6], "[CH4:1]", "not-a-smiles")
    }
    rows, audit = build_processed_rows(
        raw,
        recovery,
        expected_raw_counts={"train": 1, "val": 0, "test": 0},
        expected_retained_counts={"train": 0, "val": 0, "test": 0},
        quarantined_reaction_ids=(reaction_id,),
    )
    assert rows == {"train": [], "val": [], "test": []}
    assert audit["quarantined_endpoint_failures"] == {reaction_id: ["product"]}

    with pytest.raises(PreprocessingError, match="unquarantined strict chemistry failure"):
        build_processed_rows(
            raw,
            recovery,
            expected_raw_counts={"train": 1, "val": 0, "test": 0},
            expected_retained_counts={"train": 1, "val": 0, "test": 0},
            quarantined_reaction_ids=(),
        )
