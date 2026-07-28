"""Strict RDKit conversion for the frozen Transition1x recovery manifest.

The recovery SMILES are the sole source of reactant/product connectivity and
attributes.  In particular, endpoint edges from the historical pickle are
never reused: the recovered SMILES corrected hundreds of those graphs.
"""

from __future__ import annotations

from collections.abc import Sequence

from rdkit import Chem

from .constants import ALLOWED_BOND_ORDERS
from .schema import Atom, Edge, MolecularState, canonical_edges


class ChemistryError(ValueError):
    """Raised when a recovered endpoint violates the frozen chemistry contract."""


def _strict_smiles_parser_params() -> Chem.SmilesParserParams:
    params = Chem.SmilesParserParams()
    params.sanitize = True
    # Transition1x stores every hydrogen as an independently mapped atom.  The
    # RDKit default removes explicit [H] atoms, destroying the shared ID table.
    params.removeHs = False
    return params


def parse_strict_mapped_smiles(
    smiles: str,
    atomic_numbers: Sequence[int],
    *,
    context: str = "endpoint",
) -> tuple[Chem.Mol, tuple[int, ...]]:
    """Parse and validate one atom-mapped endpoint SMILES.

    Returns the sanitized molecule and a tuple mapping each RDKit atom index to
    the zero-based Transition1x atom ID.  Map labels must be a permutation of
    ``1..N`` and the mapped element sequence must exactly match the raw record.
    """

    if not isinstance(smiles, str) or not smiles:
        raise ChemistryError(f"{context}: recovered SMILES is missing")
    expected_z = tuple(int(value) for value in atomic_numbers)
    if not expected_z or any(value <= 0 for value in expected_z):
        raise ChemistryError(f"{context}: invalid atomic-number table")

    mol = Chem.MolFromSmiles(smiles, _strict_smiles_parser_params())
    if mol is None:
        raise ChemistryError(f"{context}: strict RDKit sanitization failed")
    if mol.GetNumAtoms() != len(expected_z):
        raise ChemistryError(
            f"{context}: parsed {mol.GetNumAtoms()} atoms, expected {len(expected_z)}"
        )

    map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
    expected_maps = tuple(range(1, len(expected_z) + 1))
    if tuple(sorted(map_numbers)) != expected_maps:
        raise ChemistryError(
            f"{context}: atom maps must occur exactly once as 1..{len(expected_z)}"
        )
    atom_ids = tuple(map_number - 1 for map_number in map_numbers)
    for rdkit_idx, atom_id in enumerate(atom_ids):
        actual_z = mol.GetAtomWithIdx(rdkit_idx).GetAtomicNum()
        if actual_z != expected_z[atom_id]:
            raise ChemistryError(
                f"{context}: map {atom_id + 1} has Z={actual_z}, "
                f"expected Z={expected_z[atom_id]}"
            )

    # Force CIP and double-bond stereo assignment from the parsed SMILES before
    # extracting the graph.  Existing tags are cleaned so no stale property can
    # survive a molecule copy or a future caller mutation.
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol, atom_ids


def _atom_stereo(atom: Chem.Atom) -> str | None:
    if not atom.HasProp("_CIPCode"):
        return None
    value = atom.GetProp("_CIPCode")
    if value not in {"R", "S"}:
        raise ChemistryError(f"unsupported atom CIP code: {value!r}")
    return value


def _bond_stereo(bond: Chem.Bond) -> str | None:
    value = bond.GetStereo()
    if value == Chem.BondStereo.STEREONONE:
        return None
    if value in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}:
        return "E"
    if value in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}:
        return "Z"
    raise ChemistryError(f"unsupported or unresolved bond stereo: {value!s}")


def molecular_state_from_mapped_smiles(
    smiles: str,
    atomic_numbers: Sequence[int],
    *,
    context: str = "endpoint",
) -> MolecularState:
    """Regenerate a complete endpoint state in shared Transition1x atom order."""

    mol, atom_ids = parse_strict_mapped_smiles(smiles, atomic_numbers, context=context)
    atoms_by_id: list[Atom | None] = [None] * len(atom_ids)
    for rdkit_idx, atom_id in enumerate(atom_ids):
        rd_atom = mol.GetAtomWithIdx(rdkit_idx)
        atoms_by_id[atom_id] = Atom(
            atom_id=atom_id,
            atomic_number=rd_atom.GetAtomicNum(),
            symbol=rd_atom.GetSymbol(),
            formal_charge=rd_atom.GetFormalCharge(),
            radical_electrons=rd_atom.GetNumRadicalElectrons(),
            stereo=_atom_stereo(rd_atom),
        )
    if any(atom is None for atom in atoms_by_id):  # defensive; maps were checked above
        raise ChemistryError(f"{context}: atom map did not populate every shared ID")

    edges: list[Edge] = []
    for bond in mol.GetBonds():
        atom_i = atom_ids[bond.GetBeginAtomIdx()]
        atom_j = atom_ids[bond.GetEndAtomIdx()]
        atom_i, atom_j = sorted((atom_i, atom_j))
        bond_order = float(bond.GetBondTypeAsDouble())
        if bond_order not in ALLOWED_BOND_ORDERS:
            raise ChemistryError(
                f"{context}: unsupported endpoint bond order {bond_order} "
                f"for ({atom_i}, {atom_j})"
            )
        edges.append(
            Edge(
                atom_i=atom_i,
                atom_j=atom_j,
                bond_order=bond_order,
                stereo=_bond_stereo(bond),
            )
        )

    components = []
    for fragment in Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False):
        components.append(tuple(sorted(atom_ids[rdkit_idx] for rdkit_idx in fragment)))
    components.sort(key=lambda component: (component[0], component))
    return MolecularState(
        atoms=tuple(atom for atom in atoms_by_id if atom is not None),
        edges=canonical_edges(edges),
        components=tuple(components),
    )


def canonical_mapless_endpoint_smiles(
    smiles: str,
    atomic_numbers: Sequence[int],
    *,
    context: str = "endpoint",
) -> str:
    """Return a canonical isomeric, mapless, component-sorted endpoint key."""

    mol, _ = parse_strict_mapped_smiles(smiles, atomic_numbers, context=context)
    mapless = Chem.Mol(mol)
    for atom in mapless.GetAtoms():
        atom.SetAtomMapNum(0)
    fragments = Chem.GetMolFrags(mapless, asMols=True, sanitizeFrags=True)
    component_smiles = sorted(
        Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        for fragment in fragments
    )
    if not component_smiles:
        raise ChemistryError(f"{context}: molecule contains no components")
    return ".".join(component_smiles)


def directional_reaction_signature(
    reactant_smiles: str,
    product_smiles: str,
    atomic_numbers: Sequence[int],
    *,
    context: str = "reaction",
) -> str:
    """Return the frozen directional full-reaction signature ``R>>P``.

    Endpoint-only overlap is deliberately not considered leakage.  This key is
    used only as a complete directional reaction pair by the split auditor.
    """

    reactant = canonical_mapless_endpoint_smiles(
        reactant_smiles, atomic_numbers, context=f"{context}/reactant"
    )
    product = canonical_mapless_endpoint_smiles(
        product_smiles, atomic_numbers, context=f"{context}/product"
    )
    return f"{reactant}>>{product}"
