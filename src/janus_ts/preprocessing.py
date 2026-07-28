"""Deterministic Transition1x preprocessing and processed-data audit.

Only four content-pinned files are consumed: three trusted raw split pickles
and the frozen hybrid recovery JSONL.  The pickles define membership and TS
Wiberg bond orders; the recovery manifest defines corrected mapped R/P SMILES.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from rdkit import rdBase

from .artifacts import (
    COMPLETE_MARKER,
    ArtifactError,
    mark_complete,
    read_complete_manifest,
)
from .chemistry import (
    ChemistryError,
    directional_reaction_signature,
    molecular_state_from_mapped_smiles,
)
from .config import ExperimentConfig
from .constants import (
    MAX_SEQUENCE_LENGTH,
    MIN_TS_EDGE_WEIGHT,
    MODEL_ID,
    MODEL_REVISION,
    QUARANTINED_REACTION_IDS,
    REPRESENTATION_VERSION,
    SEED,
    TRANSITION1X_SOURCE_SHA256,
)
from .discretization import normalize_ts_edge_weight
from .parsing import parse_ts_edges
from .representation import serialize_input, serialize_target
from .schema import Atom, Edge, MolecularState, ReactionRecord, canonical_edges
from .tokenization import (
    EpochAwareTokenizedDataset,
    QwenTrainingEncoder,
    TokenizationContractError,
    audit_dataset_lengths,
    validate_qwen_tokenizer,
)

PROCESSED_SCHEMA_VERSION = "janus-ts-transition1x-arrow-v1"
EXPECTED_RECOVERY_RECORDS = 10_073
SPLITS = ("train", "val", "test")
_REACTION_ID_RE = re.compile(r"^rxn[0-9]{4,}$")


class PreprocessingError(RuntimeError):
    """Raised when a source, record, or persisted dataset fails a hard gate."""


@dataclass(frozen=True, slots=True)
class SourceFile:
    logical_name: str
    path: Path
    size_bytes: int
    sha256: str

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    output_path: Path
    fingerprint: str
    manifest: Mapping[str, Any]
    audit: Mapping[str, Any]


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
    except OSError as exc:
        raise PreprocessingError(f"cannot hash source {source}: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_paths(config: ExperimentConfig) -> dict[str, Path]:
    return {
        **{f"{split}.pkl": config.data.raw_dir / f"{split}.pkl" for split in SPLITS},
        "recovery.jsonl": config.data.recovery_jsonl,
    }


def inventory_sources(config: ExperimentConfig) -> dict[str, SourceFile]:
    """Hash and validate every consumed source against the frozen identities."""

    expected = dict(config.data.expected_source_sha256)
    if expected != TRANSITION1X_SOURCE_SHA256:
        raise PreprocessingError("configured source hashes differ from the frozen protocol")
    paths = _source_paths(config)
    if set(paths) != set(expected):
        raise PreprocessingError(
            f"source inventory mismatch: paths={sorted(paths)}, expected={sorted(expected)}"
        )

    result: dict[str, SourceFile] = {}
    for logical_name, path in paths.items():
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise PreprocessingError(f"source is unavailable: {path}: {exc}") from exc
        actual = sha256_file(path)
        if actual != expected[logical_name]:
            raise PreprocessingError(
                f"SHA256 mismatch for {logical_name}: got {actual}, "
                f"expected {expected[logical_name]}"
            )
        result[logical_name] = SourceFile(logical_name, path, size_bytes, actual)
    return result


def _code_inventory() -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    names = (
        "artifacts.py",
        "chemistry.py",
        "config.py",
        "constants.py",
        "discretization.py",
        "preprocessing.py",
        "representation.py",
        "schema.py",
        "tokenization.py",
    )
    return {name: sha256_file(package_root / name) for name in names}


def preprocessing_fingerprint(
    config: ExperimentConfig,
    sources: Mapping[str, SourceFile] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return the output fingerprint and the canonical payload that defines it."""

    source_inventory = dict(sources or inventory_sources(config))
    code_sha256 = _code_inventory()
    payload: dict[str, Any] = {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "config_sha256": config.sha256,
        "sources": {
            name: {
                "size_bytes": source.size_bytes,
                "sha256": source.sha256,
            }
            for name, source in sorted(source_inventory.items())
        },
        "code_sha256": code_sha256,
        "protocol": {
            "representation": REPRESENTATION_VERSION,
            "min_ts_edge_weight": MIN_TS_EDGE_WEIGHT,
            "signature_algorithm": config.data.signature_algorithm,
            "quarantined_reaction_ids": list(QUARANTINED_REACTION_IDS),
            "seed": SEED,
            "tokenizer_model_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        },
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "datasets": version("datasets"),
            "pyarrow": version("pyarrow"),
        },
    }
    return _canonical_sha256(payload), payload


def expected_processed_path(
    config: ExperimentConfig,
    sources: Mapping[str, SourceFile] | None = None,
) -> Path:
    fingerprint, _ = preprocessing_fingerprint(config, sources)
    return config.data.processed_root / config.data.name / fingerprint


def load_recovery_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the SHA-validated hybrid JSONL and reject duplicate/malformed IDs."""

    source = Path(path)
    records: dict[str, dict[str, Any]] = {}
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                    reaction_id = record["rxn"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise PreprocessingError(
                        f"invalid recovery JSON on line {line_number}: {exc}"
                    ) from exc
                if not isinstance(reaction_id, str) or not _REACTION_ID_RE.fullmatch(reaction_id):
                    raise PreprocessingError(
                        f"invalid recovery reaction ID on line {line_number}: {reaction_id!r}"
                    )
                if reaction_id in records:
                    raise PreprocessingError(f"duplicate recovery reaction ID: {reaction_id}")
                required = {
                    "atomic_nums",
                    "R_smi",
                    "P_smi",
                    "R_source",
                    "P_source",
                    "R_method",
                    "P_method",
                    "atom_map_aligned_R",
                    "atom_map_aligned_P",
                }
                missing = sorted(required - record.keys())
                if missing:
                    raise PreprocessingError(
                        f"recovery {reaction_id} lacks fields: {', '.join(missing)}"
                    )
                aligned = (
                    record["atom_map_aligned_R"] is True
                    and record["atom_map_aligned_P"] is True
                )
                if not aligned:
                    raise PreprocessingError(f"recovery {reaction_id} is not atom-map aligned")
                records[reaction_id] = record
    except OSError as exc:
        raise PreprocessingError(f"cannot read recovery manifest {source}: {exc}") from exc
    if len(records) != EXPECTED_RECOVERY_RECORDS:
        raise PreprocessingError(
            f"recovery manifest has {len(records)} records, expected {EXPECTED_RECOVERY_RECORDS}"
        )
    return records


def load_raw_splits(
    raw_dir: str | Path,
    expected_counts: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Load the three trusted project pickles and validate their outer schema."""

    root = Path(raw_dir)
    if set(expected_counts) != set(SPLITS):
        raise PreprocessingError("expected raw counts must contain train, val, and test")
    result: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = root / f"{split}.pkl"
        try:
            # These are content-pinned, trusted local project artifacts.  Never
            # generalize this loader to untrusted pickle input.
            with path.open("rb") as handle:
                rows = pickle.load(handle)  # noqa: S301
        except (OSError, pickle.UnpicklingError) as exc:
            raise PreprocessingError(f"cannot load {path}: {exc}") from exc
        if not isinstance(rows, list):
            raise PreprocessingError(f"{path} must contain a list")
        if len(rows) != expected_counts[split]:
            raise PreprocessingError(
                f"{split} has {len(rows)} raw records, expected {expected_counts[split]}"
            )
        result[split] = rows
    return result


def discretize_ts_edges(
    raw_edges: Iterable[Sequence[Any]],
    *,
    atom_count: int,
    min_edge_weight: float = MIN_TS_EDGE_WEIGHT,
    context: str = "transition state",
) -> tuple[Edge, ...]:
    """Apply the exact GeoDiff WBO bins and omit values below the cutoff."""

    seen: set[tuple[int, int]] = set()
    edges: list[Edge] = []
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, (tuple, list)) or len(raw_edge) != 3:
            raise PreprocessingError(f"{context}: every raw edge must be (i, j, WBO)")
        try:
            atom_i, atom_j = int(raw_edge[0]), int(raw_edge[1])
        except (TypeError, ValueError) as exc:
            raise PreprocessingError(f"{context}: non-integer atom ID in {raw_edge!r}") from exc
        if atom_i == atom_j or min(atom_i, atom_j) < 0 or max(atom_i, atom_j) >= atom_count:
            raise PreprocessingError(f"{context}: invalid atom pair ({atom_i}, {atom_j})")
        pair = tuple(sorted((atom_i, atom_j)))
        if pair in seen:
            raise PreprocessingError(f"{context}: duplicate atom pair {pair}")
        seen.add(pair)
        try:
            bond_order = normalize_ts_edge_weight(
                float(raw_edge[2]), min_edge_weight=min_edge_weight
            )
        except (TypeError, ValueError) as exc:
            raise PreprocessingError(f"{context}: invalid WBO in {raw_edge!r}: {exc}") from exc
        if bond_order > 0:
            edges.append(Edge(pair[0], pair[1], bond_order))
    return canonical_edges(edges)


def _validate_raw_record(
    raw: Mapping[str, Any],
    *,
    split: str,
) -> tuple[str, tuple[int, ...]]:
    required = {"rxn", "atom_types", "reactant", "transition_state", "product"}
    missing = sorted(required - raw.keys())
    if missing:
        raise PreprocessingError(f"{split} raw record lacks fields: {', '.join(missing)}")
    reaction_id = raw["rxn"]
    if not isinstance(reaction_id, str) or not _REACTION_ID_RE.fullmatch(reaction_id):
        raise PreprocessingError(f"{split}: invalid reaction ID {reaction_id!r}")
    try:
        atomic_numbers = tuple(int(value) for value in raw["atom_types"])
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(f"{reaction_id}: invalid atom_types") from exc
    if not atomic_numbers:
        raise PreprocessingError(f"{reaction_id}: empty atom table")
    for state_name in ("reactant", "transition_state", "product"):
        state = raw[state_name]
        if not isinstance(state, Mapping):
            raise PreprocessingError(f"{reaction_id}/{state_name}: expected mapping")
        nested_z = tuple(int(value) for value in state.get("atom_types", ()))
        if nested_z != atomic_numbers:
            raise PreprocessingError(f"{reaction_id}/{state_name}: atom_types drift")
    return reaction_id, atomic_numbers


def _reaction_row(
    raw: Mapping[str, Any],
    recovery: Mapping[str, Any],
    *,
    split: str,
    min_edge_weight: float,
) -> dict[str, Any]:
    reaction_id, atomic_numbers = _validate_raw_record(raw, split=split)
    recovery_z = tuple(int(value) for value in recovery["atomic_nums"])
    if recovery_z != atomic_numbers:
        raise PreprocessingError(f"{reaction_id}: recovery atomic numbers differ from raw")
    reactant_smiles = recovery["R_smi"]
    product_smiles = recovery["P_smi"]
    try:
        reactant = molecular_state_from_mapped_smiles(
            reactant_smiles, atomic_numbers, context=f"{reaction_id}/reactant"
        )
        product = molecular_state_from_mapped_smiles(
            product_smiles, atomic_numbers, context=f"{reaction_id}/product"
        )
        signature = directional_reaction_signature(
            reactant_smiles, product_smiles, atomic_numbers, context=reaction_id
        )
    except ChemistryError:
        raise
    ts_state = raw["transition_state"]
    if "edges" not in ts_state:
        raise PreprocessingError(f"{reaction_id}/transition_state: missing edges")
    ts_edges = discretize_ts_edges(
        ts_state["edges"],
        atom_count=len(atomic_numbers),
        min_edge_weight=min_edge_weight,
        context=f"{reaction_id}/transition_state",
    )
    record = ReactionRecord(
        reaction_id=reaction_id,
        reactant=reactant,
        product=product,
        ts_edges=ts_edges,
        split=split,
        metadata={
            "reactant_source": str(recovery["R_source"]),
            "reactant_method": str(recovery["R_method"]),
            "product_source": str(recovery["P_source"]),
            "product_method": str(recovery["P_method"]),
        },
    )
    row = record.to_dict()
    row.update(
        {
            "atom_count": record.atom_count,
            "reactant_smiles": reactant_smiles,
            "product_smiles": product_smiles,
            "chemical_signature": signature,
            "input_text": serialize_input(record, training=False),
            "target_text": serialize_target(record.ts_edges),
        }
    )
    return row


def build_processed_rows(
    raw_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    recovery_records: Mapping[str, Mapping[str, Any]],
    *,
    expected_raw_counts: Mapping[str, int],
    expected_retained_counts: Mapping[str, int],
    min_edge_weight: float = MIN_TS_EDGE_WEIGHT,
    quarantined_reaction_ids: Sequence[str] = QUARANTINED_REACTION_IDS,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Join, regenerate, quarantine, and audit all records before Arrow write."""

    if set(raw_splits) != set(SPLITS):
        raise PreprocessingError("raw splits must be exactly train, val, and test")
    if min_edge_weight != MIN_TS_EDGE_WEIGHT:
        raise PreprocessingError(
            f"min_edge_weight must remain frozen at {MIN_TS_EDGE_WEIGHT}"
        )
    quarantine = set(quarantined_reaction_ids)
    if len(quarantine) != len(quarantined_reaction_ids):
        raise PreprocessingError("quarantine list contains duplicate IDs")

    output: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    global_ids: dict[str, str] = {}
    failure_sides: dict[str, list[str]] = defaultdict(list)
    signatures: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split in SPLITS:
        rows = raw_splits[split]
        if len(rows) != expected_raw_counts[split]:
            raise PreprocessingError(
                f"{split} has {len(rows)} records, expected {expected_raw_counts[split]}"
            )
        for raw in rows:
            reaction_id, atomic_numbers = _validate_raw_record(raw, split=split)
            if reaction_id in global_ids:
                raise PreprocessingError(
                    f"reaction ID overlap: {reaction_id} occurs in "
                    f"{global_ids[reaction_id]} and {split}"
                )
            global_ids[reaction_id] = split
            recovery = recovery_records.get(reaction_id)
            if recovery is None:
                raise PreprocessingError(f"{reaction_id}: missing from recovery manifest")
            try:
                recovery_z = tuple(int(value) for value in recovery["atomic_nums"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PreprocessingError(
                    f"{reaction_id}: invalid recovery atomic numbers"
                ) from exc
            if recovery_z != atomic_numbers:
                raise PreprocessingError(
                    f"{reaction_id}: recovery atomic numbers differ from raw"
                )

            # Probe each endpoint separately so the quarantine is itself an
            # audited fact, not an unconditional deletion list.
            for side, key in (("reactant", "R_smi"), ("product", "P_smi")):
                try:
                    molecular_state_from_mapped_smiles(
                        recovery[key], atomic_numbers, context=f"{reaction_id}/{side}"
                    )
                except ChemistryError:
                    failure_sides[reaction_id].append(side)

            if reaction_id in quarantine:
                continue
            if reaction_id in failure_sides:
                raise PreprocessingError(
                    f"unquarantined strict chemistry failure: {reaction_id} "
                    f"{failure_sides[reaction_id]}"
                )
            try:
                row = _reaction_row(
                    raw,
                    recovery,
                    split=split,
                    min_edge_weight=min_edge_weight,
                )
            except ChemistryError as exc:
                raise PreprocessingError(str(exc)) from exc
            output[split].append(row)
            signatures[row["chemical_signature"]].append((split, reaction_id))

    missing_quarantine = sorted(quarantine - global_ids.keys())
    if missing_quarantine:
        raise PreprocessingError(f"quarantine IDs absent from raw data: {missing_quarantine}")
    expected_failures = {reaction_id: ["product"] for reaction_id in quarantine}
    actual_failures = {key: value for key, value in sorted(failure_sides.items())}
    if actual_failures != expected_failures:
        raise PreprocessingError(
            "strict endpoint failures differ from frozen quarantine: "
            f"got={actual_failures}, expected={expected_failures}"
        )

    counts = {split: len(output[split]) for split in SPLITS}
    if counts != dict(expected_retained_counts):
        raise PreprocessingError(
            f"retained counts differ: got={counts}, expected={dict(expected_retained_counts)}"
        )
    recovery_extras = sorted(set(recovery_records) - set(global_ids))
    expected_extras = len(recovery_records) - sum(expected_raw_counts.values())
    if len(recovery_extras) != expected_extras:
        raise PreprocessingError("recovery/raw join cardinality is inconsistent")

    cross_split: dict[str, list[tuple[str, str]]] = {}
    within_split: dict[str, list[tuple[str, str]]] = {}
    for signature, occurrences in signatures.items():
        occurrence_splits = {split for split, _ in occurrences}
        if len(occurrence_splits) > 1:
            cross_split[signature] = occurrences
        elif len(occurrences) > 1:
            within_split[signature] = occurrences
    if cross_split:
        preview = list(cross_split.values())[:10]
        raise PreprocessingError(
            f"directional full-reaction signature leakage across splits: {preview}"
        )

    audit = {
        "status": "passed",
        "raw_counts": {split: len(raw_splits[split]) for split in SPLITS},
        "retained_counts": counts,
        "raw_unique_reaction_ids": len(global_ids),
        "recovery_records": len(recovery_records),
        "recovery_extra_records": len(recovery_extras),
        "quarantined_reaction_ids": sorted(quarantine),
        "quarantined_endpoint_failures": actual_failures,
        "cross_split_reaction_id_overlap": 0,
        "cross_split_directional_signature_overlap": 0,
        "within_split_directional_signature_repeat_groups": len(within_split),
        "within_split_directional_signature_repeat_rows": sum(
            len(items) for items in within_split.values()
        ),
        "within_split_directional_signature_repeat_examples": list(within_split.values())[:20],
        "token_lengths": {"checked": False},
    }
    return output, audit


def reaction_record_from_row(row: Mapping[str, Any]) -> ReactionRecord:
    """Rehydrate an Arrow row into the validated immutable graph schema."""

    def atom(value: Mapping[str, Any]) -> Atom:
        return Atom(
            atom_id=int(value["atom_id"]),
            atomic_number=int(value["atomic_number"]),
            symbol=str(value["symbol"]),
            formal_charge=int(value.get("formal_charge", 0)),
            radical_electrons=int(value.get("radical_electrons", 0)),
            stereo=value.get("stereo"),
        )

    def edge(value: Mapping[str, Any]) -> Edge:
        return Edge(
            atom_i=int(value["atom_i"]),
            atom_j=int(value["atom_j"]),
            bond_order=float(value["bond_order"]),
            stereo=value.get("stereo"),
        )

    def state(value: Mapping[str, Any]) -> MolecularState:
        return MolecularState(
            atoms=tuple(atom(item) for item in value["atoms"]),
            edges=tuple(edge(item) for item in value["edges"]),
            components=tuple(
                tuple(int(atom_id) for atom_id in component)
                for component in value["components"]
            ),
        )

    return ReactionRecord(
        reaction_id=str(row["reaction_id"]),
        reactant=state(row["reactant"]),
        product=state(row["product"]),
        ts_edges=tuple(edge(item) for item in row["ts_edges"]),
        split=str(row["split"]),
        metadata=dict(row.get("metadata") or {}),
    )


def audit_dataset_dict(
    dataset: DatasetDict,
    config: ExperimentConfig,
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Revalidate persisted chemistry, serialization, split isolation, and length."""

    if set(dataset) != set(SPLITS):
        raise PreprocessingError(f"processed dataset splits differ: {sorted(dataset)}")
    counts = {split: len(dataset[split]) for split in SPLITS}
    if counts != config.data.expected_retained_counts:
        raise PreprocessingError(
            "processed counts differ: "
            f"got={counts}, expected={config.data.expected_retained_counts}"
        )

    seen_ids: dict[str, str] = {}
    signatures: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split in SPLITS:
        for row in dataset[split]:
            record = reaction_record_from_row(row)
            if record.split != split:
                raise PreprocessingError(
                    f"{record.reaction_id}: stored split {record.split!r} != {split!r}"
                )
            if record.reaction_id in seen_ids:
                raise PreprocessingError(
                    f"processed reaction ID overlap: {record.reaction_id} in "
                    f"{seen_ids[record.reaction_id]} and {split}"
                )
            seen_ids[record.reaction_id] = split
            if int(row["atom_count"]) != record.atom_count:
                raise PreprocessingError(f"{record.reaction_id}: atom_count mismatch")

            atomic_numbers = tuple(atom.atomic_number for atom in record.reactant.atoms)
            try:
                regenerated_reactant = molecular_state_from_mapped_smiles(
                    row["reactant_smiles"],
                    atomic_numbers,
                    context=f"{record.reaction_id}/audit/reactant",
                )
                regenerated_product = molecular_state_from_mapped_smiles(
                    row["product_smiles"],
                    atomic_numbers,
                    context=f"{record.reaction_id}/audit/product",
                )
                signature = directional_reaction_signature(
                    row["reactant_smiles"],
                    row["product_smiles"],
                    atomic_numbers,
                    context=f"{record.reaction_id}/audit",
                )
            except ChemistryError as exc:
                raise PreprocessingError(str(exc)) from exc
            if regenerated_reactant != record.reactant or regenerated_product != record.product:
                raise PreprocessingError(f"{record.reaction_id}: persisted endpoint graph drift")
            if signature != row["chemical_signature"]:
                raise PreprocessingError(f"{record.reaction_id}: chemical signature drift")
            signatures[signature].append((split, record.reaction_id))

            if serialize_input(record, training=False) != row["input_text"]:
                raise PreprocessingError(f"{record.reaction_id}: canonical input text drift")
            if serialize_target(record.ts_edges) != row["target_text"]:
                raise PreprocessingError(f"{record.reaction_id}: canonical target text drift")
            parsed = parse_ts_edges(row["target_text"], atom_count=record.atom_count)
            if not parsed.valid or parsed.edges != record.ts_edges:
                raise PreprocessingError(
                    f"{record.reaction_id}: strict target parse failed ({parsed.error_code})"
                )

    cross_split = {
        signature: occurrences
        for signature, occurrences in signatures.items()
        if len({split for split, _ in occurrences}) > 1
    }
    if cross_split:
        raise PreprocessingError(
            "processed directional full-reaction signature leakage: "
            f"{list(cross_split.values())[:10]}"
        )
    within_split = {
        signature: occurrences
        for signature, occurrences in signatures.items()
        if len(occurrences) > 1
    }
    length_audit: dict[str, Any] = {"checked": tokenizer is not None}
    if tokenizer is not None:
        chat_template = getattr(tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template:
            raise PreprocessingError("pinned tokenizer has no chat template")
        try:
            encoder = QwenTrainingEncoder(
                tokenizer,
                max_length=config.model.max_sequence_length,
                seed=config.seed,
            )
            split_audits = {}
            for split in SPLITS:
                is_training = split == "train"
                tokenized = EpochAwareTokenizedDataset(
                    dataset[split], encoder, training=is_training
                )
                length = audit_dataset_lengths(
                    tokenized,
                    logical_epochs=config.train.epochs if is_training else 1,
                )
                split_audits[split] = asdict(length)
        except TokenizationContractError as exc:
            raise PreprocessingError(f"training token-length gate failed: {exc}") from exc
        length_audit.update(
            {
                "limit": config.model.max_sequence_length,
                "tokenizer_class": tokenizer.__class__.__name__,
                "tokenizer_model_id": config.model.model_id,
                "tokenizer_revision": config.model.revision,
                "chat_template_sha256": hashlib.sha256(
                    chat_template.encode("utf-8")
                ).hexdigest(),
                "by_split": split_audits,
                "maximum_length": max(
                    item["maximum_length"] for item in split_audits.values()
                ),
                "encoded_example_epochs": sum(
                    item["examples"] * item["logical_epochs"]
                    for item in split_audits.values()
                ),
                "over_limit": 0,
                "truncated": 0,
            }
        )
    return {
        "status": "passed",
        "retained_counts": counts,
        "unique_reaction_ids": len(seen_ids),
        "strict_endpoint_parse_failures": 0,
        "strict_target_parse_failures": 0,
        "canonical_serialization_mismatches": 0,
        "cross_split_reaction_id_overlap": 0,
        "cross_split_directional_signature_overlap": 0,
        "within_split_directional_signature_repeat_groups": len(within_split),
        "within_split_directional_signature_repeat_rows": sum(
            len(items) for items in within_split.values()
        ),
        "within_split_directional_signature_repeat_examples": list(within_split.values())[:20],
        "token_lengths": length_audit,
    }


def load_processed_dataset(path: str | Path) -> DatasetDict:
    """Memory-map a processed DatasetDict; never request an in-memory copy."""

    loaded = load_from_disk(str(path), keep_in_memory=False)
    if not isinstance(loaded, DatasetDict):
        raise PreprocessingError(f"{path} is not a DatasetDict")
    return loaded


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(serialized, encoding="utf-8")


def _dataset_file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Hash every persisted DatasetDict file except mutable audit metadata."""

    excluded = {"manifest.json", "audit.json", COMPLETE_MARKER}
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        if path.is_symlink():
            raise PreprocessingError(f"processed dataset contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        inventory[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not inventory:
        raise PreprocessingError(f"processed dataset has no persisted files: {root}")
    return inventory


def audit_processed_dataset(
    path: str | Path,
    config: ExperimentConfig,
    *,
    tokenizer: Any | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(path)
    try:
        manifest = read_complete_manifest(root)
    except ArtifactError as exc:
        raise PreprocessingError(f"invalid/incomplete processed artifact {root}: {exc}") from exc
    sources = inventory_sources(config)
    fingerprint, fingerprint_payload = preprocessing_fingerprint(config, sources)
    if root.name != fingerprint or manifest.get("fingerprint") != fingerprint:
        raise PreprocessingError("processed path/manifest fingerprint mismatch")
    if manifest.get("fingerprint_payload") != fingerprint_payload:
        raise PreprocessingError("processed fingerprint payload drift")
    expected_dataset_files = manifest.get("dataset_files")
    actual_dataset_files = _dataset_file_inventory(root)
    if expected_dataset_files != actual_dataset_files:
        raise PreprocessingError("persisted Arrow file inventory/hash mismatch")

    audit = audit_dataset_dict(load_processed_dataset(root), config, tokenizer=tokenizer)
    audit.update(
        {
            "fingerprint": fingerprint,
            "audited_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    if write:
        _write_json(root / "audit.json", audit)
    return audit


def preprocess_transition1x(
    config: ExperimentConfig,
    *,
    tokenizer: Any | None = None,
) -> PreprocessResult:
    """Build the content-addressed Arrow DatasetDict and run all data gates."""

    sources = inventory_sources(config)
    fingerprint, fingerprint_payload = preprocessing_fingerprint(config, sources)
    output_path = config.data.processed_root / config.data.name / fingerprint
    if output_path.exists():
        audit = audit_processed_dataset(output_path, config, tokenizer=tokenizer, write=True)
        manifest = read_complete_manifest(output_path)
        return PreprocessResult(output_path, fingerprint, manifest, audit)

    raw_splits = load_raw_splits(config.data.raw_dir, config.data.expected_raw_counts)
    recovery = load_recovery_manifest(config.data.recovery_jsonl)
    rows, build_audit = build_processed_rows(
        raw_splits,
        recovery,
        expected_raw_counts=config.data.expected_raw_counts,
        expected_retained_counts=config.data.expected_retained_counts,
        min_edge_weight=config.data.min_ts_edge_weight,
    )
    dataset = DatasetDict(
        {split: Dataset.from_list(rows[split]) for split in SPLITS}
    )
    persisted_audit = audit_dataset_dict(dataset, config, tokenizer=tokenizer)
    build_audit["token_lengths"] = persisted_audit["token_lengths"]

    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.tmp-", dir=parent))
    try:
        dataset.save_to_disk(str(temporary))
        dataset_files = _dataset_file_inventory(temporary)
        manifest: dict[str, Any] = {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "config_sha256": config.sha256,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "sources": {
                name: source.manifest_dict() for name, source in sorted(sources.items())
            },
            "configured_h5_lineage_path": str(config.data.h5_path),
            "counts": {
                "raw": dict(config.data.expected_raw_counts),
                "retained": dict(config.data.expected_retained_counts),
                "recovery": len(recovery),
            },
            "dataset_files": dataset_files,
            "build_audit": build_audit,
        }
        final_audit = {
            **persisted_audit,
            "fingerprint": fingerprint,
            "audited_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(temporary / "audit.json", final_audit)
        # Completion is the final write inside the temporary tree.  Resume and
        # run identity never infer completion from timestamps.
        mark_complete(temporary, manifest)
        if output_path.exists():
            raise PreprocessingError(f"processed output appeared concurrently: {output_path}")
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return PreprocessResult(output_path, fingerprint, manifest, final_audit)


def load_pinned_tokenizer(config: ExperimentConfig, *, local_files_only: bool = False) -> Any:
    """Load only the pinned tokenizer needed for the no-truncation audit."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        cache_dir=str(config.model.cache_dir),
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    try:
        validate_qwen_tokenizer(tokenizer)
    except TokenizationContractError as exc:
        raise PreprocessingError(f"pinned tokenizer validation failed: {exc}") from exc
    return tokenizer
