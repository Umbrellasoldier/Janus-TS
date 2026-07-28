"""Atomic, content-addressed experiment artifacts.

The gu30 system clock is not a trustworthy ordering source.  Completion and
resume decisions therefore use explicit manifests, monotonically increasing
``global_step`` values, and a marker written only after all payload files have
been flushed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

COMPLETE_MARKER = ".complete"


class ArtifactError(RuntimeError):
    """Raised when an artifact is incomplete or violates its manifest."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def write_json(path: str | Path, value: Any) -> None:
    """Write JSON durably without exposing a partially-written file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def mark_complete(directory: str | Path, manifest: Mapping[str, Any]) -> None:
    """Persist a manifest and write the completion marker last."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "manifest.json", dict(manifest))
    marker = root / COMPLETE_MARKER
    marker.write_text(sha256_file(root / "manifest.json") + "\n", encoding="ascii")
    with marker.open("rb") as handle:
        os.fsync(handle.fileno())


def read_complete_manifest(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    marker = root / COMPLETE_MARKER
    manifest_path = root / "manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        raise ArtifactError(f"incomplete artifact: {root}")
    expected = marker.read_text(encoding="ascii").strip()
    actual = sha256_file(manifest_path)
    if expected != actual:
        raise ArtifactError(f"manifest hash mismatch in {root}: {actual} != {expected}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid manifest in {root}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"manifest is not an object: {manifest_path}")
    return payload


@contextmanager
def atomic_directory(destination: str | Path) -> Iterator[Path]:
    """Build a new directory beside its destination and rename it atomically.

    Existing destinations are never overwritten.  Failed temporary trees are
    removed, while a successfully renamed tree remains immutable by contract.
    """

    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.building.", dir=target.parent)
    )
    try:
        yield temporary
        if not (temporary / COMPLETE_MARKER).is_file():
            raise ArtifactError(f"atomic artifact was not marked complete: {temporary}")
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def newest_complete_checkpoint(
    roots: list[str | Path], *, required_fingerprint: str
) -> tuple[Path, dict[str, Any]] | None:
    """Select the highest manifest step, never filesystem modification time."""

    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            try:
                manifest = read_complete_manifest(child)
            except ArtifactError:
                continue
            if manifest.get("run_fingerprint") != required_fingerprint:
                continue
            step = manifest.get("global_step")
            if isinstance(step, int) and step >= 0:
                candidates.append((step, child, manifest))
    if not candidates:
        return None
    _, path, manifest = max(candidates, key=lambda item: (item[0], item[1].name))
    return path, manifest


__all__ = [
    "ArtifactError",
    "COMPLETE_MARKER",
    "atomic_directory",
    "canonical_json_bytes",
    "mark_complete",
    "newest_complete_checkpoint",
    "read_complete_manifest",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "write_json",
]
