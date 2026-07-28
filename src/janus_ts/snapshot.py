"""Pinned Hugging Face snapshot download and offline identity audit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .constants import (
    MODEL_ID,
    MODEL_REVISION,
    QWEN_IM_END_TOKEN_ID,
    QWEN_IM_START_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
)
from .modeling import (
    assert_locked_model_stack,
    load_qwen_text_config,
    topology_summary,
    validate_checkpoint_index,
)

EXPECTED_WEIGHT_BYTES = 55_562_855_904
EXPECTED_WEIGHT_FILE_BYTES = 55_563_006_400
EXPECTED_CHECKPOINT_KEYS = 1_199
EXPECTED_TEXT_KEYS = 851
EXPECTED_VISION_KEYS = 333
EXPECTED_MTP_KEYS = 15
EXPECTED_CRITICAL_HASHES = {
    "config.json": "69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c",
    "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    "model.safetensors.index.json": (
        "a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b"
    ),
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "chat_template.jinja": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
}
_SHA256_NAME = re.compile(r"^[0-9a-f]{64}$")


class SnapshotContractError(RuntimeError):
    """The Hub snapshot does not match the frozen revision contract."""


def download_snapshot(*, cache_dir: str | Path, local_files_only: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
    )
    snapshot = Path(path).resolve()
    if snapshot.name != MODEL_REVISION:
        raise SnapshotContractError(
            f"snapshot resolved to {snapshot.name!r}, expected commit {MODEL_REVISION}"
        )
    return snapshot


def _blob_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    blob_name = resolved.name if _SHA256_NAME.fullmatch(resolved.name) else None
    return {
        "relative_path": path.name,
        "size": path.stat().st_size,
        "cache_blob_sha256": blob_name,
    }


def audit_snapshot(*, cache_dir: str | Path) -> dict[str, object]:
    """Perform an entirely offline audit of model, tokenizer, and topology."""

    snapshot = download_snapshot(cache_dir=cache_dir, local_files_only=True)
    failures: list[str] = []
    for relative, expected_hash in EXPECTED_CRITICAL_HASHES.items():
        path = snapshot / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif (actual := sha256_file(path)) != expected_hash:
            failures.append(f"{relative} sha256={actual}, expected {expected_hash}")

    index_path = snapshot / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        total_size = int(index["metadata"]["total_size"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SnapshotContractError(f"invalid checkpoint index: {exc}") from exc
    if len(weight_map) != EXPECTED_CHECKPOINT_KEYS:
        failures.append(f"checkpoint keys={len(weight_map)}, expected {EXPECTED_CHECKPOINT_KEYS}")
    if total_size != EXPECTED_WEIGHT_BYTES:
        failures.append(f"weight bytes={total_size}, expected {EXPECTED_WEIGHT_BYTES}")
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != 15:
        failures.append(f"weight shards={len(shard_names)}, expected 15")
    shard_identities: list[dict[str, Any]] = []
    actual_weight_bytes = 0
    for name in shard_names:
        shard = snapshot / name
        if not shard.is_file():
            failures.append(f"missing shard {name}")
            continue
        actual_weight_bytes += shard.stat().st_size
        shard_identities.append(_blob_identity(shard))
    if actual_weight_bytes != EXPECTED_WEIGHT_FILE_BYTES:
        failures.append(
            f"downloaded shard bytes={actual_weight_bytes}, "
            f"expected {EXPECTED_WEIGHT_FILE_BYTES}"
        )

    inventory = validate_checkpoint_index(index_path)
    counts = (
        len(inventory.mapped_text_keys),
        len(inventory.vision_keys),
        len(inventory.mtp_keys),
    )
    expected_counts = (EXPECTED_TEXT_KEYS, EXPECTED_VISION_KEYS, EXPECTED_MTP_KEYS)
    if counts != expected_counts:
        failures.append(f"text/vision/MTP keys={counts}, expected {expected_counts}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    token_contract = {
        "class": type(tokenizer).__name__,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "im_start_token_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }
    expected_tokens = {
        "class": "Qwen2Tokenizer",
        "pad_token_id": QWEN_PAD_TOKEN_ID,
        "eos_token_id": QWEN_IM_END_TOKEN_ID,
        "im_start_token_id": QWEN_IM_START_TOKEN_ID,
        "im_end_token_id": QWEN_IM_END_TOKEN_ID,
    }
    if token_contract != expected_tokens:
        failures.append(f"tokenizer={token_contract!r}, expected {expected_tokens!r}")

    text_config = load_qwen_text_config(cache_dir=cache_dir, local_files_only=True)
    topology = topology_summary(text_config)
    package_versions = assert_locked_model_stack()
    if failures:
        raise SnapshotContractError("snapshot audit failed: " + "; ".join(failures))
    return {
        "status": "pass",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot": str(snapshot),
        "weight_bytes": actual_weight_bytes,
        "checkpoint_key_counts": {
            "text": counts[0],
            "vision": counts[1],
            "mtp": counts[2],
        },
        "critical_sha256": dict(EXPECTED_CRITICAL_HASHES),
        "weight_shards": shard_identities,
        "tokenizer": token_contract,
        "topology": {
            "layers": topology.num_layers,
            "linear_attention_layers": len(topology.linear_attention_layers),
            "full_attention_layers": len(topology.full_attention_layers),
            "lora_targets": topology.target_module_count,
            "rank32_trainable_parameters": topology.trainable_parameters,
        },
        "packages": package_versions,
    }


__all__ = [
    "EXPECTED_CRITICAL_HASHES",
    "EXPECTED_WEIGHT_FILE_BYTES",
    "EXPECTED_WEIGHT_BYTES",
    "SnapshotContractError",
    "audit_snapshot",
    "download_snapshot",
]
