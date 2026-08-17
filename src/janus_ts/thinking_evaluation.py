"""Complete-test thinking evaluations for raw and fine-tuned Qwen.

The underlying reasoning trace remains an explicitly non-formal exploratory
artifact.  This module scores only the final answer after Qwen's closing
``</think>`` marker and cannot affect checkpoint selection or the formal test
lease.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .artifacts import (
    ArtifactError,
    mark_complete,
    read_complete_manifest,
    sha256_file,
    write_json,
)
from .config import ExperimentConfig, load_config
from .evaluation import aggregate_evaluations, evaluate_reaction
from .formal_eval_runtime import (
    DurableCheckpoint,
    inspect_durable_checkpoint,
)
from .schema import ReactionRecord
from .thinking_inference import (
    ThinkingExplorationReceipt,
    load_exploration_records,
    validate_exploration_artifact,
    validate_thinking_profile,
)
from .vllm_thinking import run_vllm_thinking_inference
from .zero_shot_runtime import (
    ZeroShotBaseline,
    build_zero_shot_baseline,
)

THINKING_EVALUATION_SCHEMA_VERSION = "janus-ts-thinking-test-evaluation-v3"
THINKING_SCORE_ROW_SCHEMA_VERSION = "janus-ts-thinking-score-row-v3"
THINKING_RUNTIME_RECEIPT_SCHEMA_VERSION = "janus-ts-thinking-runtime-receipt-v3"
THINKING_TEST_REPORT_K = (1,)
THINKING_TEST_SAMPLE_COUNT = 1
ThinkingRole = Literal["zero-shot", "fine-tuned"]
_ROLES: tuple[ThinkingRole, ...] = ("zero-shot", "fine-tuned")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ThinkingEvaluationError(RuntimeError):
    """A complete-test thinking evaluation violated its protocol."""


@dataclass(frozen=True, slots=True)
class ThinkingModelBinding:
    role: ThinkingRole
    data_fingerprint: str
    run_fingerprint: str
    model_fingerprint: str
    checkpoint_fingerprint: str
    epoch: float
    global_step: int
    checkpoint_path: Path | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "data_fingerprint": self.data_fingerprint,
            "run_fingerprint": self.run_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "checkpoint_path": (
                str(self.checkpoint_path.resolve()) if self.checkpoint_path is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ThinkingEvaluationReceipt:
    path: Path
    payload: Mapping[str, Any]

    @property
    def predictions_path(self) -> Path:
        return Path(str(self.payload["scored_predictions"]["path"]))

    @property
    def metrics_path(self) -> Path:
        return Path(str(self.payload["metrics"]["path"]))


@dataclass(frozen=True, slots=True)
class _ZeroShotThinkingCheckpoint:
    path: Path
    checkpoint_fingerprint: str
    data_fingerprint: str
    run_fingerprint: str
    model_fingerprint: str
    epoch: int = 0
    global_step: int = 0


def _require_role(role: str) -> ThinkingRole:
    if role not in _ROLES:
        raise ThinkingEvaluationError(f"thinking role must be one of {_ROLES}, got {role!r}")
    return role  # type: ignore[return-value]


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ThinkingEvaluationError(f"{name} must be a lowercase SHA256 digest")
    return value


def _binding_from_zero_shot(baseline: ZeroShotBaseline) -> ThinkingModelBinding:
    return ThinkingModelBinding(
        role="zero-shot",
        data_fingerprint=baseline.data_fingerprint,
        run_fingerprint=baseline.run_fingerprint,
        model_fingerprint=baseline.model_fingerprint,
        checkpoint_fingerprint=baseline.checkpoint_fingerprint,
        epoch=0,
        global_step=0,
        checkpoint_path=None,
    )


def _binding_from_checkpoint(checkpoint: DurableCheckpoint) -> ThinkingModelBinding:
    return ThinkingModelBinding(
        role="fine-tuned",
        data_fingerprint=checkpoint.data_fingerprint,
        run_fingerprint=checkpoint.run_fingerprint,
        model_fingerprint=checkpoint.model_fingerprint,
        checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
        epoch=checkpoint.epoch,
        global_step=checkpoint.global_step,
        checkpoint_path=checkpoint.path,
    )


def build_thinking_binding(
    config: ExperimentConfig,
    processed_path: str | Path,
    *,
    run_fingerprint: str,
    role: ThinkingRole,
    checkpoint_dir: str | Path | None = None,
) -> tuple[ThinkingModelBinding, ZeroShotBaseline | DurableCheckpoint]:
    """Resolve the exact raw model or selected portable checkpoint."""

    _require_sha256(run_fingerprint, name="run_fingerprint")
    role = _require_role(role)
    if role == "zero-shot":
        if checkpoint_dir is not None:
            raise ThinkingEvaluationError("zero-shot thinking must not receive a checkpoint")
        model = build_zero_shot_baseline(
            config,
            processed_path,
            run_fingerprint=run_fingerprint,
        )
        return _binding_from_zero_shot(model), model
    if checkpoint_dir is None:
        raise ThinkingEvaluationError("fine-tuned thinking requires the selected checkpoint")
    model = inspect_durable_checkpoint(checkpoint_dir, processed_path, config=config)
    if model.run_fingerprint != run_fingerprint:
        raise ThinkingEvaluationError("selected checkpoint belongs to a different run")
    return _binding_from_checkpoint(model), model


def extract_thinking_answer(raw_response: str) -> tuple[str, str | None]:
    """Return the final answer after one closing think marker."""

    normalized = raw_response.replace("\r\n", "\n").replace("\r", "\n")
    count = normalized.count("</think>")
    if count == 0:
        return "", "missing_closing_think"
    if count != 1:
        return "", "multiple_closing_think"
    reasoning, answer = normalized.split("</think>", 1)
    if "<think>" in reasoning:
        return "", "nested_opening_think"
    answer = answer.lstrip("\n").strip()
    if not answer:
        return "", "empty_final_answer"
    return answer, None


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) for row in rows
        )
        + "\n"
    ).encode("utf-8")


def _read_raw_predictions(receipt: ThinkingExplorationReceipt) -> list[dict[str, Any]]:
    path = receipt.path / "predictions.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ThinkingEvaluationError(f"cannot read thinking predictions: {exc}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ThinkingEvaluationError("thinking predictions are empty or malformed")
    return rows


def _score_root(
    output_dir: str | Path,
    role: ThinkingRole,
    request_sha256: str,
) -> Path:
    return Path(output_dir) / "thinking-evaluations" / role / request_sha256


def finalize_thinking_scores(
    config: ExperimentConfig,
    output_dir: str | Path,
    raw_receipt: ThinkingExplorationReceipt,
    binding: ThinkingModelBinding,
    records: Sequence[ReactionRecord],
) -> Path:
    """Extract and score all final answers, then seal a supplemental artifact."""

    request_sha256 = _require_sha256(
        raw_receipt.payload.get("request_sha256"), name="thinking request fingerprint"
    )
    destination = _score_root(output_dir, binding.role, request_sha256)
    if (destination / ".complete").is_file():
        return destination
    expected_count = config.data.expected_retained_counts["test"]
    if (
        len(records) != expected_count
        or raw_receipt.payload.get("selection_count") != expected_count
        or raw_receipt.payload.get("sample_count_per_reaction") != THINKING_TEST_SAMPLE_COUNT
    ):
        raise ThinkingEvaluationError(
            f"thinking test requires all {expected_count} records and predictions"
        )
    raw_rows = _read_raw_predictions(raw_receipt)
    if len(raw_rows) != expected_count:
        raise ThinkingEvaluationError(f"thinking prediction count is {len(raw_rows)}")

    scored_rows: list[dict[str, Any]] = []
    evaluations = []
    extraction_errors: dict[str, int] = {}
    parse_valid_candidate_count = 0
    for ordinal, (record, raw) in enumerate(zip(records, raw_rows, strict=True)):
        if raw.get("ordinal") != ordinal or raw.get("reaction_id") != record.reaction_id:
            raise ThinkingEvaluationError(f"thinking prediction identity mismatch at {ordinal}")
        raw_responses = raw.get("raw_responses")
        if (
            raw.get("sample_count") != THINKING_TEST_SAMPLE_COUNT
            or not isinstance(raw_responses, list)
            or len(raw_responses) != THINKING_TEST_SAMPLE_COUNT
            or any(not isinstance(response, str) for response in raw_responses)
        ):
            raise ThinkingEvaluationError(
                f"thinking prediction {ordinal} does not contain "
                f"{THINKING_TEST_SAMPLE_COUNT} ordered responses"
            )
        answers: list[str] = []
        response_errors: list[str | None] = []
        for raw_response in raw_responses:
            answer, extraction_error = extract_thinking_answer(raw_response)
            answers.append(answer)
            response_errors.append(extraction_error)
            if extraction_error is not None:
                extraction_errors[extraction_error] = extraction_errors.get(extraction_error, 0) + 1
        evaluation = evaluate_reaction(
            record.reaction_id,
            tuple(answers),
            record.ts_edges,
            atom_count=record.atom_count,
            report_k=THINKING_TEST_REPORT_K,
        )
        evaluations.append(evaluation)
        parse_valid_candidate_count += sum(parse.valid for parse in evaluation.parses)
        scored_rows.append(
            {
                "schema_version": THINKING_SCORE_ROW_SCHEMA_VERSION,
                "request_sha256": request_sha256,
                "role": binding.role,
                "ordinal": ordinal,
                "reaction_id": record.reaction_id,
                "sample_count": THINKING_TEST_SAMPLE_COUNT,
                "answers": answers,
                "extraction_errors": response_errors,
                "parse_valid": [parse.valid for parse in evaluation.parses],
                "parse_error_codes": [parse.error_code for parse in evaluation.parses],
            }
        )

    report = aggregate_evaluations(evaluations, report_k=THINKING_TEST_REPORT_K)
    predictions = destination / "scored-predictions.jsonl"
    metrics = destination / "metrics.json"
    _atomic_write(predictions, _jsonl_bytes(scored_rows))
    write_json(
        metrics,
        {
            "schema_version": THINKING_EVALUATION_SCHEMA_VERSION,
            **binding.to_json_dict(),
            "formal_eligible": False,
            "affects_checkpoint_selection": False,
            "split": "test",
            "sample_count_per_reaction": THINKING_TEST_SAMPLE_COUNT,
            "report_k": list(THINKING_TEST_REPORT_K),
            "predictions_sha256": sha256_file(predictions),
            "raw_predictions_sha256": sha256_file(raw_receipt.path / "predictions.jsonl"),
            "extraction_errors": dict(sorted(extraction_errors.items())),
            "parse_valid_candidate_count": parse_valid_candidate_count,
            "evaluation": report.to_json_dict(),
        },
    )
    inventory = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (predictions, metrics)
    }
    mark_complete(
        destination,
        {
            "schema_version": THINKING_EVALUATION_SCHEMA_VERSION,
            **binding.to_json_dict(),
            "formal_eligible": False,
            "affects_checkpoint_selection": False,
            "split": "test",
            "test_count": expected_count,
            "sample_count_per_reaction": THINKING_TEST_SAMPLE_COUNT,
            "report_k": list(THINKING_TEST_REPORT_K),
            "request_sha256": request_sha256,
            "raw_artifact_path": str(raw_receipt.path.resolve()),
            "raw_predictions_sha256": sha256_file(raw_receipt.path / "predictions.jsonl"),
            "payload_inventory": inventory,
        },
    )
    return destination


def thinking_receipt_path(
    output_dir: str | Path,
    binding: ThinkingModelBinding,
) -> Path:
    return Path(output_dir) / (
        f"thinking-{binding.role}.{binding.checkpoint_fingerprint}.runtime.json"
    )


def _validate_scored_artifact(path: Path, binding: ThinkingModelBinding) -> Mapping[str, Any]:
    try:
        manifest = read_complete_manifest(path)
    except ArtifactError as exc:
        raise ThinkingEvaluationError(str(exc)) from exc
    expected = {
        "schema_version": THINKING_EVALUATION_SCHEMA_VERSION,
        **binding.to_json_dict(),
        "formal_eligible": False,
        "affects_checkpoint_selection": False,
        "split": "test",
        "sample_count_per_reaction": THINKING_TEST_SAMPLE_COUNT,
        "report_k": list(THINKING_TEST_REPORT_K),
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ThinkingEvaluationError("thinking score artifact identity mismatch")
    inventory = manifest.get("payload_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != {
        "scored-predictions.jsonl",
        "metrics.json",
    }:
        raise ThinkingEvaluationError("thinking score artifact inventory is invalid")
    for name, raw_entry in inventory.items():
        path_value = path / name
        if (
            not isinstance(raw_entry, Mapping)
            or not path_value.is_file()
            or path_value.is_symlink()
            or raw_entry.get("size_bytes") != path_value.stat().st_size
            or raw_entry.get("sha256") != sha256_file(path_value)
        ):
            raise ThinkingEvaluationError(f"thinking score payload failed verification: {name}")
    return manifest


def validate_thinking_completion(
    output_dir: str | Path,
    binding: ThinkingModelBinding,
) -> ThinkingEvaluationReceipt | None:
    """Validate a completed runtime pointer and both referenced artifacts."""

    path = thinking_receipt_path(output_dir, binding)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThinkingEvaluationError(f"cannot read thinking receipt: {exc}") from exc
    expected = {
        "schema_version": THINKING_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        **binding.to_json_dict(),
        "sample_count_per_reaction": THINKING_TEST_SAMPLE_COUNT,
        "report_k": list(THINKING_TEST_REPORT_K),
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ThinkingEvaluationError("thinking runtime receipt identity mismatch")
    raw_path = Path(str(payload.get("raw_artifact_path", "")))
    score_path = Path(str(payload.get("score_artifact_path", "")))
    raw = validate_exploration_artifact(
        raw_path,
        request_sha256=str(payload.get("request_sha256", "")),
    )
    if raw.payload.get("sample_count_per_reaction") != THINKING_TEST_SAMPLE_COUNT:
        raise ThinkingEvaluationError("thinking runtime receipt references the wrong sample count")
    manifest = _validate_scored_artifact(score_path, binding)
    if manifest.get("request_sha256") != raw.payload.get("request_sha256") or payload.get(
        "raw_predictions_sha256"
    ) != sha256_file(raw.path / "predictions.jsonl"):
        raise ThinkingEvaluationError("thinking runtime receipt references stale predictions")
    predictions = score_path / "scored-predictions.jsonl"
    metrics = score_path / "metrics.json"
    for name, value in (("scored_predictions", predictions), ("metrics", metrics)):
        reference = payload.get(name)
        if (
            not isinstance(reference, Mapping)
            or reference.get("path") != str(value.resolve())
            or reference.get("sha256") != sha256_file(value)
        ):
            raise ThinkingEvaluationError(f"thinking receipt has a stale {name} reference")
    return ThinkingEvaluationReceipt(path=path, payload=payload)


def run_thinking_test_evaluation(
    config: ExperimentConfig,
    *,
    processed_path: str | Path,
    output_dir: str | Path,
    run_fingerprint: str,
    role: ThinkingRole,
    checkpoint_dir: str | Path | None = None,
    local_files_only: bool = True,
) -> ThinkingEvaluationReceipt:
    """Run one complete thinking test through the pinned two-GPU vLLM runtime."""

    validate_thinking_profile(config.thinking_generation)
    binding, model = build_thinking_binding(
        config,
        processed_path,
        run_fingerprint=run_fingerprint,
        role=role,
        checkpoint_dir=checkpoint_dir,
    )
    recovered = validate_thinking_completion(output_dir, binding)
    if recovered is not None:
        return recovered

    if binding.role == "zero-shot":
        baseline = model
        if not isinstance(baseline, ZeroShotBaseline):  # pragma: no cover - narrowed by role
            raise ThinkingEvaluationError("zero-shot binding did not resolve the raw model")
        synthetic = _ZeroShotThinkingCheckpoint(
            path=Path(config.model.cache_dir) / "models--Qwen--Qwen3.6-27B",
            checkpoint_fingerprint=binding.checkpoint_fingerprint,
            data_fingerprint=binding.data_fingerprint,
            run_fingerprint=binding.run_fingerprint,
            model_fingerprint=binding.model_fingerprint,
        )
        raw_receipt = run_vllm_thinking_inference(
            config,
            processed_path=processed_path,
            checkpoint=synthetic,
            output_dir=output_dir,
            split="test",
            limit=config.data.expected_retained_counts["test"],
            sample_count_per_reaction=THINKING_TEST_SAMPLE_COUNT,
            portable_adapter=None,
            local_files_only=local_files_only,
        )
    else:
        checkpoint = model
        if not isinstance(checkpoint, DurableCheckpoint):  # pragma: no cover - narrowed by role
            raise ThinkingEvaluationError("fine-tuned binding did not resolve a checkpoint")
        raw_receipt = run_vllm_thinking_inference(
            config,
            processed_path=processed_path,
            checkpoint=checkpoint,
            output_dir=output_dir,
            split="test",
            limit=config.data.expected_retained_counts["test"],
            sample_count_per_reaction=THINKING_TEST_SAMPLE_COUNT,
            portable_adapter=checkpoint.portable_adapter_path,
            local_files_only=local_files_only,
        )

    records = load_exploration_records(config, processed_path, "test")
    score_path = finalize_thinking_scores(
        config,
        output_dir,
        raw_receipt,
        binding,
        records,
    )
    request_sha256 = str(raw_receipt.payload["request_sha256"])
    predictions = score_path / "scored-predictions.jsonl"
    metrics = score_path / "metrics.json"
    receipt_payload = {
        "schema_version": THINKING_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        **binding.to_json_dict(),
        "sample_count_per_reaction": THINKING_TEST_SAMPLE_COUNT,
        "report_k": list(THINKING_TEST_REPORT_K),
        "request_sha256": request_sha256,
        "raw_artifact_path": str(raw_receipt.path.resolve()),
        "raw_predictions_sha256": sha256_file(raw_receipt.path / "predictions.jsonl"),
        "score_artifact_path": str(score_path.resolve()),
        "scored_predictions": {
            "path": str(predictions.resolve()),
            "sha256": sha256_file(predictions),
        },
        "metrics": {"path": str(metrics.resolve()), "sha256": sha256_file(metrics)},
    }
    path = thinking_receipt_path(output_dir, binding)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != receipt_payload:
            raise ThinkingEvaluationError("existing thinking receipt differs")
    else:
        write_json(path, receipt_payload)
    receipt = validate_thinking_completion(output_dir, binding)
    if receipt is None:
        raise ThinkingEvaluationError("thinking evaluation returned without completion")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=_ROLES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-fingerprint", required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = run_thinking_test_evaluation(
        load_config(arguments.config),
        processed_path=arguments.processed_path,
        output_dir=arguments.output_dir,
        run_fingerprint=arguments.run_fingerprint,
        role=arguments.role,
        checkpoint_dir=arguments.checkpoint_dir,
        local_files_only=True,
    )
    print(json.dumps(receipt.payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by torchrun
    raise SystemExit(main())


__all__ = [
    "THINKING_EVALUATION_SCHEMA_VERSION",
    "THINKING_RUNTIME_RECEIPT_SCHEMA_VERSION",
    "THINKING_TEST_REPORT_K",
    "THINKING_TEST_SAMPLE_COUNT",
    "ThinkingEvaluationError",
    "ThinkingEvaluationReceipt",
    "ThinkingModelBinding",
    "build_thinking_binding",
    "extract_thinking_answer",
    "finalize_thinking_scores",
    "run_thinking_test_evaluation",
    "thinking_receipt_path",
    "validate_thinking_completion",
]
