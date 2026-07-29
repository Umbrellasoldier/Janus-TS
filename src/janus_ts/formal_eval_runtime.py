"""Two-rank ZeRO-3 runtime for formal checkpoint evaluation.

Each invocation evaluates one durable candidate checkpoint.  The ordinary rank-64
portable adapter is loaded over the pinned *original* Qwen base (not the PiSSA
residual base) after :class:`~transformers.TrainingArguments` has activated
ZeRO-3 Init.  Validation first obtains ``eval_loss`` through ``Trainer`` and
then performs the frozen beam-10 generation.  Test generation is authorized by
the immutable selection proof and is recoverably idempotent after completion.

The checkpoint fingerprint is the SHA256 of its completed ``manifest.json``.
No decision in this module uses filesystem timestamps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from .artifacts import (
    ArtifactError,
    read_complete_manifest,
    sha256_file,
    write_json,
)
from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    PORTABLE_ADAPTER_SUBDIR,
)
from .config import ExperimentConfig, load_config
from .constants import (
    QWEN_IM_END_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
)
from .distributed import (
    TorchrunContext,
    assert_initialized_two_rank_job,
    assert_zero3_no_offload,
    configure_reproducibility,
    preflight_torchrun_environment,
)
from .evaluation import CheckpointScore
from .generation import (
    EVALUATION_SCHEMA_VERSION,
    EXPECTED_FORMAL_SPLIT_COUNTS,
    TEST_COMPLETION_SCHEMA_VERSION,
    FormalEvaluationResult,
    GenerationIdentity,
    TestEvaluationLease,
    load_selection_proof,
    merged_predictions_path,
    metrics_path,
    run_formal_generation,
    trainer_eval_loss_hook,
)
from .metrics import AggregateMetrics
from .modeling import (
    EXPECTED_PORTABLE_PARAMETERS,
    load_qwen_text_base,
    validate_portable_adapter_config,
)
from .preprocessing import (
    load_pinned_tokenizer,
    load_processed_dataset,
    reaction_record_from_row,
)
from .runtime import install_frozen_environment
from .schema import ReactionRecord
from .tokenization import (
    CausalLMCollator,
    EpochAwareTokenizedDataset,
    QwenTrainingEncoder,
)
from .training import (
    TokenNormalizedTrainer,
    assert_accelerate_zero3_precision_state,
    assert_bf16_model_parameters,
    assert_zero3_bf16_precision,
    build_training_argument_kwargs,
)

FORMAL_RUNTIME_RECEIPT_SCHEMA_VERSION = "janus-ts-formal-eval-runtime-receipt-v1"
Split = Literal["val", "test"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PORTABLE_FILES = {
    f"{PORTABLE_ADAPTER_SUBDIR}/adapter_config.json",
    f"{PORTABLE_ADAPTER_SUBDIR}/adapter_model.safetensors",
}


class FormalEvalRuntimeError(RuntimeError):
    """A formal evaluation runtime invariant was violated."""


@dataclass(frozen=True, slots=True)
class DurableCheckpoint:
    """Content-verified identity of one durable evaluation checkpoint."""

    path: Path
    portable_adapter_path: Path
    checkpoint_fingerprint: str
    data_fingerprint: str
    run_fingerprint: str
    config_fingerprint: str
    model_fingerprint: str
    epoch: float
    global_step: int
    manifest: Mapping[str, Any]

    def generation_identity(self, split: Split) -> GenerationIdentity:
        return GenerationIdentity(
            split=split,
            data_fingerprint=self.data_fingerprint,
            run_fingerprint=self.run_fingerprint,
            checkpoint_fingerprint=self.checkpoint_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class PreparedFormalData:
    """Canonical records plus the optional causal-loss dataset."""

    records: tuple[ReactionRecord, ...]
    eval_dataset: Any | None
    collator: Any


@dataclass(frozen=True, slots=True)
class FormalRuntimeReceipt:
    """Small immutable receipt consumed by the workflow selector."""

    path: Path
    payload: Mapping[str, Any]

    @property
    def split(self) -> str:
        return str(self.payload["split"])

    @property
    def checkpoint_fingerprint(self) -> str:
        return str(self.payload["checkpoint_fingerprint"])


class _LoraBf16OutputProbe:
    """Verify that explicit DeepSpeed autocast reaches portable LoRA linears."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self._handles: list[Any] = []
        self._observed: dict[str, Any] = {}

    @staticmethod
    def _label(name: str) -> str | None:
        if ".lora_A." in name or name.startswith("lora_A."):
            return "lora_A"
        if ".lora_B." in name or name.startswith("lora_B."):
            return "lora_B"
        return None

    def start(self) -> None:
        selected: dict[str, tuple[str, Any]] = {}
        for name, child in self.module.named_modules():
            label = self._label(name)
            if label is not None and label not in selected:
                selected[label] = (name, child)
            if set(selected) == {"lora_A", "lora_B"}:
                break
        if set(selected) != {"lora_A", "lora_B"}:
            raise FormalEvalRuntimeError("cannot locate portable LoRA A/B modules for dtype probe")

        def hook(label: str, name: str) -> Callable[..., None]:
            def record(_module: Any, _inputs: Any, output: Any) -> None:
                value = output[0] if isinstance(output, (tuple, list)) and output else output
                self._observed[label] = (name, getattr(value, "dtype", None))

            return record

        for label, (name, child) in selected.items():
            self._handles.append(child.register_forward_hook(hook(label, name)))

    def abort(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def finish(self) -> None:
        import torch

        self.abort()
        problems = []
        for label in ("lora_A", "lora_B"):
            observed = self._observed.get(label)
            if observed is None:
                problems.append(f"{label} forward hook did not fire")
            elif observed[1] != torch.bfloat16:
                problems.append(f"{observed[0]} output dtype={observed[1]}")
        if problems:
            raise FormalEvalRuntimeError(
                "portable LoRA generation did not compute in BF16: " + "; ".join(problems)
            )


class DeepSpeedGenerationProxy:
    """Call ``engine.module.generate`` inside DeepSpeed's autocast context."""

    def __init__(
        self,
        engine: Any,
        *,
        autocast_context: Callable[[Any], Any] | None = None,
        probe_factory: Callable[[Any], Any] | None = _LoraBf16OutputProbe,
    ) -> None:
        module = getattr(engine, "module", None)
        if module is None or not callable(getattr(module, "generate", None)):
            raise FormalEvalRuntimeError("DeepSpeed engine has no generate-capable module")
        if autocast_context is None:
            from deepspeed.runtime.torch_autocast import autocast_if_enabled

            autocast_context = autocast_if_enabled
        self.engine = engine
        self.module = module
        self._autocast_context = autocast_context
        self._probe_factory = probe_factory
        self._bf16_compute_verified = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        probe = None
        if not self._bf16_compute_verified and self._probe_factory is not None:
            probe = self._probe_factory(self.module)
            probe.start()
        try:
            with self._autocast_context(self.engine):
                result = self.module.generate(*args, **kwargs)
        except BaseException:
            if probe is not None:
                probe.abort()
            raise
        if probe is not None:
            probe.finish()
            self._bf16_compute_verified = True
        return result


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FormalEvalRuntimeError(f"{name} must be a lowercase SHA256 digest")
    return value


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalEvalRuntimeError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FormalEvalRuntimeError(f"{description} must be a JSON object: {path}")
    return payload


def _verify_portable_inventory(root: Path, manifest: Mapping[str, Any]) -> Path:
    inventory = manifest.get("payload_inventory")
    if not isinstance(inventory, Mapping) or not inventory:
        raise FormalEvalRuntimeError("durable checkpoint has no payload inventory")
    expected_entries = {name for name in inventory if str(name).startswith("portable_adapter/")}
    if expected_entries != _PORTABLE_FILES:
        raise FormalEvalRuntimeError(
            "portable adapter inventory differs from the frozen two-file layout: "
            f"{sorted(expected_entries)!r}"
        )
    adapter_root = root / PORTABLE_ADAPTER_SUBDIR
    if not adapter_root.is_dir() or adapter_root.is_symlink():
        raise FormalEvalRuntimeError(f"portable adapter directory is invalid: {adapter_root}")
    actual_entries = {
        path.relative_to(root).as_posix() for path in adapter_root.rglob("*") if path.is_file()
    }
    if actual_entries != _PORTABLE_FILES:
        raise FormalEvalRuntimeError(f"portable adapter files differ: {sorted(actual_entries)!r}")
    for relative in sorted(_PORTABLE_FILES):
        path = root / relative
        entry = inventory.get(relative)
        if path.is_symlink() or not isinstance(entry, Mapping):
            raise FormalEvalRuntimeError(f"unsafe portable adapter inventory entry: {relative}")
        expected_size = entry.get("size_bytes")
        expected_hash = entry.get("sha256")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or path.stat().st_size != expected_size
        ):
            raise FormalEvalRuntimeError(f"portable adapter size mismatch: {relative}")
        _require_sha256(expected_hash, name=f"payload_inventory[{relative}].sha256")
        if sha256_file(path) != expected_hash:
            raise FormalEvalRuntimeError(f"portable adapter digest mismatch: {relative}")
    return adapter_root


def inspect_durable_checkpoint(
    checkpoint_dir: str | Path,
    processed_path: str | Path,
    *,
    config: ExperimentConfig | None = None,
) -> DurableCheckpoint:
    """Validate identities and the portable payload without consulting mtime."""

    root = Path(checkpoint_dir).resolve(strict=True)
    processed_root = Path(processed_path).resolve(strict=True)
    try:
        manifest = read_complete_manifest(root)
        data_manifest = read_complete_manifest(processed_root)
    except ArtifactError as exc:
        raise FormalEvalRuntimeError(str(exc)) from exc

    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise FormalEvalRuntimeError("unsupported durable checkpoint schema")
    kind = manifest.get("kind")
    if kind not in {"epoch", "train-loss"}:
        raise FormalEvalRuntimeError(
            "formal evaluation requires an epoch or train-loss checkpoint"
        )
    raw_epoch = manifest.get("epoch")
    try:
        epoch_float = float(raw_epoch)
    except (TypeError, ValueError) as exc:
        raise FormalEvalRuntimeError(f"invalid durable epoch: {raw_epoch!r}") from exc
    if not math.isfinite(epoch_float) or epoch_float <= 0:
        raise FormalEvalRuntimeError(f"durable epoch is not positive and finite: {raw_epoch!r}")
    if kind == "epoch" and not math.isclose(
        epoch_float,
        round(epoch_float),
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise FormalEvalRuntimeError(f"durable epoch is not an integer: {raw_epoch!r}")
    epoch = float(round(epoch_float)) if kind == "epoch" else epoch_float
    if config is not None and epoch > config.train.epochs:
        raise FormalEvalRuntimeError(
            f"checkpoint epoch {epoch} exceeds configured {config.train.epochs} epochs"
        )
    global_step = manifest.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise FormalEvalRuntimeError(f"invalid durable global_step: {global_step!r}")
    if manifest.get("world_size") != 2:
        raise FormalEvalRuntimeError("durable checkpoint was not produced by exactly two ranks")
    if manifest.get("exclude_frozen_parameters") is not True:
        raise FormalEvalRuntimeError("durable checkpoint did not exclude the frozen base")
    if kind == "train-loss":
        train_loss = manifest.get("train_loss")
        if (
            isinstance(train_loss, bool)
            or not isinstance(train_loss, (int, float))
            or not math.isfinite(float(train_loss))
            or float(train_loss) < 0.0
        ):
            raise FormalEvalRuntimeError("train-loss checkpoint has an invalid train_loss")

    data_fingerprint = _require_sha256(
        manifest.get("data_fingerprint"), name="checkpoint data_fingerprint"
    )
    processed_fingerprint = _require_sha256(
        data_manifest.get("fingerprint"), name="processed data fingerprint"
    )
    if data_fingerprint != processed_fingerprint:
        raise FormalEvalRuntimeError("checkpoint and processed-data fingerprints differ")
    config_fingerprint = _require_sha256(
        manifest.get("config_fingerprint"), name="checkpoint config_fingerprint"
    )
    if config is not None and config_fingerprint != config.sha256:
        raise FormalEvalRuntimeError("checkpoint and experiment configuration fingerprints differ")
    run_fingerprint = _require_sha256(
        manifest.get("run_fingerprint"), name="checkpoint run_fingerprint"
    )
    model_fingerprint = _require_sha256(
        manifest.get("model_fingerprint"), name="checkpoint model_fingerprint"
    )
    _require_sha256(
        manifest.get("pissa_initial_adapter_fingerprint"),
        name="checkpoint PiSSA-initial fingerprint",
    )
    portable = _verify_portable_inventory(root, manifest)
    checkpoint_fingerprint = sha256_file(root / "manifest.json")
    return DurableCheckpoint(
        path=root,
        portable_adapter_path=portable,
        checkpoint_fingerprint=checkpoint_fingerprint,
        data_fingerprint=data_fingerprint,
        run_fingerprint=run_fingerprint,
        config_fingerprint=config_fingerprint,
        model_fingerprint=model_fingerprint,
        epoch=epoch,
        global_step=global_step,
        manifest=manifest,
    )


def build_formal_eval_arguments(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
) -> Any:
    """Create eval-only arguments; this must run before the 27B model loader."""

    kwargs = build_training_argument_kwargs(config, output_dir=Path(output_dir) / "trainer")
    kwargs.update(
        {
            "do_train": False,
            "do_eval": True,
            "per_device_eval_batch_size": 1,
            "eval_strategy": "no",
            "save_strategy": "no",
            "logging_strategy": "no",
            "report_to": [],
            "gradient_checkpointing": False,
            "prediction_loss_only": True,
        }
    )
    from transformers import TrainingArguments

    return TrainingArguments(**kwargs)


def _is_lora_parameter(name: str) -> bool:
    return (
        name.startswith("lora_A.")
        or name.startswith("lora_B.")
        or ".lora_A." in name
        or ".lora_B." in name
    )


def load_zero3_portable_model(
    config: ExperimentConfig,
    checkpoint: DurableCheckpoint,
    arguments: Any,
    *,
    local_files_only: bool = True,
    torch_module: Any | None = None,
    base_loader: Callable[..., Any] | None = None,
    adapter_loader: Callable[..., Any] | None = None,
    zero3_validator: Callable[[Any], None] = assert_zero3_no_offload,
    portable_config_validator: Callable[[str | Path, Any], None] = (
        validate_portable_adapter_config
    ),
    precision_validator: Callable[..., Mapping[str, Any]] = assert_zero3_bf16_precision,
) -> Any:
    """Load original BF16 Qwen + ordinary rank-64 adapter under ZeRO-Init."""

    zero3_validator(arguments)
    if torch_module is None:
        import torch as torch_module

    if base_loader is None:
        base_loader = load_qwen_text_base
    base = base_loader(
        cache_dir=config.model.cache_dir,
        local_files_only=local_files_only,
        low_cpu_mem_usage=False,
    )
    if getattr(base, "hf_device_map", None):
        raise FormalEvalRuntimeError("formal ZeRO-3 base load unexpectedly created a device map")
    portable_config_validator(checkpoint.portable_adapter_path / "adapter_config.json", base.config)
    if adapter_loader is None:
        from peft import PeftModel

        adapter_loader = PeftModel.from_pretrained
    model = adapter_loader(
        base,
        checkpoint.portable_adapter_path,
        is_trainable=True,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=False,
    )
    if getattr(model, "hf_device_map", None):
        raise FormalEvalRuntimeError(
            "formal portable-adapter load unexpectedly created a device map"
        )

    for name, parameter in model.named_parameters():
        adapter = _is_lora_parameter(name)
        parameter.requires_grad_(adapter)
        if adapter and parameter.dtype != torch_module.bfloat16:
            parameter.data = parameter.data.to(dtype=torch_module.bfloat16)
    model.config.use_cache = False
    model.config.pad_token_id = QWEN_PAD_TOKEN_ID
    model.config.eos_token_id = QWEN_IM_END_TOKEN_ID
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.pad_token_id = QWEN_PAD_TOKEN_ID
        generation_config.eos_token_id = QWEN_IM_END_TOKEN_ID
    report = precision_validator(
        arguments,
        model,
        expected_trainable_parameters=EXPECTED_PORTABLE_PARAMETERS,
    )
    model._janus_ts_formal_precision_report = dict(report)
    model.eval()
    return model


def prepare_formal_data(
    config: ExperimentConfig,
    processed_path: str | Path,
    tokenizer: Any,
    split: Split,
) -> PreparedFormalData:
    """Memory-map one frozen split and prepare canonical loss examples."""

    datasets = load_processed_dataset(processed_path)
    if split not in datasets:
        raise FormalEvalRuntimeError(f"processed DatasetDict has no {split!r} split")
    dataset = datasets[split]
    expected = EXPECTED_FORMAL_SPLIT_COUNTS[split]
    if len(dataset) != expected:
        raise FormalEvalRuntimeError(
            f"formal {split} requires {expected} rows, found {len(dataset)}"
        )
    records = tuple(reaction_record_from_row(dataset[index]) for index in range(len(dataset)))
    if any(record.split != split for record in records):
        raise FormalEvalRuntimeError(f"formal {split} data contains a foreign split")
    encoder = QwenTrainingEncoder(
        tokenizer,
        max_length=config.model.max_sequence_length,
        seed=config.seed,
    )
    eval_dataset = (
        EpochAwareTokenizedDataset(dataset, encoder, training=False) if split == "val" else None
    )
    return PreparedFormalData(
        records=records,
        eval_dataset=eval_dataset,
        collator=CausalLMCollator(max_length=config.model.max_sequence_length),
    )


def build_formal_trainer(
    model: Any,
    arguments: Any,
    prepared: PreparedFormalData,
) -> Any:
    trainer = TokenNormalizedTrainer(
        model=model,
        args=arguments,
        eval_dataset=prepared.eval_dataset,
        data_collator=prepared.collator,
    )
    assert_accelerate_zero3_precision_state(arguments, trainer.accelerator)
    return trainer


def assert_formal_zero3_precision(
    engine: Any,
    *,
    expected_trainable_parameters: int = EXPECTED_PORTABLE_PARAMETERS,
) -> dict[str, Any]:
    """Audit eval-only native-BF16 ZeRO shards without an optimizer."""

    import torch
    from deepspeed.runtime.torch_autocast import get_comm_dtype

    required = (
        "bfloat16_enabled",
        "fp16_enabled",
        "torch_autocast_enabled",
        "zero_optimization_stage",
    )
    missing = [name for name in required if not callable(getattr(engine, name, None))]
    if missing:
        raise FormalEvalRuntimeError(f"formal engine lacks DeepSpeed methods: {missing!r}")
    if (
        engine.zero_optimization_stage() != 3
        or not engine.bfloat16_enabled()
        or engine.fp16_enabled()
        or engine.torch_autocast_enabled()
    ):
        raise FormalEvalRuntimeError("formal engine has the wrong native-BF16 precision mode")
    module = getattr(engine, "module", None)
    if module is None:
        raise FormalEvalRuntimeError("formal DeepSpeed engine has no module")
    parameter_report = assert_bf16_model_parameters(
        module,
        expected_trainable_parameters=expected_trainable_parameters,
    )
    problems: list[str] = []
    communication_dtypes: set[str] = set()
    for name, parameter in module.named_parameters():
        shard = getattr(parameter, "ds_tensor", None)
        if not hasattr(parameter, "ds_id") or shard is None:
            problems.append(f"unpartitioned parameter {name}")
            continue
        if parameter.is_floating_point() and shard.dtype != torch.bfloat16:
            problems.append(f"wrong shard dtype {name}={shard.dtype}")
        if _is_lora_parameter(name):
            communication_dtypes.add(str(get_comm_dtype(parameter)))
    if communication_dtypes != {str(torch.bfloat16)}:
        problems.append(f"LoRA communication dtypes={sorted(communication_dtypes)!r}")
    if problems:
        raise FormalEvalRuntimeError(
            "formal ZeRO-3 storage contract failed: " + "; ".join(problems[:12])
        )
    return {
        **parameter_report,
        "zero_stage": 3,
        "native_bf16": True,
        "native_fp16": False,
        "torch_autocast": False,
        "base_shard_dtype": "bfloat16",
        "adapter_shard_dtype": "bfloat16",
        "adapter_communication_dtype": "bfloat16",
    }


def initialize_inference_engine(trainer: Any) -> Any:
    """Initialize eval-only DeepSpeed without consuming test examples."""

    if not getattr(trainer, "is_deepspeed_enabled", False):
        raise FormalEvalRuntimeError("formal inference requires DeepSpeed")
    if getattr(trainer, "deepspeed", None) is None:
        from transformers.integrations.deepspeed import deepspeed_init

        deepspeed_init(trainer, num_training_steps=0, inference=True)
    model = trainer._wrap_model(trainer.model, training=False)
    if len(trainer.accelerator._models) == 0 and model is trainer.model:
        model = trainer.accelerator.prepare(model)
        if model is not trainer.model:
            trainer.model_wrapped = model
        trainer.deepspeed = trainer.model_wrapped
    engine = getattr(trainer, "deepspeed", None)
    if engine is None or getattr(engine, "module", None) is None:
        raise FormalEvalRuntimeError("DeepSpeed inference engine was not initialized")
    return engine


def generation_module_from_trainer(trainer: Any) -> DeepSpeedGenerationProxy:
    """Return an autocast proxy over the PEFT module retaining ZeRO hooks."""

    engine = getattr(trainer, "model_wrapped", None)
    module = getattr(engine, "module", None)
    if module is None:
        raise FormalEvalRuntimeError("formal generation requires an initialized engine.module")
    if not callable(getattr(module, "generate", None)):
        raise FormalEvalRuntimeError("underlying ZeRO module has no generate method")
    parameters = tuple(module.parameters())
    if not parameters or not any(hasattr(parameter, "ds_id") for parameter in parameters):
        raise FormalEvalRuntimeError("underlying generation module lacks ZeRO-3 parameter hooks")
    module._janus_ts_formal_engine_precision_report = assert_formal_zero3_precision(engine)
    module.config.use_cache = True
    generation_config = getattr(module, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = True
        generation_config.pad_token_id = QWEN_PAD_TOKEN_ID
        generation_config.eos_token_id = QWEN_IM_END_TOKEN_ID
    module.eval()
    return DeepSpeedGenerationProxy(engine)


def runtime_receipt_path(
    output_dir: str | Path,
    identity: GenerationIdentity,
) -> Path:
    return Path(output_dir) / f"{identity.split}.{identity.checkpoint_fingerprint}.runtime.json"


def _load_metrics_payload(
    path: Path,
    checkpoint: DurableCheckpoint,
    split: Split,
) -> dict[str, Any]:
    payload = _load_json_object(path, description="formal metrics")
    expected_keys = {
        "schema_version",
        "split",
        "data_fingerprint",
        "run_fingerprint",
        "checkpoint_fingerprint",
        "predictions_sha256",
        "eval_loss",
        "evaluation",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise FormalEvalRuntimeError("formal metrics has the wrong schema")
    expected_identity = checkpoint.generation_identity(split).to_json_dict()
    if any(payload.get(name) != value for name, value in expected_identity.items()):
        raise FormalEvalRuntimeError("formal metrics identity differs from the checkpoint")
    _require_sha256(payload.get("predictions_sha256"), name="predictions_sha256")
    loss = payload.get("eval_loss")
    if split == "val":
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise FormalEvalRuntimeError("validation metrics lacks eval_loss")
        if not math.isfinite(float(loss)) or float(loss) < 0.0:
            raise FormalEvalRuntimeError("validation eval_loss is invalid")
    elif loss is not None:
        raise FormalEvalRuntimeError("test metrics must record eval_loss=null")
    if not isinstance(payload.get("evaluation"), Mapping):
        raise FormalEvalRuntimeError("formal metrics has no evaluation report")
    return payload


def _receipt_payload(
    checkpoint: DurableCheckpoint,
    split: Split,
    *,
    predictions: Path,
    metrics: Path,
    eval_loss: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "split": split,
        "epoch": checkpoint.epoch,
        "global_step": checkpoint.global_step,
        "data_fingerprint": checkpoint.data_fingerprint,
        "run_fingerprint": checkpoint.run_fingerprint,
        "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
        "checkpoint_manifest_sha256": checkpoint.checkpoint_fingerprint,
        "predictions": {
            "filename": predictions.name,
            "sha256": sha256_file(predictions),
        },
        "metrics": {
            "filename": metrics.name,
            "sha256": sha256_file(metrics),
        },
        "eval_loss": eval_loss,
    }


def _validate_completion(
    output_dir: Path,
    checkpoint: DurableCheckpoint,
) -> tuple[Path, Path, dict[str, Any]] | None:
    identity = checkpoint.generation_identity("test")
    completion = TestEvaluationLease(output_dir, identity).completion_path
    if not completion.exists():
        return None
    payload = _load_json_object(completion, description="test completion")
    expected_keys = {
        "schema_version",
        "split",
        "data_fingerprint",
        "run_fingerprint",
        "checkpoint_fingerprint",
        "predictions_sha256",
        "metrics_sha256",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != TEST_COMPLETION_SCHEMA_VERSION
    ):
        raise FormalEvalRuntimeError("test completion has the wrong schema")
    if any(payload.get(name) != value for name, value in identity.to_json_dict().items()):
        raise FormalEvalRuntimeError("test completion identity differs from the checkpoint")
    predictions = merged_predictions_path(output_dir, identity)
    metrics = metrics_path(output_dir, identity)
    if not predictions.is_file() or not metrics.is_file():
        raise FormalEvalRuntimeError("completed test is missing predictions or metrics")
    if sha256_file(predictions) != payload.get("predictions_sha256"):
        raise FormalEvalRuntimeError("completed test prediction digest mismatch")
    if sha256_file(metrics) != payload.get("metrics_sha256"):
        raise FormalEvalRuntimeError("completed test metrics digest mismatch")
    metrics_payload = _load_metrics_payload(metrics, checkpoint, "test")
    if metrics_payload["predictions_sha256"] != payload["predictions_sha256"]:
        raise FormalEvalRuntimeError("test completion and metrics prediction hashes differ")
    return predictions, metrics, payload


def _install_or_validate_receipt(
    output_dir: Path,
    checkpoint: DurableCheckpoint,
    split: Split,
    *,
    predictions: Path,
    metrics: Path,
    eval_loss: float | None,
) -> FormalRuntimeReceipt:
    identity = checkpoint.generation_identity(split)
    path = runtime_receipt_path(output_dir, identity)
    expected = _receipt_payload(
        checkpoint,
        split,
        predictions=predictions,
        metrics=metrics,
        eval_loss=eval_loss,
    )
    if path.exists():
        actual = _load_json_object(path, description="formal runtime receipt")
        if actual != expected:
            raise FormalEvalRuntimeError("existing formal runtime receipt differs from artifacts")
    else:
        write_json(path, expected)
    return FormalRuntimeReceipt(path=path, payload=expected)


def recover_completed_test(
    output_dir: str | Path,
    checkpoint: DurableCheckpoint,
) -> FormalRuntimeReceipt | None:
    """Return/rebuild a receipt for a locked test without loading the model."""

    root = Path(output_dir)
    completed = _validate_completion(root, checkpoint)
    if completed is None:
        return None
    predictions, metrics, _ = completed
    return _install_or_validate_receipt(
        root,
        checkpoint,
        "test",
        predictions=predictions,
        metrics=metrics,
        eval_loss=None,
    )


def validate_runtime_receipt(
    receipt_path: str | Path,
    checkpoint_dir: str | Path,
    processed_path: str | Path,
    *,
    config: ExperimentConfig | None = None,
) -> FormalRuntimeReceipt:
    """Strictly validate a receipt and both content-addressed result files."""

    checkpoint = inspect_durable_checkpoint(checkpoint_dir, processed_path, config=config)
    path = Path(receipt_path)
    payload = _load_json_object(path, description="formal runtime receipt")
    split = payload.get("split")
    if split not in ("val", "test"):
        raise FormalEvalRuntimeError(f"invalid receipt split: {split!r}")
    identity = checkpoint.generation_identity(split)
    if path.name != runtime_receipt_path(path.parent, identity).name:
        raise FormalEvalRuntimeError("runtime receipt filename differs from its identity")
    predictions = merged_predictions_path(path.parent, identity)
    metrics = metrics_path(path.parent, identity)
    metrics_payload = _load_metrics_payload(metrics, checkpoint, split)
    if (
        not predictions.is_file()
        or sha256_file(predictions) != metrics_payload["predictions_sha256"]
    ):
        raise FormalEvalRuntimeError("receipt prediction artifact is missing or corrupt")
    expected = _receipt_payload(
        checkpoint,
        split,
        predictions=predictions,
        metrics=metrics,
        eval_loss=None if split == "test" else float(metrics_payload["eval_loss"]),
    )
    if payload != expected:
        raise FormalEvalRuntimeError("runtime receipt content differs from result artifacts")
    if split == "test" and _validate_completion(path.parent, checkpoint) is None:
        raise FormalEvalRuntimeError("test receipt has no locked completion")
    return FormalRuntimeReceipt(path=path, payload=payload)


def _fraction_from_json(value: Any, *, name: str) -> Fraction:
    if not isinstance(value, Mapping):
        raise FormalEvalRuntimeError(f"{name} must be an exact fraction object")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    display = value.get("value")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
        or isinstance(display, bool)
        or not isinstance(display, (int, float))
    ):
        raise FormalEvalRuntimeError(f"{name} has invalid fraction fields")
    result = Fraction(numerator, denominator)
    if not math.isclose(float(result), float(display), rel_tol=0.0, abs_tol=1e-15):
        raise FormalEvalRuntimeError(f"{name} float display differs from exact fraction")
    return result


def load_checkpoint_score(
    checkpoint_dir: str | Path,
    metrics_file: str | Path,
    *,
    processed_path: str | Path,
    config: ExperimentConfig | None = None,
) -> CheckpointScore:
    """Load the exact @10 validation score used by checkpoint selection."""

    checkpoint = inspect_durable_checkpoint(checkpoint_dir, processed_path, config=config)
    metrics_source = Path(metrics_file)
    identity = checkpoint.generation_identity("val")
    if metrics_source.name != metrics_path(metrics_source.parent, identity).name:
        raise FormalEvalRuntimeError("validation metrics filename differs from its identity")
    payload = _load_metrics_payload(metrics_source, checkpoint, "val")
    predictions = merged_predictions_path(metrics_source.parent, identity)
    if not predictions.is_file() or sha256_file(predictions) != payload["predictions_sha256"]:
        raise FormalEvalRuntimeError("validation predictions are missing or corrupt")
    evaluation = payload["evaluation"]
    metrics_by_k = evaluation.get("metrics")
    if not isinstance(metrics_by_k, Mapping) or "@10" not in metrics_by_k:
        raise FormalEvalRuntimeError("validation report has no @10 metrics")
    summary = metrics_by_k["@10"]
    if not isinstance(summary, Mapping):
        raise FormalEvalRuntimeError("validation @10 summary is invalid")
    count = summary.get("count")
    if count != EXPECTED_FORMAL_SPLIT_COUNTS["val"]:
        raise FormalEvalRuntimeError(f"validation @10 count is {count!r}, expected 994")

    connectivity = summary.get("connectivity")
    exact = summary.get("exact")
    edit = summary.get("edit_bond")
    iou = summary.get("edge_iou")
    f1 = summary.get("edge_f1")
    if not all(isinstance(value, Mapping) for value in (connectivity, exact, edit, iou, f1)):
        raise FormalEvalRuntimeError("validation @10 metric schema is invalid")
    connectivity_fraction = _fraction_from_json(connectivity, name="Connectivity@10")
    exact_fraction = _fraction_from_json(exact, name="Exact@10")
    connectivity_successes = connectivity.get("successes")
    exact_successes = exact.get("successes")
    edit_total = edit.get("total")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (connectivity_successes, exact_successes, edit_total)
    ):
        raise FormalEvalRuntimeError("validation @10 integer totals are invalid")
    if connectivity_fraction != Fraction(connectivity_successes, count):
        raise FormalEvalRuntimeError("Connectivity@10 successes/fraction disagree")
    if exact_fraction != Fraction(exact_successes, count):
        raise FormalEvalRuntimeError("Exact@10 successes/fraction disagree")
    if _fraction_from_json(edit, name="edit bond@10") != Fraction(edit_total, count):
        raise FormalEvalRuntimeError("edit bond@10 total/fraction disagree")
    iou_sum = _fraction_from_json(iou.get("sum"), name="IoU@10 sum")
    f1_sum = _fraction_from_json(f1.get("sum"), name="F1@10 sum")
    if _fraction_from_json(iou, name="IoU@10") != iou_sum / count:
        raise FormalEvalRuntimeError("IoU@10 mean/sum disagree")
    if _fraction_from_json(f1, name="F1@10") != f1_sum / count:
        raise FormalEvalRuntimeError("F1@10 mean/sum disagree")
    aggregate = AggregateMetrics(
        count=count,
        connectivity_successes=connectivity_successes,
        exact_successes=exact_successes,
        edit_bond_total=edit_total,
        edge_iou_sum=iou_sum,
        edge_f1_sum=f1_sum,
    )
    return CheckpointScore(
        checkpoint_id=checkpoint.checkpoint_fingerprint,
        epoch=checkpoint.epoch,
        metrics_at_10=aggregate,
        eval_loss=float(payload["eval_loss"]),
    )


ArgumentsFactory = Callable[[ExperimentConfig], Any]
ModelLoader = Callable[[ExperimentConfig, DurableCheckpoint, Any], Any]
TokenizerLoader = Callable[[ExperimentConfig], Any]
DataPreparer = Callable[[ExperimentConfig, str | Path, Any, Split], PreparedFormalData]
TrainerBuilder = Callable[[Any, Any, PreparedFormalData], Any]


def run_formal_checkpoint_evaluation(
    config: ExperimentConfig,
    *,
    processed_path: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    split: Split,
    selection_proof_path: str | Path | None = None,
    local_files_only: bool = True,
    environment_installer: Callable[[], Any] = install_frozen_environment,
    context_loader: Callable[[], TorchrunContext] = preflight_torchrun_environment,
    arguments_factory: Callable[..., Any] = build_formal_eval_arguments,
    context_validator: Callable[[Any, TorchrunContext, Any], None] = (
        assert_initialized_two_rank_job
    ),
    reproducibility_configurer: Callable[..., None] = configure_reproducibility,
    model_loader: Callable[..., Any] = load_zero3_portable_model,
    tokenizer_loader: Callable[..., Any] = load_pinned_tokenizer,
    data_preparer: Callable[..., PreparedFormalData] = prepare_formal_data,
    trainer_builder: Callable[..., Any] = build_formal_trainer,
    inference_initializer: Callable[[Any], Any] = initialize_inference_engine,
    module_resolver: Callable[[Any], Any] = generation_module_from_trainer,
    generation_runner: Callable[..., FormalEvaluationResult | None] = run_formal_generation,
    torch_module: Any | None = None,
) -> FormalRuntimeReceipt | None:
    """Evaluate one durable epoch; all real calls run under two-rank torchrun."""

    if split not in ("val", "test"):
        raise FormalEvalRuntimeError(f"formal split must be val or test, got {split!r}")
    if split == "val" and selection_proof_path is not None:
        raise FormalEvalRuntimeError("validation must not receive a selection proof")
    if split == "test" and selection_proof_path is None:
        raise FormalEvalRuntimeError("test requires a locked selection proof")
    environment_installer()
    checkpoint = inspect_durable_checkpoint(checkpoint_dir, processed_path, config=config)
    output_root = Path(output_dir)
    if split == "test":
        proof = load_selection_proof(
            selection_proof_path,
            checkpoint.generation_identity("test"),
        )
        if (
            proof.selected_global_step != checkpoint.global_step
            or not math.isclose(
                float(proof.selected_epoch),
                checkpoint.epoch,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise FormalEvalRuntimeError(
                "selection proof training position differs from selected checkpoint"
            )
        recovered = recover_completed_test(output_root, checkpoint)
        if recovered is not None:
            return recovered

    context = context_loader()
    # Do not move model_loader above this call: TrainingArguments owns Zero.Init.
    arguments = arguments_factory(config, output_dir=output_root)
    if torch_module is None:
        import torch as torch_module
    reproducibility_configurer(torch_module, seed=config.seed)
    context_validator(arguments, context, torch_module)
    model = model_loader(
        config,
        checkpoint,
        arguments,
        local_files_only=local_files_only,
    )
    tokenizer = tokenizer_loader(config, local_files_only=local_files_only)
    prepared = data_preparer(config, processed_path, tokenizer, split)
    trainer = trainer_builder(model, arguments, prepared)

    if split == "val":
        if prepared.eval_dataset is None:
            raise FormalEvalRuntimeError("validation has no causal-loss dataset")
        eval_loss = trainer_eval_loss_hook(trainer, eval_dataset=prepared.eval_dataset)()
    else:
        inference_initializer(trainer)
        eval_loss = None
    generation_model = module_resolver(trainer)
    identity = checkpoint.generation_identity(split)
    result = generation_runner(
        generation_model,
        tokenizer,
        prepared.records,
        identity,
        output_dir=output_root,
        eval_loss=eval_loss,
        selection_proof_path=selection_proof_path,
    )
    if result is None:
        return None
    if result.eval_loss != eval_loss:
        raise FormalEvalRuntimeError("formal generation changed eval_loss")
    metrics_payload = _load_metrics_payload(result.metrics_path, checkpoint, split)
    if sha256_file(result.predictions_path) != metrics_payload["predictions_sha256"]:
        raise FormalEvalRuntimeError("generated predictions differ from the metrics receipt")
    return _install_or_validate_receipt(
        output_root,
        checkpoint,
        split,
        predictions=result.predictions_path,
        metrics=result.metrics_path,
        eval_loss=eval_loss,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("val", "test"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-proof", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_config(arguments.config)
    receipt = run_formal_checkpoint_evaluation(
        config,
        processed_path=arguments.processed_path,
        checkpoint_dir=arguments.checkpoint_dir,
        output_dir=arguments.output_dir,
        split=arguments.split,
        selection_proof_path=arguments.selection_proof,
        local_files_only=True,
    )
    if receipt is not None and int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(receipt.payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by torchrun
    raise SystemExit(main())


__all__ = [
    "DurableCheckpoint",
    "DeepSpeedGenerationProxy",
    "FORMAL_RUNTIME_RECEIPT_SCHEMA_VERSION",
    "FormalEvalRuntimeError",
    "FormalRuntimeReceipt",
    "PreparedFormalData",
    "assert_formal_zero3_precision",
    "build_formal_eval_arguments",
    "generation_module_from_trainer",
    "initialize_inference_engine",
    "inspect_durable_checkpoint",
    "load_checkpoint_score",
    "load_zero3_portable_model",
    "prepare_formal_data",
    "recover_completed_test",
    "run_formal_checkpoint_evaluation",
    "runtime_receipt_path",
    "validate_runtime_receipt",
]
