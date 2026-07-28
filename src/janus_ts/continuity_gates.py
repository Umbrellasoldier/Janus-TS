"""Real two-rank continuity gates for training and portable checkpoints.

This module is intentionally separate from the formal workflow.  It provides
two expensive, fail-closed probes which are launched only through ``torchrun``:

``resume``
    Train the pinned residual-base/PiSSA model for two updates, atomically save
    a complete Janus checkpoint after update one, then compare update two with
    a fresh-engine restore of that checkpoint.

``portable-parity``
    Compare last-token logits from the trained rank-32 adapter over the PiSSA
    residual base with the rank-64 portable adapter over the untouched base.
    The same portable engine also runs one full beam-10/512-token formal
    generation for the longest canonical validation prompt as a memory gate.

The module never uses filesystem timestamps.  Temporary model/checkpoint
state lives below ``/home/caoxiangyu/.cache/janus-ts/gates``; only the requested
small atomic receipt is written outside that cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import random
import shutil
import socket
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from transformers import TrainerCallback

from .artifacts import (
    canonical_json_bytes,
    read_complete_manifest,
    sha256_file,
    sha256_json,
    write_json,
)
from .checkpointing import (
    RESUME_ADAPTER_SUBDIR,
    CheckpointIdentity,
    CheckpointManager,
    ResumeCheckpoint,
    TorchDistributedCoordinator,
)
from .config import ExperimentConfig, load_config
from .constants import (
    MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    QWEN_IM_END_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
)
from .distributed import (
    JanusManagedTrainer,
    PostEngineModelContractCallback,
    SyntheticSequenceDataset,
    assert_initialized_two_rank_job,
    assert_zero3_no_offload,
    checkpoint_manager_restore_hook,
    configure_reproducibility,
    load_zero3_prepared_model,
    preflight_torchrun_environment,
)
from .evaluation import evaluate_reaction
from .formal_eval_runtime import (
    DurableCheckpoint,
    PreparedFormalData,
    assert_formal_zero3_precision,
    build_formal_eval_arguments,
    build_formal_trainer,
    generation_module_from_trainer,
    initialize_inference_engine,
    inspect_durable_checkpoint,
    load_zero3_portable_model,
)
from .generation import (
    EXPECTED_FORMAL_SPLIT_COUNTS,
    FORMAL_NUM_BEAMS,
    encode_formal_prompt,
    formal_generation_kwargs,
    generate_reaction,
)
from .modeling import (
    EXPECTED_PORTABLE_PARAMETERS,
    EXPECTED_TRAINABLE_PARAMETERS,
    assert_pissa_adapter_contract,
    validate_loading_info,
    validate_pissa_initialization_config,
    validate_qwen_text_config,
)
from .preprocessing import (
    load_pinned_tokenizer,
    load_processed_dataset,
    reaction_record_from_row,
)
from .runtime import install_frozen_environment
from .tokenization import CausalLMCollator
from .training import (
    assert_accelerate_zero3_precision_state,
    assert_zero3_bf16_precision,
    assert_zero3_engine_precision_contract,
    build_training_argument_kwargs,
)

CONTINUITY_GATE_SCHEMA_VERSION = "janus-ts-continuity-gate-v1"
RESUME_GATE_NAME = "zero3-real-resume-next-update-bitwise"
PORTABLE_GATE_NAME = "pissa-portable-logit-and-generation-parity"
OOM_EXIT_CODE = 42
MAX_SWAP_GROWTH_KIB = 256 * 1024
_CACHE_ROOT = Path("/home/caoxiangyu/.cache/janus-ts/gates")
_SHA256_LENGTH = 64


class ContinuityGateError(RuntimeError):
    """A real-model continuity invariant was violated."""


@dataclass(slots=True)
class _TrainingBranch:
    """Evidence plus live objects which must be released between branches."""

    evidence: dict[str, Any]
    checkpoint: ResumeCheckpoint | None
    trainer: Any
    model: Any


@dataclass(slots=True)
class _InferenceBranch:
    """One inference engine and its small CPU logit probe."""

    evidence: dict[str, Any]
    logits: Any
    trainer: Any
    model: Any
    engine: Any


def _is_lora_parameter(name: str) -> bool:
    return (
        name.startswith("lora_A.")
        or name.startswith("lora_B.")
        or ".lora_A." in name
        or ".lora_B." in name
    )


def _logical_numel(parameter: Any) -> int:
    return int(getattr(parameter, "ds_numel", parameter.numel()))


def _update_stable_hash(digest: Any, value: Any) -> None:
    """Hash nested state without pickle, timestamps, or object addresses."""

    import torch

    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        header = {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(canonical_json_bytes(header))
        array = tensor.numpy()
        digest.update(memoryview(array).cast("B"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        header = {
            "kind": "ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        digest.update(canonical_json_bytes(header))
        digest.update(memoryview(array).cast("B"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping[")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_stable_hash(digest, key)
            _update_stable_hash(digest, value[key])
        digest.update(b"]")
        return
    if isinstance(value, tuple):
        digest.update(b"tuple[")
        for item in value:
            _update_stable_hash(digest, item)
        digest.update(b"]")
        return
    if isinstance(value, list):
        digest.update(b"list[")
        for item in value:
            _update_stable_hash(digest, item)
        digest.update(b"]")
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        digest.update(canonical_json_bytes({"kind": type(value).__name__, "value": value}))
        return
    raise ContinuityGateError(f"cannot stably hash state value {type(value).__name__}")


def stable_state_sha256(value: Any) -> str:
    """Return a deterministic hash for scheduler/RNG evidence."""

    digest = hashlib.sha256()
    _update_stable_hash(digest, value)
    return digest.hexdigest()


def rank_local_rng_sha256(torch_module: Any) -> str:
    """Hash exactly the RNG domains persisted by :class:`CheckpointManager`."""

    cuda_initialized = torch_module.cuda.is_initialized()
    device = torch_module.cuda.current_device() if cuda_initialized else None
    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
        "cuda_device": device,
        "torch_cuda": (torch_module.cuda.get_rng_state(device) if cuda_initialized else None),
    }
    return stable_state_sha256(payload)


def _gather_context(parameter: Any) -> Any:
    if not hasattr(parameter, "ds_id"):
        return nullcontext()
    import deepspeed

    return deepspeed.zero.GatheredParameters([parameter], modifier_rank=None)


def stream_lora_tensor_hashes(
    engine: Any,
    coordinator: Any,
    *,
    expected_trainable_parameters: int = EXPECTED_TRAINABLE_PARAMETERS,
    gather_context: Callable[[Any], Any] = _gather_context,
) -> dict[str, Any]:
    """Gather and hash one LoRA tensor at a time, never materializing a state dict."""

    import torch

    module = getattr(engine, "module", None)
    if module is None:
        raise ContinuityGateError("LoRA hash requires a DeepSpeed engine.module")
    trainable = sorted(
        (
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ),
        key=lambda item: item[0],
    )
    logical_parameters = sum(_logical_numel(parameter) for _, parameter in trainable)
    problems = [name for name, _ in trainable if not _is_lora_parameter(name)]
    wrong_dtype = [name for name, parameter in trainable if parameter.dtype != torch.float32]
    if (
        not trainable
        or logical_parameters != expected_trainable_parameters
        or problems
        or wrong_dtype
    ):
        raise ContinuityGateError(
            "streaming LoRA hash contract failed: "
            f"logical_parameters={logical_parameters}, "
            f"expected={expected_trainable_parameters}, "
            f"non_lora={problems[:4]!r}, wrong_dtype={wrong_dtype[:4]!r}"
        )

    aggregate = hashlib.sha256() if coordinator.rank == 0 else None
    tensors: list[dict[str, Any]] | None = [] if coordinator.rank == 0 else None
    for name, parameter in trainable:
        with gather_context(parameter):
            if coordinator.rank != 0:
                continue
            assert aggregate is not None and tensors is not None
            tensor = parameter.detach().cpu().contiguous()
            header = {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "numel": int(tensor.numel()),
            }
            raw = memoryview(tensor.numpy()).cast("B")
            tensor_digest = hashlib.sha256()
            tensor_digest.update(canonical_json_bytes(header))
            tensor_digest.update(raw)
            entry = {**header, "sha256": tensor_digest.hexdigest()}
            tensors.append(entry)
            aggregate.update(canonical_json_bytes(header))
            aggregate.update(raw)

    payload: dict[str, Any] | None = None
    if coordinator.rank == 0:
        assert aggregate is not None and tensors is not None
        payload = {
            "algorithm": "ordered-name-shape-dtype-raw-bytes-sha256-v1",
            "aggregate_sha256": aggregate.hexdigest(),
            "tensor_count": len(tensors),
            "logical_parameters": logical_parameters,
            "tensors": tensors,
        }
    payload = coordinator.broadcast(payload, source=0)
    if not isinstance(payload, dict):
        raise ContinuityGateError("rank zero did not broadcast LoRA hash evidence")
    return payload


def make_resume_gate_identity(
    *,
    config_sha256: str,
    source_fingerprint: str,
    bundle_manifest_sha256: str,
) -> CheckpointIdentity:
    """Create a content-only identity for the disposable resume checkpoint."""

    for name, value in {
        "config_sha256": config_sha256,
        "source_fingerprint": source_fingerprint,
        "bundle_manifest_sha256": bundle_manifest_sha256,
    }.items():
        if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
            raise ContinuityGateError(f"{name} is not a SHA256 digest")
    model_fingerprint = sha256_json(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "pissa_manifest_sha256": bundle_manifest_sha256,
        }
    )
    return CheckpointIdentity.derive(
        config_fingerprint=config_sha256,
        data_fingerprint=source_fingerprint,
        model_fingerprint=model_fingerprint,
    )


def execute_restart_sequence(
    continuous_runner: Callable[[], Any],
    releaser: Callable[[Any], None],
    resumed_runner: Callable[[Any], Any],
    comparator: Callable[[Any, Any], Any],
) -> Any:
    """Enforce continuous -> release -> fresh-resume -> compare ordering."""

    continuous = continuous_runner()
    releaser(continuous)
    resumed = resumed_runner(continuous)
    return comparator(continuous, resumed)


def compare_resume_evidence(
    continuous: Mapping[str, Any], resumed: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail unless the second update is bitwise and statewise identical."""

    exact_fields = (
        "rank",
        "global_step",
        "engine_global_step",
        "scheduler_sha256",
        "rng_sha256",
    )
    mismatches = [name for name in exact_fields if continuous.get(name) != resumed.get(name)]
    left_lora = continuous.get("lora_stream")
    right_lora = resumed.get("lora_stream")
    if left_lora != right_lora:
        mismatches.append("lora_stream")
    if continuous.get("global_step") != 2 or resumed.get("global_step") != 2:
        mismatches.append("expected_global_step_2")
    if mismatches:
        raise ContinuityGateError("continuous/resumed update differs: " + ", ".join(mismatches))
    if not isinstance(left_lora, Mapping):
        raise ContinuityGateError("resume comparison lacks LoRA stream evidence")
    return {
        "status": "pass",
        "rank": int(continuous["rank"]),
        "global_step": 2,
        "scheduler_sha256": str(continuous["scheduler_sha256"]),
        "rng_sha256": str(continuous["rng_sha256"]),
        "lora_aggregate_sha256": str(left_lora["aggregate_sha256"]),
        "lora_tensor_count": int(left_lora["tensor_count"]),
        "lora_logical_parameters": int(left_lora["logical_parameters"]),
    }


def assert_logits_parity(
    residual_logits: Any,
    portable_logits: Any,
    *,
    atol: float = 0.125,
    rtol: float = 0.02,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Apply the frozen finite/argmax/numeric portable-parity contract."""

    if torch_module is None:
        import torch as torch_module

    left = torch_module.as_tensor(residual_logits).detach().cpu().float().contiguous()
    right = torch_module.as_tensor(portable_logits).detach().cpu().float().contiguous()
    if left.ndim != 1 or right.shape != left.shape or left.numel() == 0:
        raise ContinuityGateError(
            f"logit shapes differ or are invalid: {tuple(left.shape)} / {tuple(right.shape)}"
        )
    if not bool(torch_module.isfinite(left).all()) or not bool(torch_module.isfinite(right).all()):
        raise ContinuityGateError("portable parity logits contain NaN or infinity")
    left_argmax = int(left.argmax().item())
    right_argmax = int(right.argmax().item())
    if left_argmax != right_argmax:
        raise ContinuityGateError(
            f"portable parity argmax differs: {left_argmax} != {right_argmax}"
        )
    try:
        torch_module.testing.assert_close(left, right, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise ContinuityGateError(f"portable logits exceed tolerance: {exc}") from exc
    maximum_error = float((left - right).abs().max().item())
    return {
        "status": "pass",
        "vocabulary_size": int(left.numel()),
        "argmax_token_id": left_argmax,
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": maximum_error,
    }


def select_longest_canonical_prompt(
    records: Sequence[Any],
    encoder: Callable[[Any], Any],
) -> tuple[int, Any, Any]:
    """Select the first longest legal prompt by stable dataset ordinal."""

    if not records:
        raise ContinuityGateError("cannot select a prompt from an empty validation split")
    selected: tuple[int, Any, Any] | None = None
    selected_length = -1
    for ordinal, record in enumerate(records):
        encoded = encoder(record)
        length = len(encoded.input_ids)
        if length <= 0 or length > MAX_SEQUENCE_LENGTH:
            raise ContinuityGateError(f"validation prompt {ordinal} has illegal length {length}")
        if length > selected_length:
            selected = (ordinal, record, encoded)
            selected_length = length
    assert selected is not None
    return selected


def validate_generation_smoke_evidence(
    payload: Mapping[str, Any],
    *,
    max_device_memory_mib: int = 45056,
    max_swap_growth_kib: int = MAX_SWAP_GROWTH_KIB,
) -> None:
    """Validate the exact nested receipt consumed before first formal eval."""

    required = {
        "status",
        "reaction_id",
        "ordinal",
        "prompt_sha256",
        "prompt_tokens",
        "num_beams",
        "num_return_sequences",
        "max_new_tokens",
        "output_shape",
        "parsed_beams",
        "valid_parses",
        "swap_growth_kib",
        "max_swap_growth_kib",
        "max_device_memory_mib",
        "stress",
        "ranks",
    }
    if set(payload) != required:
        raise ContinuityGateError(
            "generation smoke receipt keys differ: "
            f"missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}"
        )
    if (
        payload.get("status") != "pass"
        or payload.get("num_beams") != FORMAL_NUM_BEAMS
        or payload.get("num_return_sequences") != FORMAL_NUM_BEAMS
        or payload.get("max_new_tokens") != 512
        or payload.get("parsed_beams") != FORMAL_NUM_BEAMS
        or payload.get("max_device_memory_mib") != max_device_memory_mib
        or payload.get("max_swap_growth_kib") != max_swap_growth_kib
    ):
        raise ContinuityGateError("generation smoke frozen decoding/threshold fields differ")
    if (
        isinstance(payload["swap_growth_kib"], bool)
        or not isinstance(payload["swap_growth_kib"], int)
        or payload["swap_growth_kib"] > max_swap_growth_kib
    ):
        raise ContinuityGateError("generation smoke exceeded the global swap-growth limit")
    if (
        not isinstance(payload["reaction_id"], str)
        or not payload["reaction_id"]
        or isinstance(payload["ordinal"], bool)
        or not isinstance(payload["ordinal"], int)
        or payload["ordinal"] < 0
        or not isinstance(payload["prompt_sha256"], str)
        or len(payload["prompt_sha256"]) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in payload["prompt_sha256"])
        or isinstance(payload["prompt_tokens"], bool)
        or not isinstance(payload["prompt_tokens"], int)
        or not 0 < payload["prompt_tokens"] <= MAX_SEQUENCE_LENGTH
        or isinstance(payload["valid_parses"], bool)
        or not isinstance(payload["valid_parses"], int)
        or not 0 <= payload["valid_parses"] <= FORMAL_NUM_BEAMS
    ):
        raise ContinuityGateError("generation smoke prompt/parse identity is invalid")
    shape = payload.get("output_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape[0] != FORMAL_NUM_BEAMS
        or isinstance(shape[1], bool)
        or not isinstance(shape[1], int)
    ):
        raise ContinuityGateError(f"invalid beam output shape: {shape!r}")
    if not payload["prompt_tokens"] < shape[1] <= payload["prompt_tokens"] + 512:
        raise ContinuityGateError("generation output length exceeds prompt plus max_new_tokens")
    stress = payload.get("stress")
    expected_stress = {
        "status": "pass",
        "purpose": "non-formal-worst-case-memory-only",
        "num_beams": FORMAL_NUM_BEAMS,
        "num_return_sequences": FORMAL_NUM_BEAMS,
        "min_new_tokens": 512,
        "max_new_tokens": 512,
        "actual_new_tokens": 512,
        "output_shape": [FORMAL_NUM_BEAMS, payload["prompt_tokens"] + 512],
    }
    if stress != expected_stress:
        raise ContinuityGateError(f"generation full-length stress evidence differs: {stress!r}")
    ranks = payload.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise ContinuityGateError("generation smoke requires exactly two rank reports")
    if {rank.get("rank") for rank in ranks if isinstance(rank, Mapping)} != {0, 1}:
        raise ContinuityGateError("generation smoke rank identities are incomplete")
    for rank in ranks:
        if not isinstance(rank, Mapping):
            raise ContinuityGateError("generation smoke rank report is not an object")
        rank_index = rank.get("rank")
        integer_fields = (
            "peak_allocated_mib",
            "peak_reserved_mib",
            "device_memory_used_mib",
            "effective_peak_mib",
        )
        if (
            isinstance(rank_index, bool)
            or not isinstance(rank_index, int)
            or rank.get("local_rank") != rank_index
            or any(
                isinstance(rank.get(name), bool) or not isinstance(rank.get(name), int)
                for name in integer_fields
            )
            or any(int(rank[name]) < 0 for name in integer_fields)
        ):
            raise ContinuityGateError(f"generation smoke rank resources are invalid: {rank!r}")
        recomputed_peak = max(
            int(rank["peak_reserved_mib"]),
            int(rank["device_memory_used_mib"]),
        )
        if (
            rank.get("output_shape") != shape
            or rank.get("parsed_beams") != FORMAL_NUM_BEAMS
            or isinstance(rank.get("valid_parses"), bool)
            or not isinstance(rank.get("valid_parses"), int)
            or not 0 <= rank["valid_parses"] <= FORMAL_NUM_BEAMS
            or rank.get("stress_output_shape") != stress["output_shape"]
            or rank.get("stress_actual_new_tokens") != 512
            or rank["effective_peak_mib"] != recomputed_peak
            or rank["effective_peak_mib"] > max_device_memory_mib
        ):
            raise ContinuityGateError(f"generation smoke rank failed: {rank!r}")


def _source_fingerprint(config: ExperimentConfig) -> str:
    return sha256_json(
        {
            "expected_source_sha256": config.data.expected_source_sha256,
            "expected_raw_counts": config.data.expected_raw_counts,
            "expected_retained_counts": config.data.expected_retained_counts,
        }
    )


def _invocation_workspace(
    config: ExperimentConfig,
    *,
    gate: str,
    identity: str,
    output: str | Path,
) -> Path:
    invocation = sha256_json(
        {
            "gate": gate,
            "identity": identity,
            "output": str(Path(output).resolve()),
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": os.environ.get("MASTER_PORT"),
            "torchelastic_run_id": os.environ.get("TORCHELASTIC_RUN_ID"),
        }
    )
    root = Path(config.runtime.local_cache_root).resolve() / "gates"
    if root != _CACHE_ROOT:
        raise ContinuityGateError(f"unexpected gate cache root: {root}")
    workspace = root / gate / identity / invocation
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _build_two_step_arguments(
    config: ExperimentConfig,
    *,
    output_dir: Path,
    micro_batch_size: int,
) -> Any:
    if micro_batch_size == config.train.micro_batch_size_per_gpu:
        accumulation = config.train.gradient_accumulation_steps
    elif micro_batch_size == config.train.candidate_micro_batch_size_per_gpu:
        accumulation = config.train.candidate_gradient_accumulation_steps
    else:
        raise ContinuityGateError("resume gate micro batch is not a confirmed geometry")
    kwargs = build_training_argument_kwargs(
        config,
        output_dir=output_dir,
        micro_batch_size_per_gpu=micro_batch_size,
        gradient_accumulation_steps=accumulation,
    )
    kwargs.update(
        {
            "do_eval": False,
            "eval_strategy": "no",
            "save_strategy": "no",
            "logging_strategy": "no",
            "report_to": [],
            "num_train_epochs": 1.0,
            "max_steps": 2,
        }
    )
    from transformers import TrainingArguments

    return TrainingArguments(**kwargs)


class _StepOneCheckpointCallback(TrainerCallback):
    """Save exactly one real rolling checkpoint after update one."""

    def __init__(self, manager: CheckpointManager, destination: Path) -> None:
        self.manager = manager
        self.destination = destination
        self.trainer: Any | None = None
        self.checkpoint: ResumeCheckpoint | None = None

    def bind(self, trainer: Any) -> None:
        if self.trainer is not None:
            raise ContinuityGateError("step-one callback was bound twice")
        self.trainer = trainer

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if int(state.global_step) == 1 and self.checkpoint is None:
            if self.trainer is None:
                raise ContinuityGateError("step-one callback has no Trainer")
            self.checkpoint = self.manager.save(
                self.trainer.model_wrapped,
                self.destination,
                kind="rolling",
                global_step=1,
                epoch=float(state.epoch or 0.0),
                trainer_state=state,
                lr_scheduler=kwargs.get("lr_scheduler"),
            )
        return control


def _build_synthetic_trainer(
    config: ExperimentConfig,
    *,
    bundle: Path,
    output_dir: Path,
    micro_batch_size: int,
    restore_hook: Any | None = None,
    extra_callbacks: Sequence[Any] = (),
) -> tuple[Any, Any]:
    """Build one real two-update Trainer; arguments deliberately precede model."""

    arguments = _build_two_step_arguments(
        config,
        output_dir=output_dir,
        micro_batch_size=micro_batch_size,
    )
    import torch

    context = preflight_torchrun_environment()
    configure_reproducibility(torch, seed=config.seed)
    assert_initialized_two_rank_job(arguments, context, torch)
    model = load_zero3_prepared_model(bundle, arguments)
    accumulation = int(arguments.gradient_accumulation_steps)
    examples = int(arguments.per_device_train_batch_size) * context.world_size * accumulation * 2
    dataset = SyntheticSequenceDataset(examples)
    precision_callback = PostEngineModelContractCallback()
    trainer = JanusManagedTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=CausalLMCollator(),
        callbacks=[precision_callback, *extra_callbacks],
        janus_managed_checkpoints=True,
        janus_restore_hook=restore_hook,
    )
    precision_callback.bind_engine_getter(lambda: trainer.model_wrapped)
    assert_accelerate_zero3_precision_state(arguments, trainer.accelerator)
    return trainer, model


def _training_branch_evidence(trainer: Any, *, rank: int) -> dict[str, Any]:
    import torch

    engine = trainer.model_wrapped
    precision = assert_zero3_engine_precision_contract(
        engine,
        require_optimizer_states=True,
    )
    coordinator = TorchDistributedCoordinator()
    lora = stream_lora_tensor_hashes(engine, coordinator)
    scheduler_state = trainer.lr_scheduler.state_dict()
    return {
        "rank": rank,
        "global_step": int(trainer.state.global_step),
        "engine_global_step": int(engine.global_steps),
        "scheduler_sha256": stable_state_sha256(scheduler_state),
        "rng_sha256": rank_local_rng_sha256(torch),
        "lora_stream": lora,
        "zero3_dtype_no_offload": precision,
    }


def _release_training_branch(branch: _TrainingBranch) -> None:
    """Destroy the first DeepSpeed engine before constructing the fresh one."""

    import torch

    torch.distributed.barrier()
    trainer = branch.trainer
    model = branch.model
    engine = getattr(trainer, "model_wrapped", None)
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        accelerator.free_memory(engine, model)
    for name in ("model", "model_wrapped", "deepspeed", "optimizer", "lr_scheduler"):
        if hasattr(trainer, name):
            setattr(trainer, name, None)
    branch.trainer = None
    branch.model = None
    del engine, model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    _reset_accelerator_plugin_state()


def _reset_accelerator_plugin_state() -> None:
    """Forget a destroyed engine's plugin while retaining the live NCCL group.

    Accelerate keeps ``AcceleratorState`` as a process singleton.  A second
    Trainer in the same torchrun process would otherwise silently retain the
    first TrainingArguments' DeepSpeed plugin.  ``PartialState`` owns the
    already-initialized process group and is deliberately not reset.
    """

    from accelerate.state import AcceleratorState, PartialState

    if not PartialState._shared_state:
        raise ContinuityGateError("cannot retain an uninitialized distributed PartialState")
    AcceleratorState._reset_state(reset_partial_state=False)
    if not PartialState._shared_state:
        raise ContinuityGateError("AcceleratorState reset unexpectedly destroyed PartialState")


def run_resume_gate(
    config: ExperimentConfig,
    *,
    bundle: str | Path,
    output: str | Path,
    micro_batch_size: int,
) -> dict[str, Any]:
    """Run the real step-1 save / step-2 restart equivalence gate."""

    install_frozen_environment()
    context = preflight_torchrun_environment()
    bundle_path = Path(bundle).resolve(strict=True)
    read_complete_manifest(bundle_path)
    bundle_sha = sha256_file(bundle_path / "manifest.json")
    identity = make_resume_gate_identity(
        config_sha256=config.sha256,
        source_fingerprint=_source_fingerprint(config),
        bundle_manifest_sha256=bundle_sha,
    )
    workspace = _invocation_workspace(
        config,
        gate="resume",
        identity=identity.run_fingerprint,
        output=output,
    )
    checkpoint_root = workspace / "checkpoints"
    checkpoint_path = CheckpointManager(identity).destination(checkpoint_root, 1)

    def continuous_runner() -> _TrainingBranch:
        manager = CheckpointManager(identity)
        callback = _StepOneCheckpointCallback(manager, checkpoint_path)
        trainer, model = _build_synthetic_trainer(
            config,
            bundle=bundle_path,
            output_dir=workspace / "continuous-trainer",
            micro_batch_size=micro_batch_size,
            extra_callbacks=(callback,),
        )
        callback.bind(trainer)
        trainer.train()
        if callback.checkpoint is None:
            raise ContinuityGateError("continuous branch did not save step-one checkpoint")
        evidence = _training_branch_evidence(trainer, rank=context.rank)
        return _TrainingBranch(evidence, callback.checkpoint, trainer, model)

    def resumed_runner(continuous: _TrainingBranch) -> _TrainingBranch:
        checkpoint = continuous.checkpoint
        if checkpoint is None:
            raise ContinuityGateError("released branch lost its resume checkpoint")
        manager = CheckpointManager(identity)
        restore_hook = checkpoint_manager_restore_hook(manager, checkpoint)
        trainer, model = _build_synthetic_trainer(
            config,
            bundle=bundle_path,
            output_dir=workspace / "resumed-trainer",
            micro_batch_size=micro_batch_size,
            restore_hook=restore_hook,
        )
        trainer.train(resume_from_checkpoint=str(checkpoint.path))
        restored = trainer.janus_restore_result
        if (
            restored is None
            or restored.checkpoint.global_step != 1
            or int(restored.trainer_state.get("global_step", -1)) != 1
        ):
            raise ContinuityGateError("fresh branch did not restore the complete step-one state")
        evidence = _training_branch_evidence(trainer, rank=context.rank)
        evidence["restore"] = {
            "checkpoint_global_step": restored.checkpoint.global_step,
            "trainer_state_global_step": int(restored.trainer_state["global_step"]),
            "optimizer_and_scheduler_loaded_by_deepspeed": True,
            "rng_replayed_at_resume_boundary": True,
        }
        return _TrainingBranch(evidence, checkpoint, trainer, model)

    def comparator(
        continuous: _TrainingBranch, resumed: _TrainingBranch
    ) -> tuple[_TrainingBranch, dict[str, Any], dict[str, Any]]:
        comparison = compare_resume_evidence(continuous.evidence, resumed.evidence)
        return resumed, comparison, dict(continuous.evidence)

    resumed_branch, local_comparison, continuous_evidence = execute_restart_sequence(
        continuous_runner,
        _release_training_branch,
        resumed_runner,
        comparator,
    )
    checkpoint = resumed_branch.checkpoint
    if checkpoint is None:
        raise ContinuityGateError("resume branch has no bound checkpoint")
    local_branch = {
        "rank": context.rank,
        "continuous": continuous_evidence,
        "resumed": resumed_branch.evidence,
        "comparison": local_comparison,
    }
    # Keep only the resumed live branch at this point; release it before sealing.
    resumed_evidence = dict(resumed_branch.evidence)
    _release_training_branch(resumed_branch)

    import torch

    gathered: list[Any] = [None] * context.world_size
    torch.distributed.all_gather_object(gathered, local_branch)
    if any(not isinstance(item, Mapping) for item in gathered):
        raise ContinuityGateError("resume rank evidence gather was incomplete")
    checkpoint_manifest_sha = sha256_file(checkpoint.path / "manifest.json")
    checkpoint_manifest = read_complete_manifest(checkpoint.path)
    report = {
        "schema_version": CONTINUITY_GATE_SCHEMA_VERSION,
        "gate": RESUME_GATE_NAME,
        "status": "pass",
        "host": socket.gethostname(),
        "world_size": context.world_size,
        "identity": identity.as_dict(),
        "bundle_manifest_sha256": bundle_sha,
        "micro_batch_size_per_gpu": micro_batch_size,
        "sequence_length": MAX_SEQUENCE_LENGTH,
        "checkpoint": {
            "kind": checkpoint_manifest["kind"],
            "global_step": checkpoint_manifest["global_step"],
            "manifest_sha256": checkpoint_manifest_sha,
            "complete_marker_verified": True,
            "exclude_frozen_parameters": checkpoint_manifest["exclude_frozen_parameters"],
        },
        "comparison": {
            "status": "pass",
            "global_step": 2,
            "optimizer_state_restored_before_next_update": True,
            "scheduler_state_equal": True,
            "rank_local_rng_equal": True,
            "lora_bitwise_equal": True,
            "lora_stream": resumed_evidence["lora_stream"],
        },
        "ranks": list(gathered),
    }
    if context.rank == 0:
        write_json(output, report)
    torch.distributed.barrier()
    # This tree is explicitly disposable and content-bound to this invocation.
    if context.rank == 0 and workspace.exists():
        shutil.rmtree(workspace)
    torch.distributed.barrier()
    return report


def _load_zero3_resume_adapter_model(
    bundle: Path,
    checkpoint: DurableCheckpoint,
    arguments: Any,
) -> Any:
    """Load residual BF16 base plus the checkpoint's trained rank-32 adapter."""

    import torch
    from peft import PeftModel
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    assert_zero3_no_offload(arguments)
    read_complete_manifest(bundle)
    residual_dir = bundle / "residual_base"
    resume_dir = checkpoint.path / RESUME_ADAPTER_SUBDIR
    if not residual_dir.is_dir() or not resume_dir.is_dir():
        raise ContinuityGateError("residual base or rank-32 resume adapter is missing")
    text_config = Qwen3_5TextConfig.from_pretrained(
        residual_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    validate_qwen_text_config(text_config)
    validate_pissa_initialization_config(
        resume_dir / "adapter_config.json",
        text_config,
        prepared_reference=True,
    )
    base, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        residual_dir,
        config=text_config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=False,
    )
    validate_loading_info(loading_info)
    if getattr(base, "hf_device_map", None):
        raise ContinuityGateError("resume parity base unexpectedly has a device map")
    base.config.use_cache = False
    base.config.pad_token_id = QWEN_PAD_TOKEN_ID
    base.config.eos_token_id = QWEN_IM_END_TOKEN_ID
    model = PeftModel.from_pretrained(
        base,
        resume_dir,
        is_trainable=True,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=False,
    )
    for name, parameter in model.named_parameters():
        adapter = _is_lora_parameter(name)
        parameter.requires_grad_(adapter)
        if adapter and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    model.config.use_cache = False
    model.config.pad_token_id = QWEN_PAD_TOKEN_ID
    model.config.eos_token_id = QWEN_IM_END_TOKEN_ID
    assert_pissa_adapter_contract(model, adapter_dtype=torch.bfloat16)
    report = assert_zero3_bf16_precision(arguments, model)
    model._janus_ts_parity_precision_report = report
    model.eval()
    return model


class _LoraOutputProbe:
    """Observe one LoRA A and B output during an actual forward pass."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self.handles: list[Any] = []
        self.observed: dict[str, str] = {}

    def start(self) -> None:
        selected: dict[str, tuple[str, Any]] = {}
        for name, child in self.module.named_modules():
            label = None
            if ".lora_A." in name or name.startswith("lora_A."):
                label = "lora_A"
            elif ".lora_B." in name or name.startswith("lora_B."):
                label = "lora_B"
            if label is not None and label not in selected:
                selected[label] = (name, child)
            if len(selected) == 2:
                break
        if set(selected) != {"lora_A", "lora_B"}:
            raise ContinuityGateError("cannot locate LoRA A/B modules for autocast probe")

        def hook(label: str) -> Callable[..., None]:
            def record(_module: Any, _inputs: Any, output: Any) -> None:
                value = output[0] if isinstance(output, (tuple, list)) else output
                self.observed[label] = str(getattr(value, "dtype", None))

            return record

        for label, (_, child) in selected.items():
            self.handles.append(child.register_forward_hook(hook(label)))

    def abort(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finish(self) -> dict[str, str]:
        import torch

        self.abort()
        expected = str(torch.bfloat16)
        if self.observed != {"lora_A": expected, "lora_B": expected}:
            raise ContinuityGateError(f"LoRA forward outputs were not both BF16: {self.observed!r}")
        return dict(self.observed)


def _assert_inference_engine_contract(
    engine: Any,
    arguments: Any,
    *,
    expected_trainable_parameters: int,
) -> dict[str, Any]:
    assert_zero3_no_offload(arguments)
    report = assert_formal_zero3_precision(
        engine,
        expected_trainable_parameters=expected_trainable_parameters,
    )
    return {
        **report,
        "offload": False,
    }


def _forward_last_token_logits(
    engine: Any,
    prompt: Any,
    *,
    local_rank: int,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from deepspeed.runtime.torch_autocast import autocast_if_enabled

    module = engine.module
    device = torch.device("cuda", local_rank)
    input_ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    probe = _LoraOutputProbe(module)
    probe.start()
    try:
        with torch.inference_mode(), autocast_if_enabled(engine):
            output = module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
    except BaseException:
        probe.abort()
        raise
    observed = probe.finish()
    logits = output.logits[0, -1].detach().cpu().float().contiguous()
    if not bool(torch.isfinite(logits).all()):
        raise ContinuityGateError("last-token logits contain NaN or infinity")
    evidence = {
        "shape": list(logits.shape),
        "dtype_before_cpu_cast": str(output.logits.dtype),
        "sha256": stable_state_sha256(logits),
        "argmax_token_id": int(logits.argmax().item()),
        "lora_output_dtypes": observed,
    }
    del output, input_ids, attention_mask
    return logits, evidence


def _build_inference_branch(
    config: ExperimentConfig,
    *,
    checkpoint: DurableCheckpoint,
    bundle: Path,
    prompt: Any,
    output_dir: Path,
    kind: Literal["residual-rank32", "original-portable-rank64"],
) -> _InferenceBranch:
    arguments = build_formal_eval_arguments(config, output_dir=output_dir)
    import torch

    context = preflight_torchrun_environment()
    configure_reproducibility(torch, seed=config.seed)
    assert_initialized_two_rank_job(arguments, context, torch)
    if kind == "residual-rank32":
        model = _load_zero3_resume_adapter_model(bundle, checkpoint, arguments)
        expected = EXPECTED_TRAINABLE_PARAMETERS
        semantics = "prepared-pissa-residual-base-plus-trained-rank32-resume-adapter"
    else:
        model = load_zero3_portable_model(
            config,
            checkpoint,
            arguments,
            local_files_only=True,
        )
        expected = EXPECTED_PORTABLE_PARAMETERS
        semantics = "pinned-untouched-original-base-plus-rank64-portable-adapter"
    prepared = PreparedFormalData(
        records=(),
        eval_dataset=None,
        collator=CausalLMCollator(max_length=config.model.max_sequence_length),
    )
    trainer = build_formal_trainer(model, arguments, prepared)
    engine = initialize_inference_engine(trainer)
    contract = _assert_inference_engine_contract(
        engine,
        arguments,
        expected_trainable_parameters=expected,
    )
    logits, logit_evidence = _forward_last_token_logits(
        engine,
        prompt,
        local_rank=context.local_rank,
    )
    evidence = {
        "kind": kind,
        "base_semantics": semantics,
        "adapter_parameters": expected,
        "engine_contract": contract,
        "last_token_logits": logit_evidence,
    }
    return _InferenceBranch(evidence, logits, trainer, model, engine)


def _release_inference_branch(branch: _InferenceBranch) -> None:
    import torch

    torch.distributed.barrier()
    trainer = branch.trainer
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        accelerator.free_memory(branch.engine, branch.model)
    for name in ("model", "model_wrapped", "deepspeed", "optimizer", "lr_scheduler"):
        if hasattr(trainer, name):
            setattr(trainer, name, None)
    branch.trainer = None
    branch.model = None
    branch.engine = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier()
    _reset_accelerator_plugin_state()


class _RecordingGenerationModel:
    """Transparent proxy which retains the exact returned sequence shape."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.output_shape: list[int] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        output = self.model.generate(*args, **kwargs)
        sequences = getattr(output, "sequences", output)
        self.output_shape = [int(value) for value in sequences.shape]
        return output


def _swap_used_kib() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    return values["SwapTotal"] - values["SwapFree"]


def _device_memory_used_mib(local_rank: int) -> int | None:
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={local_rank}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return None


def _run_full_length_generation_stress(
    model: Any,
    prompt: Any,
    *,
    local_rank: int,
    torch_module: Any | None = None,
    device: Any | None = None,
) -> dict[str, Any]:
    """Exercise 512 new tokens without treating the output as a prediction."""

    if torch_module is None:
        import torch as torch_module

    kwargs = formal_generation_kwargs(synchronized=True)
    kwargs["min_new_tokens"] = 512
    if (
        kwargs.get("num_beams") != FORMAL_NUM_BEAMS
        or kwargs.get("num_return_sequences") != FORMAL_NUM_BEAMS
        or kwargs.get("max_new_tokens") != 512
        or kwargs.get("min_new_tokens") != 512
        or kwargs.get("synced_gpus") is not True
    ):
        raise ContinuityGateError("full-length stress decoding geometry drifted")
    if device is None:
        device = torch_module.device("cuda", local_rank)
    input_ids = torch_module.tensor([prompt.input_ids], dtype=torch_module.long, device=device)
    attention_mask = torch_module.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs,
    )
    sequences = getattr(output, "sequences", output)
    if not isinstance(sequences, torch_module.Tensor):
        sequences = torch_module.as_tensor(sequences)
    sequences = sequences.detach().cpu()
    shape = [int(value) for value in sequences.shape]
    actual_new_tokens = shape[1] - len(prompt.input_ids) if len(shape) == 2 else -1
    if shape != [FORMAL_NUM_BEAMS, len(prompt.input_ids) + 512]:
        raise ContinuityGateError(
            f"full-length stress did not produce exactly 512 new tokens: {shape!r}"
        )
    expected_prefix = torch_module.tensor(prompt.input_ids, dtype=sequences.dtype)
    prefixes = sequences[:, : len(prompt.input_ids)]
    if not torch_module.equal(prefixes, expected_prefix.expand_as(prefixes)):
        raise ContinuityGateError("full-length stress changed the decoder prompt prefix")
    del output, sequences, input_ids, attention_mask
    return {
        "status": "pass",
        "purpose": "non-formal-worst-case-memory-only",
        "num_beams": FORMAL_NUM_BEAMS,
        "num_return_sequences": FORMAL_NUM_BEAMS,
        "min_new_tokens": 512,
        "max_new_tokens": 512,
        "actual_new_tokens": actual_new_tokens,
        "output_shape": shape,
    }


def _generation_memory_smoke(
    config: ExperimentConfig,
    *,
    branch: _InferenceBranch,
    tokenizer: Any,
    record: Any,
    ordinal: int,
    prompt: Any,
) -> dict[str, Any]:
    """Generate once per rank with the frozen formal kwargs and parse all beams."""

    import torch

    context = preflight_torchrun_environment()
    kwargs = formal_generation_kwargs(synchronized=True)
    expected_kwargs = {
        "num_beams": FORMAL_NUM_BEAMS,
        "num_return_sequences": FORMAL_NUM_BEAMS,
        "max_new_tokens": 512,
        "synced_gpus": True,
    }
    if any(kwargs.get(key) != value for key, value in expected_kwargs.items()):
        raise ContinuityGateError("formal generation kwargs drifted before memory smoke")
    torch.distributed.barrier()
    swap_before = _swap_used_kib() if context.rank == 0 else None
    values = [swap_before]
    torch.distributed.broadcast_object_list(values, src=0)
    swap_before = int(values[0])
    device = torch.device("cuda", context.local_rank)
    torch.cuda.reset_peak_memory_stats(device)
    recording = _RecordingGenerationModel(generation_module_from_trainer(branch.trainer))
    row = generate_reaction(
        recording,
        tokenizer,
        record,
        ordinal=ordinal,
        synchronized=True,
    )
    formal_output_shape = recording.output_shape
    if formal_output_shape is None:
        raise ContinuityGateError("generation shape recorder did not observe formal output")
    stress = _run_full_length_generation_stress(
        recording,
        prompt,
        local_rank=context.local_rank,
    )
    evaluation = evaluate_reaction(
        row.reaction_id,
        row.raw_beams,
        record.ts_edges,
        atom_count=record.atom_count,
        report_k=config.generation.report_k or (),
    )
    torch.cuda.synchronize(device)
    device_memory_used = _device_memory_used_mib(context.local_rank)
    if device_memory_used is None:
        raise ContinuityGateError("nvidia-smi could not report whole-device memory use")
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "peak_allocated_mib": int(torch.cuda.max_memory_allocated(device) // 1024**2),
        "peak_reserved_mib": int(torch.cuda.max_memory_reserved(device) // 1024**2),
        "device_memory_used_mib": device_memory_used,
        "output_shape": formal_output_shape,
        "parsed_beams": len(evaluation.parses),
        "valid_parses": sum(parse.valid for parse in evaluation.parses),
        "stress_output_shape": stress["output_shape"],
        "stress_actual_new_tokens": stress["actual_new_tokens"],
    }
    local["effective_peak_mib"] = max(
        int(local["peak_reserved_mib"]),
        int(local["device_memory_used_mib"]),
    )
    ranks: list[Any] = [None] * context.world_size
    torch.distributed.all_gather_object(ranks, local)
    torch.distributed.barrier()
    swap_after = _swap_used_kib() if context.rank == 0 else None
    values = [swap_after]
    torch.distributed.broadcast_object_list(values, src=0)
    swap_growth = int(values[0]) - swap_before
    if any(rank["output_shape"] != local["output_shape"] for rank in ranks):
        raise ContinuityGateError("generation output shape differs between ranks")
    payload = {
        "status": "pass",
        "reaction_id": row.reaction_id,
        "ordinal": ordinal,
        "prompt_sha256": prompt.prompt_sha256,
        "prompt_tokens": len(prompt.input_ids),
        "num_beams": int(kwargs["num_beams"]),
        "num_return_sequences": int(kwargs["num_return_sequences"]),
        "max_new_tokens": int(kwargs["max_new_tokens"]),
        "output_shape": formal_output_shape,
        "parsed_beams": len(evaluation.parses),
        "valid_parses": sum(parse.valid for parse in evaluation.parses),
        "swap_growth_kib": swap_growth,
        "max_swap_growth_kib": MAX_SWAP_GROWTH_KIB,
        "max_device_memory_mib": config.runtime.max_gpu_peak_mib,
        "stress": stress,
        "ranks": ranks,
    }
    validate_generation_smoke_evidence(
        payload,
        max_device_memory_mib=config.runtime.max_gpu_peak_mib,
    )
    return payload


def _verify_checkpoint_bundle_binding(
    config: ExperimentConfig,
    checkpoint: DurableCheckpoint,
    bundle: Path,
) -> str:
    bundle_sha = sha256_file(bundle / "manifest.json")
    expected_model = sha256_json(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "pissa_manifest_sha256": bundle_sha,
        }
    )
    if checkpoint.model_fingerprint != expected_model:
        raise ContinuityGateError("durable checkpoint is bound to a different PiSSA bundle")
    if checkpoint.config_fingerprint != config.sha256:
        raise ContinuityGateError("durable checkpoint is bound to a different configuration")
    return bundle_sha


def run_portable_parity_gate(
    config: ExperimentConfig,
    *,
    bundle: str | Path,
    processed_path: str | Path,
    checkpoint_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Run real residual/rank32 versus original/rank64 inference parity."""

    install_frozen_environment()
    context = preflight_torchrun_environment()
    bundle_path = Path(bundle).resolve(strict=True)
    read_complete_manifest(bundle_path)
    checkpoint = inspect_durable_checkpoint(
        checkpoint_dir,
        processed_path,
        config=config,
    )
    bundle_sha = _verify_checkpoint_bundle_binding(config, checkpoint, bundle_path)
    tokenizer = load_pinned_tokenizer(config, local_files_only=True)
    datasets = load_processed_dataset(processed_path)
    if "val" not in datasets:
        raise ContinuityGateError("processed dataset lacks validation split")
    if len(datasets["val"]) != EXPECTED_FORMAL_SPLIT_COUNTS["val"]:
        raise ContinuityGateError("portable parity requires the complete frozen validation split")
    records = tuple(
        reaction_record_from_row(datasets["val"][index]) for index in range(len(datasets["val"]))
    )
    if any(record.split != "val" for record in records):
        raise ContinuityGateError("portable parity validation data contains a foreign split")
    ordinal, record, prompt = select_longest_canonical_prompt(
        records,
        lambda value: encode_formal_prompt(tokenizer, value),
    )
    workspace = _invocation_workspace(
        config,
        gate="portable-parity",
        identity=checkpoint.checkpoint_fingerprint,
        output=output,
    )

    residual = _build_inference_branch(
        config,
        checkpoint=checkpoint,
        bundle=bundle_path,
        prompt=prompt,
        output_dir=workspace / "residual-trainer",
        kind="residual-rank32",
    )
    residual_evidence = dict(residual.evidence)
    residual_logits = residual.logits
    _release_inference_branch(residual)

    portable = _build_inference_branch(
        config,
        checkpoint=checkpoint,
        bundle=bundle_path,
        prompt=prompt,
        output_dir=workspace / "portable-trainer",
        kind="original-portable-rank64",
    )
    parity = assert_logits_parity(residual_logits, portable.logits)
    if checkpoint.epoch == 1:
        generation_smoke = _generation_memory_smoke(
            config,
            branch=portable,
            tokenizer=tokenizer,
            record=record,
            ordinal=ordinal,
            prompt=prompt,
        )
    else:
        generation_smoke = {
            "status": "not-run",
            "reason": "frozen-full-length-generation-memory-gate-runs-on-epoch-1-only",
            "epoch": checkpoint.epoch,
        }
    portable_evidence = dict(portable.evidence)

    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "residual_logits_sha256": residual_evidence["last_token_logits"]["sha256"],
        "portable_logits_sha256": portable_evidence["last_token_logits"]["sha256"],
        "argmax_token_id": parity["argmax_token_id"],
        "parity": parity,
    }
    gathered: list[Any] = [None] * context.world_size
    import torch

    torch.distributed.all_gather_object(gathered, local)
    if any(rank["argmax_token_id"] != parity["argmax_token_id"] for rank in gathered):
        raise ContinuityGateError("portable parity argmax differs between ranks")
    if len({rank["residual_logits_sha256"] for rank in gathered}) != 1:
        raise ContinuityGateError("residual logits differ between distributed ranks")
    if len({rank["portable_logits_sha256"] for rank in gathered}) != 1:
        raise ContinuityGateError("portable logits differ between distributed ranks")
    _release_inference_branch(portable)
    report = {
        "schema_version": CONTINUITY_GATE_SCHEMA_VERSION,
        "gate": PORTABLE_GATE_NAME,
        "status": "pass",
        "host": socket.gethostname(),
        "world_size": context.world_size,
        "checkpoint": {
            "path": str(checkpoint.path),
            "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
            "run_fingerprint": checkpoint.run_fingerprint,
            "config_fingerprint": checkpoint.config_fingerprint,
            "data_fingerprint": checkpoint.data_fingerprint,
            "model_fingerprint": checkpoint.model_fingerprint,
            "epoch": checkpoint.epoch,
            "global_step": checkpoint.global_step,
            "bundle_manifest_sha256": bundle_sha,
        },
        "probe": {
            "reaction_id": record.reaction_id,
            "ordinal": ordinal,
            "prompt_sha256": prompt.prompt_sha256,
            "prompt_tokens": len(prompt.input_ids),
            "selection": "first-longest-legal-canonical-validation-prompt",
        },
        "residual_rank32": residual_evidence,
        "original_portable_rank64": portable_evidence,
        "logit_parity": parity,
        "generation_smoke": generation_smoke,
        "ranks": gathered,
    }
    if context.rank == 0:
        write_json(output, report)
    torch.distributed.barrier()
    if context.rank == 0 and workspace.exists():
        shutil.rmtree(workspace)
    torch.distributed.barrier()
    return report


def _is_cuda_oom(error: BaseException) -> bool:
    try:
        import torch

        if isinstance(error, torch.OutOfMemoryError):
            return True
    except ImportError:  # pragma: no cover - runtime always has torch
        pass
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _best_effort_manifest_sha256(path_value: str | None) -> str | None:
    if not path_value:
        return None
    try:
        path = Path(path_value).resolve(strict=True)
        manifest = path / "manifest.json" if path.is_dir() else path
        return sha256_file(manifest) if manifest.is_file() else None
    except OSError:
        return None


def _oom_sidecar(
    output: str | Path,
    *,
    command: str,
    error: BaseException,
    config: ExperimentConfig,
) -> Path:
    rank = int(os.environ.get("RANK", "-1"))
    destination = Path(output)
    sidecar = destination.with_name(f"{destination.name}.rank-{rank:05d}.oom.json")
    bundle_path = os.environ.get("JANUS_CONTINUITY_BUNDLE")
    checkpoint_path = os.environ.get("JANUS_CONTINUITY_CHECKPOINT")
    payload = {
        "schema_version": CONTINUITY_GATE_SCHEMA_VERSION,
        "gate": command,
        "status": "oom",
        "exit_code": OOM_EXIT_CODE,
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "world_size": int(os.environ.get("WORLD_SIZE", "-1")),
        "config": os.environ.get("JANUS_CONTINUITY_CONFIG"),
        "config_sha256": config.sha256,
        "bundle": bundle_path,
        "bundle_manifest_sha256": _best_effort_manifest_sha256(bundle_path),
        "checkpoint": checkpoint_path,
        "checkpoint_manifest_sha256": _best_effort_manifest_sha256(checkpoint_path),
        "exception_type": type(error).__name__,
        "exception": str(error)[:1000],
    }
    write_json(sidecar, payload)
    return sidecar


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--config", required=True)
    resume.add_argument("--bundle", required=True)
    resume.add_argument("--output", required=True)
    resume.add_argument("--micro-batch-size", required=True, type=int, choices=(1, 2))
    portable = subparsers.add_parser("portable-parity")
    portable.add_argument("--config", required=True)
    portable.add_argument("--bundle", required=True)
    portable.add_argument("--processed-path", required=True)
    portable.add_argument("--checkpoint", required=True)
    portable.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    os.environ["JANUS_CONTINUITY_CONFIG"] = str(arguments.config)
    os.environ["JANUS_CONTINUITY_BUNDLE"] = str(arguments.bundle)
    os.environ["JANUS_CONTINUITY_CHECKPOINT"] = str(getattr(arguments, "checkpoint", ""))
    config = load_config(arguments.config)
    try:
        if arguments.command == "resume":
            run_resume_gate(
                config,
                bundle=arguments.bundle,
                output=arguments.output,
                micro_batch_size=arguments.micro_batch_size,
            )
        else:
            run_portable_parity_gate(
                config,
                bundle=arguments.bundle,
                processed_path=arguments.processed_path,
                checkpoint_dir=arguments.checkpoint,
                output=arguments.output,
            )
    except BaseException as exc:
        if _is_cuda_oom(exc):
            _oom_sidecar(
                arguments.output,
                command=arguments.command,
                error=exc,
                config=config,
            )
            return OOM_EXIT_CODE
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by torchrun
    raise SystemExit(main())


__all__ = [
    "CONTINUITY_GATE_SCHEMA_VERSION",
    "ContinuityGateError",
    "assert_logits_parity",
    "compare_resume_evidence",
    "execute_restart_sequence",
    "make_resume_gate_identity",
    "rank_local_rng_sha256",
    "run_portable_parity_gate",
    "run_resume_gate",
    "select_longest_canonical_prompt",
    "stable_state_sha256",
    "stream_lora_tensor_hashes",
    "validate_generation_smoke_evidence",
]
