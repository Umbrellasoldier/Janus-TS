"""Two-rank ZeRO-3 model loading, memory smoke, and full training entrypoint.

The order in this module is intentional.  ``TrainingArguments`` is created
before any model object so Transformers installs its ZeRO-3 initialization
context before ``from_pretrained`` sees a single 27B parameter.  The prepared
PiSSA residual and its canonical rank-32 initializer are then loaded without a
device map, CPU offload, or disk offload.

This module is directly runnable under ``torchrun``::

    torchrun --standalone --nproc-per-node=2 -m janus_ts.distributed memory-smoke ...
    torchrun --standalone --nproc-per-node=2 -m janus_ts.distributed train ...

Checkpoint/evaluation policy remains injectable through callback factories.
That keeps construction testable and lets the crash-safe checkpoint manager
own persistence without teaching the training entrypoint a second checkpoint
format.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from transformers import TrainerCallback

from .artifacts import read_complete_manifest, sha256_file, sha256_json, write_json
from .config import ExperimentConfig, load_config
from .constants import (
    MAX_SEQUENCE_LENGTH,
    QWEN_IM_END_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
    SEED,
)
from .modeling import (
    EXPECTED_TRAINABLE_PARAMETERS,
    assert_pissa_adapter_contract,
    validate_loading_info,
    validate_pissa_initialization_config,
    validate_qwen_text_config,
)
from .preprocessing import load_pinned_tokenizer, load_processed_dataset
from .runtime import install_frozen_environment
from .tokenization import CausalLMCollator, EpochAwareTokenizedDataset, QwenTrainingEncoder
from .training import (
    TokenNormalizedTrainer,
    assert_accelerate_zero3_precision_state,
    assert_zero3_bf16_precision,
    assert_zero3_engine_precision_contract,
    build_epoch_callback,
    build_training_argument_kwargs,
    build_training_arguments,
)

SMOKE_OOM_EXIT_CODE = 42
MAX_SWAP_GROWTH_KIB = 256 * 1024
RunMode = Literal["memory-smoke", "train"]
CallbackFactory = Callable[[Callable[[], Any]], Any]
ManagedRestoreHook = Callable[[Any, Path], Any]


class DistributedTrainingError(RuntimeError):
    """A distributed launch or training invariant was violated."""


class JanusManagedTrainer(TokenNormalizedTrainer):
    """Trainer bridge for the atomic :mod:`janus_ts.checkpointing` format.

    Transformers' native DeepSpeed resume expects a different directory
    layout and must never see a Janus checkpoint.  Under explicit Janus
    ownership this subclass still lets Trainer load the top-level
    ``trainer_state.json`` for epoch/batch skipping, but initializes a fresh
    engine and delegates the actual ZeRO/optimizer/scheduler/RNG restore to one
    supplied hook before callbacks or the first batch.  Native checkpoint
    writes and native RNG reloads are suppressed only in that explicit mode.
    """

    def __init__(
        self,
        *args: Any,
        janus_managed_checkpoints: bool = False,
        janus_restore_hook: ManagedRestoreHook | None = None,
        **kwargs: Any,
    ) -> None:
        if janus_restore_hook is not None and not janus_managed_checkpoints:
            raise DistributedTrainingError(
                "a Janus restore hook requires explicit Janus checkpoint ownership"
            )
        self.janus_managed_checkpoints = bool(janus_managed_checkpoints)
        self.janus_restore_hook = janus_restore_hook
        self.janus_restore_result: Any | None = None
        self._janus_active_resume_path: Path | None = None
        super().__init__(*args, **kwargs)

    def _prepare_for_training(
        self,
        max_steps: int,
        train_dataloader: Any,
        resume_from_checkpoint: str | None,
    ) -> tuple[Any, Any]:
        if not self.janus_managed_checkpoints or resume_from_checkpoint is None:
            return super()._prepare_for_training(
                max_steps, train_dataloader, resume_from_checkpoint
            )
        if self.janus_restore_hook is None:
            raise DistributedTrainingError("Janus-managed resume requires an explicit restore hook")
        checkpoint_path = Path(resume_from_checkpoint).resolve(strict=True)
        model, prepared_dataloader = super()._prepare_for_training(
            max_steps,
            train_dataloader,
            None,
        )
        # At this point the fresh DeepSpeed engine and optimizer exist, while
        # Trainer's epoch loop and dataloader have not consumed an example.
        self._janus_active_resume_path = checkpoint_path
        self.janus_restore_result = self.janus_restore_hook(self, checkpoint_path)
        return model, prepared_dataloader

    def _load_rng_state(self, checkpoint: str | None) -> None:
        if self.janus_managed_checkpoints and self._janus_active_resume_path is not None:
            if (
                checkpoint is None
                or Path(checkpoint).resolve(strict=True) != self._janus_active_resume_path
            ):
                raise DistributedTrainingError("Trainer attempted RNG restore from the wrong path")
            restore_rng = getattr(self.janus_restore_hook, "restore_rng_state", None)
            if not callable(restore_rng):
                raise DistributedTrainingError(
                    "Janus restore hook cannot replay RNG after dataloader skipping"
                )
            # CheckpointManager.restore performed an initial replay. Trainer
            # deliberately requests the same state again after fast-forwarding
            # a resumed dataloader, because skipping can consume RNG.
            restore_rng()
            return
        super()._load_rng_state(checkpoint)

    def _save_checkpoint(self, model: Any, trial: Any) -> None:
        if self.janus_managed_checkpoints:
            # JanusCheckpointCallback owns atomic resume/portable artifacts.
            return
        super()._save_checkpoint(model, trial)


def checkpoint_manager_restore_hook(
    manager: Any,
    checkpoint: Any,
) -> ManagedRestoreHook:
    """Adapt :class:`CheckpointManager.restore` to ``JanusManagedTrainer``.

    The selected ``ResumeCheckpoint`` remains the source of truth; the path
    passed through Trainer is accepted only when it resolves to that exact
    complete artifact.  Trainer state is loaded independently from the same
    top-level JSON, so the hook also cross-checks step and epoch before the
    first dataloader access.
    """

    expected_path = Path(checkpoint.path).resolve(strict=True)

    def restore(trainer: Any, checkpoint_path: Path) -> Any:
        if checkpoint_path != expected_path:
            raise DistributedTrainingError(
                f"managed resume path {checkpoint_path} != selected checkpoint {expected_path}"
            )
        restored = manager.restore(
            trainer.model_wrapped,
            checkpoint,
            lr_scheduler=trainer.lr_scheduler,
        )
        payload = restored.trainer_state
        actual_step = int(trainer.state.global_step)
        expected_step = int(payload["global_step"])
        actual_epoch = float(trainer.state.epoch or 0.0)
        expected_epoch = float(payload.get("epoch") or 0.0)
        if actual_step != expected_step or not math.isclose(
            actual_epoch,
            expected_epoch,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DistributedTrainingError(
                "Trainer/Janus restored state mismatch: "
                f"step {actual_step}/{expected_step}, epoch {actual_epoch}/{expected_epoch}"
            )
        return restored

    def restore_rng_state() -> None:
        manager.restore_rng(checkpoint)

    restore.restore_rng_state = restore_rng_state  # type: ignore[attr-defined]

    return restore


@dataclass(frozen=True, slots=True)
class TorchrunContext:
    """The exact process identity of one member of the two-rank job."""

    rank: int
    local_rank: int
    world_size: int
    device_index: int


@dataclass(slots=True)
class DistributedRun:
    """Objects needed to train and to attach late-bound persistence hooks."""

    arguments: Any
    model: Any
    tokenizer: Any
    train_dataset: EpochAwareTokenizedDataset
    eval_dataset: EpochAwareTokenizedDataset
    collator: CausalLMCollator
    trainer: Any
    context: TorchrunContext


@dataclass(frozen=True, slots=True)
class MemorySmokeOutcome:
    """JSON-safe result for a frozen smoke geometry.

    ``oom`` and ``rejected`` are expected only for the optional microbatch-2
    candidate and both select the already-confirmed microbatch-1 default.
    """

    status: Literal["pass", "oom", "rejected"]
    report: Mapping[str, Any]

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "pass" else SMOKE_OOM_EXIT_CODE


class SyntheticSequenceDataset:
    """Worst-case fixed-length causal examples for one optimizer update."""

    def __init__(self, examples: int, *, sequence_length: int = MAX_SEQUENCE_LENGTH) -> None:
        if isinstance(examples, bool) or not isinstance(examples, int) or examples <= 0:
            raise DistributedTrainingError("synthetic example count must be a positive integer")
        if sequence_length != MAX_SEQUENCE_LENGTH:
            raise DistributedTrainingError(
                f"memory smoke is frozen at sequence length {MAX_SEQUENCE_LENGTH}"
            )
        input_ids = [QWEN_PAD_TOKEN_ID] * sequence_length
        input_ids[-1] = QWEN_IM_END_TOKEN_ID
        self._feature = {
            "input_ids": input_ids,
            "attention_mask": [1] * sequence_length,
            # Full supervision retains the maximum logits/loss memory path.
            "labels": list(input_ids),
        }
        self._examples = examples

    def __len__(self) -> int:
        return self._examples

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        if index < 0 or index >= self._examples:
            raise IndexError(index)
        return {name: list(values) for name, values in self._feature.items()}


class PostEngineModelContractCallback(TrainerCallback):
    """Re-check mixed precision after DeepSpeed has partitioned the model."""

    def __init__(self) -> None:
        self._engine_getter: Callable[[], Any] | None = None
        self._optimizer_states_verified = False

    def bind_engine_getter(self, engine_getter: Callable[[], Any]) -> None:
        if self._engine_getter is not None:
            raise DistributedTrainingError("post-engine contract callback was bound twice")
        self._engine_getter = engine_getter

    def _engine(self) -> Any:
        if self._engine_getter is None:
            raise DistributedTrainingError("post-engine contract callback was not bound")
        engine = self._engine_getter()
        if engine is None:
            raise DistributedTrainingError("DeepSpeed engine is not initialized")
        return engine

    def _validate(self, *, require_optimizer_states: bool) -> None:
        engine = self._engine()
        assert_zero3_engine_precision_contract(
            engine,
            require_optimizer_states=require_optimizer_states,
        )
        model = getattr(engine, "module", None)
        if model is None:
            raise DistributedTrainingError("DeepSpeed engine has no module")
        if getattr(model.config, "use_cache", None) is not False:
            raise DistributedTrainingError("use_cache must remain false during training")
        if getattr(model, "_janus_ts_input_grads_enabled", None) is not True:
            raise DistributedTrainingError("input gradients were not enabled")
        if getattr(model, "_janus_ts_reentrant_gc_enabled", None) is not True:
            raise DistributedTrainingError("reentrant gradient checkpointing was not enabled")

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        resumed = int(state.global_step) > 0
        self._validate(require_optimizer_states=resumed)
        self._optimizer_states_verified = resumed
        return control

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        if int(state.global_step) >= 1 and not self._optimizer_states_verified:
            self._validate(require_optimizer_states=True)
            self._optimizer_states_verified = True
        return control


def preflight_torchrun_environment(
    environ: Mapping[str, str] | None = None,
) -> TorchrunContext:
    """Validate torchrun identity without importing or allocating with Torch."""

    environment = os.environ if environ is None else environ
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in environment]
    if missing:
        raise DistributedTrainingError(
            "launch with torchrun; missing environment variables " + ", ".join(missing)
        )
    try:
        rank = int(environment["RANK"])
        local_rank = int(environment["LOCAL_RANK"])
        world_size = int(environment["WORLD_SIZE"])
    except ValueError as exc:
        raise DistributedTrainingError("torchrun rank variables must be integers") from exc
    if world_size != 2 or rank not in (0, 1) or local_rank not in (0, 1):
        raise DistributedTrainingError(
            "the frozen run requires WORLD_SIZE=2 with ranks/local ranks 0 and 1; "
            f"got rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )
    return TorchrunContext(rank, local_rank, world_size, local_rank)


def configure_reproducibility(torch_module: Any, *, seed: int = SEED) -> None:
    """Install the confirmed reproducibility policy without strict algorithms."""

    if seed != SEED:
        raise DistributedTrainingError(f"the frozen seed is {SEED}, got {seed}")
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(False)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cuda.matmul.allow_tf32 = False


def _load_deepspeed_payload(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistributedTrainingError(f"invalid DeepSpeed config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DistributedTrainingError("DeepSpeed config must be a JSON object")
    return payload


def assert_zero3_no_offload(
    arguments: Any,
    *,
    zero3_enabled: Callable[[], bool] | None = None,
) -> None:
    """Prove that TrainingArguments activated ZeRO-3 with no offload."""

    if zero3_enabled is None:
        from transformers.integrations import is_deepspeed_zero3_enabled

        zero3_enabled = is_deepspeed_zero3_enabled
    if not zero3_enabled():
        raise DistributedTrainingError(
            "ZeRO-3 was not active before model construction; TrainingArguments order is wrong"
        )
    plugin = getattr(arguments, "deepspeed_plugin", None)
    problems: list[str] = []
    if plugin is None:
        problems.append("missing DeepSpeed plugin")
    else:
        if int(getattr(plugin, "zero_stage", -1)) != 3:
            problems.append(f"zero_stage={getattr(plugin, 'zero_stage', None)!r}")
        if str(getattr(plugin, "offload_optimizer_device", "")).lower() != "none":
            problems.append("optimizer offload is enabled")
        if str(getattr(plugin, "offload_param_device", "")).lower() != "none":
            problems.append("parameter offload is enabled")
        if getattr(plugin, "zero3_init_flag", None) is not True:
            problems.append("zero3_init_flag is not true")

    payload = _load_deepspeed_payload(arguments.deepspeed)
    zero = payload.get("zero_optimization")
    if not isinstance(zero, dict) or zero.get("stage") != 3:
        problems.append("JSON zero_optimization.stage is not 3")
    else:
        for name in ("offload_optimizer", "offload_param"):
            section = zero.get(name)
            if not isinstance(section, dict) or str(section.get("device", "")).lower() != "none":
                problems.append(f"JSON {name}.device is not none")
    if problems:
        raise DistributedTrainingError("unsafe DeepSpeed configuration: " + "; ".join(problems))


def assert_initialized_two_rank_job(
    arguments: Any,
    context: TorchrunContext,
    torch_module: Any,
) -> None:
    """Cross-check TrainingArguments, CUDA, and the initialized NCCL group."""

    dist = torch_module.distributed
    if not dist.is_available() or not dist.is_initialized():
        raise DistributedTrainingError("TrainingArguments did not initialize torch.distributed")
    actual = (dist.get_rank(), dist.get_world_size())
    if actual != (context.rank, context.world_size):
        raise DistributedTrainingError(
            f"process-group identity {actual!r} disagrees with torchrun {context!r}"
        )
    if int(getattr(arguments, "world_size", -1)) != 2:
        raise DistributedTrainingError(
            f"TrainingArguments world_size={getattr(arguments, 'world_size', None)!r}, expected 2"
        )
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 2:
        raise DistributedTrainingError("exactly two visible CUDA devices are required")
    torch_module.cuda.set_device(context.local_rank)
    current = int(torch_module.cuda.current_device())
    if current != context.local_rank:
        raise DistributedTrainingError(
            f"current CUDA device {current} does not equal LOCAL_RANK={context.local_rank}"
        )


def _is_lora_parameter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def assert_model_precision_and_freezing(
    model: Any,
    torch_module: Any,
    *,
    expected_trainable_parameters: int = EXPECTED_TRAINABLE_PARAMETERS,
) -> None:
    """Check BF16 frozen base and BF16 trainable adapters under ZeRO-3."""

    trainable = 0
    problems: list[str] = []
    for name, parameter in model.named_parameters():
        logical_numel = int(getattr(parameter, "ds_numel", parameter.numel()))
        adapter = _is_lora_parameter(name)
        if adapter:
            if not parameter.requires_grad:
                problems.append(f"frozen adapter {name}")
            else:
                trainable += logical_numel
            if parameter.dtype != torch_module.bfloat16:
                problems.append(f"non-BF16 adapter {name}: {parameter.dtype}")
        else:
            if parameter.requires_grad:
                problems.append(f"trainable base {name}")
            if parameter.dtype != torch_module.bfloat16:
                problems.append(f"non-BF16 base {name}: {parameter.dtype}")
    if trainable != expected_trainable_parameters:
        problems.append(
            f"trainable logical parameters={trainable:,}, expected "
            f"{expected_trainable_parameters:,}"
        )
    if problems:
        raise DistributedTrainingError(
            "model precision/freezing contract failed: " + "; ".join(problems[:12])
        )


def load_zero3_prepared_model(
    bundle_dir: str | Path,
    arguments: Any,
    *,
    torch_module: Any | None = None,
    base_loader: Callable[..., Any] | None = None,
    adapter_loader: Callable[..., Any] | None = None,
    require_complete: bool = True,
) -> Any:
    """Load residual base + PiSSA init after ZeRO-3 arguments exist.

    Loader injection is only for CPU unit tests.  Production callers use the
    pinned Qwen and PEFT classes imported inside this function.
    """

    assert_zero3_no_offload(arguments)
    checkpointing_kwargs = getattr(arguments, "gradient_checkpointing_kwargs", None)
    if checkpointing_kwargs != {"use_reentrant": True}:
        raise DistributedTrainingError(
            "ZeRO-3 training requires frozen reentrant gradient checkpointing"
        )
    if torch_module is None:
        import torch as torch_module

    from .artifacts import read_complete_manifest

    bundle = Path(bundle_dir)
    if require_complete:
        read_complete_manifest(bundle)
    residual_dir = bundle / "residual_base"
    initial_dir = bundle / "pissa_init"
    if not residual_dir.is_dir() or not initial_dir.is_dir():
        raise DistributedTrainingError(
            f"prepared bundle lacks residual_base or pissa_init: {bundle}"
        )

    if base_loader is None:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        text_config = Qwen3_5TextConfig.from_pretrained(
            residual_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        validate_qwen_text_config(text_config)
        validate_pissa_initialization_config(
            initial_dir / "adapter_config.json", text_config, prepared_reference=True
        )

        def base_loader(**kwargs: Any) -> Any:
            return Qwen3_5ForCausalLM.from_pretrained(residual_dir, **kwargs)

    else:
        text_config = None

    loaded = base_loader(
        config=text_config,
        dtype=torch_module.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=False,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise DistributedTrainingError("base loader must return (model, loading_info)")
    base, loading_info = loaded
    validate_loading_info(loading_info)
    if text_config is None:
        validate_qwen_text_config(base.config)
        validate_pissa_initialization_config(
            initial_dir / "adapter_config.json", base.config, prepared_reference=True
        )
    if getattr(base, "hf_device_map", None):
        raise DistributedTrainingError("distributed model load must not create a device map")
    base.config.use_cache = False
    base.config.pad_token_id = QWEN_PAD_TOKEN_ID
    base.config.eos_token_id = QWEN_IM_END_TOKEN_ID

    if adapter_loader is None:
        from peft import PeftModel

        def adapter_loader(base_model: Any, adapter_path: Path, **kwargs: Any) -> Any:
            return PeftModel.from_pretrained(base_model, adapter_path, **kwargs)

    model = adapter_loader(
        base,
        initial_dir,
        is_trainable=True,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=False,
    )
    if getattr(model, "hf_device_map", None):
        raise DistributedTrainingError("PEFT load unexpectedly created a device map")

    # The serialized PiSSA initializer is FP32. Cast it explicitly so the
    # training dtype does not depend on PEFT loading defaults.
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
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(checkpointing_kwargs))
    model._janus_ts_input_grads_enabled = True
    model._janus_ts_reentrant_gc_enabled = True
    model.train()
    assert_pissa_adapter_contract(model, adapter_dtype=torch_module.bfloat16)
    assert_model_precision_and_freezing(model, torch_module)
    precision = assert_zero3_bf16_precision(arguments, model)
    model._janus_ts_precision_report = precision
    return model


def build_memory_smoke_arguments(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    micro_batch_size_per_gpu: int,
) -> Any:
    """Instantiate one-update TrainingArguments before the smoke model load."""

    if micro_batch_size_per_gpu == config.train.micro_batch_size_per_gpu:
        accumulation = config.train.gradient_accumulation_steps
    elif micro_batch_size_per_gpu == config.train.candidate_micro_batch_size_per_gpu:
        accumulation = config.train.candidate_gradient_accumulation_steps
    else:
        raise DistributedTrainingError(
            "memory smoke micro batch must be the frozen default or candidate value"
        )
    kwargs = build_training_argument_kwargs(
        config,
        output_dir=output_dir,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
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
            "max_steps": 1,
        }
    )
    from transformers import TrainingArguments

    return TrainingArguments(**kwargs)


def _swap_used_kib() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    return values["SwapTotal"] - values["SwapFree"]


def _is_cuda_oom(error: BaseException, torch_module: Any) -> bool:
    oom_type = getattr(torch_module, "OutOfMemoryError", ())
    if oom_type and isinstance(error, oom_type):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _oom_sidecar_path(report_path: str | Path, rank: int) -> Path:
    path = Path(report_path)
    return path.with_name(f"{path.name}.rank-{rank}.oom.json")


def _local_gpu_used_mib(local_rank: int) -> int | None:
    """Read whole-device memory use; allocator peaks alone omit CUDA libraries."""

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


def run_memory_smoke(
    config: ExperimentConfig,
    *,
    bundle_dir: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    micro_batch_size_per_gpu: int,
    model_loader: Callable[[str | Path, Any], Any] = load_zero3_prepared_model,
    trainer_class: type = TokenNormalizedTrainer,
) -> MemorySmokeOutcome:
    """Run exactly one worst-case 2048-token optimizer update on two ranks."""

    install_frozen_environment()
    context = preflight_torchrun_environment()
    bundle = Path(bundle_dir).resolve(strict=True)
    read_complete_manifest(bundle)
    smoke_identity = sha256_json(
        {
            "gate": "zero3-worst-case-2048-one-update",
            "config_fingerprint": config.sha256,
            "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "deepspeed_config_sha256": sha256_file(config.train.deepspeed_config),
            "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
            "sequence_length": MAX_SEQUENCE_LENGTH,
            "world_size": 2,
        }
    )
    # This must precede model_loader: it activates Transformers' ZeRO-3 Init.
    arguments = build_memory_smoke_arguments(
        config,
        output_dir=output_dir,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
    )
    import torch

    configure_reproducibility(torch, seed=config.seed)
    assert_initialized_two_rank_job(arguments, context, torch)
    device = torch.device("cuda", context.local_rank)
    torch.cuda.reset_peak_memory_stats(device)
    swap_before = _swap_used_kib()

    model: Any | None = None
    trainer: Any | None = None
    try:
        model = model_loader(bundle_dir, arguments)
        accumulation = int(arguments.gradient_accumulation_steps)
        examples = int(arguments.per_device_train_batch_size) * context.world_size * accumulation
        dataset = SyntheticSequenceDataset(examples)
        precision_callback = PostEngineModelContractCallback()
        trainer = trainer_class(
            model=model,
            args=arguments,
            train_dataset=dataset,
            data_collator=CausalLMCollator(),
            callbacks=[precision_callback],
        )
        precision_callback.bind_engine_getter(lambda: trainer.model_wrapped)
        assert_accelerate_zero3_precision_state(arguments, trainer.accelerator)
        train_output = trainer.train()
        if int(trainer.state.global_step) != 1:
            raise DistributedTrainingError(
                f"memory smoke completed {trainer.state.global_step} updates, expected 1"
            )
        torch.cuda.synchronize(device)
    except BaseException as exc:
        if not _is_cuda_oom(exc, torch):
            raise
        trainer = None
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        local_report = {
            "gate": "zero3-worst-case-2048-one-update",
            "gate_identity": smoke_identity,
            "config_fingerprint": config.sha256,
            "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "status": "oom",
            "worker_cleanup_attempted": True,
            "exit_code": SMOKE_OOM_EXIT_CODE,
            "rank": context.rank,
            "local_rank": context.local_rank,
            "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
            "exception_type": type(exc).__name__,
            "exception": str(exc)[:1000],
        }
        write_json(_oom_sidecar_path(report_path, context.rank), local_report)
        return MemorySmokeOutcome("oom", local_report)

    if trainer is None:
        raise DistributedTrainingError("memory smoke lost its Trainer without raising")
    training_loss = float(train_output.training_loss)
    if not math.isfinite(training_loss):
        raise DistributedTrainingError(f"memory smoke produced non-finite loss: {training_loss}")
    local_report = {
        "gate_identity": smoke_identity,
        "config_fingerprint": config.sha256,
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "rank": context.rank,
        "local_rank": context.local_rank,
        "gpu_name": torch.cuda.get_device_name(device),
        "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "gradient_accumulation_steps": int(arguments.gradient_accumulation_steps),
        "sequence_length": MAX_SEQUENCE_LENGTH,
        "global_step": int(trainer.state.global_step),
        "training_loss": training_loss,
        "precision": getattr(model, "_janus_ts_precision_report", None),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) // 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) // 1024**2,
        "device_memory_used_mib": _local_gpu_used_mib(context.local_rank),
        "swap_growth_kib": _swap_used_kib() - swap_before,
    }
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    torch.distributed.all_gather_object(gathered, local_report)
    torch.distributed.barrier()
    ranks = [item for item in gathered if item is not None]
    failures: list[str] = []
    for item in ranks:
        peak = max(
            int(item["peak_reserved_mib"]),
            int(item["device_memory_used_mib"] or 0),
        )
        if peak > config.runtime.max_gpu_peak_mib:
            failures.append(
                f"rank {item['rank']} peak {peak}MiB > {config.runtime.max_gpu_peak_mib}MiB"
            )
        if int(item["swap_growth_kib"]) > MAX_SWAP_GROWTH_KIB:
            failures.append(
                f"rank {item['rank']} swap growth {item['swap_growth_kib']}KiB > "
                f"{MAX_SWAP_GROWTH_KIB}KiB"
            )
    if failures:
        report = {
            "gate": "zero3-worst-case-2048-one-update",
            "gate_identity": smoke_identity,
            "config_fingerprint": config.sha256,
            "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "status": "rejected",
            "reason": "resource-threshold",
            "host": socket.gethostname(),
            "world_size": context.world_size,
            "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
            "failures": failures,
            "ranks": ranks,
        }
        if context.rank == 0:
            write_json(report_path, report)
        torch.distributed.barrier()
        if micro_batch_size_per_gpu == config.train.candidate_micro_batch_size_per_gpu:
            return MemorySmokeOutcome("rejected", report)
        raise DistributedTrainingError(
            "default memory smoke resource gate failed: " + "; ".join(failures)
        )
    report = {
        "gate": "zero3-worst-case-2048-one-update",
        "gate_identity": smoke_identity,
        "config_fingerprint": config.sha256,
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "status": "pass",
        "host": socket.gethostname(),
        "world_size": context.world_size,
        "ranks": ranks,
    }
    if context.rank == 0:
        write_json(report_path, report)
    return MemorySmokeOutcome("pass", report)


def _attach_callback_factories(
    trainer: Any,
    factories: Sequence[CallbackFactory],
) -> None:
    def engine_getter() -> Any:
        return trainer.model_wrapped

    for factory in factories:
        callback = factory(engine_getter)
        if callback is None:
            raise DistributedTrainingError("callback factory returned None")
        trainer.add_callback(callback)


def build_full_distributed_run(
    config: ExperimentConfig,
    *,
    bundle_dir: str | Path,
    processed_path: str | Path,
    output_dir: str | Path,
    micro_batch_size_per_gpu: int | None = None,
    gradient_accumulation_steps: int | None = None,
    callback_factories: Sequence[CallbackFactory] = (),
    janus_managed_checkpoints: bool = False,
    janus_restore_hook: ManagedRestoreHook | None = None,
    local_files_only: bool = True,
    model_loader: Callable[[str | Path, Any], Any] = load_zero3_prepared_model,
    trainer_class: type = JanusManagedTrainer,
) -> DistributedRun:
    """Build the five-epoch Arrow-backed trainer, with arguments first."""

    install_frozen_environment()
    context = preflight_torchrun_environment()
    # Do not move this below model_loader.  The ordering is a 27B memory gate.
    arguments = build_training_arguments(
        config,
        output_dir=output_dir,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    import torch

    configure_reproducibility(torch, seed=config.seed)
    assert_initialized_two_rank_job(arguments, context, torch)
    model = model_loader(bundle_dir, arguments)
    tokenizer = load_pinned_tokenizer(config, local_files_only=local_files_only)
    datasets = load_processed_dataset(processed_path)
    missing = {"train", "val"}.difference(datasets)
    if missing:
        raise DistributedTrainingError(
            f"processed DatasetDict lacks required splits: {sorted(missing)!r}"
        )
    encoder = QwenTrainingEncoder(
        tokenizer,
        max_length=config.model.max_sequence_length,
        seed=config.seed,
    )
    train_dataset = EpochAwareTokenizedDataset(datasets["train"], encoder, training=True)
    eval_dataset = EpochAwareTokenizedDataset(datasets["val"], encoder, training=False)
    collator = CausalLMCollator(max_length=config.model.max_sequence_length)
    precision_callback = PostEngineModelContractCallback()
    if janus_restore_hook is not None and not janus_managed_checkpoints:
        raise DistributedTrainingError(
            "a Janus restore hook requires explicit Janus checkpoint ownership"
        )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": arguments,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "callbacks": [precision_callback, build_epoch_callback(train_dataset)],
    }
    if issubclass(trainer_class, JanusManagedTrainer):
        trainer_kwargs.update(
            {
                "janus_managed_checkpoints": janus_managed_checkpoints,
                "janus_restore_hook": janus_restore_hook,
            }
        )
    elif janus_managed_checkpoints:
        raise DistributedTrainingError("Janus checkpoint ownership requires JanusManagedTrainer")
    trainer = trainer_class(
        **trainer_kwargs,
    )
    precision_callback.bind_engine_getter(lambda: trainer.model_wrapped)
    assert_accelerate_zero3_precision_state(arguments, trainer.accelerator)
    _attach_callback_factories(trainer, callback_factories)
    return DistributedRun(
        arguments=arguments,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
        trainer=trainer,
        context=context,
    )


def run_full_training(
    config: ExperimentConfig,
    *,
    bundle_dir: str | Path,
    processed_path: str | Path,
    output_dir: str | Path,
    resume_from_checkpoint: str | Path | bool | None = None,
    micro_batch_size_per_gpu: int | None = None,
    gradient_accumulation_steps: int | None = None,
    callback_factories: Sequence[CallbackFactory] = (),
    janus_managed_checkpoints: bool = False,
    janus_restore_hook: ManagedRestoreHook | None = None,
) -> tuple[DistributedRun, Any]:
    """Run five epochs with either native or explicitly managed persistence.

    A Janus ``CheckpointManager`` directory is *not* a native Transformers
    checkpoint.  Pass ``janus_managed_checkpoints=True`` plus a hook from
    :func:`checkpoint_manager_restore_hook`; otherwise such a path is rejected
    instead of being forwarded to an incompatible DeepSpeed loader.
    """

    if isinstance(resume_from_checkpoint, Path):
        resume_from_checkpoint = str(resume_from_checkpoint)
    if isinstance(resume_from_checkpoint, str) and not Path(resume_from_checkpoint).is_dir():
        raise DistributedTrainingError(
            f"resume checkpoint directory does not exist: {resume_from_checkpoint}"
        )
    if janus_managed_checkpoints and resume_from_checkpoint is True:
        raise DistributedTrainingError(
            "Janus resume requires an exact manifest-selected checkpoint path, not True"
        )
    if isinstance(resume_from_checkpoint, str):
        resume_path = Path(resume_from_checkpoint)
        looks_managed = (resume_path / ".complete").is_file() and (
            resume_path / "deepspeed"
        ).is_dir()
        if looks_managed and not janus_managed_checkpoints:
            raise DistributedTrainingError(
                "Janus CheckpointManager artifacts require explicit managed ownership"
            )
    if (
        janus_managed_checkpoints
        and resume_from_checkpoint not in (None, False)
        and janus_restore_hook is None
    ):
        raise DistributedTrainingError("Janus-managed resume has no restore hook")
    run = build_full_distributed_run(
        config,
        bundle_dir=bundle_dir,
        processed_path=processed_path,
        output_dir=output_dir,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        callback_factories=callback_factories,
        janus_managed_checkpoints=janus_managed_checkpoints,
        janus_restore_hook=janus_restore_hook,
    )
    output = run.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return run, output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("memory-smoke", "train"):
        sub = subparsers.add_parser(mode)
        sub.add_argument("--config", default="configs/transition1x.yaml")
        sub.add_argument("--bundle", required=True)
        sub.add_argument("--output-dir", required=True)
        sub.add_argument("--micro-batch-size", type=int, choices=(1, 2), default=1)
    smoke = subparsers.choices["memory-smoke"]
    smoke.add_argument("--report", required=True)
    train = subparsers.choices["train"]
    train.add_argument("--processed", required=True)
    train.add_argument("--resume-from-checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.micro_batch_size == config.train.micro_batch_size_per_gpu:
        accumulation = config.train.gradient_accumulation_steps
    else:
        accumulation = config.train.candidate_gradient_accumulation_steps
    if args.mode == "memory-smoke":
        outcome = run_memory_smoke(
            config,
            bundle_dir=args.bundle,
            output_dir=args.output_dir,
            report_path=args.report,
            micro_batch_size_per_gpu=args.micro_batch_size,
        )
        if outcome.status != "pass":
            raise SystemExit(outcome.exit_code)
    else:
        run_full_training(
            config,
            bundle_dir=args.bundle,
            processed_path=args.processed,
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            micro_batch_size_per_gpu=args.micro_batch_size,
            gradient_accumulation_steps=accumulation,
        )


if __name__ == "__main__":
    main()


__all__ = [
    "CallbackFactory",
    "DistributedRun",
    "DistributedTrainingError",
    "JanusManagedTrainer",
    "ManagedRestoreHook",
    "MemorySmokeOutcome",
    "PostEngineModelContractCallback",
    "SMOKE_OOM_EXIT_CODE",
    "SyntheticSequenceDataset",
    "TorchrunContext",
    "assert_initialized_two_rank_job",
    "assert_model_precision_and_freezing",
    "assert_zero3_no_offload",
    "build_full_distributed_run",
    "build_memory_smoke_arguments",
    "checkpoint_manager_restore_hook",
    "configure_reproducibility",
    "load_zero3_prepared_model",
    "main",
    "preflight_torchrun_environment",
    "run_full_training",
    "run_memory_smoke",
]
