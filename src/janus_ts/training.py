"""Frozen Hugging Face Trainer contract for the Transition1x run."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from transformers import Trainer, TrainerCallback, TrainingArguments

from .config import ExperimentConfig
from .constants import SEED
from .modeling import EXPECTED_TRAINABLE_PARAMETERS
from .tokenization import EpochAwareTokenizedDataset

# Keep this empty. DeepSpeed groups ZeRO-3 all-gathers by communication dtype
# and then requires each coalesced group to have one storage dtype. Marking
# every Linear as BF16-communication-safe would mix BF16 base weights with
# FP32 LoRA weights in one group. The engine autocast context still performs
# Linear compute in BF16; communication follows each parameter's storage dtype.
ZERO3_AUTOCAST_SAFE_MODULES: tuple[str, ...] = ()


class TrainingContractError(ValueError):
    """Raised when Trainer would deviate from the confirmed protocol."""


def _is_lora_parameter(name: str) -> bool:
    return (
        name.startswith("lora_A.")
        or name.startswith("lora_B.")
        or ".lora_A." in name
        or ".lora_B." in name
    )


def _logical_numel(parameter: Any) -> int:
    """Return the unpartitioned size of an ordinary or ZeRO-3 parameter."""

    return int(getattr(parameter, "ds_numel", parameter.numel()))


def assert_bf16_base_fp32_lora_parameters(
    model: Any,
    *,
    expected_trainable_parameters: int = EXPECTED_TRAINABLE_PARAMETERS,
) -> dict[str, Any]:
    """Require BF16 frozen base parameters and FP32 trainable LoRA storage.

    The check is valid both before and after ZeRO-3 initialization.  For a
    partitioned parameter, ``Parameter.dtype`` still describes its persistent
    low-precision shard; ``ds_numel`` describes the full logical tensor.
    """

    import torch

    adapter_count = 0
    adapter_tensors = 0
    base_count = 0
    base_tensors = 0
    problems: list[str] = []
    for name, parameter in model.named_parameters():
        logical_numel = _logical_numel(parameter)
        if _is_lora_parameter(name):
            adapter_count += logical_numel
            adapter_tensors += 1
            if not parameter.requires_grad:
                problems.append(f"frozen adapter {name}")
            if parameter.dtype != torch.float32:
                problems.append(f"non-FP32 adapter {name}={parameter.dtype}")
        else:
            base_count += logical_numel
            base_tensors += 1
            if parameter.requires_grad:
                problems.append(f"trainable base {name}")
            if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                problems.append(f"non-BF16 base {name}={parameter.dtype}")

    if adapter_count != expected_trainable_parameters:
        problems.append(
            f"adapter parameters={adapter_count:,}, expected={expected_trainable_parameters:,}"
        )
    if adapter_tensors == 0 or base_tensors == 0:
        problems.append(
            "missing parameter class: "
            f"adapter_tensors={adapter_tensors}, base_tensors={base_tensors}"
        )
    if problems:
        raise TrainingContractError(
            "mixed-precision parameter contract failed: " + "; ".join(problems[:12])
        )

    return {
        "base_dtype": "bfloat16",
        "base_parameters": base_count,
        "base_tensors": base_tensors,
        "base_trainable": False,
        "adapter_dtype": "float32",
        "adapter_parameters": adapter_count,
        "adapter_tensors": adapter_tensors,
        "adapter_trainable": True,
    }


def _deepspeed_runtime_handles(args: TrainingArguments) -> tuple[Any, Any, dict[str, Any]]:
    hf_config = getattr(args, "hf_deepspeed_config", None)
    plugin = getattr(args, "deepspeed_plugin", None)
    if hf_config is None or plugin is None:
        raise TrainingContractError("TrainingArguments has no initialized DeepSpeed config/plugin")
    config = getattr(hf_config, "config", None)
    if not isinstance(config, dict):
        raise TrainingContractError("HfTrainerDeepSpeedConfig has no mutable config dictionary")
    if getattr(plugin, "hf_ds_config", None) is not hf_config:
        raise TrainingContractError("Accelerate and Transformers do not share one DeepSpeed config")
    if getattr(plugin, "deepspeed_config", None) is not config:
        raise TrainingContractError("DeepSpeed plugin config is not the Transformers config object")
    return hf_config, plugin, config


def _assert_zero3_no_offload(config: dict[str, Any]) -> None:
    zero = config.get("zero_optimization")
    if not isinstance(zero, dict) or zero.get("stage") != 3:
        raise TrainingContractError("mixed-precision training requires DeepSpeed ZeRO stage 3")
    for key in ("offload_optimizer", "offload_param"):
        value = zero.get(key)
        if not isinstance(value, dict) or value.get("device") != "none":
            raise TrainingContractError(f"{key} must explicitly use device=none")


def assert_zero3_bf16_load_phase(args: TrainingArguments) -> dict[str, Any]:
    """Audit the temporary native-BF16 phase used only by ZeRO-Init loading."""

    import torch

    hf_config, _, config = _deepspeed_runtime_handles(args)
    _assert_zero3_no_offload(config)
    autocast = config.get("torch_autocast")
    if (
        config.get("bf16", {}).get("enabled") is not True
        or config.get("fp16", {}).get("enabled") is not False
        or not isinstance(autocast, dict)
        or autocast.get("enabled") is not False
        or autocast.get("dtype") != "bfloat16"
        or tuple(autocast.get("lower_precision_safe_modules", ())) != ZERO3_AUTOCAST_SAFE_MODULES
        or args.bf16 is not True
        or args.fp16 is not False
        or args.mixed_precision != "bf16"
        or hf_config.dtype() != torch.bfloat16
    ):
        raise TrainingContractError(
            "DeepSpeed load phase must be native BF16 with torch_autocast disabled"
        )
    return {
        "phase": "zero3_init_load",
        "zero_stage": 3,
        "native_bf16": True,
        "torch_autocast": False,
        "parameter_creation_dtype": "bfloat16",
        "offload": False,
    }


def assert_zero3_fp32_lora_training_phase(args: TrainingArguments) -> dict[str, Any]:
    """Audit the runtime config after switching from loading to training."""

    import torch

    hf_config, _, config = _deepspeed_runtime_handles(args)
    _assert_zero3_no_offload(config)
    autocast = config.get("torch_autocast")
    if (
        config.get("bf16", {}).get("enabled") is not False
        or config.get("fp16", {}).get("enabled") is not False
        or not isinstance(autocast, dict)
        or autocast.get("enabled") is not True
        or autocast.get("dtype") != "bfloat16"
        or tuple(autocast.get("lower_precision_safe_modules", ())) != ZERO3_AUTOCAST_SAFE_MODULES
        or args.bf16 is not False
        or args.fp16 is not False
        or args.mixed_precision != "no"
        or hf_config.dtype() != torch.float32
    ):
        raise TrainingContractError(
            "DeepSpeed training phase must use BF16 torch_autocast with native BF16 disabled"
        )
    return {
        "phase": "zero3_fp32_lora_training",
        "zero_stage": 3,
        "native_bf16": False,
        "torch_autocast": True,
        "autocast_dtype": "bfloat16",
        "autocast_safe_modules": list(ZERO3_AUTOCAST_SAFE_MODULES),
        "optimizer_master_dtype": "float32",
        "offload": False,
    }


def activate_zero3_fp32_lora_training_phase(
    args: TrainingArguments,
    model: Any,
    *,
    expected_trainable_parameters: int = EXPECTED_TRAINABLE_PARAMETERS,
) -> dict[str, Any]:
    """Atomically switch the in-memory DS config after BF16 model loading.

    DeepSpeed's native BF16 engine mode casts *every* floating parameter to
    BF16.  It therefore cannot preserve FP32 LoRA storage.  The pinned
    DeepSpeed 0.19.2 torch-autocast path supports heterogeneous persistent
    parameter dtypes and FP32 ZeRO optimizer masters while executing Linear
    operations in BF16.  The on-disk config remains the immutable load-phase
    input; this explicit, audited mutation is part of the run manifest.
    """

    parameter_report = assert_bf16_base_fp32_lora_parameters(
        model, expected_trainable_parameters=expected_trainable_parameters
    )
    load_report = assert_zero3_bf16_load_phase(args)
    hf_config, _, config = _deepspeed_runtime_handles(args)

    config["bf16"]["enabled"] = False
    config["fp16"]["enabled"] = False
    config["torch_autocast"]["enabled"] = True

    # Accelerator derives the native DeepSpeed mode from this frozen field.
    # Once the model exists, DeepSpeed itself owns the autocast context.
    args.bf16 = False
    args.fp16 = False
    args.mixed_precision = "no"
    if getattr(hf_config, "mismatches", None):
        raise TrainingContractError(
            f"DeepSpeed config had pre-existing mismatches: {hf_config.mismatches!r}"
        )
    hf_config.trainer_config_process(args)
    if getattr(hf_config, "mismatches", None):
        raise TrainingContractError(
            f"DeepSpeed config mismatched after precision switch: {hf_config.mismatches!r}"
        )

    training_report = assert_zero3_fp32_lora_training_phase(args)
    return {
        "load_phase": load_report,
        "training_phase": training_report,
        "parameters": parameter_report,
    }


def assert_accelerate_zero3_precision_state(
    args: TrainingArguments,
    accelerator: Any | None = None,
) -> dict[str, Any]:
    """Prove Accelerate selected and copied the post-switch DS config.

    ``DeepSpeedPlugin.select()`` creates a defensive ``HfDeepSpeedConfig``
    copy and makes that copy Transformers' process-global ZeRO-3 weakref.  A
    stale pre-switch copy would make later model construction or an engine
    rebuild silently re-enable native BF16.  Call this after constructing
    Trainer and before ``trainer.train()``.
    """

    from transformers.integrations.deepspeed import deepspeed_config

    training_report = assert_zero3_fp32_lora_training_phase(args)
    _, plugin, config = _deepspeed_runtime_handles(args)
    selected_config = getattr(plugin, "dschf", None)
    selected_payload = getattr(selected_config, "config", None)
    if not isinstance(selected_payload, dict):
        raise TrainingContractError("Accelerate has not selected the DeepSpeed plugin")
    if selected_payload is config:
        raise TrainingContractError(
            "Accelerate DeepSpeed weakref config was not defensively copied"
        )
    if selected_payload != config:
        raise TrainingContractError("Accelerate selected a stale DeepSpeed config copy")
    if deepspeed_config() is not selected_payload:
        raise TrainingContractError(
            "Transformers global DeepSpeed weakref is not Accelerate's config"
        )

    accelerator_report: dict[str, Any] = {"checked": accelerator is not None}
    if accelerator is not None:
        state = getattr(accelerator, "state", None)
        active_plugin = getattr(state, "deepspeed_plugin", None)
        distributed_type = str(getattr(state, "distributed_type", ""))
        mixed_precision = getattr(state, "mixed_precision", None)
        if active_plugin is not plugin:
            raise TrainingContractError("Accelerator state selected a different DeepSpeed plugin")
        if distributed_type not in {"DistributedType.DEEPSPEED", "DEEPSPEED"}:
            raise TrainingContractError(
                f"Accelerator distributed_type is not DeepSpeed: {distributed_type!r}"
            )
        if mixed_precision != "no":
            raise TrainingContractError(
                f"Accelerator native mixed precision must be disabled, got {mixed_precision!r}"
            )
        accelerator_report = {
            "checked": True,
            "distributed_type": "DEEPSPEED",
            "mixed_precision": "no",
            "active_plugin_identity": True,
        }

    return {
        "training_phase": training_report,
        "plugin_selected": True,
        "selected_config_is_copy": True,
        "transformers_weakref_identity": True,
        "accelerator": accelerator_report,
    }


def assert_zero3_engine_precision_contract(
    engine: Any,
    *,
    expected_trainable_parameters: int = EXPECTED_TRAINABLE_PARAMETERS,
    require_optimizer_states: bool = False,
) -> dict[str, Any]:
    """Verify real ZeRO-3 shards, masters, communication, and Adam states."""

    import torch
    from deepspeed.runtime.torch_autocast import get_comm_dtype

    required_methods = (
        "bfloat16_enabled",
        "fp16_enabled",
        "torch_autocast_enabled",
        "torch_autocast_dtype",
        "zero_optimization_stage",
    )
    missing_methods = [
        name for name in required_methods if not callable(getattr(engine, name, None))
    ]
    if missing_methods:
        raise TrainingContractError(f"not a DeepSpeed engine; missing {missing_methods!r}")
    if (
        engine.zero_optimization_stage() != 3
        or engine.bfloat16_enabled()
        or engine.fp16_enabled()
        or not engine.torch_autocast_enabled()
        or engine.torch_autocast_dtype() != torch.bfloat16
    ):
        raise TrainingContractError("initialized DeepSpeed engine has the wrong precision mode")

    module = getattr(engine, "module", None)
    optimizer = getattr(engine, "optimizer", None)
    if module is None or optimizer is None:
        raise TrainingContractError("DeepSpeed engine has no module or ZeRO optimizer")
    parameter_report = assert_bf16_base_fp32_lora_parameters(
        module, expected_trainable_parameters=expected_trainable_parameters
    )

    storage_problems: list[str] = []
    adapter_comm_dtypes: set[str] = set()
    for name, parameter in module.named_parameters():
        shard = getattr(parameter, "ds_tensor", None)
        if not hasattr(parameter, "ds_id") or shard is None:
            storage_problems.append(f"unpartitioned parameter {name}")
            continue
        expected_dtype = torch.float32 if _is_lora_parameter(name) else torch.bfloat16
        if parameter.is_floating_point() and shard.dtype != expected_dtype:
            storage_problems.append(f"wrong shard dtype {name}={shard.dtype}")
        if _is_lora_parameter(name):
            adapter_comm_dtypes.add(str(get_comm_dtype(parameter)))
    if adapter_comm_dtypes != {str(torch.float32)}:
        storage_problems.append(f"LoRA communication dtypes={sorted(adapter_comm_dtypes)!r}")

    low_precision_groups = tuple(getattr(optimizer, "fp16_partitioned_groups_flat", ()))
    master_groups = tuple(getattr(optimizer, "fp32_partitioned_groups_flat", ()))
    if not low_precision_groups or any(
        group.dtype != torch.float32 for group in low_precision_groups
    ):
        storage_problems.append("ZeRO LoRA partitions are not all FP32")
    if not master_groups or any(group.dtype != torch.float32 for group in master_groups):
        storage_problems.append("ZeRO optimizer master partitions are not all FP32")
    if getattr(optimizer, "dtype", None) != torch.float32:
        storage_problems.append(f"ZeRO optimizer dtype={getattr(optimizer, 'dtype', None)}")
    if getattr(optimizer, "master_weights_and_grads_dtype", None) != torch.float32:
        storage_problems.append(
            "ZeRO master_weights_and_grads_dtype="
            f"{getattr(optimizer, 'master_weights_and_grads_dtype', None)}"
        )

    inner_optimizer = getattr(optimizer, "optimizer", None)
    state_tensors = 0
    wrong_state_dtypes: list[str] = []
    if inner_optimizer is not None:
        for state in inner_optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    state_tensors += 1
                    if value.dtype != torch.float32:
                        wrong_state_dtypes.append(f"{key}={value.dtype}")
    if wrong_state_dtypes:
        storage_problems.append(f"non-FP32 optimizer states={wrong_state_dtypes[:8]!r}")
    if require_optimizer_states and state_tensors == 0:
        storage_problems.append("AdamW state tensors have not been initialized by an update")
    if storage_problems:
        raise TrainingContractError(
            "ZeRO-3 storage contract failed: " + "; ".join(storage_problems[:12])
        )

    return {
        **parameter_report,
        "zero_stage": 3,
        "native_bf16": False,
        "torch_autocast_dtype": "bfloat16",
        "adapter_communication_dtype": "float32",
        "adapter_shard_dtype": "float32",
        "master_dtype": "float32",
        "optimizer_state_tensors": state_tensors,
        "optimizer_state_dtype": "float32" if state_tensors else "not_initialized",
    }


def build_training_argument_kwargs(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    micro_batch_size_per_gpu: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> dict[str, Any]:
    """Return auditable kwargs for Transformers 5.9 ``TrainingArguments``.

    The optional batch geometry exists solely for the confirmed 2x4090 smoke
    gate: default is ``1 x 2 x 8``; it may switch to ``2 x 2 x 4`` only after
    that gate passes.  Both paths retain global batch size 16.
    """

    train = config.train
    data = config.data
    runtime = config.runtime
    micro = (
        train.micro_batch_size_per_gpu
        if micro_batch_size_per_gpu is None
        else micro_batch_size_per_gpu
    )
    accumulation = (
        train.gradient_accumulation_steps
        if gradient_accumulation_steps is None
        else gradient_accumulation_steps
    )
    if isinstance(micro, bool) or not isinstance(micro, int) or micro <= 0:
        raise TrainingContractError("micro batch size must be a positive integer")
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
        raise TrainingContractError("gradient accumulation must be a positive integer")
    global_batch = micro * runtime.required_gpu_count * accumulation
    if runtime.required_gpu_count != 2:
        raise TrainingContractError("the frozen run requires exactly two distributed GPU ranks")
    if global_batch != train.global_batch_size:
        raise TrainingContractError(
            f"batch geometry gives global batch {global_batch}, expected {train.global_batch_size}"
        )
    allowed_geometry = {
        (train.micro_batch_size_per_gpu, train.gradient_accumulation_steps),
        (
            train.candidate_micro_batch_size_per_gpu,
            train.candidate_gradient_accumulation_steps,
        ),
    }
    if (micro, accumulation) not in allowed_geometry:
        raise TrainingContractError(
            f"unconfirmed batch geometry {(micro, accumulation)!r}; "
            f"allowed {sorted(allowed_geometry)!r}"
        )
    if config.seed != SEED:
        raise TrainingContractError(f"seed must remain frozen at {SEED}")
    if not train.gradient_checkpointing or not train.gradient_checkpointing_use_reentrant:
        raise TrainingContractError(
            "ZeRO-3 training requires reentrant activation checkpointing"
        )
    if not train.average_tokens_across_devices:
        raise TrainingContractError("global supervised-token normalization must remain enabled")

    output_path = Path(output_dir)
    deepspeed_path = Path(train.deepspeed_config)
    if not deepspeed_path.is_file():
        raise TrainingContractError(f"DeepSpeed config does not exist: {deepspeed_path}")

    return {
        "output_dir": str(output_path),
        "run_name": config.experiment,
        "do_train": True,
        # Formal validation (including eval_loss) is run once for each sealed
        # epoch checkpoint after training. Running Trainer's implicit epoch
        # evaluation here would duplicate that work and, more importantly,
        # place RNG-consuming dataloader iteration after the durable checkpoint.
        "do_eval": False,
        "num_train_epochs": float(train.epochs),
        "per_device_train_batch_size": micro,
        "per_device_eval_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "learning_rate": train.learning_rate,
        "lr_scheduler_type": train.scheduler,
        # Transformers 5.9 accepts a float in [0,1) here as a ratio.  Its old
        # warmup_ratio alias is already deprecated in the locked version.
        "warmup_steps": train.warmup_ratio,
        "optim": "adamw_torch",
        "adam_beta1": train.betas[0],
        "adam_beta2": train.betas[1],
        "adam_epsilon": train.epsilon,
        "weight_decay": train.weight_decay,
        "max_grad_norm": train.max_grad_norm,
        "bf16": True,
        "fp16": False,
        "tf32": False,
        "gradient_checkpointing": train.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": train.gradient_checkpointing_use_reentrant
        },
        "average_tokens_across_devices": True,
        "eval_strategy": "no",
        "prediction_loss_only": True,
        "save_strategy": "steps",
        "save_steps": train.checkpoint_steps,
        "save_total_limit": train.keep_local_checkpoints,
        "save_only_model": False,
        "logging_strategy": "steps",
        "logging_steps": train.log_steps,
        "logging_dir": str(output_path / "tensorboard"),
        "report_to": ["tensorboard"],
        "load_best_model_at_end": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "full_determinism": False,
        "dataloader_num_workers": data.dataloader_workers,
        "dataloader_pin_memory": data.pin_memory,
        "dataloader_persistent_workers": False,
        "dataloader_drop_last": False,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "ignore_data_skip": False,
        "restore_callback_states_from_checkpoint": False,
        "ddp_backend": "nccl",
        "ddp_timeout": 600,
        "ddp_find_unused_parameters": False,
        "deepspeed": str(deepspeed_path),
    }


def build_training_arguments(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    micro_batch_size_per_gpu: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> TrainingArguments:
    """Instantiate the pinned Transformers training arguments."""

    kwargs = build_training_argument_kwargs(
        config,
        output_dir=output_dir,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return TrainingArguments(**kwargs)


def logical_epoch_from_trainer_state(epoch: float | None) -> int:
    """Map a fresh or resumed fractional Trainer epoch to its data epoch."""

    if epoch is None:
        return 0
    value = float(epoch)
    if not math.isfinite(value) or value < 0:
        raise TrainingContractError(f"invalid Trainer epoch state: {epoch!r}")
    # Floating point state can contain 0.9999999999999999 at an exact boundary.
    return int(math.floor(value + 1e-12))


class DatasetEpochCallback(TrainerCallback):
    """Set online serialization epoch from persisted ``TrainerState.epoch``.

    The callback intentionally has no independent checkpoint state.  Deriving
    the value from Trainer's restored state prevents two sources of truth.
    """

    def __init__(self, train_dataset: EpochAwareTokenizedDataset) -> None:
        if not train_dataset.training:
            raise TrainingContractError("epoch callback requires a training dataset")
        self.train_dataset = train_dataset

    def _restore_epoch(self, state: Any) -> None:
        self.train_dataset.set_epoch(logical_epoch_from_trainer_state(state.epoch))

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._restore_epoch(state)
        return control

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._restore_epoch(state)
        return control


class TokenNormalizedTrainer(Trainer):
    """Trainer that refuses to fall back from global token normalization.

    Transformers 5.9 counts ``labels != -100`` across all accumulated
    micro-batches and, with ``average_tokens_across_devices=True``, all-reduces
    that count across ranks.  Qwen's native causal-LM loss then uses the global
    count as the denominator.  This guard makes that behavior contractual.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.args.average_tokens_across_devices:
            raise TrainingContractError("Trainer must set average_tokens_across_devices=True")
        if not self.model_accepts_loss_kwargs:
            raise TrainingContractError(
                "wrapped model does not accept num_items_in_batch; global token-normalized "
                "loss would be unavailable"
            )


def build_epoch_callback(
    train_dataset: EpochAwareTokenizedDataset,
) -> DatasetEpochCallback:
    """Small integration helper for the CLI's Trainer callback list."""

    return DatasetEpochCallback(train_dataset)
