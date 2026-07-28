from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from test_tokenization import FakeTokenizer, reaction_record
from torch import nn

from janus_ts.config import load_config
from janus_ts.dtype_gate import (
    build_tiny_zero3_dtype_gate_config,
    measure_trainable_zero3_shard_updates,
    snapshot_trainable_zero3_shards,
)
from janus_ts.tokenization import EpochAwareTokenizedDataset, QwenTrainingEncoder
from janus_ts.training import (
    DatasetEpochCallback,
    TrainingContractError,
    activate_zero3_fp32_lora_training_phase,
    assert_accelerate_zero3_precision_state,
    assert_bf16_base_fp32_lora_parameters,
    assert_zero3_bf16_load_phase,
    assert_zero3_engine_precision_contract,
    assert_zero3_fp32_lora_training_phase,
    build_training_argument_kwargs,
    build_training_arguments,
    logical_epoch_from_trainer_state,
)


class TinyMixedDtypeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 1, bias=False, dtype=torch.float32)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False, dtype=torch.float32)})
        self.base.requires_grad_(False)


class FakeZero3Optimizer:
    def __init__(self, model: nn.Module, *, state_initialized: bool) -> None:
        adapter = [
            parameter
            for name, parameter in model.named_parameters()
            if "lora_A." in name or "lora_B." in name
        ]
        flat = torch.cat([parameter.detach().flatten() for parameter in adapter])
        self.dtype = torch.float32
        self.master_weights_and_grads_dtype = torch.float32
        self.fp16_partitioned_groups_flat = [flat]
        self.fp32_partitioned_groups_flat = [flat.clone()]
        state = {}
        if state_initialized:
            state[self.fp32_partitioned_groups_flat[0]] = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros_like(flat),
                "exp_avg_sq": torch.zeros_like(flat),
            }
        self.optimizer = SimpleNamespace(state=state)


class FakeZero3Engine:
    def __init__(self, model: nn.Module, *, state_initialized: bool = True) -> None:
        self.module = model
        for index, (name, parameter) in enumerate(model.named_parameters()):
            parameter.ds_id = index
            parameter.ds_numel = parameter.numel()
            parameter.ds_tensor = parameter.detach().clone()
            if "lora_A." in name or "lora_B." in name:
                parameter.comm_dtype = torch.bfloat16
        self.optimizer = FakeZero3Optimizer(model, state_initialized=state_initialized)

    @staticmethod
    def zero_optimization_stage():
        return 3

    @staticmethod
    def bfloat16_enabled():
        return False

    @staticmethod
    def fp16_enabled():
        return False

    @staticmethod
    def torch_autocast_enabled():
        return True

    @staticmethod
    def torch_autocast_dtype():
        return torch.bfloat16


def test_frozen_training_argument_contract():
    config = load_config("configs/transition1x.yaml")
    kwargs = build_training_argument_kwargs(config, output_dir="artifacts/runs/test")

    assert kwargs["num_train_epochs"] == 5.0
    assert kwargs["per_device_train_batch_size"] == 1
    assert kwargs["gradient_accumulation_steps"] == 8
    assert kwargs["learning_rate"] == 1e-4
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert kwargs["warmup_steps"] == 0.05
    assert kwargs["optim"] == "adamw_torch"
    assert kwargs["bf16"] is True and kwargs["fp16"] is False
    assert kwargs["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert kwargs["average_tokens_across_devices"] is True
    assert kwargs["save_strategy"] == "steps" and kwargs["save_steps"] == 50
    assert kwargs["save_total_limit"] == 2
    assert kwargs["eval_strategy"] == "no"
    assert kwargs["do_eval"] is False
    assert kwargs["ddp_timeout"] == 600
    assert kwargs["prediction_loss_only"] is True
    assert kwargs["report_to"] == ["tensorboard"]
    assert kwargs["seed"] == kwargs["data_seed"] == 42
    assert kwargs["ddp_backend"] == "nccl"
    assert kwargs["deepspeed"] == "configs/deepspeed_zero3.json"

    candidate = build_training_argument_kwargs(
        config,
        output_dir="artifacts/runs/test",
        micro_batch_size_per_gpu=2,
        gradient_accumulation_steps=4,
    )
    assert candidate["per_device_train_batch_size"] == 2
    assert candidate["gradient_accumulation_steps"] == 4

    with pytest.raises(TrainingContractError, match="unconfirmed batch geometry"):
        build_training_argument_kwargs(
            config,
            output_dir="artifacts/runs/test",
            micro_batch_size_per_gpu=4,
            gradient_accumulation_steps=2,
        )


def test_training_arguments_instantiate_against_locked_transformers():
    config = load_config("configs/transition1x.yaml")
    arguments = build_training_arguments(config, output_dir="artifacts/runs/test")

    assert arguments.average_tokens_across_devices is True
    assert str(arguments.eval_strategy) == "IntervalStrategy.NO"
    assert str(arguments.save_strategy) == "SaveStrategy.STEPS"
    assert str(arguments.optim) == "OptimizerNames.ADAMW_TORCH"


def test_resume_epoch_is_derived_from_trainer_state():
    assert logical_epoch_from_trainer_state(None) == 0
    assert logical_epoch_from_trainer_state(0.75) == 0
    assert logical_epoch_from_trainer_state(2.375) == 2
    assert logical_epoch_from_trainer_state(0.9999999999999999) == 1
    with pytest.raises(TrainingContractError):
        logical_epoch_from_trainer_state(float("nan"))


def test_epoch_callback_restores_dataset_epoch():
    dataset = EpochAwareTokenizedDataset(
        [reaction_record()], QwenTrainingEncoder(FakeTokenizer()), training=True
    )
    callback = DatasetEpochCallback(dataset)
    control = object()

    assert callback.on_train_begin(None, SimpleNamespace(epoch=3.25), control) is control
    assert dataset.epoch == 3
    assert callback.on_epoch_begin(None, SimpleNamespace(epoch=4.0), control) is control
    assert dataset.epoch == 4


def test_epoch_callback_rejects_eval_dataset():
    dataset = EpochAwareTokenizedDataset(
        [reaction_record()], QwenTrainingEncoder(FakeTokenizer()), training=False
    )
    with pytest.raises(TrainingContractError, match="training dataset"):
        DatasetEpochCallback(dataset)


def test_two_phase_deepspeed_precision_switch_preserves_fp32_lora():
    config = load_config("configs/transition1x.yaml")
    arguments = build_training_arguments(config, output_dir="artifacts/runs/test")
    model = TinyMixedDtypeModel()

    load_report = assert_zero3_bf16_load_phase(arguments)
    parameter_report = assert_bf16_base_fp32_lora_parameters(model, expected_trainable_parameters=5)
    switch_report = activate_zero3_fp32_lora_training_phase(
        arguments, model, expected_trainable_parameters=5
    )
    runtime_report = assert_zero3_fp32_lora_training_phase(arguments)

    assert load_report["native_bf16"] is True
    assert parameter_report["adapter_dtype"] == "float32"
    assert switch_report["training_phase"] == runtime_report
    assert arguments.bf16 is False
    assert arguments.mixed_precision == "no"
    assert arguments.hf_deepspeed_config.config["bf16"]["enabled"] is False
    assert arguments.hf_deepspeed_config.config["torch_autocast"]["enabled"] is True


def test_zero3_engine_precision_gate_checks_shards_masters_and_states():
    model = TinyMixedDtypeModel()
    report = assert_zero3_engine_precision_contract(
        FakeZero3Engine(model),
        expected_trainable_parameters=5,
        require_optimizer_states=True,
    )

    assert report["adapter_shard_dtype"] == "float32"
    assert report["master_dtype"] == "float32"
    assert report["adapter_communication_dtype"] == "bfloat16"
    assert report["optimizer_state_dtype"] == "float32"


def test_zero3_engine_precision_gate_rejects_missing_adam_state():
    model = TinyMixedDtypeModel()
    with pytest.raises(TrainingContractError, match="state tensors"):
        assert_zero3_engine_precision_contract(
            FakeZero3Engine(model, state_initialized=False),
            expected_trainable_parameters=5,
            require_optimizer_states=True,
        )


def test_two_rank_dtype_gate_config_uses_runtime_autocast_contract():
    gate = build_tiny_zero3_dtype_gate_config()

    assert gate["bf16"]["enabled"] is False
    assert gate["fp16"]["enabled"] is False
    assert gate["torch_autocast"] == {
        "enabled": True,
        "dtype": "bfloat16",
        "lower_precision_safe_modules": ["torch.nn.modules.linear.Linear"],
    }
    assert gate["zero_optimization"]["stage"] == 3
    assert gate["zero_optimization"]["offload_optimizer"]["device"] == "none"
    assert gate["zero_optimization"]["offload_param"]["device"] == "none"


def test_dtype_gate_tracks_top_level_trainable_zero3_shards_without_name_matching():
    model = TinyMixedDtypeModel()
    for parameter in model.parameters():
        parameter.ds_tensor = parameter.detach().clone()

    before = snapshot_trainable_zero3_shards(model)
    assert set(before) == {"lora_A.default.weight", "lora_B.default.weight"}

    model.lora_B["default"].weight.ds_tensor[0, 0].add_(0.125)
    changed_names, max_abs_delta = measure_trainable_zero3_shard_updates(model, before)

    assert changed_names == ["lora_B.default.weight"]
    assert max_abs_delta == pytest.approx(0.125)


def test_dtype_gate_rejects_an_empty_trainable_zero3_snapshot():
    model = TinyMixedDtypeModel()
    model.requires_grad_(False)

    with pytest.raises(RuntimeError, match="no trainable ZeRO shards"):
        snapshot_trainable_zero3_shards(model)


def test_dtype_gate_requires_the_same_trainable_shards_after_step():
    model = TinyMixedDtypeModel()
    for parameter in model.parameters():
        parameter.ds_tensor = parameter.detach().clone()
    before = snapshot_trainable_zero3_shards(model)
    model.lora_B["default"].weight.requires_grad_(False)

    with pytest.raises(RuntimeError, match="shard names changed"):
        measure_trainable_zero3_shard_updates(model, before)


def test_accelerate_plugin_selects_post_switch_config_without_gpu():
    from transformers.integrations.deepspeed import unset_hf_deepspeed_config

    config = load_config("configs/transition1x.yaml")
    arguments = build_training_arguments(config, output_dir="artifacts/runs/test")
    activate_zero3_fp32_lora_training_phase(
        arguments, TinyMixedDtypeModel(), expected_trainable_parameters=5
    )
    plugin = arguments.deepspeed_plugin
    try:
        # These are the two real Accelerate calls made during AcceleratorState
        # initialization for a selected DeepSpeed plugin.
        plugin.set_mixed_precision(arguments.mixed_precision)
        plugin.select(_from_accelerator_state=True)
        report = assert_accelerate_zero3_precision_state(arguments)
    finally:
        unset_hf_deepspeed_config()

    assert report["plugin_selected"] is True
    assert report["selected_config_is_copy"] is True
    assert report["transformers_weakref_identity"] is True
