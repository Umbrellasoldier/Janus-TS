from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import janus_ts.distributed as distributed
from janus_ts.config import load_config
from janus_ts.constants import MAX_SEQUENCE_LENGTH, QWEN_IM_END_TOKEN_ID
from janus_ts.distributed import (
    SMOKE_OOM_EXIT_CODE,
    DistributedTrainingError,
    MemorySmokeOutcome,
    SyntheticSequenceDataset,
    assert_model_precision_and_freezing,
    assert_zero3_no_offload,
    build_memory_smoke_arguments,
    configure_reproducibility,
    preflight_torchrun_environment,
)


def _deepspeed_arguments(config_path: Path, **overrides):
    values = {
        "deepspeed": str(config_path),
        "deepspeed_plugin": SimpleNamespace(
            zero_stage=3,
            offload_optimizer_device="none",
            offload_param_device="none",
            zero3_init_flag=True,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_zero3(path: Path, *, offload_param: str = "none") -> None:
    path.write_text(
        json.dumps(
            {
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {"device": "none"},
                    "offload_param": {"device": offload_param},
                }
            }
        ),
        encoding="utf-8",
    )


def test_torchrun_preflight_is_exactly_two_ranks():
    first = preflight_torchrun_environment({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"})
    second = preflight_torchrun_environment({"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"})
    assert (first.rank, first.local_rank, first.world_size) == (0, 0, 2)
    assert (second.rank, second.local_rank, second.world_size) == (1, 1, 2)

    with pytest.raises(DistributedTrainingError, match="torchrun"):
        preflight_torchrun_environment({})
    with pytest.raises(DistributedTrainingError, match="WORLD_SIZE=2"):
        preflight_torchrun_environment({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"})


def test_zero3_contract_requires_early_activation_and_no_offload(tmp_path):
    config_path = tmp_path / "ds.json"
    _write_zero3(config_path)
    arguments = _deepspeed_arguments(config_path)
    assert_zero3_no_offload(arguments, zero3_enabled=lambda: True)

    with pytest.raises(DistributedTrainingError, match="not active"):
        assert_zero3_no_offload(arguments, zero3_enabled=lambda: False)

    _write_zero3(config_path, offload_param="cpu")
    with pytest.raises(DistributedTrainingError, match="offload_param"):
        assert_zero3_no_offload(arguments, zero3_enabled=lambda: True)

    _write_zero3(config_path)
    bad_plugin = _deepspeed_arguments(
        config_path,
        deepspeed_plugin=SimpleNamespace(
            zero_stage=2,
            offload_optimizer_device="cpu",
            offload_param_device="none",
            zero3_init_flag=False,
        ),
    )
    with pytest.raises(DistributedTrainingError, match="zero_stage=2"):
        assert_zero3_no_offload(bad_plugin, zero3_enabled=lambda: True)


def test_zero3_contract_accepts_in_memory_deepspeed_config() -> None:
    arguments = _deepspeed_arguments(
        Path("unused.json"),
        deepspeed={
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "none"},
                "offload_param": {"device": "none"},
            }
        },
    )

    assert_zero3_no_offload(arguments, zero3_enabled=lambda: True)


def test_worst_case_synthetic_examples_are_exactly_2048_tokens():
    dataset = SyntheticSequenceDataset(3)
    feature = dataset[0]

    assert len(dataset) == 3
    assert len(feature["input_ids"]) == MAX_SEQUENCE_LENGTH == 2048
    assert feature["input_ids"][-1] == QWEN_IM_END_TOKEN_ID
    assert feature["labels"] == feature["input_ids"]
    assert set(feature["attention_mask"]) == {1}
    feature["input_ids"][0] = -1
    assert dataset[0]["input_ids"][0] != -1

    with pytest.raises(DistributedTrainingError, match="frozen"):
        SyntheticSequenceDataset(1, sequence_length=1024)


def test_memory_smoke_training_arguments_are_one_update_and_preserve_geometry():
    config = load_config("configs/transition1x.yaml")
    default = build_memory_smoke_arguments(
        config,
        output_dir="artifacts/test-smoke-default",
        micro_batch_size_per_gpu=1,
    )
    assert default.max_steps == 1
    assert default.per_device_train_batch_size == 1
    assert default.gradient_accumulation_steps == 8
    assert str(default.eval_strategy).endswith("NO")
    assert str(default.save_strategy).endswith("NO")

    candidate = build_memory_smoke_arguments(
        config,
        output_dir="artifacts/test-smoke-candidate",
        micro_batch_size_per_gpu=2,
    )
    assert candidate.max_steps == 1
    assert candidate.per_device_train_batch_size == 2
    assert candidate.gradient_accumulation_steps == 4

    with pytest.raises(DistributedTrainingError, match="default or candidate"):
        build_memory_smoke_arguments(
            config,
            output_dir="artifacts/test-smoke-bad",
            micro_batch_size_per_gpu=4,
        )


class _Parameter:
    def __init__(self, numel: int, dtype: torch.dtype, requires_grad: bool):
        self._numel = numel
        self.dtype = dtype
        self.requires_grad = requires_grad

    def numel(self):
        return self._numel


class _TinyContractModel:
    def __init__(self, adapter_count: int = 12):
        self.parameters = [
            ("base_model.model.embed_tokens.weight", _Parameter(100, torch.bfloat16, False)),
            (
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
                _Parameter(adapter_count // 2, torch.bfloat16, True),
            ),
            (
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
                _Parameter(adapter_count - adapter_count // 2, torch.bfloat16, True),
            ),
        ]

    def named_parameters(self):
        return iter(self.parameters)


def test_model_precision_contract_uses_logical_adapter_count():
    model = _TinyContractModel(adapter_count=12)
    # Simulate a partitioned parameter whose local shard numel differs.
    model.parameters[1][1].ds_numel = 7
    model.parameters[2][1].ds_numel = 5
    assert_model_precision_and_freezing(
        model,
        torch,
        expected_trainable_parameters=12,
    )

    model.parameters[0][1].dtype = torch.float32
    with pytest.raises(DistributedTrainingError, match="non-BF16 base"):
        assert_model_precision_and_freezing(
            model,
            torch,
            expected_trainable_parameters=12,
        )


def test_reproducibility_policy_is_seed_42_and_non_strict(monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("manual_seed", seed)),
        use_deterministic_algorithms=lambda value: calls.append(("deterministic", value)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("cuda_seed", seed)),
        ),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(benchmark=True, deterministic=True, allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        ),
    )
    configure_reproducibility(fake_torch)

    assert ("manual_seed", 42) in calls
    assert ("cuda_seed", 42) in calls
    assert ("deterministic", False) in calls
    assert fake_torch.backends.cudnn.benchmark is False
    assert fake_torch.backends.cudnn.deterministic is False
    assert fake_torch.backends.cudnn.allow_tf32 is False
    assert fake_torch.backends.cuda.matmul.allow_tf32 is False
    with pytest.raises(DistributedTrainingError, match="frozen seed"):
        configure_reproducibility(fake_torch, seed=7)


def test_clean_oom_outcome_has_dedicated_exit_code():
    passing = MemorySmokeOutcome("pass", {"status": "pass"})
    oom = MemorySmokeOutcome("oom", {"status": "oom"})
    rejected = MemorySmokeOutcome("rejected", {"status": "rejected"})
    assert passing.exit_code == 0
    assert oom.exit_code == SMOKE_OOM_EXIT_CODE == 42
    assert rejected.exit_code == SMOKE_OOM_EXIT_CODE == 42


def test_post_engine_callback_checks_engine_then_initialized_optimizer(monkeypatch):
    calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(use_cache=False),
        _janus_ts_input_grads_enabled=True,
        _janus_ts_reentrant_gc_enabled=True,
    )
    engine = SimpleNamespace(module=model)
    monkeypatch.setattr(
        distributed,
        "assert_zero3_engine_precision_contract",
        lambda received, *, require_optimizer_states: calls.append(
            (received, require_optimizer_states)
        ),
    )
    callback = distributed.PostEngineModelContractCallback()
    callback.bind_engine_getter(lambda: engine)
    control = object()

    assert callback.on_train_begin(None, SimpleNamespace(global_step=0), control) is control
    assert callback.on_step_end(None, SimpleNamespace(global_step=1), control) is control
    assert callback.on_step_end(None, SimpleNamespace(global_step=2), control) is control
    assert calls == [(engine, False), (engine, True)]

    with pytest.raises(DistributedTrainingError, match="bound twice"):
        callback.bind_engine_getter(lambda: engine)


def test_zero3_model_loader_never_requests_a_device_map_or_offload(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "residual_base").mkdir(parents=True)
    (bundle / "pissa_init").mkdir()
    calls = {}
    config = SimpleNamespace(use_cache=True, pad_token_id=None, eos_token_id=None)
    base = SimpleNamespace(config=config, hf_device_map=None)

    def base_loader(**kwargs):
        calls["base"] = kwargs
        return base, {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }

    class FakeModel:
        hf_device_map = None

        def __init__(self):
            self.config = config
            self.generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=None)
            self.events = []

        def named_parameters(self):
            return iter(())

        def enable_input_require_grads(self):
            self.events.append("input-grads")

        def gradient_checkpointing_enable(self, **kwargs):
            self.events.append(("checkpointing", kwargs))

        def train(self):
            self.events.append("train")

    model = FakeModel()

    def adapter_loader(base_model, adapter_path, **kwargs):
        calls["adapter"] = (base_model, adapter_path, kwargs)
        return model

    monkeypatch.setattr(distributed, "assert_zero3_no_offload", lambda arguments: None)
    monkeypatch.setattr(distributed, "validate_qwen_text_config", lambda value: None)
    monkeypatch.setattr(
        distributed, "validate_pissa_initialization_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        distributed,
        "assert_pissa_adapter_contract",
        lambda value, *, adapter_dtype: (
            value is model and adapter_dtype == torch.bfloat16
        )
        or pytest.fail("training loader must validate the BF16 adapter dtype"),
    )
    monkeypatch.setattr(
        distributed, "assert_model_precision_and_freezing", lambda value, torch_module: None
    )
    monkeypatch.setattr(
        distributed,
        "assert_zero3_bf16_precision",
        lambda arguments, value: {"phase": "training"},
    )

    with pytest.raises(DistributedTrainingError, match="reentrant"):
        distributed.load_zero3_prepared_model(
            bundle,
            SimpleNamespace(gradient_checkpointing_kwargs={"use_reentrant": False}),
            torch_module=torch,
            base_loader=lambda **kwargs: pytest.fail("base load must not start"),
            adapter_loader=adapter_loader,
            require_complete=False,
        )
    assert calls == {}

    loaded = distributed.load_zero3_prepared_model(
        bundle,
        SimpleNamespace(gradient_checkpointing_kwargs={"use_reentrant": True}),
        torch_module=torch,
        base_loader=base_loader,
        adapter_loader=adapter_loader,
        require_complete=False,
    )

    assert loaded is model
    assert calls["base"]["dtype"] == torch.bfloat16
    assert calls["base"]["low_cpu_mem_usage"] is False
    assert "device_map" not in calls["base"]
    assert not any("offload" in name for name in calls["base"])
    assert calls["adapter"][2] == {
        "is_trainable": True,
        "autocast_adapter_dtype": False,
        "low_cpu_mem_usage": False,
    }
    assert model.events == [
        "input-grads",
        ("checkpointing", {"gradient_checkpointing_kwargs": {"use_reentrant": True}}),
        "train",
    ]
    assert model.config.use_cache is False
    assert model._janus_ts_precision_report == {"phase": "training"}


def test_full_builder_constructs_arguments_before_model_and_exposes_callback_hook(
    monkeypatch,
):
    config = load_config("configs/transition1x.yaml")
    events = []
    context = distributed.TorchrunContext(0, 0, 2, 0)
    arguments = SimpleNamespace()
    model = object()
    tokenizer = object()

    monkeypatch.setattr(distributed, "install_frozen_environment", lambda: events.append("env"))
    monkeypatch.setattr(distributed, "preflight_torchrun_environment", lambda: context)

    def arguments_builder(*args, **kwargs):
        events.append("arguments")
        return arguments

    monkeypatch.setattr(distributed, "build_training_arguments", arguments_builder)
    monkeypatch.setattr(distributed, "configure_reproducibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed, "assert_initialized_two_rank_job", lambda *args: None)
    monkeypatch.setattr(distributed, "assert_accelerate_zero3_precision_state", lambda *args: None)

    def model_loader(bundle, received_arguments):
        assert received_arguments is arguments
        assert events == ["env", "arguments"]
        events.append("model")
        return model

    monkeypatch.setattr(distributed, "load_pinned_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(
        distributed,
        "load_processed_dataset",
        lambda path: {"train": ["train-row"], "val": ["val-row"]},
    )

    class FakeEncoder:
        def __init__(self, received_tokenizer, **kwargs):
            assert received_tokenizer is tokenizer

    class FakeDataset:
        def __init__(self, records, encoder, *, training):
            self.records = records
            self.training = training

    class FakeCollator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    epoch_callback = object()
    monkeypatch.setattr(distributed, "QwenTrainingEncoder", FakeEncoder)
    monkeypatch.setattr(distributed, "EpochAwareTokenizedDataset", FakeDataset)
    monkeypatch.setattr(distributed, "CausalLMCollator", FakeCollator)
    monkeypatch.setattr(distributed, "build_epoch_callback", lambda dataset: epoch_callback)

    late_callback = object()

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model_wrapped = "engine"
            self.accelerator = "accelerator"
            self.added = []

        def add_callback(self, callback):
            self.added.append(callback)

    def callback_factory(engine_getter):
        assert engine_getter() == "engine"
        return late_callback

    run = distributed.build_full_distributed_run(
        config,
        bundle_dir="bundle",
        processed_path="processed",
        output_dir="output",
        callback_factories=[callback_factory],
        model_loader=model_loader,
        trainer_class=FakeTrainer,
    )

    assert events == ["env", "arguments", "model"]
    assert run.model is model
    assert isinstance(
        run.trainer.kwargs["callbacks"][0], distributed.PostEngineModelContractCallback
    )
    assert run.trainer.kwargs["callbacks"][1] is epoch_callback
    assert run.trainer.added == [late_callback]
    assert run.train_dataset.training is True
    assert run.eval_dataset.training is False


def test_managed_trainer_restores_fresh_engine_and_suppresses_native_io(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "resume-step-000000050"
    checkpoint.mkdir()
    calls = []

    def parent_prepare(self, max_steps, dataloader, resume):
        calls.append(("prepare", resume))
        self.model_wrapped = "fresh-engine"
        return "model", dataloader

    monkeypatch.setattr(
        distributed.TokenNormalizedTrainer,
        "_prepare_for_training",
        parent_prepare,
    )
    monkeypatch.setattr(
        distributed.TokenNormalizedTrainer,
        "_save_checkpoint",
        lambda *args: calls.append(("native-save", None)),
    )
    monkeypatch.setattr(
        distributed.TokenNormalizedTrainer,
        "_load_rng_state",
        lambda *args: calls.append(("native-rng", None)),
    )
    trainer = object.__new__(distributed.JanusManagedTrainer)
    trainer.janus_managed_checkpoints = True

    def restore(received, path):
        calls.append(("restore", received.model_wrapped, path))

    restore.restore_rng_state = lambda: calls.append(("restore-rng", None))
    trainer.janus_restore_hook = restore
    trainer.janus_restore_result = None
    trainer._janus_active_resume_path = None

    result = trainer._prepare_for_training(100, "loader", str(checkpoint))
    trainer._load_rng_state(str(checkpoint))
    trainer._save_checkpoint("model", None)

    assert result == ("model", "loader")
    assert calls == [
        ("prepare", None),
        ("restore", "fresh-engine", checkpoint.resolve()),
        ("restore-rng", None),
    ]


def test_checkpoint_manager_restore_hook_cross_checks_state(tmp_path):
    checkpoint_path = tmp_path / "resume-step-000000050"
    checkpoint_path.mkdir()
    checkpoint = SimpleNamespace(path=checkpoint_path)
    restored = SimpleNamespace(trainer_state={"global_step": 50, "epoch": 0.1})
    rng_calls = []
    scheduler = object()

    def restore(engine, selected, *, lr_scheduler):
        assert lr_scheduler is scheduler
        return restored

    manager = SimpleNamespace(
        restore=restore,
        restore_rng=lambda selected: rng_calls.append(selected),
    )
    trainer = SimpleNamespace(
        model_wrapped="engine",
        lr_scheduler=scheduler,
        state=SimpleNamespace(global_step=50, epoch=0.1),
    )

    hook = distributed.checkpoint_manager_restore_hook(manager, checkpoint)
    assert hook(trainer, checkpoint_path.resolve()) is restored
    hook.restore_rng_state()
    assert rng_calls == [checkpoint]

    trainer.state.global_step = 49
    with pytest.raises(DistributedTrainingError, match="state mismatch"):
        hook(trainer, checkpoint_path.resolve())


def test_custom_checkpoint_is_not_forwarded_to_native_deepspeed(tmp_path):
    checkpoint = tmp_path / "resume-step-000000050"
    (checkpoint / "deepspeed").mkdir(parents=True)
    (checkpoint / ".complete").write_text("manifest-hash\n", encoding="ascii")

    with pytest.raises(DistributedTrainingError, match="explicit managed ownership"):
        distributed.run_full_training(
            load_config("configs/transition1x.yaml"),
            bundle_dir="bundle",
            processed_path="processed",
            output_dir="output",
            resume_from_checkpoint=checkpoint,
        )
