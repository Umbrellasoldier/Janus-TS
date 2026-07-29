from __future__ import annotations

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from peft import LoraConfig, TaskType
from torch import nn

import janus_ts.checkpointing as checkpointing
from janus_ts.artifacts import mark_complete, read_complete_manifest, sha256_file
from janus_ts.checkpointing import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointManager,
    JanusCheckpointCallback,
    ResumeCheckpoint,
    convert_pissa_to_portable_state,
    gather_lora_state_dict,
    materialize_train_loss_checkpoint,
    normalize_lora_state_dict,
    select_resume_checkpoint,
)


class ToyProjection(nn.Module):
    def __init__(self, fill: float) -> None:
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})
        self.base_layer.weight.requires_grad_(False)
        self.lora_A["default"].weight.data.fill_(fill)
        self.lora_B["default"].weight.data.fill_(fill + 1)


class ToyPeftModel(nn.Module):
    def __init__(self, fill: float) -> None:
        super().__init__()
        self.proj = ToyProjection(fill)
        self.peft_config = {
            "default": LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=32,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                use_rslora=True,
                init_lora_weights=True,
                target_modules=["proj"],
            )
        }
        self.config = SimpleNamespace(model_type="toy")


class FakeDeepSpeedEngine:
    def __init__(self, module: nn.Module, *, global_steps: int) -> None:
        self.module = module
        self.global_rank = 0
        self.world_size = 1
        self.global_steps = global_steps
        self.last_save_kwargs: dict | None = None
        self.last_load_kwargs: dict | None = None

    def save_checkpoint(
        self,
        save_dir,
        tag=None,
        client_state=None,
        save_latest=True,
        exclude_frozen_parameters=False,
    ):
        self.last_save_kwargs = {
            "save_dir": save_dir,
            "tag": tag,
            "client_state": client_state,
            "save_latest": save_latest,
            "exclude_frozen_parameters": exclude_frozen_parameters,
        }
        root = Path(save_dir) / str(tag)
        root.mkdir(parents=True)
        trainable = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.module.named_parameters()
            if parameter.requires_grad
        }
        torch.save(
            {"module": trainable, "client_state": client_state},
            root / "mp_rank_00_model_states.pt",
        )
        torch.save(
            {"optimizer": "state", "scheduler": "state"},
            root / "zero_pp_rank_0_mp_rank_00_optim_states.pt",
        )
        return True

    def load_checkpoint(self, load_dir, **kwargs):
        self.last_load_kwargs = {"load_dir": load_dir, **kwargs}
        tag = kwargs["tag"]
        payload = torch.load(
            Path(load_dir) / tag / "mp_rank_00_model_states.pt",
            map_location="cpu",
            weights_only=False,
        )
        named = dict(self.module.named_parameters())
        for name, tensor in payload["module"].items():
            named[name].data.copy_(tensor)
        self.global_steps = int(payload["client_state"]["janus_ts_checkpoint"]["global_step"])
        return str(Path(load_dir) / tag), payload["client_state"]


class ToyScheduler:
    def __init__(self, step: int) -> None:
        self.state = {
            "base_lrs": [0.0001],
            "last_epoch": step,
            "_step_count": step + 1,
            "_last_lr": [0.0001],
            "lr_lambdas": [{}],
        }

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)


def identity() -> CheckpointIdentity:
    return CheckpointIdentity.derive(
        config_fingerprint="a" * 64,
        data_fingerprint="b" * 64,
        model_fingerprint="c" * 64,
    )


def trainer_state(step: int, epoch: float) -> dict:
    return {"global_step": step, "epoch": epoch, "log_history": []}


def complete_manifest(
    path: Path,
    *,
    run_identity: CheckpointIdentity,
    step: int,
    kind: str = "rolling",
    epoch: float = 0.5,
) -> None:
    path.mkdir(parents=True)
    payload = path / "payload.txt"
    payload.write_text("complete", encoding="utf-8")
    tag = f"global_step{step:09d}"
    ds_root = path / "deepspeed" / tag
    ds_root.mkdir(parents=True)
    model_state = ds_root / "mp_rank_00_model_states.pt"
    optimizer_state = ds_root / "zero_pp_rank_0_mp_rank_00_optim_states.pt"
    model_state.write_bytes(b"model")
    optimizer_state.write_bytes(b"optimizer")
    scheduler_state = path / checkpointing.SCHEDULER_STATE_NAME
    torch.save(ToyScheduler(step).state_dict(), scheduler_state)
    manifest = {
        "schema_version": 1,
        "kind": kind,
        **run_identity.as_dict(),
        "global_step": step,
        "epoch": epoch,
        "world_size": 1,
        "deepspeed_tag": tag,
        "exclude_frozen_parameters": True,
        "package_versions": checkpointing._installed_versions(),
        "scheduler_state_name": checkpointing.SCHEDULER_STATE_NAME,
        "scheduler_class": "ToyScheduler",
        "payload_inventory": {
            "payload.txt": {
                "size_bytes": payload.stat().st_size,
                "sha256": sha256_file(payload),
            },
            scheduler_state.relative_to(path).as_posix(): {
                "size_bytes": scheduler_state.stat().st_size,
                "sha256": sha256_file(scheduler_state),
            },
            model_state.relative_to(path).as_posix(): {"size_bytes": model_state.stat().st_size},
            optimizer_state.relative_to(path).as_posix(): {
                "size_bytes": optimizer_state.stat().st_size
            },
        },
    }
    mark_complete(path, manifest)


def tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        candidate.relative_to(path).as_posix(): candidate.read_bytes()
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
    }


class SequentialBroadcastBus:
    def __init__(self) -> None:
        self.value = None
        self.calls = []


class SequentialCoordinator:
    """Sequential two-rank stand-in for one rank-zero object broadcast."""

    def __init__(self, rank: int, bus: SequentialBroadcastBus) -> None:
        self.rank = rank
        self.world_size = 2
        self.bus = bus

    def barrier(self) -> None:
        return None

    def broadcast(self, value, *, source: int = 0):
        self.bus.calls.append((self.rank, value))
        if self.rank == source:
            self.bus.value = value
        return self.bus.value


def test_identity_is_content_derived_and_strict():
    value = identity()
    assert len(value.run_fingerprint) == 64
    assert value.run_fingerprint == identity().run_fingerprint
    with pytest.raises(CheckpointError, match="SHA-256"):
        CheckpointIdentity("run", "a" * 64, "b" * 64, "c" * 64)


def test_pissa_portable_conversion_is_exact_rank_doubling():
    trained = {
        "x.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "x.lora_B.weight": torch.tensor([[3.0], [4.0]]),
    }
    initial = {
        "x.lora_A.weight": torch.tensor([[5.0, 6.0]]),
        "x.lora_B.weight": torch.tensor([[7.0], [8.0]]),
    }
    portable = convert_pissa_to_portable_state(trained, initial)

    assert torch.equal(portable["x.lora_A.weight"], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
    assert torch.equal(portable["x.lora_B.weight"], torch.tensor([[3.0, -7.0], [4.0, -8.0]]))
    with pytest.raises(CheckpointError, match="keys differ"):
        convert_pissa_to_portable_state(trained, {"other.lora_A.weight": torch.ones(1, 2)})


def test_rolling_checkpoint_materializes_as_compact_train_loss_candidate(tmp_path):
    from safetensors.torch import load_file, save_file

    run_identity = identity()
    source = tmp_path / "resume-step-000000050"
    complete_manifest(
        source,
        run_identity=run_identity,
        step=50,
        kind="rolling",
        epoch=0.5,
    )
    source_manifest = read_complete_manifest(source)
    tag_root = source / "deepspeed" / "global_step000000050"
    (tag_root / "zero_pp_rank_1_mp_rank_00_optim_states.pt").write_bytes(
        b"optimizer-rank-1"
    )

    adapter_config = {
        "r": 32,
        "lora_alpha": 16.0,
        "use_rslora": True,
        "init_lora_weights": True,
        "inference_mode": True,
        "rank_pattern": {},
        "alpha_pattern": {},
        "target_modules": ["proj"],
    }
    trained = {
        "model.proj.lora_A.weight": torch.tensor([[3.0, 4.0]]),
        "model.proj.lora_B.weight": torch.tensor([[5.0], [6.0]]),
    }
    initial = {
        "model.proj.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "model.proj.lora_B.weight": torch.tensor([[2.0], [3.0]]),
    }
    resume = source / "resume_adapter"
    resume.mkdir()
    (resume / "adapter_config.json").write_text(
        json.dumps(adapter_config),
        encoding="utf-8",
    )
    save_file(trained, resume / "adapter_model.safetensors")

    initial_root = tmp_path / "pissa_init"
    initial_root.mkdir()
    (initial_root / "adapter_config.json").write_text(
        json.dumps(adapter_config),
        encoding="utf-8",
    )
    save_file(initial, initial_root / "adapter_model.safetensors")

    portable_config = {
        **adapter_config,
        "r": 64,
        "lora_alpha": 16.0 * (2.0**0.5),
    }
    template = tmp_path / "portable-config.json"
    template.write_text(json.dumps(portable_config), encoding="utf-8")

    source_manifest["world_size"] = 2
    source_manifest["payload_inventory"] = checkpointing._payload_inventory(source)
    mark_complete(source, source_manifest)
    destination = tmp_path / "evaluation-checkpoint"
    result = materialize_train_loss_checkpoint(
        source,
        destination,
        identity=run_identity,
        initial_adapter_dir=initial_root,
        portable_config_template=template,
        expected_initial_adapter_fingerprint=checkpointing._hash_tree(initial_root),
        train_loss=0.125,
    )

    assert result == destination
    manifest = read_complete_manifest(destination)
    assert manifest["kind"] == "train-loss"
    assert manifest["global_step"] == 50
    assert manifest["train_loss"] == 0.125
    assert set(manifest["payload_inventory"]) == {
        "resume_adapter/adapter_config.json",
        "resume_adapter/adapter_model.safetensors",
        "portable_adapter/adapter_config.json",
        "portable_adapter/adapter_model.safetensors",
    }
    portable = load_file(destination / "portable_adapter" / "adapter_model.safetensors")
    assert torch.equal(
        portable["model.proj.lora_A.weight"],
        torch.tensor([[3.0, 4.0], [1.0, 2.0]]),
    )
    assert torch.equal(
        portable["model.proj.lora_B.weight"],
        torch.tensor([[5.0, -2.0], [6.0, -3.0]]),
    )
    assert (
        materialize_train_loss_checkpoint(
            source,
            destination,
            identity=run_identity,
            initial_adapter_dir=initial_root,
            portable_config_template=template,
            expected_initial_adapter_fingerprint=checkpointing._hash_tree(initial_root),
            train_loss=0.125,
        )
        == destination
    )


def test_adapter_key_normalization_rejects_wrong_adapter():
    state = {
        "base.proj.lora_A.default.weight": torch.ones(1, 2),
        "base.proj.lora_B.default.weight": torch.ones(2, 1),
    }
    assert set(normalize_lora_state_dict(state)) == {
        "base.proj.lora_A.weight",
        "base.proj.lora_B.weight",
    }
    with pytest.raises(CheckpointError, match="unexpected LoRA"):
        normalize_lora_state_dict(state, adapter_name="other")


def test_bf16_training_adapter_is_gathered_as_portable_fp32():
    model = ToyPeftModel(fill=2.0).to(dtype=torch.bfloat16)
    engine = FakeDeepSpeedEngine(model, global_steps=1)

    gathered = gather_lora_state_dict(
        engine,
        SimpleNamespace(rank=0),
        expected_trainable_parameters=4,
    )

    assert gathered is not None
    assert gathered
    assert {tensor.dtype for tensor in gathered.values()} == {torch.float32}


def test_atomic_save_excludes_frozen_base_and_exactly_restores(tmp_path):
    manager = CheckpointManager(identity(), expected_trainable_parameters=4)
    original = ToyPeftModel(fill=2.0)
    engine = FakeDeepSpeedEngine(original, global_steps=50)
    destination = manager.destination(tmp_path / "local", 50)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    saved = manager.save(
        engine,
        destination,
        kind="rolling",
        global_step=50,
        epoch=0.25,
        trainer_state=trainer_state(50, 0.25),
        lr_scheduler=ToyScheduler(50),
    )

    assert saved.path == destination
    assert engine.last_save_kwargs is not None
    assert engine.last_save_kwargs["exclude_frozen_parameters"] is True
    assert engine.last_save_kwargs["save_latest"] is False
    manifest = read_complete_manifest(destination)
    assert manifest["global_step"] == 50
    assert manifest["exclude_frozen_parameters"] is True
    assert (destination / ".complete").is_file()
    assert not any(path.name == "latest" for path in destination.rglob("*"))

    ds_payload = torch.load(
        destination / "deepspeed" / manifest["deepspeed_tag"] / "mp_rank_00_model_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ds_payload["module"]
    assert all("lora_" in name for name in ds_payload["module"])
    assert not any("base_layer" in name for name in ds_payload["module"])

    expected_random = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.random()
    np.random.random()
    torch.rand(())

    fresh = ToyPeftModel(fill=-10.0)
    fresh_engine = FakeDeepSpeedEngine(fresh, global_steps=0)
    fresh_scheduler = ToyScheduler(0)
    restored = manager.restore(
        fresh_engine,
        ResumeCheckpoint(destination, manifest),
        lr_scheduler=fresh_scheduler,
    )
    actual_random = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert restored.trainer_state["global_step"] == 50
    assert fresh_engine.global_steps == 50
    assert fresh_scheduler.state_dict()["last_epoch"] == 50
    assert fresh_engine.last_load_kwargs is not None
    assert fresh_engine.last_load_kwargs["load_module_strict"] is False
    assert fresh_engine.last_load_kwargs["load_optimizer_states"] is True
    assert fresh_engine.last_load_kwargs["load_lr_scheduler_states"] is True
    assert actual_random == pytest.approx(expected_random)
    for name, parameter in fresh.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter, dict(original.named_parameters())[name])


def test_restore_requires_a_fresh_engine(tmp_path):
    manager = CheckpointManager(identity(), expected_trainable_parameters=4)
    engine = FakeDeepSpeedEngine(ToyPeftModel(fill=1), global_steps=50)
    saved = manager.save(
        engine,
        manager.destination(tmp_path, 50),
        kind="rolling",
        global_step=50,
        epoch=0.5,
        trainer_state=trainer_state(50, 0.5),
        lr_scheduler=ToyScheduler(50),
    )
    with pytest.raises(CheckpointError, match="freshly initialized"):
        manager.restore(engine, saved, lr_scheduler=ToyScheduler(0))


def test_refuses_engine_without_frozen_parameter_exclusion(tmp_path):
    class UnsafeEngine:
        module = ToyPeftModel(fill=1)
        global_rank = 0
        world_size = 1
        global_steps = 0

        def save_checkpoint(self, save_dir, tag=None):
            raise AssertionError("must not be called")

    manager = CheckpointManager(identity(), expected_trainable_parameters=4)
    with pytest.raises(CheckpointError, match="refusing to save the 27B base"):
        manager.save(
            UnsafeEngine(),
            tmp_path / "unsafe",
            kind="checkpoint0",
            global_step=0,
            epoch=0,
            trainer_state=trainer_state(0, 0),
            lr_scheduler=ToyScheduler(0),
        )
    assert not (tmp_path / "unsafe").exists()


def test_resume_selection_uses_manifest_step_and_durable_tie(tmp_path):
    current = identity()
    local = tmp_path / "local"
    durable = tmp_path / "durable"
    complete_manifest(local / "name-looks-new", run_identity=current, step=50)
    complete_manifest(local / "name-looks-old", run_identity=current, step=100)
    complete_manifest(durable / "durable-copy", run_identity=current, step=100)
    foreign = CheckpointIdentity.derive(
        config_fingerprint="d" * 64,
        data_fingerprint="e" * 64,
        model_fingerprint="f" * 64,
    )
    complete_manifest(durable / "foreign", run_identity=foreign, step=999)

    # Deliberately make the lower-step path look newer at the filesystem level.
    os.utime(local / "name-looks-new", (2_000_000_000, 2_000_000_000))
    os.utime(local / "name-looks-old", (1, 1))

    selected = select_resume_checkpoint([local, durable], identity=current)
    assert selected is not None
    assert selected.global_step == 100
    assert selected.path == durable / "durable-copy"


def test_local_rotation_keeps_highest_two_steps_not_newest_mtime(tmp_path):
    manager = CheckpointManager(identity(), expected_trainable_parameters=4)
    root = tmp_path / "rolling"
    complete_manifest(
        root / "zero", run_identity=manager.identity, step=0, kind="checkpoint0", epoch=0
    )
    complete_manifest(root / "fifty", run_identity=manager.identity, step=50)
    complete_manifest(root / "hundred", run_identity=manager.identity, step=100)
    os.utime(root / "zero", (2_000_000_000, 2_000_000_000))

    removed = manager.rotate_local(root, keep=2)
    assert removed == (root / "zero",)
    assert {path.name for path in root.iterdir()} == {"fifty", "hundred"}


def test_payload_tamper_is_not_selectable(tmp_path):
    root = tmp_path / "checkpoints"
    path = root / "step"
    complete_manifest(path, run_identity=identity(), step=50)
    (path / "payload.txt").write_text("tampered", encoding="utf-8")

    assert select_resume_checkpoint([root], identity=identity()) is None


def test_corrupt_current_checkpoint0_is_quarantined_then_rebuilt(tmp_path, monkeypatch):
    current = identity()
    local = tmp_path / "local"
    manager = CheckpointManager(current, expected_trainable_parameters=4)
    destination = manager.destination(local, 0)
    complete_manifest(
        destination,
        run_identity=current,
        step=0,
        kind="checkpoint0",
        epoch=0,
    )
    (destination / "payload.txt").write_text("tampered", encoding="utf-8")
    damaged_bytes = tree_bytes(destination)
    save_calls = []

    def fake_save(engine, target, **kwargs):
        target = Path(target)
        assert not target.exists()
        save_calls.append((engine, target, kwargs))
        complete_manifest(
            target,
            run_identity=current,
            step=0,
            kind="checkpoint0",
            epoch=0,
        )
        return ResumeCheckpoint(target, read_complete_manifest(target))

    monkeypatch.setattr(manager, "save", fake_save)
    engine = object()
    callback = JanusCheckpointCallback(
        manager,
        engine_getter=lambda: engine,
        local_root=local,
        durable_root=tmp_path / "durable",
        initial_adapter_dir=tmp_path / "initial",
    )
    state = SimpleNamespace(global_step=0, epoch=0.0)
    control = object()

    assert (
        callback.on_train_begin(
            None,
            state,
            control,
            lr_scheduler=ToyScheduler(0),
        )
        is control
    )

    assert len(save_calls) == 1
    assert save_calls[0][1] == destination
    assert save_calls[0][2]["kind"] == "checkpoint0"
    assert (destination / "payload.txt").read_text(encoding="utf-8") == "complete"
    quarantines = tuple(local.glob(f".{destination.name}.quarantine.*"))
    assert len(quarantines) == 1
    quarantine = quarantines[0]
    assert tree_bytes(quarantine / "checkpoint") == damaged_bytes
    receipt = json.loads((quarantine / "quarantine.json").read_text(encoding="utf-8"))
    assert receipt["run_fingerprint"] == current.run_fingerprint
    assert receipt["payload_name"] == "checkpoint"
    assert "payload digest mismatch" in receipt["validation_error"]
    assert "timestamp" not in receipt
    assert "mtime" not in receipt


def test_valid_checkpoint0_is_reused_only_after_full_payload_validation(tmp_path):
    current = identity()
    manager = CheckpointManager(current, expected_trainable_parameters=4)
    destination = manager.destination(tmp_path / "local", 0)
    complete_manifest(
        destination,
        run_identity=current,
        step=0,
        kind="checkpoint0",
        epoch=0,
    )
    before = tree_bytes(destination)

    decision = manager._prepare_checkpoint0_destination(destination)

    assert decision == {"action": "reuse", "quarantine": None, "error": None}
    assert tree_bytes(destination) == before


def test_foreign_checkpoint0_is_left_byte_exact_and_not_quarantined(tmp_path):
    current = identity()
    foreign = CheckpointIdentity.derive(
        config_fingerprint="d" * 64,
        data_fingerprint="e" * 64,
        model_fingerprint="f" * 64,
    )
    manager = CheckpointManager(current, expected_trainable_parameters=4)
    destination = manager.destination(tmp_path / "local", 0)
    complete_manifest(
        destination,
        run_identity=foreign,
        step=0,
        kind="checkpoint0",
        epoch=0,
    )
    # Even a damaged foreign tree must never be moved aside for this run.
    (destination / "payload.txt").write_text("tampered", encoding="utf-8")
    before = tree_bytes(destination)

    with pytest.raises(CheckpointError, match="foreign identity.*left untouched"):
        manager._prepare_checkpoint0_destination(destination)

    assert destination.is_dir()
    assert tree_bytes(destination) == before
    assert not tuple(destination.parent.glob(f".{destination.name}.quarantine.*"))


def test_checkpoint0_quarantine_save_decision_is_identical_across_ranks(tmp_path):
    current = identity()
    destination = tmp_path / "local" / CheckpointManager.checkpoint_name(0)
    complete_manifest(
        destination,
        run_identity=current,
        step=0,
        kind="checkpoint0",
        epoch=0,
    )
    (destination / "payload.txt").write_text("tampered", encoding="utf-8")
    bus = SequentialBroadcastBus()
    rank0 = CheckpointManager(
        current,
        coordinator=SequentialCoordinator(0, bus),
        expected_trainable_parameters=4,
    )
    rank1 = CheckpointManager(
        current,
        coordinator=SequentialCoordinator(1, bus),
        expected_trainable_parameters=4,
    )

    rank0_decision = rank0._prepare_checkpoint0_destination(destination)
    rank1_decision = rank1._prepare_checkpoint0_destination(destination)

    assert rank0_decision == rank1_decision
    assert rank0_decision["action"] == "save"
    quarantine = Path(rank0_decision["quarantine"])
    assert quarantine.is_dir()
    assert (quarantine / "checkpoint" / "payload.txt").read_text(encoding="utf-8") == (
        "tampered"
    )
    assert not destination.exists()
    assert [rank for rank, _ in bus.calls] == [0, 1]


def test_checkpoint_manifest_contains_no_timestamp_fields(tmp_path):
    manager = CheckpointManager(identity(), expected_trainable_parameters=4)
    engine = FakeDeepSpeedEngine(ToyPeftModel(fill=1), global_steps=0)
    saved = manager.save(
        engine,
        manager.destination(tmp_path, 0),
        kind="checkpoint0",
        global_step=0,
        epoch=0,
        trainer_state=trainer_state(0, 0),
        lr_scheduler=ToyScheduler(0),
    )
    manifest_text = json.dumps(saved.manifest)
    assert "timestamp" not in manifest_text
    assert "mtime" not in manifest_text


def test_scheduler_step_must_match_trainer_step(tmp_path):
    manager = CheckpointManager(identity(), expected_trainable_parameters=4)
    engine = FakeDeepSpeedEngine(ToyPeftModel(fill=1), global_steps=50)
    with pytest.raises(CheckpointError, match="scheduler/Trainer step mismatch"):
        manager.save(
            engine,
            manager.destination(tmp_path, 50),
            kind="rolling",
            global_step=50,
            epoch=0.5,
            trainer_state=trainer_state(50, 0.5),
            lr_scheduler=ToyScheduler(49),
        )


def test_rng_capture_and_restore_touch_only_current_cuda_device(tmp_path, monkeypatch):
    cuda_state = torch.tensor([4, 2], dtype=torch.uint8)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: calls.append(("get", device)) or cuda_state.clone(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, *, device: calls.append(("set", device, state.clone())),
    )

    payload = checkpointing._capture_rng_state()
    assert payload["cuda_device"] == 1
    assert torch.equal(payload["torch_cuda"], cuda_state)
    assert calls == [("get", 1)]

    path = tmp_path / "rng.pt"
    torch.save(payload, path)
    checkpointing._restore_rng_state(path)
    assert calls[1][0:2] == ("set", 1)
    assert torch.equal(calls[1][2], cuda_state)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with pytest.raises(CheckpointError, match="device differs"):
        checkpointing._restore_rng_state(path)
