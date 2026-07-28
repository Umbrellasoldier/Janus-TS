from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from janus_ts.constants import MODEL_ID
from janus_ts.modeling import (
    EXPECTED_CHECKPOINT_KEY_COUNTS,
    EXPECTED_PORTABLE_PARAMETERS,
    EXPECTED_TARGET_MODULES,
    EXPECTED_TRAINABLE_PARAMETERS,
    LORA_ALPHA,
    LORA_INIT,
    TEXT_ONLY_KEY_MAPPING,
    ModelContractError,
    PissaPreparationSpec,
    assert_gpu_only_dispatch,
    build_lora_target_modules,
    classify_checkpoint_keys,
    create_lora_config,
    expected_lora_parameter_count,
    portable_adapter_expectation,
    save_pissa_initialization_reference,
    save_portable_adapter,
    topology_summary,
    validate_checkpoint_index,
    validate_loading_info,
    validate_pinned_checkpoint_index,
    validate_pissa_initialization_config,
    validate_portable_adapter_config,
    validate_qwen_text_config,
)


def qwen_text_config(**overrides):
    values = {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 262144,
        "vocab_size": 248320,
        "layer_types": [
            "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
            for layer_idx in range(64)
        ],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def initial_adapter_payload(*, init_lora_weights):
    return {
        "base_model_name_or_path": MODEL_ID,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 32,
        "lora_alpha": 16.0,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": True,
        "init_lora_weights": init_lora_weights,
        "target_modules": list(build_lora_target_modules(qwen_text_config())),
    }


def test_text_only_key_mapping_is_anchored():
    assert TEXT_ONLY_KEY_MAPPING == {r"^model\.language_model\.": "model."}


def test_validate_qwen_text_config_rejects_architecture_drift():
    validate_qwen_text_config(qwen_text_config())

    with pytest.raises(ModelContractError, match="hidden_size"):
        validate_qwen_text_config(qwen_text_config(hidden_size=4096))

    wrong_layers = qwen_text_config().layer_types.copy()
    wrong_layers[0] = "full_attention"
    with pytest.raises(ModelContractError, match="layer_types"):
        validate_qwen_text_config(qwen_text_config(layer_types=wrong_layers))


def test_exact_target_module_enumeration():
    targets = build_lora_target_modules(qwen_text_config())

    assert len(targets) == EXPECTED_TARGET_MODULES == 496
    assert len(set(targets)) == len(targets)
    assert targets[:8] == (
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.0.linear_attn.in_proj_z",
        "model.layers.0.linear_attn.in_proj_b",
        "model.layers.0.linear_attn.in_proj_a",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    )
    assert "model.layers.3.self_attn.q_proj" in targets
    assert "model.layers.3.linear_attn.in_proj_qkv" not in targets
    assert "model.layers.63.self_attn.o_proj" in targets


def test_exact_rank32_and_portable_parameter_counts():
    config = qwen_text_config()
    summary = topology_summary(config)
    portable = portable_adapter_expectation(config)

    assert len(summary.linear_attention_layers) == 48
    assert len(summary.full_attention_layers) == 16
    assert summary.trainable_parameters == EXPECTED_TRAINABLE_PARAMETERS == 233_455_616
    assert expected_lora_parameter_count(config, rank=64) == 466_911_232
    assert portable.rank == 64
    assert portable.parameter_count == EXPECTED_PORTABLE_PARAMETERS == 466_911_232
    assert portable.alpha == pytest.approx(LORA_ALPHA * math.sqrt(2.0))
    assert portable.scaling == pytest.approx(summary.rank**-0.5 * LORA_ALPHA)


def test_lora_config_is_frozen_contract():
    pytest.importorskip("peft")
    lora = create_lora_config(qwen_text_config())

    assert lora.r == 32
    assert lora.lora_alpha == 16.0
    assert lora.lora_dropout == 0.0
    assert lora.bias == "none"
    assert lora.use_rslora is True
    assert lora.init_lora_weights == "pissa_niter_16"
    assert len(lora.target_modules) == 496


def test_pissa_preparation_spec_is_two_gpu_and_has_no_cpu_fallback():
    spec = PissaPreparationSpec()
    kwargs = spec.load_kwargs()
    assert kwargs == {
        "device_map": "balanced",
        "max_memory": {0: "44GiB", 1: "44GiB"},
        "low_cpu_mem_usage": True,
    }
    with pytest.raises(ModelContractError, match="two distinct"):
        PissaPreparationSpec(gpu_ids=(0, 0))
    with pytest.raises(ModelContractError, match="frozen at 2GB"):
        PissaPreparationSpec(max_save_shard_size="5GB")


def test_gpu_dispatch_rejects_cpu_offload_or_one_card():
    spec = PissaPreparationSpec()
    good = SimpleNamespace(hf_device_map={"model.embed_tokens": 0, "model.layers.32": 1})
    assert set(assert_gpu_only_dispatch(good, spec).values()) == {0, 1}

    with pytest.raises(ModelContractError, match="GPU-only"):
        assert_gpu_only_dispatch(
            SimpleNamespace(hf_device_map={"model.embed_tokens": 0, "model.layers.32": "cpu"}),
            spec,
        )
    with pytest.raises(ModelContractError, match="GPU-only"):
        assert_gpu_only_dispatch(SimpleNamespace(hf_device_map={"": 0}), spec)


def test_initial_reference_is_normalized_for_residual_reload(tmp_path):
    class FakePeftModel:
        def save_pretrained(self, output_dir, *, safe_serialization):
            assert safe_serialization is True
            output = Path(output_dir)
            output.mkdir()
            (output / "adapter_config.json").write_text(
                json.dumps(initial_adapter_payload(init_lora_weights=LORA_INIT)),
                encoding="utf-8",
            )
            (output / "adapter_model.safetensors").write_bytes(b"fake")

    output = tmp_path / "pissa_init"
    save_pissa_initialization_reference(FakePeftModel(), output, qwen_text_config())
    payload = validate_pissa_initialization_config(
        output / "adapter_config.json",
        qwen_text_config(),
        prepared_reference=True,
    )
    assert payload["init_lora_weights"] is True


def test_portable_conversion_uses_copy_of_immutable_initial_reference(tmp_path):
    initial = tmp_path / "canonical_init"
    initial.mkdir()
    initial_config = initial / "adapter_config.json"
    initial_config.write_text(
        json.dumps(initial_adapter_payload(init_lora_weights=True)), encoding="utf-8"
    )
    (initial / "adapter_model.safetensors").write_bytes(b"immutable")
    before = initial_config.read_bytes()

    class MutatingFakePeftModel:
        config = qwen_text_config()

        def save_pretrained(
            self,
            output_dir,
            *,
            safe_serialization,
            path_initial_model_for_weight_conversion,
        ):
            assert safe_serialization is True
            reference = Path(path_initial_model_for_weight_conversion)
            assert reference != initial
            (reference / "adapter_config.json").write_text("mutated", encoding="utf-8")
            output = Path(output_dir)
            output.mkdir()
            (output / "adapter_model.safetensors").write_bytes(b"portable")

    save_portable_adapter(MutatingFakePeftModel(), tmp_path / "portable", initial)
    assert initial_config.read_bytes() == before


def test_checkpoint_key_inventory_maps_text_and_only_allows_vision_mtp():
    inventory = classify_checkpoint_keys(
        [
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "lm_head.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "mtp.layers.0.mlp.up_proj.weight",
        ]
    )

    assert inventory.mapped_text_keys == (
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
    )
    assert len(inventory.vision_keys) == 1
    assert len(inventory.mtp_keys) == 1

    with pytest.raises(ModelContractError, match="unsupported keys"):
        classify_checkpoint_keys(["model.language_model.norm.weight", "foreign.weight"])


def test_checkpoint_index_can_compare_exact_text_key_set(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.embed_tokens.weight": "shard-1.safetensors",
                    "lm_head.weight": "shard-2.safetensors",
                    "model.visual.pos_embed.weight": "shard-2.safetensors",
                    "mtp.fc.weight": "shard-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    inventory = validate_checkpoint_index(
        index_path, expected_text_keys={"model.embed_tokens.weight", "lm_head.weight"}
    )
    assert len(inventory.mapped_text_keys) == 2

    with pytest.raises(ModelContractError, match="key mismatch"):
        validate_checkpoint_index(index_path, expected_text_keys={"model.norm.weight"})


def test_pinned_checkpoint_index_counts_are_frozen(tmp_path):
    keys = {f"model.language_model.fake.{i}": "text.safetensors" for i in range(850)}
    keys["lm_head.weight"] = "text.safetensors"
    keys.update({f"model.visual.fake.{i}": "vision.safetensors" for i in range(333)})
    keys.update({f"mtp.fake.{i}": "mtp.safetensors" for i in range(15)})
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": keys}), encoding="utf-8")

    inventory = validate_pinned_checkpoint_index(index_path)
    assert len(inventory.mapped_text_keys) == EXPECTED_CHECKPOINT_KEY_COUNTS["text"]

    keys.pop("mtp.fake.14")
    index_path.write_text(json.dumps({"weight_map": keys}), encoding="utf-8")
    with pytest.raises(ModelContractError, match="inventory changed"):
        validate_pinned_checkpoint_index(index_path)


def test_loading_info_rejects_any_incomplete_language_load():
    validate_loading_info(
        {
            "missing_keys": [],
            "mismatched_keys": [],
            "unexpected_keys": ["model.visual.pos_embed.weight", "mtp.fc.weight"],
            "error_msgs": [],
        }
    )

    with pytest.raises(ModelContractError, match="incomplete"):
        validate_loading_info({"missing_keys": ["model.layers.0.mlp.up_proj.weight"]})
    with pytest.raises(ModelContractError, match="incomplete"):
        validate_loading_info({"unexpected_keys": ["foreign.weight"]})
    with pytest.raises(ModelContractError, match="incomplete"):
        validate_loading_info({"mismatched_keys": [("lm_head.weight", (1,), (2,))]})


def test_portable_adapter_config_validation(tmp_path):
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "base_model_name_or_path": MODEL_ID,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "bias": "none",
                "lora_dropout": 0.0,
                "r": 64,
                "lora_alpha": 16.0 * math.sqrt(2.0),
                "use_rslora": True,
                "init_lora_weights": True,
                "target_modules": list(build_lora_target_modules(qwen_text_config())),
            }
        ),
        encoding="utf-8",
    )
    validate_portable_adapter_config(config_path, qwen_text_config())

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["r"] = 32
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelContractError, match="portable adapter config mismatch"):
        validate_portable_adapter_config(config_path, qwen_text_config())
