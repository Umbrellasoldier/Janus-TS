"""Qwen3.6 text-only loading and the frozen PiSSA/LoRA topology.

The Hub checkpoint is multimodal and stores language weights below
``model.language_model``.  Janus-TS deliberately instantiates
``Qwen3_5ForCausalLM`` instead, so loading must rename that prefix while
rejecting any loss of language-model weights.  Vision and MTP tensors are the
only checkpoint tensors that may be unused.

This module keeps the inexpensive topology calculations separate from model
construction.  As a result, all target names and parameter counts can be
audited without allocating (or downloading) a 27B model.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from janus_ts.constants import (
    MODEL_ID,
    MODEL_REVISION,
    QWEN_IM_END_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
)

LOCKED_MODEL_STACK: Mapping[str, str] = {
    "torch": "2.9.1",
    "transformers": "5.9.0",
    "peft": "0.19.1",
    "causal-conv1d": "1.6.2.post1+cu128torch2.9cxx11abitrueglibc228",
    "fla-core": "0.5.0",
    "flash-linear-attention": "0.5.0",
}

TEXT_CHECKPOINT_PREFIX = "model.language_model."
TEXT_MODEL_PREFIX = "model."

# Transformers 5.9 interprets ``key_mapping`` entries as regular-expression
# WeightRenaming rules.  Anchoring the rule prevents accidental rewrites in
# nested names.
TEXT_ONLY_KEY_MAPPING: Mapping[str, str] = {
    r"^model\.language_model\.": TEXT_MODEL_PREFIX,
}

LINEAR_ATTN_LORA_LEAVES = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)
FULL_ATTN_LORA_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_LORA_LEAVES = ("gate_proj", "up_proj", "down_proj")

LORA_RANK = 32
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.0
LORA_INIT = "pissa_niter_16"
EXPECTED_TARGET_MODULES = 496
EXPECTED_TRAINABLE_PARAMETERS = 233_455_616
PORTABLE_LORA_RANK = 64
EXPECTED_PORTABLE_PARAMETERS = 466_911_232
EXPECTED_CHECKPOINT_KEY_COUNTS: Mapping[str, int] = {
    "text": 851,
    "vision": 333,
    "mtp": 15,
    "total": 1199,
}

_ALLOWED_UNUSED_CHECKPOINT_PATTERNS = (
    re.compile(r"^model\.visual\."),
    re.compile(r"^mtp\."),
)


class ModelContractError(ValueError):
    """Raised when the frozen model or adapter contract is violated."""


@dataclass(frozen=True, slots=True)
class TopologySummary:
    """Auditable description of the selected Qwen and LoRA topology."""

    num_layers: int
    linear_attention_layers: tuple[int, ...]
    full_attention_layers: tuple[int, ...]
    target_modules: tuple[str, ...]
    rank: int
    trainable_parameters: int

    @property
    def target_module_count(self) -> int:
        return len(self.target_modules)


@dataclass(frozen=True, slots=True)
class PortableAdapterExpectation:
    """Expected config after PEFT converts PiSSA to ordinary LoRA."""

    rank: int
    alpha: float
    use_rslora: bool
    init_lora_weights: bool
    parameter_count: int
    target_module_count: int

    @property
    def scaling(self) -> float:
        return self.alpha / math.sqrt(self.rank)


@dataclass(frozen=True, slots=True)
class CheckpointKeyInventory:
    """Classification of keys in the original multimodal checkpoint."""

    mapped_text_keys: tuple[str, ...]
    vision_keys: tuple[str, ...]
    mtp_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PissaPreparationSpec:
    """Memory-bounded, GPU-only settings for one-time PiSSA preparation."""

    gpu_ids: tuple[int, int] = (0, 1)
    max_memory_per_gpu_gib: int = 44
    max_save_shard_size: str = "2GB"

    def __post_init__(self) -> None:
        if len(set(self.gpu_ids)) != 2 or any(index < 0 for index in self.gpu_ids):
            raise ModelContractError(f"exactly two distinct GPU IDs are required: {self.gpu_ids!r}")
        if not 32 <= self.max_memory_per_gpu_gib <= 44:
            raise ModelContractError(
                "PiSSA preparation max_memory_per_gpu_gib must be in [32, 44]"
            )
        if self.max_save_shard_size != "2GB":
            raise ModelContractError("PiSSA residual shards are frozen at 2GB")

    def load_kwargs(self) -> dict[str, Any]:
        # Deliberately omit a CPU entry. If the text model cannot fit in these
        # two budgets, Accelerate must fail instead of silently offloading.
        return {
            "device_map": "balanced",
            "max_memory": {
                gpu_id: f"{self.max_memory_per_gpu_gib}GiB" for gpu_id in self.gpu_ids
            },
            "low_cpu_mem_usage": True,
        }


DEFAULT_PISSA_PREPARATION_SPEC = PissaPreparationSpec()


def assert_locked_model_stack() -> dict[str, str]:
    """Fail unless every model/runtime package matches the experiment lock."""

    installed: dict[str, str] = {}
    failures: list[str] = []
    for package, expected in LOCKED_MODEL_STACK.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "not-installed"
        installed[package] = actual
        if actual != expected:
            failures.append(f"{package}: expected {expected}, found {actual}")
    if failures:
        raise ModelContractError("locked model stack mismatch: " + "; ".join(failures))
    return installed


def _config_value(config: Any, name: str) -> Any:
    try:
        return getattr(config, name)
    except AttributeError as exc:
        raise ModelContractError(f"Qwen text config has no {name!r}") from exc


def validate_qwen_text_config(config: Any) -> None:
    """Validate every architecture value that determines the adapter topology."""

    expected_scalars: Mapping[str, Any] = {
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
    }
    mismatches = []
    for name, expected in expected_scalars.items():
        actual = _config_value(config, name)
        if actual != expected:
            mismatches.append(f"{name}={actual!r} (expected {expected!r})")

    layer_types = tuple(_config_value(config, "layer_types"))
    expected_layer_types = tuple(
        "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
        for layer_idx in range(64)
    )
    if layer_types != expected_layer_types:
        mismatches.append("layer_types does not match 3 linear + 1 full attention × 16")

    if mismatches:
        raise ModelContractError("unexpected Qwen3.6-27B text config: " + "; ".join(mismatches))


def validate_qwen_outer_config(config: Any) -> None:
    """Ensure the Hub config is the expected Qwen multimodal wrapper."""

    if _config_value(config, "model_type") != "qwen3_5":
        raise ModelContractError(f"unexpected outer model_type: {config.model_type!r}")
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if architectures != ("Qwen3_5ForConditionalGeneration",):
        raise ModelContractError(f"unexpected Hub architectures: {architectures!r}")
    text_config = _config_value(config, "text_config")
    validate_qwen_text_config(text_config)
    if _config_value(text_config, "use_cache") is not True:
        raise ModelContractError("the untouched Hub text config must set use_cache=true")


def build_lora_target_modules(config: Any) -> tuple[str, ...]:
    """Enumerate exact module paths; never rely on a broad suffix selector."""

    validate_qwen_text_config(config)
    targets: list[str] = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == "linear_attention":
            targets.extend(
                f"model.layers.{layer_idx}.linear_attn.{leaf}"
                for leaf in LINEAR_ATTN_LORA_LEAVES
            )
        elif layer_type == "full_attention":
            targets.extend(
                f"model.layers.{layer_idx}.self_attn.{leaf}" for leaf in FULL_ATTN_LORA_LEAVES
            )
        else:  # guarded by validate_qwen_text_config; retained for defensive clarity
            raise ModelContractError(f"unsupported layer type {layer_type!r} at layer {layer_idx}")
        targets.extend(f"model.layers.{layer_idx}.mlp.{leaf}" for leaf in MLP_LORA_LEAVES)

    result = tuple(targets)
    if len(result) != EXPECTED_TARGET_MODULES or len(set(result)) != len(result):
        raise ModelContractError(
            f"LoRA target enumeration produced {len(result)} paths, expected "
            f"{EXPECTED_TARGET_MODULES} unique paths"
        )
    return result


def expected_lora_parameter_count(config: Any, rank: int = LORA_RANK) -> int:
    """Calculate A+B parameters from Qwen projection shapes."""

    validate_qwen_text_config(config)
    if rank <= 0:
        raise ModelContractError(f"LoRA rank must be positive, got {rank}")

    hidden = int(config.hidden_size)
    key_dim = int(config.linear_num_key_heads) * int(config.linear_key_head_dim)
    value_dim = int(config.linear_num_value_heads) * int(config.linear_value_head_dim)
    linear_projection_dimensions = (
        (hidden, 2 * key_dim + value_dim),  # in_proj_qkv
        (hidden, value_dim),  # in_proj_z
        (hidden, int(config.linear_num_value_heads)),  # in_proj_b
        (hidden, int(config.linear_num_value_heads)),  # in_proj_a
        (value_dim, hidden),  # out_proj
    )

    attention_width = int(config.num_attention_heads) * int(config.head_dim)
    kv_width = int(config.num_key_value_heads) * int(config.head_dim)
    full_projection_dimensions = (
        (hidden, 2 * attention_width),  # q_proj also emits the output gate
        (hidden, kv_width),
        (hidden, kv_width),
        (attention_width, hidden),
    )
    intermediate = int(config.intermediate_size)
    mlp_projection_dimensions = (
        (hidden, intermediate),
        (hidden, intermediate),
        (intermediate, hidden),
    )

    linear_layers = sum(layer == "linear_attention" for layer in config.layer_types)
    full_layers = sum(layer == "full_attention" for layer in config.layer_types)

    def adapter_parameters(dimensions: Sequence[tuple[int, int]]) -> int:
        return rank * sum(input_dim + output_dim for input_dim, output_dim in dimensions)

    total = (
        linear_layers * adapter_parameters(linear_projection_dimensions)
        + full_layers * adapter_parameters(full_projection_dimensions)
        + int(config.num_hidden_layers) * adapter_parameters(mlp_projection_dimensions)
    )
    return total


def topology_summary(config: Any) -> TopologySummary:
    """Return and hard-check the frozen rank-32 experiment topology."""

    targets = build_lora_target_modules(config)
    count = expected_lora_parameter_count(config, rank=LORA_RANK)
    if count != EXPECTED_TRAINABLE_PARAMETERS:
        raise ModelContractError(
            f"analytical LoRA parameter count is {count:,}, expected "
            f"{EXPECTED_TRAINABLE_PARAMETERS:,}"
        )
    linear = tuple(i for i, kind in enumerate(config.layer_types) if kind == "linear_attention")
    full = tuple(i for i, kind in enumerate(config.layer_types) if kind == "full_attention")
    return TopologySummary(
        num_layers=int(config.num_hidden_layers),
        linear_attention_layers=linear,
        full_attention_layers=full,
        target_modules=targets,
        rank=LORA_RANK,
        trainable_parameters=count,
    )


def portable_adapter_expectation(config: Any) -> PortableAdapterExpectation:
    """Describe PEFT's lossless PiSSA-to-standard-LoRA conversion.

    PEFT concatenates the trained and initial factors, doubling rank.  With
    rsLoRA it multiplies alpha by sqrt(2), preserving ``alpha / sqrt(rank)``.
    """

    summary = topology_summary(config)
    parameter_count = expected_lora_parameter_count(config, rank=PORTABLE_LORA_RANK)
    if parameter_count != EXPECTED_PORTABLE_PARAMETERS:
        raise ModelContractError(
            f"portable parameter count is {parameter_count:,}, expected "
            f"{EXPECTED_PORTABLE_PARAMETERS:,}"
        )
    return PortableAdapterExpectation(
        rank=PORTABLE_LORA_RANK,
        alpha=LORA_ALPHA * math.sqrt(2.0),
        use_rslora=True,
        init_lora_weights=True,
        parameter_count=parameter_count,
        target_module_count=summary.target_module_count,
    )


def _selectors_cover_frozen_targets(selectors: Sequence[str], text_config: Any) -> bool:
    if not selectors or len(set(selectors)) != len(selectors):
        return False

    def is_selected(path: str) -> bool:
        return any(path == selector or path.endswith("." + selector) for selector in selectors)

    expected_paths = build_lora_target_modules(text_config)
    return all(is_selected(path) for path in expected_paths) and not is_selected("lm_head")


def create_lora_config(config: Any) -> Any:
    """Create the one permitted PEFT configuration for this experiment."""

    summary = topology_summary(config)
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_rslora=True,
        init_lora_weights=LORA_INIT,
        target_modules=list(summary.target_modules),
    )


def validate_pissa_initialization_config(
    config_path: str | Path,
    text_config: Any,
    *,
    prepared_reference: bool,
) -> dict[str, Any]:
    """Validate a PiSSA init config before or after residual-base preparation.

    PEFT must see ``init_lora_weights=True`` when loading the saved factors on
    top of an already-residualized base.  Otherwise it performs PiSSA a second
    time and subtracts another low-rank decomposition.  The manifest retains
    ``pissa_niter_16`` as the algorithm provenance.
    """

    path = Path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelContractError(f"invalid PiSSA initialization config {path}: {exc}") from exc

    expected_init: bool | str = True if prepared_reference else LORA_INIT
    exact_fields: Mapping[str, Any] = {
        "base_model_name_or_path": MODEL_ID,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "bias": "none",
        "use_rslora": True,
        "init_lora_weights": expected_init,
    }
    problems = [
        f"{field}={payload.get(field)!r}"
        for field, expected in exact_fields.items()
        if payload.get(field) != expected
    ]
    selectors = tuple(payload.get("target_modules", ()))
    if not _selectors_cover_frozen_targets(selectors, text_config):
        problems.append(
            f"target_modules={len(selectors)} does not select exactly the frozen topology"
        )
    if problems:
        raise ModelContractError("PiSSA initialization config mismatch: " + "; ".join(problems))
    return payload


def save_pissa_initialization_reference(
    model: Any,
    output_dir: str | Path,
    text_config: Any,
) -> None:
    """Save canonical rank-32 init factors for an unmerged residual base.

    The adapter is first saved with its actual PiSSA initializer metadata, then
    normalized to ``init_lora_weights=True``.  This is the PEFT-required load
    form for a base whose PiSSA component has already been subtracted.
    """

    from janus_ts.artifacts import write_json

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    model.save_pretrained(output, safe_serialization=True)
    config_path = output / "adapter_config.json"
    payload = validate_pissa_initialization_config(
        config_path, text_config, prepared_reference=False
    )
    payload["init_lora_weights"] = True
    write_json(config_path, payload)
    validate_pissa_initialization_config(config_path, text_config, prepared_reference=True)


def _is_allowed_unused_checkpoint_key(key: str) -> bool:
    return any(pattern.match(key) is not None for pattern in _ALLOWED_UNUSED_CHECKPOINT_PATTERNS)


def classify_checkpoint_keys(keys: Iterable[str]) -> CheckpointKeyInventory:
    """Map language keys and reject unknown tensors before loading shards."""

    mapped_text: list[str] = []
    vision: list[str] = []
    mtp: list[str] = []
    rejected: list[str] = []
    for key in keys:
        if key.startswith(TEXT_CHECKPOINT_PREFIX):
            mapped_text.append(TEXT_MODEL_PREFIX + key.removeprefix(TEXT_CHECKPOINT_PREFIX))
        elif key.startswith("lm_head."):
            mapped_text.append(key)
        elif key.startswith("model.visual."):
            vision.append(key)
        elif key.startswith("mtp."):
            mtp.append(key)
        else:
            rejected.append(key)
    if rejected:
        preview = ", ".join(sorted(rejected)[:8])
        raise ModelContractError(f"checkpoint contains unsupported keys: {preview}")
    if len(mapped_text) != len(set(mapped_text)):
        raise ModelContractError("checkpoint text-key mapping is not one-to-one")
    if not mapped_text:
        raise ModelContractError("checkpoint has no language-model weights")
    return CheckpointKeyInventory(
        mapped_text_keys=tuple(sorted(mapped_text)),
        vision_keys=tuple(sorted(vision)),
        mtp_keys=tuple(sorted(mtp)),
    )


def validate_checkpoint_index(
    index_path: str | Path, expected_text_keys: Iterable[str] | None = None
) -> CheckpointKeyInventory:
    """Audit a downloaded safetensors index without opening any weight shard."""

    path = Path(index_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = payload["weight_map"].keys()
    except (OSError, json.JSONDecodeError, KeyError, AttributeError) as exc:
        raise ModelContractError(f"invalid safetensors index {path}: {exc}") from exc

    inventory = classify_checkpoint_keys(keys)
    if expected_text_keys is not None:
        expected = set(expected_text_keys)
        actual = set(inventory.mapped_text_keys)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ModelContractError(
                "checkpoint/text-model key mismatch: "
                f"missing={missing[:8]!r}, extra={extra[:8]!r}"
            )
    return inventory


def validate_pinned_checkpoint_index(index_path: str | Path) -> CheckpointKeyInventory:
    """Enforce the complete key inventory at the frozen Hub revision."""

    inventory = validate_checkpoint_index(index_path)
    actual = {
        "text": len(inventory.mapped_text_keys),
        "vision": len(inventory.vision_keys),
        "mtp": len(inventory.mtp_keys),
        "total": (
            len(inventory.mapped_text_keys)
            + len(inventory.vision_keys)
            + len(inventory.mtp_keys)
        ),
    }
    if actual != EXPECTED_CHECKPOINT_KEY_COUNTS:
        raise ModelContractError(
            f"pinned checkpoint key inventory changed: {actual!r}, "
            f"expected {dict(EXPECTED_CHECKPOINT_KEY_COUNTS)!r}"
        )
    return inventory


def validate_loading_info(loading_info: Mapping[str, Any]) -> None:
    """Reject missing/mismatched text tensors and non-vision/MTP leftovers."""

    missing = tuple(loading_info.get("missing_keys", ()) or ())
    mismatched = tuple(loading_info.get("mismatched_keys", ()) or ())
    errors = tuple(loading_info.get("error_msgs", ()) or ())
    unexpected = tuple(loading_info.get("unexpected_keys", ()) or ())
    disallowed_unexpected = tuple(
        key for key in unexpected if not _is_allowed_unused_checkpoint_key(key)
    )
    if missing or mismatched or errors or disallowed_unexpected:
        raise ModelContractError(
            "text-only checkpoint load was incomplete: "
            f"missing={missing[:8]!r}, mismatched={mismatched[:8]!r}, "
            f"unexpected={disallowed_unexpected[:8]!r}, errors={errors[:3]!r}"
        )


def load_qwen_text_config(
    *, cache_dir: str | Path | None = None, local_files_only: bool = False
) -> Any:
    """Load and validate the pinned Hub config, returning its text sub-config."""

    from transformers import Qwen3_5Config

    outer_config = Qwen3_5Config.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    validate_qwen_outer_config(outer_config)
    text_config = outer_config.text_config

    # Chat completion terminates on im_end, not the upstream endoftext token.
    text_config.pad_token_id = QWEN_PAD_TOKEN_ID
    text_config.eos_token_id = QWEN_IM_END_TOKEN_ID
    text_config.use_cache = False
    text_config._attn_implementation = "sdpa"
    text_config._name_or_path = MODEL_ID
    return text_config


def load_qwen_text_base(
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    **from_pretrained_kwargs: Any,
) -> Any:
    """Load the pinned multimodal checkpoint into a text-only BF16 model.

    ``from_pretrained_kwargs`` exists for distributed loader controls such as a
    DeepSpeed/Accelerate device context.  Frozen identity, revision, config,
    dtype, attention implementation, key mapping, and loading report cannot be
    overridden.
    """

    forbidden = {
        "config",
        "revision",
        "dtype",
        "torch_dtype",
        "attn_implementation",
        "key_mapping",
        "output_loading_info",
        "trust_remote_code",
    }.intersection(from_pretrained_kwargs)
    if forbidden:
        raise ModelContractError(f"cannot override frozen load arguments: {sorted(forbidden)!r}")

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Qwen3_5ForCausalLM

    config = load_qwen_text_config(cache_dir=cache_dir, local_files_only=local_files_only)
    index_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors.index.json",
        revision=MODEL_REVISION,
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=local_files_only,
    )
    validate_pinned_checkpoint_index(index_path)
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        key_mapping=dict(TEXT_ONLY_KEY_MAPPING),
        output_loading_info=True,
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=local_files_only,
        trust_remote_code=False,
        **from_pretrained_kwargs,
    )
    validate_loading_info(loading_info)
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise ModelContractError(f"loaded wrong model class: {type(model).__name__}")
    validate_qwen_text_config(model.config)
    return model


def assert_gpu_only_dispatch(model: Any, spec: PissaPreparationSpec) -> dict[str, Any]:
    """Require both selected GPUs and reject any CPU/disk offload."""

    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict) or not device_map:
        raise ModelContractError("prepared base has no Accelerate hf_device_map")

    used_gpu_ids: set[int] = set()
    disallowed: dict[str, Any] = {}
    for module_name, raw_device in device_map.items():
        if isinstance(raw_device, int):
            used_gpu_ids.add(raw_device)
            continue
        device = str(raw_device)
        if device.isdigit():
            used_gpu_ids.add(int(device))
        elif device.startswith("cuda:") and device.removeprefix("cuda:").isdigit():
            used_gpu_ids.add(int(device.removeprefix("cuda:")))
        else:
            disallowed[module_name] = raw_device

    expected = set(spec.gpu_ids)
    if used_gpu_ids != expected or disallowed:
        raise ModelContractError(
            "PiSSA preparation must be GPU-only across both cards: "
            f"used={sorted(used_gpu_ids)}, expected={sorted(expected)}, "
            f"offloaded={dict(list(disallowed.items())[:8])!r}"
        )
    return dict(device_map)


def load_qwen_text_base_for_pissa_preparation(
    *,
    cache_dir: str | Path | None,
    spec: PissaPreparationSpec = DEFAULT_PISSA_PREPARATION_SPEC,
    local_files_only: bool = False,
) -> Any:
    """Stream the pinned BF16 text model directly onto two GPUs.

    The absence of a CPU budget in ``max_memory`` is intentional: with only
    about 30 GiB host RAM, CPU offload is not a safe fallback.
    """

    model = load_qwen_text_base(
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        **spec.load_kwargs(),
    )
    assert_gpu_only_dispatch(model, spec)
    return model


def _freeze_base_and_cast_lora_fp32(model: Any) -> None:
    import torch

    for name, parameter in model.named_parameters():
        is_adapter = _is_lora_parameter(name)
        parameter.requires_grad_(is_adapter)
        if is_adapter and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(dtype=torch.float32)


def load_prepared_pissa_model(
    bundle_dir: str | Path,
    *,
    spec: PissaPreparationSpec = DEFAULT_PISSA_PREPARATION_SPEC,
    require_complete: bool = True,
) -> Any:
    """Reload residual BF16 base + canonical rank-32 PiSSA factors.

    This is the single-process, two-GPU parity loader used during preparation.
    A torchrun/ZeRO-3 training process must use the same residual and adapter
    artifacts through its distributed loader, never this cross-GPU device map.
    """

    import torch
    from peft import PeftModel
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from janus_ts.artifacts import read_complete_manifest

    bundle = Path(bundle_dir)
    if require_complete:
        read_complete_manifest(bundle)
    residual_dir = bundle / "residual_base"
    initial_dir = bundle / "pissa_init"
    config = Qwen3_5TextConfig.from_pretrained(
        residual_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    validate_qwen_text_config(config)
    validate_pissa_initialization_config(
        initial_dir / "adapter_config.json", config, prepared_reference=True
    )

    base, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        residual_dir,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
        local_files_only=True,
        trust_remote_code=False,
        **spec.load_kwargs(),
    )
    validate_loading_info(loading_info)
    validate_qwen_text_config(base.config)
    base.config.eos_token_id = QWEN_IM_END_TOKEN_ID
    base.config.pad_token_id = QWEN_PAD_TOKEN_ID
    base.config.use_cache = False
    assert_gpu_only_dispatch(base, spec)

    model = PeftModel.from_pretrained(
        base,
        initial_dir,
        is_trainable=True,
        autocast_adapter_dtype=True,
        low_cpu_mem_usage=True,
    )
    _freeze_base_and_cast_lora_fp32(model)
    assert_pissa_adapter_contract(model)
    return model


def _hash_artifact_tree(root: Path) -> dict[str, str]:
    from janus_ts.artifacts import COMPLETE_MARKER, sha256_file

    hashes: dict[str, str] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {COMPLETE_MARKER, "manifest.json"}:
            continue
        hashes[relative] = sha256_file(path)
    return hashes


def prepare_pissa_residual_bundle(
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None,
    parity_hook: Callable[[str, Any], Mapping[str, Any]],
    manifest_metadata: Mapping[str, Any] | None = None,
    spec: PissaPreparationSpec = DEFAULT_PISSA_PREPARATION_SPEC,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Create and verify the one-time immutable PiSSA preparation bundle.

    ``parity_hook`` is called with ``original_base``, ``pissa_initialized``,
    and ``pissa_reloaded``.  It must retain its own reference probe, compare
    subsequent stages, return JSON-safe diagnostics, and raise on a failed
    tolerance/generation check.  It must retain neither the model nor any GPU
    tensor between calls; only small detached CPU probes are safe.  Keeping the
    scientific parity policy in the CLI/evaluation layer avoids silently
    weakening it here.

    Peak host memory is bounded by low-memory streaming and 2GB save shards.
    The base is dispatched only across the two GPUs.  The PiSSA adapter is
    saved first; ``unload()`` (never ``merge_and_unload()``) then exposes and
    saves the residual BF16 base.  Finally, both are reloaded and parity-tested
    before the atomic completion marker is written.
    """

    import gc

    import torch

    from janus_ts.artifacts import (
        atomic_directory,
        canonical_json_bytes,
        mark_complete,
        read_complete_manifest,
        sha256_json,
    )

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    metadata = dict(manifest_metadata or {})
    canonical_json_bytes(metadata)
    stack = assert_locked_model_stack()
    topology: TopologySummary
    parity_records: dict[str, Mapping[str, Any]] = {}

    def run_parity(stage: str, model: Any) -> None:
        record = dict(parity_hook(stage, model))
        # Validate JSON safety before investing in the multi-gigabyte save.
        canonical_json_bytes(record)
        parity_records[stage] = record

    base = load_qwen_text_base_for_pissa_preparation(
        cache_dir=cache_dir,
        spec=spec,
        local_files_only=local_files_only,
    )
    topology = topology_summary(base.config)
    run_parity("original_base", base)
    pissa_model = attach_pissa_adapter(base)
    del base
    run_parity("pissa_initialized", pissa_model)

    manifest: dict[str, Any]
    with atomic_directory(destination) as building:
        save_pissa_initialization_reference(
            pissa_model,
            building / "pissa_init",
            pissa_model.config,
        )

        # This is intentionally unmerged: the returned model contains W minus
        # the rank-32 PiSSA component and is not a standalone original Qwen.
        residual_model = pissa_model.unload()
        del pissa_model
        if any(_is_lora_parameter(name) for name, _ in residual_model.named_parameters()):
            raise ModelContractError("unload() left LoRA parameters in the residual base")
        residual_model.save_pretrained(
            building / "residual_base",
            safe_serialization=True,
            max_shard_size=spec.max_save_shard_size,
            save_original_format=False,
        )
        del residual_model
        gc.collect()
        torch.cuda.empty_cache()

        reloaded = load_prepared_pissa_model(building, spec=spec, require_complete=False)
        run_parity("pissa_reloaded", reloaded)
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()

        file_hashes = _hash_artifact_tree(building)
        manifest = {
            "artifact_type": "janus-ts-pissa-residual-bundle",
            "format_version": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "base_semantics": "unmerged_pissa_residual_not_standalone",
            "initialization_algorithm": LORA_INIT,
            "initial_adapter_load_init_lora_weights": True,
            "stack": stack,
            "preparation": {
                "gpu_ids": list(spec.gpu_ids),
                "max_memory_per_gpu_gib": spec.max_memory_per_gpu_gib,
                "cpu_offload": False,
                "max_save_shard_size": spec.max_save_shard_size,
                "base_dtype": "bfloat16",
                "adapter_dtype": "float32",
                "unload_without_merge": True,
            },
            "topology": {
                "target_modules": topology.target_module_count,
                "rank": topology.rank,
                "trainable_parameters": topology.trainable_parameters,
            },
            "parity": parity_records,
            "files": file_hashes,
            "files_sha256": sha256_json(file_hashes),
            "metadata": metadata,
        }
        canonical_json_bytes(manifest)
        mark_complete(building, manifest)

    return read_complete_manifest(destination)


def assert_target_modules_present(model: Any, target_modules: Sequence[str]) -> None:
    """Verify that each exact target resolves to one and only one linear module."""

    import torch

    names = tuple(name for name, _ in model.named_modules())
    problems: list[str] = []
    module_by_name = dict(model.named_modules())
    for target in target_modules:
        matches = tuple(name for name in names if name == target or name.endswith("." + target))
        if len(matches) != 1:
            problems.append(f"{target}: {len(matches)} matches")
        elif not isinstance(module_by_name[matches[0]], torch.nn.Linear):
            problems.append(f"{target}: resolved module is not torch.nn.Linear")
    if problems:
        raise ModelContractError("invalid LoRA targets: " + "; ".join(problems[:12]))


def _is_lora_parameter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def _logical_numel(parameter: Any) -> int:
    """Return full parameter size for both ordinary and ZeRO-3 parameters."""

    return int(getattr(parameter, "ds_numel", parameter.numel()))


def attach_pissa_adapter(model: Any) -> Any:
    """Inject PiSSA, freeze everything else, and enforce FP32 adapter weights."""

    from peft import get_peft_model

    summary = topology_summary(model.config)
    assert_target_modules_present(model, summary.target_modules)
    peft_model = get_peft_model(model, create_lora_config(model.config))
    _freeze_base_and_cast_lora_fp32(peft_model)
    assert_pissa_adapter_contract(peft_model, summary.target_modules)
    return peft_model


def assert_pissa_adapter_contract(model: Any, target_modules: Sequence[str] | None = None) -> None:
    """Hard-check module coverage, freezing, FP32 dtype, and logical count."""

    import torch

    targets = tuple(target_modules or topology_summary(model.config).target_modules)
    modules = tuple(model.named_modules())
    missing_or_ambiguous: list[str] = []
    for target in targets:
        matches = tuple(
            module
            for name, module in modules
            if name == target or name.endswith("." + target)
        )
        if len(matches) != 1 or not all(
            hasattr(matches[0], attribute) for attribute in ("lora_A", "lora_B")
        ):
            missing_or_ambiguous.append(target)
    if missing_or_ambiguous:
        raise ModelContractError(
            f"{len(missing_or_ambiguous)} injected LoRA targets are missing or ambiguous: "
            f"{missing_or_ambiguous[:8]!r}"
        )

    trainable_count = 0
    wrong_dtype: list[str] = []
    wrongly_trainable: list[str] = []
    frozen_adapters: list[str] = []
    for name, parameter in model.named_parameters():
        is_adapter = _is_lora_parameter(name)
        if parameter.requires_grad:
            trainable_count += _logical_numel(parameter)
            if not is_adapter:
                wrongly_trainable.append(name)
            if parameter.dtype != torch.float32:
                wrong_dtype.append(name)
        elif is_adapter:
            frozen_adapters.append(name)

    if (
        trainable_count != EXPECTED_TRAINABLE_PARAMETERS
        or wrongly_trainable
        or frozen_adapters
        or wrong_dtype
    ):
        raise ModelContractError(
            "PiSSA adapter contract failed: "
            f"trainable={trainable_count:,} (expected {EXPECTED_TRAINABLE_PARAMETERS:,}), "
            f"non_lora_trainable={wrongly_trainable[:8]!r}, "
            f"frozen_lora={frozen_adapters[:8]!r}, non_fp32={wrong_dtype[:8]!r}"
        )


def save_portable_adapter(
    model: Any, output_dir: str | Path, initial_adapter_dir: str | Path
) -> None:
    """Export ordinary LoRA relative to the untouched base-model weights.

    ``initial_adapter_dir`` must be the permanent rank-32 PiSSA initialization
    reference saved before optimization.  PEFT performs the exact factor
    subtraction and rank-doubling conversion.
    """

    output = Path(output_dir)
    initial = Path(initial_adapter_dir)
    if output.exists():
        raise FileExistsError(f"portable adapter output already exists: {output}")
    if not (initial / "adapter_config.json").is_file():
        raise ModelContractError(f"missing PiSSA initialization reference: {initial}")
    validate_pissa_initialization_config(
        initial / "adapter_config.json", model.config, prepared_reference=True
    )

    # PEFT 0.19.1 rewrites adapter_config.json in the directory passed as
    # path_initial_model_for_weight_conversion.  Always give it a disposable
    # copy so the permanent, content-hashed initialization reference remains
    # byte-for-byte unchanged.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pissa-conversion-reference.", dir=output.parent
    ) as temporary_root:
        conversion_reference = Path(temporary_root) / "initial_adapter"
        shutil.copytree(initial, conversion_reference)
        model.save_pretrained(
            output,
            safe_serialization=True,
            path_initial_model_for_weight_conversion=conversion_reference,
        )


def validate_portable_adapter_config(config_path: str | Path, text_config: Any) -> None:
    """Validate PEFT's saved rank-64 portable adapter metadata."""

    path = Path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelContractError(f"invalid portable adapter config {path}: {exc}") from exc

    expected = portable_adapter_expectation(text_config)
    actual_targets = tuple(payload.get("target_modules", ()))
    problems: list[str] = []
    frozen_metadata = {
        "base_model_name_or_path": MODEL_ID,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "lora_dropout": LORA_DROPOUT,
    }
    for field, field_expected in frozen_metadata.items():
        if payload.get(field) != field_expected:
            problems.append(f"{field}={payload.get(field)!r}")
    if payload.get("r") != expected.rank:
        problems.append(f"r={payload.get('r')!r}")
    if not math.isclose(float(payload.get("lora_alpha", math.nan)), expected.alpha, rel_tol=1e-12):
        problems.append(f"lora_alpha={payload.get('lora_alpha')!r}")
    if payload.get("use_rslora") is not expected.use_rslora:
        problems.append(f"use_rslora={payload.get('use_rslora')!r}")
    if payload.get("init_lora_weights") is not expected.init_lora_weights:
        problems.append(f"init_lora_weights={payload.get('init_lora_weights')!r}")
    if not actual_targets or len(set(actual_targets)) != len(actual_targets):
        problems.append(f"target_modules={len(actual_targets)}")
    else:
        expected_paths = build_lora_target_modules(text_config)

        def is_selected(path: str) -> bool:
            return any(
                path == selector or path.endswith("." + selector) for selector in actual_targets
            )

        selected_expected = tuple(path for path in expected_paths if is_selected(path))
        if selected_expected != expected_paths:
            problems.append(
                "target_modules select "
                f"{len(selected_expected)}/{expected.target_module_count} modules"
            )
        if is_selected("lm_head"):
            problems.append("target_modules unexpectedly select the frozen lm_head")
    if problems:
        raise ModelContractError("portable adapter config mismatch: " + "; ".join(problems))


__all__ = [
    "DEFAULT_PISSA_PREPARATION_SPEC",
    "EXPECTED_CHECKPOINT_KEY_COUNTS",
    "EXPECTED_PORTABLE_PARAMETERS",
    "EXPECTED_TARGET_MODULES",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "FULL_ATTN_LORA_LEAVES",
    "LINEAR_ATTN_LORA_LEAVES",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_INIT",
    "LORA_RANK",
    "LOCKED_MODEL_STACK",
    "MLP_LORA_LEAVES",
    "ModelContractError",
    "PissaPreparationSpec",
    "PORTABLE_LORA_RANK",
    "TEXT_ONLY_KEY_MAPPING",
    "TopologySummary",
    "assert_locked_model_stack",
    "assert_gpu_only_dispatch",
    "assert_pissa_adapter_contract",
    "assert_target_modules_present",
    "attach_pissa_adapter",
    "build_lora_target_modules",
    "classify_checkpoint_keys",
    "create_lora_config",
    "expected_lora_parameter_count",
    "load_qwen_text_base",
    "load_prepared_pissa_model",
    "load_qwen_text_base_for_pissa_preparation",
    "load_qwen_text_config",
    "portable_adapter_expectation",
    "prepare_pissa_residual_bundle",
    "save_pissa_initialization_reference",
    "save_portable_adapter",
    "topology_summary",
    "validate_checkpoint_index",
    "validate_loading_info",
    "validate_portable_adapter_config",
    "validate_pinned_checkpoint_index",
    "validate_pissa_initialization_config",
    "validate_qwen_outer_config",
    "validate_qwen_text_config",
]
