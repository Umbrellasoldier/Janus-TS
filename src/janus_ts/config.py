"""Strict experiment configuration and stable hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import (
    FORMAL_EVAL_K,
    MAX_NEW_TOKENS,
    MAX_SEQUENCE_LENGTH,
    MIN_TS_EDGE_WEIGHT,
    MODEL_ID,
    MODEL_REVISION,
    QWEN_IM_END_TOKEN_ID,
    QWEN_IM_START_TOKEN_ID,
    QWEN_PAD_TOKEN_ID,
    SEED,
    SIGNATURE_ALGORITHM,
    TRANSITION1X_SOURCE_SHA256,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    name: str
    raw_dir: Path
    h5_path: Path
    recovery_jsonl: Path
    processed_root: Path
    expected_source_sha256: dict[str, str]
    expected_raw_counts: dict[str, int]
    expected_retained_counts: dict[str, int]
    min_ts_edge_weight: float
    signature_algorithm: str
    keep_in_memory: bool
    dataloader_workers: int = Field(ge=0)
    pin_memory: bool


class ModelConfig(StrictModel):
    model_id: str
    revision: str
    cache_dir: Path
    dtype: Literal["bfloat16"]
    max_sequence_length: int
    pad_token_id: int
    im_start_token_id: int
    eos_token_id: int


class AdapterConfig(StrictModel):
    rank: int
    alpha: float
    dropout: float
    bias: Literal["none"]
    use_rslora: bool
    init_lora_weights: str
    expected_target_modules: int
    expected_trainable_parameters: int
    expected_portable_rank: int
    expected_portable_parameters: int


class TrainConfig(StrictModel):
    epochs: int
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    scheduler: Literal["cosine"]
    warmup_ratio: float
    max_grad_norm: float
    micro_batch_size_per_gpu: int
    gradient_accumulation_steps: int
    candidate_micro_batch_size_per_gpu: int
    candidate_gradient_accumulation_steps: int
    global_batch_size: int
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool
    average_tokens_across_devices: bool
    checkpoint_steps: int
    keep_local_checkpoints: int
    log_steps: int
    deepspeed_config: Path


class GenerationConfig(StrictModel):
    enable_thinking: bool
    num_beams: int
    num_return_sequences: int
    do_sample: bool
    max_new_tokens: int
    length_penalty: float | None = None
    early_stopping: bool | None = None
    report_k: tuple[int, ...] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None


class RuntimeConfig(StrictModel):
    artifacts_root: Path
    local_cache_root: Path
    gpu_lock_path: Path
    required_gpu_count: int
    max_gpu_peak_mib: int
    retry_delays_minutes: tuple[int, ...]


class ExperimentConfig(StrictModel):
    experiment: str
    seed: int
    data: DataConfig
    model: ModelConfig
    adapter: AdapterConfig
    train: TrainConfig
    generation: GenerationConfig
    thinking_generation: GenerationConfig
    runtime: RuntimeConfig

    @model_validator(mode="after")
    def validate_frozen_protocol(self) -> ExperimentConfig:
        expected = {
            "seed": (self.seed, SEED),
            "min_ts_edge_weight": (self.data.min_ts_edge_weight, MIN_TS_EDGE_WEIGHT),
            "model_id": (self.model.model_id, MODEL_ID),
            "revision": (self.model.revision, MODEL_REVISION),
            "max_sequence_length": (self.model.max_sequence_length, MAX_SEQUENCE_LENGTH),
            "pad_token_id": (self.model.pad_token_id, QWEN_PAD_TOKEN_ID),
            "im_start_token_id": (self.model.im_start_token_id, QWEN_IM_START_TOKEN_ID),
            "eos_token_id": (self.model.eos_token_id, QWEN_IM_END_TOKEN_ID),
            "max_new_tokens": (self.generation.max_new_tokens, MAX_NEW_TOKENS),
            "report_k": (self.generation.report_k, FORMAL_EVAL_K),
            "expected_source_sha256": (
                self.data.expected_source_sha256,
                TRANSITION1X_SOURCE_SHA256,
            ),
            "signature_algorithm": (
                self.data.signature_algorithm,
                SIGNATURE_ALGORITHM,
            ),
        }
        mismatches = [
            f"{name}: got {actual!r}, expected {wanted!r}"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if mismatches:
            raise ValueError("frozen protocol mismatch: " + "; ".join(mismatches))
        if self.generation.enable_thinking or self.generation.do_sample:
            raise ValueError("formal generation must be deterministic non-thinking")
        if not self.thinking_generation.enable_thinking:
            raise ValueError("optional thinking profile must enable thinking")
        if self.train.global_batch_size != (
            self.train.micro_batch_size_per_gpu
            * self.runtime.required_gpu_count
            * self.train.gradient_accumulation_steps
        ):
            raise ValueError("default batch geometry does not equal global batch size")
        if self.train.global_batch_size != (
            self.train.candidate_micro_batch_size_per_gpu
            * self.runtime.required_gpu_count
            * self.train.candidate_gradient_accumulation_steps
        ):
            raise ValueError("candidate batch geometry does not equal global batch size")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(raw)
