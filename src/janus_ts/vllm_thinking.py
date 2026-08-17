"""High-throughput vLLM runtime for complete-test thinking inference."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .artifacts import sha256_file, write_json
from .config import ExperimentConfig, GenerationConfig
from .constants import QWEN_IM_END_TOKEN_ID
from .preprocessing import load_pinned_tokenizer
from .runtime import install_frozen_environment
from .schema import ReactionRecord
from .thinking_inference import (
    WORLD_SIZE,
    ScheduledRecord,
    ThinkingExplorationReceipt,
    ThinkingInferenceError,
    ThinkingPrediction,
    _atomic_write,
    _batch_persistence_payload,
    _read_jsonl,
    _request_payload,
    _validate_prediction_rows,
    encode_thinking_prompt,
    exploration_batch_path,
    exploration_run_path,
    finalize_exploration_artifact,
    load_exploration_records,
    schedule_equal_rank_calls,
    select_exploration_records,
    validate_exploration_artifact,
    validate_thinking_profile,
    write_exploration_batch,
    write_exploration_fragment,
)

VLLM_VERSION = "0.25.1"
VLLM_ENVIRONMENT = f"vllm-{VLLM_VERSION}-py312"
VLLM_BASE_MODEL_NAME = "qwen3.6-27b"
VLLM_LORA_MODEL_NAME = "janus-fine-tuned"
VLLM_REQUEST_CONCURRENCY = 8
VLLM_MAX_MODEL_LENGTH = 10_240
VLLM_MAX_BATCHED_TOKENS = 16_384
VLLM_GPU_MEMORY_UTILIZATION = 0.88
VLLM_STARTUP_TIMEOUT_SECONDS = 20 * 60
VLLM_REQUEST_TIMEOUT_SECONDS = 4 * 60 * 60
VLLM_ADAPTER_SCHEMA_VERSION = "janus-ts-qwen3.6-vllm-lora-prefix-v1"
VLLM_RUNTIME_SCHEMA_VERSION = "janus-ts-vllm-thinking-runtime-v1"
_SOURCE_LORA_PREFIX = "base_model.model.model.layers."
_VLLM_LORA_PREFIX = "base_model.model.model.language_model.layers."


class VllmThinkingError(ThinkingInferenceError):
    """The complete-test vLLM inference contract was violated."""


@dataclass(frozen=True, slots=True)
class VllmAdapter:
    path: Path
    source_sha256: str
    converted_sha256: str
    key_count: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VLLM_ADAPTER_SCHEMA_VERSION,
            "path": str(self.path.resolve()),
            "source_sha256": self.source_sha256,
            "converted_sha256": self.converted_sha256,
            "key_count": self.key_count,
        }


@dataclass(frozen=True, slots=True)
class VllmEndpoint:
    base_url: str
    base_model: str
    request_model: str
    log_path: Path


@dataclass(frozen=True, slots=True)
class VllmCompletion:
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    finish_reason: str


def _rename_lora_key(name: str) -> str:
    if not name.startswith(_SOURCE_LORA_PREFIX):
        raise VllmThinkingError(f"unexpected portable LoRA key: {name}")
    return _VLLM_LORA_PREFIX + name.removeprefix(_SOURCE_LORA_PREFIX)


def rewrite_safetensors_lora_prefix(source: str | Path, destination: str | Path) -> int:
    """Rewrite only safetensors header keys; tensor bytes and offsets stay unchanged."""

    source_path = Path(source)
    destination_path = Path(destination)
    with source_path.open("rb") as source_handle:
        raw_length = source_handle.read(8)
        if len(raw_length) != 8:
            raise VllmThinkingError("portable adapter has a truncated safetensors header")
        header_length = int.from_bytes(raw_length, "little")
        raw_header = source_handle.read(header_length)
        if len(raw_header) != header_length:
            raise VllmThinkingError("portable adapter has a truncated safetensors manifest")
        try:
            header = json.loads(raw_header)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VllmThinkingError(f"portable adapter header is invalid: {exc}") from exc
        if not isinstance(header, dict):
            raise VllmThinkingError("portable adapter header is not an object")

        converted: dict[str, Any] = {}
        key_count = 0
        for name, value in header.items():
            if name == "__metadata__":
                converted[name] = value
                continue
            renamed = _rename_lora_key(name)
            if renamed in converted:
                raise VllmThinkingError(f"duplicate converted LoRA key: {renamed}")
            converted[renamed] = value
            key_count += 1
        if key_count <= 0:
            raise VllmThinkingError("portable adapter contains no LoRA tensors")

        encoded = json.dumps(converted, separators=(",", ":"), ensure_ascii=False).encode()
        encoded += b" " * (-len(encoded) % 8)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_path.open("wb") as destination_handle:
            destination_handle.write(len(encoded).to_bytes(8, "little"))
            destination_handle.write(encoded)
            shutil.copyfileobj(source_handle, destination_handle, length=16 * 1024 * 1024)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    return key_count


def _validate_vllm_adapter(
    destination: Path,
    *,
    source_sha256: str,
) -> VllmAdapter | None:
    manifest_path = destination / "manifest.json"
    config_path = destination / "adapter_config.json"
    weights_path = destination / "adapter_model.safetensors"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VllmThinkingError(f"cannot read converted adapter manifest: {exc}") from exc
    expected = {
        "schema_version": VLLM_ADAPTER_SCHEMA_VERSION,
        "source_sha256": source_sha256,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(name) != value for name, value in expected.items()
    ):
        raise VllmThinkingError("converted vLLM adapter identity mismatch")
    if not config_path.is_file() or not weights_path.is_file():
        raise VllmThinkingError("converted vLLM adapter is incomplete")
    converted_sha256 = sha256_file(weights_path)
    key_count = manifest.get("key_count")
    if (
        manifest.get("converted_sha256") != converted_sha256
        or isinstance(key_count, bool)
        or not isinstance(key_count, int)
        or key_count <= 0
    ):
        raise VllmThinkingError("converted vLLM adapter failed digest validation")
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if len(keys) != key_count or any(not key.startswith(_VLLM_LORA_PREFIX) for key in keys):
        raise VllmThinkingError("converted vLLM adapter has invalid module names")
    return VllmAdapter(destination, source_sha256, converted_sha256, key_count)


def prepare_vllm_adapter(
    portable_adapter: str | Path,
    *,
    cache_root: str | Path,
    checkpoint_fingerprint: str,
) -> VllmAdapter:
    """Create one content-verified adapter whose keys match stock vLLM Qwen3.6."""

    source = Path(portable_adapter)
    source_config = source / "adapter_config.json"
    source_weights = source / "adapter_model.safetensors"
    if not source_config.is_file() or not source_weights.is_file():
        raise VllmThinkingError(f"portable adapter is incomplete: {source}")
    source_sha256 = sha256_file(source_weights)
    destination = (
        Path(cache_root) / "vllm" / "adapters" / f"{checkpoint_fingerprint}-{source_sha256[:16]}"
    )
    recovered = _validate_vllm_adapter(destination, source_sha256=source_sha256)
    if recovered is not None:
        return recovered
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "adapter_model.safetensors"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".adapter_model.", suffix=".safetensors.tmp", dir=destination
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        key_count = rewrite_safetensors_lora_prefix(source_weights, temporary)
        os.replace(temporary, weights_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _atomic_write(destination / "adapter_config.json", source_config.read_bytes())
    converted_sha256 = sha256_file(weights_path)
    write_json(
        destination / "manifest.json",
        {
            "schema_version": VLLM_ADAPTER_SCHEMA_VERSION,
            "source_path": str(source.resolve()),
            "source_sha256": source_sha256,
            "converted_sha256": converted_sha256,
            "key_count": key_count,
        },
    )
    converted = _validate_vllm_adapter(destination, source_sha256=source_sha256)
    if converted is None:  # pragma: no cover - manifest was just written
        raise VllmThinkingError("converted vLLM adapter did not validate")
    return converted


def _vllm_environment_root(config: ExperimentConfig) -> Path:
    return Path(config.runtime.local_cache_root) / "venvs" / VLLM_ENVIRONMENT


def _model_snapshot_path(config: ExperimentConfig) -> Path:
    cache_name = "models--" + config.model.model_id.replace("/", "--")
    return Path(config.model.cache_dir) / cache_name / "snapshots" / config.model.revision


def _vllm_version(environment: Path) -> str:
    candidates = tuple(environment.glob("lib/python*/site-packages/vllm-*.dist-info/METADATA"))
    if len(candidates) != 1:
        raise VllmThinkingError(f"cannot identify vLLM installation in {environment}")
    for line in candidates[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ").strip()
    raise VllmThinkingError("vLLM package metadata has no version")


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _get_json(url: str, *, timeout: float) -> Mapping[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise VllmThinkingError(f"vLLM returned a non-object response from {url}")
    return payload


def _post_json(url: str, payload: Mapping[str, Any], *, timeout: float) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        return _open_post(request, timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise VllmThinkingError(f"vLLM HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise VllmThinkingError(f"vLLM request failed: {exc}") from exc


def _open_post(request: urllib.request.Request, timeout: float) -> Mapping[str, Any]:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise VllmThinkingError("vLLM returned a non-object completion")
    return payload


def _tail(path: Path, *, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
    except ProcessLookupError:
        return


@contextmanager
def serve_vllm(
    config: ExperimentConfig,
    *,
    request_sha256: str,
    adapter: VllmAdapter | None,
) -> Iterator[VllmEndpoint]:
    """Start one pinned local vLLM server and always stop its process group."""

    environment = _vllm_environment_root(config)
    executable = environment / "bin" / "vllm"
    model = _model_snapshot_path(config)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise VllmThinkingError(f"vLLM executable is unavailable: {executable}")
    version = _vllm_version(environment)
    if version != VLLM_VERSION:
        raise VllmThinkingError(f"vLLM version is {version}, expected {VLLM_VERSION}")
    if not (model / "config.json").is_file():
        raise VllmThinkingError(f"pinned Qwen snapshot is unavailable: {model}")

    port = _free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = (
        Path(config.runtime.local_cache_root) / "vllm" / "runs" / request_sha256 / "server.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "serve",
        str(model),
        "--served-model-name",
        VLLM_BASE_MODEL_NAME,
        "--dtype",
        "bfloat16",
        "--pipeline-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--language-model-only",
        "--max-model-len",
        str(VLLM_MAX_MODEL_LENGTH),
        "--gpu-memory-utilization",
        str(VLLM_GPU_MEMORY_UTILIZATION),
        "--max-num-seqs",
        str(VLLM_REQUEST_CONCURRENCY),
        "--max-num-batched-tokens",
        str(VLLM_MAX_BATCHED_TOKENS),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    request_model = VLLM_BASE_MODEL_NAME
    if adapter is not None:
        request_model = VLLM_LORA_MODEL_NAME
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                "64",
                "--max-loras",
                "1",
                "--lora-modules",
                f"{request_model}={adapter.path}",
            ]
        )
    environment_variables = os.environ.copy()
    environment_variables.update(
        {
            "PATH": f"{environment / 'bin'}:{environment_variables.get('PATH', '')}",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "VLLM_LOGGING_LEVEL": "INFO",
        }
    )
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment_variables,
            start_new_session=True,
        )
    started = time.monotonic()
    announced_at = started
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_shutdown(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("vLLM thinking runtime received SIGTERM")

    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        while True:
            if process.poll() is not None:
                raise VllmThinkingError(
                    f"vLLM exited during startup with {process.returncode}:\n{_tail(log_path)}"
                )
            try:
                models = _get_json(f"{base_url}/v1/models", timeout=2)
            except (OSError, urllib.error.URLError, VllmThinkingError):
                models = {}
            identifiers = {
                item.get("id") for item in models.get("data", []) if isinstance(item, Mapping)
            }
            if {VLLM_BASE_MODEL_NAME, request_model}.issubset(identifiers):
                break
            now = time.monotonic()
            if now - started > VLLM_STARTUP_TIMEOUT_SECONDS:
                raise VllmThinkingError(f"vLLM startup timed out:\n{_tail(log_path)}")
            if now - announced_at >= 30:
                print(
                    f"[thinking-vllm] starting server: {int(now - started)}s elapsed",
                    flush=True,
                )
                announced_at = now
            time.sleep(2)
        print(
            f"[thinking-vllm] server ready in {time.monotonic() - started:.1f}s",
            flush=True,
        )
        yield VllmEndpoint(base_url, VLLM_BASE_MODEL_NAME, request_model, log_path)
    finally:
        _stop_server(process)
        signal.signal(signal.SIGTERM, previous_sigterm)


def _completion_payload(
    endpoint: VllmEndpoint,
    input_ids: Sequence[int],
    profile: GenerationConfig,
    *,
    seed: int,
    request_id: str,
) -> dict[str, Any]:
    return {
        "model": endpoint.request_model,
        "prompt": list(input_ids),
        "max_tokens": profile.max_new_tokens,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "top_k": profile.top_k,
        "min_p": profile.min_p,
        "presence_penalty": profile.presence_penalty,
        "frequency_penalty": 0.0,
        "repetition_penalty": profile.repetition_penalty,
        "seed": seed,
        "n": 1,
        "stop_token_ids": [QWEN_IM_END_TOKEN_ID],
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "return_token_ids": True,
        "stream": False,
        "request_id": request_id,
    }


def request_completion(
    endpoint: VllmEndpoint,
    input_ids: Sequence[int],
    profile: GenerationConfig,
    *,
    seed: int,
    request_id: str,
) -> VllmCompletion:
    payload = _post_json(
        f"{endpoint.base_url}/v1/completions",
        _completion_payload(endpoint, input_ids, profile, seed=seed, request_id=request_id),
        timeout=VLLM_REQUEST_TIMEOUT_SECONDS,
    )
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise VllmThinkingError("vLLM completion has the wrong choice count")
    choice = choices[0]
    prompt_ids = choice.get("prompt_token_ids")
    token_ids = choice.get("token_ids")
    finish_reason = choice.get("finish_reason")
    if prompt_ids != list(input_ids):
        raise VllmThinkingError("vLLM changed the pre-tokenized thinking prompt")
    if not isinstance(token_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in token_ids
    ):
        raise VllmThinkingError("vLLM completion omitted token IDs")
    if finish_reason not in {"stop", "length"}:
        raise VllmThinkingError(f"vLLM completion has invalid finish reason {finish_reason!r}")
    return VllmCompletion(tuple(prompt_ids), tuple(token_ids), str(finish_reason))


def validate_adapter_application(
    endpoint: VllmEndpoint,
    input_ids: Sequence[int],
    *,
    output_path: Path,
) -> None:
    if endpoint.request_model == endpoint.base_model:
        return
    responses: dict[str, Mapping[str, Any]] = {}
    for model in (endpoint.base_model, endpoint.request_model):
        payload = {
            "model": model,
            "prompt": list(input_ids),
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "logprobs": 10,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "stop_token_ids": [QWEN_IM_END_TOKEN_ID],
            "seed": 42,
        }
        responses[model] = _post_json(
            f"{endpoint.base_url}/v1/completions",
            payload,
            timeout=300,
        )
    base_choice = responses[endpoint.base_model]["choices"][0]
    adapter_choice = responses[endpoint.request_model]["choices"][0]
    if base_choice["logprobs"]["top_logprobs"] == adapter_choice["logprobs"]["top_logprobs"]:
        raise VllmThinkingError("vLLM LoRA gate failed: adapter logits equal base logits")
    write_json(
        output_path,
        {
            "schema_version": "janus-ts-vllm-lora-gate-v1",
            "status": "pass",
            "base_model": endpoint.base_model,
            "adapter_model": endpoint.request_model,
            "base_token_ids": base_choice["token_ids"],
            "adapter_token_ids": adapter_choice["token_ids"],
            "base_token_logprobs": base_choice["logprobs"]["token_logprobs"],
            "adapter_token_logprobs": adapter_choice["logprobs"]["token_logprobs"],
        },
    )


def _engine_manifest(config: ExperimentConfig, adapter: VllmAdapter | None) -> dict[str, Any]:
    environment = _vllm_environment_root(config)
    version = _vllm_version(environment)
    if version != VLLM_VERSION:
        raise VllmThinkingError(f"vLLM version is {version}, expected {VLLM_VERSION}")
    return {
        "schema_version": VLLM_RUNTIME_SCHEMA_VERSION,
        "name": "vllm",
        "version": version,
        "execution": "openai-compatible-local-server",
        "dtype": "bfloat16",
        "pipeline_parallel_size": 2,
        "request_concurrency": VLLM_REQUEST_CONCURRENCY,
        "max_model_length": VLLM_MAX_MODEL_LENGTH,
        "max_num_batched_tokens": VLLM_MAX_BATCHED_TOKENS,
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "adapter": None if adapter is None else adapter.to_json_dict(),
    }


def _vllm_request_payload(
    config: ExperimentConfig,
    checkpoint: Any,
    *,
    split: str,
    records: Sequence[ReactionRecord],
    sample_count_per_reaction: int,
    adapter: VllmAdapter | None,
) -> dict[str, Any]:
    payload = _request_payload(
        config,
        checkpoint,
        split=split,
        records=records,
        sample_count_per_reaction=sample_count_per_reaction,
        reaction_batch_size=1,
    )
    payload["runtime_execution"] = "vllm-openai-compatible-two-gpu"
    payload["inference_engine"] = _engine_manifest(config, adapter)
    profile = dict(payload["generation_profile"])
    profile.pop("presence_penalty_forwarded_to_transformers", None)
    profile.pop("presence_penalty_note", None)
    profile["presence_penalty_forwarded_to_vllm"] = True
    payload["generation_profile"] = profile
    execution = dict(payload["generation_execution"])
    execution.update(
        {
            "candidates_batched_in_one_generate_call": False,
            "reaction_batch_size": 1,
            "request_concurrency": VLLM_REQUEST_CONCURRENCY,
            "synced_gpus": False,
        }
    )
    payload["generation_execution"] = execution
    payload["batch_persistence"] = _batch_persistence_payload()
    return payload


def _decode_completion(tokenizer: Any, completion: VllmCompletion) -> str:
    try:
        value = tokenizer.decode(
            list(completion.token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        raise VllmThinkingError(f"cannot decode vLLM thinking response: {exc}") from exc
    if not isinstance(value, str):
        raise VllmThinkingError("tokenizer returned a non-string vLLM response")
    return value


def _prediction_row(
    item: ScheduledRecord,
    encoding: Any,
    completion: VllmCompletion,
    tokenizer: Any,
    *,
    split: str,
    rank: int,
    request_sha256: str,
    checkpoint_fingerprint: str,
) -> dict[str, Any] | None:
    prediction = ThinkingPrediction(
        reaction_id=item.record.reaction_id,
        ordinal=item.ordinal,
        atom_count=item.record.atom_count,
        prompt_sha256=encoding.prompt_sha256,
        prompt_tokens=len(encoding.input_ids),
        raw_responses=(_decode_completion(tokenizer, completion),),
    )
    if item.is_dummy:
        return None
    return prediction.to_json_dict(
        split=split,
        rank=rank,
        request_sha256=request_sha256,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )


def run_vllm_thinking_inference(
    config: ExperimentConfig,
    *,
    processed_path: str | Path,
    checkpoint: Any,
    output_dir: str | Path,
    split: str = "test",
    limit: int | None = None,
    sample_count_per_reaction: int = 1,
    portable_adapter: str | Path | None = None,
    local_files_only: bool = True,
    environment_installer: Callable[[], Any] = install_frozen_environment,
    record_loader: Callable[..., tuple[ReactionRecord, ...]] = load_exploration_records,
    tokenizer_loader: Callable[..., Any] = load_pinned_tokenizer,
    adapter_preparer: Callable[..., VllmAdapter] = prepare_vllm_adapter,
    server_factory: Callable[..., Any] = serve_vllm,
    completion_requester: Callable[..., VllmCompletion] = request_completion,
    server_validator: Callable[..., None] = validate_adapter_application,
) -> ThinkingExplorationReceipt:
    """Generate a complete split with eight concurrent, individually durable requests."""

    if split not in {"val", "test"}:
        raise VllmThinkingError(f"thinking split must be val or test, got {split!r}")
    if sample_count_per_reaction != 1:
        raise VllmThinkingError("vLLM thinking runtime is frozen at one candidate per reaction")
    validate_thinking_profile(config.thinking_generation)
    environment_installer()
    records = record_loader(config, processed_path, split)
    selected = select_exploration_records(records, limit=limit)
    adapter = None
    if portable_adapter is not None:
        adapter = adapter_preparer(
            portable_adapter,
            cache_root=config.runtime.local_cache_root,
            checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
        )
    request_payload = _vllm_request_payload(
        config,
        checkpoint,
        split=split,
        records=selected,
        sample_count_per_reaction=sample_count_per_reaction,
        adapter=adapter,
    )
    run_root, request_sha256 = exploration_run_path(output_dir, request_payload)
    if (run_root / ".complete").is_file():
        return validate_exploration_artifact(run_root, request_sha256=request_sha256)
    run_root.mkdir(parents=True, exist_ok=True)

    scheduled = schedule_equal_rank_calls(selected)
    per_rank = len(scheduled) // WORLD_SIZE
    persisted: dict[tuple[int, int], list[dict[str, Any]]] = {}
    missing_seen = False
    for rank in range(WORLD_SIZE):
        rank_items = scheduled[rank * per_rank : (rank + 1) * per_rank]
        for batch_index, item in enumerate(rank_items):
            path = exploration_batch_path(run_root, rank=rank, batch_index=batch_index)
            if path.exists() or path.is_symlink():
                if missing_seen:
                    raise VllmThinkingError("persisted vLLM batches are not one contiguous prefix")
                if path.is_symlink() or not path.is_file():
                    raise VllmThinkingError(f"invalid persisted vLLM batch: {path}")
                rows = _read_jsonl(path)
                _validate_prediction_rows(
                    rows,
                    rank=rank,
                    request_sha256=request_sha256,
                    checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
                    sample_count=1,
                    expected_items=(item,),
                )
                persisted[(rank, batch_index)] = rows
            else:
                missing_seen = True

    expected_batches = len(scheduled)
    if len(persisted) == expected_batches:
        for rank in range(WORLD_SIZE):
            rows = [row for index in range(per_rank) for row in persisted[(rank, index)]]
            write_exploration_fragment(run_root, rank=rank, rows=rows)
        return finalize_exploration_artifact(
            run_root,
            request_payload,
            request_sha256=request_sha256,
        )

    tokenizer = tokenizer_loader(config, local_files_only=local_files_only)
    completed_records = sum(len(rows) for rows in persisted.values())
    with server_factory(
        config,
        request_sha256=request_sha256,
        adapter=adapter,
    ) as endpoint:
        first_encoding = encode_thinking_prompt(
            tokenizer,
            selected[0],
            max_length=config.model.max_sequence_length,
        )
        server_validator(
            endpoint,
            first_encoding.input_ids,
            output_path=endpoint.log_path.parent / "adapter-gate.json",
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=VLLM_REQUEST_CONCURRENCY
        ) as executor:
            for rank in range(WORLD_SIZE):
                rank_rows: list[dict[str, Any]] = []
                rank_items = scheduled[rank * per_rank : (rank + 1) * per_rank]
                batch_index = 0
                while batch_index < per_rank:
                    existing = persisted.get((rank, batch_index))
                    if existing is not None:
                        rank_rows.extend(existing)
                        batch_index += 1
                        continue
                    wave_stop = min(batch_index + VLLM_REQUEST_CONCURRENCY, per_rank)
                    wave = list(enumerate(rank_items[batch_index:wave_stop], start=batch_index))
                    encodings = {
                        index: encode_thinking_prompt(
                            tokenizer,
                            item.record,
                            max_length=config.model.max_sequence_length,
                        )
                        for index, item in wave
                    }
                    futures = {
                        index: executor.submit(
                            completion_requester,
                            endpoint,
                            encodings[index].input_ids,
                            config.thinking_generation,
                            seed=config.seed + rank * per_rank + index,
                            request_id=f"janus-{request_sha256[:12]}-{item.ordinal:05d}",
                        )
                        for index, item in wave
                    }
                    for index, item in wave:
                        completion = futures[index].result()
                        row = _prediction_row(
                            item,
                            encodings[index],
                            completion,
                            tokenizer,
                            split=split,
                            rank=rank,
                            request_sha256=request_sha256,
                            checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
                        )
                        rows = [] if row is None else [row]
                        write_exploration_batch(
                            run_root,
                            rank=rank,
                            batch_index=index,
                            rows=rows,
                        )
                        rank_rows.extend(rows)
                        completed_records += len(rows)
                    print(
                        f"[thinking-vllm:{split}] {completed_records}/{len(selected)}",
                        flush=True,
                    )
                    batch_index = wave_stop
                write_exploration_fragment(run_root, rank=rank, rows=rank_rows)

    return finalize_exploration_artifact(
        run_root,
        request_payload,
        request_sha256=request_sha256,
    )


__all__ = [
    "VLLM_ADAPTER_SCHEMA_VERSION",
    "VLLM_REQUEST_CONCURRENCY",
    "VLLM_RUNTIME_SCHEMA_VERSION",
    "VLLM_VERSION",
    "VllmAdapter",
    "VllmCompletion",
    "VllmEndpoint",
    "VllmThinkingError",
    "prepare_vllm_adapter",
    "request_completion",
    "rewrite_safetensors_lora_prefix",
    "run_vllm_thinking_inference",
    "serve_vllm",
]
