from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from janus_ts.continuity_gates import (
    ContinuityGateError,
    _run_full_length_generation_stress,
    assert_logits_parity,
    compare_resume_evidence,
    execute_restart_sequence,
    make_resume_gate_identity,
    select_longest_canonical_prompt,
    stable_state_sha256,
    stream_lora_tensor_hashes,
    validate_generation_smoke_evidence,
)


class _Coordinator:
    rank = 0
    world_size = 2

    def broadcast(self, value, *, source=0):
        assert source == 0
        return value


class _TinyLora(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
        self.lora_B = torch.nn.Linear(1, 2, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.lora_A.weight.copy_(torch.tensor([[1.0, 2.0]]))
            self.lora_B.weight.copy_(torch.tensor([[3.0], [4.0]]))


def test_streaming_lora_hash_is_ordered_bitwise_and_does_not_need_state_dict():
    model = _TinyLora()
    engine = SimpleNamespace(module=model)
    coordinator = _Coordinator()

    first = stream_lora_tensor_hashes(
        engine,
        coordinator,
        expected_trainable_parameters=4,
        gather_context=lambda _parameter: nullcontext(),
    )
    second = stream_lora_tensor_hashes(
        engine,
        coordinator,
        expected_trainable_parameters=4,
        gather_context=lambda _parameter: nullcontext(),
    )
    assert first == second
    assert first["tensor_count"] == 2
    assert [entry["name"] for entry in first["tensors"]] == [
        "lora_A.weight",
        "lora_B.weight",
    ]

    with torch.no_grad():
        model.lora_B.weight[0, 0].add_(1e-5)
    changed = stream_lora_tensor_hashes(
        engine,
        coordinator,
        expected_trainable_parameters=4,
        gather_context=lambda _parameter: nullcontext(),
    )
    assert changed["aggregate_sha256"] != first["aggregate_sha256"]
    assert changed["tensors"][0] == first["tensors"][0]
    assert changed["tensors"][1]["sha256"] != first["tensors"][1]["sha256"]


def test_stable_state_hash_is_mapping_order_independent_and_tensor_bitwise():
    left = {"step": 2, "values": torch.tensor([1.0, 2.0]), "nested": [None, True]}
    right = {"nested": [None, True], "values": torch.tensor([1.0, 2.0]), "step": 2}
    assert stable_state_sha256(left) == stable_state_sha256(right)
    right["values"][1] = 3.0
    assert stable_state_sha256(left) != stable_state_sha256(right)


def test_restart_sequence_enforces_release_before_fresh_engine():
    events = []

    def continuous():
        events.append("continuous")
        return {"checkpoint": "step1"}

    def release(branch):
        assert branch["checkpoint"] == "step1"
        events.append("release")

    def resumed(branch):
        assert events == ["continuous", "release"]
        events.append("fresh-resume")
        return {"checkpoint": branch["checkpoint"], "step": 2}

    def compare(left, right):
        events.append("compare")
        return left["checkpoint"], right["step"]

    assert execute_restart_sequence(continuous, release, resumed, compare) == ("step1", 2)
    assert events == ["continuous", "release", "fresh-resume", "compare"]


def test_resume_evidence_requires_rng_scheduler_step_and_all_lora_hashes():
    lora = {
        "aggregate_sha256": "a" * 64,
        "tensor_count": 2,
        "logical_parameters": 4,
        "tensors": [
            {"name": "lora_A.weight", "sha256": "b" * 64},
            {"name": "lora_B.weight", "sha256": "c" * 64},
        ],
    }
    evidence = {
        "rank": 0,
        "global_step": 2,
        "engine_global_step": 2,
        "scheduler_sha256": "d" * 64,
        "rng_sha256": "e" * 64,
        "lora_stream": lora,
    }
    comparison = compare_resume_evidence(evidence, dict(evidence))
    assert comparison["status"] == "pass"
    assert comparison["lora_aggregate_sha256"] == "a" * 64

    bad = dict(evidence)
    bad["rng_sha256"] = "f" * 64
    with pytest.raises(ContinuityGateError, match="rng_sha256"):
        compare_resume_evidence(evidence, bad)

    bad = dict(evidence)
    bad["lora_stream"] = {**lora, "aggregate_sha256": "0" * 64}
    with pytest.raises(ContinuityGateError, match="lora_stream"):
        compare_resume_evidence(evidence, bad)


def test_resume_identity_binds_config_data_and_bundle_content():
    identity = make_resume_gate_identity(
        config_sha256="1" * 64,
        source_fingerprint="2" * 64,
        bundle_manifest_sha256="3" * 64,
    )
    again = make_resume_gate_identity(
        config_sha256="1" * 64,
        source_fingerprint="2" * 64,
        bundle_manifest_sha256="3" * 64,
    )
    changed = make_resume_gate_identity(
        config_sha256="1" * 64,
        source_fingerprint="2" * 64,
        bundle_manifest_sha256="4" * 64,
    )
    assert identity == again
    assert identity.run_fingerprint != changed.run_fingerprint
    assert identity.model_fingerprint != changed.model_fingerprint
    with pytest.raises(ContinuityGateError, match="SHA256"):
        make_resume_gate_identity(
            config_sha256="short",
            source_fingerprint="2" * 64,
            bundle_manifest_sha256="3" * 64,
        )


def test_logit_parity_checks_finite_argmax_and_frozen_tolerance():
    residual = torch.tensor([0.0, 3.0, -2.0])
    portable = torch.tensor([0.01, 3.02, -2.01])
    report = assert_logits_parity(residual, portable)
    assert report["status"] == "pass"
    assert report["argmax_token_id"] == 1
    assert report["atol"] == 0.125
    assert report["rtol"] == 0.02

    with pytest.raises(ContinuityGateError, match="argmax"):
        assert_logits_parity(residual, torch.tensor([4.0, 3.0, -2.0]))
    with pytest.raises(ContinuityGateError, match="NaN"):
        assert_logits_parity(residual, torch.tensor([0.0, float("nan"), -2.0]))


def test_longest_prompt_selection_is_legal_and_stable_on_ties():
    records = ["short", "first-long", "second-long"]
    lengths = {"short": 5, "first-long": 11, "second-long": 11}

    def encode(record):
        return SimpleNamespace(input_ids=tuple(range(lengths[record])))

    ordinal, record, prompt = select_longest_canonical_prompt(records, encode)
    assert (ordinal, record, len(prompt.input_ids)) == (1, "first-long", 11)

    with pytest.raises(ContinuityGateError, match="empty"):
        select_longest_canonical_prompt([], encode)


def _generation_smoke_payload() -> dict:
    shape = [10, 2400]
    ranks = [
        {
            "rank": rank,
            "local_rank": rank,
            "peak_allocated_mib": 42000,
            "peak_reserved_mib": 43000,
            "device_memory_used_mib": 43500,
            "effective_peak_mib": 43500,
            "output_shape": shape,
            "parsed_beams": 10,
            "valid_parses": 3,
            "stress_output_shape": [10, 2400],
            "stress_actual_new_tokens": 512,
        }
        for rank in (0, 1)
    ]
    return {
        "status": "pass",
        "reaction_id": "rxn-1",
        "ordinal": 17,
        "prompt_sha256": "a" * 64,
        "prompt_tokens": 1888,
        "num_beams": 10,
        "num_return_sequences": 10,
        "max_new_tokens": 512,
        "output_shape": shape,
        "parsed_beams": 10,
        "valid_parses": 3,
        "swap_growth_kib": 128,
        "max_swap_growth_kib": 256 * 1024,
        "max_device_memory_mib": 45056,
        "stress": {
            "status": "pass",
            "purpose": "non-formal-worst-case-memory-only",
            "num_beams": 10,
            "num_return_sequences": 10,
            "min_new_tokens": 512,
            "max_new_tokens": 512,
            "actual_new_tokens": 512,
            "output_shape": [10, 2400],
        },
        "ranks": ranks,
    }


def test_generation_smoke_schema_locks_beam512_memory_and_parse_evidence():
    payload = _generation_smoke_payload()
    validate_generation_smoke_evidence(payload)

    payload["max_new_tokens"] = 511
    with pytest.raises(ContinuityGateError, match="decoding"):
        validate_generation_smoke_evidence(payload)

    payload = _generation_smoke_payload()
    payload["ranks"][1]["effective_peak_mib"] = 45057
    with pytest.raises(ContinuityGateError, match="rank failed"):
        validate_generation_smoke_evidence(payload)

    payload = _generation_smoke_payload()
    payload["swap_growth_kib"] = 256 * 1024 + 1
    with pytest.raises(ContinuityGateError, match="swap"):
        validate_generation_smoke_evidence(payload)

    payload = _generation_smoke_payload()
    payload["stress"]["actual_new_tokens"] = 511
    with pytest.raises(ContinuityGateError, match="full-length stress"):
        validate_generation_smoke_evidence(payload)


def test_full_length_stress_derives_formal_kwargs_and_forces_512_on_cpu():
    class FakeModel:
        def __init__(self):
            self.kwargs = None

        def generate(self, *, input_ids, attention_mask, **kwargs):
            self.kwargs = kwargs
            assert torch.equal(attention_mask, torch.ones_like(input_ids))
            prefix = input_ids.expand(10, -1)
            continuation = torch.full((10, 512), 7, dtype=torch.long)
            return torch.cat((prefix, continuation), dim=1)

    model = FakeModel()
    prompt = SimpleNamespace(input_ids=(1, 2, 3))
    report = _run_full_length_generation_stress(
        model,
        prompt,
        local_rank=0,
        torch_module=torch,
        device=torch.device("cpu"),
    )

    assert report["purpose"] == "non-formal-worst-case-memory-only"
    assert report["actual_new_tokens"] == 512
    assert report["output_shape"] == [10, 515]
    assert model.kwargs["num_beams"] == 10
    assert model.kwargs["num_return_sequences"] == 10
    assert model.kwargs["min_new_tokens"] == 512
    assert model.kwargs["max_new_tokens"] == 512
    assert model.kwargs["synced_gpus"] is True
