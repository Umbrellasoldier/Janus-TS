from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from janus_ts.gates import (
    PISSA_MIN_MEM_AVAILABLE_KIB,
    GateError,
    PissaParityProbe,
    _assert_pissa_behavioral_parity,
    _pissa_behavioral_metrics,
    _pissa_resource_evidence,
)


class _Tokenizer:
    def __call__(self, _text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }


class _ParityModel(torch.nn.Module):
    def __init__(self, *, adapter: bool, changed_argmax: bool = False) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2, dtype=torch.bfloat16)
        self.changed_argmax = changed_argmax
        if adapter:
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(1, 2, bias=False, dtype=torch.float32)}
            )
            torch.nn.init.zeros_(self.lora_B["default"].weight)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids, attention_mask, use_cache):
        assert attention_mask.shape == input_ids.shape
        assert use_cache is False
        hidden = self.embedding(input_ids[:, -1:])
        adapter = getattr(self, "lora_A", None)
        if adapter is not None:
            hidden = hidden + self.lora_B["default"](self.lora_A["default"](hidden))
        logits = torch.zeros((*hidden.shape[:-1], 4), dtype=hidden.dtype, device=hidden.device)
        if self.changed_argmax:
            logits[..., 1] = 1
        return SimpleNamespace(logits=logits)


def test_pissa_parity_uses_bf16_autocast_and_observes_adapter_linears() -> None:
    probe = PissaParityProbe(_Tokenizer())
    original = probe("original_base", _ParityModel(adapter=False))
    initialized = probe("pissa_initialized", _ParityModel(adapter=True))

    assert original["autocast_device_type"] == "cpu"
    assert original["autocast_dtype"] == "torch.bfloat16"
    assert original["logits_dtype_before_cpu_cast"] == "torch.bfloat16"
    assert initialized["lora_output_dtypes"] == {
        "lora_A": "torch.bfloat16",
        "lora_B": "torch.bfloat16",
    }
    assert initialized["argmax_equal"] is True


def test_pissa_parity_still_fails_closed_on_changed_argmax() -> None:
    probe = PissaParityProbe(_Tokenizer())
    probe("original_base", _ParityModel(adapter=False))

    with pytest.raises(GateError, match="behavioral parity failed.*argmax changed"):
        probe("pissa_initialized", _ParityModel(adapter=True, changed_argmax=True))


def test_pissa_behavioral_parity_is_invariant_to_global_logit_shift() -> None:
    reference = torch.tensor([-4.0, -1.0, 0.5, 3.0, 1.0])
    metrics = _pissa_behavioral_metrics(reference, reference + 2.0)

    _assert_pissa_behavioral_parity(metrics, stage="test")
    assert metrics["argmax_equal"] is True
    assert metrics["total_variation"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["jensen_shannon"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["centered_nrmse"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["mean_abs"] == pytest.approx(2.0)


def test_pissa_behavioral_parity_rejects_distribution_and_ranking_drift() -> None:
    reference = torch.linspace(-4.0, 4.0, 64)
    changed = reference.flip(0)
    metrics = _pissa_behavioral_metrics(reference, changed)

    assert metrics["max_probability_delta"] <= metrics["total_variation"] + 1e-7
    with pytest.raises(GateError, match="argmax changed.*top-k overlap"):
        _assert_pissa_behavioral_parity(metrics, stage="test")


def test_pissa_behavioral_metrics_reject_nonfinite_or_shape_drift() -> None:
    with pytest.raises(GateError, match="invalid shapes"):
        _pissa_behavioral_metrics(torch.ones(4), torch.ones(3))
    with pytest.raises(GateError, match="NaN or infinity"):
        _pissa_behavioral_metrics(torch.ones(4), torch.tensor([1.0, 2.0, 3.0, float("nan")]))


def test_pissa_serialization_records_swap_without_applying_compute_limit() -> None:
    before = SimpleNamespace(
        mem_available_kib=PISSA_MIN_MEM_AVAILABLE_KIB + 1024,
        swap_free_kib=4 * 1024 * 1024,
        to_dict=lambda: {"mem_available_kib": PISSA_MIN_MEM_AVAILABLE_KIB + 1024},
    )
    after = SimpleNamespace(
        mem_available_kib=PISSA_MIN_MEM_AVAILABLE_KIB,
        swap_free_kib=512 * 1024,
        to_dict=lambda: {"mem_available_kib": PISSA_MIN_MEM_AVAILABLE_KIB},
    )

    evidence = _pissa_resource_evidence(before, after, scope="full_serialization")

    assert evidence["observed_swap_growth_kib"] == 3584 * 1024
    assert evidence["swap_growth_is_diagnostic"] is True
    assert evidence["other_gpu_phases_max_swap_growth_kib"] == 256 * 1024


def test_pissa_serialization_still_rejects_low_available_memory() -> None:
    low = SimpleNamespace(
        mem_available_kib=PISSA_MIN_MEM_AVAILABLE_KIB - 1,
        swap_free_kib=1024,
        to_dict=lambda: {},
    )

    with pytest.raises(GateError, match="MemAvailable"):
        _pissa_resource_evidence(low, low, scope="cached_payload_verification")
