from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from janus_ts.device_map_runtime import compute_global_target_token_eval_loss


class _LossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.denominators: list[int] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        use_cache: bool,
        num_items_in_batch: torch.Tensor,
    ) -> Any:
        del attention_mask, labels
        assert use_cache is False
        self.denominators.append(int(num_items_in_batch))
        loss = 1.0 if int(input_ids[0, 0]) == 1 else 3.0
        return SimpleNamespace(loss=torch.tensor(loss, device=input_ids.device))


def test_eval_loss_is_normalized_over_all_supervised_target_tokens() -> None:
    model = _LossModel()
    dataset = [
        {
            "input_ids": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "labels": [-100, -100, 3, 4],
        },
        {
            "input_ids": [2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1],
            "labels": [2, 3, 4, 5],
        },
    ]

    def collator(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        feature = features[0]
        return {name: torch.tensor([values]) for name, values in feature.items()}

    loss = compute_global_target_token_eval_loss(
        model,
        dataset,
        collator,
        progress_interval=10,
    )

    assert loss == pytest.approx((1.0 * 2 + 3.0 * 4) / 6)
    assert model.denominators == [2, 4]
