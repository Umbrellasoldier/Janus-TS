from __future__ import annotations

from janus_ts.config import load_config


def test_locked_transition1x_config() -> None:
    config = load_config("configs/transition1x.yaml")
    assert config.seed == 42
    assert config.model.revision == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert config.generation.report_k == (1, 2, 3, 4, 5, 10)
    assert len(config.sha256) == 64
