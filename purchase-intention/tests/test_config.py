"""Tests for pipeline configuration loading."""
from pathlib import Path

from purchase_intention.config import PipelineConfig


def test_load_config_from_yaml() -> None:
    """The sample config.yaml must load into a valid PipelineConfig."""
    config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
    config = PipelineConfig.from_yaml(config_path)

    assert config.model.name == "purchase_intention_classifier"
    assert 0 < config.data.test_size < 1