from __future__ import annotations

from typing import Any

from benchrep.assembly.schemas.training_config_schema import TrainingConfig


def parse_training_config(raw_config: dict[str, Any]) -> TrainingConfig:
    """Validate a raw config dictionary and return a typed training config."""
    return TrainingConfig.model_validate(raw_config)