from __future__ import annotations

from typing import Any

from benchrep.assembly.schemas.training_config_schema import TrainingConfig
from benchrep.assembly.schemas.prediction_config_schema import PredictionConfig
from benchrep.assembly.schemas.evaluation_config_schema import EvaluationConfig


def parse_training_config(
        raw_config: dict[str, Any],
        *,
        model_overridden: bool = False,
        datamodule_overridden: bool = False,
) -> TrainingConfig:
    """Validate a raw config dictionary and return a typed training config."""
    return TrainingConfig.model_validate(
        raw_config,
        context={
            "model_overridden": model_overridden,
            "datamodule_overridden": datamodule_overridden,
        }
    )


def parse_prediction_config(
        raw_config: dict[str, Any],
        *,
        training_manifest_path_overridden: bool = False,
) -> PredictionConfig:
    """Validate a raw config dictionary and return a typed prediction config."""
    return PredictionConfig.model_validate(
        raw_config,
        context={
            "training_manifest_path_overridden": training_manifest_path_overridden,
        },
    )


def parse_evaluation_config(raw_config: dict[str, Any]) -> EvaluationConfig:
    """Validate a raw config dictionary and return a typed evaluation config."""
    return EvaluationConfig.model_validate(raw_config)