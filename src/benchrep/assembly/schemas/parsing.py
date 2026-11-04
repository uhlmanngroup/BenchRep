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
    """Validate a raw training configuration with runtime-override context.

    Parameters
    ----------
    raw_config:
        Training configuration mapping to validate.
    model_overridden:
        Whether an external model is supplied through Python. Allows
        omission of sections required only for a config-built model.
    datamodule_overridden:
        Whether an external datamodule is supplied through Python. Allows
        omission of sections required only for a config-built datamodule.

    Returns
    -------
    TrainingConfig
        Schema-validated training configuration.

    Notes
    -----
    Non-null declarations under `overrides` can also establish external
    component intent during schema validation. These flags provide context;
    they do not supply objects or verify their runtime compatibility.

    This helper performs schema validation, including validators defined
    by the schema. It does not replace workflow resolution, component
    construction, or runtime validation.
    """
    return TrainingConfig.model_validate(
        raw_config,
        context={
            "model_is_external": model_overridden,
            "datamodule_overridden": datamodule_overridden,
        }
    )


def parse_prediction_config(
    raw_config: dict[str, Any],
    *,
    model_overridden: bool = False,
    datamodule_overridden: bool = False,
    training_manifest_path_overridden: bool = False,
) -> PredictionConfig:
    """Validate a raw prediction configuration with workflow-override context.

    Parameters
    ----------
    raw_config:
        Prediction configuration mapping to validate.
    model_overridden:
        External-model context passed to schema validation. Actual model
        requirements and compatibility are checked during resolution.
    datamodule_overridden:
        External-datamodule context passed to schema validation. Actual
        datamodule requirements are checked during resolution.
    training_manifest_path_overridden:
        Whether a training-manifest path is supplied through
        `PredictionOverrides.training_manifest_path`. Allows omission of
        `source.training_manifest_path` from the configuration.

    Returns
    -------
    PredictionConfig
        Schema-validated prediction configuration.

    Notes
    -----
    Context flags do not supply Python components or a manifest path.
    The workflow must supply those values separately.

    This helper does not load the manifest or checkpoint, resolve inherited
    settings, construct components, or perform prediction runtime checks.
    """
    return PredictionConfig.model_validate(
        raw_config,
        context={
            "model_is_external": model_overridden,
            "datamodule_overridden": datamodule_overridden,
            "training_manifest_path_overridden": (
                training_manifest_path_overridden
            ),
        },
    )


def parse_evaluation_config(
    raw_config: dict[str, Any],
    *,
    prediction_manifest_path_overridden: bool = False,
) -> EvaluationConfig:
    """Validate a raw evaluation configuration with manifest-override context.

    Parameters
    ----------
    raw_config:
        Evaluation configuration mapping to validate.
    prediction_manifest_path_overridden:
        Whether a prediction-manifest path is supplied through
        `EvaluationOverrides.prediction_manifest_path`. Allows schema
        validation when the configuration itself provides no source paths.

    Returns
    -------
    EvaluationConfig
        Schema-validated evaluation configuration.

    Notes
    -----
    The context flag does not supply or load the manifest. The workflow
    must supply the actual path separately.

    This helper does not inspect source artifacts, resolve step enablement,
    validate backend dependencies, or execute evaluation.
    """
    return EvaluationConfig.model_validate(
        raw_config,
        context={
            "prediction_manifest_path_overridden": (
                prediction_manifest_path_overridden
            ),
        },
    )