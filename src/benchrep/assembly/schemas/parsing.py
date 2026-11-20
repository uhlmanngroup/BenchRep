from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from benchrep.assembly.schemas.training_config_schema import TrainingConfig
from benchrep.assembly.schemas.prediction_config_schema import PredictionConfig
from benchrep.assembly.schemas.evaluation_config_schema import EvaluationConfig
from benchrep.interfaces.model_families import ModelFamilySpec


@dataclass(frozen=True, kw_only=True)
class TrainingConfigContext:
    """Runtime information needed to validate a training config.

    Attributes
    ----------
    external_model : bool
        Whether a whole model will be supplied through runtime overrides.
        Allows omission of sections needed only for a config-built model.
    external_datamodule : bool
        Whether a datamodule will be supplied through runtime overrides.
        Allows omission of sections needed only for a config-built datamodule.

    Notes
    -----
    All fields are required. Use False for components built from config.
    Non-null declarations under the config's `overrides` section also establish
    external-component intent. This context supplies no Python objects and
    does not check their runtime compatibility.
    """

    external_model: bool
    external_datamodule: bool


@dataclass(frozen=True, kw_only=True)
class PredictionConfigContext:
    """Runtime information needed to validate a prediction config.

    Attributes
    ----------
    model_family : ModelFamilySpec
        Family selected for prediction. Use AUTOENCODER_FAMILY, VAE_FAMILY,
        COMPOSITE_FAMILY, or YOYO_FAMILY from `benchrep.interfaces.model_families`.
        The YOYO family permits a missing training manifest and a null checkpoint.
    training_manifest_path_overridden : bool
        Whether `PredictionOverrides.training_manifest_path` will supply the
        training manifest separately. Use False when the path is configured
        in `source`, or when no manifest is used for a YOYO model.

    Notes
    -----
    All fields are required. This context supplies no model, datamodule, or
    manifest. Resolution checks the actual runtime overrides and manifest
    compatibility, including that a linked manifest belongs to the same family.
    """

    model_family: ModelFamilySpec
    training_manifest_path_overridden: bool


@dataclass(frozen=True, kw_only=True)
class EvaluationConfigContext:
    """Runtime information needed to validate an evaluation config.

    Attributes
    ----------
    prediction_manifest_path_overridden : bool
        Whether `EvaluationOverrides.prediction_manifest_path` will supply
        the source manifest separately. Use False when sources are configured
        in the config itself.

    Notes
    -----
    The field is required. This context does not supply or load the manifest.
    """

    prediction_manifest_path_overridden: bool


ConfigContext: TypeAlias = (
    TrainingConfigContext | PredictionConfigContext | EvaluationConfigContext
)


def _validation_context(
    schema: type[TrainingConfig] | type[PredictionConfig] | type[EvaluationConfig],
    context: ConfigContext,
) -> dict[str, Any]:
    """Check the config/context pairing and translate to Pydantic's context keys."""
    expected: (
        type[TrainingConfigContext]
        | type[PredictionConfigContext]
        | type[EvaluationConfigContext]
    )
    if issubclass(schema, TrainingConfig):
        expected = TrainingConfigContext
    elif issubclass(schema, PredictionConfig):
        expected = PredictionConfigContext
    elif issubclass(schema, EvaluationConfig):
        expected = EvaluationConfigContext
    else:
        raise TypeError(f"Unsupported config schema: {schema.__name__}.")

    if not isinstance(context, expected):
        raise TypeError(
            f"{schema.__name__} requires {expected.__name__}; "
            f"got {type(context).__name__}."
        )

    if isinstance(context, TrainingConfigContext):
        return {
            "model_is_external": context.external_model,
            "datamodule_overridden": context.external_datamodule,
        }
    if isinstance(context, PredictionConfigContext):
        return {
            "model_family_name": context.model_family.name,
            "training_manifest_path_overridden": context.training_manifest_path_overridden,
        }
    return {
        "prediction_manifest_path_overridden": context.prediction_manifest_path_overridden,
    }


def parse_training_config(
    raw_config: dict[str, Any],
    *,
    context: TrainingConfigContext,
) -> TrainingConfig:
    """Validate a raw training configuration with runtime-override context.

    Parameters
    ----------
    raw_config:
        Training configuration mapping to validate.
    context:
        Required training context describing external components. The same
        context can be reused with `update_config()`.

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
        context=_validation_context(TrainingConfig, context),
    )


def parse_prediction_config(
    raw_config: dict[str, Any],
    *,
    context: PredictionConfigContext,
) -> PredictionConfig:
    """Validate a raw prediction configuration with workflow-override context.

    Parameters
    ----------
    raw_config:
        Prediction configuration mapping to validate.
    context:
        Required prediction context identifying the family and whether the
        training manifest is supplied separately. The same context can be
        reused with `update_config()`.

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
        context=_validation_context(PredictionConfig, context),
    )


def parse_evaluation_config(
    raw_config: dict[str, Any],
    *,
    context: EvaluationConfigContext,
) -> EvaluationConfig:
    """Validate a raw evaluation configuration with manifest-override context.

    Parameters
    ----------
    raw_config:
        Evaluation configuration mapping to validate.
    context:
        Required evaluation context describing whether the prediction manifest
        is supplied separately. The same context can be reused with
        `update_config()`.

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
        context=_validation_context(EvaluationConfig, context),
    )
