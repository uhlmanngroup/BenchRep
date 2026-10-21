from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    Annotated,
    TypeAlias,
    Literal,
    TypeVar,
    Generic,
    Mapping,
    overload,
    get_args,
    get_origin,
    Union,
)

import yaml

from pydantic import BaseModel

from benchrep.utils.workflow_utils import (
    TrainingConfigInput,
    PredictionConfigInput,
    EvaluationConfigInput,
    route_config_input,
)
from benchrep.assembly.schemas.training_config_schema import (
    TrainingConfig,
    TrainingRunConfig,
    TrainingReproducibilityConfig,
    TrainingModelConfig,
    TrainingEncoderConfig,
    TrainingDecoderConfig,
    SupportedLossRole,
    TrainingLossTermConfig,
    TrainingOptimizerConfig,
    TrainingTransformPipelineConfig,
    SupportedDatasetConfig,
    TrainingDataModuleConfig,
    TrainingTrainerConfig,
    TrainingLoggerConfig,
    TrainingCheckpointConfig,
    TrainingEarlyStoppingConfig,
    TrainingAdditionalCallbackConfig,
    TrainingInspectionConfig,
)
from benchrep.assembly.schemas.composite_model_config_schema import (
    CompositeModelAssemblyStepConfig,
    CompositeModelComponentConfig,
    CompositeModelDeclarationsConfig,
)
from benchrep.assembly.schemas.prediction_config_schema import (
    PredictionConfig,
    PredictionSourceConfig,
    PredictionDataConfig,
    PredictionInferenceConfig,
    PredictionTransformPipelineConfig,
    PredictionExportConfig,
)
from benchrep.assembly.schemas.evaluation_config_schema import (
    EvaluationConfig,
    EvaluationSourceConfig,
    EvaluationRunConfig,
    EvaluationReductionsConfig,
    EvaluationClusteringConfig,
    EvaluationMetricsConfig,
    EvaluationReconstructionConfig,
    EvaluationPlotsConfig,
)
from benchrep.assembly.schemas.parsing import (
    parse_training_config,
    parse_prediction_config,
    parse_evaluation_config,
)
from benchrep.assembly.schemas.runtime_override_config_schema import (
    RuntimeOverridesConfig,
)


SUPPORTED_CONFIG_SCHEMAS = (TrainingConfig, PredictionConfig, EvaluationConfig)
SupportedConfigType: TypeAlias = TrainingConfig | PredictionConfig | EvaluationConfig
ConfigT = TypeVar("ConfigT", bound=SupportedConfigType)

ConfigSource: TypeAlias = Literal[
    "yaml",
    "config_object",
    "components",
    "yaml_with_components",
    "config_object_with_components",
]

LossesConfig: TypeAlias = dict[
    SupportedLossRole,
    list[TrainingLossTermConfig],
]

CompositeModelComponentsConfig: TypeAlias = dict[
    str,
    CompositeModelComponentConfig,
]

CompositeModelAssemblyConfig: TypeAlias = dict[
    str,
    CompositeModelAssemblyStepConfig,
]

TrainingTransformPipelinesConfig: TypeAlias = list[
    TrainingTransformPipelineConfig
]
AdditionalCallbacksConfig: TypeAlias = list[TrainingAdditionalCallbackConfig]
PredictionTransformPipelinesConfig: TypeAlias = list[
    PredictionTransformPipelineConfig
]

SupportedTrainingConfigComponent: TypeAlias = (
        RuntimeOverridesConfig
        | TrainingRunConfig
        | TrainingReproducibilityConfig
        | TrainingModelConfig
        | TrainingEncoderConfig
        | TrainingDecoderConfig
        | CompositeModelDeclarationsConfig
        | CompositeModelComponentsConfig
        | CompositeModelAssemblyConfig
        | LossesConfig
        | TrainingOptimizerConfig
        | TrainingTransformPipelinesConfig
        | SupportedDatasetConfig
        | TrainingDataModuleConfig
        | TrainingTrainerConfig
        | TrainingLoggerConfig
        | TrainingCheckpointConfig
        | TrainingEarlyStoppingConfig
        | AdditionalCallbacksConfig
        | TrainingInspectionConfig
)

SupportedPredictionConfigComponent: TypeAlias = (
        RuntimeOverridesConfig
        | PredictionSourceConfig
        | SupportedDatasetConfig
        | PredictionDataConfig
        | PredictionInferenceConfig
        | PredictionTransformPipelinesConfig
        | PredictionExportConfig
)

SupportedEvaluationConfigComponent: TypeAlias = (
        EvaluationSourceConfig
        | EvaluationRunConfig
        | EvaluationReductionsConfig
        | EvaluationClusteringConfig
        | EvaluationMetricsConfig
        | EvaluationReconstructionConfig
        | EvaluationPlotsConfig
)

SupportedConfigComponent: TypeAlias = (
    SupportedTrainingConfigComponent
    | SupportedPredictionConfigComponent
    | SupportedEvaluationConfigComponent
)


@dataclass(frozen=True)
class ConfigCompositionResult(Generic[ConfigT]):
    original_config_path: Path | None
    original_config_raw: dict[str, Any] | None
    effective_config: ConfigT
    effective_source: ConfigSource
    yaml_supplied: bool
    yaml_used_as_base: bool
    composition_messages: tuple[str, ...] = ()
    composition_warnings: tuple[str, ...] = ()


@overload
def compose_effective_config(
    *,
    schema: type[TrainingConfig],
    config: TrainingConfigInput = None,
    config_components: Mapping[str, SupportedTrainingConfigComponent] | None = None,
    external_model: bool = False,
    external_datamodule: bool = False,
    training_manifest_path_overridden: bool = False,
    prediction_manifest_path_overridden: bool = False,
) -> ConfigCompositionResult[TrainingConfig]: ...


@overload
def compose_effective_config(
    *,
    schema: type[PredictionConfig],
    config: PredictionConfigInput = None,
    config_components: Mapping[str, SupportedPredictionConfigComponent] | None = None,
    external_model: bool = False,
    external_datamodule: bool = False,
    training_manifest_path_overridden: bool = False,
    prediction_manifest_path_overridden: bool = False,
) -> ConfigCompositionResult[PredictionConfig]: ...


@overload
def compose_effective_config(
    *,
    schema: type[EvaluationConfig],
    config: EvaluationConfigInput = None,
    config_components: Mapping[str, SupportedEvaluationConfigComponent] | None = None,
    external_model: bool = False,
    external_datamodule: bool = False,
    training_manifest_path_overridden: bool = False,
    prediction_manifest_path_overridden: bool = False,
) -> ConfigCompositionResult[EvaluationConfig]: ...


def compose_effective_config(
    *,
    schema: type[ConfigT],
    config: ConfigT | Path | str | None = None,
    config_components: Mapping[str, Any] | None = None,
    external_model: bool = False,
    external_datamodule: bool = False,
    training_manifest_path_overridden: bool = False,
    prediction_manifest_path_overridden: bool = False,
) -> ConfigCompositionResult[ConfigT]:
    """Compose a typed workflow config from a base and section overrides.

    ``config`` may be a YAML path or a complete typed config object.
    ``config_components`` replaces matching top-level sections of either
    base. Each component replaces its whole section, not individual fields.
    The combined config is validated before workflow-specific resolution.

    If no ``config`` is supplied, the components must provide enough sections
    to construct a valid config. A typed config object without components is
    returned directly. For YAML input, the original path and payload are
    retained for provenance.

    Runtime model, datamodule, and manifest override flags provide validation
    context for the relevant workflow schema.

    Parameters
    ----------
    schema
        TrainingConfig, PredictionConfig, or EvaluationConfig.
    config
        Optional YAML path or complete typed config object used as the base.
    config_components
        Optional mapping from top-level section names to typed section objects.
    external_model
        Whether a runtime model object was supplied.
    external_datamodule
        Whether a runtime datamodule object was supplied.
    training_manifest_path_overridden
        Whether a training manifest path was supplied outside prediction config.
    prediction_manifest_path_overridden
        Whether a prediction manifest path was supplied outside evaluation config.

    Returns
    -------
    ConfigCompositionResult[ConfigT]
        The effective config, composition metadata, and original YAML details
        when the input was a path.

    Raises
    ------
    ValueError
        If no config source is supplied or a component key is unsupported.
    TypeError
        If the schema, config input, or a component type is unsupported.
    """
    composition_messages: list[str] = []
    composition_warnings: list[str] = []

    if schema not in SUPPORTED_CONFIG_SCHEMAS:
        raise TypeError(f"Unsupported config schema: {schema.__name__}")

    config_path, full_config_object = route_config_input(config, schema)

    # Load original YAML, if supplied and keep for records regardless.
    if config_path is not None:
        original_config_path = Path(config_path).resolve()
        original_config_raw = load_yaml(original_config_path)
        yaml_supplied = True
    else:
        original_config_path = None
        original_config_raw = None
        yaml_supplied = False

    if full_config_object is None and not config_components and not yaml_supplied:
        raise ValueError(
            "At least one of config or config_components is required."
        )

    component_raw = normalize_config_components(
        schema=schema,
        config_components=config_components,
    )

    if full_config_object is not None:
        yaml_used_as_base = False

        if component_raw:
            config_source: ConfigSource = "config_object_with_components"
            raw_config = full_config_object.model_dump(mode="python")
            raw_config.update(component_raw)
            effective_config = _parse_effective_config(
                schema=schema,
                raw_config=raw_config,
                external_model=external_model,
                external_datamodule=external_datamodule,
                training_manifest_path_overridden=training_manifest_path_overridden,
                prediction_manifest_path_overridden=prediction_manifest_path_overridden,
            )
            composition_messages.append(
                "config_components replace matching top-level sections "
                "of the config object."
            )
        else:
            config_source = "config_object"
            effective_config = full_config_object

        return ConfigCompositionResult(
            original_config_path=original_config_path,
            original_config_raw=original_config_raw,
            effective_config=effective_config,
            effective_source=config_source,
            yaml_supplied=yaml_supplied,
            yaml_used_as_base=yaml_used_as_base,
            composition_messages=tuple(composition_messages),
            composition_warnings=tuple(composition_warnings),
        )

    if yaml_supplied and component_raw:
        assert original_config_raw is not None

        config_source = "yaml_with_components"
        yaml_used_as_base = True
        raw_config = dict(original_config_raw)
        raw_config.update(component_raw)

        composition_messages.append(
            "Both config_path and config_components provided; config_components "
            "will replace matching top-level YAML sections."
        )

    elif yaml_supplied:
        assert original_config_raw is not None

        config_source = "yaml"
        yaml_used_as_base = True
        raw_config = original_config_raw

        composition_messages.append("Only config_path provided; using YAML as effective config.")

    else:
        config_source = "components"
        yaml_used_as_base = False
        raw_config = component_raw

        composition_messages.append(
            "Only config_components provided; composing effective config from "
            "top-level components. Components must provide enough information to "
            f"validate as {schema.__name__}; schema defaults may fill omitted optional fields."
        )

    effective_config = _parse_effective_config(
        schema=schema,
        raw_config=raw_config,
        external_model=external_model,
        external_datamodule=external_datamodule,
        training_manifest_path_overridden=training_manifest_path_overridden,
        prediction_manifest_path_overridden=prediction_manifest_path_overridden,
    )

    return ConfigCompositionResult(
        original_config_path=original_config_path,
        original_config_raw=original_config_raw,
        effective_config=effective_config,
        effective_source=config_source,
        yaml_supplied=yaml_supplied,
        yaml_used_as_base=yaml_used_as_base,
        composition_messages=tuple(composition_messages),
        composition_warnings=tuple(composition_warnings),
    )


def load_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """Load a YAML config or manifest file as a dictionary.

    Parameters
    ----------
    yaml_path:
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed dictionary.
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")

    if not yaml_path.is_file():
        raise ValueError(f"YAML path must point to a file, got: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        yaml_file = yaml.safe_load(file)

    if yaml_file is None:
        raise ValueError(f"YAML file is empty: {yaml_path}")

    if not isinstance(yaml_file, dict):
        raise TypeError(
            f"YAML file must define a YAML mapping/dictionary at the top level, "
            f"got {type(yaml_file).__name__}."
        )

    return yaml_file


def _strip_none_from_type(annotation: Any) -> Any:
    origin = get_origin(annotation)

    if origin in {Union, UnionType}:
        args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
        if len(args) == 1:
            return args[0]

    return annotation


def _get_runtime_types(annotation: Any) -> tuple[type, ...]:
    origin = get_origin(annotation)

    if origin is Annotated:
        return _get_runtime_types(get_args(annotation)[0])

    if origin in {Union, UnionType}:
        runtime_types: list[type] = []

        for argument in get_args(annotation):
            if argument is not type(None):
                runtime_types.extend(_get_runtime_types(argument))

        return tuple(runtime_types)

    if isinstance(annotation, type):
        return (annotation,)

    return ()


def get_expected_component_types(
    schema: type[BaseModel],
    *,
    exclude_fields: set[str] | None = None,
) -> dict[str, Any]:
    if exclude_fields is None:
        exclude_fields = {"stage"}

    return {
        name: _strip_none_from_type(field.annotation)
        for name, field in schema.model_fields.items()
        if name not in exclude_fields
    }


def normalize_config_components(
    *,
    schema: type[SupportedConfigType],
    config_components: Mapping[str, SupportedConfigComponent] | None,
) -> dict[str, Any]:
    if not config_components:
        return {}

    if schema not in SUPPORTED_CONFIG_SCHEMAS:
        raise TypeError(f"Unsupported config schema: {schema.__name__}")

    expected_component_types = get_expected_component_types(schema)
    allowed_keys = set(expected_component_types)

    unknown_keys = set(config_components) - allowed_keys
    if unknown_keys:
        raise ValueError(
            f"Unsupported config component keys for {schema.__name__}: "
            f"{sorted(unknown_keys)}. Expected keys: {sorted(allowed_keys)}."
        )

    normalized: dict[str, Any] = {}

    for key, component in config_components.items():
        if schema is TrainingConfig and key == "losses":
            if not isinstance(component, dict):
                raise TypeError(
                    "Training config component 'losses' must be a dictionary of "
                    "loss groups, not "
                    f"{type(component).__name__}."
                )
            normalized[key] = component
            continue

        if schema is TrainingConfig and key == "transform_pipelines":
            if not isinstance(component, list):
                raise TypeError(
                    "TrainingConfig component 'transform_pipelines' must be a list "
                    "of TrainingTransformPipelineConfig objects, got "
                    f"{type(component).__name__}."
                )

            if not all(
                    isinstance(pipeline, TrainingTransformPipelineConfig)
                    for pipeline in component
            ):
                raise TypeError(
                    "Every item in TrainingConfig component "
                    "'transform_pipelines' must be a "
                    "TrainingTransformPipelineConfig object."
                )

            normalized[key] = [
                pipeline.model_dump(mode="python")
                for pipeline in component
            ]
            continue

        if schema is PredictionConfig and key == "transform_pipelines":
            if not isinstance(component, list):
                raise TypeError(
                    "PredictionConfig component 'transform_pipelines' must be a list "
                    "of PredictionTransformPipelineConfig objects, got "
                    f"{type(component).__name__}."
                )

            if not all(
                    isinstance(pipeline, PredictionTransformPipelineConfig)
                    for pipeline in component
            ):
                raise TypeError(
                    "Every item in PredictionConfig component "
                    "'transform_pipelines' must be a "
                    "PredictionTransformPipelineConfig object."
                )

            normalized[key] = [
                pipeline.model_dump(mode="python")
                for pipeline in component
            ]
            continue

        if schema is TrainingConfig and key == "additional_callbacks":
            if not isinstance(component, list):
                raise TypeError(
                    "Training config component 'additional_callbacks' must "
                    "be a list of TrainingAdditionalCallbackConfig objects, got "
                    f"{type(component).__name__}."
                )

            if not all(
                isinstance(callback, TrainingAdditionalCallbackConfig)
                for callback in component
            ):
                raise TypeError(
                    "Every item in TrainingConfig component "
                    "'additional_callbacks' must be an "
                    "TrainingAdditionalCallbackConfig object."
                )

            normalized[key] = [
                callback.model_dump(mode="python")
                for callback in component
            ]
            continue

        if key == "dataset":
            dataset_config_types = _get_runtime_types(SupportedDatasetConfig)

            if not isinstance(component, dataset_config_types):
                expected_names = ", ".join(
                    config_type.__name__
                    for config_type in dataset_config_types
                )
                raise TypeError(
                    f"Config component 'dataset' for {schema.__name__} must be "
                    f"one of ({expected_names}), got {type(component).__name__}."
                )

            normalized[key] = _config_component_to_raw(component)
            continue

        if (
            schema is TrainingConfig
            and key in {
                "composite_model_components",
                "composite_model_assembly",
            }
        ):
            entry_type = (
                CompositeModelComponentConfig
                if key == "composite_model_components"
                else CompositeModelAssemblyStepConfig
            )

            if not isinstance(component, dict):
                raise TypeError(
                    f"Training config component {key!r} must be a dictionary "
                    f"mapping names to {entry_type.__name__} objects, got "
                    f"{type(component).__name__}."
                )

            if not all(
                    isinstance(name, str) and isinstance(value, entry_type)
                    for name, value in component.items()
            ):
                raise TypeError(
                    f"Every entry in TrainingConfig component {key!r} must "
                    f"map a string name to a {entry_type.__name__} object."
                )

            normalized[key] = {
                name: value.model_dump(mode="python")
                for name, value in component.items()
            }
            continue

        expected_type = expected_component_types[key]

        if not isinstance(component, expected_type):
            raise TypeError(
                f"Config component {key!r} for {schema.__name__} must be "
                f"{expected_type.__name__}, got {type(component).__name__}."
            )

        normalized[key] = _config_component_to_raw(component)

    return normalized


def _config_component_to_raw(component: SupportedConfigComponent) -> Any:
    if isinstance(component, BaseModel):
        return component.model_dump(mode="python")

    return component


def _parse_effective_config(
    *,
    schema: type[ConfigT],
    raw_config: dict[str, Any],
    external_model: bool,
    external_datamodule: bool,
    training_manifest_path_overridden: bool,
    prediction_manifest_path_overridden: bool,
) -> ConfigT:
    if schema is TrainingConfig:
        return parse_training_config(
            raw_config=raw_config,
            model_overridden=external_model,
            datamodule_overridden=external_datamodule,
        )

    if schema is PredictionConfig:
        return parse_prediction_config(
            raw_config=raw_config,
            model_overridden=external_model,
            datamodule_overridden=external_datamodule,
            training_manifest_path_overridden=(
                training_manifest_path_overridden
            ),
        )

    if schema is EvaluationConfig:
        return parse_evaluation_config(
            raw_config=raw_config,
            prediction_manifest_path_overridden=prediction_manifest_path_overridden,
        )

    raise TypeError(f"Unsupported config schema: {schema.__name__}")
