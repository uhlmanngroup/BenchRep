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

from benchrep.assembly.schemas.training_config_schema import (
    TrainingConfig,
    RunConfig,
    ReproducibilityConfig,
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    LossTermConfig,
    OptimizerConfig,
    SupportedDatasetConfig,
    DataModuleConfig,
    TrainerConfig,
    LoggerConfig,
    CheckpointConfig,
    InspectionConfig,
)
from benchrep.assembly.schemas.prediction_config_schema import (
    PredictionConfig,
    PredictionSourceConfig,
    PredictionDataConfig,
    PredictionInferenceConfig,
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


SUPPORTED_CONFIG_SCHEMAS = (TrainingConfig, PredictionConfig, EvaluationConfig)
SupportedConfigType: TypeAlias = TrainingConfig | PredictionConfig | EvaluationConfig
ConfigT = TypeVar("ConfigT", bound=SupportedConfigType)

ConfigSource: TypeAlias = Literal[
    "yaml",
    "config_object",
    "components",
    "yaml_with_components",
]

LossesConfig: TypeAlias = dict[str, dict[str, LossTermConfig]]
SupportedTrainingConfigComponent: TypeAlias = (
        RunConfig
        | ReproducibilityConfig
        | ModelConfig
        | EncoderConfig
        | DecoderConfig
        | LossesConfig
        | OptimizerConfig
        | SupportedDatasetConfig
        | DataModuleConfig
        | TrainerConfig
        | LoggerConfig
        | CheckpointConfig
        | InspectionConfig
)

SupportedPredictionConfigComponent: TypeAlias = (
        PredictionSourceConfig
        | SupportedDatasetConfig
        | PredictionDataConfig
        | PredictionInferenceConfig
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
    config_path: Path | str | None = None,
    full_config_object: TrainingConfig | None = None,
    config_components: Mapping[str, SupportedTrainingConfigComponent] | None = None,
    external_model: bool = False,
    external_datamodule: bool = False,
    training_manifest_path_overridden: bool = False,
) -> ConfigCompositionResult[TrainingConfig]: ...


@overload
def compose_effective_config(
    *,
    schema: type[PredictionConfig],
    config_path: Path | str | None = None,
    full_config_object: PredictionConfig | None = None,
    config_components: Mapping[str, SupportedPredictionConfigComponent] | None = None,
    external_model: bool = False,
    external_datamodule: bool = False,
    training_manifest_path_overridden: bool = False,
) -> ConfigCompositionResult[PredictionConfig]: ...


@overload
def compose_effective_config(
    *,
    schema: type[EvaluationConfig],
    config_path: Path | str | None = None,
    full_config_object: EvaluationConfig | None = None,
    config_components: Mapping[str, SupportedEvaluationConfigComponent] | None = None,
    external_model: bool = False,
    external_datamodule: bool = False,
    training_manifest_path_overridden: bool = False,
) -> ConfigCompositionResult[EvaluationConfig]: ...


def compose_effective_config(
    *,
    schema: type[ConfigT],
    config_path: Path | str | None = None,
    full_config_object: ConfigT | None = None,
    config_components: Mapping[str, SupportedConfigComponent] | None = None,
    external_model: bool = False,
    external_datamodule: bool = False,
    training_manifest_path_overridden: bool = False,
) -> ConfigCompositionResult[ConfigT]:
    """Compose the effective typed config object for a BenchRep workflow.

    This helper centralizes the early entrypoint logic that decides which config
    input should be used before workflow-specific resolution. It accepts an
    optional YAML config path, an optional complete typed config object, and/or
    optional keyed top-level config components. It returns the final effective
    config object that should be passed to the stage resolver, together with the
    original YAML payload if one was supplied.

    Precedence
    ----------
    A complete ``full_config_object`` takes absolute precedence. If it is
    provided, it becomes the effective config. Any YAML file is still loaded and
    returned as ``original_config_raw`` for provenance, but it is not used as the
    effective config. Any ``config_components`` are ignored.

    If no ``full_config_object`` is provided and ``config_path`` is provided, the
    YAML file is used as the base config. If ``config_components`` are also
    provided, they replace matching top-level YAML sections before validation.

        NOTE: downstream logic in entrypoint runners will override any configs
        related to model (config.model/encoder/decoder/losses/optimizer) and
        datamodule (config.dataset and config.datamodule) if they were provided
        as instantiated objects in the top level _train() or _predict() functions.

    If neither ``full_config_object`` nor ``config_path`` is provided,
    ``config_components`` must provide enough top-level sections to construct a
    valid config for the selected ``schema``.

    In short::

        full_config_object
            > YAML base with top-level config_components overrides
            > YAML only
            > config_components only

    When both ``config_path`` and ``config_components`` are provided, the YAML
    config is used only as the base. Each supplied component replaces the
    matching top-level YAML section. Therefore, for overlapping sections,
    ``config_components`` take precedence over YAML.

    Component behavior
    ------------------
    ``config_components`` is a mapping from top-level config field name to the
    corresponding typed config section, for example ``{"run": RunConfig(...)}``
    or ``{"metrics": EvaluationMetricsConfig(...)}``. Component merging is
    shallow by design: a provided component replaces the entire matching
    top-level section rather than recursively patching nested fields.

    Validation context
    ------------------
    The ``external_model`` and ``external_datamodule`` flags are forwarded to
    training config parsing so externally supplied runtime objects can relax the
    corresponding required config sections. ``training_manifest_path_overridden``
    is forwarded to prediction config parsing so a manifest path supplied outside
    the config can satisfy prediction source validation.

    Parameters
    ----------
    schema
        The target workflow config schema: ``TrainingConfig``,
        ``PredictionConfig``, or ``EvaluationConfig``.
    config_path
        Optional path to a YAML config file. When provided, it is always loaded
        and returned as the original config for provenance.
    full_config_object
        Optional complete typed config object. If provided, it becomes the
        effective config and all other effective config inputs are ignored.
    config_components
        Optional mapping of top-level config section names to typed config
        section objects. Used either as the full config source or as top-level
        overrides on top of YAML.
    external_model
        Whether a runtime model object was supplied to the training entrypoint.
    external_datamodule
        Whether a runtime datamodule object was supplied to the training
        entrypoint.
    training_manifest_path_overridden
        Whether a training manifest path was supplied to the prediction
        entrypoint outside the config object.

    Returns
    -------
    ConfigCompositionResult[ConfigT]
        The original YAML path/raw mapping, the effective typed config object,
        and provenance flags describing how the effective config was composed.

    Raises
    ------
    ValueError
        If no usable config input is provided, if component keys are unsupported,
        or if the composed config fails schema validation.
    TypeError
        If ``schema`` is unsupported, ``full_config_object`` has the wrong type,
        or a component value does not match the expected top-level section type.
    """
    composition_messages: list[str] = []
    composition_warnings: list[str] = []

    if schema not in SUPPORTED_CONFIG_SCHEMAS:
        raise TypeError(f"Unsupported config schema: {schema.__name__}")

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
            "At least one of config_path, full_config_object, or config_components "
            "is required."
        )

    if full_config_object is not None and not isinstance(full_config_object, schema):
        raise TypeError(
            f"Expected full_config_object to be an instance of {schema.__name__}; "
            f"got {type(full_config_object).__name__}."
        )

    # Full typed config object takes highest precedence
    if full_config_object is not None:
        config_source: ConfigSource = "config_object"
        yaml_used_as_base = False
        effective_config = full_config_object

        if yaml_supplied:
            composition_warnings.append(
                "full_config_object provided; config YAML will be ignored as "
                f"effective config: {original_config_path}"
            )

        if config_components:
            component_names = ", ".join(config_components.keys())
            composition_warnings.append(
                "full_config_object provided; config_components will be ignored: "
                f"{component_names}"
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

    # Convert keyed config components into raw top-level config sections.
    component_raw = _normalize_config_components(
        schema=schema,
        config_components=config_components,
    )

    if yaml_supplied and component_raw:
        config_source = "yaml_with_components"
        yaml_used_as_base = True
        raw_config = dict(original_config_raw)
        raw_config.update(component_raw)

        composition_messages.append(
            "Both config_path and config_components provided; config_components "
            "will replace matching top-level YAML sections."
        )

    elif yaml_supplied:
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


def _normalize_config_components(
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
            training_manifest_path_overridden=training_manifest_path_overridden,
        )

    if schema is EvaluationConfig:
        return parse_evaluation_config(raw_config=raw_config)

    raise TypeError(f"Unsupported config schema: {schema.__name__}")
