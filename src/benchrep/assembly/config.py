from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, Literal, TypeVar, Generic, Mapping, overload

import yaml

from pydantic import BaseModel

from benchrep.records.logs import get_run_logger
from benchrep.assembly.schemas.training_config_schema import (
    TrainingConfig,
    RunConfig,
    ReproducibilityConfig,
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    LossTermConfig,
    OptimizerConfig,
    DataSelectionConfig,
    DatasetConfig,
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
from benchrep.assembly.schemas.validation import (
    parse_training_config,
    parse_prediction_config,
    parse_evaluation_config,
)


SupportedConfig: TypeAlias = TrainingConfig | PredictionConfig | EvaluationConfig
ConfigT = TypeVar("ConfigT", bound=SupportedConfig)

ConfigSource: TypeAlias = Literal[
    "yaml",
    "config_object",
    "components",
    "yaml_with_components",
    "config_object_with_components",
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
        | DataSelectionConfig
        | DatasetConfig
        | DataModuleConfig
        | TrainerConfig
        | LoggerConfig
        | CheckpointConfig
        | InspectionConfig
)

SupportedPredictionConfigComponent: TypeAlias = (
        PredictionSourceConfig
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

STAGE_BY_SCHEMA = {
    TrainingConfig: "train",
    PredictionConfig: "predict",
    EvaluationConfig: "evaluate",
}

TRAINING_COMPONENT_TYPES: dict[str, type[Any]] = {
    "run": RunConfig,
    "reproducibility": ReproducibilityConfig,
    "model": ModelConfig,
    "encoder": EncoderConfig,
    "decoder": DecoderConfig,
    "optimizer": OptimizerConfig,
    "data": DataSelectionConfig,
    "dataset": DatasetConfig,
    "datamodule": DataModuleConfig,
    "trainer": TrainerConfig,
    "logger": LoggerConfig,
    "checkpointing": CheckpointConfig,
    "inspection": InspectionConfig,
}

PREDICTION_COMPONENT_TYPES: dict[str, type[Any]] = {
    "source": PredictionSourceConfig,
    "data": PredictionDataConfig,
    "inference": PredictionInferenceConfig,
    "exports": PredictionExportConfig,
}

EVALUATION_COMPONENT_TYPES: dict[str, type[Any]] = {
    "source": EvaluationSourceConfig,
    "run": EvaluationRunConfig,
    "reductions": EvaluationReductionsConfig,
    "clustering": EvaluationClusteringConfig,
    "metrics": EvaluationMetricsConfig,
    "reconstruction": EvaluationReconstructionConfig,
    "plots": EvaluationPlotsConfig,
}


@dataclass(frozen=True)
class ConfigCompositionResult(Generic[ConfigT]):
    original_config_path: Path | None
    original_config_raw: dict[str, Any] | None
    effective_config: ConfigT
    effective_source: ConfigSource
    yaml_supplied: bool
    yaml_used_as_base: bool


@overload
def build_effective_config(
    *,
    schema: type[TrainingConfig],
    config_path: Path | str | None = None,
    full_config_object: TrainingConfig | None = None,
    config_components: Mapping[str, SupportedTrainingConfigComponent] | None = None,
) -> ConfigCompositionResult[TrainingConfig]: ...


@overload
def build_effective_config(
    *,
    schema: type[PredictionConfig],
    config_path: Path | str | None = None,
    full_config_object: PredictionConfig | None = None,
    config_components: Mapping[str, SupportedPredictionConfigComponent] | None = None,
) -> ConfigCompositionResult[PredictionConfig]: ...


@overload
def build_effective_config(
    *,
    schema: type[EvaluationConfig],
    config_path: Path | str | None = None,
    full_config_object: EvaluationConfig | None = None,
    config_components: Mapping[str, SupportedEvaluationConfigComponent] | None = None,
) -> ConfigCompositionResult[EvaluationConfig]: ...


def build_effective_config(
    *,
    schema: type[ConfigT],
    config_path: Path | str | None = None,
    full_config_object: ConfigT | None = None,
    config_components: Mapping[str, SupportedConfigComponent] | None = None,
    external_model: bool = False,
    external_datamodule: bool = False,
    training_manifest_path_overridden: bool = False,
) -> ConfigCompositionResult[ConfigT]:
    """Build the effective typed config object for a BenchRep workflow.

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
    run_log = get_run_logger()

    if schema not in STAGE_BY_SCHEMA:
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
            run_log.warning(
                "full_config_object provided; config YAML will be ignored as "
                "effective config: %s",
                original_config_path,
            )

        if config_components:
            component_names = ", ".join(config_components.keys())
            run_log.warning(
                "full_config_object provided; config_components will be ignored: %s",
                component_names,
            )

        return ConfigCompositionResult(
            original_config_path=original_config_path,
            original_config_raw=original_config_raw,
            effective_config=effective_config,
            effective_source=config_source,
            yaml_supplied=yaml_supplied,
            yaml_used_as_base=yaml_used_as_base,
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

        run_log.info(
            "Both config_path and config_components provided; config_components "
            "will replace matching top-level YAML sections."
        )

    elif yaml_supplied:
        config_source = "yaml"
        yaml_used_as_base = True
        raw_config = original_config_raw

        run_log.info("Only config_path provided; using YAML as effective config.")

    else:
        config_source = "components"
        yaml_used_as_base = False
        raw_config = component_raw

        run_log.info(
            "Only config_components provided; composing effective config from "
            "top-level components."
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


def _normalize_config_components(
    *,
    schema: type[SupportedConfig],
    config_components: Mapping[str, SupportedConfigComponent] | None,
) -> dict[str, Any]:
    if not config_components:
        return {}

    if schema is TrainingConfig:
        allowed_component_types = TRAINING_COMPONENT_TYPES
        allowed_keys = set(allowed_component_types) | {"losses"}

    elif schema is PredictionConfig:
        allowed_component_types = PREDICTION_COMPONENT_TYPES
        allowed_keys = set(allowed_component_types)

    elif schema is EvaluationConfig:
        allowed_component_types = EVALUATION_COMPONENT_TYPES
        allowed_keys = set(allowed_component_types)

    else:
        raise TypeError(f"Unsupported config schema: {schema.__name__}")

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

        expected_type = allowed_component_types[key]

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