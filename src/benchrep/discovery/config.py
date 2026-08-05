from __future__ import annotations

import inspect
from textwrap import indent
from types import UnionType
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from benchrep.discovery.registry import _resolve_registry, _support_label


_CONFIG_REGISTRY_ROUTES: Final[dict[str, tuple[str, ...]]] = {
    # Training and prediction
    "DatasetConfig": ("dataset",),
    "TransformConfig": ("transform",),
    "EncoderConfig": ("encoder",),
    "DecoderConfig": ("decoder",),
    "ModelConfig": ("model",),
    "LossTermConfig": (
        "reconstruction_loss",
        "regularization_loss",
    ),
    "OptimizerConfig": ("optimizer",),
    "LoggerConfig": ("logger",),

    # Evaluation reductions
    "EvaluationReductionsConfig": ("reduction",),
    "PCAConfig": ("reduction",),
    "UMAPConfig": ("reduction",),
    "TSNEConfig": ("reduction",),

    # Evaluation clustering
    "EvaluationClusteringConfig": ("clustering_method",),
    "KMeansConfig": ("clustering_method",),
    "LeidenConfig": ("clustering_method",),

    # Evaluation metrics
    "InternalClusteringMetricConfig": ("internal_clustering_metric",),
    "ExternalClusteringMetricConfig": ("external_clustering_metric",),
    "EmbeddingMetricConfig": ("embedding_metric",),
    "ReconstructionMetricConfig": ("reconstruction_metric",),

    # Evaluation predictability
    "EvaluationPredictabilityConfig": ("predictability_probe",),
    "DummyProbeConfig": ("predictability_probe",),
    "LogisticRegressionProbeConfig": ("predictability_probe",),
    "RidgeProbeConfig": ("predictability_probe",),
    "KNNProbeConfig": ("predictability_probe",),
    "RandomForestProbeConfig": ("predictability_probe",),
    "XGBoostProbeConfig": ("predictability_probe",),
    "SVMRBFProbeConfig": ("predictability_probe",),
}


def inspect_config(
    config: type[BaseModel] | None = None,
) -> None:
    """Print guidance or inspect a Pydantic configuration class."""
    if config is None:
        print(
            "Pass any BenchRep configuration class to inspect its fields.\n\n"
            "Suggested starting points:\n"
            "- TrainingConfig\n"
            "- PredictionConfig\n"
            "- EvaluationConfig\n\n"
            "Nested configuration types shown in the output can be inspected "
            "the same way.\n"
            "Configuration classes are available from "
            "`benchrep.assembly.schemas`.\n\n"
            "Example: `benchrep.inspect_config(TrainingConfig)`"
        )
        return

    if not inspect.isclass(config) or not issubclass(config, BaseModel):
        raise TypeError(
            "config must be a Pydantic BaseModel class, for example "
            "`TrainingConfig`, not a config instance."
        )

    print(f"Config: {config.__name__}")

    docstring = inspect.getdoc(config)
    if docstring is not None:
        print("\nDescription:")
        print(indent(docstring, "  "))

    _print_config_customization(config)

    print("\nFields:")

    for field_name, field in config.model_fields.items():
        print(f"\n  {field_name}")
        print(f"    type: {_format_annotation(field.annotation)}")
        print(
            "    required: "
            + ("yes" if field.is_required() else "no")
        )

        if not field.is_required():
            print(f"    default: {_format_default(field)}")

        if field.description is not None:
            print("    description:")
            print(indent(field.description, "      "))

        constraints = _format_constraints(field.metadata)
        if constraints:
            print(f"    constraints: {', '.join(constraints)}")

        extra = field.json_schema_extra
        if isinstance(extra, dict):
            omit_behavior = extra.get("omit_behavior")
            null_behavior = extra.get("null_behavior")
            notes = extra.get("notes")

            if omit_behavior is not None:
                print("    omitted:")
                print(indent(str(omit_behavior), "      "))

            if null_behavior is not None:
                print("    null:")
                print(indent(str(null_behavior), "      "))

            if notes:
                print("    notes:")

                if isinstance(notes, str):
                    notes = [notes]

                for note in notes:
                    print(f"      - {note}")


def _print_config_customization(
    config: type[BaseModel],
) -> None:
    registry_names = _find_config_registries(config)

    if registry_names:
        print("\nCustomization:")

        for index, requested_registry_name in enumerate(registry_names):
            registry_name, registry_info = _resolve_registry(
                requested_registry_name
            )

            if index > 0:
                print()

            print(f"  registry: {registry_name}")
            print(
                "  custom registration: "
                f"{_support_label(registry_info.custom_registration_supported)}"
            )
            print(
                "  runtime instance override: "
                f"{_support_label(registry_info.runtime_instance_override_supported)}"
            )

            if registry_info.runtime_override is not None:
                print("  runtime override:")
                print(indent(registry_info.runtime_override, "    "))

            print(
                "  details: "
                f'benchrep.inspect_registry("{registry_name}")'
            )

        return

    if _is_benchrep_schema_type(config, "DataModuleConfig"):
        print("\nCustomization:")
        print("  registry: none")
        print("  custom registration: not supported")
        print("  runtime instance override: supported")
        print("  runtime override:")
        print(
            indent(
                "Pass a compatible LightningDataModule instance through the "
                "`datamodule` argument of a training or prediction workflow.",
                "    ",
            )
        )


def _find_config_registries(
    config: type[BaseModel],
) -> tuple[str, ...]:
    for base in config.__mro__:
        if not base.__module__.startswith(
            "benchrep.assembly.schemas"
        ):
            continue

        registry_names = _CONFIG_REGISTRY_ROUTES.get(base.__name__)

        if registry_names is not None:
            return registry_names

    return ()


def _is_benchrep_schema_type(
    config: type[BaseModel],
    expected_name: str,
) -> bool:
    return any(
        base.__name__ == expected_name
        and base.__module__.startswith("benchrep.assembly.schemas")
        for base in config.__mro__
    )


def _format_annotation(annotation: Any) -> str:
    if annotation is Any:
        return "Any"

    if annotation is type(None):
        return "None"

    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin is Annotated:
        base_annotation, *metadata = arguments
        rendered = _format_annotation(base_annotation)
        constraints = _format_constraints(metadata)

        if constraints:
            return f"{rendered} ({', '.join(constraints)})"

        return rendered

    if origin in {Union, UnionType}:
        return " | ".join(
            _format_annotation(argument)
            for argument in arguments
        )

    if origin is Literal:
        values = ", ".join(repr(value) for value in arguments)
        return f"Literal[{values}]"

    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        rendered_arguments = ", ".join(
            _format_annotation(argument)
            for argument in arguments
        )

        if rendered_arguments:
            return f"{origin_name}[{rendered_arguments}]"

        return origin_name

    name = getattr(annotation, "__name__", None)
    if name is not None:
        return name

    return str(annotation).replace("typing.", "")


def _format_default(field: FieldInfo) -> str:
    if field.default_factory is not None:
        factory_name = getattr(
            field.default_factory,
            "__name__",
            type(field.default_factory).__name__,
        )
        return f"{factory_name}()"

    return repr(field.default)


def _format_constraints(metadata: Any) -> list[str]:
    constraints: list[str] = []

    attribute_labels = {
        "gt": ">",
        "ge": ">=",
        "lt": "<",
        "le": "<=",
        "min_length": "minimum length",
        "max_length": "maximum length",
        "multiple_of": "multiple of",
        "pattern": "pattern",
    }

    for item in metadata:
        for attribute, label in attribute_labels.items():
            value = getattr(item, attribute, None)

            if value is not None:
                constraints.append(f"{label} {value!r}")

    return list(dict.fromkeys(constraints))