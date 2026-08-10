from __future__ import annotations

import inspect
from dataclasses import dataclass
from textwrap import fill, indent
from typing import Any, Final

from benchrep.assembly.registries.core import (
    DATASETS,
    TRANSFORMS,
    ENCODERS,
    DECODERS,
    MODELS,
    RECONSTRUCTION_LOSSES,
    REGULARIZATION_LOSSES,
    OPTIMIZERS,
    LOGGERS,
    CALLBACKS,
    EVAL_REDUCTIONS,
    EVAL_CLUSTERING_METHODS,
    EVAL_INTERNAL_CLUSTERING_METRICS,
    EVAL_EXTERNAL_CLUSTERING_METRICS,
    EVAL_EMBEDDING_METRICS,
    EVAL_PREDICTABILITY_PROBES,
    EVAL_RECONSTRUCTION_METRICS,
    Registry,
)


@dataclass(frozen=True)
class ComponentRegistryInfo:
    symbol: str
    registry: Registry
    custom_registration_supported: bool | None = None
    runtime_instance_override_supported: bool | None = None
    config_locations: tuple[str, ...] = ()
    contract: str | None = None
    runtime_override: str | None = None


_COMPONENT_REGISTRIES: Final[dict[str, ComponentRegistryInfo]] = {
    "dataset": ComponentRegistryInfo(
        symbol="DATASETS",
        registry=DATASETS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=(
            "TrainingConfig.dataset",
            "PredictionConfig.dataset",
        ),
        contract=(
            "The registered callable must accept the configured `params` and return a "
            "`BaseDataset` instance."
        ),
    ),
    "transform": ComponentRegistryInfo(
        symbol="TRANSFORMS",
        registry=TRANSFORMS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=(
            "TrainingConfig.transforms",
            "PredictionConfig.transforms",
        ),
        contract=(
            "The registered factory must accept the configured `params` and return a "
            "callable that maps one sample's `x` tensor to a `torch.Tensor` before "
            "batching."
        ),
    ),
    "encoder": ComponentRegistryInfo(
        symbol="ENCODERS",
        registry=ENCODERS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.encoder",),
        contract=(
            "The registered class must satisfy BenchRep's `BaseEncoder` "
            "interface."
        ),
    ),
    "decoder": ComponentRegistryInfo(
        symbol="DECODERS",
        registry=DECODERS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.decoder",),
        contract=(
            "The registered class must satisfy BenchRep's `BaseDecoder` "
            "interface."
        ),
    ),
    "model": ComponentRegistryInfo(
        symbol="MODELS",
        registry=MODELS,
        custom_registration_supported=False,
        runtime_instance_override_supported=True,
        config_locations=("TrainingConfig.model",),
        runtime_override=(
            "Pass a compatible model instance through the `model` argument of "
            "`train_ae()`, `train_vae()`, `predict_ae()`, or `predict_vae()`."
        ),
    ),
    "reconstruction_loss": ComponentRegistryInfo(
        symbol="RECONSTRUCTION_LOSSES",
        registry=RECONSTRUCTION_LOSSES,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.losses.reconstruction",),
        contract=(
            "The registered loss must accept `reconstruction` and `target` "
            "keyword arguments and return a loss tensor."
        ),
    ),
    "regularization_loss": ComponentRegistryInfo(
        symbol="REGULARIZATION_LOSSES",
        registry=REGULARIZATION_LOSSES,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.losses.regularization",),
        contract=(
            "The registered loss must accept `z_mu` and `z_logvar` keyword "
            "arguments and return a loss tensor."
        ),
    ),
    "optimizer": ComponentRegistryInfo(
        symbol="OPTIMIZERS",
        registry=OPTIMIZERS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.optimizer",),
        contract=(
            "The registered callable must accept model parameters as its first argument "
            "and the configured `params` as keyword arguments, and return a "
            "`torch.optim.Optimizer`."
        ),
    ),
    "logger": ComponentRegistryInfo(
        symbol="LOGGERS",
        registry=LOGGERS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.logger",),
        contract=(
            "The registered class must satisfy Lightning's logger interface."
        ),
    ),
    "callback": ComponentRegistryInfo(
        symbol="CALLBACKS",
        registry=CALLBACKS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.additional_callbacks",),
        contract=(
            "The registered class or factory must accept the configured "
            "`params` and return a Lightning `Callback` instance."
        ),
    ),
    "reduction": ComponentRegistryInfo(
        symbol="EVAL_REDUCTIONS",
        registry=EVAL_REDUCTIONS,
        custom_registration_supported=False,
        runtime_instance_override_supported=False,
        config_locations=("EvaluationConfig.reductions",),
    ),
    "clustering_method": ComponentRegistryInfo(
        symbol="EVAL_CLUSTERING_METHODS",
        registry=EVAL_CLUSTERING_METHODS,
        custom_registration_supported=False,
        runtime_instance_override_supported=False,
        config_locations=("EvaluationConfig.clustering",),
    ),
    "internal_clustering_metric": ComponentRegistryInfo(
        symbol="EVAL_INTERNAL_CLUSTERING_METRICS",
        registry=EVAL_INTERNAL_CLUSTERING_METRICS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=(
            "EvaluationConfig.metrics.clustering.internal",
        ),
        contract=(
            "The registered callable must accept an embedding matrix and cluster "
            "labels, followed by the configured parameters, and return a scalar."
        ),
    ),
    "external_clustering_metric": ComponentRegistryInfo(
        symbol="EVAL_EXTERNAL_CLUSTERING_METRICS",
        registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=(
            "EvaluationConfig.metrics.clustering.external",
        ),
        contract=(
            "The registered callable must accept reference labels and predicted "
            "cluster labels, followed by the configured parameters, and return "
            "a scalar."
        ),
    ),
    "embedding_metric": ComponentRegistryInfo(
        symbol="EVAL_EMBEDDING_METRICS",
        registry=EVAL_EMBEDDING_METRICS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=("EvaluationConfig.metrics.embedding",),
        contract=(
            "The registered callable must accept an embedding matrix and the "
            "configured parameters, and return dimension-wise values or a mapping "
            "of named dimension-wise values."
        ),
    ),
    "predictability_probe": ComponentRegistryInfo(
        symbol="EVAL_PREDICTABILITY_PROBES",
        registry=EVAL_PREDICTABILITY_PROBES,
        custom_registration_supported=False,
        runtime_instance_override_supported=False,
        config_locations=("EvaluationConfig.metrics.predictability",),
    ),
    "reconstruction_metric": ComponentRegistryInfo(
        symbol="EVAL_RECONSTRUCTION_METRICS",
        registry=EVAL_RECONSTRUCTION_METRICS,
        custom_registration_supported=True,
        runtime_instance_override_supported=False,
        config_locations=("EvaluationConfig.metrics.reconstruction",),
        contract=(
            "The registered callable must accept input and reconstruction arrays, "
            "followed by the configured parameters, and return a scalar."
        ),
    ),
}


def inspect_registry(
    registry: str | None = None,
    component: str | None = None,
) -> None:
    """Print human-readable information about BenchRep registries.

    With no arguments, summarizes all available registries. With a registry
    name, lists its registered components and customization policy. With both
    a registry and component name, describes the selected implementation.

    Component names may be canonical names or registered aliases.
    """
    if registry is None:
        if component is not None:
            raise ValueError(
                "component cannot be provided without a registry."
            )

        print("BenchRep registries")

        for registry_name, registry_info in _COMPONENT_REGISTRIES.items():
            selected_registry = registry_info.registry
            n_components = len(selected_registry.canonical_keys())
            component_label = (
                "component" if n_components == 1 else "components"
            )

            print(
                f"- {registry_name}: {n_components} {component_label} "
                f"({_registry_object_path(registry_info)})"
            )
            print(
                "  custom registration: "
                f"{_support_label(registry_info.custom_registration_supported)}; "
                "runtime instance override: "
                f"{_support_label(registry_info.runtime_instance_override_supported)}"
            )

        return

    registry_name, registry_info = _resolve_registry(registry)
    selected_registry = registry_info.registry
    aliases_by_canonical = selected_registry.aliases_by_canonical()

    if component is None:
        print(f"Registry: {registry_name}")
        print(
            "Registry object: "
            f"{_registry_object_path(registry_info)}"
        )

        _print_registry_customization(registry_info)

        print("\nComponents:")

        for canonical_name, aliases in aliases_by_canonical.items():
            item = selected_registry.get(canonical_name)

            print(f"\n  {canonical_name}")
            print(
                "    aliases: "
                + (", ".join(aliases) if aliases else "none")
            )
            print(f"    target: {_qualified_name(item)}")

        return

    canonical_name = selected_registry.resolve_key(component)
    item = selected_registry.get(canonical_name)
    aliases = aliases_by_canonical[canonical_name]
    signature = _callable_signature(item)
    signature_label = (
        "Constructor signature"
        if inspect.isclass(item)
        else "Callable signature"
    )
    docstring = inspect.getdoc(item)

    print(f"Registry: {registry_name}")
    print(f"Component: {canonical_name}")

    _print_registry_customization(registry_info)

    print(
        "Aliases: "
        + (", ".join(aliases) if aliases else "none")
    )
    print(f"Target: {_qualified_name(item)}")
    print(
        f"{signature_label}: "
        + (signature if signature is not None else "unavailable")
    )

    print("\nDescription:")
    if docstring is None:
        print("  No docstring available.")
    else:
        print(indent(docstring, "  "))

    print(
        "\nNote: the callable signature describes the registered "
        "implementation. BenchRep may supply some arguments internally, so "
        "it does not always map directly to the component's config `params`."
    )


def _resolve_registry(
    registry: str,
) -> tuple[str, ComponentRegistryInfo]:
    if not isinstance(registry, str):
        raise TypeError(
            f"registry must be a string, got {type(registry).__name__}."
        )

    registry_name = registry.lower().strip().replace("-", "_")

    if registry_name not in _COMPONENT_REGISTRIES:
        available = tuple(_COMPONENT_REGISTRIES)
        raise ValueError(
            f"Unknown registry {registry!r}. "
            f"Available registries: {available}. "
            "Run `benchrep.inspect_registry()` with no arguments for a "
            "registry overview."
        )

    return registry_name, _COMPONENT_REGISTRIES[registry_name]


def _print_registry_customization(
    registry_info: ComponentRegistryInfo,
) -> None:
    print(
        "Custom registration: "
        f"{_support_label(registry_info.custom_registration_supported)}"
    )
    print(
        "Runtime instance override: "
        f"{_support_label(registry_info.runtime_instance_override_supported)}"
    )

    if registry_info.config_locations:
        print("Config locations:")
        for location in registry_info.config_locations:
            print(f"  - {location}")

    if registry_info.contract is not None:
        print("Contract:")
        print(
            indent(
                fill(
                    registry_info.contract,
                    width=88,
                ),
                "  ",
            )
        )

    if registry_info.runtime_override is not None:
        print("Runtime override:")
        print(
            indent(
                fill(
                    registry_info.runtime_override,
                    width=88,
                ),
                "  ",
            )
        )


def _support_label(value: bool | None) -> str:
    if value is True:
        return "supported"

    if value is False:
        return "not supported"

    return "not documented"


def _registry_object_path(
    registry_info: ComponentRegistryInfo,
) -> str:
    return (
        "benchrep.assembly.registries."
        f"{registry_info.symbol}"
    )


def _qualified_name(item: Any) -> str:
    module = getattr(item, "__module__", None)
    qualname = getattr(item, "__qualname__", None)

    if module is not None and qualname is not None:
        return f"{module}.{qualname}"

    return type(item).__name__


def _callable_signature(item: Any) -> str | None:
    try:
        signature = inspect.signature(item)
    except (TypeError, ValueError):
        return None

    if inspect.isclass(item):
        signature = signature.replace(
            return_annotation=inspect.Signature.empty,
        )

    return str(signature)


def list_registries() -> dict[str, str]:
    """Return registry names and their public registry-object paths."""
    return {
        name: _registry_object_path(registry_info)
        for name, registry_info in _COMPONENT_REGISTRIES.items()
    }


def list_registered_components(
    registry: str,
    *,
    include_aliases: bool = False,
) -> tuple[str, ...] | dict[str, tuple[str, ...]]:
    """Return the components registered in a named registry."""
    _, registry_info = _resolve_registry(registry)
    selected_registry = registry_info.registry

    if include_aliases:
        return selected_registry.aliases_by_canonical()

    return selected_registry.canonical_keys()
