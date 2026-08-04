from __future__ import annotations

import inspect
from textwrap import indent
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
    EVAL_REDUCTIONS,
    EVAL_CLUSTERING_METHODS,
    EVAL_INTERNAL_CLUSTERING_METRICS,
    EVAL_EXTERNAL_CLUSTERING_METRICS,
    EVAL_EMBEDDING_METRICS,
    EVAL_PREDICTABILITY_PROBES,
    EVAL_RECONSTRUCTION_METRICS,
    Registry,
)


_COMPONENT_REGISTRIES: Final[dict[str, tuple[str, Registry]]] = {
    "dataset": ("DATASETS", DATASETS),
    "transform": ("TRANSFORMS", TRANSFORMS),
    "encoder": ("ENCODERS", ENCODERS),
    "decoder": ("DECODERS", DECODERS),
    "model": ("MODELS", MODELS),
    "reconstruction_loss": (
        "RECONSTRUCTION_LOSSES",
        RECONSTRUCTION_LOSSES,
    ),
    "regularization_loss": (
        "REGULARIZATION_LOSSES",
        REGULARIZATION_LOSSES,
    ),
    "optimizer": ("OPTIMIZERS", OPTIMIZERS),
    "logger": ("LOGGERS", LOGGERS),
    "reduction": ("EVAL_REDUCTIONS", EVAL_REDUCTIONS),
    "clustering_method": (
        "EVAL_CLUSTERING_METHODS",
        EVAL_CLUSTERING_METHODS,
    ),
    "internal_clustering_metric": (
        "EVAL_INTERNAL_CLUSTERING_METRICS",
        EVAL_INTERNAL_CLUSTERING_METRICS,
    ),
    "external_clustering_metric": (
        "EVAL_EXTERNAL_CLUSTERING_METRICS",
        EVAL_EXTERNAL_CLUSTERING_METRICS,
    ),
    "embedding_metric": (
        "EVAL_EMBEDDING_METRICS",
        EVAL_EMBEDDING_METRICS,
    ),
    "predictability_probe": (
        "EVAL_PREDICTABILITY_PROBES",
        EVAL_PREDICTABILITY_PROBES,
    ),
    "reconstruction_metric": (
        "EVAL_RECONSTRUCTION_METRICS",
        EVAL_RECONSTRUCTION_METRICS,
    ),
}


def inspect_registry(
    registry: str | None = None,
    component: str | None = None,
) -> None:
    """Print human-readable information about BenchRep registries.

    With no arguments, summarizes all available registries. With a registry
    name, lists its registered components. With both a registry and component
    name, describes the selected registered callable in detail.

    Component names may be canonical names or registered aliases.
    """
    if registry is None:
        if component is not None:
            raise ValueError(
                "component cannot be provided without a registry."
            )

        print("BenchRep registries")

        for registry_name, (symbol, selected_registry) in (
            _COMPONENT_REGISTRIES.items()
        ):
            n_components = len(selected_registry.canonical_keys())
            component_label = (
                "component" if n_components == 1 else "components"
            )

            print(
                f"- {registry_name}: {n_components} {component_label} "
                f"(benchrep.assembly.registries.{symbol})"
            )

        return

    registry_name, symbol, selected_registry = _resolve_registry(registry)
    aliases_by_canonical = selected_registry.aliases_by_canonical()

    if component is None:
        print(f"Registry: {registry_name}")
        print(
            "Registry object: "
            f"benchrep.assembly.registries.{symbol}"
        )
        print("Components:")

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
) -> tuple[str, str, Registry]:
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

    symbol, selected_registry = _COMPONENT_REGISTRIES[registry_name]
    return registry_name, symbol, selected_registry


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
        name: f"benchrep.assembly.registries.{symbol}"
        for name, (symbol, _) in _COMPONENT_REGISTRIES.items()
    }


def list_registered_components(
    registry: str,
    *,
    include_aliases: bool = False,
) -> tuple[str, ...] | dict[str, tuple[str, ...]]:
    """Return the components registered in a named registry."""
    _, _, selected_registry = _resolve_registry(registry)

    if include_aliases:
        return selected_registry.aliases_by_canonical()

    return selected_registry.canonical_keys()
