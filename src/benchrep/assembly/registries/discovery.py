from __future__ import annotations

from typing import Final

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
    if not isinstance(registry, str):
        raise TypeError(
            f"registry must be a string, got {type(registry).__name__}."
        )

    registry_name = registry.lower().strip().replace("-", "_")

    if registry_name not in _COMPONENT_REGISTRIES:
        available = tuple(_COMPONENT_REGISTRIES)
        raise ValueError(
            f"Unknown registry {registry!r}. "
            f"Available registries: {available}."
        )

    _, selected_registry = _COMPONENT_REGISTRIES[registry_name]

    if include_aliases:
        return selected_registry.aliases_by_canonical()

    return selected_registry.canonical_keys()
