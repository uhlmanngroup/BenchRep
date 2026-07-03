from __future__ import annotations

from typing import Any


class Registry:
    """Name-to-object registry used by builders.

    The registry maps string names from config files to Python classes or
    callables that can be instantiated by builders.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Any] = {}
        self._canonical_keys: dict[str, str] = {}

    def register(self, key: str, item: Any, *aliases: str) -> None:
        canonical_key = self._normalize_key(key)
        # Silently collapse duplicates
        names = tuple(
            dict.fromkeys(self._normalize_key(name) for name in (key, *aliases))
        )

        # Refuse overwrites
        for name in names:
            if name in self._items:
                raise KeyError(
                    f"{self.name} registry already contains key {name!r}. "
                    "Choose a different name or remove the existing registration."
                )

        for name in names:
            self._items[name] = item
            self._canonical_keys[name] = canonical_key

    def get(self, key: str) -> Any:
        _ensure_builtins_registered()

        # Retrieve a registered object by name, with a debuggable error for unknown keys.
        key = self._normalize_key(key)

        if key not in self._items:
            available = tuple(sorted(self._items))
            raise KeyError(
                f"Unknown {self.name} key {key!r}. "
                f"Available options: {available}."
            )

        return self._items[key]

    def resolve_key(self, key: str) -> str:
        """Resolve a registered key or alias to its canonical registry key."""
        _ensure_builtins_registered()

        key = self._normalize_key(key)

        if key not in self._canonical_keys:
            available = tuple(sorted(self._items))
            raise KeyError(
                f"Unknown {self.name} key {key!r}. "
                f"Available options: {available}."
            )

        return self._canonical_keys[key]

    def create(self, key: str, **kwargs: Any) -> Any:
        # Retrieve a registered class/callable and instantiate it with keyword arguments.
        item = self.get(key)
        return item(**kwargs)

    def keys(self) -> tuple[str, ...]:
        _ensure_builtins_registered()

        # Return registered keys in deterministic order for errors, debugging, and validation.
        return tuple(sorted(self._items))

    def canonical_keys(self) -> tuple[str, ...]:
        _ensure_builtins_registered()

        # Return canonical registered keys in deterministic order.
        return tuple(sorted(set(self._canonical_keys.values())))

    @staticmethod
    def _normalize_key(key: str) -> str:
        if not isinstance(key, str):
            raise TypeError(f"Registry keys must be strings, got {type(key).__name__}.")

        key = key.lower().strip().replace("-", "_")

        if not key:
            raise ValueError("Registry key must be a non-empty string.")

        return key


def _ensure_builtins_registered() -> None:
    """Ensure BenchRep built-in registry entries are available."""

    from benchrep.assembly.registries.builtins import register_builtins

    register_builtins()


# Data
DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")
# Architecture and training
ENCODERS = Registry("encoder")
DECODERS = Registry("decoder")
MODELS = Registry("model")
RECONSTRUCTION_LOSSES = Registry("reconstruction loss")
REGULARIZATION_LOSSES = Registry("regularization loss")
OPTIMIZERS = Registry("optimizer")
LOGGERS = Registry("logger")
# Evaluation
EVAL_REDUCTIONS = Registry("reduction")
EVAL_CLUSTERING_METHODS = Registry("clustering method")
EVAL_INTERNAL_CLUSTERING_METRICS = Registry("internal clustering metric")
EVAL_EXTERNAL_CLUSTERING_METRICS = Registry("external clustering metric")
EVAL_EMBEDDING_METRICS = Registry("embedding metric")
EVAL_PREDICTABILITY_PROBES = Registry("predictability probe")
EVAL_RECONSTRUCTION_METRICS = Registry("reconstruction metric")