from __future__ import annotations

from typing import Any


class Registry:
    """Name-to-object registry used by builders.

    The registry maps string names from config files to Python classes or
    callables that can be instantiated by builders.
    """

    def __init__(
        self,
        name: str,
        *,
        custom_registration_supported: bool = True,
    ) -> None:
        self.name = name
        self._custom_registration_supported = (
            custom_registration_supported
        )
        self._items: dict[str, Any] = {}
        self._canonical_keys: dict[str, str] = {}

    @property
    def custom_registration_supported(self) -> bool:
        """Whether users may register custom components."""
        return self._custom_registration_supported

    def register(
        self,
        key: str,
        item: Any,
        *aliases: str,
    ) -> None:
        """Register a user-provided component."""
        if not self.custom_registration_supported:
            raise RuntimeError(
                "Custom registration is not supported for the "
                f"{self.name} registry. This registry exposes BenchRep's "
                "built-in components for discovery and configuration only."
            )

        # Register built-ins first so custom components cannot claim a built-in
        # name merely because registration has not yet been initialized.
        _ensure_builtins_registered()
        self._register(key, item, *aliases)

    def _register_builtin(
        self,
        key: str,
        item: Any,
        *aliases: str,
    ) -> None:
        """Register a BenchRep-owned built-in component."""
        self._register(key, item, *aliases)

    def _register(
        self,
        key: str,
        item: Any,
        *aliases: str,
    ) -> None:
        canonical_key = self._normalize_key(key)

        # Silently collapse duplicate names within one registration call.
        names = tuple(
            dict.fromkeys(
                self._normalize_key(name)
                for name in (key, *aliases)
            )
        )

        # Refuse overwrites.
        for name in names:
            if name in self._items:
                raise KeyError(
                    f"{self.name} registry already contains key {name!r}. "
                    "Choose a different name or remove the existing "
                    "registration."
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

    def aliases_by_canonical(self) -> dict[str, tuple[str, ...]]:
        """Return aliases grouped by canonical registry key."""
        _ensure_builtins_registered()

        grouped: dict[str, list[str]] = {
            key: []
            for key in self.canonical_keys()
        }

        for key, canonical_key in self._canonical_keys.items():
            if key != canonical_key:
                grouped[canonical_key].append(key)

        return {
            key: tuple(sorted(aliases))
            for key, aliases in grouped.items()
        }

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
MODELS = Registry(
    "model",
    custom_registration_supported=False,
)
RECONSTRUCTION_LOSSES = Registry("reconstruction loss")
REGULARIZATION_LOSSES = Registry("regularization loss")
CUSTOM_OBJECTIVE_LOSSES = Registry("custom objective loss")
OPTIMIZERS = Registry("optimizer")
LOGGERS = Registry("logger")
CALLBACKS = Registry("callback")
# Evaluation
EVAL_REDUCTIONS = Registry(
    "reduction",
    custom_registration_supported=False,
)
EVAL_CLUSTERING_METHODS = Registry(
    "clustering method",
    custom_registration_supported=False,
)
EVAL_INTERNAL_CLUSTERING_METRICS = Registry("internal clustering metric")
EVAL_EXTERNAL_CLUSTERING_METRICS = Registry("external clustering metric")
EVAL_EMBEDDING_METRICS = Registry("embedding metric")
EVAL_PREDICTABILITY_PROBES = Registry(
    "predictability probe",
    custom_registration_supported=False,
)
EVAL_RECONSTRUCTION_METRICS = Registry("reconstruction metric")