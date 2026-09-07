from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
)
from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
)
from benchrep.evaluation.metrics import EvaluationMetric


RegistryEntry: TypeAlias = (
    Callable[..., Any]
    | ArchitectureComponent
    | LossComponent
    | EvaluationMetric
)

RegistryEntryType: TypeAlias = (
    type[ArchitectureComponent]
    | type[LossComponent]
    | type[EvaluationMetric]
)


class Registry:
    """Map configuration names to registered implementations or contract wrappers.

    BenchRep registries use four entry systems.

    Registries with ``entry_type=None`` store callables directly. Depending on
    the registry, an entry may be a class constructor, a factory function, or
    an execution function. The registry-specific discovery contract describes
    the callable's required interface and return value. ``get()`` returns the
    callable unchanged, while ``create()`` invokes it with the supplied
    arguments when construction is appropriate.

    Architecture registries store ``ArchitectureComponent`` entries. The wrapper
    carries both the module class and its Composite runtime contract. ``get()``
    returns the complete wrapper for discovery and graph validation, whereas
    ``create()`` unwraps ``entry.component`` and instantiates the module class.

    Loss registries store ``LossComponent`` entries. The wrapper carries the loss
    module class and its Composite runtime-input contract. ``get()`` returns the
    complete wrapper, while ``create()`` unwraps ``entry.component`` and
    instantiates the loss module.

    Evaluation-metric registries store ``EvaluationMetric`` entries. Evaluation
    machinery retrieves and interprets the complete wrapper, including its
    callable and result contract. These entries are not constructors and
    therefore cannot be used through ``create()``.

    ``entry_type`` declares and enforces the wrapper required by a particular
    registry. Users register one appropriately formed entry under a name and
    then select that name through configuration; component instantiation and
    metric execution remain BenchRep responsibilities.
    """

    def __init__(
        self,
        name: str,
        *,
        custom_registration_supported: bool = True,
        entry_type: RegistryEntryType | None = None,
    ) -> None:

        if (
            entry_type is not None
            and not isinstance(entry_type, type)
        ):
            raise TypeError(
                "entry_type must be a registry-entry class or None, got "
                f"{type(entry_type).__name__}."
            )

        self.name = name
        self._custom_registration_supported = (
            custom_registration_supported
        )
        self._entry_type = entry_type
        self._items: dict[str, RegistryEntry] = {}
        self._canonical_keys: dict[str, str] = {}

    @property
    def entry_type(self) -> RegistryEntryType | None:
        """Required wrapper type, or None for directly callable entries."""

        return self._entry_type

    @property
    def custom_registration_supported(self) -> bool:
        """Whether users may register custom components."""
        return self._custom_registration_supported

    def register(
        self,
        key: str,
        item: RegistryEntry,
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
        item: RegistryEntry,
        *aliases: str,
    ) -> None:
        """Register a BenchRep-owned built-in component."""
        self._register(key, item, *aliases)

    def _register(
        self,
        key: str,
        item: RegistryEntry,
        *aliases: str,
    ) -> None:
        canonical_key = self._normalize_key(key)
        self._validate_entry(item)

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

    def _validate_entry(self, item: Any) -> None:
        """Validate a value against this registry's required entry type."""

        entry_type = self.entry_type

        if entry_type is None:
            if not callable(item):
                raise TypeError(
                    f"{self.name} registry entries must be callable, got "
                    f"{type(item).__name__}."
                )

            return

        if not isinstance(item, entry_type):
            raise TypeError(
                f"{self.name} registry entries must be "
                f"{entry_type.__name__} instances, got "
                f"{type(item).__name__}."
            )

    def get(self, key: str) -> RegistryEntry:
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
        """Instantiate the implementation represented by a registry entry."""

        entry = self.get(key)

        if isinstance(entry, (ArchitectureComponent, LossComponent)):
            factory = entry.component
        elif callable(entry):
            factory = entry
        else:
            raise TypeError(
                f"{self.name} registry entries of type "
                f"{type(entry).__name__} cannot be instantiated with "
                "Registry.create()."
            )

        return factory(**kwargs)

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
ENCODERS = Registry(
    "encoder",
    entry_type=ArchitectureComponent,
)
DECODERS = Registry(
    "decoder",
    entry_type=ArchitectureComponent,
)
HEADS = Registry(
    "head",
    entry_type=ArchitectureComponent,
)
MODELS = Registry(
    "model",
    custom_registration_supported=False,
)
RECONSTRUCTION_LOSSES = Registry(
    "reconstruction loss",
    entry_type=LossComponent,
)
REGULARIZATION_LOSSES = Registry(
    "regularization loss",
    entry_type=LossComponent,
)
CONTRASTIVE_LOSSES = Registry(
    "contrastive loss",
    entry_type=LossComponent,
)
CLASSIFICATION_LOSSES = Registry(
    "classification loss",
    entry_type=LossComponent,
)
REGRESSION_LOSSES = Registry(
    "regression loss",
    entry_type=LossComponent,
)
CUSTOM_OBJECTIVE_LOSSES = Registry(
    "custom objective loss",
    entry_type=LossComponent,
)
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
EVAL_INTERNAL_CLUSTERING_METRICS = Registry(
    "internal clustering metric",
    entry_type=EvaluationMetric,
)
EVAL_EXTERNAL_CLUSTERING_METRICS = Registry(
    "external clustering metric",
    entry_type=EvaluationMetric,
)
EVAL_EMBEDDING_METRICS = Registry(
    "embedding metric",
    entry_type=EvaluationMetric,
)
EVAL_PREDICTABILITY_PROBES = Registry(
    "predictability probe",
    custom_registration_supported=False,
)
EVAL_RECONSTRUCTION_METRICS = Registry(
    "reconstruction metric",
    entry_type=EvaluationMetric,
)
