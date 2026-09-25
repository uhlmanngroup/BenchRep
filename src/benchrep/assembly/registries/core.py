from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any, TypeAlias, Final

from torch import nn

from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
)
from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
)
from benchrep.architecture.losses.custom_objective import (
    BaseCustomObjectiveLoss,
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

    Loss registries are ``LossRegistry`` instances and store ``LossComponent``
    entries. The wrapper always identifies the loss module class and may also
    carry an explicit Composite runtime-input contract. Ordinary entries without
    that contract are canonical-only, while custom objectives use their fixed
    registry-level interface under both canonical and Composite models.
    ``get()`` returns the wrapper, while ``create()`` unwraps and instantiates
    its component.

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


class LossRegistry(Registry):
    """Validate loss components against role-level calling conventions.

    LossComponent validates explicitly declared Composite runtime inputs. This
    registry handles information the component does not know by itself: whether
    the loss role has a fixed canonical calling convention and whether its
    component must inherit from a particular base class. A component without
    runtime inputs is accepted only when the registry defines such a convention.
    """

    def __init__(
        self,
        name: str,
        *,
        canonical_runtime_input_names: tuple[str, ...] | None = None,
        required_component_base: type[nn.Module] = nn.Module,
        explicit_composite_runtime_inputs_supported: bool = True,
    ) -> None:
        super().__init__(
            name,
            entry_type=LossComponent,
        )

        self._canonical_runtime_input_names = (
            canonical_runtime_input_names
        )
        self._required_component_base = required_component_base
        self._explicit_composite_runtime_inputs_supported = (
            explicit_composite_runtime_inputs_supported
        )

    @property
    def canonical_runtime_input_names(
        self,
    ) -> tuple[str, ...] | None:
        """Return the fixed runtime inputs used by canonical models."""

        return self._canonical_runtime_input_names

    @property
    def required_component_base(self) -> type[nn.Module]:
        """Return the base class required for registered loss components."""

        return self._required_component_base

    @property
    def explicit_composite_runtime_inputs_supported(self) -> bool:
        """Whether entries may declare Composite runtime inputs."""

        return self._explicit_composite_runtime_inputs_supported

    def _validate_entry(self, item: Any) -> None:
        """Validate one loss component against its registry-level contract."""

        super()._validate_entry(item)

        if not issubclass(
            item.component,
            self.required_component_base,
        ):
            raise TypeError(
                f"{self.name} registry components must subclass "
                f"`{self.required_component_base.__name__}`; got "
                f"`{item.component.__module__}."
                f"{item.component.__qualname__}`."
            )

        if (
            item.runtime_inputs is not None
            and not self.explicit_composite_runtime_inputs_supported
        ):
            raise ValueError(
                f"{self.name} registry entries must not declare "
                "`runtime_inputs` because BenchRep supplies this role's "
                "fixed runtime inputs automatically."
            )

        if item.runtime_inputs is not None:
            return

        if self.canonical_runtime_input_names is None:
            raise ValueError(
                f"{self.name} registry entries must declare "
                "`runtime_inputs` because this loss role has no fixed "
                "canonical calling convention."
            )

        self.validate_canonical_compatibility(item)

    def validate_canonical_compatibility(
        self,
        loss_component: LossComponent,
    ) -> None:
        """Validate compatibility with this role's canonical loss call."""

        canonical_runtime_input_names = (
            self.canonical_runtime_input_names
        )

        if canonical_runtime_input_names is None:
            raise ValueError(
                f"{self.name} registry entries cannot be used by canonical "
                "models because this loss role has no canonical calling "
                "convention."
            )

        runtime_input_placeholders = {
            runtime_input_name: object()
            for runtime_input_name in canonical_runtime_input_names
        }
        forward_signature = inspect.signature(
            loss_component.component.forward
        )

        try:
            forward_signature.bind(
                None,
                **runtime_input_placeholders,
            )
        except TypeError as error:
            raise TypeError(
                f"{self.name} registry component "
                f"`{loss_component.component.__module__}."
                f"{loss_component.component.__qualname__}` must accept "
                f"the canonical runtime inputs "
                f"{canonical_runtime_input_names}; "
                f"`forward{forward_signature}` is incompatible: {error}"
            ) from error


def _ensure_builtins_registered() -> None:
    """Ensure BenchRep built-in registry entries are available."""

    from benchrep.assembly.registries.builtins import register_builtins

    register_builtins()


# Data
DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")

# Architecture
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

ARCHITECTURE_REGISTRIES_BY_KIND: Final[dict[str, Registry]] = {
    "encoder": ENCODERS,
    "decoder": DECODERS,
    "head": HEADS,
}

# Training
OPTIMIZERS = Registry("optimizer")
LOGGERS = Registry("logger")
CALLBACKS = Registry("callback")

# Losses
RECONSTRUCTION_LOSSES = LossRegistry(
    "reconstruction loss",
    canonical_runtime_input_names=(
        "reconstruction",
        "target",
    ),
)
REGULARIZATION_LOSSES = LossRegistry(
    "regularization loss",
    canonical_runtime_input_names=(
        "z_mu",
        "z_logvar",
    ),
)
CONTRASTIVE_LOSSES = LossRegistry(
    "contrastive loss",
)
CLASSIFICATION_LOSSES = LossRegistry(
    "classification loss",
)
REGRESSION_LOSSES = LossRegistry(
    "regression loss",
)
CUSTOM_OBJECTIVE_LOSSES = LossRegistry(
    "custom objective loss",
    canonical_runtime_input_names=(
        "batch",
        "model_output",
    ),
    required_component_base=BaseCustomObjectiveLoss,
    explicit_composite_runtime_inputs_supported=False,
)

LOSS_REGISTRIES_BY_ROLE: Final[dict[str, LossRegistry]] = {
    "reconstruction": RECONSTRUCTION_LOSSES,
    "regularization": REGULARIZATION_LOSSES,
    "contrastive": CONTRASTIVE_LOSSES,
    "classification": CLASSIFICATION_LOSSES,
    "regression": REGRESSION_LOSSES,
    "custom_objective": CUSTOM_OBJECTIVE_LOSSES,
}

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
