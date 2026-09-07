"""Runtime contracts for losses used by Composite models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, get_args

from torch import nn

from benchrep.architecture.composite_model_roles import (
    CompositeModelInputRole,
    CompositeModelOutputRole,
    CompositeModelTensorRole,
)


_VALID_COMPOSITE_MODEL_TENSOR_ROLES: Final[frozenset[str]] = frozenset(
    (
        *get_args(CompositeModelInputRole),
        *get_args(CompositeModelOutputRole),
    )
)

LossContextSource: TypeAlias = Literal[
    "batch",
    "model_output",
]

_VALID_LOSS_CONTEXT_SOURCES: Final[frozenset[str]] = frozenset(
    get_args(LossContextSource)
)


def _validate_port_name(name: str, *, class_name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(
            f"{class_name}.name must be a string, got "
            f"{type(name).__name__}."
        )

    name = name.strip()

    if not name:
        raise ValueError(f"{class_name}.name must be a non-empty string.")

    return name


@dataclass(frozen=True)
class LossTensorPort:
    """Describe one loss argument supplied from a Composite declaration.

    ``name`` must match the corresponding argument in the loss component's
    ``forward`` method. ``supported_roles`` declares the semantic tensor roles
    that may be wired to that argument.
    """

    name: str
    supported_roles: tuple[CompositeModelTensorRole, ...]

    def __post_init__(self) -> None:
        name = _validate_port_name(
            self.name,
            class_name="LossTensorPort",
        )
        object.__setattr__(self, "name", name)

        if not isinstance(self.supported_roles, tuple):
            raise TypeError(
                "LossTensorPort.supported_roles must be a tuple, got "
                f"{type(self.supported_roles).__name__}."
            )

        if not self.supported_roles:
            raise ValueError(
                "LossTensorPort.supported_roles must not be empty."
            )

        if not all(
            isinstance(role, str)
            for role in self.supported_roles
        ):
            raise TypeError(
                "LossTensorPort.supported_roles must contain only strings."
            )

        unsupported_roles = sorted(
            set(self.supported_roles)
            - _VALID_COMPOSITE_MODEL_TENSOR_ROLES
        )

        if unsupported_roles:
            raise ValueError(
                "LossTensorPort.supported_roles contains unsupported roles: "
                f"{unsupported_roles}."
            )

        if len(set(self.supported_roles)) != len(self.supported_roles):
            raise ValueError(
                "LossTensorPort.supported_roles must not contain duplicates."
            )


@dataclass(frozen=True)
class LossContextPort:
    """Describe one automatically supplied custom-objective argument.

    Context ports receive either the complete runtime batch or the complete
    model-output mapping. They are not configured through
    ``composite_wiring``.
    """

    name: str
    source: LossContextSource

    def __post_init__(self) -> None:
        name = _validate_port_name(
            self.name,
            class_name="LossContextPort",
        )
        object.__setattr__(self, "name", name)

        if not isinstance(self.source, str):
            raise TypeError(
                "LossContextPort.source must be a string, got "
                f"{type(self.source).__name__}."
            )

        if self.source not in _VALID_LOSS_CONTEXT_SOURCES:
            raise ValueError(
                f"Unsupported loss context source {self.source!r}. "
                f"Available sources: "
                f"{tuple(sorted(_VALID_LOSS_CONTEXT_SOURCES))}."
            )


LossPort: TypeAlias = LossTensorPort | LossContextPort


@dataclass(frozen=True)
class LossComponent:
    """Associate a loss module class with its Composite runtime contract.

    ``component`` is instantiated from the loss term's constructor ``params``.
    ``runtime_inputs`` declares the keyword arguments accepted by its
    ``forward`` method.

    A loss must use either tensor ports, which require explicit Composite
    wiring, or context ports, which BenchRep supplies automatically for custom
    objectives. Every loss is required to return a scalar tensor, so that
    invariant is enforced globally rather than repeated in each contract.
    """

    component: type[nn.Module]
    runtime_inputs: tuple[LossPort, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.component, type)
            or not issubclass(self.component, nn.Module)
        ):
            raise TypeError(
                "LossComponent.component must be an nn.Module class."
            )

        if not isinstance(self.runtime_inputs, tuple):
            raise TypeError(
                "LossComponent.runtime_inputs must be a tuple, got "
                f"{type(self.runtime_inputs).__name__}."
            )

        if not self.runtime_inputs:
            raise ValueError(
                "LossComponent.runtime_inputs must not be empty."
            )

        if not all(
            isinstance(runtime_input, (LossTensorPort, LossContextPort))
            for runtime_input in self.runtime_inputs
        ):
            raise TypeError(
                "LossComponent.runtime_inputs must contain only "
                "LossTensorPort or LossContextPort instances."
            )

        input_names = [
            runtime_input.name
            for runtime_input in self.runtime_inputs
        ]

        if len(set(input_names)) != len(input_names):
            raise ValueError(
                "LossComponent.runtime_inputs must have unique names."
            )

        has_tensor_ports = any(
            isinstance(runtime_input, LossTensorPort)
            for runtime_input in self.runtime_inputs
        )
        has_context_ports = any(
            isinstance(runtime_input, LossContextPort)
            for runtime_input in self.runtime_inputs
        )

        if has_tensor_ports and has_context_ports:
            raise ValueError(
                "LossComponent.runtime_inputs cannot mix tensor and "
                "context ports."
            )