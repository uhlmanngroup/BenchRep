"""Runtime contracts for losses used by Composite models."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Final, Literal, TypeAlias, get_args

from torch import nn
from torch._C._jit_tree_views import Raise

from benchrep.architecture.composite_model_roles import (
    CompositeModelInputRole,
    CompositeModelOutputRole,
    CompositeModelTensorRole,
)
from benchrep.architecture.losses.custom_objective import BaseCustomObjectiveLoss


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

    A LossComponent using context ports must declare exactly ``batch`` sourced
    from the complete runtime batch and ``model_output`` sourced from the
    complete model-output mapping. These inputs are not configured through
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
    """Associate a loss module class with its runtime contract.

    ``component`` must be an ``nn.Module`` class and is instantiated from the
    loss term's constructor ``params``. ``runtime_inputs`` declares the keyword
    arguments through which BenchRep supplies data to its ``forward`` method.

    A loss must use either tensor ports, which require explicit Composite
    wiring, or context ports, which BenchRep supplies automatically. Context
    ports are reserved for ``BaseCustomObjectiveLoss`` subclasses and must
    declare exactly ``batch`` and ``model_output``.

    Declared runtime input names must be compatible with the component's
    ``forward`` signature. Every loss must return a scalar tensor; that
    invariant is enforced when the loss is executed.
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

        if has_context_ports:
            # Enforce the custom-objective interface.
            if not issubclass(
                    self.component,
                    BaseCustomObjectiveLoss,
            ):
                raise TypeError(
                    "LossComponent components using context ports must "
                    "subclass `BaseCustomObjectiveLoss`; got "
                    f"`{self.component.__module__}."
                    f"{self.component.__qualname__}`."
                )

            # Enforce the fixed context inputs supplied by BenchRep.
            context_inputs_by_parameter_name = {
                runtime_input.name: runtime_input.source
                for runtime_input in self.runtime_inputs
                if isinstance(runtime_input, LossContextPort)
            }

            required_context_inputs_by_parameter_name: dict[
                str,
                LossContextSource,
            ] = {
                "batch": "batch",
                "model_output": "model_output",
            }

            if (
                    context_inputs_by_parameter_name
                    != required_context_inputs_by_parameter_name
            ):
                raise ValueError(
                    "LossComponent context ports must declare exactly "
                    "`batch` sourced from `batch` and `model_output` "
                    "sourced from `model_output`; got "
                    f"{context_inputs_by_parameter_name}."
                )

        runtime_input_placeholders = {
            runtime_input.name: object()
            for runtime_input in self.runtime_inputs
        }
        forward_signature = inspect.signature(self.component.forward)

        try:
            forward_signature.bind(
                None,
                **runtime_input_placeholders,
            )
        except TypeError as error:
            raise TypeError(
                f"LossComponent runtime inputs "
                f"{tuple(runtime_input_placeholders)} are incompatible with "
                f"`{self.component.__qualname__}.forward"
                f"{forward_signature}`: {error}"
            ) from error
