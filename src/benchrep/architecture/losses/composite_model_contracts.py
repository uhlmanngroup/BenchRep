"""Runtime contracts for losses used by Composite models."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Final, get_args

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
class LossComponent:
    """Associate a loss module class with its optional Composite contract.

    ``component`` must be an ``nn.Module`` class and is instantiated from the
    loss term's constructor ``params``.

    ``runtime_inputs`` declares the keyword arguments and semantic roles used
    to wire an ordinary loss under a Composite model. ``None`` means that no
    Composite runtime contract is declared; it does not generate or imply a
    default contract.

    When runtime inputs are declared, their names must be compatible with the
    component's ``forward`` signature. Every loss must return a scalar tensor;
    that invariant is enforced when the loss is executed.
    """

    component: type[nn.Module]
    runtime_inputs: tuple[LossTensorPort, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.component, type)
            or not issubclass(self.component, nn.Module)
        ):
            raise TypeError(
                "LossComponent.component must be an nn.Module class."
            )

        # There is no Composite contract to validate here. Whether this loss
        # can be used without one depends on its registry role and model type,
        # which LossComponent does not know and is validated later.
        if self.runtime_inputs is None:
            return

        if not isinstance(self.runtime_inputs, tuple):
            raise TypeError(
                "LossComponent.runtime_inputs must be a tuple or None, got "
                f"{type(self.runtime_inputs).__name__}."
            )

        if not self.runtime_inputs:
            raise ValueError(
                "LossComponent.runtime_inputs must not be empty."
            )

        if not all(
            isinstance(runtime_input, LossTensorPort)
            for runtime_input in self.runtime_inputs
        ):
            raise TypeError(
                "LossComponent.runtime_inputs must contain only "
                "LossTensorPort instances."
            )

        input_names = [
            runtime_input.name
            for runtime_input in self.runtime_inputs
        ]

        if len(set(input_names)) != len(input_names):
            raise ValueError(
                "LossComponent.runtime_inputs must have unique names."
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
