"""Runtime wiring contracts for architecture components used by Composite models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias, get_args

from torch import nn

from benchrep.architecture.composite_model_roles import TensorStructure


_VALID_TENSOR_STRUCTURES: Final[frozenset[str]] = frozenset(
    get_args(TensorStructure)
)


def _validate_structures(
    supported_structures: tuple[TensorStructure, ...],
    *,
    field_name: str,
) -> None:
    if not isinstance(supported_structures, tuple):
        raise TypeError(
            f"{field_name} must be a tuple, got "
            f"{type(supported_structures).__name__}."
        )

    if not supported_structures:
        raise ValueError(f"{field_name} must not be empty.")

    if not all(
        isinstance(structure, str)
        for structure in supported_structures
    ):
        raise TypeError(
            f"{field_name} must contain only strings."
        )

    unsupported_structures = sorted(
        set(supported_structures) - _VALID_TENSOR_STRUCTURES
    )

    if unsupported_structures:
        raise ValueError(
            f"{field_name} contains unsupported tensor structures: "
            f"{unsupported_structures}. Available structures: "
            f"{tuple(sorted(_VALID_TENSOR_STRUCTURES))}."
        )

    if len(set(supported_structures)) != len(supported_structures):
        raise ValueError(f"{field_name} must not contain duplicates.")


@dataclass(frozen=True)
class ComponentPort:
    """Describe one named runtime port.

    ``supported_structures`` lists the supported per-sample tensor structures, excluding
    the batch dimension. Ports represent component input arguments or keys in a
    mapping returned by a component.
    """

    name: str
    supported_structures: tuple[TensorStructure, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(
                "ComponentPort.name must be a string, got "
                f"{type(self.name).__name__}."
            )

        name = self.name.strip()

        if not name:
            raise ValueError(
                "ComponentPort.name must be a non-empty string."
            )

        object.__setattr__(self, "name", name)

        _validate_structures(
            self.supported_structures,
            field_name=f"ComponentPort {name!r}.supported_structures",
        )


@dataclass(frozen=True)
class ComponentTensorResult:
    """Describe a component that returns one unnamed tensor.

    ``supported_structures`` lists the tensor structures the component can produce. The
    assembly binds the returned tensor directly to one declared ``produces``
    value without an artificial output-port name.
    """

    supported_structures: tuple[TensorStructure, ...]

    def __post_init__(self) -> None:
        _validate_structures(
            self.supported_structures,
            field_name="ComponentTensorResult.supported_structures",
        )


@dataclass(frozen=True)
class ComponentMappingResult:
    """Describe a component that returns a mapping of named tensors.

    ``outputs`` declares the exact mapping keys exposed by the component and the
    tensor structures supported by each value.
    """

    outputs: tuple[ComponentPort, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, tuple):
            raise TypeError(
                "ComponentMappingResult.outputs must be a tuple, got "
                f"{type(self.outputs).__name__}."
            )

        if not self.outputs:
            raise ValueError(
                "ComponentMappingResult.outputs must not be empty."
            )

        if not all(
            isinstance(output, ComponentPort)
            for output in self.outputs
        ):
            raise TypeError(
                "ComponentMappingResult.outputs must contain only "
                "ComponentPort instances."
            )

        output_names = [
            output.name
            for output in self.outputs
        ]

        if len(set(output_names)) != len(output_names):
            raise ValueError(
                "ComponentMappingResult.outputs must have unique names."
            )


ComponentResult: TypeAlias = (
    ComponentTensorResult | ComponentMappingResult
)


@dataclass(frozen=True)
class ArchitectureComponent:
    """Associate an architecture module class with its Composite runtime contract.

    ``component`` is instantiated from the constructor parameters supplied in
    ``composite_model_components``. Constructor parameters are intentionally
    separate from this contract.

    ``runtime_inputs`` declares the required keyword arguments accepted by the
    component's ``forward`` method and the tensor structures supported by each
    argument. ``runtime_result`` declares whether ``forward`` returns one tensor or
    a mapping of named tensors.

    Composite resolution uses this metadata to validate assembly wiring.
    Canonical model builders instantiate the registered component without needing
    to interpret its Composite runtime contract.
    """

    component: type[nn.Module]
    runtime_inputs: tuple[ComponentPort, ...]
    runtime_result: ComponentResult

    def __post_init__(self) -> None:
        if (
            not isinstance(self.component, type)
            or not issubclass(self.component, nn.Module)
        ):
            raise TypeError(
                "ArchitectureComponent.component must be an nn.Module "
                "class."
            )

        if not isinstance(self.runtime_inputs, tuple):
            raise TypeError(
                "ArchitectureComponent.runtime_inputs must be a tuple, got "
                f"{type(self.runtime_inputs).__name__}."
            )

        if not self.runtime_inputs:
            raise ValueError(
                "ArchitectureComponent.runtime_inputs must not be empty."
            )

        if not all(
            isinstance(runtime_input, ComponentPort)
            for runtime_input in self.runtime_inputs
        ):
            raise TypeError(
                "ArchitectureComponent.runtime_inputs must contain only "
                "ComponentPort instances."
            )

        input_names = [
            runtime_input.name
            for runtime_input in self.runtime_inputs
        ]

        if len(set(input_names)) != len(input_names):
            raise ValueError(
                "ArchitectureComponent.runtime_inputs must have unique names."
            )

        if not isinstance(
            self.runtime_result,
            (ComponentTensorResult, ComponentMappingResult),
        ):
            raise TypeError(
                "ArchitectureComponent.runtime_result must be a "
                "ComponentTensorResult or ComponentMappingResult."
            )