"""Resolve Composite model configuration into an executable specification.

The configuration schema describes what the user requested. This module
resolves registry aliases, checks component contracts, validates data wiring,
and orders component calls so every produced tensor exists before it is used.

The resulting dataclasses contain instructions for later graph rendering and
model building; they do not instantiate PyTorch modules themselves.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

from torch import nn

from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
    ComponentMappingResult,
    ComponentTensorResult,
)
from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
)
from benchrep.architecture.composite_model_roles import (
    CompositeModelBatchMetadataRole,
    CompositeModelComponentKind,
    CompositeModelInputRole,
    CompositeModelOutputRole,
    CompositeModelTensorRole,
    TENSOR_STRUCTURE_BY_ROLE,
    TensorStructure,
)
from benchrep.architecture.decoders import BaseDecoder
from benchrep.architecture.encoders import BaseEncoder
from benchrep.architecture.heads import BaseHead
from benchrep.assembly.registries.core import (
    ARCHITECTURE_REGISTRIES_BY_KIND,
    LOSS_REGISTRIES_BY_ROLE,
)
from benchrep.assembly.schemas.composite_model_config_schema import (
    CompositeModelDeclarationsConfig,
    CompositeModelAssemblyStepConfig,
    CompositeModelComponentConfig,
)
from benchrep.assembly.schemas.training_config_schema import (
    SupportedLossRole,
    TrainingLossTermConfig,
)


# ---------------------------------------------------------------------------
# Shared resolver vocabulary
# ---------------------------------------------------------------------------
CompositeModelContextSource: TypeAlias = Literal[
    "batch",
    "model_output",
]


# Mapping from component registry to architecture interface.
_COMPONENT_INTERFACES_BY_KIND: Final[
    dict[CompositeModelComponentKind, type[nn.Module]]
] = {
    "encoder": BaseEncoder,
    "decoder": BaseDecoder,
    "head": BaseHead,
}


# ---------------------------------------------------------------------------
# Resolved specification types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompositeModelDeclarationsSpec:
    """Store roles under their arbitrary user-defined declaration names."""

    model_input_roles_by_name: dict[str, CompositeModelInputRole]
    batch_metadata_roles_by_name: dict[str, CompositeModelBatchMetadataRole]
    model_output_roles_by_name: dict[str, CompositeModelOutputRole]


@dataclass(frozen=True)
class CompositeModelComponentSpec:
    """Describe one registered component selected for the model."""

    component_kind: CompositeModelComponentKind
    registry_entry_name: str
    constructor_params: dict[str, Any]


@dataclass(frozen=True)
class CompositeModelAssemblyStepSpec:
    """Describe one component invocation and its resolved tensor bindings.

    Input-binding keys are component ``forward()`` parameter names. Their values
    are user-defined model input or output declaration names.

    ``results_to_model_outputs`` contains one model-output declaration name for
    a tensor result, or maps component result keys to model-output declaration
    names for a mapping result.
    """

    step_id: str
    component_id: str
    inputs_from_model_inputs: dict[str, str]
    inputs_from_model_outputs: dict[str, str]
    results_to_model_outputs: str | dict[str, str]


@dataclass(frozen=True)
class CompositeModelLossSpec:
    """Describe one resolved weighted loss invocation."""

    loss_role: SupportedLossRole
    configured_loss_name: str
    registry_entry_name: str
    weight: float
    constructor_params: dict[str, Any]

    inputs_from_model_inputs: dict[str, str]
    inputs_from_model_outputs: dict[str, str]

    # Used only by custom objectives: forward parameter -> complete context.
    context_inputs: dict[str, CompositeModelContextSource]


@dataclass(frozen=True)
class CompositeModelSpec:
    """Contain the complete resolved plan for one Composite model."""

    declarations: CompositeModelDeclarationsSpec
    components_by_id: dict[str, CompositeModelComponentSpec]
    assembly_steps_by_dependency_level: dict[
        int,
        tuple[CompositeModelAssemblyStepSpec, ...],
    ]
    loss_terms: tuple[CompositeModelLossSpec, ...]


def resolve_composite_model_config(
    *,
    declarations_config: CompositeModelDeclarationsConfig,
    components_config: dict[
        str,
        CompositeModelComponentConfig,
    ],
    assembly_config: dict[
        str,
        CompositeModelAssemblyStepConfig,
    ],
    losses_config: dict[
        SupportedLossRole,
        dict[str, TrainingLossTermConfig],
    ],
) -> CompositeModelSpec:
    """Resolve Composite model configuration into one executable specification.

    This function resolves each configured declaration, component, assembly
    step, and loss term, validates the complete data-flow graph, and orders
    component invocations by dependency level. It does not instantiate any
    PyTorch modules.
    """
    declarations_spec = _resolve_declarations(
        declarations_config
    )

    component_specs_by_id = {
        component_id: _resolve_component(
            component_id,
            component_config,
        )
        for component_id, component_config
        in components_config.items()
    }

    assembly_step_specs_by_id = {
        step_id: _resolve_assembly_step(
            step_id,
            assembly_step_config,
            declarations=declarations_spec,
            components_by_id=component_specs_by_id,
        )
        for step_id, assembly_step_config
        in assembly_config.items()
    }

    loss_term_specs: list[CompositeModelLossSpec] = []

    for loss_role, configured_losses in losses_config.items():
        loss_role: SupportedLossRole

        for configured_loss_name, loss_config in configured_losses.items():
            loss_term_spec = _resolve_loss_term(
                loss_role,
                configured_loss_name,
                loss_config,
                declarations=declarations_spec,
            )
            loss_term_specs.append(loss_term_spec)

    assembly_steps_by_dependency_level = (
        _validate_and_order_assembly_steps(
            assembly_step_specs_by_id,
            declarations=declarations_spec,
            components_by_id=component_specs_by_id,
        )
    )

    return CompositeModelSpec(
        declarations=declarations_spec,
        components_by_id=component_specs_by_id,
        assembly_steps_by_dependency_level=(
            assembly_steps_by_dependency_level
        ),
        loss_terms=tuple(loss_term_specs),
    )


def _validate_and_order_assembly_steps(
    assembly_steps_by_id: dict[
        str,
        CompositeModelAssemblyStepSpec,
    ],
    *,
    declarations: CompositeModelDeclarationsSpec,
    components_by_id: dict[str, CompositeModelComponentSpec],
) -> dict[int, tuple[CompositeModelAssemblyStepSpec, ...]]:
    """Validate the complete assembly graph and order its dependency levels."""

    used_component_ids = {
        assembly_step.component_id
        for assembly_step in assembly_steps_by_id.values()
    }
    unused_component_ids = sorted(
        set(components_by_id) - used_component_ids
    )

    if unused_component_ids:
        raise ValueError(
            "The following Composite model component IDs are configured "
            "under `composite_model_components` but are not used by any "
            f"assembly step: {unused_component_ids}."
        )

    producing_step_by_model_output: dict[str, str] = {}

    # Record which assembly step produces each declared model output.
    for step_id, assembly_step in assembly_steps_by_id.items():
        # Normalize both result contracts to the model-output declaration
        # names produced by this step. Since components can produce either
        # a tensor (e.g. encoder) or a mapping (e.g. variational head).
        if isinstance(assembly_step.results_to_model_outputs, str):
            produced_model_outputs = (
                assembly_step.results_to_model_outputs,
            )
        else:
            produced_model_outputs = tuple(
                assembly_step.results_to_model_outputs.values()
            )

        # Exact same output cannot be produced by more than one assembly step.
        for model_output_name in produced_model_outputs:
            existing_producing_step = (
                producing_step_by_model_output.get(model_output_name)
            )

            if existing_producing_step is not None:
                raise ValueError(
                    f"Composite model output {model_output_name!r} is "
                    f"produced by both assembly steps "
                    f"{existing_producing_step!r} and {step_id!r}. Each "
                    "declared model output must be produced exactly once."
                )

            producing_step_by_model_output[model_output_name] = step_id

    # All declared outputs must be produced by a component.
    unproduced_model_outputs = sorted(
        set(declarations.model_output_roles_by_name)
        - set(producing_step_by_model_output)
    )

    if unproduced_model_outputs:
        raise ValueError(
            "The following Composite model outputs are declared under "
            "`composite_model_declarations.produces` but are not produced "
            "by any assembly step: "
            f"{unproduced_model_outputs}."
        )

    # Map each consuming assembly step to the steps that produce its inputs:
    # {consumer_step_id: {producer_step_id, ...}}.
    producer_step_ids_by_consumer_step_id: dict[str, set[str]] = {}

    for step_id, assembly_step in assembly_steps_by_id.items():
        producer_step_ids: set[str] = set()

        for model_output_name in (
            assembly_step.inputs_from_model_outputs.values()
        ):
            producer_step_ids.add(
                producing_step_by_model_output[model_output_name]
            )

        producer_step_ids_by_consumer_step_id[step_id] = producer_step_ids

    assembly_steps_by_dependency_level: dict[
        int,
        tuple[CompositeModelAssemblyStepSpec, ...],
    ] = {}

    # Repeatedly collect one graph layer. Steps in the same layer depend only
    # on model inputs or on outputs produced by earlier layers.
    remaining_step_ids = set(assembly_steps_by_id)
    ordered_step_ids: set[str] = set()
    dependency_level = 0

    while remaining_step_ids:
        ready_step_ids: list[str] = []

        # A step is ready once every step producing one of its inputs has
        # already been assigned to an earlier dependency level.
        for step_id in assembly_steps_by_id:
            if step_id not in remaining_step_ids:
                continue

            required_producer_step_ids = (
                producer_step_ids_by_consumer_step_id[step_id]
            )

            if required_producer_step_ids.issubset(ordered_step_ids):
                ready_step_ids.append(step_id)

        # If steps remain but none can run next, those remaining steps
        # ultimately depend on one another and therefore form a cycle.
        if not ready_step_ids:
            unresolved_dependency_descriptions: list[str] = []

            for step_id in assembly_steps_by_id:
                if step_id not in remaining_step_ids:
                    continue

                unresolved_producer_step_ids = sorted(
                    producer_step_ids_by_consumer_step_id[
                        step_id
                    ].intersection(remaining_step_ids)
                )
                unresolved_dependency_descriptions.append(
                    f"{step_id!r} depends on "
                    f"{unresolved_producer_step_ids}"
                )

            raise ValueError(
                "Composite model assembly contains a dependency cycle: "
                f"{'; '.join(unresolved_dependency_descriptions)}."
            )

        # Store this complete layer in the original configuration order.
        assembly_steps_by_dependency_level[dependency_level] = tuple(
            assembly_steps_by_id[step_id]
            for step_id in ready_step_ids
        )

        # Remove this layer from future consideration. Its steps can now
        # satisfy the dependencies of consumers in the next iteration.
        ordered_step_ids.update(ready_step_ids)
        remaining_step_ids.difference_update(ready_step_ids)
        dependency_level += 1

    return assembly_steps_by_dependency_level


def _resolve_declarations(
    config: CompositeModelDeclarationsConfig,
) -> CompositeModelDeclarationsSpec:
    """Construct the resolved declarations spec used for Composite model assembly."""

    model_input_roles: dict[str, CompositeModelInputRole] = (
        config.expects.copy()
    )
    # Normalize optionally omitted batch metadata to an empty mapping.
    batch_metadata_roles: dict[str, CompositeModelBatchMetadataRole] = (
        config.batch_metadata.copy()
        if config.batch_metadata is not None
        else {}
    )
    model_output_roles: dict[str, CompositeModelOutputRole] = (
        config.produces.copy()
    )

    return CompositeModelDeclarationsSpec(
        model_input_roles_by_name=model_input_roles,
        batch_metadata_roles_by_name=batch_metadata_roles,
        model_output_roles_by_name=model_output_roles,
    )


def _resolve_component(
    component_id: str,
    config: CompositeModelComponentConfig,
) -> CompositeModelComponentSpec:
    """Resolve and validate one configured Composite architecture component."""

    component_config_path = (
        f"composite_model_components.{component_id}"
    )

    # Resolve the configured registry alias and retrieve its contract.
    registry = ARCHITECTURE_REGISTRIES_BY_KIND[config.kind]
    registry_entry_name = registry.resolve_key(config.name)
    entry = registry.get(registry_entry_name)

    if not isinstance(entry, ArchitectureComponent):
        raise TypeError(
            f"`{component_config_path}.name` resolves to "
            f"{config.kind} registry entry {registry_entry_name!r}, "
            "which must be an ArchitectureComponent."
        )

    # Confirm that the component implements its architecture interface.
    expected_interface = _COMPONENT_INTERFACES_BY_KIND[config.kind]

    if not issubclass(entry.component, expected_interface):
        raise TypeError(
            f"`{component_config_path}.name` resolves to "
            f"{config.kind} registry entry {registry_entry_name!r}, "
            f"which wraps `{entry.component.__module__}."
            f"{entry.component.__qualname__}` and must subclass "
            f"`{expected_interface.__name__}`."
        )

    # Check the configured constructor arguments without instantiation.
    constructor_params: dict[str, Any] = config.params.copy()

    try:
        inspect.signature(entry.component).bind(
            **constructor_params
        )
    except TypeError as exc:
        raise ValueError(
            f"Invalid constructor parameters at "
            f"`{component_config_path}.params` for {config.kind} "
            f"registry entry {registry_entry_name!r}: {exc}"
        ) from exc

    return CompositeModelComponentSpec(
        component_kind=config.kind,
        registry_entry_name=registry_entry_name,
        constructor_params=constructor_params,
    )


def _resolve_assembly_output_reference(
    configured_reference: str,
    *,
    output_config_path: str,
    declarations: CompositeModelDeclarationsSpec,
    supported_structures: tuple[TensorStructure, ...],
    component_id: str,
) -> str:
    """Resolve one assembly result binding to a declared model output."""

    source, separator, declaration_name = configured_reference.partition(".")

    if separator != "." or source != "produces":
        raise ValueError(
            f"`{output_config_path}` must be a `produces.<name>` "
            f"reference; got {configured_reference!r}."
        )

    if declaration_name not in declarations.model_output_roles_by_name:
        raise ValueError(
            f"`{output_config_path}` references "
            f"{configured_reference!r}, but {declaration_name!r} is not "
            "declared under `composite_model_declarations.produces`."
        )

    role = declarations.model_output_roles_by_name[declaration_name]
    structure = TENSOR_STRUCTURE_BY_ROLE[role]

    if structure not in supported_structures:
        raise ValueError(
            f"`{output_config_path}` binds a runtime result to "
            f"{configured_reference!r}, which has role {role!r} and "
            f"tensor structure {structure!r}, but component ID "
            f"{component_id!r} supports only "
            f"{supported_structures!r} for that runtime result."
        )

    return declaration_name


def _resolve_assembly_step(
    step_id: str,
    config: CompositeModelAssemblyStepConfig,
    *,
    declarations: CompositeModelDeclarationsSpec,
    components_by_id: dict[str, CompositeModelComponentSpec],
) -> CompositeModelAssemblyStepSpec:
    """Resolve and validate one Composite model component invocation."""

    assembly_step_config_path = f"composite_model_assembly.{step_id}"

    # Retrieve the resolved component and its runtime contract.
    component = components_by_id.get(config.component)

    if component is None:
        raise ValueError(
            f"`{assembly_step_config_path}.component` references undefined "
            f"component ID {config.component!r}. Available component IDs: "
            f"{tuple(sorted(components_by_id))}."
        )

    registry = ARCHITECTURE_REGISTRIES_BY_KIND[
        component.component_kind
    ]
    component_contract = registry.get(
        component.registry_entry_name
    )

    if not isinstance(component_contract, ArchitectureComponent):
        raise TypeError(
            f"Component ID {config.component!r} resolves to "
            f"{component.component_kind} registry entry "
            f"{component.registry_entry_name!r}, which must be an "
            "ArchitectureComponent."
        )

    # Require exact agreement with the component's forward parameters.
    runtime_inputs_by_name = {
        runtime_input.name: runtime_input
        for runtime_input in component_contract.runtime_inputs
    }

    configured_input_names = set(config.inputs)
    required_input_names = set(runtime_inputs_by_name)

    missing_inputs = sorted(
        required_input_names - configured_input_names
    )
    unexpected_inputs = sorted(
        configured_input_names - required_input_names
    )

    if missing_inputs or unexpected_inputs:
        problems: list[str] = []

        if missing_inputs:
            problems.append(f"missing keys {missing_inputs}")

        if unexpected_inputs:
            problems.append(f"unexpected keys {unexpected_inputs}")

        raise ValueError(
            f"`{assembly_step_config_path}.inputs` does not match the "
            f"runtime-input contract for component ID "
            f"{config.component!r}: {'; '.join(problems)}."
        )

    # Resolve each forward parameter according to its declared source.
    inputs_from_model_inputs: dict[str, str] = {}
    inputs_from_model_outputs: dict[str, str] = {}

    for input_name, runtime_input in runtime_inputs_by_name.items():
        configured_reference = config.inputs[input_name]
        source, separator, declaration_name = (
            configured_reference.partition(".")
        )

        if separator != "." or source not in {"expects", "produces"}:
            raise ValueError(
                f"`{assembly_step_config_path}.inputs.{input_name}` must "
                f"be an `expects.<name>` or `produces.<name>` reference; "
                f"got {configured_reference!r}."
            )

        if source == "expects":
            if declaration_name not in declarations.model_input_roles_by_name:
                raise ValueError(
                    f"`{assembly_step_config_path}.inputs.{input_name}` "
                    f"references {configured_reference!r}, but "
                    f"{declaration_name!r} is not declared under "
                    "`composite_model_declarations.expects`."
                )

            role: CompositeModelTensorRole = (
                declarations.model_input_roles_by_name[declaration_name]
            )
            inputs_from_model_inputs[input_name] = declaration_name

        else:
            if declaration_name not in declarations.model_output_roles_by_name:
                raise ValueError(
                    f"`{assembly_step_config_path}.inputs.{input_name}` "
                    f"references {configured_reference!r}, but "
                    f"{declaration_name!r} is not declared under "
                    "`composite_model_declarations.produces`."
                )

            role = declarations.model_output_roles_by_name[declaration_name]
            inputs_from_model_outputs[input_name] = declaration_name

        structure = TENSOR_STRUCTURE_BY_ROLE[role]

        if structure not in runtime_input.supported_structures:
            raise ValueError(
                f"`{assembly_step_config_path}.inputs.{input_name}` "
                f"references {configured_reference!r}, which has role "
                f"{role!r} and tensor structure {structure!r}, but "
                f"component ID {config.component!r} supports only "
                f"{runtime_input.supported_structures!r} for runtime "
                f"input {input_name!r}."
            )

    # Resolve a single-tensor or named-mapping component result.
    runtime_result = component_contract.runtime_result
    results_to_model_outputs: str | dict[str, str]

    if isinstance(runtime_result, ComponentTensorResult):
        if not isinstance(config.outputs, str):
            raise ValueError(
                f"`{assembly_step_config_path}.outputs` must be a direct "
                f"`produces.<name>` string because component ID "
                f"{config.component!r} returns one unnamed tensor; got "
                f"{config.outputs!r}."
            )

        results_to_model_outputs = _resolve_assembly_output_reference(
            config.outputs,
            output_config_path=f"{assembly_step_config_path}.outputs",
            declarations=declarations,
            supported_structures=runtime_result.supported_structures,
            component_id=config.component,
        )

    elif isinstance(runtime_result, ComponentMappingResult):
        if not isinstance(config.outputs, dict):
            raise ValueError(
                f"`{assembly_step_config_path}.outputs` must map runtime "
                f"result keys to `produces.<name>` references because "
                f"component ID {config.component!r} returns a named "
                f"mapping; got {config.outputs!r}."
            )

        runtime_outputs_by_name = {
            runtime_output.name: runtime_output
            for runtime_output in runtime_result.outputs
        }

        configured_output_names = set(config.outputs)
        required_output_names = set(runtime_outputs_by_name)

        missing_outputs = sorted(
            required_output_names - configured_output_names
        )
        unexpected_outputs = sorted(
            configured_output_names - required_output_names
        )

        if missing_outputs or unexpected_outputs:
            problems = []

            if missing_outputs:
                problems.append(f"missing keys {missing_outputs}")

            if unexpected_outputs:
                problems.append(f"unexpected keys {unexpected_outputs}")

            raise ValueError(
                f"`{assembly_step_config_path}.outputs` does not match "
                f"the runtime-result contract for component ID "
                f"{config.component!r}: {'; '.join(problems)}."
            )

        resolved_outputs: dict[str, str] = {}

        for output_name, runtime_output in runtime_outputs_by_name.items():
            configured_reference = config.outputs[output_name]

            resolved_outputs[output_name] = (
                _resolve_assembly_output_reference(
                    configured_reference,
                    output_config_path=(
                        f"{assembly_step_config_path}.outputs."
                        f"{output_name}"
                    ),
                    declarations=declarations,
                    supported_structures=(
                        runtime_output.supported_structures
                    ),
                    component_id=config.component,
                )
            )

        results_to_model_outputs = resolved_outputs

    else:
        raise TypeError(
            f"Component ID {config.component!r} declares an unsupported "
            f"runtime result contract: "
            f"{type(runtime_result).__name__}."
        )

    return CompositeModelAssemblyStepSpec(
        step_id=step_id,
        component_id=config.component,
        inputs_from_model_inputs=inputs_from_model_inputs,
        inputs_from_model_outputs=inputs_from_model_outputs,
        results_to_model_outputs=results_to_model_outputs,
    )


def _resolve_loss_term(
    loss_role: SupportedLossRole,
    configured_loss_name: str,
    config: TrainingLossTermConfig,
    *,
    declarations: CompositeModelDeclarationsSpec,
) -> CompositeModelLossSpec:
    """Resolve and validate one configured Composite model loss term."""

    loss_term_config_path = (
        f"losses.{loss_role}.{configured_loss_name}"
    )

    # Resolve the registry alias and retrieve the loss contract.
    registry = LOSS_REGISTRIES_BY_ROLE[loss_role]
    registry_entry_name = registry.resolve_key(configured_loss_name)
    loss_component = registry.get(registry_entry_name)

    if not isinstance(loss_component, LossComponent):
        raise TypeError(
            f"`{loss_term_config_path}` resolves to registry entry "
            f"{registry_entry_name!r}, which must be a LossComponent."
        )

    # Check the configured constructor arguments without instantiation.
    constructor_params: dict[str, Any] = config.params.copy()

    try:
        inspect.signature(loss_component.component).bind(
            **constructor_params
        )
    except TypeError as exc:
        raise ValueError(
            f"Invalid constructor parameters at "
            f"`{loss_term_config_path}.params` for registry entry "
            f"{registry_entry_name!r}: {exc}"
        ) from exc

    inputs_from_model_inputs: dict[str, str] = {}
    inputs_from_model_outputs: dict[str, str] = {}
    context_inputs: dict[str, CompositeModelContextSource] = {}

    if loss_role == "custom_objective":
        # Resolve automatically supplied full-context inputs.
        if config.composite_wiring is not None:
            raise ValueError(
                f"`{loss_term_config_path}.composite_wiring` must not "
                "be configured for a custom objective because its "
                "runtime inputs are supplied automatically."
            )

        # Custom objectives always receive these complete mappings, so there
        # is no per-loss Composite runtime contract to resolve.
        context_inputs = {
            "batch": "batch",
            "model_output": "model_output",
        }

    else:
        # Composite wiring cannot be inferred from a canonical calling
        # convention because it also requires declared semantic roles.
        if loss_component.runtime_inputs is None:
            raise ValueError(
                f"`{loss_term_config_path}` resolves to loss registry "
                f"entry {registry_entry_name!r}, which does not declare "
                "a Composite runtime contract."
            )

        # Validate and resolve explicit tensor inputs.
        if config.composite_wiring is None:
            raise ValueError(
                f"`{loss_term_config_path}.composite_wiring` is required "
                f"for loss role {loss_role!r} under a Composite model."
            )

        runtime_inputs_by_name = {
            runtime_input.name: runtime_input
            for runtime_input in loss_component.runtime_inputs
        }

        configured_input_names = set(config.composite_wiring)
        required_input_names = set(runtime_inputs_by_name)

        missing_inputs = sorted(
            required_input_names - configured_input_names
        )
        unexpected_inputs = sorted(
            configured_input_names - required_input_names
        )

        if missing_inputs or unexpected_inputs:
            problems: list[str] = []

            if missing_inputs:
                problems.append(f"missing keys {missing_inputs}")

            if unexpected_inputs:
                problems.append(
                    f"unexpected keys {unexpected_inputs}"
                )

            raise ValueError(
                f"`{loss_term_config_path}.composite_wiring` does not "
                f"match the runtime-input contract for loss registry "
                f"entry {registry_entry_name!r}: "
                f"{'; '.join(problems)}."
            )

        for input_name, runtime_input in runtime_inputs_by_name.items():
            configured_reference = config.composite_wiring[input_name]

            source, separator, declaration_name = (
                configured_reference.partition(".")
            )

            if (
                separator != "."
                or source not in {"expects", "produces"}
            ):
                raise ValueError(
                    f"`{loss_term_config_path}.composite_wiring."
                    f"{input_name}` must be an `expects.<name>` or "
                    f"`produces.<name>` reference; got "
                    f"{configured_reference!r}."
                )

            if source == "expects":
                if declaration_name not in declarations.model_input_roles_by_name:
                    raise ValueError(
                        f"`{loss_term_config_path}.composite_wiring."
                        f"{input_name}` references "
                        f"{configured_reference!r}, but "
                        f"{declaration_name!r} is not declared under "
                        "`composite_model_declarations.expects`."
                    )

                role: CompositeModelTensorRole = (
                    declarations.model_input_roles_by_name[declaration_name]
                )
                inputs_from_model_inputs[input_name] = declaration_name

            else:
                if declaration_name not in declarations.model_output_roles_by_name:
                    raise ValueError(
                        f"`{loss_term_config_path}.composite_wiring."
                        f"{input_name}` references "
                        f"{configured_reference!r}, but "
                        f"{declaration_name!r} is not declared under "
                        "`composite_model_declarations.produces`."
                    )

                role = declarations.model_output_roles_by_name[declaration_name]
                inputs_from_model_outputs[input_name] = declaration_name

            if role not in runtime_input.supported_roles:
                raise ValueError(
                    f"`{loss_term_config_path}.composite_wiring."
                    f"{input_name}` references "
                    f"{configured_reference!r}, which has role "
                    f"{role!r}, but loss registry entry "
                    f"{registry_entry_name!r} supports only "
                    f"{runtime_input.supported_roles!r} for runtime "
                    f"input {input_name!r}."
                )

    return CompositeModelLossSpec(
        loss_role=loss_role,
        configured_loss_name=configured_loss_name,
        registry_entry_name=registry_entry_name,
        weight=config.weight,
        constructor_params=constructor_params,
        inputs_from_model_inputs=inputs_from_model_inputs,
        inputs_from_model_outputs=inputs_from_model_outputs,
        context_inputs=context_inputs,
    )
