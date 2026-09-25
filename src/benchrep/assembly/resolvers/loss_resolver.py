"""Resolve configured losses into shared executable specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
)
from benchrep.assembly.registries.core import LOSS_REGISTRIES_BY_ROLE
from benchrep.assembly.schemas.training_config_schema import (
    SupportedLossRole,
    TrainingLossTermConfig,
)
from benchrep.architecture.composite_model_roles import (
    CompositeModelInputRole,
    CompositeModelOutputRole,
    CompositeModelTensorRole,
)


LossContextSource: TypeAlias = Literal[
    "batch",
    "model_output",
]

_LOSS_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class LossSpec:
    """Describe one resolved weighted loss invocation."""

    loss_role: SupportedLossRole
    loss_id: str
    registry_entry_name: str
    weight: float
    constructor_params: dict[str, Any]

    # Composite models populate these after declaration-aware wiring resolution.
    inputs_from_model_inputs: dict[str, str] = field(default_factory=dict)
    inputs_from_model_outputs: dict[str, str] = field(default_factory=dict)

    # Used by custom objectives: forward parameter -> complete context.
    context_inputs: dict[str, LossContextSource] = field(default_factory=dict)


def resolve_loss_configs(
    losses_config: Mapping[
        SupportedLossRole,
        Sequence[TrainingLossTermConfig],
    ],
) -> tuple[LossSpec, ...]:
    """Resolve model-family-independent properties of configured losses."""

    loss_specs: list[LossSpec] = []

    for loss_role, configured_losses in losses_config.items():
        for loss_index, loss_config in enumerate(configured_losses):
            loss_specs.append(
                _resolve_loss_term(
                    loss_role,
                    loss_config,
                    config_path=f"losses.{loss_role}[{loss_index}]",
                )
            )

    _validate_unique_loss_identities(loss_specs)

    return tuple(loss_specs)


def resolve_canonical_loss_configs(
    losses_config: Mapping[
        SupportedLossRole,
        Sequence[TrainingLossTermConfig],
    ],
) -> tuple[LossSpec, ...]:
    """Resolve and validate losses for a canonical model."""

    loss_specs = resolve_loss_configs(losses_config)

    for loss_spec in loss_specs:
        registry = LOSS_REGISTRIES_BY_ROLE[loss_spec.loss_role]
        loss_component = registry.get(
            loss_spec.registry_entry_name
        )

        if not isinstance(loss_component, LossComponent):
            raise TypeError(
                f"Loss registry entry "
                f"{loss_spec.registry_entry_name!r} for role "
                f"{loss_spec.loss_role!r} must be a LossComponent."
            )

        registry.validate_canonical_compatibility(
            loss_component
        )

    return loss_specs


def resolve_composite_loss_configs(
    losses_config: Mapping[
        SupportedLossRole,
        Sequence[TrainingLossTermConfig],
    ],
    *,
    model_input_roles_by_name: Mapping[str, CompositeModelInputRole],
    model_output_roles_by_name: Mapping[str, CompositeModelOutputRole],
) -> tuple[LossSpec, ...]:
    """Resolve losses and validate Composite declaration-aware wiring."""

    base_loss_specs = resolve_loss_configs(losses_config)

    configured_loss_terms = [
        (
            f"losses.{loss_role}[{loss_index}]",
            loss_config,
        )
        for loss_role, configured_losses in losses_config.items()
        for loss_index, loss_config in enumerate(configured_losses)
    ]

    return tuple(
        _resolve_composite_loss_spec(
            base_loss_spec,
            loss_config,
            config_path=config_path,
            model_input_roles_by_name=model_input_roles_by_name,
            model_output_roles_by_name=model_output_roles_by_name,
        )
        for (config_path, loss_config), base_loss_spec in zip(
            configured_loss_terms,
            base_loss_specs,
            strict=True,
        )
    )


def _resolve_composite_loss_spec(
    loss_spec: LossSpec,
    config: TrainingLossTermConfig,
    *,
    config_path: str,
    model_input_roles_by_name: Mapping[str, CompositeModelInputRole],
    model_output_roles_by_name: Mapping[str, CompositeModelOutputRole],
) -> LossSpec:
    """Resolve Composite-specific wiring for one resolved loss term."""

    loss_role = loss_spec.loss_role
    registry_entry_name = loss_spec.registry_entry_name

    # Retrieve the already-resolved loss contract.
    registry = LOSS_REGISTRIES_BY_ROLE[loss_role]
    loss_component = registry.get(registry_entry_name)

    if not isinstance(loss_component, LossComponent):
        raise TypeError(
            f"`{config_path}.name` resolves to registry entry "
            f"{registry_entry_name!r}, which must be a LossComponent."
        )

    inputs_from_model_inputs: dict[str, str] = {}
    inputs_from_model_outputs: dict[str, str] = {}
    context_inputs: dict[str, LossContextSource] = {}

    if loss_role == "custom_objective":
        # Resolve automatically supplied full-context inputs.
        if config.composite_wiring is not None:
            raise ValueError(
                f"`{config_path}.composite_wiring` must not be "
                "configured for a custom objective because its "
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
                f"`{config_path}` resolves to loss registry entry "
                f"{registry_entry_name!r}, which does not declare "
                "a Composite runtime contract."
            )

        # Validate and resolve explicit tensor inputs.
        if config.composite_wiring is None:
            raise ValueError(
                f"`{config_path}.composite_wiring` is required for "
                f"loss role {loss_role!r} under a Composite model."
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
                f"`{config_path}.composite_wiring` does not match "
                f"the runtime-input contract for loss registry entry "
                f"{registry_entry_name!r}: "
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
                    f"`{config_path}.composite_wiring.{input_name}` "
                    "must be an `expects.<name>` or "
                    f"`produces.<name>` reference; got "
                    f"{configured_reference!r}."
                )

            if source == "expects":
                if (
                    declaration_name
                    not in model_input_roles_by_name
                ):
                    raise ValueError(
                        f"`{config_path}.composite_wiring."
                        f"{input_name}` references "
                        f"{configured_reference!r}, but "
                        f"{declaration_name!r} is not declared under "
                        "`composite_model_declarations.expects`."
                    )

                role: CompositeModelTensorRole = (
                    model_input_roles_by_name[declaration_name]
                )
                inputs_from_model_inputs[input_name] = (
                    declaration_name
                )

            else:
                if (
                    declaration_name
                    not in model_output_roles_by_name
                ):
                    raise ValueError(
                        f"`{config_path}.composite_wiring."
                        f"{input_name}` references "
                        f"{configured_reference!r}, but "
                        f"{declaration_name!r} is not declared under "
                        "`composite_model_declarations.produces`."
                    )

                role = model_output_roles_by_name[declaration_name]
                inputs_from_model_outputs[input_name] = (
                    declaration_name
                )

            if role not in runtime_input.supported_roles:
                raise ValueError(
                    f"`{config_path}.composite_wiring.{input_name}` "
                    f"references {configured_reference!r}, which has "
                    f"role {role!r}, but loss registry entry "
                    f"{registry_entry_name!r} supports only "
                    f"{runtime_input.supported_roles!r} for runtime "
                    f"input {input_name!r}."
                )

    return replace(
        loss_spec,
        inputs_from_model_inputs=inputs_from_model_inputs,
        inputs_from_model_outputs=inputs_from_model_outputs,
        context_inputs=context_inputs,
    )


def _resolve_loss_term(
    loss_role: SupportedLossRole,
    config: TrainingLossTermConfig,
    *,
    config_path: str,
) -> LossSpec:
    """Resolve one configured loss independently of model-family wiring."""

    registry = LOSS_REGISTRIES_BY_ROLE[loss_role]

    registry_entry_name = registry.resolve_key(config.name)
    loss_component = registry.get(registry_entry_name)

    if not isinstance(loss_component, LossComponent):
        raise TypeError(
            f"`{config_path}.name` resolves to registry entry "
            f"{registry_entry_name!r}, which must be a LossComponent."
        )

    constructor_params: dict[str, Any] = config.params.copy()

    try:
        inspect.signature(loss_component.component).bind(
            **constructor_params
        )
    except TypeError as exc:
        raise ValueError(
            f"Invalid constructor parameters at `{config_path}.params` "
            f"for registry entry {registry_entry_name!r}: {exc}"
        ) from exc

    loss_id = (
        f"{config.id}_{registry_entry_name}"
        if config.id is not None
        else registry_entry_name
    )

    if _LOSS_ID_PATTERN.fullmatch(loss_id) is None:
        raise ValueError(
            f"`{config_path}` resolves to loss ID {loss_id!r}, which is not "
            "safe for use as a module key and metric-path component. Loss "
            "IDs must start with an alphanumeric character and contain only "
            "letters, numbers, underscores, or hyphens."
        )

    return LossSpec(
        loss_role=loss_role,
        loss_id=loss_id,
        registry_entry_name=registry_entry_name,
        weight=config.weight,
        constructor_params=constructor_params,
    )


def _validate_unique_loss_identities(
    loss_specs: Sequence[LossSpec],
) -> None:
    """Reject role-scoped loss IDs or generated metric names that collide."""

    loss_ids_by_role: dict[SupportedLossRole, set[str]] = {}
    metric_owners_by_role: dict[
        SupportedLossRole,
        dict[str, str],
    ] = {}

    for loss_spec in loss_specs:
        loss_role = loss_spec.loss_role
        loss_id = loss_spec.loss_id

        role_loss_ids = loss_ids_by_role.setdefault(
            loss_role,
            set(),
        )

        if loss_id in role_loss_ids:
            raise ValueError(
                f"Multiple configured losses under role {loss_role!r} "
                f"resolve to loss ID {loss_id!r}. Loss IDs must be unique "
                "within a role; configure a distinct `id` for one of the "
                "colliding loss terms."
            )

        role_loss_ids.add(loss_id)

        metric_owners = metric_owners_by_role.setdefault(
            loss_role,
            {},
        )

        for metric_name in (
            loss_id,
            f"{loss_id}_weighted",
        ):
            existing_owner = metric_owners.get(metric_name)

            if existing_owner is not None:
                raise ValueError(
                    f"Loss IDs {existing_owner!r} and {loss_id!r} under role "
                    f"{loss_role!r} produce a metric-name collision at "
                    f"{metric_name!r}. BenchRep logs each loss using both its "
                    "`loss_id` and `<loss_id>_weighted`, so these two IDs cannot "
                    "be used together. Configure a different `id` for one of the "
                    "colliding loss terms."
                )

            metric_owners[metric_name] = loss_id
