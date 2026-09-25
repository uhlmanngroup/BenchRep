"""Resolve configured losses into shared executable specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
)
from benchrep.assembly.registries.core import LOSS_REGISTRIES_BY_ROLE
from benchrep.assembly.schemas.training_config_schema import (
    SupportedLossRole,
    TrainingLossTermConfig,
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
                    f"Loss IDs {existing_owner!r} and {loss_id!r} under "
                    f"role {loss_role!r} generate the same metric name "
                    f"{metric_name!r}. Configure a different `id` for one "
                    "of these loss terms."
                )

            metric_owners[metric_name] = loss_id
