"""Build configured loss terms from role-specific registries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from benchrep.architecture.losses.base import LossTerm
from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
)
from benchrep.assembly.schemas import TrainingLossTermConfig
from benchrep.assembly.schemas.training_config_schema import (
    SupportedLossRole,
)
from benchrep.records import get_run_logger
from benchrep.assembly.registries.core import (
    LOSS_REGISTRIES_BY_ROLE,
)

LossTermInput: TypeAlias = TrainingLossTermConfig | LossTerm


def build_loss_terms(
    losses_by_role: Mapping[
        SupportedLossRole,
        Mapping[str, LossTermInput],
    ],
) -> dict[SupportedLossRole, dict[str, LossTerm]]:
    """Build and log configured loss terms for a canonical model."""

    resolved_losses: dict[
        SupportedLossRole,
        dict[str, LossTerm],
    ] = {}

    sources_by_role: dict[
        SupportedLossRole,
        dict[str, str],
    ] = {}

    for role, configured_losses in losses_by_role.items():
        role: SupportedLossRole
        registry = LOSS_REGISTRIES_BY_ROLE.get(role)

        if registry is None:
            raise ValueError(
                f"No loss registry is configured for role {role!r}."
            )

        resolved_losses[role] = {}
        sources_by_role[role] = {}

        for loss_name, loss_config in configured_losses.items():
            if isinstance(loss_config, LossTerm):
                loss_term = loss_config
                source = "provided"
            else:
                loss_component = registry.get(loss_name)

                if not isinstance(loss_component, LossComponent):
                    raise TypeError(
                        f"Loss registry entry {loss_name!r} for role "
                        f"{role!r} must be a LossComponent."
                    )

                registry.validate_canonical_compatibility(
                    loss_component
                )

                loss_term = LossTerm(
                    loss=registry.create(
                        loss_name,
                        **loss_config.params,
                    ),
                    weight=loss_config.weight,
                )
                source = "config"

            resolved_losses[role][loss_name] = loss_term
            sources_by_role[role][loss_name] = source

    _log_resolved_losses(
        losses_by_role=resolved_losses,
        sources_by_role=sources_by_role,
    )

    return resolved_losses


def _log_resolved_losses(
    *,
    losses_by_role: Mapping[
        SupportedLossRole,
        Mapping[str, LossTerm],
    ],
    sources_by_role: Mapping[
        SupportedLossRole,
        Mapping[str, str],
    ],
) -> None:
    run_log = get_run_logger()
    descriptions: list[str] = []

    for role, loss_terms in losses_by_role.items():
        role: SupportedLossRole
        if not loss_terms:
            continue

        sources = sources_by_role[role]

        terms = ", ".join(
            (
                f"{loss_name} ({sources[loss_name]})"
                f" -> {type(loss_term.loss).__name__}"
                f" (weight={loss_term.weight})"
            )
            for loss_name, loss_term in loss_terms.items()
        )

        descriptions.append(f"{role}=[{terms}]")

    run_log.info(
        "Resolved losses: %s",
        "; ".join(descriptions),
    )