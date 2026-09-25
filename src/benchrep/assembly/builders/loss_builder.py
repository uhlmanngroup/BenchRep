"""Instantiate resolved loss specifications."""

from __future__ import annotations

from collections.abc import Sequence

from benchrep.architecture.losses.base import LossTerm
from benchrep.assembly.registries.core import LOSS_REGISTRIES_BY_ROLE
from benchrep.assembly.resolvers.loss_resolver import LossSpec
from benchrep.assembly.schemas.training_config_schema import (
    SupportedLossRole,
)
from benchrep.records import get_run_logger


def build_loss_terms(
    loss_specs: Sequence[LossSpec],
) -> dict[SupportedLossRole, dict[str, LossTerm]]:
    """Instantiate resolved loss specifications."""

    losses_by_role: dict[
        SupportedLossRole,
        dict[str, LossTerm],
    ] = {}

    log_entries_by_role: dict[
        SupportedLossRole,
        list[str],
    ] = {}

    for loss_spec in loss_specs:
        loss_role = loss_spec.loss_role

        registry = LOSS_REGISTRIES_BY_ROLE[loss_role]

        loss_module = registry.create(
            loss_spec.registry_entry_name,
            **loss_spec.constructor_params,
        )

        loss_term = LossTerm(
            loss=loss_module,
            weight=loss_spec.weight,
        )

        losses_by_role.setdefault(loss_role, {})[
            loss_spec.loss_id
        ] = loss_term

        log_entries_by_role.setdefault(loss_role, []).append(
            f"{loss_spec.loss_id}"
            f" -> {type(loss_module).__name__}"
            f" (registry={loss_spec.registry_entry_name}, "
            f"weight={loss_spec.weight})"
        )

    _log_built_losses(log_entries_by_role)

    return losses_by_role


def _log_built_losses(
    log_entries_by_role: dict[
        SupportedLossRole,
        list[str],
    ],
) -> None:
    run_log = get_run_logger()

    descriptions = [
        f"{loss_role}=[{', '.join(entries)}]"
        for loss_role, entries in log_entries_by_role.items()
        if entries
    ]

    run_log.info(
        "Built losses: %s",
        "; ".join(descriptions),
    )