"""Instantiate resolved loss specifications."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

from benchrep.architecture.losses.base import LossTerm
from benchrep.architecture.losses.weight_schedules import (
    CyclicSchedule,
    LossWeightSchedule,
)
from benchrep.assembly.schemas.training_config_schema import (
    SupportedLossRole,
)
from benchrep.records import get_run_logger
from benchrep.assembly.registries.core import (
    LOSS_REGISTRIES_BY_ROLE,
    LOSS_WEIGHT_SCHEDULES,
)
from benchrep.assembly.resolvers.loss_resolver import (
    LossSpec,
    LossWeightScheduleSpec,
)


def _build_loss_weight_schedule(
    spec: LossWeightScheduleSpec | None,
) -> LossWeightSchedule | None:
    if spec is None:
        return None

    schedule = LOSS_WEIGHT_SCHEDULES.create(
        spec.registry_entry_name,
        **spec.constructor_params,
    )

    if not callable(getattr(schedule, "factor", None)):
        raise TypeError(
            f"Schedule {spec.registry_entry_name!r} must provide "
            "a callable factor(position) method."
        )

    try:
        inspect.signature(schedule.factor).bind(0)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Schedule {spec.registry_entry_name!r} must support "
            "factor(position) with one positional argument."
        ) from exc

    if spec.cycle_length is not None:
        schedule = CyclicSchedule(
            schedule=schedule,
            length=spec.cycle_length,
        )

    return schedule


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

        schedule_spec = loss_spec.weight_schedule
        loss_term = LossTerm(
            loss=loss_module,
            weight=loss_spec.weight,
            weight_schedule=_build_loss_weight_schedule(schedule_spec),
            schedule_interval=(
                schedule_spec.interval
                if schedule_spec is not None
                else "epoch"
            ),
        )

        losses_by_role.setdefault(loss_role, {})[
            loss_spec.loss_id
        ] = loss_term

        schedule_description = "constant"
        if schedule_spec is not None:
            schedule_description = (
                f"{schedule_spec.registry_entry_name}"
                f", interval={schedule_spec.interval}"
            )
            if schedule_spec.cycle_length is not None:
                schedule_description += (
                    f", cycle_length={schedule_spec.cycle_length}"
                )

        log_entries_by_role.setdefault(loss_role, []).append(
            f"{loss_spec.loss_id}"
            f" -> {type(loss_module).__name__}"
            f" (registry={loss_spec.registry_entry_name}, "
            f"base_weight={loss_spec.weight}, "
            f"schedule={schedule_description})"
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
        f"    {loss_role}: {', '.join(entries)}"
        for loss_role, entries in log_entries_by_role.items()
        if entries
    ]

    run_log.info(
        "Built losses:\n%s",
        "\n".join(descriptions),
    )