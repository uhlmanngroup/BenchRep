from __future__ import annotations

import math
from numbers import Real
from typing import Literal

from torch import nn

from benchrep.architecture.losses.weight_schedules import LossWeightSchedule


class LossTerm(nn.Module):
    """Container for a loss, its base weight, and an optional schedule."""

    def __init__(
        self,
        *,
        loss: nn.Module,
        weight: float,
        weight_schedule: LossWeightSchedule | None = None,
        schedule_interval: Literal["epoch", "step"] = "epoch",
    ) -> None:
        super().__init__()

        if schedule_interval not in {"epoch", "step"}:
            raise ValueError("`schedule_interval` must be 'epoch' or 'step'.")

        self.loss = loss
        self.weight = weight
        self.weight_schedule = weight_schedule
        self.schedule_interval = schedule_interval

    def effective_weight(self, *, epoch: int, step: int) -> float:
        """Evaluate the weight without changing the configured base weight."""
        if self.weight_schedule is None:
            return self.weight

        position = epoch if self.schedule_interval == "epoch" else step
        factor = self.weight_schedule.factor(position)

        if isinstance(factor, bool) or not isinstance(factor, Real):
            raise TypeError(
                "Loss-weight schedules must return a real scalar, "
                f"got {type(factor).__name__}."
            )

        factor = float(factor)
        if not math.isfinite(factor) or factor < 0:
            raise ValueError(
                f"Schedule factor must be finite and nonnegative, got {factor}."
            )

        effective_weight = self.weight * factor
        if not math.isfinite(effective_weight):
            raise ValueError(
                "Scheduled loss weight overflowed: "
                f"base weight={self.weight}, factor={factor}."
            )

        return effective_weight
