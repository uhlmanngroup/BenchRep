from __future__ import annotations

from collections.abc import Sequence
from typing import Literal


TrainingStatus = Literal[
    "completed",
    "completed_with_warnings",
    "completed_after_interruption",
    "failed",
]

TrainingInterruptionSignal = Literal["sigint", "sigterm"]

SUCCESSFUL_TRAINING_STATUSES = frozenset[TrainingStatus](
    {
        "completed",
        "completed_with_warnings",
        "completed_after_interruption",
    }
)


def summarize_training_status(
    *,
    errors: Sequence[str],
    warnings: Sequence[str],
    interruption_signal: TrainingInterruptionSignal | None,
) -> TrainingStatus:
    if errors:
        return "failed"

    if interruption_signal is not None:
        return "completed_after_interruption"

    if warnings:
        return "completed_with_warnings"

    return "completed"