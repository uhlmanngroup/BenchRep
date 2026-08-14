from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


TrainingStatus = Literal[
    "completed",
    "completed_with_warnings",
    "completed_after_interruption",
    "failed",
]

TrainingInterruptionSignal = Literal["sigint", "sigterm"]

ACCEPTABLE_TRAINING_STATUSES = frozenset[TrainingStatus](
    {
        "completed",
        "completed_with_warnings",
        "completed_after_interruption",
    }
)


@dataclass(frozen=True)
class TrainingStatusReport:
    status: TrainingStatus
    issues: tuple[str, ...]
    interruption_signal: TrainingInterruptionSignal | None


def build_training_status_report(
    *,
    errors: Sequence[str],
    warnings: Sequence[str],
    interruption_signal: TrainingInterruptionSignal | None,
) -> TrainingStatusReport:
    if errors:
        status: TrainingStatus = "failed"
    elif interruption_signal is not None:
        status = "completed_after_interruption"
    elif warnings:
        status = "completed_with_warnings"
    else:
        status = "completed"

    issues = (
        *(f"Error: {error}" for error in errors),
        *(f"Warning: {warning}" for warning in warnings),
    )

    return TrainingStatusReport(
        status=status,
        issues=issues,
        interruption_signal=interruption_signal,
    )
