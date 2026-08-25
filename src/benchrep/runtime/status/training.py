from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from lightning.pytorch.callbacks import EarlyStopping


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


@dataclass(frozen=True)
class EarlyStoppingRecord:
    triggered: bool
    reason: str
    monitor: str
    stopped_epoch: int
    best_score: float | None
    wait_count: int


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


def build_early_stopping_record(
    callback: EarlyStopping | None,
) -> EarlyStoppingRecord | None:
    if callback is None:
        return None

    reason = callback.stopping_reason.name.lower()
    best_score_tensor = torch.as_tensor(
        callback.best_score
    ).detach().cpu()

    best_score = (
        float(best_score_tensor)
        if torch.isfinite(best_score_tensor)
        else None
    )

    return EarlyStoppingRecord(
        triggered=reason != "not_stopped",
        reason=reason,
        monitor=callback.monitor,
        stopped_epoch=int(callback.stopped_epoch),
        best_score=best_score,
        wait_count=int(callback.wait_count),
    )
