from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PredictionOutcomeStatus = Literal[
    "disabled",
    "completed",
    "completed_with_warnings",
    "skipped",
    "failed",
]

PredictionStatus = Literal[
    "completed",
    "completed_with_warnings",
    "partially_completed",
    "failed",
]


@dataclass(frozen=True)
class PredictionOutcome:
    name: str
    status: PredictionOutcomeStatus
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionStatusReport:
    inference: PredictionOutcome
    embeddings_export: PredictionOutcome
    reconstructions_export: PredictionOutcome
    status: PredictionStatus


def build_prediction_status_report(
    *,
    inference: PredictionOutcome,
    embeddings_export: PredictionOutcome,
    reconstructions_export: PredictionOutcome,
) -> PredictionStatusReport:
    exports = (
        embeddings_export,
        reconstructions_export,
    )

    if inference.status in {"disabled", "skipped"}:
        raise ValueError(
            "Prediction inference cannot be disabled or skipped in a "
            "final status report."
        )

    if inference.status == "failed":
        status: PredictionStatus = "failed"
    else:
        active_exports = tuple(
            outcome
            for outcome in exports
            if outcome.status != "disabled"
        )

        has_successful_export = any(
            outcome.status in {
                "completed",
                "completed_with_warnings",
            }
            for outcome in active_exports
        )
        has_incomplete_export = any(
            outcome.status in {"failed", "skipped"}
            for outcome in active_exports
        )
        has_warnings = (
            inference.status == "completed_with_warnings"
            or any(
                outcome.status == "completed_with_warnings"
                for outcome in active_exports
            )
        )

        if has_incomplete_export:
            status = (
                "partially_completed"
                if has_successful_export
                else "failed"
            )
        elif has_warnings:
            status = "completed_with_warnings"
        else:
            status = "completed"

    return PredictionStatusReport(
        inference=inference,
        embeddings_export=embeddings_export,
        reconstructions_export=reconstructions_export,
        status=status,
    )