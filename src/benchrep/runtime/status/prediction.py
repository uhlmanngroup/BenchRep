from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PredictionOutcomeStatus = Literal[
    "disabled",
    "completed",
    "completed_with_warnings",
    "partially_completed",
    "skipped",
    "failed",
]

PredictionStatus = Literal[
    "completed",
    "completed_with_warnings",
    "partially_completed",
    "failed",
]

ACCEPTABLE_PREDICTION_STATUSES = frozenset[PredictionStatus](
    {
        "completed",
        "completed_with_warnings",
        "partially_completed",
    }
)


@dataclass(frozen=True)
class PredictionOutcome:
    """Outcome of inference or a prediction export branch.

    Attributes
    ----------
    name : str
        Operation identifier.
    status : PredictionOutcomeStatus
        ``disabled``, ``completed``, ``completed_with_warnings``, ``partially_completed``, ``skipped``, or ``failed``.
    issues : tuple[str, ...]
        Reasons, warnings, or errors associated with this operation.
    """
    name: str
    status: PredictionOutcomeStatus
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionStatusReport:
    """Prediction status with separate inference and export outcomes.

    Attributes
    ----------
    inference : PredictionOutcome
        Outcome of model inference.
    anndata_export : PredictionOutcome
        Outcome of AnnData export.
    reconstructions_export : PredictionOutcome
        Aggregate outcome of reconstruction exports.
    status : PredictionStatus
        ``completed``, ``completed_with_warnings``, ``partially_completed``, or ``failed``. Disabled exports are excluded from aggregation.
    """
    inference: PredictionOutcome
    anndata_export: PredictionOutcome
    reconstructions_export: PredictionOutcome
    status: PredictionStatus


def build_prediction_status_report(
    *,
    inference: PredictionOutcome,
    anndata_export: PredictionOutcome,
    reconstructions_export: PredictionOutcome,
) -> PredictionStatusReport:
    exports = (
        anndata_export,
        reconstructions_export,
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
                "partially_completed",
            }
            for outcome in active_exports
        )

        has_incomplete_export = any(
            outcome.status in {
                "failed",
                "skipped",
                "partially_completed",
            }
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
        anndata_export=anndata_export,
        reconstructions_export=reconstructions_export,
        status=status,
    )