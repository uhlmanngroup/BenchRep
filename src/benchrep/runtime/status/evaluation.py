from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


EvaluationOutcomeStatus = Literal[
    "pending",
    "running",
    "disabled",
    "completed",
    "completed_with_warnings",
    "skipped",
    "failed",
]

EvaluationStatus = Literal[
    "disabled",
    "completed",
    "completed_with_warnings",
    "partially_completed",
    "failed",
]


@dataclass(frozen=True)
class EvaluationOutcome:
    """State and issues for an individual evaluation operation.

    Attributes
    ----------
    name : str
        Operation identifier.
    category : str
        Category used to group the operation.
    status : EvaluationOutcomeStatus
        ``pending``, ``running``, ``disabled``, ``completed``, ``completed_with_warnings``, ``skipped``, or ``failed``. Pending and running states must be finalized before aggregation.
    issues : tuple[str, ...]
        Reasons, warnings, or errors associated with the operation.
    """
    name: str
    category: str
    status: EvaluationOutcomeStatus
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationSectionStatus:
    """Aggregate status and individual outcomes for one evaluation section.

    Attributes
    ----------
    status : EvaluationStatus
        ``disabled``, ``completed``, ``completed_with_warnings``, ``partially_completed``, or ``failed``.
    outcomes : tuple[EvaluationOutcome, ...]
        Individual operation outcomes, including disabled operations.
    """
    status: EvaluationStatus
    outcomes: tuple[EvaluationOutcome, ...]


@dataclass(frozen=True)
class EvaluationStatusReport:
    """Overall evaluation status and its section-level outcomes.

    Attributes
    ----------
    anndata : EvaluationSectionStatus
        Outcomes of analyses performed on AnnData.
    reconstructions : EvaluationSectionStatus
        Outcomes of reconstruction analyses.
    exports : EvaluationSectionStatus
        Outcomes of artifact and figure exports.
    status : EvaluationStatus
        Overall status. A fatal issue makes the run ``failed``; mixed successful and unsuccessful active sections produce ``partially_completed``.
    fatal_issue : str | None
        Fatal workflow error, otherwise ``None``.
    """
    anndata: EvaluationSectionStatus
    reconstructions: EvaluationSectionStatus
    exports: EvaluationSectionStatus
    status: EvaluationStatus
    fatal_issue: str | None = None


def summarize_evaluation_outcomes(
    outcomes: Sequence[EvaluationOutcome],
) -> EvaluationSectionStatus:
    outcomes = tuple(outcomes)

    unfinished = [
        outcome.name
        for outcome in outcomes
        if outcome.status in {"pending", "running"}
    ]

    if unfinished:
        raise RuntimeError(
            "Cannot summarize unfinished evaluation outcomes: "
            + ", ".join(unfinished)
        )

    active_outcomes = tuple(
        outcome
        for outcome in outcomes
        if outcome.status != "disabled"
    )

    if not active_outcomes:
        return EvaluationSectionStatus(
            status="disabled",
            outcomes=outcomes,
        )

    has_success = any(
        outcome.status in {"completed", "completed_with_warnings"}
        for outcome in active_outcomes
    )
    has_incomplete = any(
        outcome.status in {"failed", "skipped"}
        for outcome in active_outcomes
    )
    has_warnings = any(
        outcome.status == "completed_with_warnings"
        for outcome in active_outcomes
    )

    if has_incomplete:
        status: EvaluationStatus = (
            "partially_completed"
            if has_success
            else "failed"
        )
    elif has_warnings:
        status = "completed_with_warnings"
    else:
        status = "completed"

    return EvaluationSectionStatus(
        status=status,
        outcomes=outcomes,
    )


def build_evaluation_status_report(
    *,
    anndata_outcomes: Sequence[EvaluationOutcome],
    reconstruction_outcomes: Sequence[EvaluationOutcome],
    export_outcomes: Sequence[EvaluationOutcome],
    fatal_issue: str | None = None,
) -> EvaluationStatusReport:
    anndata = summarize_evaluation_outcomes(anndata_outcomes)
    reconstructions = summarize_evaluation_outcomes(
        reconstruction_outcomes
    )
    exports = summarize_evaluation_outcomes(export_outcomes)

    status = _summarize_workflow_status(
        sections=(anndata, reconstructions, exports),
        fatal_issue=fatal_issue,
    )

    return EvaluationStatusReport(
        anndata=anndata,
        reconstructions=reconstructions,
        exports=exports,
        status=status,
        fatal_issue=fatal_issue,
    )


def _summarize_workflow_status(
    *,
    sections: Sequence[EvaluationSectionStatus],
    fatal_issue: str | None,
) -> EvaluationStatus:
    if fatal_issue is not None:
        return "failed"

    active_sections = tuple(
        section
        for section in sections
        if section.status != "disabled"
    )

    if not active_sections:
        return "completed"

    has_usable_output = any(
        section.status
        in {
            "completed",
            "completed_with_warnings",
            "partially_completed",
        }
        for section in active_sections
    )
    has_incomplete_output = any(
        section.status in {"partially_completed", "failed"}
        for section in active_sections
    )
    has_warnings = any(
        section.status == "completed_with_warnings"
        for section in active_sections
    )

    if has_incomplete_output:
        return (
            "partially_completed"
            if has_usable_output
            else "failed"
        )

    if has_warnings:
        return "completed_with_warnings"

    return "completed"