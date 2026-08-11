import warnings

import anndata as ad
import numpy as np
import pytest

from benchrep.evaluation.pipelines import AnnDataEvaluationStep
from benchrep.runtime.status.evaluation import (
    EvaluationOutcome,
    EvaluationOutcomeStatus,
    EvaluationSummaryStatus,
    build_evaluation_status_report,
    summarize_evaluation_outcomes,
)


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((), "disabled"),
        (("disabled",), "disabled"),
        (("completed",), "completed"),
        (
            ("completed", "completed_with_warnings"),
            "completed_with_warnings",
        ),
        (("completed", "failed"), "partially_completed"),
        (("failed", "skipped"), "failed"),
    ],
)
def test_summarize_evaluation_outcomes(
    statuses: tuple[EvaluationOutcomeStatus, ...],
    expected: EvaluationSummaryStatus,
) -> None:
    outcomes = tuple(
        EvaluationOutcome(
            name=f"outcome_{index}",
            category="test",
            status=status,
        )
        for index, status in enumerate(statuses)
    )

    assert summarize_evaluation_outcomes(outcomes).status == expected


def test_summarize_evaluation_outcomes_rejects_unfinished() -> None:
    outcomes = (
        EvaluationOutcome(
            name="unfinished",
            category="test",
            status="running",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Cannot summarize unfinished evaluation outcomes",
    ):
        summarize_evaluation_outcomes(outcomes)


def test_build_evaluation_status_report_aggregates_sections() -> None:
    report = build_evaluation_status_report(
        embedding_outcomes=(
            EvaluationOutcome(
                name="pca",
                category="reductions",
                status="completed",
            ),
            EvaluationOutcome(
                name="umap",
                category="reductions",
                status="failed",
            ),
        ),
        reconstruction_outcomes=(),
        export_outcomes=(
            EvaluationOutcome(
                name="evaluated_embeddings",
                category="exports",
                status="completed",
            ),
        ),
    )

    assert report.embeddings.status == "partially_completed"
    assert report.reconstructions.status == "disabled"
    assert report.exports.status == "completed"
    assert report.status == "partially_completed"


def test_evaluation_step_always_captures_relevant_warnings() -> None:
    adata = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32)
    )

    def warn_and_return(value: ad.AnnData) -> ad.AnnData:
        warnings.warn("record me", RuntimeWarning)
        return value

    step = AnnDataEvaluationStep(
        name="warning_step",
        category="test",
        fn=warn_and_return,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        step.run(adata)

    assert step.status == "completed_with_warnings"
    assert step.issues == ["Warning (RuntimeWarning): record me"]
    assert step.to_outcome().category == "test"
