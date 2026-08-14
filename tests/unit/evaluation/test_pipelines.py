import anndata as ad
import numpy as np
import pytest

from benchrep.evaluation.pipelines import AnnDataEvaluationStep, AnnDataEvaluationPipeline
from benchrep.evaluation.utils import RecoverableEvaluationStepError


def _make_adata() -> ad.AnnData:
    return ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32)
    )


def test_successful_anndata_step_commits_mutations() -> None:
    original = _make_adata()

    def mutate(value: ad.AnnData) -> ad.AnnData:
        value.obs["committed"] = True
        return value

    step = AnnDataEvaluationStep(
        name="successful_step",
        category="test",
        fn=mutate,
    )

    result = step.run(original)

    assert step.status == "completed"
    assert result is not original
    assert "committed" in result.obs
    assert "committed" not in original.obs


def test_recoverably_failed_anndata_step_discards_mutations() -> None:
    original = _make_adata()

    def mutate_then_fail(value: ad.AnnData) -> ad.AnnData:
        value.obs["partial"] = True
        raise RecoverableEvaluationStepError("expected failure")

    step = AnnDataEvaluationStep(
        name="failed_step",
        category="test",
        fn=mutate_then_fail,
    )

    result = step.run(original)

    assert step.status == "failed"
    assert result is original
    assert "partial" not in result.obs


def test_fatally_failed_anndata_step_discards_mutations() -> None:
    original = _make_adata()

    def mutate_then_fail(value: ad.AnnData) -> ad.AnnData:
        value.obs["partial"] = True
        raise RuntimeError("expected failure")

    step = AnnDataEvaluationStep(
        name="failed_step",
        category="test",
        fn=mutate_then_fail,
    )

    with pytest.raises(RuntimeError, match="expected failure"):
        step.run(original)

    assert step.status == "failed"
    assert "partial" not in original.obs


def test_pipeline_discards_failed_step_mutations_before_next_step() -> None:
    original = _make_adata()

    def commit_first(value: ad.AnnData) -> ad.AnnData:
        value.obs["committed"] = True
        return value

    def mutate_then_fail(value: ad.AnnData) -> ad.AnnData:
        value.obs["partial"] = True
        raise RecoverableEvaluationStepError("expected failure")

    def inspect_and_continue(value: ad.AnnData) -> ad.AnnData:
        assert "committed" in value.obs
        assert "partial" not in value.obs
        value.obs["continued"] = True
        return value

    pipeline = AnnDataEvaluationPipeline(
        steps=[
            AnnDataEvaluationStep(
                name="first",
                category="test",
                fn=commit_first,
            ),
            AnnDataEvaluationStep(
                name="failed",
                category="test",
                fn=mutate_then_fail,
            ),
            AnnDataEvaluationStep(
                name="last",
                category="test",
                fn=inspect_and_continue,
            ),
        ]
    )

    result = pipeline.run(original)

    assert pipeline.steps[0].status == "completed"
    assert pipeline.steps[1].status == "failed"
    assert pipeline.steps[2].status == "completed"
    assert "committed" in result.obs
    assert "continued" in result.obs
    assert "partial" not in result.obs