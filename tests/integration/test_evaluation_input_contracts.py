"""Validate real saved inputs through the evaluation workflow."""

import json

import anndata as ad
import numpy as np
import pytest
from scipy import sparse
import torch

from benchrep import evaluate
from benchrep.assembly.schemas import EvaluationConfig


def _embedding_config(tmp_path, path):
    return EvaluationConfig.model_validate(
        {
            "source": {"anndata_path": path},
            "run": {"output_root": tmp_path / "outputs"},
            "reductions": {
                name: {"enabled": False} for name in ("pca", "umap", "tsne")
            },
            "clustering": {
                name: {"enabled": False} for name in ("kmeans", "leiden", "hdbscan")
            },
            "metrics": {
                "embedding": {"enabled": True, "selected": ["mean"]},
                "predictability": {"enabled": False},
            },
            "plots": {"enabled": False},
        }
    )


@pytest.mark.parametrize(
    "matrix_type", [sparse.csr_matrix, sparse.csc_matrix], ids=["csr", "csc"]
)
def test_sparse_embeddings_are_evaluated_with_implicit_zeros(tmp_path, matrix_type):
    dense = np.array([[0, 2], [4, 0], [0, 0]], dtype=np.float32)
    path = tmp_path / "sparse.h5ad"
    ad.AnnData(X=matrix_type(dense)).write_h5ad(path)

    result = evaluate(config=_embedding_config(tmp_path, path))

    assert result.status_report.status == "completed"
    metrics = json.loads(result.export_paths.metrics_json_path.read_text())
    np.testing.assert_allclose(
        metrics["embedding"]["metrics"]["mean"]["values"],
        dense.mean(axis=0),
    )
    # Evaluation must not mutate the source artifact.
    original = ad.read_h5ad(path)
    assert "benchrep" not in original.uns
    np.testing.assert_array_equal(original.X.toarray(), dense)


@pytest.mark.parametrize(
    "matrix_type", [sparse.csr_matrix, sparse.csc_matrix], ids=["csr", "csc"]
)
def test_sparse_nonfinite_error_counts_values_and_distinct_observations(
    tmp_path, matrix_type
):
    # Two bad values share a row: three bad values affect only two observations.
    matrix = matrix_type(
        (
            [np.nan, np.inf, -np.inf],
            ([1, 1, 3], [0, 2, 1]),
        ),
        shape=(5, 3),
    )
    path = tmp_path / "nonfinite.h5ad"
    ad.AnnData(X=matrix).write_h5ad(path)

    with pytest.raises(
        ValueError,
        match=r"1 NaN values and 2 infinite values across 2 observations.*\[1, 3\]",
    ):
        evaluate(config=_embedding_config(tmp_path, path))
    assert not list((tmp_path / "outputs").rglob("metrics.json"))


@pytest.mark.parametrize(
    ("matrix", "error_type", "message"),
    [
        pytest.param(
            np.empty((0, 3)),
            ValueError,
            "at least one observation",
            id="no-observations",
        ),
        pytest.param(
            np.empty((3, 0)), ValueError, "at least one variable", id="no-variables"
        ),
        pytest.param(
            np.ones((3, 2), dtype=complex),
            TypeError,
            "real numeric values",
            id="complex-embeddings",
        ),
    ],
)
def test_invalid_embedding_contract_fails_before_evaluation(
    tmp_path, matrix, error_type, message
):
    path = tmp_path / "invalid.h5ad"
    ad.AnnData(X=matrix).write_h5ad(path)
    with pytest.raises(error_type, match=message):
        evaluate(config=_embedding_config(tmp_path, path))
    assert not list((tmp_path / "outputs").rglob("metrics.json"))


@pytest.mark.parametrize(
    ("obs", "error_type", "message"),
    [
        pytest.param(
            {"sample_id": ["a", "b"], "label": [0]},
            ValueError,
            "columns with inconsistent lengths",
            id="unequal-columns",
        ),
        pytest.param(
            {"sample_id": ["a"], "label": [0]},
            ValueError,
            "obs length=1, reconstruction examples=2",
            id="wrong-row-count",
        ),
        pytest.param(
            {"description": "not a column", "version": 1},
            TypeError,
            "Could not infer",
            id="no-observation-columns",
        ),
    ],
)
def test_reconstruction_metadata_must_align_with_examples(
    tmp_path, obs, error_type, message
):
    bundle = tmp_path / "reconstructions"
    bundle.mkdir()
    torch.save(torch.zeros(2, 1, 2, 2), bundle / "input.pt")
    torch.save(torch.ones(2, 1, 2, 2), bundle / "reconstruction.pt")
    torch.save(obs, bundle / "obs.pt")
    config = EvaluationConfig.model_validate(
        {
            "source": {"reconstructions_path": bundle},
            "run": {"output_root": tmp_path / "outputs"},
            "plots": {"enabled": False},
        }
    )

    with pytest.raises(error_type, match=message):
        evaluate(config=config)
    assert not list((tmp_path / "outputs").rglob("metrics.json"))
