"""Regression coverage for low-precision built-in clustering metrics."""

import warnings

import anndata as ad
import numpy as np
import pytest
from sklearn.metrics import calinski_harabasz_score as sklearn_score

from benchrep.assembly.registries.core import EVAL_INTERNAL_CLUSTERING_METRICS
from benchrep.evaluation.embeddings.clustering_metrics import (
    compute_internal_clustering_metrics,
)


@pytest.mark.parametrize(
    "dtype,selected",
    [(np.float16, "calinski_harabasz"), (np.float16, "ch"),
     (np.float32, "calinski_harabasz"), (np.float64, "calinski_harabasz")],
)
def test_calinski_harabasz_uses_stable_precision_without_changing_embeddings(
    dtype, selected,
):
    # Modest finite values, but enough observations/features for float16
    # dispersion products to overflow in scikit-learn 1.6 with NumPy 2.
    labels = np.arange(1024) % 10
    X = (np.random.default_rng(137).normal(size=(1024, 64))
         + labels[:, None] / 3).astype(dtype)
    original = X.copy()
    expected = sklearn_score(X.astype(np.float64), labels)
    adata = ad.AnnData(X=X)
    adata.obs["clusters"] = labels

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = compute_internal_clustering_metrics(
            adata, cluster_key="clusters", selected=[selected],
        )

    metric = result.uns["benchrep"]["metrics"]["clustering"]["internal"][
        "clusters"
    ]["metrics"]["calinski_harabasz"]
    assert metric["kind"] == "scalar"
    assert np.isfinite(metric["value"])
    assert metric["value"] == pytest.approx(expected, rel=1e-12)
    assert result is adata
    assert adata.X is X
    assert adata.X.dtype == dtype
    np.testing.assert_array_equal(adata.X, original)


@pytest.mark.parametrize(
    "X,labels,message",
    [
        (np.ones((4, 2)), [0, 0, 0, 0], "Number of labels"),
        (np.array([[np.nan], [0.0], [1.0], [2.0]]), [0, 0, 1, 1], "NaN"),
        (np.ones((4, 2), dtype=complex), [0, 0, 1, 1], "Complex data"),
    ],
)
def test_calinski_harabasz_retains_input_validation(X, labels, message):
    metric = EVAL_INTERNAL_CLUSTERING_METRICS.get("calinski_harabasz")
    with pytest.raises(ValueError, match=message):
        metric.fn(X, labels)
