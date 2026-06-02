from __future__ import annotations

from typing import Any

import anndata as ad
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    silhouette_score,
)

from benchrep.evaluation.utils import validate_adata_x


def compute_external_clustering_metrics(
    adata: ad.AnnData,
    *,
    label_key: str = "label",
    cluster_key: str,
    key_added: str = "external_clustering_metrics",
    ami_average_method: str = "arithmetic",
    overwrite: bool = False,
) -> ad.AnnData:
    """
    Compute external clustering metrics against known labels in ``adata.obs``.

    External metrics compare cluster assignments to an existing annotation,
    such as class labels, cell types, or transferred labels. Results are stored
    in ``adata.uns[key_added][cluster_key]``.

    Parameters
    ----------
    adata:
        AnnData object containing labels and cluster assignments in ``adata.obs``.
    label_key:
        Column in ``adata.obs`` containing known labels.
    cluster_key:
        Column in ``adata.obs`` containing cluster assignments.
    key_added:
        Key under which metrics are stored in ``adata.uns``.
    ami_average_method:
        Averaging method passed to ``adjusted_mutual_info_score``.
    overwrite:
        If ``False``, raise an error when metrics for ``cluster_key`` already
        exist under ``adata.uns[key_added]``. If ``True``, replace them.

    Returns
    -------
    AnnData
        The input AnnData object, modified in place and returned for convenience.
    """
    _validate_obs_key(adata, label_key)
    _validate_obs_key(adata, cluster_key)
    _check_metric_key_available(
        adata,
        key_added=key_added,
        cluster_key=cluster_key,
        overwrite=overwrite,
    )

    labels = adata.obs[label_key]
    clusters = adata.obs[cluster_key]

    metrics = {
        "adjusted_rand_index": adjusted_rand_score(labels, clusters),
        "adjusted_mutual_info": adjusted_mutual_info_score(
            labels,
            clusters,
            average_method=ami_average_method,
        ),
        "homogeneity": homogeneity_score(labels, clusters),
        "label_key": label_key,
        "cluster_key": cluster_key,
        "average_method": ami_average_method,
        "n_labels": int(labels.nunique()),
        "n_clusters": int(clusters.nunique()),
    }

    if key_added not in adata.uns:
        adata.uns[key_added] = {}

    adata.uns[key_added][cluster_key] = metrics

    return adata


def compute_internal_clustering_metrics(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    key_added: str = "internal_clustering_metrics",
    metric: str = "euclidean",
    overwrite: bool = False,
    **silhouette_kwargs: Any,
) -> ad.AnnData:
    """
    Compute internal clustering metrics from ``adata.X`` and cluster assignments.

    Internal metrics evaluate cluster structure using the feature space itself,
    without requiring known labels. For now, this computes silhouette score.
    Results are stored in ``adata.uns[key_added][cluster_key]``.

    Parameters
    ----------
    adata:
        AnnData object whose ``X`` matrix contains the clustered representation.
    cluster_key:
        Column in ``adata.obs`` containing cluster assignments.
    key_added:
        Key under which metrics are stored in ``adata.uns``.
    metric:
        Distance metric passed to ``sklearn.metrics.silhouette_score``.
    overwrite:
        If ``False``, raise an error when metrics for ``cluster_key`` already
        exist under ``adata.uns[key_added]``. If ``True``, replace them.
    **silhouette_kwargs:
        Additional keyword arguments passed to ``silhouette_score``.

    Returns
    -------
    AnnData
        The input AnnData object, modified in place and returned for convenience.
    """
    validate_adata_x(adata)
    _validate_obs_key(adata, cluster_key)
    _check_metric_key_available(
        adata,
        key_added=key_added,
        cluster_key=cluster_key,
        overwrite=overwrite,
    )

    clusters = adata.obs[cluster_key]

    if clusters.nunique() < 2:
        raise ValueError(
            "Silhouette score requires at least 2 clusters, got "
            f"{clusters.nunique()}."
        )

    if clusters.nunique() >= adata.n_obs:
        raise ValueError(
            "Silhouette score requires fewer clusters than observations, got "
            f"{clusters.nunique()} clusters for {adata.n_obs} observations."
        )

    metrics = {
        "silhouette": silhouette_score(
            adata.X,
            clusters,
            metric=metric,
            **silhouette_kwargs,
        ),
        "cluster_key": cluster_key,
        "metric": metric,
        "n_clusters": int(clusters.nunique()),
    }

    if key_added not in adata.uns:
        adata.uns[key_added] = {}

    adata.uns[key_added][cluster_key] = metrics

    return adata


def _validate_obs_key(adata: ad.AnnData, key: str) -> None:
    """Validate that ``key`` exists in ``adata.obs``."""

    if key not in adata.obs.columns:
        raise KeyError(
            f"adata.obs does not contain {key!r}. "
            f"Available columns: {list(adata.obs.columns)}"
        )


def _check_metric_key_available(
    adata: ad.AnnData,
    *,
    key_added: str,
    cluster_key: str,
    overwrite: bool,
) -> None:
    """Check whether a metric entry can be written."""

    if (
        key_added in adata.uns
        and cluster_key in adata.uns[key_added]
        and not overwrite
    ):
        raise KeyError(
            f"adata.uns[{key_added!r}] already contains metrics for "
            f"{cluster_key!r}. Pass overwrite=True to replace them."
        )