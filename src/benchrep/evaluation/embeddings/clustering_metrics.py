from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad

from benchrep.assembly.registries.core import (
    EVAL_EXTERNAL_CLUSTERING_METRICS,
    EVAL_INTERNAL_CLUSTERING_METRICS,
)
from benchrep.assembly.registries.utils import (
    resolve_registry_keys,
    resolve_registry_param_keys,
)
from benchrep.evaluation.utils import (
    to_python_scalar,
    validate_adata_x,
    validate_metric_params,
    validate_obs_key,
)


def compute_external_clustering_metrics(
    adata: ad.AnnData,
    *,
    label_key: str = "label",
    cluster_key: str,
    selected: Sequence[str] | None = None,
    metric_params: Mapping[str, Mapping[str, Any]] | None = None,
    overwrite: bool = False,
) -> ad.AnnData:
    """Compute external clustering metrics against labels in ``adata.obs``.

    External clustering metrics compare cluster assignments to a reference
    annotation, such as class labels, cell types, or transferred labels.

    Registered external metric callables must follow the BenchRep external
    clustering metric contract:

        metric_fn(labels_true, labels_pred, **params) -> scalar

    where ``labels_true`` comes from ``adata.obs[label_key]`` and
    ``labels_pred`` comes from ``adata.obs[cluster_key]``. Custom metrics with a
    different native signature should be wrapped before registration.

    ``selected`` should contain canonical metric names or aliases registered in
    ``EVAL_EXTERNAL_CLUSTERING_METRICS``. If ``selected`` is ``None``, all
    canonical registered external clustering metrics are computed.

    Results are stored under:

        adata.uns["benchrep"]["metrics"]["clustering"]["external"][cluster_key]
    """
    validate_obs_key(adata, label_key)
    validate_obs_key(adata, cluster_key)
    _check_metric_result_available(
        adata,
        metric_group="external",
        cluster_key=cluster_key,
        overwrite=overwrite,
    )

    labels = adata.obs[label_key]
    clusters = adata.obs[cluster_key]

    metric_names = resolve_registry_keys(
        selected=selected,
        registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
        none_policy="all",
    )
    metric_params = resolve_registry_param_keys(
        params=metric_params,
        registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
    )

    results: dict[str, Any] = {}

    for metric_name in metric_names:
        metric_fn = EVAL_EXTERNAL_CLUSTERING_METRICS.get(metric_name)
        params = metric_params.get(metric_name, {})

        validate_metric_params(
            metric_name=metric_name,
            metric_fn=metric_fn,
            params=params,
            metric_kind="external clustering metric",
        )

        try:
            value = metric_fn(labels, clusters, **params)
        except Exception as error:
            raise RuntimeError(
                f"Failed to compute external clustering metric {metric_name!r}. "
                "The metric callable was found, but execution failed."
            ) from error

        results[metric_name] = to_python_scalar(value)

    _store_clustering_metric_result(
        adata,
        metric_group="external",
        cluster_key=cluster_key,
        result={
            "metrics": results,
            "params": metric_params,
            "label_key": label_key,
            "cluster_key": cluster_key,
            "n_labels": int(labels.nunique()),
            "n_clusters": int(clusters.nunique()),
            "n_obs": int(adata.n_obs),
        },
    )

    return adata


def compute_internal_clustering_metrics(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    selected: Sequence[str] | None = None,
    metric_params: Mapping[str, Mapping[str, Any]] | None = None,
    overwrite: bool = False,
) -> ad.AnnData:
    """Compute internal clustering metrics from ``adata.X`` and cluster labels.

    Internal clustering metrics evaluate cluster structure in the representation
    space without requiring ground-truth labels.

    Registered internal metric callables must follow the BenchRep internal
    clustering metric contract:

        metric_fn(X, labels, **params) -> scalar

    where ``X`` is ``adata.X`` and ``labels`` comes from
    ``adata.obs[cluster_key]``. Custom metrics with a different native signature
    should be wrapped before registration.

    ``selected`` should contain canonical metric names or aliases registered in
    ``EVAL_INTERNAL_CLUSTERING_METRICS``. If ``selected`` is ``None``, all
    canonical registered internal clustering metrics are computed.

    Results are stored under:

        adata.uns["benchrep"]["metrics"]["clustering"]["internal"][cluster_key]
    """
    validate_adata_x(adata)
    validate_obs_key(adata, cluster_key)
    _check_metric_result_available(
        adata,
        metric_group="internal",
        cluster_key=cluster_key,
        overwrite=overwrite,
    )

    clusters = adata.obs[cluster_key]
    n_clusters = int(clusters.nunique())

    if n_clusters < 2:
        raise ValueError(
            "Internal clustering metrics require at least 2 clusters, got "
            f"{n_clusters}."
        )

    if n_clusters >= adata.n_obs:
        raise ValueError(
            "Internal clustering metrics require fewer clusters than observations, "
            f"got {n_clusters} clusters for {adata.n_obs} observations."
        )

    metric_names = resolve_registry_keys(
        selected=selected,
        registry=EVAL_INTERNAL_CLUSTERING_METRICS,
        none_policy="all",
    )
    metric_params = resolve_registry_param_keys(
        params=metric_params,
        registry=EVAL_INTERNAL_CLUSTERING_METRICS,
    )

    results: dict[str, Any] = {}

    for metric_name in metric_names:
        metric_fn = EVAL_INTERNAL_CLUSTERING_METRICS.get(metric_name)
        params = metric_params.get(metric_name, {})

        validate_metric_params(
            metric_name=metric_name,
            metric_fn=metric_fn,
            params=params,
            metric_kind="internal clustering metric",
        )

        try:
            value = metric_fn(adata.X, clusters, **params)
        except Exception as error:
            raise RuntimeError(
                f"Failed to compute internal clustering metric {metric_name!r}. "
                "The metric callable was found, but execution failed."
            ) from error

        results[metric_name] = to_python_scalar(value)

    _store_clustering_metric_result(
        adata,
        metric_group="internal",
        cluster_key=cluster_key,
        result={
            "metrics": results,
            "params": metric_params,
            "cluster_key": cluster_key,
            "n_clusters": n_clusters,
            "n_obs": int(adata.n_obs),
        },
    )

    return adata


def _check_metric_result_available(
    adata: ad.AnnData,
    *,
    metric_group: str,
    cluster_key: str,
    overwrite: bool,
) -> None:
    """Check whether a clustering metric result can be written."""

    group_results = (
        adata.uns
        .get("benchrep", {})
        .get("metrics", {})
        .get("clustering", {})
        .get(metric_group, {})
    )

    if cluster_key in group_results and not overwrite:
        raise KeyError(
            f"BenchRep {metric_group} clustering metrics already contain results "
            f"for {cluster_key!r}. Pass overwrite=True to replace them."
        )


def _store_clustering_metric_result(
    adata: ad.AnnData,
    *,
    metric_group: str,
    cluster_key: str,
    result: Mapping[str, Any],
) -> None:
    """Store clustering metric results under the BenchRep namespace.

    Results are written to:

        adata.uns["benchrep"]["metrics"]["clustering"][metric_group][cluster_key]

    where ``metric_group`` should usually be ``"internal"`` or ``"external"``,
    and ``cluster_key`` is the ``adata.obs`` column containing the cluster labels,
    such as ``"leiden"`` or ``"kmeans"``.

    The resulting structure is:

        adata.uns["benchrep"] = {
            "metrics": {
                "clustering": {
                    metric_group: {
                        cluster_key: result,
                    },
                },
            },
        }

    For example, internal Leiden metrics are stored at:

        adata.uns["benchrep"]["metrics"]["clustering"]["internal"]["leiden"]

    and external KMeans metrics are stored at:

        adata.uns["benchrep"]["metrics"]["clustering"]["external"]["kmeans"].
    """

    benchrep_uns = adata.uns.setdefault("benchrep", {})
    metrics_uns = benchrep_uns.setdefault("metrics", {})
    clustering_uns = metrics_uns.setdefault("clustering", {})
    group_uns = clustering_uns.setdefault(metric_group, {})

    group_uns[cluster_key] = dict(result)
