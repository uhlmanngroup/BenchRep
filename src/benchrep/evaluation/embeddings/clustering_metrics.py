from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import anndata as ad
import numpy as np

from benchrep.assembly.registries.core import (
    EVAL_EXTERNAL_CLUSTERING_METRICS,
    EVAL_INTERNAL_CLUSTERING_METRICS,
)
from benchrep.assembly.registries.utils import (
    resolve_registry_keys,
    resolve_registry_param_keys,
)
from benchrep.evaluation.metrics.execution import execute_metric_group
from benchrep.evaluation.utils import (
    RecoverableEvaluationStepError,
    validate_adata_x,
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
    """Compute external clustering metrics against reference labels.

    External clustering metrics compare cluster assignments in
    ``adata.obs[cluster_key]`` with reference labels in
    ``adata.obs[label_key]``.

    Registered entries must be ``EvaluationMetric`` descriptors whose callables
    accept:

        metric.fn(labels_true, labels_pred, **metric_kwargs)

    Return values are validated and normalized according to the metric's declared
    ``result_kind``. Vector and vector-mapping metrics may declare either of these
    axes:

    - ``"true_label"``: unique reference labels in order of first appearance.
    - ``"cluster"``: unique cluster labels in order of first appearance.

    Metric vectors must follow the ordering of their declared axis.

    For HDBSCAN results, observations assigned to the noise cluster are excluded
    before metrics and axis labels are computed. The metric step fails recoverably
    if no non-noise observations remain.

    ``selected`` may contain canonical names or aliases from
    ``EVAL_EXTERNAL_CLUSTERING_METRICS``. Under the current selection behavior,
    ``None`` selects every canonical metric registered in the process.

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

    non_noise_mask = _hdbscan_non_noise_mask(
        adata,
        cluster_key=cluster_key,
    )

    if non_noise_mask is not None:
        labels = labels.iloc[non_noise_mask]
        clusters = clusters.iloc[non_noise_mask]

        if clusters.empty:
            raise RecoverableEvaluationStepError(
                "External clustering metrics require at least one non-noise "
                "HDBSCAN observation."
            )

    metric_names = resolve_registry_keys(
        selected=selected,
        registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
        none_policy="all",
    )
    resolved_metric_kwargs_by_name = resolve_registry_param_keys(
        params=metric_params,
        registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
    )

    if not metric_names:
        raise ValueError(
            "At least one external clustering metric must be selected."
        )

    results, failures = execute_metric_group(
        registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
        metric_names=metric_names,
        metric_positional_args=(
            labels,
            clusters,
        ),
        metric_kwargs_by_name=resolved_metric_kwargs_by_name,
        axis_labels_by_name={
            "true_label": labels.unique(),
            "cluster": clusters.unique(),
        },
    )

    _finalize_recoverable_metric_failures(
        metric_kind="external clustering",
        results=results,
        failures=failures,
    )

    _store_clustering_metric_result(
        adata,
        metric_group="external",
        cluster_key=cluster_key,
        result={
            "metrics": results,
            "params": resolved_metric_kwargs_by_name,
            "label_key": label_key,
            "cluster_key": cluster_key,
            "n_labels": int(labels.nunique()),
            "n_clusters": int(clusters.nunique()),
            "n_obs": int(len(clusters)),
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
    """Compute internal clustering metrics from embeddings and cluster labels.

    Internal clustering metrics evaluate cluster structure in ``adata.X`` using
    assignments from ``adata.obs[cluster_key]``, without requiring reference
    labels.

    Registered entries must be ``EvaluationMetric`` descriptors whose callables
    accept:

        metric.fn(X, cluster_labels, **metric_kwargs)

    Return values are validated and normalized according to the metric's declared
    ``result_kind``. Vector and vector-mapping metrics may declare the
    ``"cluster"`` axis, whose labels are the unique cluster assignments in order
    of first appearance. Metric vectors must follow that same ordering.

    For HDBSCAN results, observations assigned to the noise cluster are excluded
    before metrics and axis labels are computed. Internal metrics require at least
    two remaining clusters and fewer clusters than observations; violations fail
    the metric step recoverably.

    ``selected`` may contain canonical names or aliases from
    ``EVAL_INTERNAL_CLUSTERING_METRICS``. Under the current selection behavior,
    ``None`` selects every canonical metric registered in the process.

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
    metric_input = adata.X

    non_noise_mask = _hdbscan_non_noise_mask(
        adata,
        cluster_key=cluster_key,
    )

    if non_noise_mask is not None:
        clusters = clusters.iloc[non_noise_mask]
        metric_input = adata.X[non_noise_mask]

    n_observations = len(clusters)
    n_clusters = int(clusters.nunique())

    if n_clusters < 2:
        raise RecoverableEvaluationStepError(
            "Internal clustering metrics require at least 2 clusters, got "
            f"{n_clusters}."
        )

    if n_clusters >= n_observations:
        raise RecoverableEvaluationStepError(
            "Internal clustering metrics require fewer clusters than "
            "observations, got "
            f"{n_clusters} clusters for {n_observations} observations."
        )

    metric_names = resolve_registry_keys(
        selected=selected,
        registry=EVAL_INTERNAL_CLUSTERING_METRICS,
        none_policy="all",
    )
    resolved_metric_kwargs_by_name = resolve_registry_param_keys(
        params=metric_params,
        registry=EVAL_INTERNAL_CLUSTERING_METRICS,
    )

    if not metric_names:
        raise ValueError(
            "At least one internal clustering metric must be selected."
        )

    results, failures = execute_metric_group(
        registry=EVAL_INTERNAL_CLUSTERING_METRICS,
        metric_names=metric_names,
        metric_positional_args=(
            metric_input,
            clusters,
        ),
        metric_kwargs_by_name=resolved_metric_kwargs_by_name,
        axis_labels_by_name={
            "cluster": clusters.unique(),
        },
    )

    _finalize_recoverable_metric_failures(
        metric_kind="internal clustering",
        results=results,
        failures=failures,
    )

    _store_clustering_metric_result(
        adata,
        metric_group="internal",
        cluster_key=cluster_key,
        result={
            "metrics": results,
            "params": resolved_metric_kwargs_by_name,
            "cluster_key": cluster_key,
            "n_clusters": n_clusters,
            "n_obs": int(n_observations),
        },
    )

    return adata


def _finalize_recoverable_metric_failures(
    *,
    metric_kind: str,
    results: Mapping[str, Any],
    failures: Mapping[str, str],
) -> None:
    """Warn for partial failures or fail when no metric succeeded."""

    if not failures:
        return

    failure_details = "; ".join(
        f"{metric_name}: {reason}"
        for metric_name, reason in failures.items()
    )

    if not results:
        raise RecoverableEvaluationStepError(
            f"All selected {metric_kind} metrics failed recoverably. "
            f"{failure_details}"
        )

    for metric_name, reason in failures.items():
        warnings.warn(
            f"Skipped {metric_kind} metric {metric_name!r}: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )


def _hdbscan_non_noise_mask(
    adata: ad.AnnData,
    *,
    cluster_key: str,
) -> np.ndarray | None:
    """Return the non-noise mask when the cluster output came from HDBSCAN."""

    metadata = (
        adata.uns
        .get("benchrep", {})
        .get("clustering", {})
        .get(cluster_key, {})
    )

    if (
        not isinstance(metadata, Mapping)
        or metadata.get("method") != "hdbscan"
    ):
        return None

    clusters = adata.obs[cluster_key]
    return np.asarray(clusters.astype(str) != "-1")


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
