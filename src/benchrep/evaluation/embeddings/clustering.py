from __future__ import annotations

from typing import Any, Literal
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN

from benchrep.evaluation.utils import (
    RecoverableEvaluationStepError,
    validate_adata_x,
    load_scanpy_backend,
)

MAX_CLUSTERS_WARN = 50


def run_kmeans(
    adata: ad.AnnData,
    *,
    n_clusters: int,
    key_added: str = "kmeans",
    random_state: int = 137,
    n_init: str | int = "auto",
    overwrite: bool = False,
    **kmeans_kwargs: Any,
) -> ad.AnnData:
    """
    Run KMeans clustering on ``adata.X`` and store cluster labels in ``adata.obs``.

    Parameters
    ----------
    adata:
        AnnData object whose ``X`` matrix contains the representation to cluster.
    n_clusters:
        Number of KMeans clusters.
    key_added:
        Key under which cluster labels are stored in ``adata.obs``.
    random_state:
        Random seed passed to scikit-learn KMeans.
    n_init:
        Number of KMeans initializations. Passed directly to scikit-learn.
    overwrite:
        If ``False``, raise an error when ``key_added`` already exists. If
        ``True``, replace existing entries.
    **kmeans_kwargs:
        Additional keyword arguments passed to ``sklearn.cluster.KMeans``.

    Returns
    -------
    AnnData
        The input AnnData object, modified in place and returned for convenience.
    """
    validate_adata_x(adata)
    _check_obs_key_available(adata, key_added=key_added, overwrite=overwrite)

    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}.")

    if n_clusters > adata.n_obs:
        raise RecoverableEvaluationStepError(
            "n_clusters cannot exceed adata.n_obs, got "
            f"n_clusters={n_clusters} and n_obs={adata.n_obs}."
        )
    requested_n_clusters = n_clusters

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        **kmeans_kwargs,
    )

    labels = _validate_clustering_labels(
        kmeans.fit_predict(adata.X),
        method_name="KMeans",
        n_observations=adata.n_obs,
        require_integer=True,
    )

    if np.any(labels < 0):
        raise RuntimeError(
            "KMeans returned negative cluster labels."
        )

    actual_n_clusters = int(np.unique(labels).size)

    adata.obs[key_added] = labels.astype(str)
    adata.obs[key_added] = adata.obs[key_added].astype("category")

    _store_clustering_metadata(
        adata,
        key_added=key_added,
        metadata={
            "method": "kmeans",
            "cluster_key": key_added,
            "n_clusters": actual_n_clusters,
            "requested_n_clusters": requested_n_clusters,
            "random_state": random_state,
            "n_init": n_init,
            "input_shape": list(adata.X.shape),
            "inertia": float(kmeans.inertia_),
            "params": dict(kmeans_kwargs),
        },
    )

    if actual_n_clusters == 1:
        warnings.warn(
            "KMeans produced only one cluster. Internal clustering metrics "
            "requiring at least two clusters will be unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )

    _warn_if_many_clusters(
        cluster_key=key_added,
        n_clusters=actual_n_clusters,
    )

    return adata


def run_leiden(
    adata: ad.AnnData,
    *,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    use_rep: str | None = None,
    metric: str = "euclidean",
    neighbors_kwargs: dict[str, Any] | None = None,
    resolution: float = 1.0,
    key_added: str = "leiden",
    leiden_kwargs: dict[str, Any] | None = None,
    neighbors_key: str = "neighbors",
    random_state: int = 137,
    overwrite: bool = False,
) -> ad.AnnData:
    """Construct a Scanpy neighbor graph and run Leiden clustering.

    The representation used for neighbor construction is controlled by
    ``use_rep`` and ``n_pcs``. When ``use_rep`` is ``None``, Scanpy selects
    between ``adata.X`` and ``adata.obsm["X_pca"]`` automatically.

    Cluster labels are stored in ``adata.obs[key_added]``. Neighbor metadata is
    stored in ``adata.uns[neighbors_key]``, with distance and connectivity
    matrices in ``adata.obsp``. BenchRep provenance is stored under
    ``adata.uns["benchrep"]["clustering"][key_added]``.

    Parameters
    ----------
    adata:
        AnnData object containing the representation to cluster.
    n_neighbors:
        Number of kNN neighbors passed to ``scanpy.pp.neighbors()``.
    n_pcs:
        Number of principal components passed to ``scanpy.pp.neighbors()``.
        ``None`` delegates selection to Scanpy; zero forces ``adata.X`` when
        ``use_rep`` is unset.
    use_rep:
        Representation passed to ``scanpy.pp.neighbors()``. ``"X"`` selects
        ``adata.X``; other values name an entry in ``adata.obsm``. ``None``
        delegates selection to Scanpy.
    metric:
        Distance metric passed to ``scanpy.pp.neighbors()``.
    neighbors_kwargs:
        Additional keyword arguments passed to ``scanpy.pp.neighbors()``.
    resolution:
        Resolution passed to ``scanpy.tl.leiden()``. Higher values generally
        produce more clusters.
    key_added:
        Key used by ``scanpy.tl.leiden()`` to store labels in ``adata.obs``.
    leiden_kwargs:
        Additional keyword arguments passed to ``scanpy.tl.leiden()``.
    neighbors_key:
        Namespace used to store the neighbor graph and retrieve it for Leiden.
    random_state:
        Random seed passed to ``scanpy.pp.neighbors()`` and
        ``scanpy.tl.leiden()``.
    overwrite:
        Whether existing Leiden or neighbor-graph outputs may be replaced.

    Returns
    -------
    AnnData
        The input AnnData object, modified in place and returned.
    """
    sc = load_scanpy_backend(
        feature="Leiden clustering",
        require_leiden=True,
    )

    validate_adata_x(adata)
    _check_obs_key_available(adata, key_added=key_added, overwrite=overwrite)

    if neighbors_key in adata.uns and not overwrite:
        raise RecoverableEvaluationStepError(
            f"adata.uns already contains {neighbors_key!r}. "
            "Pass overwrite=True to replace it."
        )
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}.")

    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")

    if n_neighbors >= adata.n_obs:
        raise RecoverableEvaluationStepError(
            "n_neighbors must be smaller than adata.n_obs, got "
            f"n_neighbors={n_neighbors} and n_obs={adata.n_obs}."
        )

    neighbors_kwargs = (
        {} if neighbors_kwargs is None else dict(neighbors_kwargs)
    )
    leiden_kwargs = (
        {} if leiden_kwargs is None else dict(leiden_kwargs)
    )

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        metric=metric,
        key_added=None if neighbors_key == "neighbors" else neighbors_key,
        random_state=random_state,
        **neighbors_kwargs,
    )

    # Adopt Scanpy's announced future Leiden defaults explicitly to avoid the
    # FutureWarning while preserving user overrides through leiden_kwargs.
    leiden_kwargs.setdefault("flavor", "igraph")
    leiden_kwargs.setdefault("n_iterations", 2)
    leiden_kwargs.setdefault("directed", False)

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        key_added=key_added,
        neighbors_key=neighbors_key,
        **leiden_kwargs,
    )

    labels = _validate_clustering_labels(
        adata.obs[key_added].to_numpy(),
        method_name="Leiden",
        n_observations=adata.n_obs,
        require_integer=False,
    )

    n_clusters = int(pd.unique(labels).size)

    _store_clustering_metadata(
        adata,
        key_added=key_added,
        metadata={
            "method": "leiden",
            "n_neighbors": n_neighbors,
            "n_pcs": n_pcs,
            "use_rep": use_rep,
            "metric": metric,
            "neighbors_params": dict(neighbors_kwargs),
            "resolution": resolution,
            "cluster_key": key_added,
            "leiden_params": dict(leiden_kwargs),
            "neighbors_key": neighbors_key,
            "random_state": random_state,
            "n_clusters": n_clusters,
            "adata_x_shape": list(adata.X.shape),
        },
    )

    if n_clusters == 1:
        warnings.warn(
            "Leiden produced only one cluster. Internal clustering metrics "
            "requiring at least two clusters will be unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )

    _warn_if_many_clusters(
        cluster_key=key_added,
        n_clusters=n_clusters,
    )

    return adata


def run_hdbscan(
    adata: ad.AnnData,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
    cluster_selection_method: Literal["eom", "leaf"] = "eom",
    allow_single_cluster: bool = False,
    key_added: str = "hdbscan",
    overwrite: bool = False,
    **hdbscan_kwargs: Any,
) -> ad.AnnData:
    """Run HDBSCAN clustering on ``adata.X``.

    Cluster labels are stored in ``adata.obs[key_added]``. The label ``-1``
    identifies density noise. If scikit-learn returns labels below ``-1``,
    indicating non-finite input values, this function raises an error.

    Per-sample cluster-membership strengths are stored in
    ``adata.obs[f"{key_added}_probability"]``.

    Parameters
    ----------
    adata:
        AnnData object whose ``X`` matrix contains the representation to cluster.
    min_cluster_size:
        Minimum number of samples required for a grouping to be considered a
        cluster.
    min_samples:
        Number of neighboring samples required for a point to be considered a
        core point. If ``None``, scikit-learn uses ``min_cluster_size``.
    cluster_selection_epsilon:
        Distance threshold below which clusters are merged.
    metric:
        Distance metric passed to scikit-learn HDBSCAN.
    cluster_selection_method:
        Method used to select clusters from the condensed tree.
    allow_single_cluster:
        Whether HDBSCAN may return a single non-noise cluster.
    key_added:
        Key under which cluster labels are stored in ``adata.obs``.
    overwrite:
        If ``False``, raise when either output key already exists. If ``True``,
        replace existing outputs.
    **hdbscan_kwargs:
        Additional keyword arguments passed to ``sklearn.cluster.HDBSCAN``.

    Returns
    -------
    AnnData
        The input AnnData object, modified in place and returned for convenience.
    """
    validate_adata_x(adata)

    probability_key = f"{key_added}_probability"

    _check_obs_key_available(
        adata,
        key_added=key_added,
        overwrite=overwrite,
    )
    _check_obs_key_available(
        adata,
        key_added=probability_key,
        overwrite=overwrite,
    )

    if min_cluster_size < 2:
        raise ValueError(
            "min_cluster_size must be >= 2, "
            f"got {min_cluster_size}."
        )

    if min_samples is not None and min_samples < 1:
        raise ValueError(
            f"min_samples must be >= 1 or None, got {min_samples}."
        )

    if cluster_selection_epsilon < 0:
        raise ValueError(
            "cluster_selection_epsilon must be >= 0, "
            f"got {cluster_selection_epsilon}."
        )

    if adata.n_obs < 2:
        raise RecoverableEvaluationStepError(
            "HDBSCAN requires at least 2 observations, got "
            f"n_obs={adata.n_obs}."
        )

    resolved_min_samples = (
        min_cluster_size
        if min_samples is None
        else min_samples
    )

    if resolved_min_samples > adata.n_obs:
        raise RecoverableEvaluationStepError(
            "HDBSCAN min_samples cannot exceed adata.n_obs, got "
            f"min_samples={resolved_min_samples} and n_obs={adata.n_obs}."
        )

    # Preserve scikit-learn's original HDBSCAN behavior and suppress its
    # copy-default transition warning while still allowing an explicit override.
    hdbscan_kwargs.setdefault("copy", False)

    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        **hdbscan_kwargs,
    )

    labels = _validate_clustering_labels(
        hdbscan.fit_predict(adata.X),
        method_name="HDBSCAN",
        n_observations=adata.n_obs,
        require_integer=True,
    )
    probabilities = np.asarray(hdbscan.probabilities_, dtype=float)

    invalid_input_mask = labels < -1

    if np.any(invalid_input_mask):
        invalid_labels, invalid_counts = np.unique(
            labels[invalid_input_mask],
            return_counts=True,
        )
        invalid_label_counts = {
            int(label): int(count)
            for label, count in zip(
                invalid_labels,
                invalid_counts,
                strict=True,
            )
        }

        raise ValueError(
            "HDBSCAN returned labels indicating non-finite input values: "
            f"{invalid_label_counts}. Evaluation embeddings must contain only "
            "finite values."
        )

    clustered_mask = labels >= 0
    n_samples = int(labels.size)
    n_clusters = int(np.unique(labels[clustered_mask]).size)
    n_noise = int(np.count_nonzero(labels == -1))

    adata.obs[key_added] = pd.Categorical(labels.astype(str))
    adata.obs[probability_key] = probabilities

    params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_epsilon": cluster_selection_epsilon,
        "metric": metric,
        "cluster_selection_method": cluster_selection_method,
        "allow_single_cluster": allow_single_cluster,
        **hdbscan_kwargs,
    }

    metadata: dict[str, Any] = {
        "method": "hdbscan",
        "cluster_key": key_added,
        "probability_key": probability_key,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": n_noise / n_samples,
        "input_shape": list(adata.X.shape),
        "params": params,
    }

    if hasattr(hdbscan, "centroids_"):
        metadata["centroids"] = hdbscan.centroids_

    if hasattr(hdbscan, "medoids_"):
        metadata["medoids"] = hdbscan.medoids_

    _store_clustering_metadata(
        adata,
        key_added=key_added,
        metadata=metadata,
    )

    if n_clusters == 0:
        warnings.warn(
            "HDBSCAN produced no non-noise clusters: "
            f"all {n_samples} observations were classified as noise. "
            "Dependent clustering metrics may be unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )

    elif n_clusters == 1:
        warnings.warn(
            "HDBSCAN produced only one non-noise cluster after excluding "
            f"{n_noise} noise observations. Internal clustering metrics "
            "requiring at least two clusters will be unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )

    _warn_if_many_clusters(
        cluster_key=key_added,
        n_clusters=n_clusters,
    )

    return adata


def _warn_if_many_clusters(
    *,
    cluster_key: str,
    n_clusters: int,
) -> None:
    """Warn when a clustering result may be difficult to interpret."""

    if n_clusters <= MAX_CLUSTERS_WARN:
        return

    warnings.warn(
        f"Clustering key {cluster_key!r} produced {n_clusters} clusters, "
        f"exceeding the warning threshold of {MAX_CLUSTERS_WARN}. "
        "Cluster-colored reduction plots and cluster-size plots may be "
        "difficult to interpret.",
        RuntimeWarning,
        stacklevel=2,
    )


def _validate_clustering_labels(
    labels: Any,
    *,
    method_name: str,
    n_observations: int,
    require_integer: bool,
) -> np.ndarray:
    """Return labels after validating the clustering output contract."""

    label_array = np.asarray(labels)

    if label_array.ndim != 1:
        raise RuntimeError(
            f"{method_name} must return a one-dimensional label array, got "
            f"shape {label_array.shape}."
        )

    if label_array.shape[0] != n_observations:
        raise RuntimeError(
            f"{method_name} returned {label_array.shape[0]} labels for "
            f"{n_observations} observations."
        )

    if pd.isna(label_array).any():
        raise RuntimeError(
            f"{method_name} returned missing cluster labels."
        )

    if (
        require_integer
        and not np.issubdtype(label_array.dtype, np.integer)
    ):
        raise RuntimeError(
            f"{method_name} must return integer cluster labels, got dtype "
            f"{label_array.dtype}."
        )

    return label_array


def _check_obs_key_available(
    adata: ad.AnnData,
    *,
    key_added: str,
    overwrite: bool,
) -> None:
    """Check whether an ``adata.obs`` key can be written."""

    if key_added in adata.obs.columns and not overwrite:
        raise RecoverableEvaluationStepError(
            f"adata.obs already contains {key_added!r}. "
            "Pass overwrite=True to replace it."
        )


def _store_clustering_metadata(
    adata: ad.AnnData,
    *,
    key_added: str,
    metadata: dict[str, Any],
) -> None:
    """Store clustering metadata under the BenchRep namespace.

    Metadata are written to:

        adata.uns["benchrep"]["clustering"][key_added]

    where ``key_added`` is the ``adata.obs`` column containing the cluster labels,
    such as ``"leiden"`` or ``"kmeans"``.
    """

    benchrep_uns = adata.uns.setdefault("benchrep", {})
    clustering_uns = benchrep_uns.setdefault("clustering", {})

    clustering_uns[key_added] = metadata