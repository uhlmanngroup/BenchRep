from __future__ import annotations

from typing import Any

import anndata as ad
import scanpy as sc
from sklearn.cluster import KMeans

from benchrep.evaluation.utils import validate_adata_x


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

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        **kmeans_kwargs,
    )

    labels = kmeans.fit_predict(adata.X)

    adata.obs[key_added] = labels.astype(str)
    adata.obs[key_added] = adata.obs[key_added].astype("category")

    adata.uns[key_added] = {
        "method": "kmeans",
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_init": n_init,
        "input_shape": tuple(adata.X.shape),
        "inertia": float(kmeans.inertia_),
    }

    return adata


def run_leiden(
    adata: ad.AnnData,
    *,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    metric: str = "euclidean",
    key_added: str = "leiden",
    neighbors_key: str = "neighbors",
    random_state: int = 137,
    overwrite: bool = False,
    neighbors_kwargs: dict[str, Any] | None = None,
    leiden_kwargs: dict[str, Any] | None = None,
) -> ad.AnnData:
    """
    Build a Scanpy neighbor graph and run Leiden clustering on ``adata.X``.

    Parameters
    ----------
    adata:
        AnnData object whose ``X`` matrix contains the representation to cluster.
    resolution:
        Leiden resolution parameter. Higher values generally produce more
        clusters.
    n_neighbors:
        Number of neighbors used to construct the neighbor graph.
    n_pcs:
        Number of PCs passed to Scanpy neighbors. If ``None``, Scanpy decides
        based on the input.
    metric:
        Distance metric passed to Scanpy neighbors. Common useful options
        include ``"euclidean"``, ``"cosine"``, ``"correlation"``,
        ``"manhattan"``, ``"l1"``, and ``"l2"``.
    key_added:
        Key under which Leiden cluster labels are stored in ``adata.obs``.
    neighbors_key:
        Key used by Scanpy to store and retrieve the neighbor graph.
    random_state:
        Random seed passed to Scanpy Leiden.
    overwrite:
        If ``False``, raise an error when ``key_added`` or ``neighbors_key``
        already exists. If ``True``, replace existing entries.
    neighbors_kwargs:
        Additional keyword arguments passed to ``scanpy.pp.neighbors``.
    leiden_kwargs:
        Additional keyword arguments passed to ``scanpy.tl.leiden``.

    Returns
    -------
    AnnData
        The input AnnData object, modified in place and returned for convenience.
    """
    validate_adata_x(adata)
    _check_obs_key_available(adata, key_added=key_added, overwrite=overwrite)

    if neighbors_key in adata.uns and not overwrite:
        raise KeyError(
            f"adata.uns already contains {neighbors_key!r}. "
            "Pass overwrite=True to replace it."
        )

    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}.")

    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")

    neighbors_kwargs = {} if neighbors_kwargs is None else neighbors_kwargs
    leiden_kwargs = {} if leiden_kwargs is None else leiden_kwargs

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        metric=metric,
        key_added=neighbors_key,
        **neighbors_kwargs,
    )

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        key_added=key_added,
        neighbors_key=neighbors_key,
        **leiden_kwargs,
    )

    adata.uns[key_added] = {
        "method": "leiden",
        "resolution": resolution,
        "n_neighbors": n_neighbors,
        "n_pcs": n_pcs,
        "metric": metric,
        "neighbors_key": neighbors_key,
        "random_state": random_state,
        "input_shape": tuple(adata.X.shape),
        "n_clusters": int(adata.obs[key_added].nunique()),
    }

    return adata


def _check_obs_key_available(
    adata: ad.AnnData,
    *,
    key_added: str,
    overwrite: bool,
) -> None:
    """Check whether an ``adata.obs`` key can be written."""

    if key_added in adata.obs.columns and not overwrite:
        raise KeyError(
            f"adata.obs already contains {key_added!r}. "
            "Pass overwrite=True to replace it."
        )