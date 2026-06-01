from __future__ import annotations

import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.decomposition import PCA


def run_pca(
    adata: ad.AnnData,
    *,
    n_components: int = 2,
    key_added: str = "X_pca",
    random_state: int = 137,
    overwrite: bool = False,
) -> ad.AnnData:
    """
    Run PCA on ``adata.X`` and store the coordinates in ``adata.obsm``.

    This function treats ``adata.X`` as the feature space being evaluated,
    such as a learned embedding, baseline feature matrix, or other
    representation. PCA coordinates are written to ``adata.obsm[key_added]``.
    Basic PCA provenance and explained-variance summaries are written to
    ``adata.uns[key_added]``.

    Parameters
    ----------
    adata:
        AnnData object whose ``X`` matrix contains the representation to reduce.
    n_components:
        Number of principal components to compute.
    key_added:
        Key under which PCA coordinates are stored in ``adata.obsm``.
    random_state:
        Random seed passed to scikit-learn PCA.
    overwrite:
        If ``False``, raise an error when ``key_added`` already exists.
        If ``True``, replace existing entries.

    Returns
    -------
    AnnData
       The input AnnData object, modified in place and returned for convenience.
    """
    _validate_adata_x(adata)

    if key_added in adata.obsm and not overwrite:
        raise KeyError(
            f"adata.obsm already contains {key_added!r}. "
            "Pass overwrite=True to replace it."
        )

    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}.")

    max_components = min(adata.n_obs, adata.n_vars)
    if n_components > max_components:
        raise ValueError(
            "n_components cannot exceed min(adata.n_obs, adata.n_vars), "
            f"got n_components={n_components} and max={max_components}."
        )

    pca = PCA(
        n_components=n_components,
        random_state=random_state,
    )

    adata.obsm[key_added] = pca.fit_transform(adata.X)

    adata.uns[key_added] = {
        "method": "pca",
        "n_components": n_components,
        "random_state": random_state,
        "input_shape": tuple(adata.X.shape),
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance_ratio": np.cumsum(
            pca.explained_variance_ratio_
        ).tolist(),
    }

    return adata


def run_umap(
    adata: ad.AnnData,
    *,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    metric: str = "euclidean",
    key_added: str = "X_umap",
    neighbors_key: str = "neighbors",
    random_state: int = 137,
    overwrite: bool = False,
) -> ad.AnnData:
    """
    Run Scanpy neighbors followed by UMAP on ``adata.X``.

    This function builds a neighbor graph from ``adata.X`` using
    ``scanpy.pp.neighbors`` and then computes UMAP coordinates with
    ``scanpy.tl.umap``. The UMAP coordinates are stored in
    ``adata.obsm[key_added]``. Neighbor-graph information is stored under
    ``neighbors_key`` using Scanpy's standard AnnData fields, and UMAP
    parameters are recorded in ``adata.uns[key_added]``.

    Parameters
    ----------
    adata:
        AnnData object whose ``X`` matrix contains the representation to reduce.
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
        Key under which UMAP coordinates are stored in ``adata.obsm``.
    neighbors_key:
        Key used by Scanpy to store and retrieve the neighbor graph.
    random_state:
        Random seed passed to Scanpy UMAP.
    overwrite:
        If ``False``, raise an error when ``key_added`` or ``neighbors_key``
        already exists. If ``True``, replace existing entries.

    Returns
    -------
    AnnData
        The input AnnData object, modified in place and returned for convenience.
    """
    _validate_adata_x(adata)

    if key_added in adata.obsm and not overwrite:
        raise KeyError(
            f"adata.obsm already contains {key_added!r}. "
            "Pass overwrite=True to replace it."
        )

    if neighbors_key in adata.uns and not overwrite:
        raise KeyError(
            f"adata.uns already contains {neighbors_key!r}. "
            "Pass overwrite=True to replace it."
        )

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        metric=metric,
        key_added=neighbors_key,
    )

    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        random_state=random_state,
        key_added=key_added,
    )

    adata.uns[key_added] = {
        "method": "umap",
        "n_neighbors": n_neighbors,
        "n_pcs": n_pcs,
        "metric": metric,
        "neighbors_key": neighbors_key,
        "random_state": random_state,
    }

    return adata


def run_tsne(
    adata: ad.AnnData,
    *,
    n_pcs: int | None = None,
    perplexity: float = 30.0,
    key_added: str = "X_tsne",
    random_state: int = 137,
    overwrite: bool = False,
) -> ad.AnnData:
    """
    Run t-SNE on ``adata.X`` and store the coordinates in ``adata.obsm``.

    This function uses Scanpy's t-SNE implementation and treats ``adata.X`` as
    the representation being evaluated. Coordinates are written to
    ``adata.obsm[key_added]`` and basic run parameters are recorded in
    ``adata.uns[key_added]``.

    Parameters
    ----------
    adata:
        AnnData object whose ``X`` matrix contains the representation to reduce.
    n_pcs:
        Number of PCs used by Scanpy before t-SNE. If ``None``, Scanpy decides
        based on the input.
    perplexity:
        t-SNE perplexity. Must be greater than 0 and smaller than
        ``adata.n_obs``.
    key_added:
        Key under which t-SNE coordinates are stored in ``adata.obsm``.
    random_state:
        Random seed passed to Scanpy t-SNE.
    overwrite:
        If ``False``, raise an error when ``key_added`` already exists. If
        ``True``, replace existing entries.

    Returns
    -------
    AnnData
        The input AnnData object, modified in place and returned for convenience.
    """
    _validate_adata_x(adata)

    if key_added in adata.obsm and not overwrite:
        raise KeyError(
            f"adata.obsm already contains {key_added!r}. "
            "Pass overwrite=True to replace it."
        )

    if perplexity <= 0:
        raise ValueError(f"perplexity must be > 0, got {perplexity}.")

    if perplexity >= adata.n_obs:
        raise ValueError(
            "perplexity must be smaller than adata.n_obs, got "
            f"perplexity={perplexity} and n_obs={adata.n_obs}."
        )

    sc.tl.tsne(
        adata,
        n_pcs=n_pcs,
        perplexity=perplexity,
        random_state=random_state,
        key_added=key_added,
    )

    adata.uns[key_added] = {
        "method": "tsne",
        "n_pcs": n_pcs,
        "perplexity": perplexity,
        "random_state": random_state,
    }

    return adata


def _validate_adata_x(adata: ad.AnnData) -> None:
    """Validate that adata.X is present and 2D."""

    if adata.X is None:
        raise ValueError("adata.X is required for dimensionality reduction.")

    if adata.X.ndim != 2:
        raise ValueError(f"adata.X must be 2D, got shape {adata.X.shape}.")