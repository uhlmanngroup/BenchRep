from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA

from benchrep.evaluation.utils import (
    RecoverableEvaluationStepError,
    validate_adata_x,
    load_scanpy_backend,
)


DEFAULT_PCA_N_COMPONENTS = 30


def run_pca(
    adata: ad.AnnData,
    *,
    n_components: int | None = None,
    key_added: str = "X_pca",
    random_state: int = 137,
    overwrite: bool = False,
    **pca_kwargs: Any,
) -> ad.AnnData:
    """
    Run PCA on ``adata.X`` and store the coordinates in ``adata.obsm``.

    This function treats ``adata.X`` as the feature space being evaluated,
    such as a learned embedding, baseline feature matrix, or other
    representation. PCA coordinates are written to ``adata.obsm[key_added]``.
    Basic PCA provenance and explained-variance summaries are written to
    ``adata.uns["benchrep"]["reductions"][key_added]``.

    Parameters
    ----------
    adata:
        AnnData object whose ``X`` matrix contains the representation to reduce.
    n_components:
        Number of principal components to compute. If ``None``, compute
        ``min(30, adata.n_obs, adata.n_vars)`` components.
    key_added:
        Key under which PCA coordinates are stored in ``adata.obsm``.
    random_state:
        Random seed passed to scikit-learn PCA.
    overwrite:
        If ``False``, raise an error when ``key_added`` already exists.
        If ``True``, replace existing entries.
    **pca_kwargs:
        Additional keyword arguments passed to ``sklearn.decomposition.PCA``.

    Returns
    -------
    AnnData
       The input AnnData object, modified in place and returned for convenience.
    """

    validate_adata_x(adata)

    if key_added in adata.obsm and not overwrite:
        raise RecoverableEvaluationStepError(
            f"adata.obsm already contains {key_added!r}. "
            "Pass overwrite=True to replace it."
        )

    max_components = min(adata.n_obs, adata.n_vars)

    if n_components is None:
        resolved_n_components = min(DEFAULT_PCA_N_COMPONENTS, max_components)
    else:
        resolved_n_components = n_components

    if resolved_n_components < 1:
        raise ValueError(
            f"n_components must be >= 1, got {resolved_n_components}."
        )

    if resolved_n_components > max_components:
        raise RecoverableEvaluationStepError(
            "n_components cannot exceed min(adata.n_obs, adata.n_vars), "
            f"got n_components={resolved_n_components} and max={max_components}."
        )

    pca = PCA(
        n_components=resolved_n_components,
        random_state=random_state,
        **pca_kwargs,
    )

    coordinates = _validate_reduction_coordinates(
        pca.fit_transform(adata.X),
        method_name="PCA",
        n_observations=adata.n_obs,
        expected_n_dimensions=resolved_n_components,
    )

    adata.obsm[key_added] = coordinates

    _store_reduction_metadata(
        adata,
        key_added=key_added,
        metadata={
            "method": "pca",
            "n_components": resolved_n_components,
            "requested_n_components": n_components,
            "default_n_components": DEFAULT_PCA_N_COMPONENTS,
            "random_state": random_state,
            "input_shape": list(adata.X.shape),
            "explained_variance": pca.explained_variance_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "params": dict(pca_kwargs),
        },
    )

    return adata


def run_umap(
    adata: ad.AnnData,
    *,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    use_rep: str | None = None,
    metric: str = "euclidean",
    neighbors_kwargs: dict[str, Any] | None = None,
    min_dist: float = 0.1,
    key_added: str = "X_umap",
    umap_kwargs: dict[str, Any] | None = None,
    neighbors_key: str = "neighbors",
    random_state: int = 137,
    overwrite: bool = False,
) -> ad.AnnData:
    """Construct a Scanpy neighbor graph and compute UMAP coordinates.

    The representation used for neighbor construction is controlled by
    ``use_rep`` and ``n_pcs``. When ``use_rep`` is ``None``, Scanpy selects
    between ``adata.X`` and ``adata.obsm["X_pca"]`` automatically.

    UMAP coordinates are stored in ``adata.obsm[key_added]``. Neighbor metadata
    is stored in ``adata.uns[neighbors_key]``, with distance and connectivity
    matrices in ``adata.obsp``. BenchRep provenance is stored under
    ``adata.uns["benchrep"]["reductions"][key_added]``.

    Parameters
    ----------
    adata:
        AnnData object containing the representation to reduce.
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
    min_dist:
        Minimum distance passed to ``scanpy.tl.umap()``.
    key_added:
        Key used by ``scanpy.tl.umap()`` to store coordinates in ``adata.obsm``.
    umap_kwargs:
        Additional keyword arguments passed to ``scanpy.tl.umap()``.
    neighbors_key:
        Namespace used to store the neighbor graph and retrieve it for UMAP.
    random_state:
        Random seed passed to ``scanpy.pp.neighbors()`` and
        ``scanpy.tl.umap()``.
    overwrite:
        Whether existing UMAP or neighbor-graph outputs may be replaced.

    Returns
    -------
    AnnData
        The input AnnData object, modified in place and returned.
    """

    sc = load_scanpy_backend(feature="UMAP")

    validate_adata_x(adata)

    if key_added in adata.obsm and not overwrite:
        raise RecoverableEvaluationStepError(
            f"adata.obsm already contains {key_added!r}. "
            "Pass overwrite=True to replace it."
        )

    if neighbors_key in adata.uns and not overwrite:
        raise RecoverableEvaluationStepError(
            f"adata.uns already contains {neighbors_key!r}. "
            "Pass overwrite=True to replace it."
        )

    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")

    if n_neighbors >= adata.n_obs:
        raise RecoverableEvaluationStepError(
            "n_neighbors must be smaller than adata.n_obs, got "
            f"n_neighbors={n_neighbors} and n_obs={adata.n_obs}."
        )

    neighbors_kwargs = {} if neighbors_kwargs is None else dict(neighbors_kwargs)
    umap_kwargs = {} if umap_kwargs is None else dict(umap_kwargs)

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

    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        random_state=random_state,
        key_added=key_added,
        min_dist=min_dist,
        **umap_kwargs,
    )

    if key_added not in adata.obsm:
        raise RuntimeError(
            f"UMAP did not create adata.obsm[{key_added!r}]."
        )

    adata.obsm[key_added] = _validate_reduction_coordinates(
        adata.obsm[key_added],
        method_name="UMAP",
        n_observations=adata.n_obs,
        expected_n_dimensions=umap_kwargs.get("n_components", 2),
    )

    _store_reduction_metadata(
        adata,
        key_added=key_added,
        metadata={
            "method": "umap",
            "n_neighbors": n_neighbors,
            "n_pcs": n_pcs,
            "use_rep": use_rep,
            "metric": metric,
            "neighbors_params": neighbors_kwargs,
            "min_dist": min_dist,
            "umap_params": umap_kwargs,
            "neighbors_key": neighbors_key,
            "random_state": random_state,
            "adata_x_shape": list(adata.X.shape),
        },
    )

    return adata


def run_tsne(
    adata: ad.AnnData,
    *,
    n_pcs: int | None = None,
    use_rep: str | None = None,
    perplexity: float = 30.0,
    key_added: str = "X_tsne",
    random_state: int = 137,
    overwrite: bool = False,
    **tsne_kwargs: Any,
) -> ad.AnnData:
    """Compute t-SNE coordinates with Scanpy.

    The input representation is controlled by ``use_rep`` and ``n_pcs``. When
    ``use_rep`` is ``None``, Scanpy selects between ``adata.X`` and
    ``adata.obsm["X_pca"]`` automatically.

    Coordinates are stored in ``adata.obsm[key_added]`` and BenchRep provenance
    under ``adata.uns["benchrep"]["reductions"][key_added]``.

    Parameters
    ----------
    adata:
        AnnData object containing the representation to reduce.
    n_pcs:
        Number of principal components passed to ``scanpy.tl.tsne()``.
        ``None`` delegates selection to Scanpy; zero forces ``adata.X`` when
        ``use_rep`` is unset.
    use_rep:
        Representation passed to ``scanpy.tl.tsne()``. ``"X"`` selects
        ``adata.X``; other values name an entry in ``adata.obsm``. ``None``
        delegates selection to Scanpy.
    perplexity:
        Perplexity passed to ``scanpy.tl.tsne()``. It must be positive and
        smaller than ``adata.n_obs``.
    key_added:
        Key used by ``scanpy.tl.tsne()`` to store coordinates in ``adata.obsm``.
    random_state:
        Random seed passed to ``scanpy.tl.tsne()``.
    overwrite:
        Whether an existing ``adata.obsm[key_added]`` entry may be replaced.
    **tsne_kwargs:
        Additional keyword arguments passed to ``scanpy.tl.tsne()``.

    Returns
    -------
    AnnData
        The input AnnData object, modified in place and returned.
    """

    sc = load_scanpy_backend(feature="t-SNE")

    validate_adata_x(adata)

    if key_added in adata.obsm and not overwrite:
        raise RecoverableEvaluationStepError(
            f"adata.obsm already contains {key_added!r}. "
            "Pass overwrite=True to replace it."
        )

    if perplexity <= 0:
        raise ValueError(f"perplexity must be > 0, got {perplexity}.")

    if perplexity >= adata.n_obs:
        raise RecoverableEvaluationStepError(
            "perplexity must be smaller than adata.n_obs, got "
            f"perplexity={perplexity} and n_obs={adata.n_obs}."
        )

    sc.tl.tsne(
        adata,
        n_pcs=n_pcs,
        use_rep=use_rep,
        perplexity=perplexity,
        random_state=random_state,
        key_added=key_added,
        **tsne_kwargs,
    )

    if key_added not in adata.obsm:
        raise RuntimeError(
            f"t-SNE did not create adata.obsm[{key_added!r}]."
        )

    adata.obsm[key_added] = _validate_reduction_coordinates(
        adata.obsm[key_added],
        method_name="t-SNE",
        n_observations=adata.n_obs,
        expected_n_dimensions=tsne_kwargs.get("n_components", 2),
    )

    _store_reduction_metadata(
        adata,
        key_added=key_added,
        metadata={
            "method": "tsne",
            "n_pcs": n_pcs,
            "use_rep": use_rep,
            "perplexity": perplexity,
            "random_state": random_state,
            "adata_x_shape": list(adata.X.shape),
            "params": dict(tsne_kwargs),
        },
    )

    return adata


def _validate_reduction_coordinates(
    value: Any,
    *,
    method_name: str,
    n_observations: int,
    expected_n_dimensions: int,
) -> np.ndarray:
    """Validate and return one reduction coordinate matrix."""

    try:
        coordinates = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{method_name} coordinates could not be converted to a NumPy "
            "array."
        ) from error

    if coordinates.ndim != 2:
        raise ValueError(
            f"{method_name} coordinates must be two-dimensional, got shape "
            f"{coordinates.shape}."
        )

    expected_shape = (
        n_observations,
        expected_n_dimensions,
    )

    if coordinates.shape != expected_shape:
        raise ValueError(
            f"{method_name} coordinates must have shape {expected_shape}, got "
            f"{coordinates.shape}."
        )

    if (
        not np.issubdtype(coordinates.dtype, np.number)
        or np.issubdtype(coordinates.dtype, np.complexfloating)
    ):
        raise TypeError(
            f"{method_name} coordinates must contain real numeric values, "
            f"got dtype {coordinates.dtype}."
        )

    if not np.isfinite(coordinates).all():
        raise RecoverableEvaluationStepError(
            f"{method_name} produced non-finite coordinates."
        )

    return coordinates


def _store_reduction_metadata(
    adata: ad.AnnData,
    *,
    key_added: str,
    metadata: dict[str, Any],
) -> None:
    """Store reduction metadata under the BenchRep namespace.

    Metadata are written to:

        adata.uns["benchrep"]["reductions"][key_added]

    where ``key_added`` is the ``adata.obsm`` key containing the reduced
    coordinates, such as ``"X_pca"``, ``"X_umap"``, or ``"X_tsne"``.
    """

    benchrep_uns = adata.uns.setdefault("benchrep", {})
    reductions_uns = benchrep_uns.setdefault("reductions", {})

    reductions_uns[key_added] = metadata