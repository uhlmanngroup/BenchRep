from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import numpy as np

from benchrep.assembly.registries.core import EVAL_EMBEDDING_METRICS
from benchrep.assembly.registries.utils import (
    resolve_registry_keys,
    resolve_registry_param_keys,
)
from benchrep.evaluation.utils import (
    ArrayLike,
    validate_adata_x,
    validate_embedding_matrix,
    validate_metric_params,
)


def compute_embedding_metrics(
    adata: ad.AnnData,
    *,
    selected: Sequence[str],
    metric_params: Mapping[str, Mapping[str, Any]] | None = None,
    overwrite: bool = False,
) -> ad.AnnData:
    """Compute descriptive statistics for each embedding dimension.

    Registered embedding metric callables must follow the contract:

        metric_fn(embeddings, **params) -> dimensionwise values

    A metric may return either:

    - a one-dimensional array with one value per embedding dimension; or
    - a mapping of named one-dimensional arrays, as used by the quantile metric.

    Results are stored under:

        adata.uns["benchrep"]["metrics"]["embedding"]
    """

    validate_adata_x(adata)
    _check_embedding_metric_result_available(
        adata,
        overwrite=overwrite,
    )

    embedding_array = validate_embedding_matrix(adata.X)

    metric_names = resolve_registry_keys(
        selected=selected,
        registry=EVAL_EMBEDDING_METRICS,
        none_policy="preserve",
    )

    if not metric_names:
        raise ValueError(
            "At least one embedding metric must be selected."
        )

    resolved_metric_params = resolve_registry_param_keys(
        params=metric_params,
        registry=EVAL_EMBEDDING_METRICS,
    )

    results: dict[
        str,
        list[float] | dict[str, list[float]],
    ] = {}

    for metric_name in metric_names:
        metric_fn = EVAL_EMBEDDING_METRICS.get(metric_name)
        params = resolved_metric_params.get(metric_name, {})

        validate_metric_params(
            metric_name=metric_name,
            metric_fn=metric_fn,
            params=params,
            metric_kind="embedding metric",
        )

        try:
            value = metric_fn(
                embedding_array,
                **params,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to compute embedding metric {metric_name!r}. "
                f"Original error ({type(error).__name__}): {error}"
            ) from error

        results[metric_name] = (
            _normalize_dimensionwise_metric_result(
                value,
                metric_name=metric_name,
                n_dimensions=embedding_array.shape[1],
            )
        )

    _store_embedding_metric_result(
        adata,
        result={
            "dimension_names": [
                str(name)
                for name in adata.var_names
            ],
            "metrics": results,
            "params": resolved_metric_params,
            "n_samples": int(embedding_array.shape[0]),
            "n_dimensions": int(embedding_array.shape[1]),
        },
    )

    return adata


def _normalize_dimensionwise_metric_result(
    value: Any,
    *,
    metric_name: str,
    n_dimensions: int,
) -> list[float] | dict[str, list[float]]:
    """Validate and normalize one embedding metric result."""

    if isinstance(value, Mapping):
        if not value:
            raise ValueError(
                f"Embedding metric {metric_name!r} returned an empty mapping."
            )

        normalized: dict[str, list[float]] = {}

        for key, nested_value in value.items():
            normalized_key = str(key)

            if normalized_key in normalized:
                raise ValueError(
                    f"Embedding metric {metric_name!r} returned duplicate "
                    f"result key {normalized_key!r}."
                )

            normalized[normalized_key] = (
                _normalize_dimensionwise_vector(
                    nested_value,
                    metric_name=metric_name,
                    result_key=normalized_key,
                    n_dimensions=n_dimensions,
                )
            )

        return normalized

    return _normalize_dimensionwise_vector(
        value,
        metric_name=metric_name,
        result_key=None,
        n_dimensions=n_dimensions,
    )


def _normalize_dimensionwise_vector(
    value: Any,
    *,
    metric_name: str,
    result_key: str | None,
    n_dimensions: int,
) -> list[float]:
    """Validate one dimension-wise result vector."""

    try:
        result_array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Embedding metric {metric_name!r} returned a value that "
            "could not be converted to a NumPy array."
        ) from error

    result_location = (
        metric_name
        if result_key is None
        else f"{metric_name}.{result_key}"
    )

    if result_array.ndim != 1:
        raise ValueError(
            f"Embedding metric result {result_location!r} must be "
            f"one-dimensional, got shape {result_array.shape}."
        )

    if result_array.shape[0] != n_dimensions:
        raise ValueError(
            f"Embedding metric result {result_location!r} must contain "
            f"one value per embedding dimension. Expected {n_dimensions}, "
            f"got {result_array.shape[0]}."
        )

    if (
        not np.issubdtype(result_array.dtype, np.number)
        or np.issubdtype(result_array.dtype, np.complexfloating)
    ):
        raise TypeError(
            f"Embedding metric result {result_location!r} must contain "
            f"real numeric values, got dtype {result_array.dtype}."
        )

    if not np.isfinite(result_array).all():
        raise ValueError(
            f"Embedding metric result {result_location!r} contains "
            "non-finite values."
        )

    normalized_array = result_array.astype(
        np.float64,
        copy=False,
    )
    normalized_values: list[float] = normalized_array.tolist()

    return normalized_values


def _check_embedding_metric_result_available(
    adata: ad.AnnData,
    *,
    overwrite: bool,
) -> None:
    """Check whether embedding metric results can be written."""

    metrics = (
        adata.uns
        .get("benchrep", {})
        .get("metrics", {})
    )

    if "embedding" in metrics and not overwrite:
        raise KeyError(
            "BenchRep embedding metrics already contain results. "
            "Pass overwrite=True to replace them."
        )


def _store_embedding_metric_result(
    adata: ad.AnnData,
    *,
    result: Mapping[str, Any],
) -> None:
    """Store embedding metric results under the BenchRep namespace."""

    benchrep_uns = adata.uns.setdefault("benchrep", {})
    metrics_uns = benchrep_uns.setdefault("metrics", {})

    metrics_uns["embedding"] = dict(result)


def dimensionwise_mean(embeddings: ArrayLike) -> np.ndarray:
    """Return the mean of each embedding dimension across samples."""

    embedding_array = validate_embedding_matrix(embeddings)

    return np.asarray(
        np.mean(embedding_array, axis=0, dtype=np.float64),
        dtype=np.float64,
    )


def dimensionwise_median(embeddings: ArrayLike) -> np.ndarray:
    """Return the median of each embedding dimension across samples."""

    embedding_array = validate_embedding_matrix(embeddings)

    return np.asarray(
        np.median(embedding_array, axis=0),
        dtype=np.float64,
    )


def dimensionwise_standard_deviation(
    embeddings: ArrayLike,
    *,
    ddof: int = 0,
) -> np.ndarray:
    """Return the standard deviation of each embedding dimension."""

    embedding_array = validate_embedding_matrix(embeddings)

    if not isinstance(ddof, int) or isinstance(ddof, bool):
        raise TypeError(
            f"ddof must be an integer, got {type(ddof).__name__}."
        )

    if ddof < 0:
        raise ValueError(f"ddof must be non-negative, got {ddof}.")

    if ddof >= embedding_array.shape[0]:
        raise ValueError(
            "ddof must be smaller than the number of samples, got "
            f"ddof={ddof} and n_samples={embedding_array.shape[0]}."
        )

    return np.asarray(
        np.std(
            embedding_array,
            axis=0,
            ddof=ddof,
            dtype=np.float64,
        ),
        dtype=np.float64,
    )


def dimensionwise_minimum(embeddings: ArrayLike) -> np.ndarray:
    """Return the minimum of each embedding dimension across samples."""

    embedding_array = validate_embedding_matrix(embeddings)

    return np.asarray(
        np.min(embedding_array, axis=0),
        dtype=np.float64,
    )


def dimensionwise_maximum(embeddings: ArrayLike) -> np.ndarray:
    """Return the maximum of each embedding dimension across samples."""

    embedding_array = validate_embedding_matrix(embeddings)

    return np.asarray(
        np.max(embedding_array, axis=0),
        dtype=np.float64,
    )


def dimensionwise_quantiles(
    embeddings: ArrayLike,
    *,
    q: float | Sequence[float] = (0.25, 0.75),
) -> dict[str, np.ndarray]:
    """Return requested quantiles for each embedding dimension."""

    embedding_array = validate_embedding_matrix(embeddings)

    try:
        quantiles = np.asarray(q, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "q must be a numeric value or sequence of numeric values."
        ) from error

    if quantiles.ndim == 0:
        quantiles = quantiles.reshape(1)

    if quantiles.ndim != 1 or quantiles.size == 0:
        raise ValueError(
            "q must contain at least one quantile."
        )

    if not np.isfinite(quantiles).all():
        raise ValueError("q contains non-finite values.")

    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError(
            "Every quantile in q must be between 0 and 1."
        )

    if np.unique(quantiles).size != quantiles.size:
        raise ValueError("q must not contain duplicate quantiles.")

    values = np.quantile(
        embedding_array,
        q=quantiles,
        axis=0,
        method="linear",
    )

    results: dict[str, np.ndarray] = {}

    for index in range(quantiles.size):
        quantile = quantiles[index].item()
        results[str(quantile)] = np.asarray(
            values[index],
            dtype=np.float64,
        )

    return results