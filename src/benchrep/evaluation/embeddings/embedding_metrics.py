from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import anndata as ad
import numpy as np

from benchrep.assembly.registries.core import EVAL_EMBEDDING_METRICS
from benchrep.assembly.registries.utils import resolve_registry_param_keys
from benchrep.evaluation.utils import (
    ArrayLike,
    RecoverableEvaluationStepError,
    validate_adata_x,
    validate_embedding_matrix,
)
from benchrep.evaluation.metrics.execution import execute_metric_group


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

    resolved_metric_params = resolve_registry_param_keys(
        params=metric_params,
        registry=EVAL_EMBEDDING_METRICS,
    )

    results, failures = execute_metric_group(
        registry=EVAL_EMBEDDING_METRICS,
        metric_names=selected,
        metric_positional_args=(embedding_array,),
        metric_kwargs_by_name=resolved_metric_params,
        axis_labels_by_name={
            "embedding_dimension": adata.var_names,
        },
    )

    _finalize_recoverable_embedding_metric_failures(
        results=results,
        failures=failures,
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


def _finalize_recoverable_embedding_metric_failures(
    *,
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
            "All selected embedding metrics failed recoverably. "
            f"{failure_details}"
        )

    for metric_name, reason in failures.items():
        warnings.warn(
            f"Skipped embedding metric {metric_name!r}: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )


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
        raise RecoverableEvaluationStepError(
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