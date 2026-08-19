from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import numpy as np

from benchrep.evaluation.utils import (
    RecoverableEvaluationStepError,
    to_python_scalar,
)
from benchrep.evaluation.metrics.contracts import EvaluationMetric


def _normalize_scalar(
    value: Any,
    *,
    metric_name: str,
    result_key: str | None = None,
) -> float:
    """Normalize a scalar metric result to a finite Python float.

    Expected metric return
    ----------------------
    A real scalar, such as ``0.42``, ``np.float64(0.42)``, or a
    zero-dimensional numeric array. Boolean values, vectors, complex numbers,
    and non-finite values are rejected.

    Example
    -------
    ``np.float64(0.42)`` is normalized to ``0.42``.
    """

    result_location = (
        metric_name
        if result_key is None
        else f"{metric_name}.{result_key}"
    )

    try:
        scalar = to_python_scalar(value)
    except Exception as error:
        raise RecoverableEvaluationStepError(
            f"Metric result {result_location!r} must be a scalar."
        ) from error

    if isinstance(scalar, bool) or not isinstance(scalar, Real):
        raise RecoverableEvaluationStepError(
            f"Metric result {result_location!r} must be a real numeric "
            f"scalar, got {type(scalar).__name__}."
        )

    normalized = float(scalar)

    if not np.isfinite(normalized):
        raise RecoverableEvaluationStepError(
            f"Metric result {result_location!r} is non-finite: {scalar!r}."
        )

    return normalized


def _normalize_vector(
    value: Any,
    *,
    metric_name: str,
    axis_name: str,
    expected_length: int,
    result_key: str | None = None,
) -> list[float]:
    """Normalize a vector metric result to a list of finite Python floats.

    Expected metric return
    ----------------------
    A one-dimensional real numeric array-like object with exactly one value
    for each entry of the metric's declared vector axis.

    For example, a metric with:

        vector_axis="embedding_dimension"

    may return:

        np.array([0.1, -0.2, 3.3])

    when the embedding has three dimensions. The result is normalized to:

        [0.1, -0.2, 3.3]

    The dispatcher separately stores the corresponding axis name and labels.
    """

    result_location = (
        metric_name
        if result_key is None
        else f"{metric_name}.{result_key}"
    )

    try:
        result_array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise RecoverableEvaluationStepError(
            f"Metric result {result_location!r} could not be converted "
            "to a NumPy array."
        ) from error

    if result_array.ndim != 1:
        raise RecoverableEvaluationStepError(
            f"Metric result {result_location!r} must be one-dimensional, "
            f"got shape {result_array.shape}."
        )

    if result_array.shape[0] != expected_length:
        raise RecoverableEvaluationStepError(
            f"Metric result {result_location!r} must contain one value per "
            f"{axis_name!r} entry. Expected {expected_length}, got "
            f"{result_array.shape[0]}."
        )

    if (
        not np.issubdtype(result_array.dtype, np.number)
        or np.issubdtype(result_array.dtype, np.complexfloating)
    ):
        raise RecoverableEvaluationStepError(
            f"Metric result {result_location!r} must contain real numeric "
            f"values, got dtype {result_array.dtype}."
        )

    if not np.isfinite(result_array).all():
        raise RecoverableEvaluationStepError(
            f"Metric result {result_location!r} contains non-finite values."
        )

    return result_array.astype(np.float64, copy=False).tolist()


def _normalize_mapping_items(
    value: Any,
    *,
    metric_name: str,
) -> list[tuple[str, Any]]:
    """Normalize mapping keys while preserving their original order."""

    if not isinstance(value, Mapping):
        raise RecoverableEvaluationStepError(
            f"Metric {metric_name!r} must return a mapping."
        )

    if not value:
        raise RecoverableEvaluationStepError(
            f"Metric {metric_name!r} returned an empty mapping."
        )

    normalized_items: list[tuple[str, Any]] = []
    seen_keys: set[str] = set()

    for key, nested_value in value.items():
        normalized_key = str(key)

        if normalized_key in seen_keys:
            raise RecoverableEvaluationStepError(
                f"Metric {metric_name!r} returned duplicate result key "
                f"{normalized_key!r} after key normalization."
            )

        normalized_items.append((normalized_key, nested_value))
        seen_keys.add(normalized_key)

    return normalized_items


def _normalize_scalar_mapping(
    value: Any,
    *,
    metric_name: str,
) -> tuple[list[str], list[float]]:
    """Normalize a named mapping of scalar metric results.

    Expected metric return
    ----------------------
    A non-empty mapping whose values are finite real scalars. Mapping keys are
    converted to strings and stored separately from their values.

    For example, per-label scores may be returned as:

        {
            "healthy": 0.90,
            "disease": 0.60,
        }

    and are normalized to:

        (
            ["healthy", "disease"],
            [0.90, 0.60],
        )
    """

    keys: list[str] = []
    values: list[float] = []

    for key, nested_value in _normalize_mapping_items(
        value,
        metric_name=metric_name,
    ):
        keys.append(key)
        values.append(
            _normalize_scalar(
                nested_value,
                metric_name=metric_name,
                result_key=key,
            )
        )

    return keys, values


def _normalize_vector_mapping(
    value: Any,
    *,
    metric_name: str,
    axis_name: str,
    expected_length: int,
) -> tuple[list[str], list[list[float]]]:
    """Normalize a named mapping of vector metric results.

    Expected metric return
    ----------------------
    A non-empty mapping whose values are one-dimensional real numeric
    array-like objects aligned to the same declared vector axis.

    For example, Jaccard scores for each true label across three predicted
    clusters may be returned as:

        {
            "healthy": [0.80, 0.15, 0.05],
            "disease": [0.10, 0.20, 0.70],
        }

    with:

        vector_axis="predicted_cluster"

    and axis labels:

        ["cluster_0", "cluster_1", "cluster_2"]

    The mapping is normalized to parallel keys and values:

        (
            ["healthy", "disease"],
            [
                [0.80, 0.15, 0.05],
                [0.10, 0.20, 0.70],
            ],
        )

    The dispatcher separately stores the shared axis name and labels.
    """

    keys: list[str] = []
    values: list[list[float]] = []

    for key, nested_value in _normalize_mapping_items(
        value,
        metric_name=metric_name,
    ):
        keys.append(key)
        values.append(
            _normalize_vector(
                nested_value,
                metric_name=metric_name,
                result_key=key,
                axis_name=axis_name,
                expected_length=expected_length,
            )
        )

    return keys, values


def _normalize_axis_labels(
    *,
    axis_name: str,
    axis_labels_by_name: Mapping[str, Iterable[Any]] | None,
) -> list[str]:
    """Resolve labels for one declared vector axis."""

    if axis_labels_by_name is None or axis_name not in axis_labels_by_name:
        raise RuntimeError(
            f"No labels were provided for metric vector axis "
            f"{axis_name!r}."
        )

    raw_labels = axis_labels_by_name[axis_name]

    if isinstance(raw_labels, str | bytes):
        raise TypeError(
            f"Labels for metric vector axis {axis_name!r} must be an "
            "iterable of labels, not a string."
        )

    try:
        normalized_labels = [
            str(label)
            for label in raw_labels
        ]
    except TypeError as error:
        raise TypeError(
            f"Labels for metric vector axis {axis_name!r} must be iterable."
        ) from error

    if not normalized_labels:
        raise ValueError(
            f"Metric vector axis {axis_name!r} must contain at least one label."
        )

    return normalized_labels


def normalize_metric_result(
    value: Any,
    *,
    metric_name: str,
    metric: EvaluationMetric,
    axis_labels_by_name: Mapping[str, Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """Validate and convert one metric result to its canonical representation."""

    if metric.result_kind == "scalar":
        return {
            "kind": "scalar",
            "value": _normalize_scalar(
                value,
                metric_name=metric_name,
            ),
        }

    if metric.result_kind == "scalar_mapping":
        keys, values = _normalize_scalar_mapping(
            value,
            metric_name=metric_name,
        )
        return {
            "kind": "scalar_mapping",
            "keys": keys,
            "values": values,
        }

    axis_name = metric.vector_axis

    if axis_name is None:
        raise RuntimeError(
            f"Metric {metric_name!r} declares result kind "
            f"{metric.result_kind!r} without a vector axis."
        )

    normalized_axis_labels = _normalize_axis_labels(
        axis_name=axis_name,
        axis_labels_by_name=axis_labels_by_name,
    )

    if metric.result_kind == "vector":
        return {
            "kind": "vector",
            "axis": {
                "name": axis_name,
                "labels": normalized_axis_labels,
            },
            "values": _normalize_vector(
                value,
                metric_name=metric_name,
                axis_name=axis_name,
                expected_length=len(normalized_axis_labels),
            ),
        }

    if metric.result_kind == "vector_mapping":
        keys, values = _normalize_vector_mapping(
            value,
            metric_name=metric_name,
            axis_name=axis_name,
            expected_length=len(normalized_axis_labels),
        )
        return {
            "kind": "vector_mapping",
            "keys": keys,
            "axis": {
                "name": axis_name,
                "labels": normalized_axis_labels,
            },
            "values": values,
        }

    raise RuntimeError(
        f"Metric {metric_name!r} declares unsupported result kind "
        f"{metric.result_kind!r}."
    )