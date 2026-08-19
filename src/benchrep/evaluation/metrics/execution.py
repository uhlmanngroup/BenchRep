from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from benchrep.assembly.registries.core import Registry
from benchrep.evaluation.metrics.contracts import EvaluationMetric
from benchrep.evaluation.metrics.normalization import (
    _normalize_axis_labels,
    normalize_metric_result,
)
from benchrep.evaluation.utils import (
    RecoverableEvaluationStepError,
    validate_metric_params,
)


def execute_metric(
    *,
    registry: Registry,
    canonical_name: str,
    metric_positional_args: tuple[Any, ...],
    metric_kwargs: Mapping[str, Any],
    axis_labels_by_name: Mapping[str, Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """Execute and normalize one registered evaluation metric.

    This function operates on exactly one metric. It retrieves the registered
    ``EvaluationMetric`` descriptor, validates the supplied keyword arguments
    against its callable, checks that any declared vector axis is available,
    executes the callable, and normalizes its return value according to the
    descriptor's ``result_kind``.

    Parameters
    ----------
    registry
        Registry containing the metric. The registry supplies the metric category
        used in error messages, such as ``"embedding metric"`` or
        ``"internal clustering metric"``.
    canonical_name
        Registered canonical metric name.
    metric_positional_args
        Positional arguments supplied to the registered metric callable.
    metric_kwargs
        Keyword arguments supplied to the registered metric callable.

        The callable is invoked as:

            metric.fn(*metric_positional_args, **metric_kwargs)

    axis_labels_by_name
        Labels available for vector-result axes, keyed by axis name. A metric's
        ``vector_axis`` selects the relevant entry. For example:

            {
                "embedding_dimension": [
                    "dimension_0",
                    "dimension_1",
                    "dimension_2",
                ],
            }

        supplies labels to a metric declaring:

            vector_axis="embedding_dimension"

        Scalar and scalar-mapping metrics do not require axis labels.

    Returns
    -------
    dict[str, Any]
        The metric's canonical normalized result. For example, a vector result is
        returned as:

            {
                "kind": "vector",
                "axis": {
                    "name": "embedding_dimension",
                    "labels": [
                        "dimension_0",
                        "dimension_1",
                        "dimension_2",
                    ],
                },
                "values": [0.4, -0.2, 1.1],
            }

    Failure behavior
    ----------------
    A ``RecoverableEvaluationStepError`` raised by the callable or result
    normalizer is propagated unchanged so that ``execute_metric_group`` can record
    a per-metric failure. Invalid registration, unsupported parameters, missing
    axis metadata, and unexpected callable exceptions remain fatal and propagate
    out of the metric group.
    """

    metric = registry.get(canonical_name)

    if not isinstance(metric, EvaluationMetric):
        raise TypeError(
            f"{registry.name.capitalize()} registry entry "
            f"{canonical_name!r} must be an EvaluationMetric, got "
            f"{type(metric).__name__}."
        )

    validate_metric_params(
        metric_name=canonical_name,
        metric_fn=metric.fn,
        params=metric_kwargs,
        metric_kind=registry.name,
    )

    if metric.result_kind in {"vector", "vector_mapping"}:
        axis_name = metric.vector_axis

        if axis_name is None:
            raise RuntimeError(
                f"{registry.name.capitalize()} {canonical_name!r} declares "
                f"result kind {metric.result_kind!r} without a vector axis."
            )

        _normalize_axis_labels(
            axis_name=axis_name,
            axis_labels_by_name=axis_labels_by_name,
        )

    try:
        value = metric.fn(
            *metric_positional_args,
            **metric_kwargs,
        )
    except RecoverableEvaluationStepError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"Failed to compute {registry.name} {canonical_name!r}. "
            f"Original error ({type(error).__name__}): {error}"
        ) from error

    return normalize_metric_result(
        value,
        metric_name=canonical_name,
        metric=metric,
        axis_labels_by_name=axis_labels_by_name,
    )


def execute_metric_group(
    *,
    registry: Registry,
    canonical_metric_names: Sequence[str],
    metric_positional_args: tuple[Any, ...],
    metric_kwargs_by_name: Mapping[str, Mapping[str, Any]] | None = None,
    axis_labels_by_name: Mapping[str, Iterable[Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Execute a resolved group of metrics against shared positional arguments.

    ``canonical_metric_names`` and the outer keys of
    ``metric_kwargs_by_name`` must already be resolved to canonical registry
    names by the category runner.

    ``metric_kwargs_by_name`` maps each canonical metric name to the keyword
    arguments for that metric:

        {
            "standard_deviation": {"ddof": 1},
            "quantiles": {"q": [0.25, 0.75]},
        }

    Each metric is executed through ``execute_metric`` as:

        metric.fn(*metric_positional_args, **metric_kwargs)

    Returns ``(results, failures)``. ``results`` contains canonical normalized
    results keyed by canonical metric name. ``failures`` contains recoverable
    per-metric failures. Fatal errors propagate immediately.
    """

    if not canonical_metric_names:
        raise ValueError(
            f"At least one {registry.name} must be selected."
        )

    if metric_kwargs_by_name is None:
        metric_kwargs_by_name = {}

    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    for canonical_name in canonical_metric_names:
        try:
            results[canonical_name] = execute_metric(
                registry=registry,
                canonical_name=canonical_name,
                metric_positional_args=metric_positional_args,
                metric_kwargs=metric_kwargs_by_name.get(canonical_name, {}),
                axis_labels_by_name=axis_labels_by_name,
            )
        except RecoverableEvaluationStepError as error:
            failures[canonical_name] = str(error)

    return results, failures