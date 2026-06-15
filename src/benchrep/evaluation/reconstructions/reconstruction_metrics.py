from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from benchrep.assembly.registry import EVAL_RECONSTRUCTION_METRICS
from benchrep.assembly.registry_utils import (
    resolve_registry_keys,
    resolve_registry_param_keys,
)
from benchrep.evaluation.reconstructions.data import ReconstructionEvaluationInput

from benchrep.evaluation.utils import (
    ArrayLike,
    ensure_reconstruction_channel_axis,
    resolve_reconstruction_channel_names,
    to_python_scalar,
    validate_metric_params,
    validate_reconstruction_arrays,
)


def mean_absolute_error(inputs: ArrayLike, reconstructions: ArrayLike) -> float:
    input_array, reconstruction_array = validate_reconstruction_arrays(
        inputs=inputs,
        reconstructions=reconstructions,
    )
    return float(np.mean(np.abs(input_array - reconstruction_array)))


def mean_squared_error(inputs: ArrayLike, reconstructions: ArrayLike) -> float:
    input_array, reconstruction_array = validate_reconstruction_arrays(
        inputs=inputs,
        reconstructions=reconstructions,
    )
    return float(np.mean((input_array - reconstruction_array) ** 2))


def root_mean_squared_error(inputs: ArrayLike, reconstructions: ArrayLike) -> float:
    return float(np.sqrt(mean_squared_error(inputs, reconstructions)))


def max_absolute_error(inputs: ArrayLike, reconstructions: ArrayLike) -> float:
    input_array, reconstruction_array = validate_reconstruction_arrays(
        inputs=inputs,
        reconstructions=reconstructions,
    )
    return float(np.max(np.abs(input_array - reconstruction_array)))


def compute_reconstruction_metrics(
    reconstruction_input: ReconstructionEvaluationInput,
    *,
    selected: Sequence[str] | None = None,
    metric_params: Mapping[str, Mapping[str, Any]] | None = None,
    reduction: str = "global",
) -> dict[str, Any]:
    """Compute reconstruction metrics from reconstruction input data.

    Registered reconstruction metric callables must follow the contract:

        metric_fn(inputs, reconstructions, **params) -> scalar

    ``selected`` should contain canonical metric names or aliases registered in
    ``EVAL_RECONSTRUCTION_METRICS``. If ``selected`` is ``None``, all canonical
    registered reconstruction metrics are computed.
    """
    input_array, reconstruction_array = validate_reconstruction_arrays(
        inputs=reconstruction_input.inputs,
        reconstructions=reconstruction_input.reconstructions,
    )

    if reduction not in {"global", "per_channel", "both"}:
        raise ValueError(
            "reduction must be one of 'global', 'per_channel', or 'both', "
            f"got {reduction!r}."
        )

    metric_names = resolve_registry_keys(
        selected=selected,
        registry=EVAL_RECONSTRUCTION_METRICS,
        none_policy="all",
    )
    metric_params = resolve_registry_param_keys(
        params=metric_params,
        registry=EVAL_RECONSTRUCTION_METRICS,
    )

    results: dict[str, Any] = {}

    if reduction in {"global", "both"}:
        results["global"] = _compute_metric_group(
            input_array=input_array,
            reconstruction_array=reconstruction_array,
            metric_names=metric_names,
            metric_params=metric_params,
        )

    if reduction in {"per_channel", "both"}:
        results["per_channel"] = _compute_per_channel_metric_group(
            reconstruction_input=reconstruction_input,
            input_array=input_array,
            reconstruction_array=reconstruction_array,
            metric_names=metric_names,
            metric_params=metric_params,
        )

    return {
        "metrics": results,
        "params": metric_params,
        "reduction": reduction,
        "shape": tuple(input_array.shape),
        "n_images": int(input_array.shape[0]),
    }


def _compute_metric_group(
    *,
    input_array: np.ndarray,
    reconstruction_array: np.ndarray,
    metric_names: Sequence[str],
    metric_params: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute selected metrics for one input/reconstruction pair."""

    results: dict[str, Any] = {}

    for metric_name in metric_names:
        metric_fn = EVAL_RECONSTRUCTION_METRICS.get(metric_name)
        params = metric_params.get(metric_name, {})

        validate_metric_params(
            metric_name=metric_name,
            metric_fn=metric_fn,
            params=params,
            metric_kind="reconstruction metric",
        )

        try:
            value = metric_fn(input_array, reconstruction_array, **params)
        except Exception as error:
            raise RuntimeError(
                f"Failed to compute reconstruction metric {metric_name!r}. "
                "The metric callable was found, but execution failed."
            ) from error

        results[metric_name] = to_python_scalar(value)

    return results


def _compute_per_channel_metric_group(
    *,
    reconstruction_input: ReconstructionEvaluationInput,
    input_array: np.ndarray,
    reconstruction_array: np.ndarray,
    metric_names: Sequence[str],
    metric_params: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute selected metrics independently for each image channel."""

    input_array = ensure_reconstruction_channel_axis(input_array)
    reconstruction_array = ensure_reconstruction_channel_axis(reconstruction_array)

    n_channels = input_array.shape[1]
    channel_names = resolve_reconstruction_channel_names(
        metadata=reconstruction_input.metadata,
        n_channels=n_channels,
    )

    results: dict[str, dict[str, Any]] = {}

    for channel_index, channel_name in enumerate(channel_names):
        results[channel_name] = _compute_metric_group(
            input_array=input_array[:, channel_index, :, :],
            reconstruction_array=reconstruction_array[:, channel_index, :, :],
            metric_names=metric_names,
            metric_params=metric_params,
        )

    return results