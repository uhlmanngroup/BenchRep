from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from numbers import Real
import warnings

import numpy as np

from benchrep.assembly.registries.core import EVAL_RECONSTRUCTION_METRICS
from benchrep.assembly.registries.utils import (
    resolve_registry_keys,
    resolve_registry_param_keys,
)
from benchrep.evaluation.reconstructions.data import ReconstructionEvaluationInput

from benchrep.evaluation.utils import (
    ArrayLike,
    RecoverableEvaluationStepError,
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

    if not metric_names:
        raise ValueError(
            "At least one reconstruction metric must be selected."
        )

    metric_params = resolve_registry_param_keys(
        params=metric_params,
        registry=EVAL_RECONSTRUCTION_METRICS,
    )

    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    n_successful_results = 0

    if reduction in {"global", "both"}:
        global_results, global_failures = _compute_metric_group(
            input_array=input_array,
            reconstruction_array=reconstruction_array,
            metric_names=metric_names,
            metric_params=metric_params,
        )

        results["global"] = global_results
        n_successful_results += len(global_results)
        failures.update(
            {
                f"global.{metric_name}": reason
                for metric_name, reason in global_failures.items()
            }
        )

    if reduction in {"per_channel", "both"}:
        per_channel_results, per_channel_failures = (
            _compute_per_channel_metric_group(
                reconstruction_input=reconstruction_input,
                input_array=input_array,
                reconstruction_array=reconstruction_array,
                metric_names=metric_names,
                metric_params=metric_params,
            )
        )

        results["per_channel"] = per_channel_results
        n_successful_results += sum(
            len(channel_results)
            for channel_results in per_channel_results.values()
        )
        failures.update(
            {
                f"per_channel.{result_key}": reason
                for result_key, reason in per_channel_failures.items()
            }
        )

    _finalize_recoverable_reconstruction_metric_failures(
        failures=failures,
        n_successful_results=n_successful_results,
    )

    return {
        "metrics": results,
        "params": metric_params,
        "reduction": reduction,
        "shape": tuple(input_array.shape),
        "n_images": int(input_array.shape[0]),
    }


def _validate_reconstruction_metric_value(
    value: Any,
    *,
    metric_name: str,
) -> Real:
    """Validate and return one reconstruction metric scalar."""

    try:
        scalar = to_python_scalar(value)
    except Exception as error:
        raise TypeError(
            f"Reconstruction metric {metric_name!r} must return a scalar."
        ) from error

    if isinstance(scalar, bool) or not isinstance(scalar, Real):
        raise TypeError(
            f"Reconstruction metric {metric_name!r} must return a real "
            f"numeric scalar, got {type(scalar).__name__}."
        )

    if not np.isfinite(float(scalar)):
        raise RecoverableEvaluationStepError(
            f"Reconstruction metric {metric_name!r} returned a non-finite "
            f"value: {scalar!r}."
        )

    return scalar


def _finalize_recoverable_reconstruction_metric_failures(
    *,
    failures: Mapping[str, str],
    n_successful_results: int,
) -> None:
    """Warn for partial failures or fail when no result succeeded."""

    if not failures:
        return

    failure_details = "; ".join(
        f"{result_key}: {reason}"
        for result_key, reason in failures.items()
    )

    if n_successful_results == 0:
        raise RecoverableEvaluationStepError(
            "All requested reconstruction metric results failed "
            f"recoverably. {failure_details}"
        )

    for result_key, reason in failures.items():
        warnings.warn(
            f"Skipped reconstruction metric result {result_key!r}: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )


def _compute_metric_group(
    *,
    input_array: np.ndarray,
    reconstruction_array: np.ndarray,
    metric_names: Sequence[str],
    metric_params: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Compute selected metrics for one input/reconstruction pair."""

    results: dict[str, Any] = {}
    failures: dict[str, str] = {}

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
            value = metric_fn(
                input_array,
                reconstruction_array,
                **params,
            )
        except RecoverableEvaluationStepError as error:
            failures[metric_name] = str(error)
            continue
        except Exception as error:
            raise RuntimeError(
                f"Failed to compute reconstruction metric {metric_name!r}. "
                "The metric callable was found, but execution failed."
            ) from error

        try:
            results[metric_name] = _validate_reconstruction_metric_value(
                value,
                metric_name=metric_name,
            )
        except RecoverableEvaluationStepError as error:
            failures[metric_name] = str(error)

    return results, failures


def _compute_per_channel_metric_group(
    *,
    reconstruction_input: ReconstructionEvaluationInput,
    input_array: np.ndarray,
    reconstruction_array: np.ndarray,
    metric_names: Sequence[str],
    metric_params: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Compute selected metrics independently for each image channel."""

    input_array = ensure_reconstruction_channel_axis(input_array)
    reconstruction_array = ensure_reconstruction_channel_axis(
        reconstruction_array
    )

    n_channels = input_array.shape[1]
    channel_names = resolve_reconstruction_channel_names(
        metadata=reconstruction_input.metadata,
        n_channels=n_channels,
    )

    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    for channel_index, channel_name in enumerate(channel_names):
        channel_results, channel_failures = _compute_metric_group(
            input_array=input_array[:, channel_index, :, :],
            reconstruction_array=(
                reconstruction_array[:, channel_index, :, :]
            ),
            metric_names=metric_names,
            metric_params=metric_params,
        )

        results[channel_name] = channel_results
        failures.update(
            {
                f"{channel_name}.{metric_name}": reason
                for metric_name, reason in channel_failures.items()
            }
        )

    return results, failures