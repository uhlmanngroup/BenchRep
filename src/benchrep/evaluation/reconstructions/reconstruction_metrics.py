from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal
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
    validate_reconstruction_arrays,
)
from benchrep.evaluation.metrics.execution import execute_metric_group


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
    reduction: Literal["global", "per_channel", "both"] = "global",
) -> dict[str, Any]:
    """Compute selected reconstruction metrics globally and/or per channel.

    Registered entries must be ``EvaluationMetric`` descriptors whose callables
    accept:

        metric.fn(inputs, reconstructions, **metric_kwargs)

    Return values are validated and normalized according to the metric's declared
    ``result_kind``. Vector and vector-mapping metrics may declare the
    ``"reconstruction_channel"`` axis.

    For global computation, that axis contains every reconstruction channel in
    array order. For per-channel computation, each metric invocation receives one
    channel and the axis contains only that channel's name.

    ``reduction`` controls whether metrics are computed over the complete arrays,
    independently for each channel, or both. Recoverable failures are retained per
    global or per-channel result; the step fails recoverably only if no requested
    result succeeds.

    Under the current selection behavior, ``selected=None`` selects every
    canonical reconstruction metric registered in the process.
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

    resolved_metric_names = resolve_registry_keys(
        selected=selected,
        registry=EVAL_RECONSTRUCTION_METRICS,
        none_policy="all",
    )

    if not resolved_metric_names:
        raise ValueError(
            "At least one reconstruction metric must be selected."
        )

    resolved_metric_kwargs_by_name = resolve_registry_param_keys(
        params=metric_params,
        registry=EVAL_RECONSTRUCTION_METRICS,
    )

    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    n_successful_results = 0

    input_array_with_channel_axis = ensure_reconstruction_channel_axis(
        input_array
    )
    reconstruction_array_with_channel_axis = (
        ensure_reconstruction_channel_axis(reconstruction_array)
    )

    channel_names = resolve_reconstruction_channel_names(
        metadata=reconstruction_input.metadata,
        n_channels=input_array_with_channel_axis.shape[1],
    )

    if reduction in {"global", "both"}:
        global_results, global_failures = execute_metric_group(
            registry=EVAL_RECONSTRUCTION_METRICS,
            canonical_metric_names=resolved_metric_names,
            metric_positional_args=(
                input_array,
                reconstruction_array,
            ),
            metric_kwargs_by_name=resolved_metric_kwargs_by_name,
            axis_labels_by_name={
                "reconstruction_channel": channel_names,
            },
        )

        results["global"] = global_results
        n_successful_results += len(global_results)
        failures.update(
            {
                f"global.{metric_name}": reason
                for metric_name, reason in global_failures.items()
            }
        )

    per_channel_results: dict[str, dict[str, Any]] = {}
    per_channel_failures: dict[str, str] = {}

    if reduction in {"per_channel", "both"}:
        for channel_index, channel_name in enumerate(channel_names):
            channel_input_array = input_array_with_channel_axis[
                :, channel_index, :, :
            ]
            channel_reconstruction_array = (
                reconstruction_array_with_channel_axis[
                    :, channel_index, :, :
                ]
            )

            channel_results, channel_failures = execute_metric_group(
                registry=EVAL_RECONSTRUCTION_METRICS,
                canonical_metric_names=resolved_metric_names,
                metric_positional_args=(
                    channel_input_array,
                    channel_reconstruction_array,
                ),
                metric_kwargs_by_name=resolved_metric_kwargs_by_name,
                axis_labels_by_name={
                    "reconstruction_channel": [channel_name],
                },
            )

            per_channel_results[channel_name] = channel_results
            per_channel_failures.update(
                {
                    f"{channel_name}.{metric_name}": reason
                    for metric_name, reason in channel_failures.items()
                }
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
        "params": resolved_metric_kwargs_by_name,
        "reduction": reduction,
        "shape": tuple(input_array.shape),
        "n_images": int(input_array.shape[0]),
    }


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
