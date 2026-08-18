from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import numpy as np

from benchrep.evaluation.reconstructions.data import ReconstructionEvaluationInput
from benchrep.evaluation.utils import (
    RecoverableEvaluationStepError,
    validate_reconstruction_arrays,
)


SUPPORTED_ERROR_MAP_KINDS = {
    "absolute",
    "squared",
    "signed",
    "relative",
    "normalized_absolute_global",
    "normalized_absolute_per_channel",
}

def compute_error_maps(
    reconstruction_input: ReconstructionEvaluationInput,
    *,
    kinds: Sequence[str] = ("absolute", "signed", "relative"),
    n_examples: int | None = None,
    denominator_floor: float = 1e-8,
    data_range: float | None = None,
) -> Mapping[str, Any]:
    """Compute in-memory reconstruction error maps.

    Supported kinds
    ----------------
    absolute
        Absolute residual: ``abs(reconstruction - input)``.
    squared
        Squared residual: ``(reconstruction - input) ** 2``.
    signed
        Signed residual: ``reconstruction - input``.
    relative
        Absolute residual divided by the absolute input intensity, with a
        denominator floor for stability.
    normalized_absolute_global
        Absolute residual divided by one global input intensity range. The
        range is inferred as ``max(input) - min(input)`` unless ``data_range``
        is provided explicitly.
    normalized_absolute_per_channel
        Absolute residual divided by a separate intensity range for each
        channel, computed across all provided samples for that channel.

    Error maps are returned as arrays. Disk export to TIFF/PNG belongs to the
    records/export layer.
    """
    kinds = tuple(dict.fromkeys(kinds))

    if not kinds:
        raise ValueError("kinds must contain at least one error map kind.")

    invalid_kinds = [kind for kind in kinds if kind not in SUPPORTED_ERROR_MAP_KINDS]

    if invalid_kinds:
        raise ValueError(
            f"Unsupported error map kinds: {invalid_kinds!r}. "
            f"Supported kinds: {sorted(SUPPORTED_ERROR_MAP_KINDS)}."
        )

    if (
        data_range is not None
        and "normalized_absolute_global" not in kinds
    ):
        warnings.warn(
            "`data_range` was ignored because it only applies to "
            "`normalized_absolute_global`, which was not requested.",
            UserWarning,
            stacklevel=2,
        )
        data_range = None

    inputs, reconstructions = validate_reconstruction_arrays(
        inputs=reconstruction_input.inputs,
        reconstructions=reconstruction_input.reconstructions,
    )

    resolved_n_examples = _resolve_n_examples(
        n_examples=n_examples,
        n_available=inputs.shape[0],
    )

    if resolved_n_examples is not None:
        inputs = inputs[:resolved_n_examples]
        reconstructions = reconstructions[:resolved_n_examples]

    error_maps_dict: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    for error_kind in kinds:
        try:
            error_maps = _compute_error_map_array(
                inputs=inputs,
                reconstructions=reconstructions,
                kind=error_kind,
                denominator_floor=denominator_floor,
                data_range=data_range,
            )

            error_maps = _validate_error_map_array(
                error_maps,
                error_kind=error_kind,
                expected_shape=inputs.shape,
            )

        except RecoverableEvaluationStepError as error:
            failures[error_kind] = str(error)
            continue

        error_maps_dict[error_kind] = {
            "error_maps": error_maps,
            "shape": tuple(error_maps.shape),
            "n_examples": error_maps.shape[0],
            "params": {
                "denominator_floor": denominator_floor,
                "data_range": (
                    data_range
                    if error_kind == "normalized_absolute_global"
                    else None
                ),
            },
        }

    _finalize_recoverable_error_map_failures(
        results=error_maps_dict,
        failures=failures,
    )

    return error_maps_dict


def _validate_error_map_array(
    value: Any,
    *,
    error_kind: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Validate and return one error-map array."""

    try:
        error_maps = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Error map kind {error_kind!r} returned a value that could not "
            "be converted to a NumPy array."
        ) from error

    if error_maps.shape != expected_shape:
        raise ValueError(
            f"Error map kind {error_kind!r} must preserve the reconstruction "
            f"shape. Expected {expected_shape}, got {error_maps.shape}."
        )

    if (
        not np.issubdtype(error_maps.dtype, np.number)
        or np.issubdtype(error_maps.dtype, np.complexfloating)
    ):
        raise TypeError(
            f"Error map kind {error_kind!r} must return real numeric values, "
            f"got dtype {error_maps.dtype}."
        )

    if not np.isfinite(error_maps).all():
        raise RecoverableEvaluationStepError(
            f"Error map kind {error_kind!r} produced non-finite values."
        )

    return error_maps


def _finalize_recoverable_error_map_failures(
    *,
    results: Mapping[str, Any],
    failures: Mapping[str, str],
) -> None:
    """Warn for partial failures or fail when no error map succeeded."""

    if not failures:
        return

    failure_details = "; ".join(
        f"{error_kind}: {reason}"
        for error_kind, reason in failures.items()
    )

    if not results:
        raise RecoverableEvaluationStepError(
            "All requested error map kinds failed recoverably. "
            f"{failure_details}"
        )

    for error_kind, reason in failures.items():
        warnings.warn(
            f"Skipped error map kind {error_kind!r}: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )


def _compute_error_map_array(
    *,
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    kind: str,
    denominator_floor: float,
    data_range: float | None,
) -> np.ndarray:
    """Compute one kind of reconstruction error map."""

    if denominator_floor <= 0:
        raise ValueError(
            f"denominator_floor must be > 0, got {denominator_floor}."
        )

    residual = reconstructions - inputs

    if kind == "absolute":
        return np.abs(residual)

    if kind == "squared":
        return np.square(residual)

    if kind == "signed":
        return residual

    if kind == "relative":
        denominator = np.maximum(np.abs(inputs), denominator_floor)
        return np.abs(residual) / denominator

    if kind == "normalized_absolute_global":
        resolved_data_range = _resolve_global_data_range(
            inputs=inputs,
            data_range=data_range,
            denominator_floor=denominator_floor,
        )
        return np.abs(residual) / resolved_data_range

    if kind == "normalized_absolute_per_channel":
        per_channel_data_range = _resolve_per_channel_data_range(
            inputs=inputs,
            denominator_floor=denominator_floor,
        )

        if inputs.ndim == 3:
            return np.abs(residual) / per_channel_data_range

        return np.abs(residual) / per_channel_data_range[None, :, None, None]

    raise RuntimeError(f"Unhandled error map kind: {kind!r}.")


def _resolve_global_data_range(
    *,
    inputs: np.ndarray,
    data_range: float | None,
    denominator_floor: float,
) -> float:
    """Resolve the intensity range used for normalized error maps."""

    if data_range is not None:
        if data_range <= 0:
            raise ValueError(f"data_range must be > 0, got {data_range}.")
        return float(data_range)

    input_min = float(np.min(inputs))
    input_max = float(np.max(inputs))
    inferred_data_range = input_max - input_min

    if inferred_data_range <= denominator_floor:
        warnings.warn(
            "The inferred global input data range was "
            f"{inferred_data_range}, so denominator_floor="
            f"{denominator_floor} was used for normalized error maps.",
            RuntimeWarning,
            stacklevel=2,
        )

    return max(inferred_data_range, denominator_floor)


def _resolve_per_channel_data_range(
    *,
    inputs: np.ndarray,
    denominator_floor: float,
) -> np.ndarray | float:
    """Resolve per-channel intensity ranges for normalized error maps."""

    if inputs.ndim == 3:
        input_min = float(np.min(inputs))
        input_max = float(np.max(inputs))
        inferred_data_range = input_max - input_min

        if inferred_data_range <= denominator_floor:
            warnings.warn(
                "The inferred input data range was "
                f"{inferred_data_range}, so denominator_floor="
                f"{denominator_floor} was used for per-channel normalized "
                "error maps.",
                RuntimeWarning,
                stacklevel=2,
            )

        return max(inferred_data_range, denominator_floor)

    channel_mins = np.min(inputs, axis=(0, 2, 3))
    channel_maxs = np.max(inputs, axis=(0, 2, 3))
    inferred_data_ranges = channel_maxs - channel_mins

    floored_channel_indices = np.flatnonzero(
        inferred_data_ranges <= denominator_floor
    )

    if floored_channel_indices.size:
        preview = floored_channel_indices[:10].tolist()

        warnings.warn(
            "Per-channel normalized error maps used denominator_floor="
            f"{denominator_floor} for {floored_channel_indices.size} channel(s). "
            f"First affected channel indices: {preview}.",
            RuntimeWarning,
            stacklevel=2,
        )

    return np.maximum(inferred_data_ranges, denominator_floor).astype(
        np.float32,
        copy=False,
    )


def _resolve_n_examples(
    *,
    n_examples: int | None,
    n_available: int,
) -> int | None:
    """Resolve how many reconstruction examples to keep."""

    if n_examples is None:
        return None

    if n_examples < 1:
        raise ValueError(f"n_examples must be >= 1, got {n_examples}.")

    return min(n_examples, n_available)