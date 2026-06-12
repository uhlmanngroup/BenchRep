from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from benchrep.evaluation.reconstructions.data import (
    ReconstructionEvaluationInput,
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
    kind: str = "absolute",
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
        Absolute residual divided by one global input intensity range
        ``max(input) - min(input)``.
    normalized_absolute_per_channel
        Absolute residual divided by a separate intensity range for each
        channel, computed across all provided samples for that channel.

    Error maps are returned as arrays. Disk export to TIFF/PNG belongs to the
    records/export layer.
    """

    inputs, reconstructions = validate_reconstruction_arrays(
        inputs=reconstruction_input.inputs,
        reconstructions=reconstruction_input.reconstructions,
    )

    resolved_n_examples = _resolve_n_examples(
        configured_n_examples=n_examples,
        fallback_n_examples=reconstruction_input.n_examples,
        n_available=inputs.shape[0],
    )

    if resolved_n_examples is not None:
        inputs = inputs[:resolved_n_examples]
        reconstructions = reconstructions[:resolved_n_examples]

    error_maps = _compute_error_map_array(
        inputs=inputs,
        reconstructions=reconstructions,
        kind=kind,
        denominator_floor=denominator_floor,
        data_range=data_range,
    )

    return {
        "kind": kind,
        "error_maps": error_maps,
        "shape": tuple(error_maps.shape),
        "n_examples": error_maps.shape[0],
        "params": {
            "denominator_floor": denominator_floor,
            "data_range": data_range,
        },
    }


def _compute_error_map_array(
    *,
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    kind: str,
    denominator_floor: float,
    data_range: float | None,
) -> np.ndarray:
    """Compute one kind of reconstruction error map."""

    if kind not in SUPPORTED_ERROR_MAP_KINDS:
        raise ValueError(
            f"Unsupported error map kind: {kind!r}. "
            f"Supported kinds: {sorted(SUPPORTED_ERROR_MAP_KINDS)}."
        )

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
        if data_range is not None:
            raise ValueError(
                "data_range is not supported for "
                "'normalized_absolute_per_channel'. Per-channel ranges are "
                "inferred automatically from the inputs."
            )

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

    inferred_data_range = float(np.max(inputs) - np.min(inputs))

    return max(inferred_data_range, denominator_floor)


def _resolve_per_channel_data_range(
    *,
    inputs: np.ndarray,
    denominator_floor: float,
) -> np.ndarray | float:
    """Resolve per-channel intensity ranges for normalized error maps."""

    if inputs.ndim == 3:
        inferred_data_range = float(np.max(inputs) - np.min(inputs))
        return max(inferred_data_range, denominator_floor)

    channel_mins = np.min(inputs, axis=(0, 2, 3))
    channel_maxs = np.max(inputs, axis=(0, 2, 3))
    inferred_data_ranges = channel_maxs - channel_mins

    return np.maximum(inferred_data_ranges, denominator_floor).astype(
        np.float32,
        copy=False,
    )


def _resolve_n_examples(
    *,
    configured_n_examples: int | None,
    fallback_n_examples: int | None,
    n_available: int,
) -> int | None:
    """Resolve how many reconstruction examples to keep."""

    n_examples = (
        configured_n_examples
        if configured_n_examples is not None
        else fallback_n_examples
    )

    if n_examples is None:
        return None

    if n_examples < 1:
        raise ValueError(f"n_examples must be >= 1, got {n_examples}.")

    return min(n_examples, n_available)