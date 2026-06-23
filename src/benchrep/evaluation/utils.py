from collections.abc import Mapping
from typing import Any, Literal
import inspect

import anndata as ad
import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor
PredictabilityTask = Literal["classification", "regression"]


def validate_adata_x(adata: ad.AnnData) -> None:
    """Validate that adata.X is present and 2D."""

    if adata.X is None:
        raise ValueError("adata.X is required for dimensionality reduction.")

    if adata.X.ndim != 2:
        raise ValueError(f"adata.X must be 2D, got shape {adata.X.shape}.")


def validate_obs_key(adata: ad.AnnData, key: str) -> None:
    """Validate that ``key`` exists in ``adata.obs``."""

    if key not in adata.obs.columns:
        raise KeyError(
            f"adata.obs does not contain {key!r}. "
            f"Available columns: {list(adata.obs.columns)}"
        )


def to_numpy(array: ArrayLike) -> np.ndarray:
    """Convert a tensor/array to a CPU NumPy array without modifying shape."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()

    if isinstance(array, np.ndarray):
        return array

    raise TypeError(
        "Expected a NumPy array or torch.Tensor, "
        f"got {type(array).__name__}."
    )


def validate_metric_params(
    *,
    metric_name: str,
    metric_fn: Any,
    params: Mapping[str, Any],
    metric_kind: str = "metric",
) -> None:
    """Validate user-provided metric params against the callable signature."""

    try:
        signature = inspect.signature(metric_fn)
    except (TypeError, ValueError):
        return

    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_var_kwargs:
        return

    unsupported_params = [
        param_name for param_name in params if param_name not in parameters
    ]

    if unsupported_params:
        raise ValueError(
            f"Unsupported parameters for {metric_kind} {metric_name!r}: "
            f"{unsupported_params}. Accepted parameters are: {tuple(parameters)}."
        )


def validate_reconstruction_arrays(
    *,
    inputs: ArrayLike,
    reconstructions: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate reconstruction input/reconstruction array compatibility."""

    input_array = np.asarray(to_numpy(inputs))
    reconstruction_array = np.asarray(to_numpy(reconstructions))

    if input_array.shape != reconstruction_array.shape:
        raise ValueError(
            "inputs and reconstructions must have matching shapes, got "
            f"{input_array.shape} and {reconstruction_array.shape}."
        )

    if input_array.ndim not in {3, 4}:
        raise ValueError(
            "Expected arrays with shape (B, H, W) or (B, C, H, W), got "
            f"{input_array.shape}."
        )

    if not np.issubdtype(input_array.dtype, np.number):
        raise TypeError(f"inputs must be numeric, got dtype {input_array.dtype}.")

    if not np.issubdtype(reconstruction_array.dtype, np.number):
        raise TypeError(
            "reconstructions must be numeric, got dtype "
            f"{reconstruction_array.dtype}."
        )

    return (
        input_array.astype(np.float32, copy=False),
        reconstruction_array.astype(np.float32, copy=False),
    )


def ensure_reconstruction_channel_axis(array: np.ndarray) -> np.ndarray:
    """Return reconstruction array with explicit channel axis.

    Converts ``(B, H, W)`` to ``(B, 1, H, W)`` and leaves ``(B, C, H, W)``
    unchanged.
    """

    if array.ndim == 3:
        return array[:, None, :, :]

    if array.ndim == 4:
        return array

    raise ValueError(
        "Expected reconstruction array with shape (B, H, W) or (B, C, H, W), "
        f"got {array.shape}."
    )


def resolve_reconstruction_channel_names(
    *,
    metadata: Mapping[str, Any] | None,
    n_channels: int,
) -> list[str]:
    """Resolve reconstruction channel names from metadata or fallback names."""

    metadata = metadata or {}
    channel_names = metadata.get("channel_names")

    if channel_names is None:
        return [f"channel_{index}" for index in range(n_channels)]

    channel_names = [str(channel_name) for channel_name in channel_names]

    if len(channel_names) != n_channels:
        raise ValueError(
            f"Expected {n_channels} channel names, got {len(channel_names)}."
        )

    return channel_names


def to_python_scalar(value: Any) -> Any:
    """Convert scalar-like values to plain Python scalars when possible."""

    if hasattr(value, "item"):
        return value.item()

    return value
