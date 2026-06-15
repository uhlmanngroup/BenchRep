"""Evaluation output exporters.

This module will contain disk-writing utilities for evaluation artifacts,
including evaluated AnnData files, metrics JSON/YAML, plots, reconstruction
examples, and error maps.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json
import math

import anndata as ad
import numpy as np

from benchrep.evaluation.utils import to_python_scalar


def save_evaluation_metrics_json(
    *,
    output_dir: str | Path,
    adata: ad.AnnData,
    reconstruction_outputs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Save all available evaluation metrics to one JSON file."""

    output_path = Path(output_dir) / "metrics.json"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Metrics JSON already exists: {output_path}")

    metrics = _collect_evaluation_metrics(
        adata=adata,
        reconstruction_outputs=reconstruction_outputs,
    )
    metrics = _to_json_safe(metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    return output_path


def _collect_evaluation_metrics(
    *,
    adata: ad.AnnData,
    reconstruction_outputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect evaluation metrics from AnnData and reconstruction outputs."""

    benchrep = adata.uns.get("benchrep", {})
    if not isinstance(benchrep, Mapping):
        raise TypeError(
            "Expected adata.uns['benchrep'] to be a mapping, "
            f"got {type(benchrep).__name__}."
        )

    adata_metrics = benchrep.get("metrics", {})
    if adata_metrics is None:
        adata_metrics = {}

    if not isinstance(adata_metrics, Mapping):
        raise TypeError(
            "Expected adata.uns['benchrep']['metrics'] to be a mapping, "
            f"got {type(adata_metrics).__name__}."
        )

    metrics = dict(adata_metrics)

    if reconstruction_outputs is None:
        return metrics

    reconstruction_metrics = reconstruction_outputs.get("reconstruction_metrics")
    if reconstruction_metrics is None:
        return metrics

    if not isinstance(reconstruction_metrics, Mapping):
        raise TypeError(
            "Expected reconstruction_outputs['reconstruction_metrics'] to be a "
            f"mapping, got {type(reconstruction_metrics).__name__}."
        )

    if "reconstruction" in metrics:
        raise ValueError(
            "Found reconstruction metrics in both adata.uns['benchrep']['metrics'] "
            "and reconstruction_outputs. Refusing to overwrite one with the other."
        )

    metrics["reconstruction"] = reconstruction_metrics

    return metrics


def _to_json_safe(value: Any) -> Any:
    """Convert nested metric structures to strict JSON-safe values."""

    value = to_python_scalar(value)

    if isinstance(value, Mapping):
        return {str(key): _to_json_safe(item) for key, item in value.items()}

    if isinstance(value, list | tuple):
        return [_to_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_json_safe(value.item())

        return {
            "array_shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if value is None or isinstance(value, str | int | float | bool):
        return value

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable "
        "as an evaluation metric value."
    )