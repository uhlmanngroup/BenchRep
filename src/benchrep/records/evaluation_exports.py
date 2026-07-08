"""Evaluation output exporters.

This module will contain disk-writing utilities for evaluation artifacts,
including evaluated AnnData files, metrics JSON/YAML, plots, reconstruction
examples, and error maps.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING
import json
import math

import anndata as ad
import numpy as np
import re
import tifffile

from benchrep.evaluation.reconstructions.data import ReconstructionEvaluationInput
from benchrep.evaluation.embeddings.plotting import plot_2d_projection
from benchrep.evaluation.utils import (
    to_python_scalar,
    ensure_reconstruction_channel_axis,
    validate_reconstruction_arrays,
)


if TYPE_CHECKING:
    from benchrep.assembly.resolvers.evaluation_config_resolver import EvaluationStepSpec


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


def export_reduction_plots(
    *,
    output_dir: str | Path,
    adata: ad.AnnData,
    step_spec: "EvaluationStepSpec",
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Export static 2D reduction plots for enabled reductions."""

    if not step_spec.plots_enabled:
        return {}

    output_dir = Path(output_dir)

    bases: list[str] = []
    if step_spec.pca_enabled:
        bases.append(step_spec.pca_params.get("key_added", "X_pca"))
    if step_spec.umap_enabled:
        bases.append(step_spec.umap_params.get("key_added", "X_umap"))
    if step_spec.tsne_enabled:
        bases.append(step_spec.tsne_params.get("key_added", "X_tsne"))

    color_by = step_spec.plot_params.get("color_by") or []
    dpi = step_spec.plot_params.get("dpi", 300)
    formats = step_spec.plot_params.get("formats", ["png"])
    formats = [fmt.strip().lower() for fmt in formats if isinstance(fmt, str) and fmt.strip()]
    formats = list(dict.fromkeys(formats))

    written_paths: dict[str, list[Path]] = {}

    for basis in bases:
        if not isinstance(basis, str) or basis not in adata.obsm:
            continue

        basis_token = _sanitize_filename_token(basis)

        # Uncolored
        uncolored_key = f"{basis}:uncolored"
        written_paths[uncolored_key] = []

        for fmt in formats:
            output_path = output_dir / f"{basis_token}.{fmt}"
            plot_2d_projection(
                adata,
                basis=basis,
                color_by=None,
                output_path=output_path,
                dpi=dpi,
                overwrite=overwrite,
            )
            written_paths[uncolored_key].append(output_path)

        # Colored
        for color in color_by:
            if not isinstance(color, str) or color not in adata.obs.columns:
                continue

            color_token = _sanitize_filename_token(color)
            colored_key = f"{basis}:colored_by:{color}"
            written_paths[colored_key] = []

            for fmt in formats:
                output_path = output_dir / (
                    f"{basis_token}_colored_by_{color_token}.{fmt}"
                )
                plot_2d_projection(
                    adata,
                    basis=basis,
                    color_by=color,
                    output_path=output_path,
                    dpi=dpi,
                    overwrite=overwrite,
                )
                written_paths[colored_key].append(output_path)

    return written_paths


def export_reconstruction_tiffs(
    *,
    output_dir: str | Path,
    reconstruction_input: ReconstructionEvaluationInput,
    reconstruction_outputs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Export reconstruction inputs, predictions, and error maps as TIFF files."""

    output_dir = Path(output_dir)

    input_array, reconstruction_array = validate_reconstruction_arrays(
        inputs=reconstruction_input.inputs,
        reconstructions=reconstruction_input.reconstructions,
    )

    available_examples = int(input_array.shape[0])

    if reconstruction_input.n_examples is None:
        n_examples = available_examples
    else:
        n_examples = min(reconstruction_input.n_examples, available_examples)

    input_array = input_array[:n_examples]
    reconstruction_array = reconstruction_array[:n_examples]

    filename_stems = _resolve_reconstruction_filenames(
        obs=reconstruction_input.obs,
        n_examples=n_examples,
    )

    written_paths: dict[str, list[Path]] = {
        "inputs": _write_reconstruction_tiffs(
            arrays=input_array,
            output_dir=output_dir / "inputs",
            filename_prefix="input",
            filename_stems=filename_stems,
            overwrite=overwrite,
        ),
        "predictions": _write_reconstruction_tiffs(
            arrays=reconstruction_array,
            output_dir=output_dir / "predictions",
            filename_prefix="prediction",
            filename_stems=filename_stems,
            overwrite=overwrite,
        ),
    }

    if reconstruction_outputs is None:
        return written_paths

    error_map_output = reconstruction_outputs.get("error_maps")
    if error_map_output is None:
        return written_paths

    if not isinstance(error_map_output, Mapping):
        raise TypeError(
            "Expected reconstruction_outputs['error_maps'] to be a mapping, "
            f"got {type(error_map_output).__name__}."
        )

    error_maps = error_map_output.get("error_maps")
    if error_maps is None:
        return written_paths

    error_map_array = ensure_reconstruction_channel_axis(np.asarray(error_maps))
    input_array_with_channels = ensure_reconstruction_channel_axis(input_array)

    if error_map_array.shape[0] < n_examples:
        raise ValueError(
            "Expected at least as many error maps as selected reconstruction "
            f"examples. Got {error_map_array.shape[0]} error maps and "
            f"{n_examples} selected examples."
        )

    error_map_array = error_map_array[:n_examples]

    if error_map_array.shape != input_array_with_channels.shape:
        raise ValueError(
            "Expected selected error maps to have the same shape as selected "
            "reconstruction inputs after adding any missing channel axis. "
            f"Got error maps shape {error_map_array.shape} and input shape "
            f"{input_array_with_channels.shape}."
        )

    written_paths["error_maps"] = _write_reconstruction_tiffs(
        arrays=error_map_array,
        output_dir=output_dir / "error_maps",
        filename_prefix="error_map",
        filename_stems=filename_stems,
        overwrite=overwrite,
    )

    return written_paths


def _sanitize_filename_token(value: Any, *, fallback: str = "unnamed") -> str:
    """Return a filesystem-safe token for generated artifact filenames."""

    token = str(value).strip()
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
    token = token.strip("._-")

    return token or fallback


def _resolve_reconstruction_filenames(
    *,
    obs: Any,
    n_examples: int,
) -> list[str]:
    """Resolve safe per-example filename stems from sample IDs or indices."""

    sample_ids = None

    if isinstance(obs, Mapping):
        sample_ids = obs.get("sample_id")
    elif hasattr(obs, "columns") and "sample_id" in obs.columns:
        sample_ids = obs["sample_id"]

    filename_stems: list[str] = []

    for example_index in range(n_examples):
        fallback = f"example_{example_index:04d}"
        sample_id = None

        if sample_ids is not None:
            try:
                if hasattr(sample_ids, "iloc"):
                    sample_id = sample_ids.iloc[example_index]
                else:
                    sample_id = sample_ids[example_index]
            except (IndexError, KeyError, TypeError):
                sample_id = None

        if sample_id is None:
            filename_stem = fallback
        else:
            filename_stem = f"{fallback}_{sample_id}"

        filename_stem = _sanitize_filename_token(filename_stem, fallback=fallback)

        filename_stems.append(filename_stem)

    return filename_stems


def _write_reconstruction_tiffs(
    *,
    arrays: np.ndarray,
    output_dir: str | Path,
    filename_prefix: str,
    filename_stems: list[str],
    overwrite: bool,
) -> list[Path]:
    """Write one reconstruction-like array collection as per-example TIFFs."""

    arrays = ensure_reconstruction_channel_axis(np.asarray(arrays))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if arrays.shape[0] != len(filename_stems):
        raise ValueError(
            "Expected the number of filename stems to match the number of "
            f"examples. Got {len(filename_stems)} filename stems and "
            f"{arrays.shape[0]} examples."
        )

    written_paths: list[Path] = []

    for example_index, filename_stem in enumerate(filename_stems):
        output_path = output_dir / f"{filename_prefix}_{filename_stem}.tif"

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"TIFF output already exists: {output_path}")

        image = arrays[example_index]

        if image.shape[0] == 1:
            output_array = image[0]
            metadata = {"axes": "YX"}
        else:
            output_array = image
            metadata = {"axes": "CYX"}

        tifffile.imwrite(
            output_path,
            output_array.astype(np.float32, copy=False),
            metadata=metadata,
        )

        written_paths.append(output_path)

    return written_paths
