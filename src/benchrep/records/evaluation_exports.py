from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING
import json
import math

import anndata as ad
import numpy as np
import re
import tifffile

from benchrep.evaluation.reconstructions.data import ReconstructionEvaluationInput
from benchrep.evaluation.reconstructions.error_maps import compute_error_maps
from benchrep.evaluation.reconstructions.plotting import (
    plot_reconstruction_grid_page,
)
from benchrep.evaluation.embeddings.plotting import (
    DEFAULT_ACCENT_COLOR,
    plot_2d_projection,
    plot_pca_variance,
    plot_cluster_sizes,
)
from benchrep.evaluation.utils import (
    to_python_scalar,
    ensure_reconstruction_channel_axis,
    validate_reconstruction_arrays,
)
from benchrep.records.logs import get_run_logger

if TYPE_CHECKING:
    from benchrep.assembly.resolvers.evaluation_config_resolver import EvaluationStepSpec


RECONSTRUCTION_GRID_PAGE_SIZE = 12
RECONSTRUCTION_GRID_FILE_WARNING_THRESHOLD = 24
RECONSTRUCTION_GRID_LABEL_VALUE_MAX_LENGTH = 32


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

    reductions_dir = output_dir / "reductions"
    uncolored_dir = reductions_dir / "uncolored"
    colored_by_dir = reductions_dir / "colored_by"
    diagnostics_dir = output_dir / "diagnostics"
    pca_diagnostics_dir = diagnostics_dir / "pca"

    bases: list[str] = []
    if step_spec.pca_enabled:
        bases.append(step_spec.pca_params.get("key_added", "X_pca"))
    if step_spec.umap_enabled:
        bases.append(step_spec.umap_params.get("key_added", "X_umap"))
    if step_spec.tsne_enabled:
        bases.append(step_spec.tsne_params.get("key_added", "X_tsne"))

    accent_color = step_spec.plot_params.get("accent_color") or DEFAULT_ACCENT_COLOR
    color_by = step_spec.plot_params.get("color_by") or []
    dpi, formats = _resolve_plot_file_options(step_spec)

    written_paths: dict[str, list[Path]] = {}

    if step_spec.pca_enabled:
        pca_key = step_spec.pca_params.get("key_added", "X_pca")

        if isinstance(pca_key, str):
            pca_metadata = _get_reduction_metadata(adata, pca_key)

            if pca_metadata is not None:
                explained_variance_ratio = pca_metadata.get("explained_variance_ratio")

                if explained_variance_ratio is not None:
                    pca_token = _sanitize_filename_token(pca_key)

                    scree_key = f"{pca_key}:scree"
                    written_paths[scree_key] = []

                    cumulative_key = f"{pca_key}:cumulative_variance"
                    written_paths[cumulative_key] = []

                    for fmt in formats:
                        scree_path = pca_diagnostics_dir / f"{pca_token}_scree.{fmt}"
                        plot_pca_variance(
                            explained_variance_ratio=explained_variance_ratio,
                            output_path=scree_path,
                            kind="scree",
                            title=f"{pca_key} explained variance",
                            dpi=dpi,
                            accent_color=accent_color,
                            overwrite=overwrite,
                        )
                        written_paths[scree_key].append(scree_path)

                        cumulative_path = pca_diagnostics_dir / f"{pca_token}_cumulative_variance.{fmt}"
                        plot_pca_variance(
                            explained_variance_ratio=explained_variance_ratio,
                            output_path=cumulative_path,
                            kind="cumulative",
                            title=f"{pca_key} cumulative explained variance",
                            dpi=dpi,
                            accent_color=accent_color,
                            overwrite=overwrite,
                        )
                        written_paths[cumulative_key].append(cumulative_path)

    for basis in bases:
        if not isinstance(basis, str) or basis not in adata.obsm:
            continue

        basis_token = _sanitize_filename_token(basis)

        # Uncolored
        uncolored_key = f"{basis}:uncolored"
        written_paths[uncolored_key] = []

        for fmt in formats:
            output_path = uncolored_dir / f"{basis_token}.{fmt}"
            plot_2d_projection(
                adata,
                basis=basis,
                accent_color=accent_color,
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

            colored_by_subdir = colored_by_dir / color_token

            for fmt in formats:
                output_path = colored_by_subdir / f"{basis_token}.{fmt}"
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


def export_cluster_size_plots(
    *,
    output_dir: str | Path,
    adata: ad.AnnData,
    step_spec: "EvaluationStepSpec",
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Export cluster-size diagnostic plots for clustering outputs."""

    if not step_spec.plots_enabled:
        return {}

    output_dir = Path(output_dir)

    diagnostics_dir = output_dir / "diagnostics"
    clustering_diagnostics_dir = diagnostics_dir / "clustering"

    accent_color = step_spec.plot_params.get("accent_color") or DEFAULT_ACCENT_COLOR
    dpi, formats = _resolve_plot_file_options(step_spec)

    clustering_uns = adata.uns.get("benchrep", {}).get("clustering", {})
    if not isinstance(clustering_uns, Mapping):
        return {}

    written_paths: dict[str, list[Path]] = {}

    for key_added, metadata in clustering_uns.items():
        if not isinstance(key_added, str):
            continue

        if not isinstance(metadata, Mapping):
            continue

        if key_added not in adata.obs.columns:
            continue

        key_token = _sanitize_filename_token(key_added)
        cluster_sizes_key = f"{key_added}:cluster_sizes"
        written_paths[cluster_sizes_key] = []

        cluster_sizes_dir = clustering_diagnostics_dir / key_token

        for fmt in formats:
            output_path = cluster_sizes_dir / f"cluster_sizes.{fmt}"

            plot_cluster_sizes(
                adata.obs[key_added],
                output_path=output_path,
                title=f"{key_added} cluster sizes",
                dpi=dpi,
                accent_color=accent_color,
                overwrite=overwrite,
            )
            written_paths[cluster_sizes_key].append(output_path)

    return written_paths


def export_reconstruction_tiffs(
    *,
    output_dir: str | Path,
    reconstruction_input: ReconstructionEvaluationInput,
    reconstruction_outputs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
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

    written_paths: dict[str, Any] = {
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

    error_map_outputs = reconstruction_outputs.get("error_maps")
    if error_map_outputs is None:
        return written_paths

    if not isinstance(error_map_outputs, Mapping):
        raise TypeError(
            "Expected reconstruction_outputs['error_maps'] to be a mapping, "
            f"got {type(error_map_outputs).__name__}."
        )

    input_array_with_channels = ensure_reconstruction_channel_axis(input_array)
    written_error_map_paths: dict[str, list[Path]] = {}

    for error_kind, error_map_output in error_map_outputs.items():
        error_kind = str(error_kind)

        if not isinstance(error_map_output, Mapping):
            raise TypeError(
                "Expected each reconstruction error-map output to be a mapping, "
                f"got {type(error_map_output).__name__} for kind {error_kind!r}."
            )

        error_maps = error_map_output.get("error_maps")
        if error_maps is None:
            continue

        error_map_array = ensure_reconstruction_channel_axis(np.asarray(error_maps))

        if error_map_array.shape[0] < n_examples:
            raise ValueError(
                "Expected at least as many error maps as selected reconstruction "
                f"examples for kind {error_kind!r}. Got {error_map_array.shape[0]} "
                f"error maps and {n_examples} selected examples."
            )

        error_map_array = error_map_array[:n_examples]

        if error_map_array.shape != input_array_with_channels.shape:
            raise ValueError(
                "Expected selected error maps to have the same shape as selected "
                "reconstruction inputs after adding any missing channel axis. "
                f"For kind {error_kind!r}, got error maps shape "
                f"{error_map_array.shape} and input shape "
                f"{input_array_with_channels.shape}."
            )

        written_error_map_paths[error_kind] = _write_reconstruction_tiffs(
            arrays=error_map_array,
            output_dir=output_dir / "error_maps" / error_kind,
            filename_prefix=f"error_map_{error_kind}",
            filename_stems=filename_stems,
            overwrite=overwrite,
        )

    if written_error_map_paths:
        written_paths["error_maps"] = written_error_map_paths

    return written_paths


def export_reconstruction_grids(
    *,
    output_dir: str | Path,
    reconstruction_input: ReconstructionEvaluationInput,
    step_spec: "EvaluationStepSpec",
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Export paginated reconstruction-summary grids."""

    if not step_spec.plots_enabled:
        return {}

    inputs, reconstructions = validate_reconstruction_arrays(
        inputs=reconstruction_input.inputs,
        reconstructions=reconstruction_input.reconstructions,
    )

    inputs = ensure_reconstruction_channel_axis(inputs)
    reconstructions = ensure_reconstruction_channel_axis(reconstructions)

    n_available, n_channels = inputs.shape[:2]

    grid_params = step_spec.plot_params.get("reconstruction_grid", {})

    if not isinstance(grid_params, Mapping):
        raise TypeError(
            "plots.params.reconstruction_grid must resolve to a mapping."
        )

    stratify_by = grid_params.get("stratify_by")
    channel_selection = grid_params.get("channel_selection")

    pages = _resolve_reconstruction_grid_pages(
        obs=reconstruction_input.obs,
        n_available=n_available,
        stratify_by=stratify_by,
        random_state=grid_params.get("random_state", 137),
    )

    selected_channels = _resolve_reconstruction_grid_channels(
        n_channels=n_channels,
        channel_selection=channel_selection,
    )

    dpi, formats = _resolve_plot_file_options(step_spec)
    output_dir = Path(output_dir)

    run_log = get_run_logger()

    if channel_selection is None and n_channels > 1:
        run_log.warning(
            "Reconstruction-grid channel_selection is null for %d-channel data; "
            "only channel 0 will be shown.",
            n_channels,
        )

    n_grid_files = len(pages) * len(selected_channels) * len(formats)

    if n_grid_files > RECONSTRUCTION_GRID_FILE_WARNING_THRESHOLD:
        run_log.warning(
            "Reconstruction grid export will write %d figure files "
            "(%d page(s) × %d channel(s) × %d format(s)).",
            n_grid_files,
            len(pages),
            len(selected_channels),
            len(formats),
        )

    written_paths: dict[str, list[Path]] = {
        f"channel_{channel:04d}": []
        for channel in selected_channels
    }

    all_selected_indices = np.concatenate(pages)

    selected_inputs = inputs[all_selected_indices]
    selected_reconstructions = reconstructions[all_selected_indices]

    selected_error_maps: dict[str, np.ndarray] = {}

    if step_spec.error_maps_enabled:
        computed_error_maps = compute_error_maps(
            ReconstructionEvaluationInput(
                inputs=selected_inputs,
                reconstructions=selected_reconstructions,
                obs=None,
            ),
            n_examples=None,
            **dict(step_spec.error_map_params),
        )

        for error_kind, error_output in computed_error_maps.items():
            selected_error_maps[str(error_kind)] = (
                ensure_reconstruction_channel_axis(
                    np.asarray(error_output["error_maps"])
                )
            )

    page_start = 0

    for page_number, selected_indices in enumerate(pages, start=1):
        page_stop = page_start + len(selected_indices)
        page_slice = slice(page_start, page_stop)

        page_inputs = selected_inputs[page_slice]
        page_reconstructions = selected_reconstructions[page_slice]

        row_labels = _resolve_reconstruction_grid_row_labels(
            obs=reconstruction_input.obs,
            selected_indices=selected_indices,
            stratify_by=stratify_by,
        )

        page_error_maps = {
            error_kind: error_maps[page_slice]
            for error_kind, error_maps in selected_error_maps.items()
        }

        for channel in selected_channels:
            channel_key = f"channel_{channel:04d}"

            channel_error_maps = {
                error_kind: error_maps[:, channel]
                for error_kind, error_maps in page_error_maps.items()
            }

            for fmt in formats:
                output_path = (
                    output_dir
                    / f"reconstruction_grid_channel_{channel:04d}"
                      f"_page_{page_number:03d}.{fmt}"
                )

                plot_reconstruction_grid_page(
                    inputs=page_inputs[:, channel],
                    reconstructions=page_reconstructions[:, channel],
                    error_maps=channel_error_maps,
                    row_labels=row_labels,
                    output_path=output_path,
                    title=(
                        f"Reconstruction summary — channel {channel} — "
                        f"page {page_number}/{len(pages)}"
                    ),
                    dpi=dpi,
                    overwrite=overwrite,
                )

                written_paths[channel_key].append(output_path)

        page_start = page_stop

    return written_paths


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


def _get_reduction_metadata(
    adata: ad.AnnData,
    reduction_key: str,
) -> Mapping[str, Any] | None:
    """Return BenchRep reduction metadata for a stored reduction key."""

    benchrep_uns = adata.uns.get("benchrep")
    if not isinstance(benchrep_uns, Mapping):
        return None

    reductions = benchrep_uns.get("reductions")
    if not isinstance(reductions, Mapping):
        return None

    metadata = reductions.get(reduction_key)
    if not isinstance(metadata, Mapping):
        return None

    return metadata


def _resolve_plot_file_options(
    step_spec: "EvaluationStepSpec",
) -> tuple[int, list[str]]:
    """Resolve shared static plot export options from the evaluation step spec."""
    plot_params = step_spec.plot_params

    dpi = plot_params.get("dpi", 300)

    formats = plot_params.get("formats", ["png"])
    formats = [
        fmt.strip().lower()
        for fmt in formats
        if isinstance(fmt, str) and fmt.strip()
    ]
    formats = list(dict.fromkeys(formats)) or ["png"]

    return dpi, formats


def _resolve_reconstruction_grid_pages(
    *,
    obs: Any,
    n_available: int,
    stratify_by: str | None,
    random_state: int,
    page_size: int = RECONSTRUCTION_GRID_PAGE_SIZE,
) -> list[np.ndarray]:
    """Select and paginate reconstruction examples for summary grids."""
    if n_available < 1:
        raise ValueError("Cannot create reconstruction grids without examples.")

    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}.")

    generator = np.random.default_rng(random_state)

    if stratify_by is None:
        n_selected = min(page_size, n_available)
        selected_indices = generator.choice(
            n_available,
            size=n_selected,
            replace=False,
        )

        return [
            np.asarray(selected_indices, dtype=np.int64)
        ]

    stratify_values = _get_reconstruction_obs_field(
        obs=obs,
        key=stratify_by,
    )

    if stratify_values is None:
        raise KeyError(
            f"Reconstruction-grid stratification requested key "
            f"{stratify_by!r}, but it is not available in reconstruction "
            f"observations. Available keys: "
            f"{_available_reconstruction_obs_keys(obs)}."
        )

    if len(stratify_values) != n_available:
        raise ValueError(
            f"Reconstruction-grid stratification field {stratify_by!r} "
            f"contains {len(stratify_values)} values, but exactly "
            f"{n_available} are required—one per reconstruction example."
        )

    strata: dict[Any, list[int]] = {}

    for example_index, value in enumerate(stratify_values):
        value = _normalize_reconstruction_stratum(value)

        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(
                f"Reconstruction-grid stratification field {stratify_by!r} "
                "must contain scalar, hashable values."
            ) from exc

        strata.setdefault(value, []).append(example_index)

    selected_indices = [
        group[int(generator.integers(len(group)))]
        for group in strata.values()
    ]
    generator.shuffle(selected_indices)

    return [
        np.asarray(
            selected_indices[start:start + page_size],
            dtype=np.int64,
        )
        for start in range(0, len(selected_indices), page_size)
    ]


def _get_reconstruction_obs_field(
    *,
    obs: Any,
    key: str,
) -> Sequence[Any] | None:
    """Return one reconstruction observation field when available."""
    values = None

    if isinstance(obs, Mapping):
        values = obs.get(key)
    elif hasattr(obs, "columns") and key in obs.columns:
        values = obs[key]

    if values is None:
        return None

    if hasattr(values, "tolist"):
        values = values.tolist()

    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(
            f"Reconstruction observation field {key!r} must contain a "
            f"sequence of per-example values, got {type(values).__name__}."
        )

    return values


def _available_reconstruction_obs_keys(obs: Any) -> list[str]:
    """Return available reconstruction observation field names."""
    if isinstance(obs, Mapping):
        return sorted(str(key) for key in obs)

    if hasattr(obs, "columns"):
        return sorted(str(key) for key in obs.columns)

    return []


def _normalize_reconstruction_stratum(value: Any) -> Any:
    """Normalize scalar stratum values, including missing values."""
    value = to_python_scalar(value)

    if value is None:
        return "<missing>"

    try:
        if bool(np.isnan(value)):
            return "<missing>"
    except (TypeError, ValueError):
        pass

    return value


def _resolve_reconstruction_grid_channels(
    *,
    n_channels: int,
    channel_selection: str | int | Sequence[int] | None,
) -> list[int]:
    """Resolve and validate channels selected for reconstruction grids."""
    if n_channels < 1:
        raise ValueError(
            f"Reconstruction arrays must contain at least one channel, got {n_channels}."
        )

    if channel_selection is None:
        selected_channels = [0]

    elif channel_selection == "all":
        selected_channels = list(range(n_channels))

    elif isinstance(channel_selection, int) and not isinstance(
        channel_selection,
        bool,
    ):
        selected_channels = [channel_selection]

    elif isinstance(channel_selection, Sequence) and not isinstance(
        channel_selection,
        str | bytes,
    ):
        selected_channels = list(dict.fromkeys(channel_selection))

        if not selected_channels:
            raise ValueError(
                "Reconstruction-grid channel selection cannot be empty."
            )

        if any(
            not isinstance(channel, int) or isinstance(channel, bool)
            for channel in selected_channels
        ):
            raise TypeError(
                "Reconstruction-grid channel selection must contain only integers."
            )

    else:
        raise TypeError(
            "Reconstruction-grid channel selection must be null, 'all', "
            f"an integer, or a sequence of integers; got "
            f"{type(channel_selection).__name__}."
        )

    invalid_channels = [
        channel
        for channel in selected_channels
        if channel < 0 or channel >= n_channels
    ]

    if invalid_channels:
        raise IndexError(
            f"Reconstruction-grid channel indices {invalid_channels} are out "
            f"of bounds for arrays containing {n_channels} channel(s)."
        )

    return selected_channels


def _resolve_reconstruction_grid_row_labels(
    *,
    obs: Any,
    selected_indices: Sequence[int],
    stratify_by: str | None,
) -> list[str]:
    """Resolve grid row labels from sample IDs, source indices, or row indices."""
    sample_ids = _get_reconstruction_obs_field(
        obs=obs,
        key="sample_id",
    )
    source_indices = _get_reconstruction_obs_field(
        obs=obs,
        key="source_index",
    )

    stratify_values = (
        _get_reconstruction_obs_field(
            obs=obs,
            key=stratify_by,
        )
        if stratify_by is not None
        else None
    )

    row_labels: list[str] = []

    for selected_index in selected_indices:
        label_parts: list[str] = []

        if stratify_values is not None and stratify_by is not None:
            stratum_label = _resolve_reconstruction_obs_label(
                values=stratify_values,
                index=selected_index,
                prefix=stratify_by,
            )

            if stratum_label is not None:
                label_parts.append(stratum_label)

        identifier = None

        if sample_ids is not None:
            identifier = _resolve_reconstruction_obs_label(
                values=sample_ids,
                index=selected_index,
                prefix="sample_id",
            )

        if identifier is None and source_indices is not None:
            identifier = _resolve_reconstruction_obs_label(
                values=source_indices,
                index=selected_index,
                prefix="source_index",
            )

        if identifier is None:
            identifier = f"index={selected_index}"

        label_parts.append(identifier)
        row_labels.append(" | ".join(label_parts))

    return row_labels


def _resolve_reconstruction_obs_label(
    *,
    values: Sequence[Any],
    index: int,
    prefix: str,
) -> str | None:
    """Resolve one non-missing observation value as a grid row label."""
    try:
        value = to_python_scalar(values[index])
    except (IndexError, KeyError, TypeError):
        return None

    if value is None:
        return None

    try:
        if bool(np.isnan(value)):
            return None
    except (TypeError, ValueError):
        pass

    value_text = str(value)

    if len(value_text) > RECONSTRUCTION_GRID_LABEL_VALUE_MAX_LENGTH:
        value_text = (
                value_text[:RECONSTRUCTION_GRID_LABEL_VALUE_MAX_LENGTH - 3]
                + "..."
        )

    return f"{prefix}={value_text}"
