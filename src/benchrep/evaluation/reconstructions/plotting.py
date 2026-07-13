from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np

from benchrep.evaluation.utils import prepare_output_path, validate_dpi


MAX_GRID_DISPLAY_SIDE = 512


def plot_reconstruction_grid_page(
    *,
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    error_maps: Mapping[str, np.ndarray],
    row_labels: Sequence[str],
    output_path: str | Path,
    title: str | None = None,
    dpi: int = 300,
    overwrite: bool = False,
) -> None:
    """Plot one reconstruction-grid page for one selected channel.

    Rows represent examples. Columns contain the input, reconstruction, and
    each requested error-map kind.
    """
    inputs = np.asarray(inputs)
    reconstructions = np.asarray(reconstructions)
    error_maps = {
        str(kind): np.asarray(values)
        for kind, values in error_maps.items()
    }

    _validate_grid_arrays(
        inputs=inputs,
        reconstructions=reconstructions,
        error_maps=error_maps,
        row_labels=row_labels,
    )
    validate_dpi(dpi)

    output_path = prepare_output_path(
        output_path,
        overwrite=overwrite,
    )

    n_examples = inputs.shape[0]
    error_kinds = list(error_maps)
    n_columns = 2 + len(error_kinds)

    fig, axes = plt.subplots(
        nrows=n_examples,
        ncols=n_columns,
        figsize=(2.2 * n_columns, 1.8 * n_examples),
        squeeze=False,
    )

    column_titles = [
        "Input",
        "Reconstruction",
        *[_format_error_kind(kind) for kind in error_kinds],
    ]

    for column_index, column_title in enumerate(column_titles):
        wrapped_title = textwrap.fill(column_title, width=16)
        axes[0, column_index].set_title(
            wrapped_title,
            fontsize=9,
        )

    error_limits = {
        error_kind: (
            _symmetric_finite_limits(error_values)
            if error_kind == "signed"
            else _shared_finite_limits(error_values)
        )
        for error_kind, error_values in error_maps.items()
    }

    for example_index in range(n_examples):
        input_image = _downsample_for_display(inputs[example_index])
        reconstruction_image = _downsample_for_display(
            reconstructions[example_index]
        )

        input_vmin, input_vmax = _shared_finite_limits(
            input_image,
            reconstruction_image,
        )

        axes[example_index, 0].imshow(
            np.ma.masked_invalid(input_image),
            cmap="gray",
            vmin=input_vmin,
            vmax=input_vmax,
        )
        axes[example_index, 1].imshow(
            np.ma.masked_invalid(reconstruction_image),
            cmap="gray",
            vmin=input_vmin,
            vmax=input_vmax,
        )

        for error_offset, error_kind in enumerate(error_kinds):
            axis = axes[example_index, error_offset + 2]
            error_image = _downsample_for_display(
                error_maps[error_kind][example_index]
            )

            vmin, vmax = error_limits[error_kind]
            cmap = "coolwarm" if error_kind == "signed" else "magma"

            axis.imshow(
                np.ma.masked_invalid(error_image),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

        axes[example_index, 0].set_ylabel(
            str(row_labels[example_index]),
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )

        for axis in axes[example_index]:
            axis.set_xticks([])
            axis.set_yticks([])

    if title is not None:
        fig.suptitle(title, y=1.0)

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.subplots_adjust(wspace=0.12, hspace=0.08)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def _validate_grid_arrays(
    *,
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    error_maps: Mapping[str, np.ndarray],
    row_labels: Sequence[str],
) -> None:
    expected_shape = inputs.shape

    if inputs.ndim != 3:
        raise ValueError(
            "Grid inputs must have shape (n_examples, height, width), "
            f"got {inputs.shape}."
        )

    if reconstructions.shape != expected_shape:
        raise ValueError(
            "Grid reconstructions must match the input shape. "
            f"Got inputs {expected_shape} and reconstructions "
            f"{reconstructions.shape}."
        )

    if expected_shape[0] == 0:
        raise ValueError("Cannot plot an empty reconstruction grid.")

    if len(row_labels) != expected_shape[0]:
        raise ValueError(
            "Expected one row label per grid example. "
            f"Got {len(row_labels)} labels for {expected_shape[0]} examples."
        )

    for error_kind, error_map in error_maps.items():
        if error_map.shape != expected_shape:
            raise ValueError(
                f"Error maps for kind {error_kind!r} must match the input "
                f"shape. Got {error_map.shape}, expected {expected_shape}."
            )


def _downsample_for_display(
    image: np.ndarray,
    *,
    max_side: int = MAX_GRID_DISPLAY_SIDE,
) -> np.ndarray:
    """Downsample a 2D image for plotting without modifying source artifacts."""
    height, width = image.shape

    row_step = max(1, math.ceil(height / max_side))
    column_step = max(1, math.ceil(width / max_side))

    return image[::row_step, ::column_step]


def _shared_finite_limits(
    *images: np.ndarray,
) -> tuple[float, float]:
    finite_values = [
        image[np.isfinite(image)]
        for image in images
    ]
    finite_values = [
        values
        for values in finite_values
        if values.size > 0
    ]

    if not finite_values:
        return 0.0, 1.0

    vmin = min(float(values.min()) for values in finite_values)
    vmax = max(float(values.max()) for values in finite_values)

    if vmin == vmax:
        return vmin - 0.5, vmax + 0.5

    return vmin, vmax


def _symmetric_finite_limits(
    image: np.ndarray,
) -> tuple[float, float]:
    finite_values = image[np.isfinite(image)]

    if finite_values.size == 0:
        return -1.0, 1.0

    limit = float(np.max(np.abs(finite_values)))

    if limit == 0.0:
        limit = 1.0

    return -limit, limit


def _format_error_kind(error_kind: str) -> str:
    return error_kind.replace("_", " ").title()
