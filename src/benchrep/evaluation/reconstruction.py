from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from benchrep.evaluation.utils import ArrayLike, to_numpy


def export_reconstruction_examples(
    inputs: ArrayLike,
    reconstructions: ArrayLike,
    output_dir: str | Path,
    *,
    n_examples: int = 16,
    prefix: str = "reconstruction",
    overwrite: bool = False,
) -> None:
    """
    Export input, reconstruction, and absolute-error examples as TIFF files.

    Arrays are saved without channel projection or reshaping. Expected input
    shapes are ``(B, H, W)`` or ``(B, C, H, W)``.
    """

    input_array = to_numpy(inputs)
    reconstruction_array = to_numpy(reconstructions)

    if input_array.shape != reconstruction_array.shape:
        raise ValueError(
            "inputs and reconstructions must have matching shapes, got "
            f"{input_array.shape} and {reconstruction_array.shape}."
        )

    if input_array.ndim not in {3, 4}:
        raise ValueError(
            "Expected inputs with shape (B, H, W) or (B, C, H, W), got "
            f"{input_array.shape}."
        )

    if n_examples < 1:
        raise ValueError(f"n_examples must be >= 1, got {n_examples}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_examples = min(n_examples, input_array.shape[0])
    input_array = input_array[:n_examples]
    reconstruction_array = reconstruction_array[:n_examples]
    error_array = np.abs(input_array - reconstruction_array)

    _save_tiff(
        input_array,
        output_dir / f"{prefix}_inputs.tif",
        overwrite=overwrite,
    )

    _save_tiff(
        reconstruction_array,
        output_dir / f"{prefix}_reconstructions.tif",
        overwrite=overwrite,
    )

    _save_tiff(
        error_array,
        output_dir / f"{prefix}_absolute_errors.tif",
        overwrite=overwrite,
    )


def _save_tiff(
    array: np.ndarray,
    path: Path,
    *,
    overwrite: bool,
) -> None:
    """Save an array as a TIFF file."""

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. Pass overwrite=True to replace it."
        )

    tifffile.imwrite(path, array)