from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchrep.evaluation.utils import validate_reconstruction_arrays


@dataclass(frozen=True)
class ReconstructionEvaluationInput:
    """Loaded reconstruction artifacts for reconstruction-side evaluation.

    This is the in-memory contract used by reconstruction evaluation functions.
    Disk export to TIFF/PNG belongs to the records/export layer.
    """

    inputs: np.ndarray
    reconstructions: np.ndarray
    obs: Any
    metadata: Mapping[str, Any] | None = None
    n_examples: int | None = None


def load_reconstruction_evaluation_input(
    *,
    input_path: str | Path,
    reconstruction_path: str | Path,
    obs_path: str | Path,
    metadata_path: str | Path | None = None,
    n_examples: int | None = None,
) -> ReconstructionEvaluationInput:
    """Load prediction-stage reconstruction artifacts for evaluation."""

    inputs = _load_pt(input_path)
    reconstructions = _load_pt(reconstruction_path)

    inputs, reconstructions = validate_reconstruction_arrays(
        inputs=inputs,
        reconstructions=reconstructions,
    )

    obs = _load_pt(obs_path)
    metadata = _load_pt(metadata_path) if metadata_path is not None else None

    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(
            "Loaded reconstruction metadata must be a mapping, got "
            f"{type(metadata).__name__}."
        )

    if n_examples is not None and n_examples < 1:
        raise ValueError(f"n_examples must be >= 1, got {n_examples}.")

    available_examples = int(inputs.shape[0])

    if n_examples is None:
        effective_n_examples = available_examples
    else:
        effective_n_examples = min(n_examples, available_examples)

    return ReconstructionEvaluationInput(
        inputs=inputs,
        reconstructions=reconstructions,
        obs=obs,
        metadata=metadata,
        n_examples=effective_n_examples,
    )


def _load_pt(path: str | Path) -> Any:
    """Load a torch-saved artifact on CPU."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Reconstruction artifact does not exist: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")