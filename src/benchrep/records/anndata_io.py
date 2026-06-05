from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from benchrep.evaluation.utils import ArrayLike, to_numpy


def read_h5ad(path: str | Path) -> ad.AnnData:
    """Read an AnnData evaluation artifact from disk."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No AnnData file found at {path}.")

    return ad.read_h5ad(path)


def write_h5ad(
    adata: ad.AnnData,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write an AnnData evaluation artifact to disk."""
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. Pass overwrite=True to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def package_matrix_as_anndata(
    matrix: ArrayLike,
    *,
    sample_ids: Sequence[str] | None = None,
    labels: Sequence[Any] | ArrayLike | None = None,
    label_key: str = "label",
    metadata: pd.DataFrame | Mapping[str, Sequence[Any]] | None = None,
) -> ad.AnnData:
    """
    Package a matrix (e.g. embeddings) and sample-level annotations into
    an AnnData object.

    BenchRep evaluation uses AnnData as its internal contract:
    `adata.X` stores the matrix, and `adata.obs` stores sample metadata.
    """

    matrix_array = to_numpy(matrix)

    if matrix_array.ndim != 2:
        raise ValueError(
            "matrix must be a 2D array with shape "
            f"(n_samples, n_features), got shape {matrix_array.shape}."
        )

    n_samples = matrix_array.shape[0]

    if sample_ids is None:
        obs_index = [f"sample_{i}" for i in range(n_samples)]
    else:
        if len(sample_ids) != n_samples:
            raise ValueError(
                "sample_ids length must match matrix.shape[0], got "
                f"{len(sample_ids)} sample IDs for {n_samples} rows."
            )
        obs_index = list(map(str, sample_ids))

    if len(set(obs_index)) != len(obs_index):
        raise ValueError("sample_ids must be unique.")

    obs = _metadata_to_obs(metadata, n_samples=n_samples)
    obs.index = pd.Index(obs_index, name="sample_id")

    if labels is not None:
        if isinstance(labels, (np.ndarray, torch.Tensor)):
            label_array = to_numpy(labels)
        else:
            label_array = np.asarray(labels)

        if label_array.ndim != 1:
            raise ValueError(
                f"labels must be 1D, got shape {label_array.shape}."
            )

        if len(label_array) != n_samples:
            raise ValueError(
                "labels length must match matrix.shape[0], got "
                f"{len(label_array)} labels for {n_samples} rows."
            )

        obs[label_key] = label_array

    return ad.AnnData(X=matrix_array, obs=obs)


def _metadata_to_obs(
    metadata: pd.DataFrame | Mapping[str, Sequence[Any]] | None,
    *,
    n_samples: int,
) -> pd.DataFrame:
    """Convert optional metadata input into an obs DataFrame."""

    # Return empty obs table so sample IDs can still be assigned downstream
    if metadata is None:
        return pd.DataFrame(index=range(n_samples))

    if isinstance(metadata, pd.DataFrame):
        obs = metadata.copy()
    elif isinstance(metadata, Mapping):
        obs = pd.DataFrame(metadata)
    else:
        raise TypeError(
            "metadata must be a pandas DataFrame, mapping, or None, "
            f"got {type(metadata).__name__}."
        )

    if len(obs) != n_samples:
        raise ValueError(
            "metadata length must match matrix.shape[0], got "
            f"{len(obs)} metadata rows for {n_samples} rows."
        )

    return obs.reset_index(drop=True)