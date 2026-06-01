from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch


ArrayLike = np.ndarray | torch.Tensor


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


def package_matrix_as_anndata(
    embeddings: ArrayLike,
    *,
    sample_ids: Sequence[str] | None = None,
    labels: Sequence[Any] | np.ndarray | torch.Tensor | None = None,
    label_key: str = "label",
    metadata: pd.DataFrame | Mapping[str, Sequence[Any]] | None = None,
) -> ad.AnnData:
    """
    Package a matrix (e.g. embeddings) and sample-level annotations into
    an AnnData object.

    BenchRep evaluation uses AnnData as its internal contract:
    `adata.X` stores embeddings, and `adata.obs` stores sample metadata.
    """

    embedding_array = to_numpy(embeddings)

    if embedding_array.ndim != 2:
        raise ValueError(
            "embeddings must be a 2D array with shape "
            f"(n_samples, n_features), got shape {embedding_array.shape}."
        )

    n_samples = embedding_array.shape[0]

    if sample_ids is None:
        obs_index = [f"sample_{i}" for i in range(n_samples)]
    else:
        if len(sample_ids) != n_samples:
            raise ValueError(
                "sample_ids length must match embeddings.shape[0], got "
                f"{len(sample_ids)} sample IDs for {n_samples} embeddings."
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
                "labels length must match embeddings.shape[0], got "
                f"{len(label_array)} labels for {n_samples} embeddings."
            )

        obs[label_key] = label_array

    return ad.AnnData(X=embedding_array, obs=obs)


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
            "metadata length must match embeddings.shape[0], got "
            f"{len(obs)} metadata rows for {n_samples} embeddings."
        )

    return obs.reset_index(drop=True)