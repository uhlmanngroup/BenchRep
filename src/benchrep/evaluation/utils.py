import anndata as ad
import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


def validate_adata_x(adata: ad.AnnData) -> None:
    """Validate that adata.X is present and 2D."""

    if adata.X is None:
        raise ValueError("adata.X is required for dimensionality reduction.")

    if adata.X.ndim != 2:
        raise ValueError(f"adata.X must be 2D, got shape {adata.X.shape}.")


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