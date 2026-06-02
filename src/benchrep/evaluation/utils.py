import anndata as ad


def validate_adata_x(adata: ad.AnnData) -> None:
    """Validate that adata.X is present and 2D."""

    if adata.X is None:
        raise ValueError("adata.X is required for dimensionality reduction.")

    if adata.X.ndim != 2:
        raise ValueError(f"adata.X must be 2D, got shape {adata.X.shape}.")