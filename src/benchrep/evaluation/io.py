from __future__ import annotations

from pathlib import Path

import anndata as ad


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