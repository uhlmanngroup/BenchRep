from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt


def plot_2d_projection(
    adata: ad.AnnData,
    *,
    basis: str,
    color: str,
    output_path: str | Path,
    title: str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Plot a 2D projection from ``adata.obsm`` colored by an ``adata.obs`` column.

    ``basis`` should point to a coordinate matrix in ``adata.obsm``, such as
    ``"X_pca"``, ``"X_umap"``, or ``"X_tsne"``.
    """

    if basis not in adata.obsm:
        raise KeyError(
            f"adata.obsm does not contain {basis!r}. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    if color not in adata.obs.columns:
        raise KeyError(
            f"adata.obs does not contain {color!r}. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    coords = adata.obsm[basis]

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{basis!r}] must have shape (n_obs, >=2), "
            f"got {coords.shape}."
        )

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {output_path}. "
            "Pass overwrite=True to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = adata.obs[color].astype("category")
    codes = values.cat.codes

    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=codes,
        s=8,
        alpha=0.8,
    )

    ax.set_xlabel(f"{basis}_1")
    ax.set_ylabel(f"{basis}_2")
    ax.set_title(title or f"{basis} colored by {color}")

    handles, _ = scatter.legend_elements()
    labels = list(values.cat.categories)

    if len(labels) <= 20:
        ax.legend(
            handles,
            labels,
            title=color,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)