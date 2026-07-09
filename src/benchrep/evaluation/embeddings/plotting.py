from __future__ import annotations

from pathlib import Path
from typing import Literal
from collections.abc import Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ColorKind = Literal["auto", "categorical", "continuous"]

DEFAULT_PCA_VARIANCE_PLOT_N_COMPONENTS = 20
PCAVariancePlotKind = Literal["scree", "cumulative"]
DEFAULT_ACCENT_COLOR = "#6A3D9A"


def plot_2d_projection(
    adata: ad.AnnData,
    *,
    basis: str,
    accent_color: str = DEFAULT_ACCENT_COLOR,
    color_by: str | None = None,
    color_kind: ColorKind = "auto",
    max_categorical_levels: int = 30,
    output_path: str | Path,
    title: str | None = None,
    dpi: int = 300,
    overwrite: bool = False,
) -> None:
    """
    Plot a 2D projection from ``adata.obsm``.

    ``basis`` should point to a coordinate matrix in ``adata.obsm``, such as
    ``"X_pca"``, ``"X_umap"``, or ``"X_tsne"``. If ``color_by`` is provided, it
    must point to an ``adata.obs`` column and is rendered as categorical or
    continuous depending on ``color_kind``.

    The output format is inferred from ``output_path`` suffix, e.g. ``.png``,
    ``.pdf``, or ``.svg``
    """

    if color_kind not in {"auto", "categorical", "continuous"}:
        raise ValueError(
            "color_kind must be one of {'auto', 'categorical', 'continuous'}, "
            f"got {color_kind!r}."
        )

    if max_categorical_levels < 1:
        raise ValueError(
            "max_categorical_levels must be >= 1, "
            f"got {max_categorical_levels}."
        )

    if basis not in adata.obsm:
        raise KeyError(
            f"adata.obsm does not contain {basis!r}. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    if color_by is not None and color_by not in adata.obs.columns:
        raise KeyError(
            f"adata.obs does not contain {color_by!r}. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    _validate_dpi(dpi)

    coords = adata.obsm[basis]

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{basis!r}] must have shape (n_obs, >=2), "
            f"got {coords.shape}."
        )

    if coords.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm[{basis!r}] must have one row per observation, "
            f"got {coords.shape[0]} rows for {adata.n_obs} observations."
        )

    output_path = _prepare_output_path(output_path, overwrite=overwrite)

    fig, ax = plt.subplots(figsize=(6, 5))

    if color_by is None:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=8,
            alpha=0.8,
            color=accent_color,
        )
        ax.set_title(title or basis)

    else:
        values = adata.obs[color_by]
        resolved_color_kind = (
            _infer_color_kind(
                values,
                max_categorical_levels=max_categorical_levels,
            )
            if color_kind == "auto"
            else color_kind
        )

        if resolved_color_kind == "continuous":
            if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
                raise TypeError(
                    f"Cannot render adata.obs[{color_by!r}] as continuous because "
                    f"its dtype is {values.dtype!r}."
                )

            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=values.to_numpy(dtype=float, na_value=np.nan),
                s=8,
                alpha=0.8,
            )
            fig.colorbar(scatter, ax=ax, label=color_by)

        else:
            categorical_values = (
                values
                .astype("category")
                .cat
                .remove_unused_categories()
            )
            codes = categorical_values.cat.codes.to_numpy(dtype=float)
            codes[codes < 0] = np.nan

            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=codes,
                s=8,
                alpha=0.8,
            )

            labels = [str(label) for label in categorical_values.cat.categories]
            if len(labels) <= 20:
                handles, _ = scatter.legend_elements()
                ax.legend(
                    handles,
                    labels,
                    title=color_by,
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    borderaxespad=0,
                )

        ax.set_title(title or f"{basis} colored by {color_by}")

    ax.set_xlabel(f"{basis}_1")
    ax.set_ylabel(f"{basis}_2")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(
    *,
    explained_variance_ratio: Sequence[float],
    output_path: str | Path,
    kind: PCAVariancePlotKind,
    max_components: int = DEFAULT_PCA_VARIANCE_PLOT_N_COMPONENTS,
    title: str | None = None,
    dpi: int = 300,
    accent_color: str = DEFAULT_ACCENT_COLOR,
    overwrite: bool = False,
) -> None:
    """Plot PCA explained-variance diagnostics."""

    if kind not in {"scree", "cumulative"}:
        raise ValueError(f"Unsupported PCA variance plot kind: {kind!r}.")

    if max_components < 1:
        raise ValueError(f"max_components must be >= 1, got {max_components}.")

    _validate_dpi(dpi)

    explained = np.asarray(explained_variance_ratio, dtype=float).reshape(-1)

    if explained.size == 0:
        raise ValueError("explained_variance_ratio must contain at least one value.")

    if not np.all(np.isfinite(explained)):
        raise ValueError("explained_variance_ratio contains NaN or infinite values.")

    if np.any(explained < 0.0):
        raise ValueError("explained_variance_ratio must contain non-negative values.")

    if float(np.max(explained)) > 1.0 + 1e-6:
        raise ValueError(
            "explained_variance_ratio appears to contain percentages or raw "
            "variance values. Expected fractions in [0, 1]."
        )

    if float(np.sum(explained)) > 1.0 + 1e-6:
        raise ValueError(
            "explained_variance_ratio sums to more than 1.0. Expected "
            "per-component fractions of total variance."
        )

    n_plot = min(max_components, explained.size)
    components = np.arange(1, n_plot + 1)

    if kind == "scree":
        values = explained[:n_plot] * 100.0
        ylabel = "Explained variance (%)"
        resolved_title = title or "PCA explained variance"
    else:
        values = np.cumsum(explained)[:n_plot] * 100.0
        ylabel = "Cumulative explained variance (%)"
        resolved_title = title or "PCA cumulative explained variance"

    output_path = _prepare_output_path(output_path, overwrite=overwrite)

    fig, ax = plt.subplots(figsize=(6, 4))

    if kind == "scree":
        ax.bar(components, values, color=accent_color)

        max_value = float(np.max(values))
        upper_ylim = min(100.0, max_value * 1.1)
        ax.set_ylim(0.0, upper_ylim)
    else:
        ax.plot(components, values, marker="o", color=accent_color)
        ax.set_ylim(0.0, 100.0)

    ax.set_xlabel("Principal component")
    ax.set_ylabel(ylabel)
    ax.set_title(resolved_title)
    ax.set_xticks(components)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_cluster_sizes(
    labels: Sequence[object] | pd.Series,
    *,
    output_path: str | Path,
    title: str | None = None,
    dpi: int = 300,
    overwrite: bool = False,
    accent_color: str = DEFAULT_ACCENT_COLOR,
) -> None:
    """Plot cluster sizes for one clustering label vector."""

    _validate_dpi(dpi)

    label_series = pd.Series(labels, dtype="object")
    label_series = label_series[label_series.notna()]

    if label_series.empty:
        raise ValueError("Cannot plot cluster sizes because labels contain no values.")

    counts = label_series.astype(str).value_counts().sort_values(ascending=True)

    output_path = _prepare_output_path(output_path, overwrite=overwrite)

    height = max(4.0, min(18.0, 0.28 * len(counts) + 1.5))
    fig, ax = plt.subplots(figsize=(7, height))

    ax.barh(
        counts.index,
        counts.to_numpy(),
        color=accent_color,
    )

    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Cluster")
    ax.set_title(title or "Cluster sizes")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _infer_color_kind(
    values,
    *,
    max_categorical_levels: int,
) -> Literal["categorical", "continuous"]:
    """Infer whether an obs column should be plotted as categorical or continuous."""

    non_missing = values.dropna()
    n_unique = int(non_missing.nunique())

    if n_unique <= 1:
        return "categorical"

    if (
        isinstance(values.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(values)
        or pd.api.types.is_string_dtype(values)
        or pd.api.types.is_bool_dtype(values)
    ):
        return "categorical"

    if pd.api.types.is_integer_dtype(values):
        if n_unique <= max_categorical_levels:
            return "categorical"
        return "continuous"

    if pd.api.types.is_numeric_dtype(values):
        array = non_missing.to_numpy(dtype=float, na_value=np.nan)
        integer_like = np.all(np.isclose(array, np.round(array)))

        if integer_like and n_unique <= max_categorical_levels:
            return "categorical"

        return "continuous"

    return "categorical"


def _validate_dpi(dpi: int) -> None:
    if not isinstance(dpi, int):
        raise TypeError(f"dpi must be an int, got {type(dpi).__name__}.")
    if dpi < 72:
        raise ValueError(f"dpi must be >= 72, got {dpi}.")


def _prepare_output_path(
    output_path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output path already exists: {output_path}. "
            "Pass overwrite=True to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path