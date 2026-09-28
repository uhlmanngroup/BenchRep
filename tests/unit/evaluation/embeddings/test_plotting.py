from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pytest

from benchrep.evaluation.embeddings import plotting


def _make_projection_adata(labels: list[str]) -> ad.AnnData:
    n_obs = len(labels)

    adata = ad.AnnData(
        X=np.ones((n_obs, 2), dtype=np.float32),
        obs={
            "label": labels,
        },
    )
    adata.obsm["X_pca"] = np.column_stack(
        (
            np.arange(n_obs, dtype=np.float32),
            np.arange(n_obs, dtype=np.float32),
        )
    )

    return adata


def test_plot_2d_projection_uses_default_categorical_cmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adata = _make_projection_adata(["a", "b", "c"])

    original_get_cmap = plt.get_cmap
    requested_cmaps: list[str] = []

    def capture_get_cmap(
        name: str,
        lut: int | None = None,
    ):
        requested_cmaps.append(name)
        return original_get_cmap(name, lut)

    monkeypatch.setattr(
        plotting.plt,
        "get_cmap",
        capture_get_cmap,
    )

    plotting.plot_2d_projection(
        adata,
        basis="X_pca",
        color_by="label",
        color_kind="categorical",
        output_path=tmp_path / "categorical.png",
    )

    assert plotting.DEFAULT_CATEGORICAL_CMAP in requested_cmaps


def test_plot_2d_projection_uses_high_cardinality_categorical_cmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_cmap = plt.get_cmap(
        plotting.DEFAULT_CATEGORICAL_CMAP
    )

    labels = [
        f"label_{index}"
        for index in range(default_cmap.N + 1)
    ]
    adata = _make_projection_adata(labels)

    original_get_cmap = plt.get_cmap
    requested_cmaps: list[str] = []

    def capture_get_cmap(
        name: str,
        lut: int | None = None,
    ):
        requested_cmaps.append(name)
        return original_get_cmap(name, lut)

    monkeypatch.setattr(
        plotting.plt,
        "get_cmap",
        capture_get_cmap,
    )

    plotting.plot_2d_projection(
        adata,
        basis="X_pca",
        color_by="label",
        color_kind="categorical",
        output_path=tmp_path / "high_cardinality.png",
    )

    assert (
        plotting.DEFAULT_HIGH_CARDINALITY_CATEGORICAL_CMAP
        in requested_cmaps
    )


def test_plot_2d_projection_warns_for_undersized_explicit_categorical_cmap(
    tmp_path: Path,
) -> None:
    explicit_cmap = plt.get_cmap("tab10")

    labels = [
        f"label_{index}"
        for index in range(explicit_cmap.N + 1)
    ]
    adata = _make_projection_adata(labels)

    with pytest.warns(
        UserWarning,
        match=(
            rf"provides {explicit_cmap.N} native colors for "
            rf"{explicit_cmap.N + 1} categories"
        ),
    ):
        plotting.plot_2d_projection(
            adata,
            basis="X_pca",
            color_by="label",
            color_kind="categorical",
            cmap="tab10",
            output_path=tmp_path / "undersized_cmap.png",
        )