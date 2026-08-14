from pathlib import Path
from types import SimpleNamespace
from typing import Any
import warnings

import anndata as ad
import numpy as np
import pytest

from benchrep.records import evaluation_exports


def _make_step_spec(**overrides: Any) -> SimpleNamespace:
    values = {
        "plots_enabled": True,
        "pca_enabled": False,
        "umap_enabled": False,
        "tsne_enabled": False,
        "kmeans_enabled": False,
        "leiden_enabled": False,
        "hdbscan_enabled": False,
        "reconstruction_tiffs_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_required_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_h5ad(
        _adata: ad.AnnData,
        output_path: Path,
        *,
        overwrite: bool,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()

    def save_metrics(
        *,
        output_dir: Path,
        **_: Any,
    ) -> Path:
        output_path = Path(output_dir) / "metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return output_path

    monkeypatch.setattr(
        evaluation_exports,
        "write_h5ad",
        write_h5ad,
    )
    monkeypatch.setattr(
        evaluation_exports,
        "save_evaluation_metrics_json",
        save_metrics,
    )


def _run_export(
    *,
    tmp_path: Path,
    step_spec: SimpleNamespace,
    reconstruction_input: Any | None = None,
):
    return evaluation_exports.export_evaluation_outputs(
        adata=ad.AnnData(
            X=np.ones((3, 2), dtype=np.float32)
        ),
        reconstruction_input=reconstruction_input,
        reconstruction_outputs=None,
        step_spec=step_spec,
        embeddings_dir=tmp_path / "embeddings",
        embeddings_figures_dir=tmp_path / "embedding_figures",
        metrics_dir=tmp_path / "metrics",
        reconstructions_dir=tmp_path / "reconstructions",
        reconstruction_figures_dir=tmp_path / "reconstruction_figures",
    )


def _outcomes_by_name(result) -> dict[str, Any]:
    return {
        outcome.name: outcome
        for outcome in result.outcomes
    }


def test_irrelevant_plot_groups_are_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_required_exports(monkeypatch)

    def fail_if_called(**_: Any):
        pytest.fail("disabled plot exporter was called")

    monkeypatch.setattr(
        evaluation_exports,
        "export_reduction_plots",
        fail_if_called,
    )
    monkeypatch.setattr(
        evaluation_exports,
        "export_cluster_size_plots",
        fail_if_called,
    )

    result = _run_export(
        tmp_path=tmp_path,
        step_spec=_make_step_spec(),
    )
    outcomes = _outcomes_by_name(result)

    assert outcomes["reduction_plots"].status == "disabled"
    assert outcomes["cluster_size_plots"].status == "disabled"
    assert outcomes["reconstruction_tiffs"].status == "disabled"
    assert outcomes["reconstruction_grids"].status == "disabled"
    assert outcomes["evaluated_embeddings"].status == "completed"
    assert outcomes["metrics_json"].status == "completed"


def test_requested_empty_plot_groups_are_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_required_exports(monkeypatch)
    monkeypatch.setattr(
        evaluation_exports,
        "export_reduction_plots",
        lambda **_: {},
    )
    monkeypatch.setattr(
        evaluation_exports,
        "export_cluster_size_plots",
        lambda **_: {},
    )

    result = _run_export(
        tmp_path=tmp_path,
        step_spec=_make_step_spec(
            pca_enabled=True,
            hdbscan_enabled=True,
        ),
    )
    outcomes = _outcomes_by_name(result)

    assert outcomes["reduction_plots"].status == "skipped"
    assert outcomes["cluster_size_plots"].status == "skipped"
    assert outcomes["reduction_plots"].issues
    assert outcomes["cluster_size_plots"].issues


def test_export_failure_does_not_prevent_later_export_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_required_exports(monkeypatch)

    def fail_reduction_export(**_: Any):
        raise RuntimeError("reduction plot failed")

    cluster_path = tmp_path / "cluster.png"
    cluster_path.touch()

    monkeypatch.setattr(
        evaluation_exports,
        "export_reduction_plots",
        fail_reduction_export,
    )
    monkeypatch.setattr(
        evaluation_exports,
        "export_cluster_size_plots",
        lambda **_: {"hdbscan": [cluster_path]},
    )

    result = _run_export(
        tmp_path=tmp_path,
        step_spec=_make_step_spec(
            pca_enabled=True,
            hdbscan_enabled=True,
        ),
    )
    outcomes = _outcomes_by_name(result)

    assert outcomes["reduction_plots"].status == "failed"
    assert outcomes["reduction_plots"].issues == (
        "Error (RuntimeError): reduction plot failed",
    )
    assert outcomes["cluster_size_plots"].status == "completed"
    assert outcomes["metrics_json"].status == "completed"


def test_export_warnings_are_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_required_exports(monkeypatch)

    def warn_and_export(**_: Any):
        warnings.warn("plot warning", RuntimeWarning)
        return {"pca": [tmp_path / "pca.png"]}

    monkeypatch.setattr(
        evaluation_exports,
        "export_reduction_plots",
        warn_and_export,
    )

    result = _run_export(
        tmp_path=tmp_path,
        step_spec=_make_step_spec(pca_enabled=True),
    )
    outcome = _outcomes_by_name(result)["reduction_plots"]

    assert outcome.status == "completed_with_warnings"
    assert outcome.issues == (
        "Warning (RuntimeWarning): plot warning",
    )


def test_metrics_json_is_not_written_when_no_metrics(
    tmp_path: Path,
) -> None:
    output_path = evaluation_exports.save_evaluation_metrics_json(
        output_dir=tmp_path,
        adata=ad.AnnData(
            X=np.ones((3, 2), dtype=np.float32)
        ),
    )

    assert output_path is None
    assert not (tmp_path / "metrics.json").exists()


def test_empty_metrics_export_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_required_exports(monkeypatch)
    monkeypatch.setattr(
        evaluation_exports,
        "save_evaluation_metrics_json",
        lambda **_: None,
    )

    result = _run_export(
        tmp_path=tmp_path,
        step_spec=_make_step_spec(),
    )
    outcome = _outcomes_by_name(result)["metrics_json"]

    assert result.paths.metrics_json_path is None
    assert outcome.status == "disabled"