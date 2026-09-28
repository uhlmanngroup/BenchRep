from pathlib import Path
from types import SimpleNamespace
from typing import Any
import warnings

import anndata as ad
import numpy as np
import pytest
import json

import benchrep.records.evaluation_exports as evaluation_exports
from benchrep.runtime.status.evaluation import EvaluationOutcome


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
        "benchrep.records.evaluation_exports.write_h5ad",
        write_h5ad,
    )
    monkeypatch.setattr(
        "benchrep.records.evaluation_exports.save_evaluation_metrics_json",
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
        anndata_outcomes=(
            EvaluationOutcome(
                name="test_anndata_step",
                category="anndata",
                status="completed",
            ),
        ),
        reconstruction_input=reconstruction_input,
        reconstruction_outputs=None,
        step_spec=step_spec,
        anndata_dir=tmp_path / "anndata",
        anndata_figures_dir=tmp_path / "embedding_figures",
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
    assert outcomes["evaluated_anndata"].status == "completed"
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


def test_metrics_json_supports_reconstruction_metrics_without_adata(
    tmp_path: Path,
) -> None:
    output_path = evaluation_exports.save_evaluation_metrics_json(
        output_dir=tmp_path,
        adata=None,
        reconstruction_outputs={
            "reconstruction_metrics": {
                "mse": 1.25,
            },
        },
    )

    assert output_path is not None

    with output_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    assert metrics == {
        "reconstruction": {
            "mse": 1.25,
        },
    }


def test_export_reduction_plots_forwards_color_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adata = ad.AnnData(
        X=np.ones((4, 2), dtype=np.float32),
        obs={
            "digit_label": ["0", "1", "0", "1"],
            "mean_stroke_width": [1.0, 2.0, 3.0, 4.0],
        },
    )
    adata.obsm["X_pca"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    calls: list[dict[str, Any]] = []

    def capture_plot_call(*args: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        "benchrep.records.evaluation_exports.plot_2d_projection",
        capture_plot_call,
    )

    step_spec = SimpleNamespace(
        plots_enabled=True,
        pca_enabled=True,
        pca_params={"key_added": "X_pca"},
        umap_enabled=False,
        umap_params={},
        tsne_enabled=False,
        tsne_params={},
        plot_params={
            "accent_color": "#6A3D9A",
            "color_by": [
                {
                    "key": "digit_label",
                    "kind": "categorical",
                    "cmap": "Set3",
                },
                {
                    "key": "mean_stroke_width",
                    "kind": "continuous",
                    "cmap": "plasma",
                },
            ],
            "dpi": 300,
            "formats": ["png"],
        },
    )

    evaluation_exports.export_reduction_plots(
        output_dir=tmp_path,
        adata=adata,
        step_spec=step_spec,
    )

    colored_calls = [
        call
        for call in calls
        if call["color_by"] is not None
    ]

    assert len(colored_calls) == 2

    assert colored_calls[0]["color_by"] == "digit_label"
    assert colored_calls[0]["color_kind"] == "categorical"
    assert colored_calls[0]["cmap"] == "Set3"

    assert colored_calls[1]["color_by"] == "mean_stroke_width"
    assert colored_calls[1]["color_kind"] == "continuous"
    assert colored_calls[1]["cmap"] == "plasma"
