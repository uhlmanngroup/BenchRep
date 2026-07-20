from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from benchrep.workflows import (
    train_ae,
    predict_ae,
    train_vae,
    predict_vae,
    evaluate,
)
from tests.fixtures.datasets import TinySyntheticDataset
from benchrep.assembly.registries.core import DATASETS


CONFIG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "configs"


@pytest.mark.parametrize(
    (
        "training_config_name",
        "train_fn",
        "predict_fn",
        "expected_embedding_keys",
    ),
    [
        pytest.param(
            "training_tiny_synthetic_ae.yaml",
            train_ae,
            predict_ae,
            {"embedding"},
            id="autoencoder",
        ),
        pytest.param(
            "training_tiny_synthetic_vae.yaml",
            train_vae,
            predict_vae,
            {"embedding", "z_mu", "z_logvar"},
            id="vae",
        ),
    ],
)
def test_internal_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    training_config_name: str,
    train_fn: Callable,
    predict_fn: Callable,
    expected_embedding_keys: set[str],
) -> None:
    """Exercise YAML-driven training, prediction, and evaluation workflows."""
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)

    monkeypatch.chdir(tmp_path)

    training_result = train_fn(
        config_path=CONFIG_DIR / training_config_name,
    )

    assert training_result.manifest_path.is_file()
    assert training_result.audit_report_path.is_file()
    assert training_result.checkpoint_callback.best_model_path
    assert Path(training_result.checkpoint_callback.best_model_path).is_file()
    assert training_result.checkpoint_callback.last_model_path
    assert Path(training_result.checkpoint_callback.last_model_path).is_file()
    _assert_completed_manifest(training_result.manifest_path, "training")
    _assert_audit_has_no_errors(training_result.audit_report_path)

    prediction_result = predict_fn(
        config_path=CONFIG_DIR / "prediction_tiny_synthetic.yaml",
        training_manifest_path=training_result.manifest_path,
    )

    assert prediction_result.manifest_path.is_file()
    assert prediction_result.audit_report_path.is_file()
    assert len(prediction_result.predictions) == 4
    _assert_completed_manifest(prediction_result.manifest_path, "prediction")
    _assert_audit_has_no_errors(prediction_result.audit_report_path)

    embedding_export = prediction_result.export_paths.embedding_export
    reconstruction_paths = prediction_result.export_paths.reconstruction_paths

    assert embedding_export is not None
    assert embedding_export.embeddings_h5ad_path is not None
    assert embedding_export.embeddings_h5ad_path.is_file()
    assert embedding_export.resolved_primary_key == "embedding"
    assert embedding_export.resolved_keys is not None
    assert expected_embedding_keys <= set(embedding_export.resolved_keys)

    assert reconstruction_paths is not None
    assert reconstruction_paths.n_examples_exported == 8
    assert reconstruction_paths.input_path is not None
    assert reconstruction_paths.reconstruction_path is not None
    assert reconstruction_paths.obs_path is not None
    assert reconstruction_paths.metadata_path is not None

    reconstruction_obs = torch.load(
        reconstruction_paths.obs_path,
        map_location="cpu",
        weights_only=False,
    )
    assert len(reconstruction_obs["source_index"]) == 8
    assert set(reconstruction_obs["label"]) == {0, 1, 2, 3}
    assert {"label_str", "continuous_target", "group"} <= set(
        reconstruction_obs
    )

    evaluation_result = evaluate(
        config_path=CONFIG_DIR / "evaluation_tiny_synthetic.yaml",
        prediction_manifest_path=prediction_result.manifest_path,
    )

    assert evaluation_result.manifest_path.is_file()
    assert evaluation_result.audit_report_path.is_file()
    _assert_completed_manifest(evaluation_result.manifest_path, "evaluation")
    _assert_audit_has_no_errors(evaluation_result.audit_report_path)

    assert evaluation_result.adata.n_obs == 32
    assert evaluation_result.adata.obsm["X_pca"].shape == (32, 4)
    assert "kmeans" in evaluation_result.adata.obs.columns
    assert {
        "label",
        "label_str",
        "continuous_target",
        "group",
    } <= set(evaluation_result.adata.obs.columns)

    export_paths = evaluation_result.export_paths
    assert export_paths.evaluated_embeddings_path.is_file()
    assert export_paths.metrics_json_path.is_file()
    assert _count_paths(export_paths.reduction_plot_paths) > 0
    assert _count_paths(export_paths.cluster_size_plot_paths) > 0
    assert _count_paths(export_paths.reconstruction_tiff_paths) > 0
    assert _count_paths(export_paths.reconstruction_grid_paths) > 0

    with export_paths.metrics_json_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    assert "clustering" in metrics
    assert "predictability" in metrics
    assert "reconstruction" in metrics


def _assert_completed_manifest(path: Path, expected_stage: str) -> None:
    with path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    assert manifest["stage"] == expected_stage
    assert manifest["status"] == "completed"


def _assert_audit_has_no_errors(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        report = yaml.safe_load(handle)

    assert report["summary"]["errors"] == 0


def _count_paths(value: Any) -> int:
    if isinstance(value, Path):
        return 1

    if isinstance(value, Mapping):
        return sum(_count_paths(item) for item in value.values())

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return sum(_count_paths(item) for item in value)

    return 0