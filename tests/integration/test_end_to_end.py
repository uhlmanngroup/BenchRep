from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)

from benchrep.workflows import (
    train_ae,
    predict_ae,
    train_vae,
    predict_vae,
    evaluate,
)
from tests.fixtures.datasets import TinySyntheticDataset
from benchrep.assembly.registries.core import DATASETS
from benchrep.architecture.models import VAE
from benchrep.assembly.schemas import (
    AdditionalCallbackConfig,
    EarlyStoppingConfig,
    LoggerConfig,
    PredictionInferenceConfig,
)


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
    assert training_result.checkpoint_callback.best_model_path
    assert Path(training_result.checkpoint_callback.best_model_path).is_file()
    assert training_result.checkpoint_callback.last_model_path
    assert Path(training_result.checkpoint_callback.last_model_path).is_file()
    _assert_completed_manifest(training_result.manifest_path, "training")

    prediction_result = predict_fn(
        config_path=CONFIG_DIR / "prediction_tiny_synthetic.yaml",
        training_manifest_path=training_result.manifest_path,
    )

    assert prediction_result.manifest_path.is_file()

    if training_config_name == "training_tiny_synthetic_vae.yaml":
        model = prediction_result.model
        assert isinstance(model, VAE)
        assert prediction_result.run_spec.reconstruction_latent_source == "mean"
        assert model.prediction_reconstruction_latent_source == "mean"

        with torch.no_grad():
            for prediction in prediction_result.predictions:
                torch.testing.assert_close(
                    prediction.reconstruction,
                    model.decode(prediction.z_mu),
                )

            forward_output = model(
                prediction_result.predictions[0].input
            )
            torch.testing.assert_close(
                forward_output["reconstruction"],
                model.decode(forward_output["z_sample"]),
            )

        _assert_vae_reconstruction_provenance(
            prediction_result,
            configured_source=None,
            effective_source="mean",
            resolution="benchrep_default",
            uses_randomness=False,
        )


    assert len(prediction_result.predictions) == 4
    _assert_completed_manifest(prediction_result.manifest_path, "prediction")
    with prediction_result.manifest_path.open(encoding="utf-8") as handle:
        prediction_manifest = yaml.safe_load(handle)

    assert prediction_manifest["source"]["checkpoint_selection"] == "best"
    assert (
            prediction_manifest["source"]["checkpoint_source"]
            == "training_manifest_best"
    )
    assert Path(
        prediction_manifest["source"]["checkpoint_path"]
    ) == prediction_result.run_spec.checkpoint_path

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
    _assert_completed_manifest(evaluation_result.manifest_path, "evaluation")

    assert evaluation_result.status_report.status == "completed"
    assert evaluation_result.status_report.embeddings.status == "completed"
    assert (
        evaluation_result.status_report.reconstructions.status
        == "completed"
    )
    assert evaluation_result.status_report.exports.status == "completed"

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


def test_internal_vae_prediction_can_reconstruct_from_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)

    monkeypatch.chdir(tmp_path)

    training_result = train_vae(
        config_path=CONFIG_DIR / "training_tiny_synthetic_vae.yaml",
    )

    prediction_result = predict_vae(
        config_path=CONFIG_DIR / "prediction_tiny_synthetic.yaml",
        config_components={
            "inference": PredictionInferenceConfig(
                reconstruction_latent_source="sample",
            ),
        },
        training_manifest_path=training_result.manifest_path,
    )

    model = prediction_result.model
    assert isinstance(model, VAE)
    assert prediction_result.run_spec.reconstruction_latent_source == "sample"
    assert model.prediction_reconstruction_latent_source == "sample"

    with torch.no_grad():
        for prediction in prediction_result.predictions:
            torch.testing.assert_close(
                prediction.reconstruction,
                model.decode(prediction.z_sample),
            )

    _assert_vae_reconstruction_provenance(
        prediction_result,
        configured_source="sample",
        effective_source="sample",
        resolution="prediction_config",
        uses_randomness=True,
    )


def _assert_completed_manifest(path: Path, expected_stage: str) -> None:
    with path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    assert manifest["stage"] == expected_stage
    assert manifest["status"] == "completed"


def _count_paths(value: Any) -> int:
    if isinstance(value, Path):
        return 1

    if isinstance(value, Mapping):
        return sum(_count_paths(item) for item in value.values())

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return sum(_count_paths(item) for item in value)

    return 0


def _assert_vae_reconstruction_provenance(
    prediction_result: Any,
    *,
    configured_source: str | None,
    effective_source: str,
    resolution: str,
    uses_randomness: bool,
) -> None:
    with prediction_result.manifest_path.open(
        encoding="utf-8"
    ) as handle:
        manifest = yaml.safe_load(handle)

    recorded_source = (
        manifest["provenance"]["prediction"]["inference"]
        ["reconstruction_latent_source"]
    )

    assert recorded_source == {
        "configured": configured_source,
        "effective": effective_source,
        "resolution": resolution,
    }

    runtime_environment_path = (
        prediction_result.run_context.metadata_dir
        / "prediction_runtime_environment.yaml"
    )

    with runtime_environment_path.open(encoding="utf-8") as handle:
        runtime_environment = yaml.safe_load(handle)

    reproducibility = runtime_environment["workflow"]["reproducibility"]

    assert (
        reproducibility["requested_overrides"]
        ["reconstruction_latent_source"]
        == configured_source
    )
    assert (
        reproducibility["resolved"]["reconstruction_latent_source"]
        == effective_source
    )

    assert reproducibility["components"]["vae_reconstruction"] == {
        "applicable": True,
        "model_source": "config",
        "latent_source": effective_source,
        "uses_randomness": uses_randomness,
        "seed": (
            prediction_result.run_spec.seed
            if uses_randomness
            else None
        ),
    }


def test_training_callbacks_are_not_inherited_by_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)

    monkeypatch.chdir(tmp_path)

    training_result = train_ae(
        config_path=(
            CONFIG_DIR
            / "training_tiny_synthetic_ae.yaml"
        ),
        config_components={
            "logger": LoggerConfig(
                name="csv",
                params={
                    "save_dir": str(tmp_path / "lightning_logs"),
                },
            ),
            "early_stopping": EarlyStoppingConfig(
                monitor="val/loss",
                mode="min",
                patience=3,
            ),
            "additional_callbacks": [
                AdditionalCallbackConfig(
                    name="learning_rate_monitor",
                    params={
                        "logging_interval": "epoch",
                    },
                ),
            ],
        },
    )

    early_stopping_callbacks = [
        callback
        for callback in training_result.trainer.callbacks
        if isinstance(callback, EarlyStopping)
    ]
    learning_rate_callbacks = [
        callback
        for callback in training_result.trainer.callbacks
        if isinstance(callback, LearningRateMonitor)
    ]

    assert len(early_stopping_callbacks) == 1
    assert early_stopping_callbacks[0].monitor == "val/loss"
    assert early_stopping_callbacks[0].mode == "min"
    assert early_stopping_callbacks[0].patience == 3

    assert len(learning_rate_callbacks) == 1
    assert learning_rate_callbacks[0].logging_interval == "epoch"

    prediction_result = predict_ae(
        config_path=(
            CONFIG_DIR
            / "prediction_tiny_synthetic.yaml"
        ),
        training_manifest_path=training_result.manifest_path,
    )

    assert not any(
        isinstance(callback, EarlyStopping)
        for callback in prediction_result.trainer.callbacks
    )
    assert not any(
        isinstance(callback, LearningRateMonitor)
        for callback in prediction_result.trainer.callbacks
    )