"""An invalid later prediction batch must prevent exports and leave diagnostics."""

from pathlib import Path

import pytest
import yaml

from benchrep import PredictionOverrides, TrainingOverrides, predict_ae, train_ae
from benchrep.assembly.schemas.parsing import (
    PredictionConfigContext,
    TrainingConfigContext,
    parse_training_config,
    parse_prediction_config,
)
from benchrep.interfaces.contracts import AutoencoderBatch
from benchrep.interfaces.model_families import AUTOENCODER_FAMILY
from tests.fixtures.datamodules import ExternalDataModule
from tests.fixtures.datasets import CompatibleAutoencoderBatchDataset
from tests.fixtures.models import (
    CompatibleExternalAutoencoder,
    ExternalAutoencoderPrediction,
)


class InvalidLaterPrediction(CompatibleExternalAutoencoder):
    def __init__(self):
        super().__init__()
        self.seen_batches = []

    def predict_step(
        self, batch: AutoencoderBatch, batch_idx: int
    ) -> ExternalAutoencoderPrediction:
        prediction = super().predict_step(batch, batch_idx)
        self.seen_batches.append(batch_idx)
        if batch_idx == 1:
            # The annotation is valid, but the actual second batch violates it.
            prediction.embedding = "invalid embedding"
        return prediction


@pytest.fixture
def trained(tmp_path):
    dataset = CompatibleAutoencoderBatchDataset(n_samples=4, seed=137)
    datamodule = ExternalDataModule(
        train_dataset=dataset,
        val_dataset=dataset,
        predict_dataset=dataset,
        batch_size=2,
    )
    config = parse_training_config(
        {
            "run": {"output_root": tmp_path / "training"},
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "num_sanity_val_steps": 0,
            },
            "logger": None,
            "inspection": {"torchview": {"enabled": False}},
        },
        context=TrainingConfigContext(external_model=True, external_datamodule=True),
    )
    result = train_ae(
        config=config,
        overrides=TrainingOverrides(
            model=CompatibleExternalAutoencoder(), datamodule=datamodule
        ),
        compatibility_policy="error",
    )
    assert result.status_report.status == "completed"
    return result.manifest_path, datamodule


@pytest.mark.parametrize(
    ("anndata_enabled", "reconstructions_enabled"),
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["both-exports", "anndata-only", "reconstructions-only", "exports-disabled"],
)
def test_invalid_later_batch_writes_failed_manifest_without_partial_exports(
    tmp_path,
    trained,
    anndata_enabled,
    reconstructions_enabled,
):
    training_manifest, datamodule = trained
    # Linked prediction inherits the training run's output root.
    output_root = tmp_path / "training" / "prediction"
    config = parse_prediction_config(
        {
            "source": {"training_manifest_path": training_manifest},
            "exports": {
                "anndata": {"enabled": anndata_enabled},
                "reconstructions": {"enabled": reconstructions_enabled},
            },
        },
        context=PredictionConfigContext(
            model_family=AUTOENCODER_FAMILY,
            training_manifest_path_overridden=False,
        ),
    )
    model = InvalidLaterPrediction()

    with pytest.raises(
        RuntimeError, match="Prediction was finalized with status 'failed'"
    ) as caught:
        predict_ae(
            config=config,
            overrides=PredictionOverrides(model=model, datamodule=datamodule),
            compatibility_policy="error",
        )

    assert model.seen_batches == [0, 1]
    manifests = list(output_root.rglob("prediction_manifest.yaml"))
    assert len(manifests) == 1
    assert str(manifests[0]) in str(caught.value)
    manifest = yaml.safe_load(manifests[0].read_text())
    assert manifest["status"] == "failed"
    assert Path(manifest["source"]["training_manifest_path"]) == training_manifest
    inference = manifest["status_report"]["inference"]
    assert inference["status"] == "failed"
    assert inference["n_batches"] == 2
    assert inference["n_observations"] is None
    assert len(inference["issues"]) == 1
    issue = inference["issues"][0]
    assert issue.startswith("Error (TypeError):")
    assert "batch 1" in issue and "embedding" in issue

    for name, enabled in (
        ("anndata", anndata_enabled),
        ("reconstructions", reconstructions_enabled),
    ):
        outcome = manifest["status_report"]["exports"][name]
        assert outcome["status"] == ("skipped" if enabled else "disabled")
        assert outcome["issues"] == (
            ["Skipped because prediction output validation failed."] if enabled else []
        )
    assert manifest["exports"]["anndata"]["path"] is None
    assert not list(output_root.rglob("*.h5ad"))
    assert not list(output_root.rglob("*.pt"))
