from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import pytest
from pydantic import ValidationError
import yaml

from benchrep.assembly.registries.core import DATASETS
from benchrep.workflows import train_ae, predict_ae, evaluate
from benchrep.workflows.predict import PredictionWorkflowResult
from benchrep.workflows.train import TrainingWorkflowResult
from tests.fixtures.datasets import (
    TinySyntheticDataset,
    CompatibleAutoencoderBatchDataset,
    PrivateImageBatchDataset,
)
from tests.fixtures.datamodules import ExternalDataModule
from tests.fixtures.models import (
    CompatibleExternalAutoencoder,
    PrivateBatchExternalAutoencoder,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "configs"
TRAINING_CONFIG_PATH = (
    CONFIG_DIR / "training_tiny_synthetic_ae.yaml"
)
PREDICTION_CONFIG_PATH = (
    CONFIG_DIR / "prediction_tiny_synthetic.yaml"
)
EVALUATION_CONFIG_PATH = (
    CONFIG_DIR / "evaluation_tiny_synthetic.yaml"
)

def test_prediction_training_manifest_override_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The training-manifest argument overrides the path configured in YAML."""
    _register_internal_dataset()
    monkeypatch.chdir(tmp_path)

    training_result = train_ae(
        config_path=TRAINING_CONFIG_PATH,
    )

    actual_manifest_path = training_result.manifest_path.resolve()
    nonexistent_manifest_path = (
        tmp_path / "nonexistent_training_manifest.yaml"
    )

    assert actual_manifest_path.is_file()
    assert not nonexistent_manifest_path.exists()

    with PREDICTION_CONFIG_PATH.open(encoding="utf-8") as handle:
        prediction_config = yaml.safe_load(handle)

    prediction_config["source"]["training_manifest_path"] = str(
        nonexistent_manifest_path
    )

    mutated_config_path = (
        tmp_path / "prediction_with_conflicting_manifest.yaml"
    )

    with mutated_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            prediction_config,
            handle,
            sort_keys=False,
        )

    prediction_result = predict_ae(
        config_path=mutated_config_path,
        training_manifest_path=actual_manifest_path,
    )

    assert prediction_result.manifest_path.is_file()

    assert (
        prediction_result.run_spec.training_manifest_path
        == actual_manifest_path
    )
    assert (
        prediction_result.config.source.training_manifest_path
        == actual_manifest_path
    )

    with prediction_result.manifest_path.open(
        encoding="utf-8"
    ) as handle:
        prediction_manifest = yaml.safe_load(handle)

    recorded_training_manifest_path = Path(
        prediction_manifest["source"]["training_manifest_path"]
    )

    assert recorded_training_manifest_path == actual_manifest_path


def test_evaluation_prediction_manifest_override_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prediction-manifest argument overrides the path configured in YAML."""
    _register_internal_dataset()
    monkeypatch.chdir(tmp_path)

    training_result = train_ae(config_path=TRAINING_CONFIG_PATH)

    prediction_result = predict_ae(
        config_path=PREDICTION_CONFIG_PATH,
        training_manifest_path=training_result.manifest_path,
    )

    actual_manifest_path = prediction_result.manifest_path.resolve()
    nonexistent_manifest_path = (
        tmp_path / "nonexistent_prediction_manifest.yaml"
    )

    assert actual_manifest_path.is_file()
    assert not nonexistent_manifest_path.exists()

    with EVALUATION_CONFIG_PATH.open(encoding="utf-8") as handle:
        evaluation_config = yaml.safe_load(handle)

    evaluation_config["source"]["prediction_manifest_path"] = str(
        nonexistent_manifest_path
    )

    mutated_config_path = (
        tmp_path / "evaluation_with_conflicting_manifest.yaml"
    )

    with mutated_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            evaluation_config,
            handle,
            sort_keys=False,
        )

    evaluation_result = evaluate(
        config_path=mutated_config_path,
        prediction_manifest_path=actual_manifest_path,
    )

    assert evaluation_result.manifest_path.is_file()

    assert (
        evaluation_result.run_spec.input_spec.prediction_manifest_path
        == actual_manifest_path
    )
    assert (
        evaluation_result.config.source.prediction_manifest_path
        == actual_manifest_path
    )

    with evaluation_result.manifest_path.open(
        encoding="utf-8"
    ) as handle:
        evaluation_manifest = yaml.safe_load(handle)

    recorded_prediction_manifest_path = Path(
        evaluation_manifest["source"]["prediction_manifest_path"]
    )

    assert recorded_prediction_manifest_path == actual_manifest_path


def test_internal_model_with_standard_external_datamodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    datamodule = _make_standard_external_datamodule()

    training_result = train_ae(
        config_path=TRAINING_CONFIG_PATH,
        datamodule=datamodule,
        compatibility_policy="error",
    )

    prediction_result = predict_ae(
        config_path=PREDICTION_CONFIG_PATH,
        training_manifest_path=training_result.manifest_path,
        datamodule=datamodule,
        compatibility_policy="error",
    )

    _assert_successful_train_predict(
        training_result=training_result,
        prediction_result=prediction_result,
        expected_model_source="config",
        expected_datamodule_source="external_object",
    )


def test_internal_model_with_private_external_datamodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    datamodule = _make_private_external_datamodule()

    with pytest.raises(
        RuntimeError,
        match=(
            "Training failed while using an external datamodule "
            "with an internal model"
        ),
    ) as exc_info:
        train_ae(
            config_path=TRAINING_CONFIG_PATH,
            datamodule=datamodule,
            compatibility_policy="error",
        )

    assert "AutoencoderBatch" in str(exc_info.value)
    assert "required fields" in str(exc_info.value)
    assert "`x`" in str(exc_info.value)
    assert "Original error (KeyError)" in str(exc_info.value)


def test_standard_external_model_with_standard_external_datamodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    datamodule = _make_standard_external_datamodule()

    training_result = train_ae(
        config_path=TRAINING_CONFIG_PATH,
        model=CompatibleExternalAutoencoder(),
        datamodule=datamodule,
        compatibility_policy="error",
    )

    prediction_result = predict_ae(
        config_path=PREDICTION_CONFIG_PATH,
        training_manifest_path=training_result.manifest_path,
        model=CompatibleExternalAutoencoder(),
        datamodule=datamodule,
        compatibility_policy="error",
    )

    _assert_successful_train_predict(
        training_result=training_result,
        prediction_result=prediction_result,
        expected_model_source="external_object",
        expected_datamodule_source="external_object",
    )


def test_standard_external_model_with_internal_datamodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_internal_dataset()
    monkeypatch.chdir(tmp_path)

    training_result = train_ae(
        config_path=TRAINING_CONFIG_PATH,
        model=CompatibleExternalAutoencoder(),
        compatibility_policy="error",
    )

    prediction_result = predict_ae(
        config_path=PREDICTION_CONFIG_PATH,
        training_manifest_path=training_result.manifest_path,
        model=CompatibleExternalAutoencoder(),
        compatibility_policy="error",
    )

    _assert_successful_train_predict(
        training_result=training_result,
        prediction_result=prediction_result,
        expected_model_source="external_object",
        expected_datamodule_source="config",
    )


def test_standard_external_model_with_private_external_datamodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    datamodule = _make_private_external_datamodule()

    with pytest.raises(KeyError, match="x"):
        train_ae(
            config_path=TRAINING_CONFIG_PATH,
            model=CompatibleExternalAutoencoder(),
            datamodule=datamodule,
            compatibility_policy="error",
        )


def test_private_external_model_with_internal_datamodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_internal_dataset()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        TypeError,
        match=r"training_step.*missing required field.*x",
    ):
        train_ae(
            config_path=TRAINING_CONFIG_PATH,
            model=PrivateBatchExternalAutoencoder(),
            compatibility_policy="error",
        )


def test_private_external_model_with_standard_external_datamodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    datamodule = _make_standard_external_datamodule()

    with pytest.raises(KeyError, match="image"):
        train_ae(
            config_path=TRAINING_CONFIG_PATH,
            model=PrivateBatchExternalAutoencoder(),
            datamodule=datamodule,
            compatibility_policy="error",
        )


def test_private_external_model_with_private_external_datamodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    datamodule = _make_private_external_datamodule()

    training_result = train_ae(
        config_path=TRAINING_CONFIG_PATH,
        model=PrivateBatchExternalAutoencoder(),
        datamodule=datamodule,
        compatibility_policy="error",
    )

    prediction_result = predict_ae(
        config_path=PREDICTION_CONFIG_PATH,
        training_manifest_path=training_result.manifest_path,
        model=PrivateBatchExternalAutoencoder(),
        datamodule=datamodule,
        compatibility_policy="error",
    )

    _assert_successful_train_predict(
        training_result=training_result,
        prediction_result=prediction_result,
        expected_model_source="external_object",
        expected_datamodule_source="external_object",
    )


def test_external_model_allows_missing_model_config_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_internal_dataset()
    monkeypatch.chdir(tmp_path)

    training_config_path = _write_config_without_sections(
        source_path=TRAINING_CONFIG_PATH,
        output_path=tmp_path / "training_without_model_config.yaml",
        sections={
            "model",
            "encoder",
            "decoder",
            "losses",
            "optimizer",
        },
    )

    training_result = train_ae(
        config_path=training_config_path,
        model=CompatibleExternalAutoencoder(),
        compatibility_policy="error",
    )

    prediction_result = predict_ae(
        config_path=PREDICTION_CONFIG_PATH,
        training_manifest_path=training_result.manifest_path,
        model=CompatibleExternalAutoencoder(),
        compatibility_policy="error",
    )

    assert training_result.config.model is None
    assert training_result.config.encoder is None
    assert training_result.config.decoder is None
    assert training_result.config.losses == {}
    assert training_result.config.optimizer is None

    _assert_successful_train_predict(
        training_result=training_result,
        prediction_result=prediction_result,
        expected_model_source="external_object",
        expected_datamodule_source="config",
    )


def test_external_datamodule_allows_missing_data_config_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    training_config_path = _write_config_without_sections(
        source_path=TRAINING_CONFIG_PATH,
        output_path=tmp_path / "training_without_data_config.yaml",
        sections={
            "dataset",
            "datamodule",
        },
    )
    prediction_config_path = _write_config_without_sections(
        source_path=PREDICTION_CONFIG_PATH,
        output_path=tmp_path / "prediction_without_dataset_config.yaml",
        sections={
            "dataset",
        },
    )

    datamodule = _make_standard_external_datamodule()

    training_result = train_ae(
        config_path=training_config_path,
        datamodule=datamodule,
        compatibility_policy="error",
    )

    prediction_result = predict_ae(
        config_path=prediction_config_path,
        training_manifest_path=training_result.manifest_path,
        datamodule=datamodule,
        compatibility_policy="error",
    )

    assert training_result.config.dataset is None

    _assert_successful_train_predict(
        training_result=training_result,
        prediction_result=prediction_result,
        expected_model_source="config",
        expected_datamodule_source="external_object",
    )


def test_external_model_and_datamodule_allow_all_related_sections_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    training_config_path = _write_config_without_sections(
        source_path=TRAINING_CONFIG_PATH,
        output_path=tmp_path / "training_without_override_sections.yaml",
        sections={
            "model",
            "encoder",
            "decoder",
            "losses",
            "optimizer",
            "dataset",
            "datamodule",
        },
    )
    prediction_config_path = _write_config_without_sections(
        source_path=PREDICTION_CONFIG_PATH,
        output_path=tmp_path / "prediction_without_dataset_config.yaml",
        sections={
            "dataset",
        },
    )

    datamodule = _make_standard_external_datamodule()

    training_result = train_ae(
        config_path=training_config_path,
        model=CompatibleExternalAutoencoder(),
        datamodule=datamodule,
        compatibility_policy="error",
    )

    prediction_result = predict_ae(
        config_path=prediction_config_path,
        training_manifest_path=training_result.manifest_path,
        model=CompatibleExternalAutoencoder(),
        datamodule=datamodule,
        compatibility_policy="error",
    )

    assert training_result.config.model is None
    assert training_result.config.encoder is None
    assert training_result.config.decoder is None
    assert training_result.config.losses == {}
    assert training_result.config.optimizer is None
    assert training_result.config.dataset is None

    _assert_successful_train_predict(
        training_result=training_result,
        prediction_result=prediction_result,
        expected_model_source="external_object",
        expected_datamodule_source="external_object",
    )


@pytest.mark.parametrize(
    ("removed_sections", "expected_missing_field"),
    [
        pytest.param(
            {
                "model",
                "encoder",
                "decoder",
                "losses",
                "optimizer",
            },
            "model",
            id="missing-model-without-override",
        ),
        pytest.param(
            {
                "dataset",
                "datamodule",
            },
            "dataset",
            id="missing-datamodule-without-override",
        ),
    ],
)
def test_missing_config_sections_are_rejected_without_override(
    tmp_path: Path,
    removed_sections: set[str],
    expected_missing_field: str,
) -> None:
    training_config_path = _write_config_without_sections(
        source_path=TRAINING_CONFIG_PATH,
        output_path=tmp_path / "incomplete_training_config.yaml",
        sections=removed_sections,
    )

    with pytest.raises(
        ValidationError,
        match=rf"{expected_missing_field}.*required",
    ):
        train_ae(
            config_path=training_config_path,
        )



def _make_standard_external_datamodule() -> ExternalDataModule:
    return ExternalDataModule(
        train_dataset=CompatibleAutoencoderBatchDataset(
            n_samples=24,
            seed=137,
        ),
        val_dataset=CompatibleAutoencoderBatchDataset(
            n_samples=8,
            seed=138,
        ),
        predict_dataset=CompatibleAutoencoderBatchDataset(
            n_samples=32,
            seed=139,
        ),
    )


def _make_private_external_datamodule() -> ExternalDataModule:
    return ExternalDataModule(
        train_dataset=PrivateImageBatchDataset(
            n_samples=24,
            seed=137,
        ),
        val_dataset=PrivateImageBatchDataset(
            n_samples=8,
            seed=138,
        ),
        predict_dataset=PrivateImageBatchDataset(
            n_samples=32,
            seed=139,
        ),
    )


def _register_internal_dataset() -> None:
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register(
            "tiny_synthetic",
            TinySyntheticDataset,
        )


def _assert_successful_train_predict(
    *,
    training_result: TrainingWorkflowResult,
    prediction_result: PredictionWorkflowResult,
    expected_model_source: str,
    expected_datamodule_source: str,
) -> None:
    assert training_result.manifest_path.is_file()
    assert training_result.audit_report_path.is_file()
    assert prediction_result.manifest_path.is_file()
    assert prediction_result.audit_report_path.is_file()

    assert training_result.checkpoint_callback.best_model_path
    assert Path(
        training_result.checkpoint_callback.best_model_path
    ).is_file()

    assert len(prediction_result.predictions) == 4

    with training_result.manifest_path.open(
        encoding="utf-8",
    ) as handle:
        training_manifest = yaml.safe_load(handle)

    assert training_manifest["status"] == "completed"
    assert (
        training_manifest["provenance"]["model"]["source"]
        == expected_model_source
    )
    assert (
        training_manifest["provenance"]["datamodule"]["source"]
        == expected_datamodule_source
    )

    expected_reconstructability = (
        expected_model_source == "config"
        and expected_datamodule_source == "config"
    )
    assert (
        training_manifest["provenance"]["config"][
            "run_reconstructable_from_resolved_config"
        ]
        is expected_reconstructability
    )

    with prediction_result.manifest_path.open(
        encoding="utf-8",
    ) as handle:
        prediction_manifest = yaml.safe_load(handle)

    prediction_provenance = prediction_manifest["provenance"][
        "prediction"
    ]

    assert prediction_manifest["status"] == "completed"
    assert (
        prediction_provenance["model"]["source"]
        == expected_model_source
    )
    assert (
        prediction_provenance["datamodule"]["source"]
        == expected_datamodule_source
    )
    assert (
            prediction_provenance["config"][
                "run_reconstructable_from_resolved_config"
            ]
            is expected_reconstructability
    )

    with training_result.audit_report_path.open(
        encoding="utf-8",
    ) as handle:
        training_audit = yaml.safe_load(handle)

    with prediction_result.audit_report_path.open(
        encoding="utf-8",
    ) as handle:
        prediction_audit = yaml.safe_load(handle)

    assert training_audit["summary"]["errors"] == 0
    assert prediction_audit["summary"]["errors"] == 0


def _write_config_without_sections(
    *,
    source_path: Path,
    output_path: Path,
    sections: Iterable[str],
) -> Path:
    with source_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for section in sections:
        config.pop(section, None)

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
        )

    return output_path