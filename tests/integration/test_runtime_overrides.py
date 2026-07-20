from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchrep.assembly.registries.core import DATASETS
from benchrep.workflows import train_ae, predict_ae, evaluate
from tests.fixtures.datasets import TinySyntheticDataset


CONFIG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "configs"


def test_prediction_training_manifest_override_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The training-manifest argument overrides the path configured in YAML."""
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)

    monkeypatch.chdir(tmp_path)

    training_result = train_ae(
        config_path=CONFIG_DIR / "training_tiny_synthetic_ae.yaml",
    )

    actual_manifest_path = training_result.manifest_path.resolve()
    nonexistent_manifest_path = (
        tmp_path / "nonexistent_training_manifest.yaml"
    )

    assert actual_manifest_path.is_file()
    assert not nonexistent_manifest_path.exists()

    with (
        CONFIG_DIR / "prediction_tiny_synthetic.yaml"
    ).open(encoding="utf-8") as handle:
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
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)

    monkeypatch.chdir(tmp_path)

    training_result = train_ae(
        config_path=CONFIG_DIR / "training_tiny_synthetic_ae.yaml",
    )

    prediction_result = predict_ae(
        config_path=CONFIG_DIR / "prediction_tiny_synthetic.yaml",
        training_manifest_path=training_result.manifest_path,
    )

    actual_manifest_path = prediction_result.manifest_path.resolve()
    nonexistent_manifest_path = (
        tmp_path / "nonexistent_prediction_manifest.yaml"
    )

    assert actual_manifest_path.is_file()
    assert not nonexistent_manifest_path.exists()

    with (
        CONFIG_DIR / "evaluation_tiny_synthetic.yaml"
    ).open(encoding="utf-8") as handle:
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