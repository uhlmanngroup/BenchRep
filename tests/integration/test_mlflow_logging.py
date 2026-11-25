"""Check the optional MLflow logger through a real, local training workflow."""

from importlib.util import find_spec
import math
from pathlib import Path

import pytest
import yaml

from benchrep.assembly.registries.core import DATASETS
from benchrep.workflows import train_vae
from tests.fixtures.datasets import TinySyntheticDataset


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures/configs/training_tiny_synthetic_vae.yaml"
)


def test_mlflow_records_training_and_checkpoint(tmp_path, monkeypatch):
    if find_spec("mlflow") is None:
        pytest.skip("MLflow is an optional dependency; install benchrep[mlflow].")

    # Import after the availability check: broken installed versions must fail,
    # rather than being skipped by pytest.importorskip.
    from mlflow.tracking import MlflowClient

    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "benchrep-integration-test"
    experiment_id = client.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["run"]["output_root"] = str(tmp_path / "outputs")
    config["logger"] = {
        "name": "mlflow",
        "params": {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "log_model": True,
        },
    }
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)

    result = train_vae(config=config_path)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"

    runs = client.search_runs([experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.metrics, "No training metrics reached MLflow."
    assert all(math.isfinite(value) for value in run.data.metrics.values())
    assert run.data.params["prediction_reconstruction_latent_source"] == (
        result.model.prediction_reconstruction_latent_source
    )

    def artifact_paths(path=""):
        for item in client.list_artifacts(run.info.run_id, path):
            if item.is_dir:
                yield from artifact_paths(item.path)
            else:
                yield item.path

    checkpoint = Path(result.checkpoint_callback.best_model_path)
    uploaded = [path for path in artifact_paths() if Path(path).name == checkpoint.name]
    assert len(uploaded) == 1, "The best training checkpoint was not logged."
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    downloaded = Path(client.download_artifacts(
        run.info.run_id, uploaded[0], dst_path=str(download_dir),
    ))
    assert downloaded.read_bytes() == checkpoint.read_bytes()
