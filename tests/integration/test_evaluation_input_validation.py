from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import json

import torch
import yaml

from benchrep import evaluate
from benchrep.assembly.schemas import (
    EvaluationConfig,
    EvaluationRunConfig,
    EvaluationSourceConfig,
    EvaluationReductionsConfig,
    UMAPConfig,
)


def test_evaluate_rejects_nonfinite_embeddings(
    tmp_path: Path,
) -> None:
    embeddings = np.arange(24, dtype=np.float32).reshape(6, 4)
    embeddings[0, 0] = np.nan
    embeddings[1, 1] = np.inf
    embeddings[2, 2] = -np.inf

    embeddings_path = tmp_path / "nonfinite_embeddings.h5ad"
    ad.AnnData(X=embeddings).write_h5ad(embeddings_path)

    config = EvaluationConfig(
        source=EvaluationSourceConfig(
            embeddings_path=embeddings_path,
        ),
        run=EvaluationRunConfig(
            output_root=tmp_path / "outputs",
            run_name="nonfinite_embeddings",
        ),
        reductions=EvaluationReductionsConfig(
            umap=UMAPConfig(enabled=False),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Found 1 NaN values and 2 infinite values across "
            r"3 observations"
        ),
    ):
        evaluate(full_config_object=config)


def test_evaluate_supports_reconstruction_only_input(
    tmp_path: Path,
) -> None:
    reconstruction_dir = tmp_path / "reconstruction_bundle"
    reconstruction_dir.mkdir()

    inputs = torch.zeros((6, 1, 4, 4), dtype=torch.float32)
    reconstructions = torch.ones_like(inputs)
    obs = {
        "sample_id": [f"sample_{index}" for index in range(6)],
        "label": [0, 0, 0, 1, 1, 1],
    }

    torch.save(inputs, reconstruction_dir / "input.pt")
    torch.save(
        reconstructions,
        reconstruction_dir / "reconstruction.pt",
    )
    torch.save(obs, reconstruction_dir / "obs.pt")

    config = EvaluationConfig.model_validate(
        {
            "source": {
                "reconstructions_path": reconstruction_dir,
            },
            "run": {
                "output_root": tmp_path / "outputs",
                "run_name": "reconstruction_only",
            },
            "plots": {
                "enabled": False,
            },
        }
    )

    result = evaluate(full_config_object=config)

    assert result.status_report.status == "completed"
    assert result.status_report.embeddings.status == "disabled"
    assert result.status_report.reconstructions.status == "completed"

    assert result.adata is None
    assert result.export_paths.evaluated_embeddings_path is None
    assert result.export_paths.metrics_json_path is not None
    assert result.export_paths.metrics_json_path.is_file()

    with result.export_paths.metrics_json_path.open(
        encoding="utf-8"
    ) as handle:
        metrics = json.load(handle)

    assert "reconstruction" in metrics

    with result.manifest_path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    assert manifest["source"]["mode"] == "direct"
    assert manifest["source"]["embeddings"] == {
        "source": None,
        "path": None,
    }
    assert manifest["exports"]["embeddings"] == {
        "path": None,
        "n_obs": None,
        "n_vars": None,
    }
    assert manifest["summary"]["has_embeddings"] is False
    assert manifest["summary"]["has_reconstructions"] is True
