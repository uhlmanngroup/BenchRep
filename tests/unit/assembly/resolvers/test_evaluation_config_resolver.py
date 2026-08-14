from pathlib import Path

import pytest
import yaml

from benchrep.assembly.resolvers.evaluation_config_resolver import (
    _load_prediction_manifest,
    resolve_step_spec,
)
from benchrep.assembly.schemas import EvaluationConfig

@pytest.mark.parametrize(
    "status",
    [
        "completed",
        "completed_with_warnings",
    ],
)
def test_load_prediction_manifest_accepts_successful_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    manifest_path = tmp_path / "prediction_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "prediction",
                "status": status,
            }
        )
    )

    manifest = _load_prediction_manifest(manifest_path)

    assert manifest["status"] == status


def test_resolve_step_spec_rejects_no_enabled_work() -> None:
    config = EvaluationConfig.model_validate(
        {
            "source": {
                "embeddings_path": "embeddings.h5ad",
            },
            "reductions": {
                "pca": {"enabled": False},
                "umap": {"enabled": False},
                "tsne": {"enabled": False},
            },
            "clustering": {
                "kmeans": {"enabled": False},
                "leiden": {"enabled": False},
                "hdbscan": {"enabled": False},
            },
            "metrics": {
                "clustering": {
                    "internal": {"enabled": False},
                    "external": {"enabled": False},
                },
                "embedding": {"enabled": False},
                "predictability": {"enabled": False},
                "reconstruction": {"enabled": False},
            },
            "reconstruction": {
                "export_tiffs": False,
            },
            "plots": {
                "enabled": False,
            },
        }
    )

    with pytest.raises(
        ValueError,
        match="resolves to no enabled evaluation",
    ):
        resolve_step_spec(
            evaluation_config=config,
            has_reconstructions=False,
        )