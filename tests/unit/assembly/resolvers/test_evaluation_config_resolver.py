from pathlib import Path

import pytest
import yaml

from benchrep.assembly.resolvers.evaluation_config_resolver import (
    _load_prediction_manifest,
)

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