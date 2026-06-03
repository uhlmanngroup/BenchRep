from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml


def write_yaml_record(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_training_manifest(
    *,
    output_path: Path,
    stage: Literal["training"],
    run_name: str,
    output_dir: Path,
    resolved_config_path: Path,
    checkpoint_dir: Path,
    best_checkpoint_path: str | None,
    best_checkpoint_score: float | None,
    last_checkpoint_path: str | None,
) -> None:
    manifest = {
        "stage": stage,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "resolved_config_path": str(resolved_config_path),
        "checkpoints": {
            "checkpoint_dir": str(checkpoint_dir),
            "best_checkpoint_path": best_checkpoint_path,
            "best_checkpoint_score": best_checkpoint_score,
            "last_checkpoint_path": last_checkpoint_path,
        },
    }

    write_yaml_record(manifest, output_path)