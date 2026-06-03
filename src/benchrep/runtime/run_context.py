from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4


@dataclass(frozen=True)
class RunContext:
    """
    Filesystem context for a single BenchRep run.

    RunContext stores the resolved run name and all standard output
    directories used by a training, inference, prediction, or test run.
    It is intended to be created through `RunContext.create()`, which
    derives a timestamped run name, handles rare folder-name collisions,
    creates the run directory structure, and returns an immutable context
    object.

    The object itself does not store model weights, configs, logs, or
    metadata contents directly. It only stores the paths where those
    artifacts should be written by the rest of the pipeline.

    Attributes
    ----------
    run_name:
        Final name of the run directory.
    output_dir:
        Root directory for this specific run.
    config_dir:
        Directory for raw and resolved configuration files.
    log_dir:
        Directory for stdout/stderr logs and run summaries.
    checkpoint_dir:
        Directory for Lightning checkpoints and trained model state.
    architecture_dir:
        Directory for architecture inspection artifacts such as torchview
        graphs, model summaries, and input signatures.
    metadata_dir:
        Directory for environment, version, and run metadata.
    artifact_dir:
        Directory for run-produced artifacts such as reconstructions,
        embeddings, predictions, and metrics.
    """
    run_name: str
    output_dir: Path
    records_dir: Path
    config_dir: Path
    log_dir: Path
    metadata_dir: Path
    checkpoint_dir: Path
    architecture_dir: Path
    artifact_dir: Path

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        stage: Literal["training", "prediction", "evaluation"],
        model_name: str,
        project_name: str | None = None,
        timestamp: str | None = None,
    ) -> "RunContext":
        output_root = Path(output_root).expanduser().resolve() / stage
        timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

        if project_name:
            base_run_name = f"{project_name}_{model_name}_{timestamp}"
        else:
            base_run_name = f"{model_name}_{timestamp}"

        run_name = base_run_name
        output_dir = output_root / run_name

        if output_dir.exists():
            suffix = uuid4().hex[:8]
            run_name = f"{base_run_name}_{suffix}"
            output_dir = output_root / run_name

        records_dir = output_dir / "records"

        config_dir = records_dir / "config"
        log_dir = records_dir / "logs"
        metadata_dir = records_dir / "metadata"

        checkpoint_dir = output_dir / "checkpoints"
        architecture_dir = output_dir / "architecture"
        artifact_dir = output_dir / "artifacts"

        for path in [
            records_dir,
            config_dir,
            log_dir,
            checkpoint_dir,
            architecture_dir,
            metadata_dir,
            artifact_dir,
        ]:
            path.mkdir(parents=True, exist_ok=False)

        return cls(
            run_name=run_name,
            output_dir=output_dir,
            records_dir=records_dir,
            config_dir=config_dir,
            log_dir=log_dir,
            metadata_dir=metadata_dir,
            checkpoint_dir=checkpoint_dir,
            architecture_dir=architecture_dir,
            artifact_dir=artifact_dir,
        )