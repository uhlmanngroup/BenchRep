from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args
from uuid import uuid4


RunStage = Literal["training", "prediction", "evaluation"]


@dataclass(frozen=True)
class RunContext:
    """
    Filesystem context for one BenchRep workflow run.

    `RunContext` is the central path registry for a single training,
    prediction, or evaluation run. It stores the resolved run name, the root
    output directory for that run, and the standard subdirectories where other
    parts of the workflow should write configs, logs, manifests, checkpoints,
    embeddings, reconstruction tensors, and other generated records.

    Instances should be created through `RunContext.create()` rather than by
    direct construction. The factory method applies BenchRep's output
    convention:

        <output_root>/<stage>/<run_name>/

    where `stage` is one of `"training"`, `"prediction"`, or `"evaluation"`.
    The run name is derived from the project name, model name, and timestamp.
    If a run directory with the same name already exists, a short UUID suffix
    is appended to avoid overwriting existing outputs.

    Common record directories are created for every stage:

        records/configs/
        records/logs/
        records/metadata/

    Stage-specific directories are created only for the stages that currently
    use them. Training creates `checkpoints/` and `architecture/`; prediction
    creates `embeddings/` and `reconstructions/`; evaluation currently creates
    only the common record directories. Even when a stage-specific directory is
    not created, its conventional path is still stored on the context object so
    downstream code has a single source of truth for path names.

    The context does not write model weights, configs, logs, embeddings, or
    metrics itself. It only owns the directory layout and exposes immutable
    paths for the rest of the pipeline.

    Attributes
    ----------
    stage:
        Workflow stage for this run: `"training"`, `"prediction"`, or
        `"evaluation"`.
    run_name:
        Final directory name for this run, including timestamp and optional
        collision suffix.
    output_dir:
        Root directory for this specific run.
    records_dir:
        Parent directory for BenchRep-managed record files.
    config_dir:
        Directory for input and resolved configuration files.
    log_dir:
        Directory for console captures and local run logs.
    metadata_dir:
        Directory for manifests, environment records, and workflow metadata.
    checkpoint_dir:
        Conventional directory for training checkpoints.
    architecture_dir:
        Conventional directory for architecture inspection artifacts, such as
        torchview graphs or model summaries.
    embedding_dir:
        Conventional directory for prediction embedding exports.
    reconstruction_dir:
        Conventional directory for selected reconstruction tensor exports.
    """

    stage: RunStage
    run_name: str
    output_dir: Path

    # Common record directories
    records_dir: Path
    config_dir: Path
    log_dir: Path
    metadata_dir: Path

    # Stage-specific conventional directories
    checkpoint_dir: Path
    architecture_dir: Path
    embedding_dir: Path
    reconstruction_dir: Path

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        stage: RunStage,
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
        config_dir = records_dir / "configs"
        log_dir = records_dir / "logs"
        metadata_dir = records_dir / "metadata"

        checkpoint_dir = output_dir / "checkpoints"
        architecture_dir = output_dir / "architecture"
        embedding_dir = output_dir / "embeddings"
        reconstruction_dir = output_dir / "reconstructions"

        dirs_to_create = [
            records_dir,
            config_dir,
            log_dir,
            metadata_dir,
        ]

        if stage == "training":
            dirs_to_create.extend([checkpoint_dir, architecture_dir])
        elif stage == "prediction":
            dirs_to_create.extend([embedding_dir, reconstruction_dir])
        elif stage == "evaluation":
            pass
        else:
            raise ValueError(f"Unsupported run stage: {stage!r}. "
                             f"Supported stages: {get_args(RunStage)}"
            )

        for path in dirs_to_create:
            path.mkdir(parents=True, exist_ok=False)

        return cls(
            stage=stage,
            run_name=run_name,
            output_dir=output_dir,
            records_dir=records_dir,
            config_dir=config_dir,
            log_dir=log_dir,
            metadata_dir=metadata_dir,
            checkpoint_dir=checkpoint_dir,
            architecture_dir=architecture_dir,
            embedding_dir=embedding_dir,
            reconstruction_dir=reconstruction_dir,
        )