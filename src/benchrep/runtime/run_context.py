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
    prediction exports, evaluation artifacts, figures, metrics, and other
    generated records.

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
    use them. Training creates `checkpoints/` and `architecture/`. Prediction
    creates top-level `embeddings/` and `reconstructions/` directories for
    prediction exports. Evaluation creates a centralized `records/metrics/`
    directory for machine-readable metric records, an `artifacts/` tree for
    exported evaluation data products, and a `figures/` tree for generated
    evaluation plots and visual summaries. Evaluation reconstruction artifacts
    are organized under `artifacts/reconstructions/inputs/`,
    `artifacts/reconstructions/predictions/`, and
    `artifacts/reconstructions/error_maps/`.

    Even when a stage-specific directory is not created, its conventional path
    is still stored on the context object so downstream code has a single source
    of truth for path names.

    The context does not write model weights, configs, logs, embeddings,
    reconstructions, metrics, or figures itself. It only owns the directory
    layout and exposes immutable paths for the rest of the pipeline.

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
    training_checkpoint_dir:
        Conventional directory for training checkpoints.
    training_architecture_dir:
        Conventional directory for architecture inspection artifacts, such as
        torchview graphs or model summaries.
    prediction_embeddings_dir:
        Conventional directory for prediction embedding exports.
    prediction_reconstructions_dir:
        Conventional directory for selected prediction reconstruction exports.
    evaluation_artifacts_dir:
        Parent directory for exported evaluation data products.
    evaluation_figures_dir:
        Parent directory for generated evaluation figures.
    evaluation_metrics_dir:
        Directory for machine-readable evaluation metric records.
    evaluation_embeddings_dir:
        Directory for embedding-side evaluation artifacts.
    evaluation_reconstructions_dir:
        Directory for reconstruction-side evaluation artifacts.
    evaluation_embeddings_figures_dir:
        Directory for embedding-side evaluation figures.
    evaluation_reconstructions_figures_dir:
        Directory for reconstruction-side evaluation figures.
    evaluation_reconstruction_inputs_dir:
        Directory for exported reconstruction input examples.
    evaluation_reconstruction_predictions_dir:
        Directory for exported reconstruction prediction examples.
    evaluation_reconstruction_error_maps_dir:
        Directory for exported reconstruction error maps.
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
    training_checkpoint_dir: Path
    training_architecture_dir: Path
    prediction_embeddings_dir: Path
    prediction_reconstructions_dir: Path
    evaluation_artifacts_dir: Path
    evaluation_figures_dir: Path
    evaluation_metrics_dir: Path
    evaluation_embeddings_dir: Path
    evaluation_reconstructions_dir: Path
    evaluation_embeddings_figures_dir: Path
    evaluation_reconstructions_figures_dir: Path
    evaluation_reconstruction_inputs_dir: Path
    evaluation_reconstruction_predictions_dir: Path
    evaluation_reconstruction_error_maps_dir: Path

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        stage: RunStage,
        run_name_stem: str | None = None,
        model_name: str | None = None,
        project_name: str | None = None,
        timestamp: str | None = None,
    ) -> "RunContext":
        output_root = Path(output_root).expanduser().resolve() / stage
        timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

        if run_name_stem is not None:
            stem = run_name_stem
        elif model_name is not None:
            stem = model_name
        else:
            stem = stage

        if project_name is not None:
            base_run_name = f"{project_name}_{stem}_{timestamp}"
        else:
            base_run_name = f"{stem}_{timestamp}"

        run_name = base_run_name
        output_dir = output_root / run_name

        if output_dir.exists():
            suffix = uuid4().hex[:8]
            run_name = f"{base_run_name}_{suffix}"
            output_dir = output_root / run_name

        # Common dirs
        records_dir = output_dir / "records"
        config_dir = records_dir / "configs"
        log_dir = records_dir / "logs"
        metadata_dir = records_dir / "metadata"

        # Stage-specific dirs
        training_checkpoint_dir = output_dir / "checkpoints"
        training_architecture_dir = output_dir / "architecture"
        prediction_embeddings_dir = output_dir / "embeddings"
        prediction_reconstructions_dir = output_dir / "reconstructions"
        evaluation_artifacts_dir = output_dir / "artifacts"
        evaluation_figures_dir = output_dir / "figures"

        evaluation_metrics_dir = records_dir / "metrics"

        evaluation_embeddings_dir = evaluation_artifacts_dir / "embeddings"
        evaluation_reconstructions_dir = evaluation_artifacts_dir / "reconstructions"
        evaluation_embeddings_figures_dir = evaluation_figures_dir / "embeddings"
        evaluation_reconstructions_figures_dir = evaluation_figures_dir / "reconstructions"
        evaluation_reconstruction_inputs_dir = evaluation_reconstructions_dir / "inputs"
        evaluation_reconstruction_predictions_dir = evaluation_reconstructions_dir / "predictions"
        evaluation_reconstruction_error_maps_dir = evaluation_reconstructions_dir / "error_maps"

        dirs_to_create = [
            records_dir,
            config_dir,
            log_dir,
            metadata_dir,
        ]

        if stage == "training":
            dirs_to_create.extend([training_checkpoint_dir, training_architecture_dir])
        elif stage == "prediction":
            dirs_to_create.extend([prediction_embeddings_dir, prediction_reconstructions_dir])
        elif stage == "evaluation":
            dirs_to_create.extend([
                evaluation_artifacts_dir,
                evaluation_figures_dir,
                evaluation_metrics_dir,
                evaluation_embeddings_dir,
                evaluation_reconstructions_dir,
                evaluation_embeddings_figures_dir,
                evaluation_reconstructions_figures_dir,
                evaluation_reconstruction_inputs_dir,
                evaluation_reconstruction_predictions_dir,
                evaluation_reconstruction_error_maps_dir,
            ])
        else:
            raise ValueError(
                f"Unsupported run stage: {stage!r}. "
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
            training_checkpoint_dir=training_checkpoint_dir,
            training_architecture_dir=training_architecture_dir,
            prediction_embeddings_dir=prediction_embeddings_dir,
            prediction_reconstructions_dir=prediction_reconstructions_dir,
            evaluation_artifacts_dir=evaluation_artifacts_dir,
            evaluation_figures_dir=evaluation_figures_dir,
            evaluation_metrics_dir=evaluation_metrics_dir,
            evaluation_embeddings_dir=evaluation_embeddings_dir,
            evaluation_reconstructions_dir=evaluation_reconstructions_dir,
            evaluation_embeddings_figures_dir=evaluation_embeddings_figures_dir,
            evaluation_reconstructions_figures_dir=evaluation_reconstructions_figures_dir,
            evaluation_reconstruction_inputs_dir=evaluation_reconstruction_inputs_dir,
            evaluation_reconstruction_predictions_dir=evaluation_reconstruction_predictions_dir,
            evaluation_reconstruction_error_maps_dir=evaluation_reconstruction_error_maps_dir,
        )