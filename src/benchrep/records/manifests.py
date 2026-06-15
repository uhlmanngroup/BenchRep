from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lightning.pytorch.callbacks import ModelCheckpoint

from benchrep.assembly.schemas import TrainingConfig
from benchrep.assembly.resolvers import PredictionRunSpec
from benchrep.records.prediction_exports import PredictionExportPaths
from benchrep.runtime import RunContext
from benchrep.records.logs import (
    RUN_LOG_FILENAME,
    STDERR_LOG_FILENAME,
    STDOUT_LOG_FILENAME,
)


def write_yaml_record(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _optional_path_to_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def write_training_manifest(
    *,
    output_path: Path,
    config: TrainingConfig,
    run_context: RunContext,
    input_config_path: Path,
    checkpoint_callback: ModelCheckpoint,
    torchview_graph_path: Path | None = None,
    created_at: str,
    completed_at: str,
    status: str = "completed",
) -> None:
    summary = {
        "model": config.model.name,
        "encoder": config.encoder.name,
        "decoder": config.decoder.name if config.decoder is not None else None,
        "dataset": config.dataset.name,
        "transform": (
            config.dataset.transform.name
            if config.dataset.transform is not None
            else None
        ),
        "batch_size": config.datamodule.batch_size,
        "losses": {
            role: list(loss_terms)
            for role, loss_terms in config.losses.items()
        },
        "optimizer": config.optimizer.name,
        "seed": config.reproducibility.seed,
        "float32_matmul_precision": config.reproducibility.float32_matmul_precision,
        "trainer": {
            "max_epochs": getattr(config.trainer, "max_epochs", None),
            "deterministic": getattr(config.trainer, "deterministic", None),
        },
        "logger": {
            "name": config.logger.name if config.logger is not None else None,
        },
    }

    manifest = {
        "stage": config.stage,
        "status": status,
        "created_at": created_at,
        "completed_at": completed_at,
        "run": {
            "run_name": run_context.run_name,
            "output_dir": str(run_context.output_dir),
        },
        "records": {
            "input_config_path": str(input_config_path),
            "resolved_config_path": str(run_context.config_dir / "resolved_config.yaml"),
            "log_dir": str(run_context.log_dir),
            "metadata_dir": str(run_context.metadata_dir),
            "run_log_path": str(run_context.log_dir / RUN_LOG_FILENAME),
            "console_stderr_path": str(run_context.log_dir / STDERR_LOG_FILENAME),
            "console_stdout_path": (
                f"{run_context.log_dir / STDOUT_LOG_FILENAME} "
                "[optional; only written when stdout capture is enabled]"
            ),
            "architecture": {
                "torchview_graph_path": (
                    str(torchview_graph_path) if torchview_graph_path is not None else None
                ),
            },
        },
        "checkpoints": {
            "checkpoint_dir": str(run_context.training_checkpoint_dir),
            "monitor": config.checkpointing.monitor,
            "mode": config.checkpointing.mode,
            "best_checkpoint_path": checkpoint_callback.best_model_path or None,
            "best_checkpoint_score": (
                float(checkpoint_callback.best_model_score)
                if checkpoint_callback.best_model_score is not None
                else None
            ),
            "last_checkpoint_path": checkpoint_callback.last_model_path or None,
            "best_k_models": {
                path: float(score)
                for path, score in checkpoint_callback.best_k_models.items()
            },
        },
        "summary": summary,
    }

    write_yaml_record(manifest, output_path)


def write_prediction_manifest(
    *,
    output_path: Path,
    run_spec: PredictionRunSpec,
    run_context: RunContext,
    input_config_path: Path,
    export_paths: PredictionExportPaths,
    created_at: str,
    completed_at: str,
    status: str = "completed",
) -> None:
    training_summary = run_spec.training_manifest.get("summary", {})

    embedding_spec = run_spec.export_spec.embeddings
    reconstruction_spec = run_spec.export_spec.reconstructions
    embedding_export = export_paths.embedding_export
    reconstruction_paths = export_paths.reconstruction_paths

    summary = {
        "project_name": run_spec.training_config.run.project_name,
        "model": training_summary.get("model"),
        "encoder": training_summary.get("encoder"),
        "decoder": training_summary.get("decoder"),
        "dataset": run_spec.dataset_config.name,
        "data_split": run_spec.split,
        "batch_size": run_spec.batch_size,
        "max_batches": run_spec.max_batches,
    }

    manifest = {
        "stage": run_spec.stage,
        "status": status,
        "created_at": created_at,
        "completed_at": completed_at,
        "run": {
            "run_name": run_context.run_name,
            "output_dir": str(run_context.output_dir),
        },
        "source": {
            "training_manifest_path": str(run_spec.training_manifest_path),
            "training_run_name": run_spec.training_run_name,
            "training_output_dir": str(run_spec.training_output_dir),
            "resolved_training_config_path": str(run_spec.resolved_training_config_path),
            "checkpoint_path": str(run_spec.checkpoint_path),
        },
        "records": {
            "input_config_path": str(input_config_path),
            "resolved_config_path": str(run_context.config_dir / "resolved_config.yaml"),
            "log_dir": str(run_context.log_dir),
            "metadata_dir": str(run_context.metadata_dir),
            "run_log_path": str(run_context.log_dir / RUN_LOG_FILENAME),
            "console_stderr_path": str(run_context.log_dir / STDERR_LOG_FILENAME),
            "console_stdout_path": (
                f"{run_context.log_dir / STDOUT_LOG_FILENAME} "
                "[optional; only written when stdout capture is enabled]"
            ),
        },
        "exports": {
            "mode": run_spec.export_spec.mode,
            "embeddings": {
                "enabled": embedding_spec.enabled,
                "requested_keys": embedding_spec.keys,
                "requested_primary_key": embedding_spec.primary_key,
                "path": (
                    _optional_path_to_str(embedding_export.embeddings_h5ad_path)
                    if embedding_export is not None
                    else None
                ),
                "resolved_keys": (
                    embedding_export.resolved_keys
                    if embedding_export is not None
                    else None
                ),
                "resolved_primary_key": (
                    embedding_export.resolved_primary_key
                    if embedding_export is not None
                    else None
                ),
            },
            "reconstructions": {
                "enabled": reconstruction_spec.enabled,
                "n_examples_requested": reconstruction_spec.n_examples,
                "n_examples_exported": (
                    reconstruction_paths.n_examples_exported
                    if reconstruction_paths is not None
                    else None
                ),
                "selection": reconstruction_spec.selection,
                "seed": reconstruction_spec.seed,
                "include_input": reconstruction_spec.include_input,
                "include_prediction": reconstruction_spec.include_prediction,
                "paths": {
                    "input": (
                        _optional_path_to_str(reconstruction_paths.input_path)
                        if reconstruction_paths is not None
                        else None
                    ),
                    "reconstruction": (
                        _optional_path_to_str(reconstruction_paths.reconstruction_path)
                        if reconstruction_paths is not None
                        else None
                    ),
                    "obs": (
                        _optional_path_to_str(reconstruction_paths.obs_path)
                        if reconstruction_paths is not None
                        else None
                    ),
                    "metadata": (
                        _optional_path_to_str(reconstruction_paths.metadata_path)
                        if reconstruction_paths is not None
                        else None
                    ),
                },
            },
        },
        "summary": summary,
    }

    write_yaml_record(manifest, output_path)