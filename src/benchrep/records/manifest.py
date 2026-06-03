from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lightning.pytorch.callbacks import ModelCheckpoint

from benchrep.assembly.schemas import TrainingConfig
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


def write_training_manifest(
    *,
    output_path: Path,
    config: TrainingConfig,
    run_context: RunContext,
    input_config_path: Path,
    checkpoint_callback: ModelCheckpoint,
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
        },
        "checkpoints": {
            "checkpoint_dir": str(run_context.checkpoint_dir),
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