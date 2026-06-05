from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchrep.assembly.config import load_yaml
from benchrep.assembly.schemas import (
    PredictionConfig,
    TrainingConfig,
    parse_training_config,
    DatasetConfig,
    DataModuleConfig,
    TrainerConfig,
)
from benchrep.assembly.resolvers.utils import (
    resolve_optional,
    get_required_nested_path,
    get_required_nested_str,
)


@dataclass(frozen=True)
class PredictionRunSpec:
    stage: Literal["prediction"]
    prediction_config: PredictionConfig
    training_config: TrainingConfig
    training_manifest: dict[str, Any]

    training_manifest_path: Path
    resolved_training_config_path: Path
    checkpoint_path: Path

    training_run_name: str
    training_output_dir: Path

    dataset_config: DatasetConfig
    datamodule_config: DataModuleConfig
    split: str
    batch_size: int
    num_workers: int
    trainer_config: TrainerConfig
    max_batches: int | None

    seed: int | None
    seed_workers: bool | None
    float32_matmul_precision: str | None


def resolve_prediction_config(
    prediction_config: PredictionConfig,
) -> PredictionRunSpec:
    """Resolve prediction config values that depend on the training run."""
    training_manifest_path = prediction_config.source.training_manifest_path.resolve()
    training_manifest = load_yaml(training_manifest_path)

    resolved_training_config_path = get_required_nested_path(
        training_manifest,
        "records",
        "resolved_config_path",
        base_dir=training_manifest_path.parent,
    )

    raw_training_config = load_yaml(resolved_training_config_path)
    training_config = parse_training_config(raw_training_config)

    checkpoint_path = _resolve_checkpoint_path(
        checkpoint=prediction_config.source.checkpoint,
        training_manifest=training_manifest,
        manifest_path=training_manifest_path,
    )

    training_run_name = get_required_nested_str(
        training_manifest,
        "run",
        "run_name",
    )

    training_output_dir = get_required_nested_path(
        training_manifest,
        "run",
        "output_dir",
        base_dir=training_manifest_path.parent,
    )

    batch_size = resolve_optional(
        prediction_config.data.batch_size,
        training_config.datamodule.batch_size,
        field_name="data.batch_size",
    )

    num_workers = resolve_optional(
        prediction_config.data.num_workers,
        training_config.datamodule.num_workers,
        field_name="data.num_workers",
    )

    dataset_config = training_config.dataset

    # Override DataModule config params from prediction config
    datamodule_config = training_config.datamodule.model_copy(
        update={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "drop_last": False, # never drop_last in prediction
        }
    )

    seed = resolve_optional(
        prediction_config.inference.seed,
        training_config.reproducibility.seed,
        field_name="inference.seed",
    )

    seed_workers = resolve_optional(
        prediction_config.inference.seed_workers,
        training_config.reproducibility.seed_workers,
        field_name="inference.seed_workers",
    )

    deterministic = resolve_optional(
        prediction_config.inference.deterministic,
        training_config.trainer.deterministic,
        field_name="inference.deterministic",
    )

    trainer_config = training_config.trainer.model_copy(
        update={
            "deterministic": deterministic,
        }
    )

    float32_matmul_precision = resolve_optional(
        prediction_config.inference.float32_matmul_precision,
        training_config.reproducibility.float32_matmul_precision,
        field_name="inference.float32_matmul_precision",
    )

    return PredictionRunSpec(
        stage=prediction_config.stage,
        prediction_config=prediction_config,
        training_config=training_config,
        training_manifest=training_manifest,
        training_manifest_path=training_manifest_path,
        resolved_training_config_path=resolved_training_config_path,
        checkpoint_path=checkpoint_path,
        training_run_name=training_run_name,
        training_output_dir=training_output_dir,
        dataset_config=dataset_config,
        datamodule_config=datamodule_config,
        split=prediction_config.data.split,
        batch_size=batch_size,
        num_workers=num_workers,
        trainer_config=trainer_config,
        max_batches=prediction_config.data.max_batches,
        seed=seed,
        seed_workers=seed_workers,
        float32_matmul_precision=float32_matmul_precision,
    )


def _resolve_checkpoint_path(
    *,
    checkpoint: str,
    training_manifest: dict[str, Any],
    manifest_path: Path,
) -> Path:
    if checkpoint == "best":
        checkpoint_path = get_required_nested_path(
            training_manifest,
            "checkpoints",
            "best_checkpoint_path",
            base_dir=manifest_path.parent,
        )
    elif checkpoint == "last":
        checkpoint_path = get_required_nested_path(
            training_manifest,
            "checkpoints",
            "last_checkpoint_path",
            base_dir=manifest_path.parent,
        )
    else:
        checkpoint_name = Path(checkpoint)

        if checkpoint_name.is_absolute() or checkpoint_name.parent != Path("."):
            raise ValueError(
                "Explicit checkpoint selection must be a checkpoint filename from "
                "the training checkpoint directory, not an absolute or relative path. "
                f"Got: {checkpoint!r}"
            )

        checkpoint_dir = get_required_nested_path(
            training_manifest,
            "checkpoints",
            "checkpoint_dir",
            base_dir=manifest_path.parent,
        )
        checkpoint_path = (checkpoint_dir / checkpoint_name).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path must point to a file, got: {checkpoint_path}")

    if checkpoint_path.suffix != ".ckpt":
        raise ValueError(f"Checkpoint file must end with '.ckpt', got: {checkpoint_path}")

    return checkpoint_path
