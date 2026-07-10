from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchrep.assembly.config import load_yaml
from benchrep.assembly.schemas import (
    PredictionConfig,
    TrainingConfig,
    PredictionExportConfig,
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


# -------------------------
# Resolved specs
# -------------------------
@dataclass(frozen=True)
class PredictionEmbeddingsExportSpec:
    enabled: bool
    keys: list[str] | Literal["auto", "all"]
    primary_key: str


@dataclass(frozen=True)
class PredictionReconstructionsExportSpec:
    enabled: bool
    n_examples: int | Literal["all"]
    selection: Literal["first", "random"]
    stratify_by: str | None
    seed: int | None
    include_input: bool
    include_prediction: bool


@dataclass(frozen=True)
class PredictionExportSpec:
    mode: Literal["standard", "all", "custom"]
    embeddings: PredictionEmbeddingsExportSpec
    reconstructions: PredictionReconstructionsExportSpec


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

    dataset_config: DatasetConfig | None
    datamodule_config: DataModuleConfig | None
    split: str
    batch_size: int | None
    num_workers: int | None
    trainer_config: TrainerConfig
    max_batches: int | None

    seed: int | None
    seed_workers: bool | None
    float32_matmul_precision: str | None

    export_spec: PredictionExportSpec


def resolve_prediction_config(
    prediction_config: PredictionConfig,
    training_manifest_path_override: Path | str | None = None,
    model_overridden: bool = False,
    datamodule_overridden: bool = False,
) -> PredictionRunSpec:
    """Resolve prediction config values that depend on the training run."""
    prediction_config, training_manifest_path = _resolve_training_manifest_path(
        prediction_config=prediction_config,
        training_manifest_path_override=training_manifest_path_override,
    )

    training_manifest = _load_training_manifest(training_manifest_path)

    training_provenance = training_manifest.get("provenance", {})
    training_model_provenance = training_provenance.get("model", {})
    training_datamodule_provenance = training_provenance.get("datamodule", {})

    training_model_external = training_model_provenance.get("source") != "config"
    training_datamodule_external = (
        training_datamodule_provenance.get("source") != "config"
    )

    resolved_training_config_path = get_required_nested_path(
        training_manifest,
        "records",
        "resolved_config_path",
        base_dir=training_manifest_path.parent,
    )

    raw_training_config = load_yaml(resolved_training_config_path)
    training_config = parse_training_config(
        raw_training_config,
        model_overridden=training_model_external,
        datamodule_overridden=training_datamodule_external,
    )

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

    if training_model_external and not model_overridden:
        raise ValueError(
            "Training manifest indicates that the trained model came from an external "
            "Python object, but no model override was provided to predict(). "
            "Pass a compatible model instance that can load the recorded checkpoint."
        )

    if datamodule_overridden:
        dataset_config = None
        datamodule_config = None
        batch_size = None
        num_workers = None
    else:
        if training_config.dataset is None or training_config.datamodule is None:
            raise ValueError(
                "Prediction requires a config-reconstructable dataset/datamodule, "
                "but the training config does not contain `dataset` and `datamodule`. "
                "Pass a datamodule override to predict()."
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

        datamodule_config = training_config.datamodule.model_copy(
            update={
                "batch_size": batch_size,
                "num_workers": num_workers,
                "drop_last": False,
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

    export_spec = resolve_prediction_exports(
        export_config=prediction_config.exports,
        seed=seed,
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
        export_spec=export_spec,
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


def resolve_prediction_exports(
    *,
    export_config: PredictionExportConfig,
    seed: int | None,
) -> PredictionExportSpec:
    """Resolve prediction export settings into a runtime export spec.

    This resolves config-level export intent only. Actual output-key validation
    is done later by the exporter after ``trainer.predict()`` has produced model
    outputs.

    ``seed`` is the already-resolved prediction/inference seed. It should already
    reflect the prediction config seed if provided, otherwise the training run seed.
    """
    mode = export_config.mode

    if mode == "standard":
        embedding_keys: list[str] | Literal["auto", "all"] = "auto"
        primary_key = "auto"

    elif mode == "all":
        embedding_keys = "all"
        primary_key = "auto"

    elif mode == "custom":
        embedding_keys = export_config.embeddings.keys
        primary_key = export_config.embeddings.primary_key

    else:
        raise ValueError(
            f"Unsupported prediction export mode {mode!r}. "
            "Available options: 'standard', 'all', 'custom'."
        )

    reconstruction_seed = (
        export_config.reconstructions.seed
        if export_config.reconstructions.seed is not None
        else seed
    )

    if (
        export_config.reconstructions.enabled
        and export_config.reconstructions.selection == "random"
        and reconstruction_seed is None
    ):
        raise ValueError(
            "Random reconstruction export requires a seed. Set "
            "`exports.reconstructions.seed`, `inference.seed`, or use a training "
            "run with a reproducibility seed."
        )

    return PredictionExportSpec(
        mode=mode,
        embeddings=PredictionEmbeddingsExportSpec(
            enabled=export_config.embeddings.enabled,
            keys=embedding_keys,
            primary_key=primary_key,
        ),
        reconstructions=PredictionReconstructionsExportSpec(
            enabled=export_config.reconstructions.enabled,
            n_examples=export_config.reconstructions.n_examples,
            selection=export_config.reconstructions.selection,
            stratify_by=export_config.reconstructions.stratify_by,
            seed=reconstruction_seed,
            include_input=export_config.reconstructions.include_input,
            include_prediction=export_config.reconstructions.include_prediction,
        ),
    )


def _resolve_training_manifest_path(
    *,
    prediction_config: PredictionConfig,
    training_manifest_path_override: Path | str | None,
) -> tuple[PredictionConfig, Path]:
    if training_manifest_path_override is not None:
        training_manifest_path = Path(training_manifest_path_override).resolve()

        prediction_config = prediction_config.model_copy(
            update={
                "source": prediction_config.source.model_copy(
                    update={"training_manifest_path": training_manifest_path}
                )
            }
        )
    else:
        training_manifest_path = prediction_config.source.training_manifest_path.resolve()

    if training_manifest_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            "source.training_manifest_path must point to a YAML file. "
            f"Got: {training_manifest_path}"
        )

    if not training_manifest_path.is_file():
        raise FileNotFoundError(
            f"Training manifest file does not exist: {training_manifest_path}"
        )

    return prediction_config, training_manifest_path


def _load_training_manifest(path: Path) -> dict[str, Any]:
    training_manifest = load_yaml(path)

    if not isinstance(training_manifest, dict):
        raise TypeError(
            "Training manifest must load as a mapping, "
            f"got {type(training_manifest).__name__}."
        )

    manifest_stage = training_manifest.get("stage")
    if manifest_stage != "training":
        raise ValueError(
            "Prediction requires a training manifest, "
            f"but manifest stage is {manifest_stage!r}."
        )

    manifest_status = training_manifest.get("status")
    if manifest_status != "completed":
        raise ValueError(
            "Prediction requires a completed training manifest, "
            f"but manifest status is {manifest_status!r}."
        )

    return training_manifest