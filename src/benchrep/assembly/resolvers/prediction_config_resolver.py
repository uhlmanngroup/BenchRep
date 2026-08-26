from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchrep.assembly.config import load_yaml
from benchrep.assembly.schemas import (
    PredictionTransformConfig,
    PredictionConfig,
    TrainingConfig,
    PredictionReconstructionsExportConfig,
    PredictionExportConfig,
    parse_training_config,
    SupportedDatasetConfig,
    TrainingDataModuleConfig,
    TrainingTrainerConfig,
)
from benchrep.assembly.resolvers.utils import (
    resolve_optional,
    get_required_nested_path,
    get_required_nested_str,
)
from benchrep.assembly.registries.utils import normalize_name
from benchrep.interfaces.model_families import (
    ModelFamilySpec,
    VAE_FAMILY,
    model_family_supports_reconstruction,
)
from benchrep.runtime.status.training import ACCEPTABLE_TRAINING_STATUSES


# -------------------------
# Resolved specs
# -------------------------
PredictionCheckpointSource = Literal[
    "training_manifest_best",
    "training_manifest_last",
    "training_checkpoint_filename",
    "explicit_path",
]

PredictionTransformSource = Literal[
    "training_config",
    "prediction_config",
    "default_identity",
    "external_datamodule",
]

ReconstructionLatentSource = Literal["mean", "sample"]


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
    checkpoint_source: PredictionCheckpointSource

    training_run_name: str
    training_output_dir: Path

    dataset_config: SupportedDatasetConfig | None
    transform_configs: tuple[PredictionTransformConfig, ...] | None
    transform_source: PredictionTransformSource
    datamodule_config: TrainingDataModuleConfig | None
    batch_size: int | None
    num_workers: int | None
    trainer_config: TrainingTrainerConfig
    max_batches: int | None

    seed: int | None
    seed_workers: bool
    float32_matmul_precision: Literal["medium", "high", "highest"]
    reconstruction_latent_source: ReconstructionLatentSource | None

    export_spec: PredictionExportSpec


def resolve_prediction_config(
    prediction_config: PredictionConfig,
    model_family: ModelFamilySpec,
    training_manifest_path_override: Path | str | None = None,
    model_overridden: bool = False,
    datamodule_overridden: bool = False,
) -> PredictionRunSpec:
    """Resolve prediction configuration against its linked training run.

      Loads and validates the training manifest and resolved training config,
      verifies model provenance and family compatibility, resolves the checkpoint
      and prediction data settings, inherits applicable runtime settings, and
      returns the complete runtime specification used by the prediction workflow.
      """
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

    _validate_prediction_model_source(
        training_model_external=training_model_external,
        model_overridden=model_overridden,
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

    _validate_prediction_model_family(
        training_model_provenance=training_model_provenance,
        training_config=training_config,
        model_family=model_family,
        model_overridden=model_overridden,
    )

    checkpoint_path, checkpoint_source = _resolve_checkpoint_path(
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

    if datamodule_overridden:
        dataset_config = None
        transform_configs = None
        transform_source: PredictionTransformSource = "external_datamodule"
        datamodule_config = None
        batch_size = None
        num_workers = None

    else:
        if prediction_config.dataset is not None:
            dataset_config = prediction_config.dataset
        elif training_datamodule_external:
            dataset_config = None
        else:
            dataset_config = training_config.dataset

        if dataset_config is None:
            raise ValueError(
                "Prediction requires a dataset configuration, but none was "
                "provided in the prediction config or reconstructable from the "
                "training config. Pass `dataset` in the prediction config or "
                "provide a datamodule override to the prediction entrypoint."
            )

        transform_configs, transform_source = _resolve_prediction_transforms(
            prediction_config=prediction_config,
            training_config=training_config,
            training_datamodule_external=training_datamodule_external,
        )

        base_datamodule_config = (
            training_config.datamodule
            if (
                    not training_datamodule_external
                    and training_config.datamodule is not None
            )
            else TrainingDataModuleConfig()
        )

        batch_size = resolve_optional(
            prediction_config.data.batch_size,
            base_datamodule_config.batch_size,
            field_name="data.batch_size",
        )

        num_workers = resolve_optional(
            prediction_config.data.num_workers,
            base_datamodule_config.num_workers,
            field_name="data.num_workers",
        )

        datamodule_config = base_datamodule_config.model_copy(
            update={
                "batch_size": batch_size,
                "num_workers": num_workers,
                "val_fraction": 0.0,
                "persistent_workers": (
                    base_datamodule_config.persistent_workers
                    if num_workers > 0
                    else False
                ),
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

    deterministic = (
        prediction_config.inference.deterministic
        if prediction_config.inference.deterministic is not None
        else training_config.trainer.deterministic
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

    reconstruction_latent_source = _resolve_reconstruction_latent_source(
        configured_source=(
            prediction_config.inference.reconstruction_latent_source
        ),
        model_family=model_family,
        model_overridden=model_overridden,
    )

    export_spec = resolve_prediction_exports(
        export_config=prediction_config.exports,
        seed=seed,
        model_family=model_family,
    )

    return PredictionRunSpec(
        stage=prediction_config.stage,
        prediction_config=prediction_config,
        training_config=training_config,
        training_manifest=training_manifest,
        training_manifest_path=training_manifest_path,
        resolved_training_config_path=resolved_training_config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_source=checkpoint_source,
        training_run_name=training_run_name,
        training_output_dir=training_output_dir,
        dataset_config=dataset_config,
        transform_configs=transform_configs,
        transform_source=transform_source,
        datamodule_config=datamodule_config,
        batch_size=batch_size,
        num_workers=num_workers,
        trainer_config=trainer_config,
        max_batches=prediction_config.data.max_batches,
        seed=seed,
        seed_workers=seed_workers,
        float32_matmul_precision=float32_matmul_precision,
        reconstruction_latent_source=reconstruction_latent_source,
        export_spec=export_spec,
    )


def _resolve_reconstruction_export_enabled(
    *,
    reconstruction_config: PredictionReconstructionsExportConfig,
    model_family: ModelFamilySpec,
) -> bool:
    """Resolve tri-state reconstruction export against model-family support.

    An explicit true value enables export and errors if the model family does
    not declare returned reconstructions. An explicit false value disables export.
    A null value enables export automatically only for model families whose
    expected prediction output contains a reconstruction field.
    """
    supports_reconstruction = model_family_supports_reconstruction(
        model_family
    )

    if reconstruction_config.enabled is True and not supports_reconstruction:
        raise ValueError(
            f"Model family {model_family.name!r} does not support "
            "reconstruction export. Set "
            "`exports.reconstructions.enabled=False` or null."
        )

    if reconstruction_config.enabled is None:
        return supports_reconstruction

    return reconstruction_config.enabled


def resolve_prediction_exports(
    *,
    export_config: PredictionExportConfig,
    seed: int | None,
    model_family: ModelFamilySpec,
) -> PredictionExportSpec:
    """Resolve prediction export settings into a runtime export spec.

    This resolves config-level export intent only. Actual output-key validation
    is done later by the exporter after ``trainer.predict()`` has produced model
    outputs.

    ``seed`` is the already-resolved prediction/inference seed. It should already
    reflect the prediction config seed if provided, otherwise the training run seed.
    """

    reconstruction_config = export_config.reconstructions
    reconstruction_enabled = _resolve_reconstruction_export_enabled(
        reconstruction_config=reconstruction_config,
        model_family=model_family,
    )

    if reconstruction_enabled:
        if (
            not reconstruction_config.include_input
            and not reconstruction_config.include_prediction
        ):
            raise ValueError(
                "Enabled reconstruction export requires at least one of "
                "`include_input` or `include_prediction` to be true."
            )

        if (
            reconstruction_config.n_examples != "all"
            and reconstruction_config.stratify_by is not None
            and reconstruction_config.selection != "random"
        ):
            raise ValueError(
                "`exports.reconstructions.selection` must be 'random' when "
                "stratifying a reconstruction subset."
            )

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
        reconstruction_config.seed
        if reconstruction_config.seed is not None
        else seed
    )

    if (
            reconstruction_enabled
            and reconstruction_config.n_examples != "all"
            and reconstruction_config.selection == "random"
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
            enabled=reconstruction_enabled,
            n_examples=reconstruction_config.n_examples,
            selection=reconstruction_config.selection,
            stratify_by=reconstruction_config.stratify_by,
            seed=reconstruction_seed,
            include_input=reconstruction_config.include_input,
            include_prediction=reconstruction_config.include_prediction,
        ),
    )


def _resolve_checkpoint_path(
    *,
    checkpoint: Literal["best", "last"] | Path,
    training_manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[Path, PredictionCheckpointSource]:
    """Resolve a checkpoint selection and record how it was selected.

    Supports the training run's best or last checkpoint, a bare filename within
    its checkpoint directory, or an explicit absolute checkpoint path.
    """

    if checkpoint == "best":
        checkpoint_path = get_required_nested_path(
            training_manifest,
            "checkpoints",
            "best_checkpoint_path",
            base_dir=manifest_path.parent,
        )
        checkpoint_source: PredictionCheckpointSource = (
            "training_manifest_best"
        )

    elif checkpoint == "last":
        checkpoint_path = get_required_nested_path(
            training_manifest,
            "checkpoints",
            "last_checkpoint_path",
            base_dir=manifest_path.parent,
        )
        checkpoint_source = "training_manifest_last"

    else:
        assert isinstance(checkpoint, Path)

        checkpoint = checkpoint.expanduser()

        if checkpoint.is_absolute():
            checkpoint_path = checkpoint.resolve()
            checkpoint_source = "explicit_path"

        else:
            if checkpoint.parent != Path("."):
                raise ValueError(
                    "Relative checkpoint selections must be bare filenames, "
                    f"got: {checkpoint}"
                )

            checkpoint_dir = get_required_nested_path(
                training_manifest,
                "checkpoints",
                "checkpoint_dir",
                base_dir=manifest_path.parent,
            )
            checkpoint_path = (checkpoint_dir / checkpoint).resolve()
            checkpoint_source = "training_checkpoint_filename"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file does not exist: {checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise ValueError(
            f"Checkpoint path must point to a file, got: {checkpoint_path}"
        )

    if checkpoint_path.suffix != ".ckpt":
        raise ValueError(
            f"Checkpoint file must end with '.ckpt', got: {checkpoint_path}"
        )

    return checkpoint_path, checkpoint_source


def _resolve_prediction_transforms(
    *,
    prediction_config: PredictionConfig,
    training_config: TrainingConfig,
    training_datamodule_external: bool,
) -> tuple[
    tuple[PredictionTransformConfig, ...],
    PredictionTransformSource,
]:
    """Resolve the ordered transforms applied during prediction."""
    if prediction_config.transforms is not None:
        return tuple(prediction_config.transforms), "prediction_config"

    if training_datamodule_external:
        return (), "default_identity"

    inherited_transforms = tuple(
        PredictionTransformConfig.model_validate(
            transform.model_dump(
                mode="python",
                exclude={"apply_to"},
            )
        )
        for transform in training_config.transforms
        if "validation" in transform.apply_to
    )

    return inherited_transforms, "training_config"


def _resolve_training_manifest_path(
    *,
    prediction_config: PredictionConfig,
    training_manifest_path_override: Path | str | None,
) -> tuple[PredictionConfig, Path]:
    """Resolve and validate the effective training-manifest path.

    An entrypoint-provided path overrides the path stored in the prediction
    config and is written into the returned config copy.
    """

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

    if manifest_status not in ACCEPTABLE_TRAINING_STATUSES:
        raise ValueError(
            "Prediction requires a training manifest with an acceptable status "
            f"status in {sorted(ACCEPTABLE_TRAINING_STATUSES)}, "
            f"but manifest status is {manifest_status!r}."
        )

    return training_manifest


def _validate_prediction_model_source(
    *,
    training_model_external: bool,
    model_overridden: bool,
) -> None:
    if training_model_external and not model_overridden:
        raise ValueError(
            "Training manifest indicates that the trained model came from an external "
            "Python object, but no model override was provided to the prediction entrypoint. "
            "Pass a compatible model instance that can load the recorded checkpoint."
        )


def _validate_prediction_model_family(
    *,
    training_model_provenance: dict[str, Any],
    training_config: TrainingConfig,
    model_family: ModelFamilySpec,
    model_overridden: bool,
) -> None:
    recorded_family = training_model_provenance.get("family")

    if recorded_family is not None and recorded_family != model_family.name:
        raise ValueError(
            "Prediction model family does not match the training run: "
            f"training family={recorded_family!r}, "
            f"prediction family={model_family.name!r}."
        )

    if model_overridden:
        return

    assert training_config.model is not None

    configured_model_name = normalize_name(
        training_config.model.name,
        field_name="config.model.name",
    )

    if configured_model_name not in model_family.config_model_names:
        raise ValueError(
            "Configured model is incompatible with the selected prediction "
            "model family: "
            f"family={model_family.name!r}, "
            f"configured_model={configured_model_name!r}, "
            f"expected one of {model_family.config_model_names!r}."
        )


def _resolve_reconstruction_latent_source(
    *,
    configured_source: ReconstructionLatentSource | None,
    model_family: ModelFamilySpec,
    model_overridden: bool,
) -> ReconstructionLatentSource | None:
    """Resolve the latent source used for prediction-time VAE reconstruction."""
    if configured_source is None:
        if model_overridden:
            return None

        if model_family == VAE_FAMILY:
            return "mean"

        return None

    if model_overridden:
        if model_family == VAE_FAMILY:
            raise ValueError(
                "`inference.reconstruction_latent_source` cannot be set when "
                "prediction uses an external model. BenchRep cannot control how "
                "the external VAE model's `predict_step()` produces reconstructions."
            )

    if model_family != VAE_FAMILY:
        raise ValueError(
            "`inference.reconstruction_latent_source` is only supported for "
            "prediction using internal VAE models."
        )

    return configured_source