from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchrep.assembly.config import load_yaml
from benchrep.assembly.schemas import (
    PredictionTransformPipelineConfig,
    PredictionTransformStepConfig,
    PredictionConfig,
    TrainingConfig,
    CompositeModelAssemblyInputOverrideConfig,
    PredictionAnnDataExportConfig,
    PredictionReconstructionsExportConfig,
    PredictionExportConfig,
    parse_training_config,
    TrainingConfigContext,
    SupportedDatasetConfig,
    TrainingDataModuleConfig,
    TrainingTrainerConfig,
    CompositeModelDeclarationsConfig,
    TrainingReproducibilityConfig,
    TrainingRunConfig,
)
from benchrep.assembly.schemas.training_config_schema import Float32MatmulPrecision
from benchrep.assembly.schemas.composite_model_config_schema import (
    CompositeModelAssemblyStepConfig,
)
from benchrep.assembly.registries.core import MODELS
from benchrep.assembly.registries.utils import normalize_path
from benchrep.assembly.resolvers.loss_resolver import (
    LossSpec,
    resolve_canonical_loss_configs,
    resolve_composite_loss_configs,

)
from benchrep.assembly.resolvers.composite_model_resolver import (
    CompositeModelSpec,
    resolve_composite_model_config,
)
from benchrep.assembly.resolvers.utils import (
    resolve_optional,
    get_required_nested_path,
    get_required_nested_str,
    ComponentSource,
    RunIdentitySpec,
    resolve_runtime_override_config,
    get_optional_nested_value,
)
from benchrep.architecture.data.datamodule import validate_dataloader_kwargs
from benchrep.architecture.composite_model_roles import (
    TENSOR_STRUCTURE_BY_ROLE,
)
from benchrep.interfaces.model_families import (
    CanonicalModelFamilySpec,
    ModelFamilySpec,
    VAE_FAMILY,
    COMPOSITE_FAMILY,
    YOYO_FAMILY,
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
    "runtime_model"
]

PredictionTransformPipelineSource = Literal[
    "training_config",
    "prediction_config",
    "default_identity",
    "external_datamodule",
]

CanonicalVAEReconstructionLatentSource = Literal["mean", "sample"]

PredictionInheritableField = Literal[
    "dataset",
    "transform_pipelines",
    "data.batch_size",
    "data.num_workers",
    "inference.seed",
    "inference.seed_workers",
    "inference.deterministic",
    "inference.float32_matmul_precision",
    "overrides.model",
    "exports.reconstructions.seed",
]


PredictionAnnDataObservationSource = Literal[
    "model_input",
    "model_output",
    "batch_metadata",
]

PredictionReconstructionObservationSource = Literal[
    "model_input",
    "batch_metadata",
]


@dataclass(frozen=True)
class PredictionAnnDataObservationSpec:
    name: str
    source: PredictionAnnDataObservationSource
    use_as_index: bool = False


@dataclass(frozen=True)
class PredictionReconstructionObservationSpec:
    name: str
    source: PredictionReconstructionObservationSource


@dataclass(frozen=True)
class PredictionAnnDataExportSpec:
    """Resolved AnnData selections and any declared output structures.

    Enabled YOYO exports leave ``output_structures_by_key`` empty here.
    Their scalar/vector structures are determined from returned tensors.
    """
    enabled: bool
    mode: Literal["all", "custom"]
    keys: tuple[str, ...]
    output_structures_by_key: dict[
        str,
        Literal["scalar", "vector"],
    ]
    primary_key: str | None
    observations: tuple[PredictionAnnDataObservationSpec, ...]


@dataclass(frozen=True)
class PredictionReconstructionPairSpec:
    id: str
    input: str
    reconstruction: str


@dataclass(frozen=True)
class PredictionReconstructionsExportSpec:
    enabled: bool
    mode: Literal["all", "custom"]
    pairs: tuple[PredictionReconstructionPairSpec, ...]
    observations: tuple[PredictionReconstructionObservationSpec, ...]
    n_examples: int | Literal["all"]
    selection: Literal["first", "random"]
    stratify_by: str | None
    seed: int | None
    include_input: bool
    include_reconstruction: bool


@dataclass(frozen=True)
class PredictionExportSpec:
    anndata: PredictionAnnDataExportSpec
    reconstructions: PredictionReconstructionsExportSpec


@dataclass(frozen=True)
class PredictionRunSpec:
    stage: Literal["prediction"]
    model_family: ModelFamilySpec
    model_source: ComponentSource
    datamodule_source: ComponentSource
    compatibility_policy: Literal["error", "warn"]
    run_identity: RunIdentitySpec
    inherited_config_fields: frozenset[PredictionInheritableField]
    prediction_config: PredictionConfig
    training_config: TrainingConfig | None
    loss_specs: tuple[LossSpec, ...]
    composite_model_spec: CompositeModelSpec | None
    training_manifest: dict[str, Any] | None

    training_manifest_path: Path | None
    resolved_training_config_path: Path | None
    checkpoint_selection: Literal["best", "last"] | Path | None
    checkpoint_path: Path | None
    checkpoint_source: PredictionCheckpointSource

    training_run_name: str | None
    training_output_dir: Path | None

    dataset_config: SupportedDatasetConfig | None
    transform_pipeline_configs: tuple[PredictionTransformPipelineConfig, ...] | None
    transform_pipeline_source: PredictionTransformPipelineSource
    datamodule_config: TrainingDataModuleConfig | None
    batch_size: int | None
    num_workers: int | None
    trainer_config: TrainingTrainerConfig
    max_batches: int | None

    seed: int | None
    seed_workers: bool
    float32_matmul_precision: Float32MatmulPrecision
    canonical_vae_reconstruction_latent_source: CanonicalVAEReconstructionLatentSource | None

    export_spec: PredictionExportSpec


def resolve_prediction_config(
    prediction_config: PredictionConfig,
    model_family: ModelFamilySpec,
    training_manifest_path_override: Path | str | None = None,
    model_source: ComponentSource = "config",
    datamodule_source: ComponentSource = "config",
    model_override_name: str | None = None,
    compatibility_policy: Literal["error", "warn"] = "error",
) -> PredictionRunSpec:
    """Resolve the model, weights, data, and runtime settings for prediction.

    Linked runs validate the training manifest and inherit applicable settings
    from its recorded configuration. Standalone prediction for YOYO models uses
    a supplied model and resolves settings without a training record.

    Without a training record, run, Trainer, and reproducibility fallbacks use
    their standard training-schema defaults. These defaults are not recorded
    as inherited settings.
    """

    if compatibility_policy not in {"error", "warn"}:
        raise ValueError(
            "`compatibility_policy` must be either 'error' or 'warn'."
        )

    model_is_external = model_source != "config"
    datamodule_is_external = datamodule_source != "config"

    if model_family == YOYO_FAMILY and not model_is_external:
        raise ValueError(
            "YOYO models require a model class or instance supplied through "
            "`PredictionOverrides.model`."
        )

    if prediction_config.source.checkpoint is None and model_family != YOYO_FAMILY:
        raise ValueError(
            "`source.checkpoint` may be null only for YOYO models."
        )

    # Record inheritance before materialization replaces omitted config values.
    inherited_config_fields: set[PredictionInheritableField] = set()

    prediction_config, training_manifest_path = _resolve_training_manifest_path(
        prediction_config=prediction_config,
        training_manifest_path_override=training_manifest_path_override,
        model_family=model_family,
    )

    if training_manifest_path is None:
        training_manifest = None
        training_config = None
        resolved_training_config_path = None
        training_run_name = None
        training_output_dir = None
        training_datamodule_external = False
    else:
        training_manifest = _load_training_manifest(training_manifest_path)

        training_construction = training_manifest.get("construction")

        if not isinstance(training_construction, dict):
            raise TypeError(
                "Training manifest field `construction` must be a mapping."
            )

        training_model_construction = training_construction.get("model")
        training_datamodule_construction = training_construction.get(
            "datamodule"
        )

        if not isinstance(training_model_construction, dict):
            raise TypeError(
                "Training manifest field `construction.model` must be a mapping."
            )

        if not isinstance(training_datamodule_construction, dict):
            raise TypeError(
                "Training manifest field `construction.datamodule` must be a "
                "mapping."
            )

        training_model_external = (
            training_model_construction.get("source") != "config"
        )
        training_datamodule_external = (
            training_datamodule_construction.get("source") != "config"
        )

        _validate_prediction_model_source(
            training_model_external=training_model_external,
            model_is_external=model_is_external,
        )

        # Retained only as an informational record path. The embedded configuration
        # below is the source consumed by prediction.
        resolved_training_config_path = get_required_nested_path(
            training_manifest,
            "records",
            "resolved_config_path",
            base_dir=training_manifest_path.parent,
        )

        training_appendix = training_manifest.get("appendix")

        if not isinstance(training_appendix, dict):
            raise TypeError(
                "Training manifest field `appendix` must be a mapping."
            )

        raw_training_config = training_appendix.get("resolved_config")

        if not isinstance(raw_training_config, dict):
            raise TypeError(
                "Training manifest field `appendix.resolved_config` must be a "
                "mapping."
            )

        training_config = parse_training_config(
            raw_training_config,
            context=TrainingConfigContext(
                external_model=training_model_external,
                external_datamodule=training_datamodule_external,
            ),
        )

        _validate_prediction_model_family(
            training_model_construction=training_model_construction,
            training_config=training_config,
            model_family=model_family,
            model_is_external=model_is_external,
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

    composite_model_spec = _resolve_prediction_composite_model_spec(
        prediction_config=prediction_config,
        training_config=training_config,
        model_family=model_family,
        model_is_external=model_is_external,
    )

    loss_specs: tuple[LossSpec, ...] = ()

    if not model_is_external:
        assert training_config is not None
        assert training_config.losses is not None

        if composite_model_spec is not None:
            loss_specs = resolve_composite_loss_configs(
                training_config.losses,
                model_input_roles_by_name=(
                    composite_model_spec.declarations
                    .model_input_roles_by_name
                ),
                model_output_roles_by_name=(
                    composite_model_spec.declarations
                    .model_output_roles_by_name
                ),
            )

        else:
            loss_specs = resolve_canonical_loss_configs(
                training_config.losses
            )

    source = prediction_config.source
    checkpoint_selection = source.checkpoint

    if model_family == YOYO_FAMILY and "checkpoint" not in source.model_fields_set:
        if training_manifest is None:
            checkpoint_selection = None
        elif get_optional_nested_value(
                training_manifest, "checkpoints", "best_checkpoint_path"
        ) is None:
            checkpoint_selection = "last"

    checkpoint_path, checkpoint_source = _resolve_checkpoint_path(
        checkpoint=checkpoint_selection,
        training_manifest=training_manifest,
        manifest_path=training_manifest_path,
    )

    # Record the resolved checkpoint path, or None when using runtime weights.
    prediction_config = prediction_config.model_copy(
        update={
            "source": prediction_config.source.model_copy(
                update={"checkpoint": checkpoint_path},
            ),
        },
    )

    if datamodule_is_external:
        dataset_config = None
        transform_pipeline_configs = None
        transform_pipeline_source: PredictionTransformPipelineSource = (
            "external_datamodule"
        )
        datamodule_config = None
        batch_size = None
        num_workers = None

    else:
        if prediction_config.dataset is not None:
            dataset_config = prediction_config.dataset
        elif training_config is None or training_datamodule_external:
            dataset_config = None
        else:
            dataset_config = training_config.dataset
            inherited_config_fields.add("dataset")

        if dataset_config is None:
            raise ValueError(
                "Prediction requires a dataset configuration, but none was "
                "provided in the prediction config or reconstructable from the "
                "training config. Pass `dataset` in the prediction config or "
                "provide a datamodule override to the prediction entrypoint."
            )

        (
            transform_pipeline_configs,
            transform_pipeline_source,
        ) = _resolve_prediction_transform_pipelines(
            prediction_config=prediction_config,
            training_config=training_config,
            training_datamodule_external=training_datamodule_external,
            model_family=model_family,
        )

        if transform_pipeline_source == "training_config":
            inherited_config_fields.add("transform_pipelines")

        if (
            training_config is not None
            and not training_datamodule_external
            and training_config.datamodule is not None
        ):
            base_datamodule_config = training_config.datamodule

            if prediction_config.data.batch_size is None:
                inherited_config_fields.add("data.batch_size")

            if prediction_config.data.num_workers is None:
                inherited_config_fields.add("data.num_workers")
        else:
            # BenchRep defaults are not inheritance from training.
            base_datamodule_config = TrainingDataModuleConfig()

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
                "prefetch_factor": (
                    base_datamodule_config.prefetch_factor
                    if num_workers > 0
                    else None
                ),
                "drop_last": False,
            }
        )

        validate_dataloader_kwargs(
            datamodule_config.dataloader_kwargs,
            num_workers=datamodule_config.num_workers,
        )

    base_reproducibility = (
        training_config.reproducibility
        if training_config is not None
        else TrainingReproducibilityConfig()
    )
    base_trainer = (
        training_config.trainer
        if training_config is not None
        else TrainingTrainerConfig()
    )

    seed = resolve_optional(
        prediction_config.inference.seed,
        base_reproducibility.seed,
        field_name="inference.seed",
    )

    seed_workers = resolve_optional(
        prediction_config.inference.seed_workers,
        base_reproducibility.seed_workers,
        field_name="inference.seed_workers",
    )

    deterministic = (
        prediction_config.inference.deterministic
        if prediction_config.inference.deterministic is not None
        else base_trainer.deterministic
    )

    trainer_config = base_trainer.model_copy(
        update={"deterministic": deterministic},
    )

    float32_matmul_precision: Float32MatmulPrecision = resolve_optional(
        prediction_config.inference.float32_matmul_precision,
        base_reproducibility.float32_matmul_precision,
        field_name="inference.float32_matmul_precision",
    )

    seed_inherited_from_training = (
            training_config is not None
            and prediction_config.inference.seed is None
    )

    if seed_inherited_from_training:
        inherited_config_fields.add("inference.seed")

    if training_config is not None:
        if prediction_config.inference.seed_workers is None:
            inherited_config_fields.add("inference.seed_workers")

        if prediction_config.inference.float32_matmul_precision is None:
            inherited_config_fields.add("inference.float32_matmul_precision")

        if prediction_config.inference.deterministic is None:
            inherited_config_fields.add("inference.deterministic")

    canonical_vae_reconstruction_latent_source = (
        _resolve_canonical_vae_reconstruction_latent_source(
            configured_source=(
                prediction_config.inference.canonical_vae_reconstruction_latent_source
            ),
            model_family=model_family,
            model_is_external=model_is_external,
        )
    )

    export_spec = resolve_prediction_exports(
        export_config=prediction_config.exports,
        seed=seed,
        model_family=model_family,
        training_config=training_config,
    )

    # The reconstruction seed inherits from training only when random subset
    # selection actually consumes the resolved inference seed.
    if (
            export_spec.reconstructions.enabled
            and export_spec.reconstructions.n_examples != "all"
            and export_spec.reconstructions.selection == "random"
            and prediction_config.exports.reconstructions.seed is None
            and seed_inherited_from_training
    ):
        inherited_config_fields.add("exports.reconstructions.seed")

    # Materialize effective inference settings, whether inherited or defaulted.
    resolved_config_updates: dict[str, Any] = {
        "inference": prediction_config.inference.model_copy(
            update={
                "seed": seed,
                "seed_workers": seed_workers,
                "deterministic": deterministic,
                "float32_matmul_precision": float32_matmul_precision,
            },
        ),
    }

    # Persist the export seed when its fallback ultimately came from training.
    if "exports.reconstructions.seed" in inherited_config_fields:
        resolved_config_updates["exports"] = prediction_config.exports.model_copy(
            update={
                "reconstructions": (
                    prediction_config.exports.reconstructions.model_copy(
                        update={"seed": export_spec.reconstructions.seed},
                    )
                ),
            },
        )

    if not datamodule_is_external:
        assert dataset_config is not None
        assert transform_pipeline_configs is not None

        # Materialize the effective dataset and transform pipelines.
        resolved_config_updates["dataset"] = dataset_config
        resolved_config_updates["transform_pipelines"] = list(
            transform_pipeline_configs
        )

        if not training_datamodule_external:
            resolved_config_updates["data"] = (
                prediction_config.data.model_copy(
                    update={
                        "batch_size": batch_size,
                        "num_workers": num_workers,
                    },
                )
            )

    prediction_config = prediction_config.model_copy(
        update=resolved_config_updates,
    )

    model_override_config = prediction_config.overrides.model

    # A supplied prediction section replaces training constructor parameters.
    # Otherwise, inherit them only when BenchRep will instantiate a model class.
    if (
        model_source == "external_class"
        and model_override_config is None
        and training_config is not None
    ):
        model_override_config = training_config.overrides.model

        if model_override_config is not None:
            inherited_config_fields.add("overrides.model")

    resolved_model_override = resolve_runtime_override_config(
        model_override_config,
        source=model_source,
        config_path="overrides.model",
    )

    resolved_datamodule_override = resolve_runtime_override_config(
        prediction_config.overrides.datamodule,
        source=datamodule_source,
        config_path="overrides.datamodule",
    )

    resolved_overrides = prediction_config.overrides.model_copy(
        update={
            "model": resolved_model_override,
            "datamodule": resolved_datamodule_override,
        },
    )

    prediction_config = prediction_config.model_copy(
        update={"overrides": resolved_overrides},
    )

    # Remove config-driven data settings that the external datamodule overrides.
    if datamodule_is_external:
        resolved_data_config = prediction_config.data.model_copy(
            update={
                "batch_size": None,
                "num_workers": None,
            },
        )

        prediction_config = prediction_config.model_copy(
            update={
                "dataset": None,
                "transform_pipelines": None,
                "data": resolved_data_config,
            },
        )

    if model_is_external:
        assert model_override_name is not None

        model_name = (
            f"{model_family.name}_external_{model_override_name}"
        )
    else:
        assert training_config is not None
        assert training_config.model is not None
        model_name = training_config.model.name

    run_config = (
        training_config.run
        if training_config is not None
        else TrainingRunConfig()
    )

    run_identity = RunIdentitySpec(
        output_root=Path(run_config.output_root),
        run_name_prefix=run_config.run_name_prefix,
        model_name=model_name,
    )

    return PredictionRunSpec(
        stage=prediction_config.stage,
        model_family=model_family,
        model_source=model_source,
        compatibility_policy=compatibility_policy,
        run_identity=run_identity,
        inherited_config_fields=frozenset[PredictionInheritableField](
            inherited_config_fields
        ),
        datamodule_source=datamodule_source,
        prediction_config=prediction_config,
        training_config=training_config,
        loss_specs=loss_specs,
        composite_model_spec=composite_model_spec,
        training_manifest=training_manifest,
        training_manifest_path=training_manifest_path,
        resolved_training_config_path=resolved_training_config_path,
        checkpoint_selection=checkpoint_selection,
        checkpoint_path=checkpoint_path,
        checkpoint_source=checkpoint_source,
        training_run_name=training_run_name,
        training_output_dir=training_output_dir,
        dataset_config=dataset_config,
        transform_pipeline_configs=transform_pipeline_configs,
        transform_pipeline_source=transform_pipeline_source,
        datamodule_config=datamodule_config,
        batch_size=batch_size,
        num_workers=num_workers,
        trainer_config=trainer_config,
        max_batches=prediction_config.data.max_batches,
        seed=seed,
        seed_workers=seed_workers,
        float32_matmul_precision=float32_matmul_precision,
        canonical_vae_reconstruction_latent_source=canonical_vae_reconstruction_latent_source,
        export_spec=export_spec,
    )


def _resolve_prediction_reconstruction_export(
    *,
    reconstruction_config: PredictionReconstructionsExportConfig,
    seed: int | None,
    model_family: ModelFamilySpec,
    training_config: TrainingConfig | None,
) -> PredictionReconstructionsExportSpec:
    mode = reconstruction_config.mode or "all"

    pairs = _resolve_prediction_reconstruction_pairs(
        reconstruction_config=reconstruction_config,
        model_family=model_family,
        training_config=training_config,
    )

    observation_specs: tuple[
        PredictionReconstructionObservationSpec,
        ...
    ] = ()

    if reconstruction_config.enabled and model_family == COMPOSITE_FAMILY:
        assert training_config is not None
        declarations = training_config.composite_model_declarations

        if declarations is None:
            raise ValueError(
                "Composite prediction requires "
                "`composite_model_declarations` in the resolved training "
                "configuration."
            )

        observation_specs = (
            _resolve_composite_reconstruction_observations(
                declarations=declarations,
            )
        )

        if (
                reconstruction_config.stratify_by is not None
                and reconstruction_config.stratify_by
                not in {
            observation.name
            for observation in observation_specs
        }
        ):
            raise ValueError(
                "`exports.reconstructions.stratify_by` references "
                f"{reconstruction_config.stratify_by!r}, which is not available "
                "among the Composite reconstruction observations: "
                f"{[observation.name for observation in observation_specs]}."
            )

    reconstruction_seed = (
        reconstruction_config.seed
        if reconstruction_config.seed is not None
        else seed
    )

    if (
        reconstruction_config.enabled
        and reconstruction_config.n_examples != "all"
        and reconstruction_config.selection == "random"
        and reconstruction_seed is None
    ):
        raise ValueError(
            "Random reconstruction export requires a seed. Set "
            "`exports.reconstructions.seed`, `inference.seed`, or use a "
            "training run with a reproducibility seed."
        )

    return PredictionReconstructionsExportSpec(
        enabled=reconstruction_config.enabled,
        mode=mode,
        pairs=pairs,
        observations=observation_specs,
        n_examples=reconstruction_config.n_examples,
        selection=reconstruction_config.selection,
        stratify_by=reconstruction_config.stratify_by,
        seed=reconstruction_seed,
        include_input=reconstruction_config.include_input,
        include_reconstruction=(
            reconstruction_config.include_reconstruction
        ),
    )


def resolve_prediction_exports(
    *,
    export_config: PredictionExportConfig,
    seed: int | None,
    model_family: ModelFamilySpec,
    training_config: TrainingConfig | None,
) -> PredictionExportSpec:
    """Resolve exports against family contracts or explicit YOYO selections.

    Canonical models use their fixed contracts; Composite models use their
    declarations. YOYO models require explicit keys and pairs, whose fields
    and tensor structures are checked during export.
    """

    anndata_spec = _resolve_prediction_anndata_export(
        anndata_config=export_config.anndata,
        model_family=model_family,
        training_config=training_config,
    )

    reconstruction_spec = _resolve_prediction_reconstruction_export(
        reconstruction_config=export_config.reconstructions,
        seed=seed,
        model_family=model_family,
        training_config=training_config,
    )

    return PredictionExportSpec(
        anndata=anndata_spec,
        reconstructions=reconstruction_spec,
    )


def _resolve_checkpoint_path(
    *,
    checkpoint: Literal["best", "last"] | Path | None,
    training_manifest: dict[str, Any] | None,
    manifest_path: Path | None,
) -> tuple[Path | None, PredictionCheckpointSource]:
    """Resolve the selected checkpoint, or use the runtime model's current weights.

    An absolute path selects a file directly. The best and last selectors and
    bare filenames are resolved against the linked training manifest.
    """
    if checkpoint is None:
        return None, "runtime_model"

    if isinstance(checkpoint, Path):
        checkpoint = checkpoint.expanduser()

    if isinstance(checkpoint, Path) and checkpoint.is_absolute():
        checkpoint_path = checkpoint.resolve()
        checkpoint_source: PredictionCheckpointSource = "explicit_path"
    else:
        if training_manifest is None or manifest_path is None:
            raise ValueError(
                f"Checkpoint selection {str(checkpoint)!r} requires a training "
                "manifest. For standalone YOYO prediction, omit `source.checkpoint` "
                "or set it to null to use the supplied model's current weights, "
                "or provide an absolute .ckpt path."
            )

        if checkpoint in ("best", "last"):
            checkpoint_path = get_required_nested_path(
                training_manifest,
                "checkpoints",
                f"{checkpoint}_checkpoint_path",
                base_dir=manifest_path.parent,
            )
            checkpoint_source = (
                "training_manifest_best"
                if checkpoint == "best"
                else "training_manifest_last"
            )
        else:
            assert isinstance(checkpoint, Path)
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


def _resolve_training_manifest_path(
    *,
    prediction_config: PredictionConfig,
    training_manifest_path_override: Path | str | None,
    model_family: ModelFamilySpec,
) -> tuple[PredictionConfig, Path | None]:
    """Resolve and validate the effective training-manifest path.

    An entrypoint-provided path overrides the configured path and is written
    into the returned config copy. YOYO models may omit both paths to run
    standalone.
    """

    if training_manifest_path_override is not None:
        training_manifest_path = normalize_path(training_manifest_path_override).resolve()

        prediction_config = prediction_config.model_copy(
            update={
                "source": prediction_config.source.model_copy(
                    update={"training_manifest_path": training_manifest_path}
                )
            }
        )
    else:
        training_manifest_path = (
            prediction_config.source.training_manifest_path
        )
        if training_manifest_path is None:
            if model_family == YOYO_FAMILY:
                return prediction_config, None
            raise ValueError(
                "A training manifest path must be provided either through "
                "`training_manifest_path_override` or `source.training_manifest_path`."
            )

        training_manifest_path = normalize_path(training_manifest_path).resolve()

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
    model_is_external: bool,
) -> None:
    if training_model_external and not model_is_external:
        raise ValueError(
            "Training manifest indicates that the trained model came from an external "
            "Python class or instance, but no model override was provided to the "
            "prediction entrypoint. Pass a compatible model class or instance that can "
            "load the recorded checkpoint."
        )


def _validate_prediction_model_family(
    *,
    training_model_construction: dict[str, Any],
    training_config: TrainingConfig,
    model_family: ModelFamilySpec,
    model_is_external: bool,
) -> None:
    recorded_family = training_model_construction.get("family")

    if recorded_family is not None and recorded_family != model_family.name:
        raise ValueError(
            "Prediction model family does not match the training run: "
            f"training family={recorded_family!r}, "
            f"prediction family={model_family.name!r}."
        )

    if model_is_external:
        return

    assert training_config.model is not None

    configured_model_name = MODELS.resolve_key(
        training_config.model.name
    )

    if configured_model_name != model_family.name:
        raise ValueError(
            "Configured model is incompatible with the selected prediction "
            "model family: "
            f"family={model_family.name!r}, "
            f"configured_model={configured_model_name!r}, "
            f"expected={model_family.name!r}."
        )


def _resolve_prediction_composite_model_spec(
    *,
    prediction_config: PredictionConfig,
    training_config: TrainingConfig | None,
    model_family: ModelFamilySpec,
    model_is_external: bool,
) -> CompositeModelSpec | None:
    """Resolve the effective Composite model graph used for prediction.

    Prediction-only assembly overrides may reroute an existing assembly-step
    input to another output produced by the same upstream component invocation.
    The complete Composite model is then resolved again so its contracts and
    data-flow graph are validated before construction.
    """
    input_overrides = (
        prediction_config.inference
        .composite_model_assembly_input_overrides
    )

    if model_family != COMPOSITE_FAMILY:
        if input_overrides is not None:
            raise ValueError(
                "`inference.composite_model_assembly_input_overrides` is "
                "supported only for composite models."
            )

        return None

    if model_is_external:
        raise ValueError(
            "Whole-model overrides are not supported for Composite "
            "prediction. BenchRep reconstructs the Composite model from "
            "the linked training configuration."
        )

    assert training_config is not None
    assert training_config.composite_model_declarations is not None
    assert training_config.composite_model_components is not None
    assert training_config.composite_model_assembly is not None

    effective_assembly_config = training_config.composite_model_assembly

    if input_overrides is not None:
        effective_assembly_config = (
            _apply_composite_model_assembly_input_overrides(
                assembly_config=effective_assembly_config,
                input_overrides=input_overrides,
            )
        )

    return resolve_composite_model_config(
        declarations_config=training_config.composite_model_declarations,
        components_config=training_config.composite_model_components,
        assembly_config=effective_assembly_config,
    )


def _apply_composite_model_assembly_input_overrides(
    *,
    assembly_config: dict[
        str,
        CompositeModelAssemblyStepConfig,
    ],
    input_overrides: dict[
        str,
        CompositeModelAssemblyInputOverrideConfig,
    ],
) -> dict[str, CompositeModelAssemblyStepConfig]:
    """Apply constrained prediction-time input rerouting.

    An overridden input must already exist on the selected consumer step.
    Its original and replacement references must both be outputs of the same
    producer step. This permits selecting another result from a variational or
    other mapping-producing component without changing the model's components
    or output bindings.
    """
    producer_step_by_output_reference: dict[str, str] = {}

    for producer_step_id, producer_step in assembly_config.items():
        output_references = (
            (producer_step.outputs,)
            if isinstance(producer_step.outputs, str)
            else tuple(producer_step.outputs.values())
        )

        for output_reference in output_references:
            producer_step_by_output_reference[output_reference] = (
                producer_step_id
            )

    effective_assembly_config = dict(assembly_config)

    for consumer_step_id, step_override in input_overrides.items():
        if consumer_step_id not in assembly_config:
            raise ValueError(
                "`inference.composite_model_assembly_input_overrides` "
                f"references unknown assembly step {consumer_step_id!r}. "
                f"Available steps: {list(assembly_config)}."
            )

        consumer_step = assembly_config[consumer_step_id]
        effective_inputs = dict(consumer_step.inputs)

        for input_name, replacement_reference in (
            step_override.inputs.items()
        ):
            if input_name not in consumer_step.inputs:
                raise ValueError(
                    "Composite assembly input override for step "
                    f"{consumer_step_id!r} references unknown component "
                    f"input {input_name!r}. Available inputs: "
                    f"{list(consumer_step.inputs)}."
                )

            original_reference = consumer_step.inputs[input_name]

            original_producer_step_id = (
                producer_step_by_output_reference.get(original_reference)
            )
            replacement_producer_step_id = (
                producer_step_by_output_reference.get(
                    replacement_reference
                )
            )

            if original_producer_step_id is None:
                raise ValueError(
                    "Composite assembly input override for step "
                    f"{consumer_step_id!r}, input {input_name!r}, cannot "
                    f"replace {original_reference!r}. Only inputs currently "
                    "routed from a `produces.*` output can be overridden."
                )

            if replacement_producer_step_id is None:
                raise ValueError(
                    "Composite assembly input override for step "
                    f"{consumer_step_id!r}, input {input_name!r}, references "
                    f"unknown model output {replacement_reference!r}."
                )

            if (
                replacement_producer_step_id
                != original_producer_step_id
            ):
                raise ValueError(
                    "Composite assembly input override for step "
                    f"{consumer_step_id!r}, input {input_name!r}, must select "
                    "another output from the same producer step "
                    f"{original_producer_step_id!r}; "
                    f"{replacement_reference!r} is produced by "
                    f"{replacement_producer_step_id!r}."
                )

            effective_inputs[input_name] = replacement_reference

        effective_assembly_config[consumer_step_id] = (
            consumer_step.model_copy(
                update={"inputs": effective_inputs},
            )
        )

    return effective_assembly_config


def _resolve_canonical_vae_reconstruction_latent_source(
    *,
    configured_source: CanonicalVAEReconstructionLatentSource | None,
    model_family: ModelFamilySpec,
    model_is_external: bool,
) -> CanonicalVAEReconstructionLatentSource | None:
    """Resolve the latent source used for prediction-time VAE reconstruction."""
    if configured_source is None:
        if model_is_external:
            return None

        if model_family == VAE_FAMILY:
            return "mean"

        return None

    if model_is_external:
        if model_family == VAE_FAMILY:
            raise ValueError(
                "`inference.canonical_vae_reconstruction_latent_source` cannot be set when "
                "prediction uses an external model. BenchRep cannot control how "
                "the external VAE model's `predict_step()` produces reconstructions."
            )

    if model_family != VAE_FAMILY:
        raise ValueError(
            "`inference.canonical_vae_reconstruction_latent_source` is only supported for "
            "prediction using internal VAE models."
        )

    return configured_source


def _resolve_prediction_transform_pipelines(
    *,
    prediction_config: PredictionConfig,
    training_config: TrainingConfig | None,
    training_datamodule_external: bool,
    model_family: ModelFamilySpec,
) -> tuple[
    tuple[PredictionTransformPipelineConfig, ...],
    PredictionTransformPipelineSource,
]:
    """Resolve configured transforms or inherit validation transforms from training.

    An explicit list replaces inherited pipelines. Without one, standalone runs
    and runs whose training datamodule was external use no transforms.
    """

    if prediction_config.transform_pipelines is not None:
        resolved_pipelines = _resolve_explicit_prediction_transform_routes(
            prediction_config.transform_pipelines,
            model_family=model_family,
            declarations_config=(
                training_config.composite_model_declarations
                if training_config is not None
                else None
            ),
        )

        return resolved_pipelines, "prediction_config"

    if training_config is None or training_datamodule_external:
        return (), "default_identity"

    inherited_pipelines: list[PredictionTransformPipelineConfig] = []

    for training_pipeline in training_config.transform_pipelines:
        inherited_steps = [
            PredictionTransformStepConfig.model_validate(
                step.model_dump(
                    mode="python",
                    exclude={"apply_to"},
                )
            )
            for step in training_pipeline.steps
            if "validation" in step.apply_to
        ]

        if not inherited_steps:
            continue

        assert training_pipeline.input is not None
        assert training_pipeline.output is not None

        inherited_pipelines.append(
            PredictionTransformPipelineConfig(
                input=training_pipeline.input,
                output=training_pipeline.output,
                steps=inherited_steps,
            )
        )

    if not inherited_pipelines:
        return (), "default_identity"

    return tuple(inherited_pipelines), "training_config"


def _resolve_explicit_prediction_transform_routes(
    configs: list[PredictionTransformPipelineConfig],
    *,
    model_family: ModelFamilySpec,
    declarations_config: CompositeModelDeclarationsConfig | None,
) -> tuple[PredictionTransformPipelineConfig, ...]:
    """Validate and materialize prediction routes for the selected model family.

    YOYO models require explicit input names; omitted outputs reuse those inputs.
    For these models, field availability and compatibility are checked at runtime.
    """

    if isinstance(model_family, CanonicalModelFamilySpec):
        for index, pipeline in enumerate(configs):
            route = (pipeline.input, pipeline.output)

            if route not in {
                (None, None),
                ("x", "x"),
            }:
                raise ValueError(
                    "Canonical model prediction transform pipelines use the "
                    "fixed route `x` to `x`; omit both fields or provide that "
                    f"resolved route: transform_pipelines[{index}]."
                )

        return tuple(
            pipeline.model_copy(
                update={
                    "input": "x",
                    "output": "x",
                }
            )
            for pipeline in configs
        )

    if model_family == YOYO_FAMILY:
        resolved_pipelines = []

        for index, pipeline in enumerate(configs):
            if pipeline.input is None:
                raise ValueError(
                    f"`transform_pipelines[{index}].input` is required for YOYO models. "
                    "Set it to the sample field this pipeline should transform."
                )

            resolved_pipelines.append(
                pipeline.model_copy(
                    update={"output": pipeline.output or pipeline.input},
                )
            )

        return tuple(resolved_pipelines)

    assert declarations_config is not None

    input_roles = declarations_config.expects
    sample_image_names = [
        name
        for name, role in input_roles.items()
        if role == "sample_image"
    ]

    resolved: list[PredictionTransformPipelineConfig] = []

    for index, pipeline in enumerate(configs):
        input_name = pipeline.input

        if input_name is None:
            if len(sample_image_names) != 1:
                raise ValueError(
                    f"`transform_pipelines[{index}].input` must be provided "
                    "when the Composite declarations do not contain exactly "
                    "one input with role `sample_image`; found "
                    f"{sample_image_names}."
                )

            input_name = sample_image_names[0]

        output_name = pipeline.output or input_name

        for field_name, declared_name in (
            ("input", input_name),
            ("output", output_name),
        ):
            if declared_name not in input_roles:
                raise ValueError(
                    f"`transform_pipelines[{index}].{field_name}` references "
                    f"{declared_name!r}, which is not declared under the "
                    "training configuration's "
                    "`composite_model_declarations.expects`."
                )

            role = input_roles[declared_name]

            if TENSOR_STRUCTURE_BY_ROLE[role] != "image":
                raise ValueError(
                    f"`transform_pipelines[{index}].{field_name}` references "
                    f"{declared_name!r}, whose declared role {role!r} is not "
                    "an image."
                )

        resolved.append(
            pipeline.model_copy(
                update={
                    "input": input_name,
                    "output": output_name,
                }
            )
        )

    return tuple(resolved)


def _find_duplicate_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for name in names:
        if name in seen:
            duplicates.add(name)
        else:
            seen.add(name)

    return sorted(duplicates)


def _resolve_composite_anndata_observations(
    *,
    declarations: CompositeModelDeclarationsConfig,
    selected_keys: tuple[str, ...],
    output_structures_by_key: dict[
        str,
        Literal["scalar", "vector"],
    ],
) -> tuple[PredictionAnnDataObservationSpec, ...]:
    """Resolve the columns written to ``adata.obs`` for a Composite model.

    AnnData observations include selected scalar model outputs, declared scalar
    model inputs, and declared batch metadata. At most one metadata field may
    provide the AnnData observation index.
    """
    batch_metadata = declarations.batch_metadata or {}

    index_names = [
        name
        for name, role in batch_metadata.items()
        if role == "index"
    ]

    if len(index_names) > 1:
        raise ValueError(
            "Composite AnnData export supports at most one batch-metadata "
            "declaration with role `index`; found "
            f"{index_names}."
        )

    observations = tuple(
        [
            PredictionAnnDataObservationSpec(
                name=key,
                source="model_output",
            )
            for key in selected_keys
            if output_structures_by_key[key] == "scalar"
        ]
        + [
            PredictionAnnDataObservationSpec(
                name=name,
                source="model_input",
            )
            for name, role in declarations.expects.items()
            if TENSOR_STRUCTURE_BY_ROLE[role] == "scalar"
        ]
        + [
            PredictionAnnDataObservationSpec(
                name=name,
                source="batch_metadata",
                use_as_index=role == "index",
            )
            for name, role in batch_metadata.items()
        ]
    )

    duplicate_names = _find_duplicate_names(
        [observation.name for observation in observations]
    )

    if duplicate_names:
        raise ValueError(
            "Composite AnnData export resolves conflicting observation names "
            "across selected scalar outputs, scalar model inputs, and batch "
            f"metadata: {duplicate_names}."
        )

    return observations


def _resolve_composite_reconstruction_observations(
    *,
    declarations: CompositeModelDeclarationsConfig,
) -> tuple[PredictionReconstructionObservationSpec, ...]:
    """Resolve contextual observations written into reconstruction bundles.

    Reconstruction observations include declared scalar model inputs and batch
    metadata. Model outputs are excluded because reconstruction observations
    describe the selected source examples rather than the model's exported
    predictions.
    """
    observations = tuple(
        [
            PredictionReconstructionObservationSpec(
                name=name,
                source="model_input",
            )
            for name, role in declarations.expects.items()
            if TENSOR_STRUCTURE_BY_ROLE[role] == "scalar"
        ]
        + [
            PredictionReconstructionObservationSpec(
                name=name,
                source="batch_metadata",
            )
            for name in (declarations.batch_metadata or {})
        ]
    )

    observation_names = [
        observation.name
        for observation in observations
    ]

    duplicate_names = _find_duplicate_names(observation_names)

    if duplicate_names:
        raise ValueError(
            "Composite reconstruction export resolves conflicting observation "
            "names across scalar model inputs and batch metadata: "
            f"{duplicate_names}."
        )

    if "source_index" in observation_names:
        raise ValueError(
            "Composite reconstruction observation name `source_index` is "
            "reserved for BenchRep's reconstruction selection index."
        )

    return observations


def _resolve_prediction_anndata_export(
    *,
    anndata_config: PredictionAnnDataExportConfig,
    model_family: ModelFamilySpec,
    training_config: TrainingConfig | None,
) -> PredictionAnnDataExportSpec:
    mode = anndata_config.mode or "all"

    if not anndata_config.enabled:
        return PredictionAnnDataExportSpec(
            enabled=False,
            mode=mode,
            keys=(),
            output_structures_by_key={},
            primary_key=None,
            observations=(),
        )

    if isinstance(model_family, CanonicalModelFamilySpec):
        # Canonical models have fixed keys.
        available_keys = (
            ("embedding", "z_mu", "z_logvar", "z_sample")
            if model_family == VAE_FAMILY
            else ("embedding",)
        )

        if mode == "all":
            # "embedding" and "z_mu" are equivalent.
            selected_keys = (
                ("embedding", "z_logvar", "z_sample")
                if model_family == VAE_FAMILY
                else ("embedding",)
            )
        else:
            assert anndata_config.keys is not None
            selected_keys = tuple(anndata_config.keys)

            unknown_keys = [
                key
                for key in selected_keys
                if key not in available_keys
            ]

            if unknown_keys:
                raise ValueError(
                    "`exports.anndata.keys` contains outputs unsupported by "
                    f"the {model_family.name!r} model family: {unknown_keys}. "
                    f"Available keys: {list(available_keys)}."
                )

        output_structures_by_key: dict[
            str,
            Literal["scalar", "vector"],
        ] = {
            key: "vector"
            for key in selected_keys
        }
        observation_specs: tuple[
            PredictionAnnDataObservationSpec,
            ...
        ] = ()

        primary_key = anndata_config.primary_key or "embedding"

    elif model_family == COMPOSITE_FAMILY:
        assert training_config is not None
        declarations = training_config.composite_model_declarations

        if declarations is None:
            raise ValueError(
                "Composite prediction requires "
                "`composite_model_declarations` in the resolved training "
                "configuration."
            )

        # All declared non-image outputs are exportable in AnnData.
        output_roles = declarations.produces
        exportable_keys = tuple(
            key
            for key, role in output_roles.items()
            if TENSOR_STRUCTURE_BY_ROLE[role] != "image"
        )

        if mode == "all":
            selected_keys = exportable_keys
        else:
            assert anndata_config.keys is not None
            selected_keys = tuple(anndata_config.keys)

            unknown_keys = [
                key
                for key in selected_keys
                if key not in output_roles
            ]
            if unknown_keys:
                raise ValueError(
                    "`exports.anndata.keys` contains outputs not declared under "
                    "`composite_model_declarations.produces`: "
                    f"{unknown_keys}."
                )

            image_keys = [
                key
                for key in selected_keys
                if TENSOR_STRUCTURE_BY_ROLE[output_roles[key]] == "image"
            ]
            if image_keys:
                raise ValueError(
                    "`exports.anndata.keys` contains image-valued outputs, "
                    "which cannot be exported to AnnData: "
                    f"{image_keys}."
                )

        output_structures_by_key: dict[
            str,
            Literal["scalar", "vector"],
        ] = {}

        for key in selected_keys:
            structure = TENSOR_STRUCTURE_BY_ROLE[output_roles[key]]

            if structure == "image":
                raise RuntimeError(
                    f"Internal error: image-valued output {key!r} reached "
                    "Composite AnnData export resolution."
                )

            output_structures_by_key[key] = structure

        observation_specs = _resolve_composite_anndata_observations(
            declarations=declarations,
            selected_keys=selected_keys,
            output_structures_by_key=output_structures_by_key,
        )

        primary_key = anndata_config.primary_key

        if primary_key is None:
            raise ValueError(
                "Composite AnnData export requires an explicit "
                "`exports.anndata.primary_key`."
            )

        if primary_key not in output_roles:
            raise ValueError(
                "`exports.anndata.primary_key` references an output not "
                "declared under `composite_model_declarations.produces`: "
                f"{primary_key!r}."
            )

        primary_role = output_roles[primary_key]

        if TENSOR_STRUCTURE_BY_ROLE[primary_role] != "vector":
            raise ValueError(
                "`exports.anndata.primary_key` must reference a vector-valued "
                f"output, but {primary_key!r} has role {primary_role!r}."
            )

    elif model_family == YOYO_FAMILY:
        if mode != "custom":
            raise ValueError(
                "AnnData export for YOYO models requires "
                "`exports.anndata.mode='custom'` and explicit "
                "`exports.anndata.keys`."
            )
        assert anndata_config.keys is not None
        selected_keys = tuple(anndata_config.keys)
        primary_key = anndata_config.primary_key
        if primary_key is None:
            raise ValueError(
                "AnnData export for YOYO models requires an explicit "
                "`exports.anndata.primary_key`."
            )
        # YOYO output structures are determined from returned tensors at runtime.
        output_structures_by_key = {}
        observation_specs = ()

    else:
        raise TypeError(
            f"Unsupported model family: {model_family.name!r}."
        )

    if not selected_keys:
        raise ValueError(
            "Enabled AnnData export did not resolve any exportable outputs."
        )

    if primary_key not in selected_keys:
        raise ValueError(
            f"AnnData primary key {primary_key!r} is not included in the "
            f"resolved export keys: {list(selected_keys)}."
        )

    return PredictionAnnDataExportSpec(
        enabled=True,
        mode=mode,
        keys=selected_keys,
        output_structures_by_key=output_structures_by_key,
        primary_key=primary_key,
        observations=observation_specs,
    )


def _resolve_prediction_reconstruction_pairs(
    *,
    reconstruction_config: PredictionReconstructionsExportConfig,
    model_family: ModelFamilySpec,
    training_config: TrainingConfig | None,
) -> tuple[PredictionReconstructionPairSpec, ...]:
    """Resolve declared pairs or explicit YOYO fields checked during export."""
    if not reconstruction_config.enabled:
        return ()

    mode = reconstruction_config.mode or "all"

    if model_family == YOYO_FAMILY:
        if mode != "custom":
            raise ValueError(
                "Reconstruction export for YOYO models requires "
                "`exports.reconstructions.mode='custom'` and explicit "
                "`exports.reconstructions.pairs`."
            )
        assert reconstruction_config.pairs is not None
        return tuple(
            PredictionReconstructionPairSpec(
                id=pair_id,
                input=pair.input,
                reconstruction=pair.reconstruction,
            )
            for pair_id, pair in reconstruction_config.pairs.items()
        )

    if mode == "custom":
        assert reconstruction_config.pairs is not None

        if isinstance(model_family, CanonicalModelFamilySpec):
            resolved_pairs = tuple(
                PredictionReconstructionPairSpec(
                    id=pair_id,
                    input=pair.input,
                    reconstruction=pair.reconstruction,
                )
                for pair_id, pair in reconstruction_config.pairs.items()
            )

            # Given the structural contracts, even external models submitted under
            # the canonical model families must output these keys.
            invalid_pairs = [
                pair.id
                for pair in resolved_pairs
                if (
                    pair.input != "input"
                    or pair.reconstruction != "reconstruction"
                )
            ]
            if invalid_pairs:
                raise ValueError(
                    "Canonical reconstruction export supports only the pair "
                    "`input` -> `reconstruction`. Invalid pair IDs: "
                    f"{invalid_pairs}."
                )

            return resolved_pairs

        if model_family != COMPOSITE_FAMILY:
            raise TypeError(
                f"Unsupported model family: {model_family.name!r}."
            )

        assert training_config is not None
        declarations = training_config.composite_model_declarations

        if declarations is None:
            raise ValueError(
                "Composite prediction requires "
                "`composite_model_declarations` in the resolved training "
                "configuration."
            )

        resolved_pairs: list[PredictionReconstructionPairSpec] = []

        for pair_id, pair in reconstruction_config.pairs.items():
            input_role = declarations.expects.get(pair.input)

            if input_role is None:
                raise ValueError(
                    f"`exports.reconstructions.pairs.{pair_id}.input` "
                    f"references {pair.input!r}, which is not declared under "
                    "`composite_model_declarations.expects`."
                )

            if TENSOR_STRUCTURE_BY_ROLE[input_role] != "image":
                raise ValueError(
                    f"`exports.reconstructions.pairs.{pair_id}.input` must "
                    "reference an image-valued model input, but "
                    f"{pair.input!r} has role {input_role!r}."
                )

            reconstruction_role = declarations.produces.get(
                pair.reconstruction
            )

            if reconstruction_role is None:
                raise ValueError(
                    f"`exports.reconstructions.pairs.{pair_id}."
                    f"reconstruction` references {pair.reconstruction!r}, "
                    "which is not declared under "
                    "`composite_model_declarations.produces`."
                )

            if reconstruction_role != "reconstruction_image":
                raise ValueError(
                    f"`exports.reconstructions.pairs.{pair_id}."
                    "reconstruction` must reference an output with role "
                    "`reconstruction_image`, but "
                    f"{pair.reconstruction!r} has role "
                    f"{reconstruction_role!r}."
                )

            resolved_pairs.append(
                PredictionReconstructionPairSpec(
                    id=pair_id,
                    input=pair.input,
                    reconstruction=pair.reconstruction,
                )
            )

        return tuple(resolved_pairs)

    # If mode = "all"
    if isinstance(model_family, CanonicalModelFamilySpec):
        return (
            PredictionReconstructionPairSpec(
                id="reconstruction_bundle_01",
                input="input",
                reconstruction="reconstruction",
            ),
        )

    if model_family != COMPOSITE_FAMILY:
        raise TypeError(
            f"Unsupported model family: {model_family.name!r}."
        )

    assert training_config is not None
    declarations = training_config.composite_model_declarations

    if declarations is None:
        raise ValueError(
            "Composite prediction requires "
            "`composite_model_declarations` in the resolved training "
            "configuration."
        )

    reconstruction_losses = (
        (training_config.losses or {}).get("reconstruction")
    )

    if not reconstruction_losses:
        raise ValueError(
            "Composite reconstruction export with `mode='all'` requires at "
            "least one configured loss under `losses.reconstruction`. Use "
            "`mode='custom'` to configure reconstruction pairs explicitly."
        )

    inferred_pairs: list[tuple[str, str]] = []

    # Infer input/reconstruction pairs from reconstruction losses.
    for loss_index, loss_config in enumerate(reconstruction_losses):
        loss_config_path = f"losses.reconstruction[{loss_index}]"
        wiring = loss_config.composite_wiring

        # Loss wiring is mandatory with Composite models.
        if wiring is None:
            raise ValueError(
                "Cannot infer a reconstruction pair because "
                f"`{loss_config_path}.composite_wiring` is missing."
            )

        input_names: list[str] = []
        reconstruction_names: list[str] = []

        for reference in wiring.values():
            source, separator, declaration_name = reference.partition(".")

            if separator != ".":
                continue

            if source == "expects":
                role = declarations.expects.get(declaration_name)

                if (
                        role is not None
                        and TENSOR_STRUCTURE_BY_ROLE[role] == "image"
                ):
                    input_names.append(declaration_name)

            elif source == "produces":
                role = declarations.produces.get(declaration_name)

                if role == "reconstruction_image":
                    reconstruction_names.append(declaration_name)

        input_names = list(dict.fromkeys(input_names))
        reconstruction_names = list(dict.fromkeys(reconstruction_names))

        if not (len(input_names) == len(reconstruction_names) == 1):
            raise ValueError(
                "Could not infer exactly one image input and one "
                "reconstruction-image output from "
                f"`{loss_config_path}.composite_wiring`; "
                f"found inputs={input_names} and "
                f"reconstructions={reconstruction_names}. Use "
                "`exports.reconstructions.mode='custom'` to configure the "
                "pair explicitly."
            )

        pair = (input_names[0], reconstruction_names[0])

        if pair not in inferred_pairs:
            inferred_pairs.append(pair)

    return tuple(
        PredictionReconstructionPairSpec(
            id=f"reconstruction_bundle_{index:02d}",
            input=input_name,
            reconstruction=reconstruction_name,
        )
        for index, (input_name, reconstruction_name) in enumerate(
            inferred_pairs,
            start=1,
        )
    )
