from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    NonNegativeInt,
    model_validator,
    ValidationInfo,
    Discriminator,
    Tag,
    field_validator,
)

from benchrep.assembly.registries.core import MODELS
from benchrep.assembly.registries.utils import normalize_name
from benchrep.architecture.models import (
    Autoencoder,
    VAE,
)


# Helper for model and datamodule overrides
def _require_present(value: object, field_name: str) -> None:
    if value is None:
        raise ValueError(
            f"`{field_name}` is required unless the corresponding object is overridden."
        )


# -------------------------
# Generic reusable blocks
# -------------------------
class _TrainingConfigBaseModel(BaseModel):
    """Base model for strict training configuration schemas."""

    model_config = ConfigDict(extra="forbid")


class NamedConfig(_TrainingConfigBaseModel):
    """Configuration for a named component."""

    name: str = Field(
        description="Name of the component to build.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used to configure the selected component.",
        json_schema_extra={
            "omit_behavior": "Uses an empty parameter mapping.",
            "null_behavior": "Not allowed; use an empty mapping instead.",
        },
    )


# -------------------------
# Run/output configuration
# -------------------------
class RunConfig(_TrainingConfigBaseModel):
    """Output location and run identity shared by linked workflows."""

    output_root: Path = Field(
        default=Path("outputs"),
        validate_default=True,
        description=(
            "Base directory for BenchRep outputs. Training and linked prediction "
            "runs write beneath separate stage subdirectories. Manifest-linked "
            "evaluation infers this base root unless its own output root is set."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `outputs/` relative to the working directory.",
            "null_behavior": "Not allowed.",
        },
    )
    project_name: str | None = Field(
        default=None,
        description=(
            "Optional prefix for generated run names. It is reused by linked "
            "prediction runs and propagated to manifest-linked evaluation runs. "
            "When constructing directory names, unsupported characters are replaced "
            "with underscores and leading or trailing punctuation is removed."
        ),
        json_schema_extra={
            "omit_behavior": "No project-name prefix is added.",
            "null_behavior": "Equivalent to omission; no project-name prefix is added.",
        },
    )

    @field_validator("output_root")
    @classmethod
    def resolve_output_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()


# -------------------------
# Architecture configuration
# -------------------------
class ModelConfig(NamedConfig):
    """Selects the BenchRep model family and its assembly parameters.

    Use `benchrep.list_registered_components("model", include_aliases=True)`
    to inspect available model names. Supported parameters and required
    encoder, decoder, and loss sections depend on the selected model.
    For supported models assembled from configuration, this configuration is
    recorded during training and reused to reconstruct the model for linked
    prediction runs.
    """


class EncoderConfig(NamedConfig):
    """Selects and configures an encoder from the encoder registry.

    Use `benchrep.list_registered_components("encoder", include_aliases=True)`
    to inspect available names. `params` are passed as keyword arguments to the
    selected encoder constructor. User-registered encoders must satisfy
    BenchRep's encoder interface and must be registered again when reconstructing
    the model in a linked prediction process.
    """


class DecoderConfig(NamedConfig):
    """Selects and configures a decoder from the decoder registry.

    Use `benchrep.list_registered_components("decoder", include_aliases=True)`
    to inspect available names. `params` are passed as keyword arguments to the
    selected decoder constructor, except for model-dependent dimensions supplied
    by BenchRep. `input_dim` is overridden by BenchRep and derived from the encoder
    output or VAE latent dimension. When required, `initial_shape` is inferred from
    `encoder.feature_shape` and must not be configured manually. User-registered
    decoders must satisfy BenchRep's decoder interface and must be registered
    again when reconstructing the model in a linked prediction process.
    """


# -------------------------
# Optimization/loss configuration
# -------------------------
class LossTermConfig(_TrainingConfigBaseModel):
    """Configuration for one weighted loss within a role-specific loss mapping.

    The surrounding mapping key is the registered loss name. Its parent role
    determines the registry and calling convention: reconstruction losses receive
    `reconstruction` and `target`, while regularization losses currently receive
    `z_mu` and `z_logvar`. Use
    `benchrep.list_registered_components("reconstruction_loss", include_aliases=True)`
    or
    `benchrep.list_registered_components("regularization_loss", include_aliases=True)`
    to inspect available names. User-registered losses must satisfy the relevant
    calling convention and must be registered again when reconstructing an
    internally assembled model for linked prediction.
    """

    weight: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Direct scalar coefficient applied to this raw loss before it is added "
            "to the total training loss. Weights are not normalized across losses "
            "or within loss roles."
        ),
        json_schema_extra={
            "omit_behavior": "Uses a weight of 1.0.",
            "null_behavior": "Not allowed.",
            "notes": [
                "A weight of 0 keeps the loss computation and logging but removes "
                "its contribution to the total loss."
            ],
        },
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments passed to the selected registered loss constructor."
        ),
        json_schema_extra={
            "omit_behavior": "Uses an empty parameter mapping.",
            "null_behavior": "Not allowed; use an empty mapping instead.",
        },
    )


class OptimizerConfig(NamedConfig):
    """Selects and configures an optimizer from the optimizer registry.

    Use `benchrep.list_registered_components("optimizer", include_aliases=True)`
    to inspect available names. `params` are keyword arguments for the selected
    optimizer constructor, excluding the model parameters. BenchRep stores the
    optimizer choice and arguments until Lightning calls
    `configure_optimizers()`, at which point the model parameters are supplied
    and the optimizer is instantiated.

    User-registered optimizers must follow the standard PyTorch optimizer
    interface. BenchRep supplies the model parameters as the first argument and
    then passes the configured `params`, equivalent to
    `MyOptimizer(model.parameters(), **params)`. They must be registered again
    when reconstructing the model in a linked prediction process.
    """


# -------------------------
# Training/runtime configuration
# -------------------------
class ReproducibilityConfig(_TrainingConfigBaseModel):
    """Controls training randomness and float32 matrix-multiplication precision.

    These settings are recorded with the training run and used as defaults by
    linked prediction runs unless prediction explicitly overrides them.
    """

    seed: int = Field(
        default=137,
        description=(
            "Passed to `lightning.seed_everything()` as the global seed and to "
            "BenchRep's internal datamodule for reproducible train-validation "
            "splitting. It also becomes the default seed for linked prediction "
            "and random reconstruction-example selection."
        ),
        json_schema_extra={
            "omit_behavior": "Uses seed 137.",
            "null_behavior": "Not allowed.",
        },
    )
    seed_workers: bool = Field(
        default=True,
        description=(
            "Passed as the `workers` argument to "
            "`lightning.seed_everything()`. When true, Lightning configures "
            "reproducible seeding for DataLoader worker processes."
        ),
        json_schema_extra={
            "omit_behavior": "Enables DataLoader-worker seeding.",
            "null_behavior": "Not allowed.",
        },
    )
    float32_matmul_precision: Literal["medium", "high", "highest"] = Field(
        default="highest",
        description=(
            "Passed to `torch.set_float32_matmul_precision()` before training and, "
            "unless overridden, linked prediction. Controls the internal precision "
            "used for float32 matrix multiplications without changing tensor dtypes."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `highest`.",
            "null_behavior": "Not allowed.",
        },
    )


class TrainerConfig(_TrainingConfigBaseModel):
    """Configuration forwarded to `lightning.Trainer`.

    All declared fields and additional non-null fields are passed as keyword
    arguments to `lightning.Trainer`. BenchRep manages `default_root_dir`,
    `logger`, `callbacks`, and `enable_checkpointing`, so these arguments cannot
    be configured here.

    The training Trainer configuration is reused for linked prediction.
    Prediction may override selected behavior through its own configuration and
    always disables Lightning logging and checkpointing.
    """

    max_epochs: PositiveInt | None = Field(
        default=None,
        description=(
            "Passed as `max_epochs` to `lightning.Trainer`. Sets the maximum "
            "number of complete training epochs."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Not passed to `lightning.Trainer`; the Trainer applies its "
                "default behavior."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )
    accelerator: str | None = Field(
        default="auto",
        description=(
            "Passed as `accelerator` to `lightning.Trainer`. Selects the "
            "hardware accelerator backend."
        ),
        json_schema_extra={
            "omit_behavior": "Passes `auto`, allowing the Trainer to select.",
            "null_behavior": (
                "Not passed to `lightning.Trainer`; the Trainer applies its "
                "default behavior."
            ),
        },
    )
    devices: str | int | list[int] | None = Field(
        default="auto",
        description=(
            "Passed as `devices` to `lightning.Trainer`. Selects how many or "
            "which devices the Trainer uses."
        ),
        json_schema_extra={
            "omit_behavior": "Passes `auto`, allowing the Trainer to select.",
            "null_behavior": (
                "Not passed to `lightning.Trainer`; the Trainer applies its "
                "default behavior."
            ),
        },
    )
    log_every_n_steps: PositiveInt | None = Field(
        default=None,
        description=(
            "Passed as `log_every_n_steps` to `lightning.Trainer`. Sets the "
            "number of training steps between metric-logging updates."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Not passed to `lightning.Trainer`; the Trainer applies its "
                "default behavior."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )
    deterministic: bool | Literal["warn"] | None = Field(
        default=None,
        description=(
            "Passed as `deterministic` to `lightning.Trainer`. Controls the use "
            "of deterministic algorithms. `warn` requests deterministic "
            "execution but warns instead of failing when an operation has no "
            "deterministic implementation."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Not passed to `lightning.Trainer`; the Trainer applies its "
                "default behavior."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )
    benchmark: bool | None = Field(
        default=False,
        description=(
            "Passed as `benchmark` to `lightning.Trainer`. Controls cuDNN "
            "benchmarking, which can improve convolution performance but may "
            "reduce reproducibility."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Passes `False` to `lightning.Trainer` to disable cuDNN benchmarking."
            ),
            "null_behavior": (
                "Not passed to `lightning.Trainer`; the Trainer applies its "
                "default behavior."
            ),
        },
    )
    precision: str | int | None = Field(
        default=None,
        description=(
            "Passed as `precision` to `lightning.Trainer`. Selects the "
            "Trainer's numerical precision mode. Accepted values depend on the "
            "installed Lightning version."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Not passed to `lightning.Trainer`; the Trainer applies its "
                "default behavior."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )

    model_config = ConfigDict(extra="allow")


class LoggerConfig(NamedConfig):
    """Selects and configures a training logger from the logger registry.

    Use `benchrep.list_registered_components("logger", include_aliases=True)`
    to inspect available names. `params` are passed unchanged as keyword
    arguments to the selected Lightning logger constructor, equivalent to
    `SelectedLogger(**params)`.

    CSV logging is available with the core installation. W&B, TensorBoard, and
    MLflow require the corresponding `wandb`, `tensorboard`, or `mlflow`
    installation extra; bundled supported external backends can be installed with
    the `logging` extra.

    For local MLflow tracking, use a database-backed tracking URI such as
    `sqlite:///mlflow.db`. Lightning's `file:<save_dir>` fallback uses MLflow's
    legacy filesystem backend, which newer supported MLflow versions reject by
    default. When `tracking_uri` is configured, MLFlowLogger ignores `save_dir`.

    Logging is currently used during training only. Linked prediction runs do
    not reconstruct or use the configured logger.
    """

    wandb_api_key_path: Path | None = Field(
        default=None,
        description=(
            "Optional path to a plain-text W&B API-key file. BenchRep expands "
            "the user directory, requires a non-empty file, reads and strips "
            "its contents, and sets `WANDB_API_KEY` before constructing "
            "`lightning.pytorch.loggers.WandbLogger`. This BenchRep-owned "
            "setting is not passed to the logger constructor and is only valid "
            "when the selected logger resolves to WandbLogger."
        ),
        json_schema_extra={
            "omit_behavior": (
                "BenchRep does not set `WANDB_API_KEY`; W&B uses its normal "
                "environment, existing-login, or offline-mode behavior."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )


class CheckpointConfig(_TrainingConfigBaseModel):
    """Controls checkpoint creation during training.

    BenchRep constructs a
    `lightning.pytorch.callbacks.ModelCheckpoint` from this configuration and
    supplies its output directory from the current training `RunContext`.
    Checkpoint paths and ranking information are recorded in the training
    manifest for use by linked prediction runs.

    When `monitor` names a metric, checkpoints are ranked using that metric and
    `mode`, `save_top_k`, and `filename` configure the ranked checkpoints.

    When `monitor=None`, BenchRep disables ranked checkpointing by constructing
    ModelCheckpoint with `monitor=None` and `save_top_k=0`. In that mode,
    `mode`, `save_top_k`, and `filename` have no effect, and `save_last=True`
    is required so that training produces a checkpoint.
    """

    monitor: str | None = Field(
        default="val/loss",
        min_length=1,
        description=(
            "Metric key passed to "
            "`lightning.pytorch.callbacks.ModelCheckpoint(monitor=...)`. "
            "The model must log a metric with this exact key. Set to null to "
            "disable metric-based ranking and save only the last checkpoint."
        ),
        json_schema_extra={
            "omit_behavior": "Monitors `val/loss`.",
            "null_behavior": (
                "Disables metric-based ranking. BenchRep forces `save_top_k=0`, "
                "ignores `mode` and `filename`, and requires `save_last=True`."
            ),
        },
    )

    mode: Literal["min", "max"] = Field(
        default="min",
        description=(
            "Direction passed to `ModelCheckpoint(mode=...)` when ranking "
            "checkpoints: `min` treats lower monitored values as better, while "
            "`max` treats higher values as better. Has no effect when "
            "`monitor=None`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `min`.",
            "null_behavior": "Not allowed.",
        },
    )

    save_top_k: int = Field(
        default=1,
        ge=-1,
        description=(
            "Number passed to `ModelCheckpoint(save_top_k=...)` when metric "
            "ranking is enabled. A positive value retains that many best "
            "checkpoints, `0` disables ranked checkpoints, and `-1` retains "
            "every checkpoint. BenchRep forces this to `0` when `monitor=None`."
        ),
        json_schema_extra={
            "omit_behavior": "Retains the single best ranked checkpoint.",
            "null_behavior": "Not allowed.",
        },
    )

    save_last: bool = Field(
        default=True,
        description=(
            "Passed to `ModelCheckpoint(save_last=...)`. When true, Lightning "
            "maintains `last.ckpt` in addition to any metric-ranked checkpoints. "
            "This is required when ranked checkpointing is disabled."
        ),
        json_schema_extra={
            "omit_behavior": "Saves `last.ckpt`.",
            "null_behavior": "Not allowed.",
        },
    )

    filename: str = Field(
        default="{epoch:03d}-{step}",
        min_length=1,
        description=(
            "Filename template passed to `ModelCheckpoint(filename=...)` for "
            "metric-ranked checkpoints. Lightning resolves placeholders from "
            "the epoch, step, and logged metrics and appends the checkpoint "
            "extension. This does not control the `last.ckpt` filename and has "
            "no effect when `monitor=None`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `{epoch:03d}-{step}`.",
            "null_behavior": "Not allowed.",
        },
    )

    @model_validator(mode="after")
    def validate_checkpoint_output(self) -> "CheckpointConfig":
        ranked_checkpoint_enabled = (
            self.monitor is not None and self.save_top_k != 0
        )

        if not ranked_checkpoint_enabled and not self.save_last:
            raise ValueError(
                "Checkpointing requires either a nonzero `save_top_k` with a "
                "configured `monitor`, or `save_last=True`; otherwise no "
                "checkpoint would be saved."
            )

        return self

# -------------------------
# Inspection configuration
# -------------------------
class TorchviewConfig(_TrainingConfigBaseModel):
    """Configures best-effort model-graph export with torchview.

    When enabled, BenchRep performs this inspection after training completes.
    It reads the shape of `batch["x"]` from the first training batch, replaces
    its batch dimension with one, and passes that synthesized input size to
    `torchview.draw_graph()`. The resulting Graphviz graph is rendered as
    `model_graph.png` in the training run's architecture directory.

    The graph represents the execution observed by torchview for one synthetic
    input shape. It may not capture alternative data-dependent branches,
    dynamic control flow, other supported input shapes, training/evaluation
    differences, or operations unsupported by torchview. It should therefore
    be treated as a diagnostic visualization rather than an authoritative
    description of every possible model execution.

    Export is best effort. Missing optional dependencies, incompatible model
    inputs, unsupported operations, tracing failures, and rendering failures
    produce warnings but do not fail an otherwise successful training run.
    Install the `model_graph` extra and ensure the Graphviz `dot` executable is
    available to enable rendering.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Whether BenchRep should attempt to export a torchview model graph "
            "after training completes."
        ),
        json_schema_extra={
            "omit_behavior": "Model-graph export is not attempted.",
            "null_behavior": "Not allowed.",
        },
    )

    expand_nested: bool = Field(
        default=True,
        description=(
            "Passed to `torchview.draw_graph(expand_nested=...)`. When true, "
            "torchview draws dashed-border boxes around nested modules to show "
            "the model's module hierarchy. This affects graph presentation, not "
            "which execution paths are inspected."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Nested modules are grouped using dashed-border boxes."
            ),
            "null_behavior": "Not allowed.",
        },
    )

    depth: int = Field(
        default=10,
        ge=0,
        description=(
            "Upper module-hierarchy depth passed to "
            "`torchview.draw_graph(depth=...)`. The main module has depth 0, "
            "its direct submodules have depth 1, and each additional nesting "
            "level increases the depth by one. Nodes deeper than this limit "
            "are omitted from the visualization."
        ),
        json_schema_extra={
            "omit_behavior": "Shows nodes through module-hierarchy depth 10.",
            "null_behavior": "Not allowed.",
        },
    )


class InspectionConfig(_TrainingConfigBaseModel):
    """Groups optional, best-effort inspection outputs for a training run."""

    torchview: TorchviewConfig = Field(
        default_factory=TorchviewConfig,
        description="Configuration for optional torchview model-graph export.",
        json_schema_extra={
            "omit_behavior": (
                "Uses the default TorchviewConfig, for which export is disabled."
            ),
            "null_behavior": (
                "Not allowed; set `torchview.enabled=False` to disable export."
            ),
        },
    )


# -------------------------
# Data configuration
# -------------------------
ParamsT = TypeVar("ParamsT")


class TransformConfig(NamedConfig):
    pass


class DatasetConfig(BaseModel, Generic[ParamsT]):
    name: str
    params: ParamsT

    @field_validator("name", mode="before")
    @classmethod
    def normalize_dataset_name(cls, value: Any) -> str:
        return normalize_name(value, field_name="dataset.name")


class MNISTDatasetParams(BaseModel):
    root: Path
    split: Literal["train", "test"] = "train"
    download: bool = False
    transform: TransformConfig | None = None


class MNISTDatasetConfig(DatasetConfig[MNISTDatasetParams]):
    name: Literal["mnist"] = "mnist"
    params: MNISTDatasetParams


class CustomDatasetConfig(DatasetConfig[dict[str, Any]]):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


def _dataset_config_discriminator(value: Any) -> str:
    if isinstance(value, dict):
        name = value.get("name")
    else:
        name = getattr(value, "name", None)

    if isinstance(name, str) and name.strip():
        normalized_name = normalize_name(name, field_name="dataset.name")

        if normalized_name == "mnist":
            return "mnist"

    return "custom"


SupportedDatasetConfig = Annotated[
    Annotated[MNISTDatasetConfig, Tag("mnist")]
    | Annotated[CustomDatasetConfig, Tag("custom")],
    Discriminator(_dataset_config_discriminator),
]


class DataModuleConfig(BaseModel):
    batch_size: PositiveInt = 32
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    num_workers: NonNegativeInt = 4
    pin_memory: bool | Literal["auto"] = "auto"
    persistent_workers: bool = False
    drop_last: bool = False


# -------------------------
# Full experiment configuration
# -------------------------
class TrainingConfig(BaseModel):
    stage: Literal["training"] = "training"
    run: RunConfig = Field(default_factory=RunConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)

    model: ModelConfig | None = None
    encoder: EncoderConfig | None = None
    decoder: DecoderConfig | None = None
    losses: dict[str, dict[str, LossTermConfig]] | None = Field(default_factory=dict)
    optimizer: OptimizerConfig | None = None

    dataset: SupportedDatasetConfig | None = None
    datamodule: DataModuleConfig | None = Field(default_factory=DataModuleConfig)

    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    logger: LoggerConfig | None = None
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    inspection: InspectionConfig = Field(default_factory=InspectionConfig)

    @model_validator(mode="after")
    def validate_override_requirements(self, info: ValidationInfo) -> "TrainingConfig":
        ctx = info.context or {}
        model_overridden = ctx.get("model_overridden", False)
        datamodule_overridden = ctx.get("datamodule_overridden", False)

        if not model_overridden:
            _require_present(self.model, "model")
            _require_present(self.encoder, "encoder")
            _require_present(self.losses, "losses")
            _require_present(self.optimizer, "optimizer")

        if not datamodule_overridden:
            _require_present(self.dataset, "dataset")
            _require_present(self.datamodule, "datamodule")

        return self

    @model_validator(mode="after")
    def validate_model_requirements(self, info: ValidationInfo) -> "TrainingConfig":
        ctx = info.context or {}
        model_overridden = ctx.get("model_overridden", False)

        if model_overridden:
            return self

        assert self.model is not None
        assert self.encoder is not None
        assert self.losses is not None
        assert self.optimizer is not None

        model_name = normalize_name(
            self.model.name,
            field_name="model.name",
        )

        model_cls = MODELS.get(model_name)

        if model_cls is Autoencoder:
            if self.decoder is None:
                raise ValueError("Autoencoder requires a decoder config section.")

            if "reconstruction" not in self.losses or not self.losses["reconstruction"]:
                raise ValueError(
                    "Autoencoder requires at least one reconstruction loss under "
                    "`losses.reconstruction`."
                )

        elif model_cls is VAE:
            if self.decoder is None:
                raise ValueError("VAE requires a decoder config section.")

            if "latent_dim" not in self.model.params:
                raise ValueError("VAE requires `model.params.latent_dim`.")

            latent_dim = self.model.params.get("latent_dim")
            if not isinstance(latent_dim, int) or latent_dim <= 0:
                raise ValueError(
                    "VAE requires `model.params.latent_dim` to be a positive integer."
                )

            if "reconstruction" not in self.losses or not self.losses["reconstruction"]:
                raise ValueError(
                    "VAE requires at least one reconstruction loss under "
                    "`losses.reconstruction`."
                )

            if "regularization" not in self.losses or not self.losses["regularization"]:
                raise ValueError(
                    "VAE requires at least one regularization loss under "
                    "`losses.regularization`."
                )

        return self
