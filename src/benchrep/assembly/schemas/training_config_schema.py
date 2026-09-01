from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeVar, TypeAlias

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

from benchrep.assembly.registries.core import CALLBACKS, MODELS
from benchrep.assembly.registries.utils import normalize_name
from benchrep.architecture.models import (
    Autoencoder,
    VAE,
)
from benchrep.assembly.schemas.runtime_override_config_schema import (
    RuntimeOverridesConfig,
)


# Helper for model and datamodule overrides
def _require_present(value: object, field_name: str) -> None:
    if value is None:
        raise ValueError(
            f"`{field_name}` is required unless the corresponding object is overridden."
        )


SupportedLossRole: TypeAlias = Literal[
    "reconstruction",
    "regularization",
    "custom_objective",
]


_LOGGER_REQUIRED_ADDITIONAL_CALLBACKS = frozenset({
    "device_stats_monitor",
    "learning_rate_monitor",
})

Float32MatmulPrecision: TypeAlias = Literal[
    "medium",
    "high",
    "highest",
]


# -------------------------
# Generic reusable blocks
# -------------------------
class _TrainingConfigBaseModel(BaseModel):
    """Base model for strict training configuration schemas."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "extra_field_behavior": (
                "Forbidden at this configuration level; unknown fields raise a "
                "validation error."
            ),
        },
    )


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
class TrainingRunConfig(_TrainingConfigBaseModel):
    """Output location and run identity shared by linked workflows."""

    output_root: Path = Field(
        default=Path("outputs"),
        validate_default=True,
        description="Base directory beneath which BenchRep creates workflow outputs.",
        json_schema_extra={
            "omit_behavior": "Uses `outputs/` relative to the working directory.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Training and linked prediction runs use separate stage subdirectories.",
                "Manifest-linked evaluation inherits this base root unless its own "
                "output root is configured.",
            ],
        },
    )

    project_name: str | None = Field(
        default=None,
        description="Optional prefix used to construct generated run names.",
        json_schema_extra={
            "omit_behavior": "No project-name prefix is added.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "The project name is reused by linked prediction runs and propagated "
                "to manifest-linked evaluation runs.",
                "Unsupported filename characters are replaced with underscores, and "
                "leading or trailing punctuation is removed.",
            ],
        },
    )

    @field_validator("output_root")
    @classmethod
    def resolve_output_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()


# -------------------------
# Architecture configuration
# -------------------------
class TrainingModelConfig(NamedConfig):
    """Selects the BenchRep model family and its assembly parameters.

    Use `benchrep.inspect_registry("model")` to inspect available model names
    and aliases, and `benchrep.inspect_registry("model", "<name>")` for details
    about a specific registered implementation.

    Supported parameters and required encoder, decoder, and loss sections depend
    on the selected model. For supported models assembled from configuration,
    this configuration is recorded during training and reused to reconstruct the
    model for linked prediction runs.
    """


class TrainingEncoderConfig(NamedConfig):
    """Selects and configures an encoder from the encoder registry.

    Use `benchrep.inspect_registry("encoder")` to inspect available names and
    aliases, and `benchrep.inspect_registry("encoder", "<name>")` for the
    registered constructor signature and documentation.

    `params` are passed as keyword arguments to the selected encoder constructor.
    User-registered encoders must satisfy BenchRep's encoder interface and must
    be registered again when reconstructing the model in a linked prediction
    process.
    """


class TrainingDecoderConfig(NamedConfig):
    """Selects and configures a decoder from the decoder registry.

    Use `benchrep.inspect_registry("decoder")` to inspect available names and
    aliases, and `benchrep.inspect_registry("decoder", "<name>")` for the
    registered constructor signature and documentation.

    `params` are passed as keyword arguments to the selected decoder constructor,
    except for model-dependent dimensions supplied by BenchRep. `input_dim` is
    overridden by BenchRep and derived from the encoder output or VAE latent
    dimension. When required, `initial_shape` is inferred from
    `encoder.feature_shape` and must not be configured manually.

    User-registered decoders must satisfy BenchRep's decoder interface and must
    be registered again when reconstructing the model in a linked prediction
    process.
    """


# -------------------------
# Optimization/loss configuration
# -------------------------
class TrainingLossTermConfig(_TrainingConfigBaseModel):
    """Configuration for one weighted term in a role-specific loss mapping.

    The surrounding mapping key is the registered component name. Its parent role
    selects the registry and runtime calling convention:

    - `reconstruction`: called with `reconstruction` and `target`.
    - `regularization`: called with `z_mu` and `z_logvar`.
    - `custom_objective`: called with `batch` and `model_output` mappings.

    `params` are passed only to the registered component's constructor. Runtime
    model tensors are supplied separately by the model when the objective is
    evaluated.

    Every configured term must return a scalar tensor. BenchRep multiplies that
    value by the configured `weight` before adding it to the other configured
    terms.

    Use `benchrep.inspect_registry("reconstruction_loss")`,
    `benchrep.inspect_registry("regularization_loss")`, or
    `benchrep.inspect_registry("custom_objective_loss")` to inspect available names,
    aliases, constructors, and calling contracts.

    User-registered components must satisfy the selected role's calling convention
    and must be registered again when reconstructing an internally assembled model
    for linked prediction.

    A custom objective may use any fields available in the batch or model output.
    If it is used in place of reconstruction or regularization losses, the custom
    objective is responsible for implementing the omitted behavior.
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


LossRoleTerms: TypeAlias = Annotated[
    dict[str, TrainingLossTermConfig],
    Field(min_length=1),
]


class TrainingOptimizerConfig(NamedConfig):
    """Selects and configures an optimizer from the optimizer registry.

    Use `benchrep.inspect_registry("optimizer")` to inspect available names and
    aliases, and `benchrep.inspect_registry("optimizer", "<name>")` for the
    registered constructor signature and documentation.

    `params` are keyword arguments for the selected optimizer constructor,
    excluding the model parameters. BenchRep stores the optimizer choice and
    arguments until Lightning calls `configure_optimizers()`, at which point the
    model parameters are supplied and the optimizer is instantiated.

    User-registered optimizers must follow the standard PyTorch optimizer
    interface. BenchRep supplies the model parameters as the first argument and
    then passes the configured `params`, equivalent to
    `MyOptimizer(model.parameters(), **params)`. They must be registered again
    when reconstructing the model in a linked prediction process.
    """


# -------------------------
# Training/runtime configuration
# -------------------------
class TrainingReproducibilityConfig(_TrainingConfigBaseModel):
    """Controls training randomness and float32 matrix-multiplication precision.

    These settings are recorded with the training run and used as defaults by
    linked prediction runs unless prediction explicitly overrides them.
    """

    seed: int = Field(
        default=137,
        description="Global random seed used for training and internal data splitting.",
        json_schema_extra={
            "omit_behavior": "Uses seed 137.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Passed to `lightning.seed_everything()` and BenchRep's internal "
                "datamodule.",
                "Inherited as the default seed for linked prediction and random "
                "reconstruction-example selection.",
            ],
        },
    )

    seed_workers: bool = Field(
        default=True,
        description="Whether DataLoader worker processes receive reproducible seeds.",
        json_schema_extra={
            "omit_behavior": "Enables DataLoader-worker seeding.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Passed as the `workers` argument to `lightning.seed_everything()`.",
            ],
        },
    )

    float32_matmul_precision: Float32MatmulPrecision = Field(
        default="highest",
        description="Internal precision used for float32 matrix multiplications.",
        json_schema_extra={
            "omit_behavior": "Uses `highest`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Passed to `torch.set_float32_matmul_precision()` before training.",
                "Does not change tensor dtypes.",
                "Inherited by linked prediction unless explicitly overridden.",
            ],
        },
    )


class TrainingTrainerConfig(_TrainingConfigBaseModel):
    """Configuration forwarded to `lightning.Trainer`.

    All declared fields and additional non-null fields are passed as keyword
    arguments to `lightning.Trainer`. BenchRep manages `default_root_dir`,
    `logger`, `callbacks`, and `enable_checkpointing`, so these arguments cannot
    be configured here. Use the top-level `logger`, `checkpointing`,
    `early_stopping`, and `additional_callbacks` sections for their supported
    BenchRep-managed equivalents.

    The general Trainer configuration is reused for linked prediction.
    Prediction may override selected behavior through its own configuration,
    always disables Lightning logging and checkpointing, and does not inherit
    training callbacks.
    """

    max_epochs: PositiveInt | None = Field(
        default=None,
        description="Maximum number of complete training epochs.",
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
        description="Hardware accelerator backend used for training.",
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
        description="Number or identifiers of devices used for training.",
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
        description="Number of training steps between metric-logging updates.",
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
        description="Whether deterministic algorithms are requested during training.",
        json_schema_extra={
            "omit_behavior": (
                "Not passed to `lightning.Trainer`; the Trainer applies its "
                "default behavior."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "`warn` requests deterministic execution but warns instead of failing "
                "when an operation lacks a deterministic implementation.",
            ],
        },
    )

    benchmark: bool | None = Field(
        default=False,
        description="Whether cuDNN benchmarking is enabled.",
        json_schema_extra={
            "omit_behavior": (
                "Passes `False` to `lightning.Trainer` to disable cuDNN benchmarking."
            ),
            "null_behavior": (
                "Not passed to `lightning.Trainer`; the Trainer applies its "
                "default behavior."
            ),
            "notes": [
                "Benchmarking can improve convolution performance but may reduce "
                "reproducibility.",
            ],
        },
    )

    precision: str | int | None = Field(
        default=None,
        description="Numerical precision mode used by the Trainer.",
        json_schema_extra={
            "omit_behavior": (
                "Not passed to `lightning.Trainer`; the Trainer applies its "
                "default behavior."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Accepted values depend on the installed Lightning version.",
            ],
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null fields are forwarded as keyword "
                "arguments to `lightning.Trainer`."
            ),
        },
    )


class TrainingLoggerConfig(NamedConfig):
    """Selects and configures a training logger from the logger registry.

    Use `benchrep.inspect_registry("logger")` to inspect available names and
    aliases, and `benchrep.inspect_registry("logger", "<name>")` for the
    registered constructor signature and documentation.

    `params` are passed unchanged as keyword arguments to the selected Lightning
    logger constructor, equivalent to `SelectedLogger(**params)`.

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
        description="Optional path to a plain-text W&B API-key file.",
        json_schema_extra={
            "omit_behavior": (
                "BenchRep does not set `WANDB_API_KEY`; W&B uses its normal "
                "environment, existing-login, or offline-mode behavior."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Only valid when the selected logger resolves to `WandbLogger`.",
                "BenchRep expands the user directory and requires a non-empty file.",
                "The stripped file contents are assigned to `WANDB_API_KEY` before the "
                "logger is constructed.",
                "This BenchRep-owned setting is not forwarded to the logger constructor.",
            ],
        },
    )


class TrainingCheckpointConfig(_TrainingConfigBaseModel):
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
        description="Metric key used to rank checkpoints.",
        json_schema_extra={
            "omit_behavior": "Monitors `val/loss`.",
            "null_behavior": (
                "Disables metric-based ranking. BenchRep forces `save_top_k=0`, "
                "ignores `mode` and `filename`, and requires `save_last=True`."
            ),
            "notes": [
                "The model must log a metric with this exact key.",
            ],
        },
    )

    mode: Literal["min", "max"] = Field(
        default="min",
        description="Direction used to rank monitored metric values.",
        json_schema_extra={
            "omit_behavior": "Uses `min`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "`min` treats lower values as better; `max` treats higher values as better.",
                "Has no effect when `monitor=None`.",
            ],
        },
    )

    save_top_k: int = Field(
        default=1,
        ge=-1,
        description="Number of metric-ranked checkpoints retained.",
        json_schema_extra={
            "omit_behavior": "Retains the single best ranked checkpoint.",
            "null_behavior": "Not allowed.",
            "notes": [
                "A positive value retains that many best checkpoints, `0` disables ranked "
                "checkpoints, and `-1` retains every checkpoint.",
                "BenchRep forces this value to `0` when `monitor=None`.",
            ],
        },
    )

    save_last: bool = Field(
        default=True,
        description="Whether Lightning maintains a `last.ckpt` checkpoint.",
        json_schema_extra={
            "omit_behavior": "Saves `last.ckpt`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Saved in addition to any metric-ranked checkpoints.",
                "Required when ranked checkpointing is disabled.",
            ],
        },
    )

    filename: str = Field(
        default="{epoch:03d}-{step}",
        min_length=1,
        description="Filename template used for metric-ranked checkpoints.",
        json_schema_extra={
            "omit_behavior": "Uses `{epoch:03d}-{step}`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Lightning resolves placeholders from the epoch, step, and logged metrics "
                "and appends the checkpoint extension.",
                "Does not control the `last.ckpt` filename.",
                "Has no effect when `monitor=None`.",
            ],
        },
    )

    @model_validator(mode="after")
    def validate_checkpoint_output(self) -> TrainingCheckpointConfig:
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
# Early stopping configuration
# -------------------------
class TrainingEarlyStoppingConfig(_TrainingConfigBaseModel):
    """Configures metric-based early stopping during training.

    BenchRep constructs a `lightning.pytorch.callbacks.EarlyStopping`
    callback when this configuration is present. Declared fields and
    additional non-null fields are passed as keyword arguments to that
    callback. The callback checks the monitored metric at Lightning's
    configured validation or training-check frequency and requests a clean
    stop when its stopping criterion is met.

    Early stopping ends training normally rather than failing the workflow.
    BenchRep can therefore continue checkpoint validation, optional inspection,
    and manifest writing after the Trainer stops.
    """

    monitor: str = Field(
        default="val/loss",
        min_length=1,
        description="Metric key monitored for early stopping.",
        json_schema_extra={
            "omit_behavior": "Monitors `val/loss`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "The model must log a metric with this exact key.",
            ],
        },
    )

    mode: Literal["min", "max"] = Field(
        default="min",
        description="Direction used to interpret improvement and stopping thresholds.",
        json_schema_extra={
            "omit_behavior": "Uses `min`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "With `min`, lower values are better, improvement requires decreasing by "
                "at least `min_delta`, `stopping_threshold` is reached below its value, "
                "and `divergence_threshold` is crossed above its value.",
                "With `max`, these directions are reversed.",
            ],
        },
    )

    patience: NonNegativeInt = Field(
        default=3,
        description="Number of consecutive insufficient-improvement checks allowed.",
        json_schema_extra={
            "omit_behavior": "Allows three checks without sufficient improvement.",
            "null_behavior": "Not allowed.",
            "notes": [
                "An improvement of at least `min_delta` resets the counter.",
                "Patience counts metric checks, not necessarily epochs.",
                "Does not apply to immediate stopping, divergence, or non-finite-value "
                "conditions.",
            ],
        },
    )

    min_delta: float = Field(
        default=0.0,
        description=(
            "Minimum change in the monitored metric required to count as an "
            "improvement."
        ),
        json_schema_extra={
            "omit_behavior": "Any improvement greater than zero resets patience.",
            "null_behavior": "Not allowed.",
        },
    )

    strict: bool = Field(
        default=True,
        description="Whether a missing monitored metric fails training.",
        json_schema_extra={
            "omit_behavior": "Fails when the monitored metric is unavailable.",
            "null_behavior": "Not allowed.",
            "notes": [
                "When false, Lightning warns instead of raising an error.",
            ],
        },
    )

    check_finite: bool = Field(
        default=True,
        description=(
            "Whether early stopping should stop training when the monitored metric "
            "becomes NaN or infinite."
        ),
        json_schema_extra={
            "omit_behavior": "Stops when the monitored metric is non-finite.",
            "null_behavior": "Not allowed.",
        },
    )

    stopping_threshold: float | None = Field(
        default=None,
        description=(
            "Optional target value that stops training immediately once reached."
        ),
        json_schema_extra={
            "omit_behavior": "No target-value threshold is used.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    divergence_threshold: float | None = Field(
        default=None,
        description=(
            "Optional value indicating that the monitored metric has become "
            "sufficiently poor to stop training immediately."
        ),
        json_schema_extra={
            "omit_behavior": "No divergence threshold is used.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    check_on_train_epoch_end: bool | None = Field(
        default=None,
        description=(
            "Whether the stopping criterion is checked at the end of each training epoch."
        ),
        json_schema_extra={
            "omit_behavior": "Lets Lightning determine when checks occur.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null fields are forwarded as keyword "
                "arguments to `lightning.pytorch.callbacks.EarlyStopping`."
            ),
        },
    )


# -------------------------
# Additional callback configuration
# -------------------------
class TrainingAdditionalCallbackConfig(NamedConfig):
    """Selects and configures one additional registered Lightning callback.

    `name` identifies a callback in BenchRep's callback registry. `params` are
    passed unchanged as keyword arguments to the callback constructor.

    Additional callbacks apply only to training and supplement, rather than
    replace, BenchRep-managed checkpointing and early stopping. BenchRep records
    their configuration but does not specially interpret their runtime behavior.
    """


# -------------------------
# Inspection configuration
# -------------------------
class TrainingTorchviewConfig(_TrainingConfigBaseModel):
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
        description="Whether nested modules are visually grouped in the model graph.",
        json_schema_extra={
            "omit_behavior": "Nested modules are grouped using dashed-border boxes.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Passed to `torchview.draw_graph(expand_nested=...)`.",
                "Affects graph presentation, not which execution paths are inspected.",
            ],
        },
    )

    depth: int = Field(
        default=10,
        ge=0,
        description="Maximum module-hierarchy depth included in the model graph.",
        json_schema_extra={
            "omit_behavior": "Shows nodes through module-hierarchy depth 10.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Passed to `torchview.draw_graph(depth=...)`.",
                "The main module has depth 0, direct submodules have depth 1, and each "
                "additional nesting level increases the depth by one.",
                "Nodes deeper than this limit are omitted.",
            ],
        },
    )


class TrainingInspectionConfig(_TrainingConfigBaseModel):
    """Groups optional, best-effort inspection outputs for a training run."""

    torchview: TrainingTorchviewConfig = Field(
        default_factory=TrainingTorchviewConfig,
        description="Configuration for optional torchview model-graph export.",
        json_schema_extra={
            "omit_behavior": (
                "Uses the default TrainingTorchviewConfig, for which export is disabled."
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


class TrainingTransformConfig(NamedConfig):
    """Configuration for one transform in an ordered transform sequence.

    Use `benchrep.inspect_registry("transform")` to inspect available names and
    aliases, and `benchrep.inspect_registry("transform", "<name>")` for the
    registered constructor signature and documentation.
    """

    apply_to: list[Literal["training", "validation"]] = Field(
        min_length=1,
        description=(
            "Split pipelines that include this transform. The transform's "
            "position in the surrounding sequence determines its order within "
            "each targeted pipeline. Validation-targeted transforms are also "
            "inherited by linked prediction runs when prediction transforms "
            "are omitted or null."
        ),
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "If `datamodule.val_fraction` is 0, validation-targeted "
                "transforms do not run during training but remain available "
                "for inheritance by linked prediction."
            ],
        },
    )

    @field_validator("apply_to")
    @classmethod
    def validate_unique_transform_targets(
        cls,
        value: list[Literal["training", "validation"]],
    ) -> list[Literal["training", "validation"]]:
        if len(value) != len(set(value)):
            raise ValueError(
                "apply_to must not contain duplicate split targets."
            )

        return value


class DatasetConfig(_TrainingConfigBaseModel, Generic[ParamsT]):
    """Selects a registered dataset and configures its construction.

    Use `benchrep.inspect_registry("dataset")` to inspect registered dataset
    names and aliases. Built-in datasets use dataset-specific configuration and
    parameter types, which can be explored by passing the concrete type to
    `benchrep.inspect_config()`.

    Other registered dataset names use `CustomDatasetConfig`, whose parameters
    are passed directly to the registered dataset constructor.
    """

    name: str = Field(
        description=(
            "Name of the registered dataset to build. Names are normalized before "
            "registry lookup."
        ),
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )

    params: ParamsT = Field(
        description=(
            "Parameters used to construct the selected dataset. The accepted "
            "fields depend on the dataset name."
        ),
        json_schema_extra={
            "omit_behavior": "Determined by the concrete dataset configuration.",
            "null_behavior": "Not allowed.",
        },
    )

    @field_validator("name", mode="before")
    @classmethod
    def normalize_dataset_name(cls, value: Any) -> str:
        return normalize_name(value, field_name="dataset.name")


class MNISTDatasetParams(_TrainingConfigBaseModel):
    """Parameters for the built-in MNIST dataset."""

    root: Path = Field(
        description="Directory containing or receiving the MNIST dataset files.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )

    split: Literal["train", "test"] = Field(
        default="train",
        description="MNIST split to load.",
        json_schema_extra={
            "omit_behavior": "Uses the training split.",
            "null_behavior": "Not allowed.",
        },
    )

    download: bool = Field(
        default=False,
        description=(
            "Whether torchvision should download the dataset when it is not "
            "available under `root`."
        ),
        json_schema_extra={
            "omit_behavior": "Does not download the dataset.",
            "null_behavior": "Not allowed.",
        },
    )


class MNISTDatasetConfig(DatasetConfig[MNISTDatasetParams]):
    """Configuration selecting BenchRep's built-in MNIST dataset."""

    name: Literal["mnist"] = Field(
        default="mnist",
        description="Registered name of the built-in MNIST dataset.",
        json_schema_extra={
            "omit_behavior": "Uses `mnist`.",
            "null_behavior": "Not allowed.",
        },
    )

    params: MNISTDatasetParams = Field(
        description="Parameters used to construct the MNIST dataset.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )


class CIFAR10DatasetParams(_TrainingConfigBaseModel):
    """Parameters for the built-in CIFAR-10 dataset."""

    root: Path = Field(
        description="Directory containing or receiving the CIFAR-10 dataset files.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )

    split: Literal["train", "test"] = Field(
        default="train",
        description="CIFAR-10 split to load.",
        json_schema_extra={
            "omit_behavior": "Uses the training split.",
            "null_behavior": "Not allowed.",
        },
    )

    download: bool = Field(
        default=False,
        description=(
            "Whether torchvision should download the dataset when it is not "
            "available under `root`."
        ),
        json_schema_extra={
            "omit_behavior": "Does not download the dataset.",
            "null_behavior": "Not allowed.",
        },
    )


class CIFAR10DatasetConfig(DatasetConfig[CIFAR10DatasetParams]):
    """Configuration selecting BenchRep's built-in CIFAR-10 dataset."""

    name: Literal["cifar10", "cifar_10"] = Field(
        default="cifar10",
        description=(
            "Registered name of the built-in CIFAR-10 dataset. `cifar_10` is "
            "accepted as an alias."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `cifar10`.",
            "null_behavior": "Not allowed.",
        },
    )

    params: CIFAR10DatasetParams = Field(
        description="Parameters used to construct the CIFAR-10 dataset.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )


class STL10DatasetParams(_TrainingConfigBaseModel):
    """Parameters for the built-in STL-10 dataset."""

    root: Path = Field(
        description="Directory containing or receiving the STL-10 dataset files.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )

    split: Literal[
        "train",
        "test",
        "unlabeled",
        "train+unlabeled",
    ] = Field(
        default="train",
        description=(
            "STL-10 split to load. Unlabeled samples have the label `-1`; "
            "`train+unlabeled` combines the training and unlabeled splits."
        ),
        json_schema_extra={
            "omit_behavior": "Uses the training split.",
            "null_behavior": "Not allowed.",
        },
    )

    download: bool = Field(
        default=False,
        description=(
            "Whether torchvision should download the dataset when it is not "
            "available under `root`."
        ),
        json_schema_extra={
            "omit_behavior": "Does not download the dataset.",
            "null_behavior": "Not allowed.",
        },
    )


class STL10DatasetConfig(DatasetConfig[STL10DatasetParams]):
    """Configuration selecting BenchRep's built-in STL-10 dataset."""

    name: Literal["stl10", "stl_10"] = Field(
        default="stl10",
        description=(
            "Registered name of the built-in STL-10 dataset. `stl_10` is accepted "
            "as an alias."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `stl10`.",
            "null_behavior": "Not allowed.",
        },
    )

    params: STL10DatasetParams = Field(
        description="Parameters used to construct the STL-10 dataset.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )


class CustomDatasetConfig(DatasetConfig[dict[str, Any]]):
    """Configuration for a user-registered dataset.

    The dataset must be registered before the training workflow builds it
    and must produce a `BaseDataset` instance. `params` are passed as keyword
    arguments to its constructor.
    """

    name: str = Field(
        description="Registered name of the custom dataset to build.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )

    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments passed to the registered dataset constructor."
        ),
        json_schema_extra={
            "omit_behavior": "Uses an empty parameter mapping.",
            "null_behavior": "Not allowed; use an empty mapping instead.",
        },
    )


def _dataset_config_discriminator(value: Any) -> str:
    if isinstance(value, dict):
        name = value.get("name")
    else:
        name = getattr(value, "name", None)

    if isinstance(name, str) and name.strip():
        normalized_name = normalize_name(name, field_name="dataset.name")

        if normalized_name in {"cifar10", "cifar_10"}:
            return "cifar10"

        if normalized_name in {"stl10", "stl_10"}:
            return "stl10"

        if normalized_name == "mnist":
            return "mnist"

    return "custom"


SupportedDatasetConfig: TypeAlias = Annotated[
    Annotated[CIFAR10DatasetConfig, Tag("cifar10")]
    | Annotated[MNISTDatasetConfig, Tag("mnist")]
    | Annotated[STL10DatasetConfig, Tag("stl10")]
    | Annotated[CustomDatasetConfig, Tag("custom")],
    Discriminator(_dataset_config_discriminator),
]


class TrainingDataModuleConfig(_TrainingConfigBaseModel):
    """Configures batching, data loading, and train-validation splitting."""

    batch_size: PositiveInt = Field(
        default=32,
        description="Number of samples loaded in each batch.",
        json_schema_extra={
            "omit_behavior": "Uses a batch size of 32.",
            "null_behavior": "Not allowed.",
        },
    )

    val_fraction: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description=(
            "Fraction of the configured dataset reserved for validation. A value "
            "of `0.0` disables validation splitting."
        ),
        json_schema_extra={
            "omit_behavior": "Reserves 10% of the dataset for validation.",
            "null_behavior": "Not allowed.",
        },
    )

    num_workers: NonNegativeInt = Field(
        default=4,
        description="Number of worker processes used by each DataLoader.",
        json_schema_extra={
            "omit_behavior": "Uses four worker processes.",
            "null_behavior": "Not allowed.",
        },
    )

    pin_memory: bool | Literal["auto"] = Field(
        default="auto",
        description=(
            "Whether DataLoaders use pinned CPU memory. `auto` enables it when "
            "CUDA is available and disables it otherwise."
        ),
        json_schema_extra={
            "omit_behavior": "Automatically selects based on CUDA availability.",
            "null_behavior": "Not allowed.",
        },
    )

    persistent_workers: bool = Field(
        default=False,
        description=(
            "Whether DataLoader worker processes remain alive between epochs. "
            "Enabling this requires `num_workers` to be greater than zero."
        ),
        json_schema_extra={
            "omit_behavior": "Workers are shut down after each epoch.",
            "null_behavior": "Not allowed.",
        },
    )

    drop_last: bool = Field(
        default=False,
        description=(
            "Whether the final incomplete training batch is discarded. Validation, "
            "test, and prediction loaders never discard their final batch."
        ),
        json_schema_extra={
            "omit_behavior": "Keeps the final incomplete training batch.",
            "null_behavior": "Not allowed.",
        },
    )


# -------------------------
# Full experiment configuration
# -------------------------
class TrainingConfig(_TrainingConfigBaseModel):
    """Complete configuration for a BenchRep training workflow.

    Requirements for model and data configuration depend on whether external
    model or datamodule objects are supplied at runtime.

    Use `benchrep.inspect_config(TrainingConfig)` to inspect this configuration.
    Nested configuration types shown in the output can be inspected the same
    way, for example `benchrep.inspect_config(TrainingTrainerConfig)` or
    `benchrep.inspect_config(MNISTDatasetConfig)`. Public configuration classes
    are available from `benchrep.assembly.schemas`.

    Use `benchrep.inspect_registry()` to discover component registries and
    `benchrep.inspect_registry("<registry>", "<component>")` to inspect a
    registered implementation.

    For machine-readable discovery, `TrainingConfig.model_json_schema()` returns
    standard JSON Schema, while `benchrep.list_registries()` and
    `benchrep.list_registered_components()` return structured registry data.
    """

    stage: Literal["training"] = Field(
        default="training",
        description="Identifies this configuration as a training workflow.",
        json_schema_extra={
            "omit_behavior": "Uses `training`.",
            "null_behavior": "Not allowed.",
        },
    )

    overrides: RuntimeOverridesConfig = Field(
        default_factory=RuntimeOverridesConfig,
        description=(
            "External model or datamodule requirements and constructor "
            "parameters."
        ),
        json_schema_extra={
            "omit_behavior": "Requires no external runtime components.",
            "null_behavior": "Not allowed.",
        },
    )

    run: TrainingRunConfig = Field(
        default_factory=TrainingRunConfig,
        description="Output location and run-identification settings.",
        json_schema_extra={
            "omit_behavior": "Uses the defaults defined by `TrainingRunConfig`.",
            "null_behavior": "Not allowed.",
        },
    )

    reproducibility: TrainingReproducibilityConfig = Field(
        default_factory=TrainingReproducibilityConfig,
        description="Training randomness and numerical reproducibility settings.",
        json_schema_extra={
            "omit_behavior": "Uses the defaults defined by `TrainingReproducibilityConfig`.",
            "null_behavior": "Not allowed.",
        },
    )

    model: TrainingModelConfig | None = Field(
        default=None,
        description="Model assembled for training.",
        json_schema_extra={
            "omit_behavior": (
                "Allowed when an external model is supplied; otherwise a model "
                "configuration is required."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Ignored when an external model object is supplied.",
            ],
        },
    )

    encoder: TrainingEncoderConfig | None = Field(
        default=None,
        description="Encoder used when assembling the configured model.",
        json_schema_extra={
            "omit_behavior": (
                "Allowed when an external model is supplied; otherwise an encoder "
                "configuration is required."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Ignored when an external model object is supplied.",
            ],
        },
    )

    decoder: TrainingDecoderConfig | None = Field(
        default=None,
        description="Decoder used when required by the configured model.",
        json_schema_extra={
            "omit_behavior": (
                "No decoder is configured. Config-built autoencoders and VAEs "
                "require this section."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Ignored when an external model object is supplied.",
            ],
        },
    )

    losses: dict[SupportedLossRole, LossRoleTerms] | None = Field(
        default_factory=dict,
        description="Loss terms grouped by role and registered component name.",
        json_schema_extra={
            "omit_behavior": (
                "Uses an empty loss mapping, which does not satisfy the loss "
                "requirements of config-built autoencoders or VAEs."
            ),
            "null_behavior": (
                "Allowed when an external model is supplied; otherwise a loss "
                "configuration is required."
            ),
            "notes": [
                "A configured custom objective bypasses only the structural "
                "requirements for standard loss roles.",
                "BenchRep does not verify that a custom objective reproduces any "
                "omitted reconstruction or regularization behavior.",
                "All configured terms across all roles contribute additively to the total loss.",
                "A nonempty `custom_objective` mapping may be used alone or alongside the "
                "standard roles.",
                "Without a custom objective, built-in autoencoders require `reconstruction`, "
                "while built-in VAEs require both `reconstruction` and `regularization`.",
                "Ignored when an external model object is supplied.",
            ],
        },
    )

    optimizer: TrainingOptimizerConfig | None = Field(
        default=None,
        description="Optimizer used to train the config-built model.",
        json_schema_extra={
            "omit_behavior": (
                "Allowed when an external model is supplied; otherwise an optimizer "
                "configuration is required."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Ignored when an external model object is supplied.",
            ],
        },
    )

    dataset: SupportedDatasetConfig | None = Field(
        default=None,
        description="Dataset built for training.",
        json_schema_extra={
            "omit_behavior": (
                "Allowed when an external datamodule is supplied; otherwise a "
                "dataset configuration is required."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Inspect registered dataset names and aliases with "
                '`benchrep.inspect_registry("dataset")`.',
                "Built-in dataset configuration types can be passed to "
                "`benchrep.inspect_config()` for detailed discovery.",
                "Parameters for user-registered datasets must match the registered "
                "dataset constructor.",
                "Ignored when an external datamodule object is supplied.",
            ],
        },
    )

    transforms: list[TrainingTransformConfig] = Field(
        default_factory=list,
        description=(
            "Ordered transform definitions applied to each dataset sample's `x` tensor "
            "before batching."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses empty training and validation transform sequences."
            ),
            "null_behavior": (
                "Not allowed; use an empty list instead."
            ),
            "notes": [
                "`apply_to` determines whether each transform is used for training, "
                "validation, or both.",
                "Relative ordering is preserved independently in the resulting training "
                "and validation pipelines.",
                "Validation-targeted transforms are inherited by linked prediction runs "
                "when prediction transforms are omitted or null.",
                "Ignored when an external datamodule object is supplied.",
            ],
        },
    )

    datamodule: TrainingDataModuleConfig | None = Field(
        default_factory=TrainingDataModuleConfig,
        description=(
            "Batching, data-loading, and train-validation splitting settings for "
            "BenchRep's internal datamodule."
        ),
        json_schema_extra={
            "omit_behavior": "Uses the defaults defined by `TrainingDataModuleConfig`.",
            "null_behavior": (
                "Allowed when an external datamodule is supplied; otherwise a "
                "datamodule configuration is required."
            ),
            "notes": [
                "Ignored when an external datamodule object is supplied.",
            ],
        },
    )

    trainer: TrainingTrainerConfig = Field(
        default_factory=TrainingTrainerConfig,
        description="Lightning Trainer settings used during training.",
        json_schema_extra={
            "omit_behavior": "Uses the defaults defined by `TrainingTrainerConfig`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Inherited as defaults by linked prediction runs.",
                "Declared and additional non-null fields are forwarded to "
                "`lightning.Trainer`, except for arguments managed internally by BenchRep.",
            ],
        },
    )

    logger: TrainingLoggerConfig | None = Field(
        default=None,
        description="Optional experiment logger used during training.",
        json_schema_extra={
            "omit_behavior": "Disables experiment logging.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    checkpointing: TrainingCheckpointConfig = Field(
        default_factory=TrainingCheckpointConfig,
        description="Checkpoint creation and selection settings for training.",
        json_schema_extra={
            "omit_behavior": "Uses the defaults defined by `TrainingCheckpointConfig`.",
            "null_behavior": "Not allowed.",
        },
    )

    early_stopping: TrainingEarlyStoppingConfig | None = Field(
        default=None,
        description="Optional metric-based early-stopping settings.",
        json_schema_extra={
            "omit_behavior": "Disables early stopping.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "When configured, BenchRep adds a Lightning `EarlyStopping` callback.",
            ],
        },
    )

    additional_callbacks: list[TrainingAdditionalCallbackConfig] = Field(
        default_factory=list,
        description="Additional registered Lightning callbacks used during training.",
        json_schema_extra={
            "omit_behavior": "Uses no additional callbacks.",
            "null_behavior": "Not allowed; use an empty list instead.",
            "notes": [
                "These supplement BenchRep-managed checkpointing and early stopping.",
                "They are not inherited by linked prediction runs.",
            ],
        },
    )

    inspection: TrainingInspectionConfig = Field(
        default_factory=TrainingInspectionConfig,
        description="Optional best-effort torchview model-graph export settings.",
        json_schema_extra={
            "omit_behavior": "Uses the defaults defined by `TrainingInspectionConfig`.",
            "null_behavior": "Not allowed.",
        },
    )

    @model_validator(mode="after")
    def validate_additional_callback_requirements(
        self,
    ) -> TrainingConfig:
        callback_names = {
            CALLBACKS.resolve_key(callback.name)
            for callback in self.additional_callbacks
        }

        callbacks_requiring_logger = sorted(
            callback_names & _LOGGER_REQUIRED_ADDITIONAL_CALLBACKS
        )

        if self.logger is None and callbacks_requiring_logger:
            formatted_names = ", ".join(
                repr(name)
                for name in callbacks_requiring_logger
            )

            raise ValueError(
                "`logger` must be configured when using the following "
                f"`additional_callbacks`: {formatted_names}."
            )

        return self

    @model_validator(mode="after")
    def validate_override_requirements(
            self,
            info: ValidationInfo,
    ) -> TrainingConfig:
        ctx = info.context or {}

        # Config declares external intent; the resolver checks actual overrides.
        model_overridden = (
                ctx.get("model_is_external", False)
                or self.overrides.model is not None
        )
        datamodule_overridden = (
                ctx.get("datamodule_overridden", False)
                or self.overrides.datamodule is not None
        )

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
    def validate_model_requirements(
            self,
            info: ValidationInfo,
    ) -> TrainingConfig:
        ctx = info.context or {}

        # Config markers declare external intent before entrypoint arguments
        # exist. The resolver verifies that those components were actually supplied.
        model_overridden = (
                ctx.get("model_is_external", False)
                or self.overrides.model is not None
        )

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

        has_reconstruction = bool(
            self.losses.get("reconstruction")
        )
        has_regularization = bool(
            self.losses.get("regularization")
        )
        has_custom_objective = bool(
            self.losses.get("custom_objective")
        )

        if model_cls is Autoencoder:
            if self.decoder is None:
                raise ValueError(
                    "Autoencoder requires a decoder config section."
                )

            if has_regularization:
                raise ValueError(
                    "Autoencoder does not support losses under "
                    "`losses.regularization`; use `losses.custom_objective` "
                    "for objectives requiring non-reconstruction model outputs."
                )

            if not has_reconstruction and not has_custom_objective:
                raise ValueError(
                    "Autoencoder requires at least one loss under either "
                    "`losses.reconstruction` or `losses.custom_objective`."
                )

        elif model_cls is VAE:
            if self.decoder is None:
                raise ValueError(
                    "VAE requires a decoder config section."
                )

            if "latent_dim" not in self.model.params:
                raise ValueError(
                    "VAE requires `model.params.latent_dim`."
                )

            latent_dim = self.model.params.get("latent_dim")

            if not isinstance(latent_dim, int) or latent_dim <= 0:
                raise ValueError(
                    "VAE requires `model.params.latent_dim` to be a "
                    "positive integer."
                )

            has_complete_standard_objective = (
                    has_reconstruction and has_regularization
            )

            if (
                    not has_custom_objective
                    and not has_complete_standard_objective
            ):
                raise ValueError(
                    "VAE requires either a nonempty "
                    "`losses.custom_objective` mapping or at least one loss "
                    "under both `losses.reconstruction` and "
                    "`losses.regularization`."
                )

        return self
