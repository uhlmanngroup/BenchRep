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
    StringConstraints,
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
from benchrep.assembly.schemas.composite_model_config_schema import (
    CompositeModelAssemblyStepConfig,
    CompositeModelComponentConfig,
    CompositeModelDeclarationsConfig,
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
    "contrastive",
    "classification",
    "regression",
    "custom_objective",
]

LossTermId: TypeAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    ),
]


_LOGGER_REQUIRED_ADDITIONAL_CALLBACKS = frozenset({
    "device_stats_monitor",
    "learning_rate_monitor",
})

_BENCHREP_MANAGED_TRAINER_ARGUMENTS = frozenset({
    "default_root_dir",
    "logger",
    "callbacks",
    "enable_checkpointing",
})

Float32MatmulPrecision: TypeAlias = Literal[
    "medium",
    "high",
    "highest",
]

StrictPositiveInt: TypeAlias = Annotated[
    int,
    Field(strict=True, gt=0),
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
    """Selects one of BenchRep's built-in model families.

    Use `benchrep.inspect_registry("model")` to inspect available model names
    and aliases, and `benchrep.inspect_registry("model", "<name>")` for details
    about a specific model family.

    Canonical autoencoders and VAEs define their architecture through the
    top-level `encoder` and `decoder` sections. Composite models instead use
    `composite_model_declarations`, `composite_model_components`, and
    `composite_model_assembly`.

    The model registry exposes BenchRep's built-in model families for discovery
    and configuration; it does not support custom model registration.
    """

    params: dict[
        Literal["latent_dim"],
        StrictPositiveInt,
    ] = Field(
        default_factory=dict,
        description="Model-family parameters used by a config-built model.",
        json_schema_extra={
            "omit_behavior": "Uses an empty parameter mapping.",
            "null_behavior": "Not allowed; use an empty mapping instead.",
            "notes": [
                "`vae` requires `latent_dim` as its only model-family parameter.",
                "`autoencoder` and `composite` do not accept model-family parameters.",
                "Constructor arguments for an externally supplied canonical model "
                "class belong in `overrides.model.params`, as described by "
                "`RuntimeOverridesConfig`.",
                "An externally supplied model instance must already be initialized "
                "and cannot receive constructor parameters.",
            ],
        },
    )


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

    `params` are passed as keyword arguments to the selected decoder constructor.
    For canonical autoencoders, BenchRep supplies `input_dim` from
    `encoder.output_dim`; for canonical VAEs, it supplies
    `model.params.latent_dim`. An explicitly configured `input_dim` is permitted
    only when it agrees with the value supplied by BenchRep.

    When the decoder accepts `initial_shape`, an explicitly configured value is
    used when present; otherwise, BenchRep attempts to infer it from
    `encoder.feature_shape`. If both are available, they must agree. Resolution
    fails when `initial_shape` is required but neither source provides it.

    User-registered decoders must satisfy BenchRep's decoder interface and must
    be registered again when reconstructing the model in a linked prediction
    process.
    """


# -------------------------
# Optimization/loss configuration
# -------------------------
class TrainingLossTermConfig(_TrainingConfigBaseModel):
    """Configuration for one weighted term in a role-specific loss list.

    `name` selects the registered loss, while the parent role selects the loss
    registry:

    - `reconstruction`: compares reconstructed and source images.
    - `regularization`: regularizes representations or model parameters.
    - `contrastive`: compares related or unrelated representations.
    - `classification`: evaluates categorical predictions.
    - `regression`: evaluates continuous predictions.
    - `custom_objective`: receives the complete batch and model-output mappings.

    `params` are passed only to the registered loss component's constructor.

    Canonical autoencoders and VAEs use fixed loss calling conventions.
    Reconstruction losses receive `reconstruction` and `target`, regularization
    losses receive `z_mu` and `z_logvar`, and custom objectives receive `batch`
    and `model_output`. A canonical-only ordinary loss may therefore be
    registered as `LossComponent(MyLoss)` without declaring runtime inputs.

    Composite models require every ordinary loss to declare its runtime
    parameter names and supported semantic roles through `LossTensorPort`
    entries. `composite_wiring` then maps those parameters to declarations under
    `expects` or `produces`. Contrastive, classification, and regression losses
    currently require this Composite contract because no canonical model uses
    those roles.

    Custom objectives follow one fixed interface under both model modes. Their
    component must subclass `BaseCustomObjectiveLoss`, `runtime_inputs` must be
    omitted, and `composite_wiring` must not be configured. BenchRep supplies
    the complete `batch` and `model_output` mappings automatically.

    Every configured term must return a scalar tensor. BenchRep multiplies that
    value by the configured `weight` before adding it to the other configured
    terms.

    Use `benchrep.inspect_registry("<loss-role>_loss")` to list registered
    losses. Pass a registered loss name as the second argument to inspect its
    constructor and Composite compatibility.

    User registrations must be repeated in each process that reconstructs an
    internally assembled model.

    A custom objective may use any fields available in the batch or model
    output. If it replaces reconstruction or regularization losses, it is
    responsible for implementing the omitted behavior.
    """

    name: str = Field(
        description="Registered loss name or alias to instantiate for this term.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )

    id: LossTermId | None = Field(
        default=None,
        description=(
            "Optional user-defined identifier used to distinguish multiple "
            "occurrences of the same registered loss within one loss role."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses the canonical registered loss name as the term identity. "
                "An identifier is required only when needed to keep resolved "
                "term identities unique within the role."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "The resolved term identity is `<id>_<canonical_loss_name>` when "
                "an identifier is configured.",
                "Aliases do not affect the canonical loss name used in the "
                "resolved identity.",
            ],
        },
    )

    weight: float = Field(
        default=1.0,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "Finite, nonnegative scalar coefficient applied to this raw loss "
            "before it is added to the total training loss. Weights are not "
            "normalized across losses or within loss roles."
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

    composite_wiring: dict[str, str] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Composite-only mapping from an ordinary loss component's "
            "forward() parameter names to declarations under `expects` "
            "or `produces`."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Omission is required for every loss under canonical models and for custom "
                "objectives under Composite. All other Composite losses require this field."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "The selected ordinary loss must declare matching "
                "LossTensorPort entries in its runtime contract.",
                "Keys must exactly match the loss component's declared "
                "runtime input names.",
                "Values must reference `expects.<name>` or "
                "`produces.<name>`.",
                "Custom objectives do not accept this field; BenchRep "
                "supplies `batch` and `model_output` automatically.",
            ],
        },
    )


LossRoleTerms: TypeAlias = Annotated[
    list[TrainingLossTermConfig],
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

    @model_validator(mode="after")
    def validate_benchrep_managed_arguments(
        self,
    ) -> TrainingTrainerConfig:
        configured_arguments = sorted(
            _BENCHREP_MANAGED_TRAINER_ARGUMENTS
            & set(self.model_extra or {})
        )

        if configured_arguments:
            formatted_arguments = ", ".join(
                repr(argument)
                for argument in configured_arguments
            )

            raise ValueError(
                "The following `trainer` arguments are managed by BenchRep "
                f"and cannot be configured directly: {formatted_arguments}. "
                "Use the top-level `run`, `logger`, `checkpointing`, "
                "`early_stopping`, and `additional_callbacks` sections instead."
            )

        return self


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

    When `monitor` names a metric and `save_top_k` is nonzero, checkpoints
    are ranked using that metric. `mode`, `save_top_k`, and `filename`
    configure these ranked checkpoints.

    Ranked checkpointing is disabled when `monitor=None` or `save_top_k=0`.
    When `monitor=None`, the resolver materializes this decision by setting
    `save_top_k=0`. Whenever ranked checkpointing is disabled,
    `save_last=True` is required so that training produces a checkpoint;
    `mode` and `filename` then have no effect.
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
                "Has no effect when `monitor=None` or `save_top_k=0`.",
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
                "Has no effect when `monitor=None` or `save_top_k=0`.",
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
    """Configures best-effort model-execution graph export with Torchview.

    When enabled, BenchRep performs this inspection after training completes.
    For canonical models, it reads `batch["x"]` from the first training batch,
    replaces its batch dimension with one, and passes the resulting input size
    to `torchview.draw_graph()`.

    For Composite models, BenchRep reads every input declared under
    `composite_model_declarations.expects` from the first training batch,
    retains the first observation from each tensor, and passes the resulting
    input mapping to the model as one positional argument.

    The resulting Graphviz graph is rendered as
    `torchview_model_graph.svg` in the training run's architecture directory.

    The graph represents the execution observed by Torchview for those inputs.
    It may not capture alternative data-dependent branches, dynamic control
    flow, other supported input shapes, training/evaluation differences, or
    operations unsupported by Torchview. It should therefore be treated as a
    diagnostic visualization rather than an authoritative description of
    every possible model execution.

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
            "notes": [
                "Validation size is calculated as "
                "`int(dataset_size * val_fraction)`; setup fails if this produces an "
                "empty validation subset.",
            ],
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

    @model_validator(mode="after")
    def validate_worker_configuration(
        self,
    ) -> TrainingDataModuleConfig:
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "`datamodule.persistent_workers=True` requires "
                "`datamodule.num_workers` to be greater than zero."
            )

        return self


# -------------------------
# Transform config
# -------------------------
class TrainingTransformStepConfig(NamedConfig):
    """Configuration for one transform in an ordered transform sequence.

    `params` are passed as keyword arguments to the registered transform
    constructor. The constructed object must be callable and must return a
    tensor when executed within a BenchRep transform pipeline.

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


class TrainingTransformPipelineConfig(_TrainingConfigBaseModel):
    """Configures one ordered, field-routed training transform pipeline.

    Pipelines execute in their configured list order. Each pipeline reads from
    the current sample mapping, applies its eligible transform steps in order,
    and assigns the resulting tensor to `output`. Consequently, a later
    pipeline may consume a field produced or overwritten by an earlier one.

    Canonical models support only the fixed in-place route `x` to `x`; both
    routing fields may be omitted or both may explicitly contain `x`.

    For Composite models, an omitted `input` resolves to the unique declaration
    having role `sample_image`. When multiple declarations have that role,
    `input` must be explicit. An omitted `output` resolves to the effective
    input name. Explicit names must reference image-valued declarations under
    `composite_model_declarations.expects`.

    When `input` and `output` differ, BenchRep clones the input tensor before
    applying transforms so the source field remains unchanged. Existing output
    fields are overwritten.
    """

    input: str | None = Field(
        default=None,
        description="Sample field from which this transform pipeline reads.",
        json_schema_extra={
            "omit_behavior": (
                "Canonical models use `x`. Composite models use the unique "
                "declaration assigned role `sample_image`; omission raises "
                "an error when multiple declarations have that role."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )

    output: str | None = Field(
        default=None,
        description="Sample field to which the transformed tensor is assigned.",
        json_schema_extra={
            "omit_behavior": "Uses the effective `input` field.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Canonical models require `x`.",
                "Composite models require an image-valued declaration under "
                "`composite_model_declarations.expects`.",
                "An existing field with this name is overwritten.",
            ],
        },
    )

    steps: list[TrainingTransformStepConfig] = Field(
        min_length=1,
        description="Ordered transform steps belonging to this routed pipeline.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Steps are filtered independently for training and validation "
                "according to `apply_to`.",
                "If no steps in this pipeline target a particular split, the "
                "pipeline is omitted from that split.",
                "Validation-targeted steps may be inherited by linked prediction.",
            ],
        },
    )

    @field_validator("input", "output")
    @classmethod
    def validate_field_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(
                "Transform pipeline input and output names must be nonempty."
            )

        return value


# -------------------------
# Full experiment configuration
# -------------------------
class TrainingConfig(_TrainingConfigBaseModel):
    """Complete configuration for a BenchRep training workflow.

    Requirements for model and data configuration depend on whether external
    model or datamodule objects are supplied at runtime.

    Config-built autoencoders and VAEs use the top-level `encoder` and `decoder`
    sections. Config-built Composite models instead define a declaration-driven
    execution graph through `composite_model_declarations`,
    `composite_model_components`, and `composite_model_assembly`.

    Whole-model runtime overrides are supported only by the canonical
    autoencoder and VAE entrypoints. Composite models must be assembled from
    configuration.

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
                "Whole-model overrides are supported only by the canonical autoencoder "
                "and VAE entrypoints.",
            ],
        },
    )

    encoder: TrainingEncoderConfig | None = Field(
        default=None,
        description="Encoder used to assemble a canonical autoencoder or VAE.",
        json_schema_extra={
            "omit_behavior": (
                "Required for config-built autoencoders and VAEs. It must be "
                "omitted for Composite models and may be omitted when an "
                "external canonical model is supplied."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Ignored and removed from the resolved configuration when an "
                "external model is supplied.",
                "Composite encoders are declared under "
                "`composite_model_components`.",
            ],
        },
    )

    decoder: TrainingDecoderConfig | None = Field(
        default=None,
        description="Decoder used to assemble a canonical autoencoder or VAE.",
        json_schema_extra={
            "omit_behavior": (
                "Required for config-built autoencoders and VAEs. It must be "
                "omitted for Composite models and may be omitted when an "
                "external canonical model is supplied."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Ignored and removed from the resolved configuration when an "
                "external model is supplied.",
                "Composite decoders are declared under "
                "`composite_model_components`.",
            ],
        },
    )

    composite_model_declarations: (
        CompositeModelDeclarationsConfig | None
    ) = Field(
        default=None,
        description=(
            "Semantic declarations for the Composite model's batch inputs, "
            "batch metadata, and produced model outputs."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Required when `model.name` resolves to `composite`; otherwise "
                "this section must be omitted."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "At least one input must have role `sample_image`.",
                "At most one batch-metadata field may have role `index`.",
                "Declared input and metadata names identify fields expected in "
                "each dataset sample and collated batch.",
                "Every declared output must be produced exactly once by the "
                "assembly graph.",
                "Ignored and removed from the resolved configuration when an "
                "external model is supplied.",
            ],
        },
    )

    composite_model_components: (
        dict[str, CompositeModelComponentConfig] | None
    ) = Field(
        default=None,
        min_length=1,
        description=(
            "Reusable architecture components available to the Composite "
            "assembly graph, keyed by user-defined component ID."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Required when `model.name` resolves to `composite`; otherwise "
                "this section must be omitted."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Each component selects the encoder, decoder, or head registry.",
                "Composite compatibility is defined by each registered "
                "`ArchitectureComponent` runtime contract; Composite encoders and "
                "decoders do not need to satisfy the canonical `BaseEncoder` or "
                "`BaseDecoder` interfaces.",
                "Each component ID is instantiated once and may be invoked by "
                "multiple assembly steps, thereby sharing parameters.",
                "Every configured component must be used by at least one "
                "assembly step.",
                "Ignored and removed from the resolved configuration when an "
                "external model is supplied.",
            ],
        },
    )

    composite_model_assembly: (
        dict[str, CompositeModelAssemblyStepConfig] | None
    ) = Field(
        default=None,
        min_length=1,
        description=(
            "Composite execution steps, keyed by user-defined step ID, that "
            "wire declared inputs and produced outputs through configured "
            "components."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Required when `model.name` resolves to `composite`; otherwise "
                "this section must be omitted."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A component ID may be reused by multiple steps.",
                "Step execution order is derived from output dependencies, not "
                "from mapping order.",
                "Dependency cycles are rejected.",
                "Inputs must reference `expects.<name>` or `produces.<name>`.",
                "Outputs must reference declarations under `produces`.",
                "Ignored and removed from the resolved configuration when an "
                "external model is supplied.",
            ],
        },
    )

    losses: dict[SupportedLossRole, LossRoleTerms] | None = Field(
        default_factory=dict,
        description="Loss terms grouped by role.",
        json_schema_extra={
            "omit_behavior": (
                "Uses an empty loss configuration, which does not satisfy the loss "
                "requirements of config-built models."
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
                "A nonempty `custom_objective` list may be used alone or alongside the "
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

    transform_pipelines: list[TrainingTransformPipelineConfig] = Field(
        default_factory=list,
        description=(
            "Ordered, split-specific transform pipelines routed between "
            "tensor-valued dataset-sample fields before batching."
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
            _require_present(self.losses, "losses")
            _require_present(self.optimizer, "optimizer")

            assert self.model is not None

            model_name = MODELS.resolve_key(self.model.name)

            if model_name != "composite":
                _require_present(self.encoder, "encoder")
                _require_present(self.decoder, "decoder")

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
        assert self.losses is not None
        assert self.optimizer is not None

        model_name = MODELS.resolve_key(self.model.name)

        if model_name != "vae" and self.model.params:
            raise ValueError(
                f"`model.params` must be empty for {model_name!r}; only `vae` "
                "accepts model-family parameters."
            )

        if model_name == "composite":
            return self

        composite_only_loss_roles = (
            "contrastive",
            "classification",
            "regression",
        )

        configured_composite_only_roles = [
            f"`losses.{role}`"
            for role in composite_only_loss_roles
            if self.losses.get(role)
        ]

        if configured_composite_only_roles:
            raise ValueError(
                "The following loss roles are supported only by Composite models: "
                + ", ".join(configured_composite_only_roles)
            )

        assert self.encoder is not None
        assert self.decoder is not None

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

            has_complete_standard_objective = (
                    has_reconstruction and has_regularization
            )

            if (
                    not has_custom_objective
                    and not has_complete_standard_objective
            ):
                raise ValueError(
                    "VAE requires either a nonempty "
                    "`losses.custom_objective` list or at least one loss "
                    "under both `losses.reconstruction` and "
                    "`losses.regularization`."
                )

        return self

    @model_validator(mode="after")
    def validate_loss_requirements(
            self,
            info: ValidationInfo,
    ) -> TrainingConfig:
        ctx = info.context or {}

        model_overridden = (
                ctx.get("model_is_external", False)
                or self.overrides.model is not None
        )

        if model_overridden or self.model is None:
            return self

        model_name = MODELS.resolve_key(self.model.name)

        if model_name == "composite" and not self.losses:
            raise ValueError(
                "Composite models require at least one configured loss."
            )

        if self.losses is None:
            return self

        configured_wiring = [
            f"`losses.{loss_role}[{loss_index}].composite_wiring`"
            for loss_role, loss_terms in self.losses.items()
            for loss_index, loss_config in enumerate(loss_terms)
            if loss_config.composite_wiring is not None
        ]

        if model_name != "composite":
            if configured_wiring:
                raise ValueError(
                    "`composite_wiring` may only be configured when "
                    "`model.name` is `composite`: "
                    + ", ".join(configured_wiring)
                )

            return self

        custom_objective_wiring = [
            f"`losses.custom_objective[{loss_index}].composite_wiring`"
            for loss_index, loss_config
            in enumerate(self.losses.get("custom_objective", []))
            if loss_config.composite_wiring is not None
        ]

        if custom_objective_wiring:
            raise ValueError(
                "Custom objective losses do not accept `composite_wiring`; "
                "`batch` and `model_output` are supplied automatically: "
                + ", ".join(custom_objective_wiring)
            )

        missing_wiring = [
            f"`losses.{loss_role}[{loss_index}].composite_wiring`"
            for loss_role, loss_terms in self.losses.items()
            if loss_role != "custom_objective"
            for loss_index, loss_config in enumerate(loss_terms)
            if loss_config.composite_wiring is None
        ]

        if missing_wiring:
            raise ValueError(
                "Every non-custom-objective loss under Composite requires "
                "`composite_wiring`: "
                + ", ".join(missing_wiring)
            )

        return self

    @model_validator(mode="after")
    def validate_composite_requirements(
            self,
            info: ValidationInfo,
    ) -> TrainingConfig:
        ctx = info.context or {}

        model_overridden = (
                ctx.get("model_is_external", False)
                or self.overrides.model is not None
        )

        if model_overridden or self.model is None:
            return self

        model_name = MODELS.resolve_key(self.model.name)

        # Reject composite-model sections when using canonical models
        composite_sections = {
            "composite_model_declarations": self.composite_model_declarations,
            "composite_model_components": self.composite_model_components,
            "composite_model_assembly": self.composite_model_assembly,
        }

        if model_name != "composite":
            configured = [
                name
                for name, value in composite_sections.items()
                if value is not None
            ]

            if configured:
                raise ValueError(
                    "Composite model configuration sections may only be used with "
                    "`model.name: composite`: "
                    + ", ".join(configured)
                )

            return self

        # Require composite-model sections and reject canonical architecture sections when using composite-model
        _require_present(self.composite_model_declarations, "composite_model_declarations")
        _require_present(self.composite_model_components, "composite_model_components")
        _require_present(self.composite_model_assembly, "composite_model_assembly")

        assert self.composite_model_declarations is not None
        assert self.composite_model_components is not None
        assert self.composite_model_assembly is not None

        if self.encoder is not None or self.decoder is not None:
            raise ValueError(
                "Composite models define architecture through `composite_model_components` "
                "and `composite_model_assembly`; top-level `encoder` and `decoder` sections "
                "are not supported."
            )

        # Validate composite-model input declarations
        sample_inputs = [
            name
            for name, role
            in self.composite_model_declarations.expects.items()
            if role == "sample_image"
        ]

        if not sample_inputs:
            raise ValueError(
                "Composite models require at least one input with role `sample_image`."
            )

        if self.composite_model_declarations.batch_metadata is not None:
            index_fields = [
                name
                for name, role
                in self.composite_model_declarations.batch_metadata.items()
                if role == "index"
            ]

            if len(index_fields) > 1:
                raise ValueError(
                    "Composite models support at most one batch metadata field "
                    "with `role: index`."
                )

        # Validate assembly component references
        unknown_components = sorted({
            step.component
            for step in self.composite_model_assembly.values()
            if step.component not in self.composite_model_components
        })

        if unknown_components:
            raise ValueError(
                "Composite model assembly references undefined components: "
                + ", ".join(repr(name) for name in unknown_components)
            )

        # Validate assembly-produced outputs
        valid_output_names = set(
            self.composite_model_declarations.produces
        )

        output_references: list[tuple[str, str]] = []

        for step_name, step in self.composite_model_assembly.items():
            if isinstance(step.outputs, str):
                output_references.append(
                    (
                        f"composite_model_assembly.{step_name}.outputs",
                        step.outputs,
                    )
                )
            else:
                output_references.extend(
                    (
                        (
                            f"composite_model_assembly.{step_name}."
                            f"outputs.{output_name}"
                        ),
                        source,
                    )
                    for output_name, source in step.outputs.items()
                )

        produced_outputs: list[str] = []
        invalid_output_references: list[str] = []

        for location, source in output_references:
            namespace, separator, name = source.partition(".")

            if (
                    separator != "."
                    or namespace != "produces"
                    or name not in valid_output_names
            ):
                invalid_output_references.append(
                    f"{location} -> {source!r}"
                )
                continue

            produced_outputs.append(name)

        if invalid_output_references:
            raise ValueError(
                "Composite model assembly outputs must reference declared "
                "`produces` values: "
                + ", ".join(invalid_output_references)
            )

        duplicate_outputs = sorted({
            name
            for name in produced_outputs
            if produced_outputs.count(name) > 1
        })

        if duplicate_outputs:
            raise ValueError(
                "Composite model produced data may originate from only one "
                "assembly step: "
                + ", ".join(repr(name) for name in duplicate_outputs)
            )

        unproduced_outputs = sorted(
            valid_output_names - set(produced_outputs)
        )

        if unproduced_outputs:
            raise ValueError(
                "Composite model declares produced data that no assembly step "
                "generates: "
                + ", ".join(repr(name) for name in unproduced_outputs)
            )

        # Validate assembly and loss input references
        valid_input_names = set(self.composite_model_declarations.expects)
        invalid_references: list[str] = []

        for step_name, step in self.composite_model_assembly.items():
            for argument_name, source in step.inputs.items():
                namespace, separator, name = source.partition(".")

                if separator != ".":
                    invalid_references.append(
                        f"{step_name}.{argument_name} -> {source!r}"
                    )
                    continue

                if namespace == "expects":
                    valid_names = valid_input_names
                elif namespace == "produces":
                    valid_names = valid_output_names
                else:
                    invalid_references.append(
                        f"{step_name}.{argument_name} -> {source!r}"
                    )
                    continue

                if name not in valid_names:
                    invalid_references.append(
                        f"{step_name}.{argument_name} -> {source!r}"
                    )

        assert self.losses is not None

        for loss_role, loss_terms in self.losses.items():
            for loss_index, loss_config in enumerate(loss_terms):
                if loss_config.composite_wiring is None:
                    continue

                for parameter_name, source in loss_config.composite_wiring.items():
                    namespace, separator, name = source.partition(".")

                    reference = (
                        f"losses.{loss_role}[{loss_index}]."
                        f"composite_wiring.{parameter_name}"
                    )

                    if separator != ".":
                        invalid_references.append(
                            f"{reference} -> {source!r}"
                        )
                        continue

                    if namespace == "expects":
                        valid_names = valid_input_names
                    elif namespace == "produces":
                        valid_names = valid_output_names
                    else:
                        invalid_references.append(
                            f"{reference} -> {source!r}"
                        )
                        continue

                    if name not in valid_names:
                        invalid_references.append(
                            f"{reference} -> {source!r}"
                        )

        if invalid_references:
            raise ValueError(
                "Composite model configuration contains invalid runtime source references: "
                + ", ".join(invalid_references)
            )

        return self

