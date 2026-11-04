from __future__ import annotations

from pathlib import Path
from typing import (
    Annotated,
    Any,
    Final,
    Generic,
    get_args,
    Literal,
    TypeVar,
    TypeAlias,
)

from pydantic import (
    BaseModel,
    BeforeValidator,
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

from benchrep.assembly.registries.core import (
    CALLBACKS,
    MODELS,
    LOSS_WEIGHT_SCHEDULES,
)
from benchrep.assembly.registries.utils import normalize_name, PathInput
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

    @field_validator("name", mode="before")
    @classmethod
    def reject_blank_name(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Component name must not be blank.")
        return value


# -------------------------
# Run/output configuration
# -------------------------
class TrainingRunConfig(_TrainingConfigBaseModel):
    """Output location and run identity shared by linked workflows."""

    output_root: PathInput = Field(
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

    run_name_prefix: str | None = Field(
        default=None,
        description="Optional prefix used to construct generated run names.",
        json_schema_extra={
            "omit_behavior": "No run name prefix is added.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "The run name prefix is reused by linked prediction runs and propagated "
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

    @field_validator("output_root", mode="before")
    @classmethod
    def reject_blank_output_root(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError(
                "output_root must not be blank. "
                "Omit it to use 'outputs/', or use '.' explicitly "
                "for the working directory."
            )
        return value


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
# Loss configuration
# -------------------------
class RampScheduleParams(_TrainingConfigBaseModel):
    """Defines the positions and endpoint factors of a bounded ramp.

    Position units are selected by the enclosing schedule's `interval`.
    Before and at `start`, the factor is `start_factor`. At and after `end`,
    it is `end_factor`. The selected schedule determines interpolation
    between those positions.

    The end position must be greater than the start position. Factors may
    increase, decrease, or remain constant.
    """

    start: int = Field(
        default=0,
        strict=True,
        ge=0,
        description="Position at which the ramp starts.",
        json_schema_extra={
            "omit_behavior": "Starts at position 0.",
            "null_behavior": "Not allowed.",
        },
    )

    end: int = Field(
        strict=True,
        gt=0,
        description="Position at which the ramp reaches its final factor.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Must be greater than `start`.",
                "With cycling enabled, must be less than `cycle.length`.",
            ],
        },
    )

    start_factor: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Multiplier held before and at the start position.",
        json_schema_extra={
            "omit_behavior": "Uses 0.0.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Multiplies the loss term's base weight.",
                "Zero removes the loss contribution but retains computation "
                "and logging.",
            ],
        },
    )

    end_factor: float = Field(
        default=1.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Multiplier held at and after the end position.",
        json_schema_extra={
            "omit_behavior": "Uses 1.0, restoring the configured base weight.",
            "null_behavior": "Not allowed.",
            "notes": [
                "May exceed one or be smaller than `start_factor`.",
            ],
        },
    )

    @model_validator(mode="after")
    def _validate_positions(self) -> RampScheduleParams:
        if self.end <= self.start:
            raise ValueError("`end` must be greater than `start`.")
        return self


class SigmoidScheduleParams(RampScheduleParams):
    """Defines an endpoint-normalized sigmoid ramp.

    Normalization makes the ramp reach the configured endpoint factors
    exactly, rather than only approaching them asymptotically.
    """

    steepness: float = Field(
        default=10.0,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Steepness over the normalized ramp interval. Larger values "
            "concentrate the transition around its midpoint."
        ),
        json_schema_extra={
            "omit_behavior": "Uses a steepness of 10.0.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Changes the transition shape without changing its positions "
                "or endpoint factors.",
            ],
        },
    )


class ExponentialScheduleParams(RampScheduleParams):
    """Defines geometric interpolation between positive endpoint factors.

    Unlike the other ramps, this schedule cannot start or end at zero.
    `start_factor` must be supplied explicitly.
    """

    start_factor: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="Strictly positive multiplier held before and at `start`.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )

    end_factor: float = Field(
        default=1.0,
        gt=0.0,
        allow_inf_nan=False,
        description="Strictly positive multiplier held at and after `end`.",
        json_schema_extra={
            "omit_behavior": "Uses 1.0, restoring the configured base weight.",
            "null_behavior": "Not allowed.",
            "notes": [
                "May be greater than, equal to, or smaller than `start_factor`.",
            ],
        },
    )


SchedulePosition: TypeAlias = Annotated[
    int, Field(strict=True, ge=0)
]
ScheduleFactor: TypeAlias = Annotated[
    float, Field(ge=0.0, allow_inf_nan=False)
]


class PiecewiseScheduleParams(_TrainingConfigBaseModel):
    """Defines a schedule using ordered position/factor pairs.

    `piecewise_linear` interpolates between consecutive milestones.
    `piecewise_constant` holds each milestone's factor until the next one,
    switching to the new factor at that milestone's position.

    Before the first milestone and after the last, the nearest endpoint
    factor is held.
    """

    milestones: list[tuple[SchedulePosition, ScheduleFactor]] = Field(
        min_length=2,
        description=(
            "At least two [position, factor] pairs with strictly increasing "
            "nonnegative integer positions and finite, nonnegative factors."
        ),
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Position units follow the enclosing schedule's `interval`.",
                "The first position need not be zero.",
                "Factors may increase, decrease, remain constant, or exceed one.",
                "With cycling enabled, the last position must be less than "
                "`cycle.length`.",
            ],
        },
    )

    @model_validator(mode="after")
    def _validate_milestones(self) -> PiecewiseScheduleParams:
        positions = [position for position, _ in self.milestones]
        if any(right <= left for left, right in zip(positions, positions[1:])):
            raise ValueError(
                "Milestone positions must be strictly increasing."
            )
        return self


class LossWeightScheduleCycleConfig(_TrainingConfigBaseModel):
    """Repeats a schedule using positions local to each cycle.

    The wrapped schedule receives `position % length`, ranging from zero
    through `length - 1`. Cycles start at training position zero and repeat
    indefinitely. Resetting may produce a jump between endpoint factors.
    """

    length: int = Field(
        strict=True,
        gt=0,
        description="Cycle length in the enclosing schedule's interval units.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "For built-in ramps, `end` must be less than this length.",
                "For piecewise schedules, the final milestone position must "
                "be less than this length.",
                "Custom schedules define their own parameter semantics; "
                "BenchRep does not infer their endpoint positions.",
            ],
        },
    )


ScheduleParamsT = TypeVar("ScheduleParamsT")


class TrainingLossWeightScheduleConfig(
    _TrainingConfigBaseModel, Generic[ScheduleParamsT]
):
    """Configures a registered multiplier for a loss term's base weight.

    The effective loss weight is `weight * schedule.factor(position)`.
    Factors must be finite and nonnegative, but may exceed one. Scheduling
    changes the loss contribution without disabling its computation.

    Position is either the zero-based epoch index or the number of completed
    optimizer steps, selected by `interval`. The same schedule is evaluated
    during training and validation using the model's current position.
    Consequently, scheduled weights also affect weighted validation losses
    used for checkpointing or early stopping.

    Built-in ramps hold their endpoint factors outside the configured ramp
    interval. Piecewise schedules hold their nearest endpoint factor outside
    the milestone range. An optional cycle repeats the schedule from position
    zero rather than holding its final factor indefinitely.

    Use `benchrep.inspect_registry("loss_weight_schedule")` to discover
    registered schedules, or supply a schedule name as the second argument
    to inspect its constructor and contract.

    Custom schedules receive constructor parameters from `params` and expose
    `factor(position)`. They do not receive the model, batch, or logged metrics.
    Their registrations must be available in each process that reconstructs
    the model.
    """

    name: str = Field(
        description="Registered schedule name.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )
    params: ScheduleParamsT = Field(
        description="Constructor parameters for the selected schedule.",
        json_schema_extra={
            "omit_behavior": (
                "Required for built-in schedules. Custom schedules default "
                "to an empty parameter mapping."
            ),
            "null_behavior": "Not allowed.",
            "notes": [
                "Built-in schedules use their corresponding typed parameter "
                "schemas. Custom schedule parameters are checked against "
                "the registered constructor.",
            ],
        },
    )
    interval: Literal["epoch", "step"] = Field(
        default="epoch",
        description=(
            "Schedule position: zero-based epoch index or number of completed "
            "optimizer steps. Gradient accumulation does not advance the "
            "step position until an optimizer step completes."
        ),
        json_schema_extra={
            "omit_behavior": "Uses epoch-based scheduling.",
            "null_behavior": "Not allowed.",
        },
    )
    cycle: LossWeightScheduleCycleConfig | None = Field(
        default=None,
        description=(
            "Optional repetition of the configured schedule. Cycles start "
            "at training position zero and repeat indefinitely."
        ),
        json_schema_extra={
            "omit_behavior": "The schedule does not repeat.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    @model_validator(mode="after")
    def _validate_cycle(self) -> TrainingLossWeightScheduleConfig:
        if self.cycle is None:
            return self

        if isinstance(self.params, RampScheduleParams):
            last_position = self.params.end
        elif isinstance(self.params, PiecewiseScheduleParams):
            last_position = self.params.milestones[-1][0]
        else:
            # Custom constructors own their parameter semantics.
            return self

        if last_position >= self.cycle.length:
            raise ValueError(
                "The schedule's final position must be less than "
                "`cycle.length` so its endpoint is reached before resetting."
            )
        return self

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalize_name(value, field_name="weight_schedule.name")
        return value


class LinearScheduleConfig(
    TrainingLossWeightScheduleConfig[RampScheduleParams]
):
    """Linear interpolation between the endpoint factors."""

    name: Literal["linear"] = Field(
        description="Selects the linear loss-weight schedule.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )


class CosineScheduleConfig(
    TrainingLossWeightScheduleConfig[RampScheduleParams]
):
    """Half-cosine interpolation between the endpoint factors."""

    name: Literal["cosine"] = Field(
        description="Selects the cosine loss-weight schedule.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )


class SigmoidScheduleConfig(
    TrainingLossWeightScheduleConfig[SigmoidScheduleParams]
):
    """Sigmoid interpolation normalized to reach both endpoint factors."""

    name: Literal["sigmoid"] = Field(
        description="Selects the sigmoid loss-weight schedule.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )


class ExponentialScheduleConfig(
    TrainingLossWeightScheduleConfig[ExponentialScheduleParams]
):
    """Geometric interpolation between endpoint factors."""

    name: Literal["exponential"] = Field(
        description="Selects the exponential loss-weight schedule.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )


class PiecewiseLinearScheduleConfig(
    TrainingLossWeightScheduleConfig[PiecewiseScheduleParams]
):
    """Linear interpolation between consecutive milestones."""

    name: Literal["piecewise_linear"] = Field(
        description="Selects the piecewise-linear loss-weight schedule.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )


class PiecewiseConstantScheduleConfig(
    TrainingLossWeightScheduleConfig[PiecewiseScheduleParams]
):
    """Hold each milestone's factor until the next milestone."""

    name: Literal["piecewise_constant"] = Field(
        description="Selects the piecewise-constant loss-weight schedule.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
        },
    )


class CustomScheduleConfig(
    TrainingLossWeightScheduleConfig[dict[str, Any]]
):
    """Constructor-validated parameters for a user-registered schedule."""

    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the registered constructor.",
        json_schema_extra={
            "omit_behavior": "Uses an empty parameter mapping.",
            "null_behavior": "Not allowed; use an empty mapping instead.",
        },
    )


def _normalize_loss_weight_schedule_config(value: Any) -> Any:
    # Revalidate supplied config objects through the same dispatch path.
    if isinstance(value, TrainingLossWeightScheduleConfig):
        value = value.model_dump()

    if not isinstance(value, dict):
        return value

    name = value.get("name")
    if not isinstance(name, str):
        return value

    name = normalize_name(name, field_name="weight_schedule.name")
    try:
        name = LOSS_WEIGHT_SCHEDULES.resolve_key(name)
    except KeyError:
        # Unknown names use the custom schema; the resolver later checks
        # that the requested implementation is registered.
        pass

    return {**value, "name": name}


def _loss_weight_schedule_discriminator(value: Any) -> str:
    name = (
        value.get("name")
        if isinstance(value, dict)
        else getattr(value, "name", None)
    )
    if isinstance(name, str) and name in BUILTIN_LOSS_WEIGHT_SCHEDULE_NAMES:
        return name
    return "custom"


SupportedLossWeightScheduleConfig: TypeAlias = Annotated[
    Annotated[LinearScheduleConfig, Tag("linear")]
    | Annotated[CosineScheduleConfig, Tag("cosine")]
    | Annotated[SigmoidScheduleConfig, Tag("sigmoid")]
    | Annotated[ExponentialScheduleConfig, Tag("exponential")]
    | Annotated[PiecewiseLinearScheduleConfig, Tag("piecewise_linear")]
    | Annotated[PiecewiseConstantScheduleConfig, Tag("piecewise_constant")]
    | Annotated[CustomScheduleConfig, Tag("custom")],
    Discriminator(_loss_weight_schedule_discriminator),
    BeforeValidator(_normalize_loss_weight_schedule_config),
]


def _get_builtin_loss_weight_schedule_names() -> frozenset[str]:
    """Extract literal names from the typed schedule union."""
    schedule_union = get_args(SupportedLossWeightScheduleConfig)[0]
    names: set[str] = set()

    for tagged_config in get_args(schedule_union):
        config_class = get_args(tagged_config)[0]
        name_annotation = config_class.model_fields["name"].annotation

        # Literal["linear"] contributes "linear"; custom's str contributes
        # nothing because get_args(str) is empty.
        names.update(get_args(name_annotation))

    return frozenset(names)


BUILTIN_LOSS_WEIGHT_SCHEDULE_NAMES: Final[frozenset[str]] = (
    _get_builtin_loss_weight_schedule_names()
)


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
    value by its effective weight before adding it to the other configured
    terms. The effective weight is `weight`, optionally multiplied by the
    factor supplied by `weight_schedule`.

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
        title="Loss term ID",
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

    weight_schedule: SupportedLossWeightScheduleConfig | None = Field(
        default=None,
        description=(
            "Optional schedule multiplying the configured base weight. "
            "Effective weight equals `weight * schedule_factor`; factors "
            "must be finite and nonnegative and may exceed 1."
        ),
        json_schema_extra={
            "omit_behavior": "Uses the configured weight unchanged.",
            "null_behavior": "Equivalent to omission.",
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



# -------------------------
# Optimizer configuration
# -------------------------
class TrainingLRSchedulerConfig(NamedConfig):
    """Configures a registered learning-rate scheduler for the optimizer.

    `name` selects a scheduler from the LR-scheduler registry. `params` are
    constructor keyword arguments, excluding `optimizer`, which BenchRep
    supplies when Lightning initializes the model's optimizer.

    The registered constructor or factory must return a PyTorch `LRScheduler`
    or `ReduceLROnPlateau` instance attached to the supplied optimizer.

    `interval` and `frequency` control when Lightning advances the scheduler.
    A `ReduceLROnPlateau` instance additionally requires `monitor` to identify
    a logged metric. Other schedulers advance without receiving that metric.

    These schedules change the optimizer's learning rate. Loss-weight
    schedules are configured separately on individual loss terms.

    Use `benchrep.inspect_registry("lr_scheduler")` to list registered
    schedulers, or supply a scheduler name as the second argument to inspect
    its constructor and documentation.

    Custom registrations must be available in each process that reconstructs
    the config-built model.
    """

    interval: Literal["epoch", "step"] = Field(
        default="epoch",
        description=(
            "Whether Lightning advances the scheduler at epoch or "
            "optimizer-step intervals."
        ),
        json_schema_extra={
            "omit_behavior": "Uses epoch-based scheduling.",
            "null_behavior": "Not allowed.",
            "notes": [
                "For step-based scheduling, gradient accumulation does not "
                "cause a scheduler update on every training batch.",
                "For metric-driven scheduling, the monitored metric must "
                "be available when Lightning updates the scheduler.",
            ],
        },
    )

    frequency: int = Field(
        default=1,
        strict=True,
        gt=0,
        description="Number of configured intervals between scheduler updates.",
        json_schema_extra={
            "omit_behavior": "Updates at every configured interval.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Passed to Lightning together with `interval`.",
                "For a validation-monitored scheduler, coordinate this "
                "frequency with the Trainer's validation-check frequency.",
            ],
        },
    )

    monitor: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Exact logged metric key supplied to a ReduceLROnPlateau scheduler."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Allowed for ordinary schedulers. ReduceLROnPlateau requires "
                "a nonblank metric key when the scheduler is constructed."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "For example, `losses/val/total_weighted_loss`.",
                "The metric must be logged by the model and available at "
                "the scheduler's update time.",
                "Configure the scheduler's optimization direction through "
                "`params.mode` when supported.",
                "This field does not select checkpointing or early-stopping "
                "metrics; those have separate monitor settings.",
            ],
        },
    )

    @field_validator("params")
    @classmethod
    def reject_optimizer_override(
        cls, value: dict[str, Any]
    ) -> dict[str, Any]:
        if "optimizer" in value:
            raise ValueError(
                "`optimizer` is supplied by BenchRep and cannot appear "
                "in scheduler params."
            )
        return value


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

    lr_scheduler: TrainingLRSchedulerConfig | None = Field(
        default=None,
        title="LR scheduler",
        description="Optional LR scheduler attached to this optimizer.",
        json_schema_extra={
            "omit_behavior": "No LR scheduler is constructed.",
            "null_behavior": "Equivalent to omission.",
        },
    )


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
        title="Float32 matrix multiplication precision",
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
        title="Maximum epochs",
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

    wandb_api_key_path: PathInput | None = Field(
        default=None,
        title="W&B API key path",
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
        default="losses/val/total_weighted_loss",
        min_length=1,
        description="Metric key used to rank checkpoints.",
        json_schema_extra={
            "omit_behavior": "Monitors `losses/val/total_weighted_loss`.",
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
        default="losses/val/total_weighted_loss",
        min_length=1,
        description="Metric key monitored for early stopping.",
        json_schema_extra={
            "omit_behavior": "Monitors `losses/val/total_weighted_loss`.",
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
                "With `min`, lower values are better. Improvement requires "
                "a decrease greater than `min_delta`; `stopping_threshold` "
                "is crossed below its value and `divergence_threshold` "
                "is crossed above its value.",
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
                "An improvement greater than `min_delta` resets the counter.",
                "Patience counts metric checks, not necessarily epochs.",
                "Does not apply to immediate stopping, divergence, or non-finite-value "
                "conditions.",
            ],
        },
    )

    min_delta: float = Field(
        default=0.0,
        title="Minimum improvement",
        description=(
            "Absolute improvement threshold. The monitored metric must "
            "improve by more than this amount to reset patience."
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
                "When false, Lightning skips the check if the metric is "
                "missing. A warning is emitted when the callback's "
                "`verbose` setting is enabled.",
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
    """Selects and configures an additional registered Lightning callback.

    `name` selects an entry in BenchRep's callback registry. `params` are
    passed as keyword arguments to the registered constructor or factory,
    which must return a Lightning Callback instance.

    Use `benchrep.inspect_registry("callback")` to list registered callbacks,
    or supply a callback name as the second argument to inspect its
    constructor and documentation. Custom callbacks must be registered
    before the training configuration is validated.

    Additional callbacks supplement BenchRep-managed checkpointing and
    early stopping. They apply only to training and are not inherited by
    linked prediction runs.

    The built-in `device_stats_monitor` and `learning_rate_monitor`
    callbacks require a configured training logger.

    Callbacks requiring live Python objects can instead be constructed by
    the caller and passed through the training entrypoint's
    `runtime_callbacks` argument. Those instances are appended after
    configured additional callbacks and do not replace them.

    Configured callback names and parameters are recorded in the resolved
    configuration. Runtime callback instances are recorded separately by
    their fully qualified class names; their constructor state is not
    serialized into the configuration.
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


class _GradientNormConfig(_TrainingConfigBaseModel):
    """Selects the model parameters and norm used by one inspection request.

    All selected gradient elements are treated as one vector. Norms are not
    averaged across parameter tensors. Frozen parameters are excluded, and
    parameters unused by the measured loss contribute no gradient.

    A connected zero gradient is reported as zero. Unavailable measurements
    are skipped with a warning and retried at subsequent sampling positions.
    """

    target: str = Field(
        min_length=1,
        description=(
            "Model component whose parameter gradients are inspected, "
            "or `whole_model`."
        ),
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Config-built canonical models support `encoder`, `decoder`, "
                "and `whole_model`. Canonical VAEs also support "
                "`variational_head`.",
                "Composite models support configured component IDs and "
                "`whole_model`.",
                "External models support only optimizer-gradient inspection "
                "with target `whole_model`.",
                "`whole_model` is reserved when Composite gradient "
                "inspection is enabled.",
            ],
        },
    )

    norm_type: Literal["l1", "l2", "inf"] = Field(
        default="l2",
        description=(
            "Norm computed over the selected gradients: `l1` sums absolute "
            "values, `l2` computes the Euclidean norm, and `inf` takes the "
            "maximum absolute value."
        ),
        json_schema_extra={
            "omit_behavior": "Uses the L2 norm.",
            "null_behavior": "Not allowed.",
        },
    )


class PerLossGradientNormConfig(_GradientNormConfig):
    """Measures individual loss gradients before normal training backward.

    Each selected loss produces a raw gradient norm and a weighted norm.
    The weighted norm is the raw norm multiplied by the effective loss
    weight, including any loss-weight schedule.

    Measurements use additional autograd calculations while preserving the
    training graph and leaving parameter `.grad` buffers unchanged. This adds
    computation and may increase memory usage.

    Measurements apply to individual training batches, before gradient
    accumulation. They are logged without epoch averaging under
    `grad_norms_wrt_<target>/per_loss/<role>/<loss_id>/<norm_type>/raw`
    and the corresponding `/weighted` key.

    Per-loss inspection requires a BenchRep-built model.
    """

    kind: Literal["per_loss"] = Field(
        default="per_loss",
        description="Selects inspection of individual loss gradients.",
        json_schema_extra={
            "omit_behavior": (
                "Defaults to `per_loss` when constructing this concrete class. "
                "Mappings in `inspection.gradient_norms` must include `kind` "
                "to select the appropriate schema."
            ),
            "null_behavior": "Not allowed.",
        },
    )

    losses: Literal["all"] | Annotated[
        list[
            Annotated[
                str,
                Field(pattern=r"^[^.\s]+\.[^.\s]+$")
            ]
        ],
        Field(min_length=1),
    ] = Field(
        default="all",
        description=(
            "`all` or a nonempty list of configured losses expressed as "
            "`<role>.<resolved_loss_identifier>`."
        ),
        json_schema_extra={
            "omit_behavior": "Inspects every configured loss term.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Without a configured loss ID, the resolved identifier is "
                "the canonical registered loss name.",
                "With a configured ID, the resolved identifier is "
                "`<id>_<canonical_loss_name>`.",
                "For example, `reconstruction.mse` selects an MSE term "
                "without a custom ID.",
                "Aliases do not change resolved loss identifiers.",
                "Unknown references and duplicate measurements are rejected.",
            ],
        },
    )

    every_n_batches: int = Field(
        default=1,
        strict=True,
        gt=0,
        description=(
            "Inspect every N training batches, starting with batch 0 "
            "of each epoch."
        ),
        json_schema_extra={
            "omit_behavior": "Inspects every training batch.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Sampling occurs when `batch_index % every_n_batches == 0`.",
                "This controls measurement frequency. Emission to an "
                "experiment logger also follows Lightning's logging cadence.",
            ],
        },
    )


class OptimizerGradientNormConfig(_GradientNormConfig):
    """Measures accumulated gradients immediately before an optimizer update.

    Inspection occurs after gradient accumulation and mixed-precision
    unscaling, but before gradient clipping. It reads existing `.grad`
    buffers without calculating new gradients or modifying them.

    Only parameters belonging to both the selected target and the active
    optimizer are included. Measurements are logged without epoch averaging
    under `grad_norms_wrt_<target>/optimizer/<norm_type>`.

    External models may use this inspection with target `whole_model`.
    """

    kind: Literal["optimizer_gradients"] = Field(
        default="optimizer_gradients",
        description="Selects inspection of stored optimizer gradients.",
        json_schema_extra={
            "omit_behavior": (
                "Defaults to `optimizer_gradients` when constructing this "
                "concrete class. Mappings in `inspection.gradient_norms` "
                "must include `kind` to select the appropriate schema."
            ),
            "null_behavior": "Not allowed.",
        },
    )

    every_n_optimizer_steps: int = Field(
        default=1,
        strict=True,
        gt=0,
        description=(
            "Inspect before optimizer updates using the completed-step count, "
            "starting before the first update."
        ),
        json_schema_extra={
            "omit_behavior": "Inspects before every optimizer update.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Sampling occurs when the number of completed optimizer "
                "steps is divisible by this value.",
                "Gradient-accumulation batches without an optimizer update "
                "do not trigger this measurement.",
                "Emission to an experiment logger also follows Lightning's "
                "logging cadence.",
            ],
        },
    )


GradientNormConfig: TypeAlias = Annotated[
    PerLossGradientNormConfig | OptimizerGradientNormConfig,
    Field(discriminator="kind"),
]


class TrainingInspectionConfig(_TrainingConfigBaseModel):
    """Configure optional model-graph export and training diagnostics."""

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

    gradient_norms: list[GradientNormConfig] = Field(
        default_factory=list,
        description="Gradient-norm inspection requests for training.",
        json_schema_extra={
            "omit_behavior": "Gradient-norm inspection is disabled.",
            "null_behavior": "Not allowed; use an empty list to disable.",
            "notes": [
                "Requires single-process training, Lightning automatic "
                "optimization, and exactly one optimizer.",
                "Each mapping must include `kind`: `per_loss` or "
                "`optimizer_gradients`.",
                "Duplicate measurements are rejected even when their "
                "sampling frequencies differ.",
                "Measurements are logged through Lightning. Configure a "
                "training logger to persist their history.",
                "Unavailable measurements produce deduplicated warnings "
                "included in training provenance. Unexpected errors "
                "propagate.",
            ],
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

    root: PathInput = Field(
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

    root: PathInput = Field(
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

    root: PathInput = Field(
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
        title="Validation fraction",
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
                "When `val_fraction > 0`, validation size is calculated as "
                "`int(dataset_size * val_fraction)`; setup fails if this "
                "produces an empty validation subset.",
            ],
        },
    )

    num_workers: NonNegativeInt = Field(
        default=4,
        title="Number of workers",
        description="Number of worker processes used by each DataLoader.",
        json_schema_extra={
            "omit_behavior": "Uses four worker processes.",
            "null_behavior": "Not allowed.",
        },
    )

    prefetch_factor: PositiveInt | None = Field(
        default=None,
        strict=True,
        description=(
            "Number of batches loaded in advance by each worker. An explicit "
            "value requires `num_workers` to be greater than zero."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses PyTorch's default: two batches per worker when "
                "num_workers > 0; otherwise no multiprocessing prefetch."
            ),
            "null_behavior": "Same as omission.",
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

    dataloader_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        title="Additional DataLoader arguments",
        description=(
            "Additional keyword arguments passed to each DataLoader. "
            "BenchRep-controlled arguments cannot be overridden."
        ),
        json_schema_extra={
            "omit_behavior": "Passes no additional DataLoader arguments.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Reserved arguments are `dataset`, `batch_size`, `shuffle`, "
                "`sampler`, `batch_sampler`, `num_workers`, `pin_memory`, "
                "`persistent_workers`, `drop_last`, and `prefetch_factor`.",
                "Other argument names must be accepted by the installed "
                "PyTorch DataLoader.",
                "With `num_workers=0`, `multiprocessing_context` must be "
                "omitted or null, and `timeout` must be zero.",
            ],
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

        if self.prefetch_factor is not None and self.num_workers == 0:
            raise ValueError(
                "`datamodule.prefetch_factor` requires "
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

    random_apply: float = Field(
        default=1.0,
        title="Application probability",
        ge=0.0,
        le=1.0,
        description=(
            "Probability of applying this transform each time the step runs. "
            "When skipped, the input tensor passes through unchanged."
        ),
        json_schema_extra={
            "omit_behavior": "Always applies the transform.",
            "null_behavior": "Not allowed.",
            "notes": [
                "At 1.0, no probability wrapper or additional random draw is used.",
                "At 0.0, the transform is never applied.",
                "This probability is independent of any application probability "
                "configured in the transform's own `params`.",
                "Validation-targeted training steps retain this setting when "
                "inherited by prediction.",
            ],
        },
    )
    apply_to: list[Literal["training", "validation"]] = Field(
        min_length=1,
        max_length=2,
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
                "Callback instances passed through the training entrypoint's "
                "`runtime_callbacks` argument supplement this list.",
            ],
        },
    )

    inspection: TrainingInspectionConfig = Field(
        default_factory=TrainingInspectionConfig,
        description=(
            "Optional model-graph export and training-time gradient-norm "
            "inspection settings."
        ),
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

