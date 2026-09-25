from __future__ import annotations

from pathlib import Path
from typing import Literal, Annotated

from pydantic import (
    ConfigDict,
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeInt,
    model_validator,
    field_validator,
    ValidationInfo,
    StringConstraints,
)

from benchrep.assembly.schemas.training_config_schema import (
    NamedConfig,
    SupportedDatasetConfig,
)
from benchrep.assembly.schemas.runtime_override_config_schema import (
    RuntimeOverridesConfig,
)


# -------------------------
# Generic reusable blocks
# -------------------------
class _PredictionConfigBaseModel(BaseModel):
    """Base model for strict prediction configuration schemas."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "extra_field_behavior": (
                "Forbidden at this configuration level; unknown fields raise a "
                "validation error."
            ),
        },
    )


# -------------------------
# Source config
# -------------------------
class PredictionSourceConfig(_PredictionConfigBaseModel):
    """Selects the training run and checkpoint used for prediction.

    The training manifest anchors prediction to a BenchRep training run. It
    provides the resolved training configuration, run identity, component
    provenance, and recorded checkpoint locations needed to reconstruct and
    validate the prediction workflow.

    The manifest path may be configured here or passed to the prediction
    entrypoint through its `training_manifest_path` argument. The entrypoint
    argument takes precedence when both are provided.

    The checkpoint may be selected through the training manifest, by filename
    within the recorded training checkpoint directory, or through an absolute
    path. An absolute checkpoint path may point outside the training run, but
    the training manifest remains required and the selected checkpoint must be
    compatible with the reconstructed or externally supplied model.
    """

    training_manifest_path: Path | None = Field(
        default=None,
        description=(
            "Path to a BenchRep training manifest. Relative paths are resolved "
            "against the current working directory. The manifest must identify a "
            "training run with an accepted status and provide the configuration and "
            "provenance required for prediction."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Must be supplied through the `training_manifest_path` argument "
                "of entrypoint functions like `predict_ae()` or `predict_vae()`."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Accepted training statuses are `completed`, "
                "`completed_with_warnings`, and `completed_after_interruption`.",
            ],
        },
    )

    checkpoint: Literal["best", "last"] | Path = Field(
        default="best",
        description=(
            "Checkpoint selection for prediction. `best` uses the best-checkpoint "
            "path recorded in the training manifest, while `last` uses its "
            "last-checkpoint path. A bare checkpoint filename is resolved inside "
            "the training checkpoint directory recorded in the manifest. An "
            "absolute path, including an expanded `~/...` path, selects that file "
            "directly. Directory-containing relative paths are not supported."
        ),
        json_schema_extra={
            "omit_behavior": "Selects the recorded best checkpoint.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Selecting `best` or `last` requires the corresponding checkpoint "
                "path to be present in the training manifest.",
                "Every resolved checkpoint must exist, be a file, and have the "
                "`.ckpt` extension.",
                "An explicitly selected checkpoint may come from outside the "
                "training run, but BenchRep cannot verify semantic compatibility "
                "beyond loading its state dictionary into the selected model.",
            ],
        },
    )

    @field_validator("training_manifest_path")
    @classmethod
    def resolve_training_manifest_path(
        cls,
        value: Path | None,
    ) -> Path | None:
        if value is None:
            return None

        value = value.expanduser().resolve()

        if value.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                "`training_manifest_path` must point to a YAML file."
            )

        return value

    @field_validator("checkpoint")
    @classmethod
    def validate_checkpoint(
        cls,
        value: Literal["best", "last"] | Path,
    ) -> Literal["best", "last"] | Path:
        if value in {"best", "last"}:
            return value

        assert isinstance(value, Path)

        checkpoint_path = value.expanduser()

        if (
            not checkpoint_path.is_absolute()
            and checkpoint_path.parent != Path(".")
        ):
            raise ValueError(
                "`checkpoint` must be 'best', 'last', a checkpoint filename, "
                "or an absolute checkpoint path; directory-containing relative "
                "paths are not supported."
            )

        if checkpoint_path.suffix != ".ckpt":
            raise ValueError(
                "`checkpoint` filenames and paths must end with '.ckpt'."
            )

        if checkpoint_path.is_absolute():
            return checkpoint_path.resolve()

        return checkpoint_path


class PredictionDataConfig(_PredictionConfigBaseModel):
    """Controls prediction-time data loading and optional batch limiting.

    `batch_size` and `num_workers` apply when BenchRep constructs the prediction
    datamodule. By default, they inherit the corresponding settings from a
    config-built training datamodule. If no reconstructable training datamodule
    configuration is available, BenchRep starts from the `TrainingDataModuleConfig`
    defaults when constructing a new prediction datamodule.

    When a datamodule is supplied directly to the prediction entrypoint,
    `batch_size` and `num_workers` are ignored because the external datamodule
    controls its own DataLoaders.

    `max_batches` limits the number of batches processed by the Lightning
    Trainer and therefore also applies when an external datamodule is used.
    """

    batch_size: PositiveInt | None = Field(
        default=None,
        description=(
            "Number of samples loaded in each prediction batch when BenchRep "
            "constructs the prediction datamodule."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Inherits the config-built training datamodule's batch size "
                "when available; otherwise uses the `TrainingDataModuleConfig` default."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Ignored when an external datamodule is supplied to the "
                "prediction entrypoint."
            ],
        },
    )

    num_workers: NonNegativeInt | None = Field(
        default=None,
        description=(
            "Number of worker processes used by the prediction DataLoader when "
            "BenchRep constructs the prediction datamodule."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Inherits the config-built training datamodule's worker count "
                "when available; otherwise uses the `TrainingDataModuleConfig` default."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Setting this to 0 disables persistent DataLoader workers.",
                "Ignored when an external datamodule is supplied to the "
                "prediction entrypoint.",
            ],
        },
    )

    max_batches: PositiveInt | None = Field(
        default=None,
        description=(
            "Maximum number of prediction batches processed. When configured, "
            "BenchRep passes this value to Lightning as "
            "`limit_predict_batches`."
        ),
        json_schema_extra={
            "omit_behavior": "Processes every available prediction batch.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "This limits batches rather than individual samples.",
                "Prediction exports and linked evaluation contain only samples "
                "from the processed batches.",
                "This setting also applies when an external datamodule is used.",
            ],
        },
    )


class CompositeModelAssemblyInputOverrideConfig(
    _PredictionConfigBaseModel
):
    """Overrides selected input routes for one composite assembly step."""

    inputs: dict[str, str] = Field(
        min_length=1,
        description=(
            "Mapping from component forward-parameter names to replacement "
            "composite graph references such as `produces.z_mu`."
        ),
    )


class PredictionInferenceConfig(_PredictionConfigBaseModel):
    """Controls prediction reproducibility and model-specific inference behavior.

    Randomness and matrix-multiplication settings inherit their corresponding
    training values unless explicitly overridden. The resolved seed and worker
    seeding settings are passed to `lightning.seed_everything()` before
    prediction, while deterministic execution is configured through the
    prediction Lightning Trainer.

    `canonical_vae_reconstruction_latent_source` applies only to internally assembled
    canonical VAE models. Composite models instead support prediction-only
    rerouting through `composite_model_assembly_input_overrides`, which modifies
    the assembly input wiring before the prediction-time model specification is
    resolved and built.
    """

    seed: int | None = Field(
        default=None,
        description="Global random seed used for prediction.",
        json_schema_extra={
            "omit_behavior": "Inherits the training reproducibility seed.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Passed to `lightning.seed_everything()` before prediction.",
                "Used as the fallback seed for random reconstruction-example selection "
                "when no export-specific seed is configured.",
            ],
        },
    )

    seed_workers: bool | None = Field(
        default=None,
        description="Whether DataLoader worker processes receive reproducible seeds.",
        json_schema_extra={
            "omit_behavior": (
                "Inherits the training DataLoader-worker seeding setting."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Passed as the `workers` argument to `lightning.seed_everything()`.",
            ],
        },
    )

    deterministic: bool | Literal["warn"] | None = Field(
        default=None,
        description="Whether deterministic algorithms are requested during prediction.",
        json_schema_extra={
            "omit_behavior": (
                "Inherits the training Trainer's deterministic setting."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "`warn` requests deterministic execution but warns instead of failing "
                "when an operation lacks a deterministic implementation.",
            ],
        },
    )

    float32_matmul_precision: Literal[
        "medium",
        "high",
        "highest",
    ] | None = Field(
        default=None,
        description="Internal precision used for float32 matrix multiplications.",
        json_schema_extra={
            "omit_behavior": (
                "Inherits the training float32 matrix-multiplication precision."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Passed to `torch.set_float32_matmul_precision()` before prediction.",
                "Does not change tensor dtypes.",
            ],
        },
    )

    canonical_vae_reconstruction_latent_source: Literal["mean", "sample"] | None = Field(
        default=None,
        description=(
            "Latent representation decoded for prediction-time reconstruction "
            "by an internally assembled VAE. `mean` decodes the posterior mean "
            "(`z_mu`), while `sample` decodes a sampled latent (`z_sample`)."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses `mean` for an internally assembled VAE. No latent source "
                "is selected for non-VAE or externally supplied models."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A non-null value is rejected for non-VAE models.",
                "A non-null value is rejected for external VAE models because "
                "BenchRep cannot control their `predict_step()` implementation.",
                "Using `mean` removes posterior sampling from the reconstruction "
                "path but does not independently guarantee that every operation "
                "in the prediction workflow is deterministic.",
                "Using `sample` introduces latent-sampling randomness governed "
                "by the resolved prediction seed and execution settings.",
            ],
        },
    )

    composite_model_assembly_input_overrides: dict[
        str,
        CompositeModelAssemblyInputOverrideConfig,
    ] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Prediction-only input-route overrides keyed by composite "
            "assembly-step identifier."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses the composite assembly recorded by the training run "
                "without modification."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Supported only for composite models.",
                "Each key identifies an existing assembly step.",
                "Each nested input key identifies an existing component "
                "forward parameter on that step.",
                "Only inputs currently routed from a `produces.*` output "
                "can be overridden.",
                "Replacement values must be complete composite graph "
                "references such as `produces.z_mu`.",
                "The original and replacement outputs must come from the "
                "same producer assembly step.",
                "Components and output bindings cannot be changed.",
            ],
        },
    )

    @model_validator(mode="after")
    def validate_model_specific_inference_options(
        self,
    ) -> PredictionInferenceConfig:
        if (
            self.canonical_vae_reconstruction_latent_source is not None
            and self.composite_model_assembly_input_overrides is not None
        ):
            raise ValueError(
                "`canonical_vae_reconstruction_latent_source` and "
                "`composite_model_assembly_input_overrides` cannot be configured "
                "together because the former applies to canonical VAEs and the "
                "latter to composite models."
            )

        return self


# -------------------------
# Transform config
# -------------------------
class PredictionTransformStepConfig(NamedConfig):
    """Select one transform step within a prediction pipeline."""


class PredictionTransformPipelineConfig(_PredictionConfigBaseModel):
    """Configure one routed prediction transform pipeline."""

    input: str | None = None
    output: str | None = None
    steps: list[PredictionTransformStepConfig] = Field(min_length=1)

    @field_validator("input", "output")
    @classmethod
    def validate_field_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(
                "Transform pipeline input and output names must be nonempty."
            )

        return value


# -------------------------
# Exports config
# -------------------------
PredictionOutputKey = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class PredictionAnnDataExportConfig(_PredictionConfigBaseModel):
    """Configures export of non-image prediction outputs to AnnData.

    `all` selects every supported non-image output. `custom` selects the
    explicitly configured output keys. Model-dependent key and primary-key
    resolution occurs later, when the linked training model is known.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Whether supported prediction outputs are exported as an AnnData "
            "artifact."
        ),
        json_schema_extra={
            "omit_behavior": "Enables AnnData export.",
            "null_behavior": "Not allowed.",
        },
    )

    mode: Literal["all", "custom"] | None = Field(
        default=None,
        description=(
            "Output selection mode. `all` exports every supported non-image "
            "output, while `custom` exports only `keys`."
        ),
        json_schema_extra={
            "omit_behavior": "Equivalent to `all`.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "When AnnData export is disabled, this field has no effect.",
            ],
        },
    )

    keys: list[PredictionOutputKey] | None = Field(
        default=None,
        description=(
            "Ordered prediction-output keys selected when `mode='custom'`."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Required when AnnData export is enabled with `mode='custom'`; "
                "otherwise no explicit keys are selected."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Custom key lists must be non-empty and contain no duplicates.",
                "Keys are validated against the linked model during prediction "
                "configuration resolution.",
            ],
        },
    )

    primary_key: PredictionOutputKey | None = Field(
        default=None,
        description=(
            "Selected vector-valued output stored in `adata.X`. For canonical "
            "models, omission selects `embedding`. Composite models require an "
            "explicit primary key."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Selects `embedding` for canonical models; rejected for "
                "composite models."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "The primary key must resolve to a selected vector-valued output.",
                "With `mode='custom'`, an explicit primary key must be included "
                "in `keys`.",
            ],
        },
    )

    @model_validator(mode="after")
    def validate_selection(self) -> PredictionAnnDataExportConfig:
        if not self.enabled:
            return self

        resolved_mode = self.mode or "all"

        if resolved_mode == "all":
            if self.keys is not None:
                raise ValueError(
                    "`exports.anndata.keys` must be null or omitted when "
                    "`exports.anndata.mode='all'`."
                )

            return self

        if not self.keys:
            raise ValueError(
                "`exports.anndata.keys` must be a non-empty list when "
                "`exports.anndata.mode='custom'`."
            )

        if len(self.keys) != len(set(self.keys)):
            raise ValueError(
                "`exports.anndata.keys` must not contain duplicate output keys."
            )

        if (
            self.primary_key is not None
            and self.primary_key not in self.keys
        ):
            raise ValueError(
                "`exports.anndata.primary_key` must be included in "
                "`exports.anndata.keys` when mode is 'custom'."
            )

        return self


PredictionReconstructionPairId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    ),
]

PredictionReconstructionFieldName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class PredictionReconstructionPairConfig(_PredictionConfigBaseModel):
    """Declares one input/reconstruction artifact pair."""

    input: PredictionReconstructionFieldName
    reconstruction: PredictionReconstructionFieldName


class PredictionReconstructionsExportConfig(_PredictionConfigBaseModel):
    """Configures export of selected inputs and model reconstructions.

    The exported tensors come from the `input` and `reconstruction` fields
    returned by the model's `predict_step()` for each batch. BenchRep
    concatenates these fields across prediction batches, selects the requested
    examples, and writes the selected tensors as separate `.pt` artifacts.

    Reconstruction export also writes `obs.pt`, containing the selected source
    indices and any available `sample_id`, `label`, and metadata fields, plus a
    metadata artifact describing the selection and exported tensors.

    Unlike embedding-key selection, these settings are not altered by the
    parent `exports.mode`.

    When `enabled` is omitted or null, BenchRep determines reconstruction
    availability from the model family's expected prediction-output contract.
    Reconstruction export is enabled when that contract declares a
    `reconstruction` field and disabled otherwise. Setting `enabled` to true
    for an unsupported model family raises during configuration resolution.
    When export is explicitly or automatically disabled, the remaining settings
    in this section have no effect.
    """

    enabled: bool = Field(
        default=False,
        description="Whether reconstruction bundles are exported.",
        json_schema_extra={
            "omit_behavior": "Disables reconstruction export.",
            "null_behavior": "Not allowed.",
            "notes": [
                "When disabled, the remaining reconstruction settings have no effect.",
            ],
        },
    )

    mode: Literal["all", "custom"] | None = Field(
        default=None,
        description=(
            "Pair selection mode. `all` derives every supported reconstruction "
            "pair, while `custom` uses the explicitly configured `pairs`."
        ),
        json_schema_extra={
            "omit_behavior": "Equivalent to `all`.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "When reconstruction export is disabled, this field has no effect.",
            ],
        },
    )

    pairs: (
            dict[
                PredictionReconstructionPairId,
                PredictionReconstructionPairConfig,
            ]
            | None
    ) = Field(
        default=None,
        description=(
            "Ordered reconstruction pairs exported when `mode='custom'`. Mapping "
            "keys are stable pair IDs used in manifests and artifact-directory names."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Required when reconstruction export is enabled with "
                "`mode='custom'`; otherwise pairs are inferred."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )

    n_examples: Literal["all"] | PositiveInt = Field(
        default=32,
        description=(
            "Number of reconstruction examples exported. A positive integer "
            "requests at most that many examples, capped by the number of "
            "predicted samples. `all` exports every predicted sample in its "
            "original order."
        ),
        json_schema_extra={
            "omit_behavior": "Exports at most 32 examples.",
            "null_behavior": "Not allowed.",
            "notes": [
                "When set to `all`, `selection` does not alter which examples "
                "are exported.",
            ],
        },
    )

    selection: Literal["first", "random"] = Field(
        default="first",
        description=(
            "Method used to select an integer number of examples. `first` "
            "selects the first examples after prediction batches are "
            "concatenated. `random` samples without replacement using the "
            "resolved reconstruction-export seed."
        ),
        json_schema_extra={
            "omit_behavior": "Selects the first requested examples.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Has no effect when `n_examples='all'`.",
                "Stratified subset selection requires `random`.",
            ],
        },
    )

    stratify_by: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ] | None = Field(
        default=None,
        description=(
            "Optional prediction observation or metadata field used for "
            "approximately balanced random subset selection. Supported fields "
            "include `sample_id`, `label`, and keys returned inside the "
            "`metadata` field of `predict_step()` outputs."
        ),
        json_schema_extra={
            "omit_behavior": "Does not stratify reconstruction selection.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "For an integer `n_examples`, stratification requires "
                "`selection='random'`.",
                "Sampling is distributed approximately evenly across strata "
                "rather than proportionally to their original frequencies.",
                "If fewer examples than strata are requested, some strata "
                "cannot be represented and BenchRep emits a warning.",
                "With `n_examples='all'`, this field does not alter selection, "
                "but the requested field must exist and is preserved in "
                "`obs.pt`.",
            ],
        },
    )

    seed: int | None = Field(
        default=None,
        description=(
            "Seed used for random reconstruction-example selection. It "
            "initializes a local PyTorch generator and does not replace the "
            "global prediction seed."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses the resolved prediction inference seed when random "
                "selection is requested."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Unused for `selection='first'` and when `n_examples='all'`.",
            ],
        },
    )

    include_input: bool = Field(
        default=True,
        description=(
            "Whether selected values from the `input` field returned by "
            "`predict_step()` are written to `input.pt`."
        ),
        json_schema_extra={
            "omit_behavior": "Exports selected model inputs.",
            "null_behavior": "Not allowed.",
        },
    )

    include_reconstruction: bool = Field(
        default=True,
        description=(
            "Whether selected reconstruction tensors are written to each "
            "reconstruction bundle."
        ),
        json_schema_extra={
            "omit_behavior": "Exports selected model reconstructions.",
            "null_behavior": "Not allowed.",
        },
    )

    @model_validator(mode="after")
    def validate_export_settings(
            self,
    ) -> PredictionReconstructionsExportConfig:
        if not self.enabled:
            return self

        resolved_mode = self.mode or "all"

        if resolved_mode == "all":
            if self.pairs is not None:
                raise ValueError(
                    "`exports.reconstructions.pairs` must be null or omitted when "
                    "`exports.reconstructions.mode='all'`."
                )

        elif not self.pairs:
            raise ValueError(
                "`exports.reconstructions.pairs` must be a non-empty mapping when "
                "`exports.reconstructions.mode='custom'`."
            )

        if self.pairs is not None:
            resolved_pairs = [
                (pair.input, pair.reconstruction)
                for pair in self.pairs.values()
            ]

            if len(resolved_pairs) != len(set(resolved_pairs)):
                raise ValueError(
                    "`exports.reconstructions.pairs` must not contain duplicate "
                    "input/reconstruction pairs."
                )

        if not self.include_input and not self.include_reconstruction:
            raise ValueError(
                "Enabled reconstruction export requires at least one of "
                "`include_input` or `include_reconstruction` to be true."
            )

        if (
                self.n_examples != "all"
                and self.stratify_by is not None
                and self.selection != "random"
        ):
            raise ValueError(
                "`exports.reconstructions.selection` must be 'random' when "
                "stratifying a reconstruction subset."
            )

        return self


class PredictionExportConfig(_PredictionConfigBaseModel):
    """Configures prediction artifact exports.

    AnnData and reconstruction exports are independent. Each branch controls
    whether it is enabled and whether its contents are selected automatically
    or explicitly.
    """

    anndata: PredictionAnnDataExportConfig = Field(
        default_factory=PredictionAnnDataExportConfig,
        description=(
            "Configures export of supported non-image prediction outputs to "
            "AnnData."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses the defaults defined by "
                "`PredictionAnnDataExportConfig`, enabling export in `all` mode."
            ),
            "null_behavior": "Not allowed.",
        },
    )

    reconstructions: PredictionReconstructionsExportConfig = Field(
        default_factory=PredictionReconstructionsExportConfig,
        description=(
            "Configures export of input/reconstruction tensor bundles."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses the defaults defined by "
                "`PredictionReconstructionsExportConfig`, disabling "
                "reconstruction export."
            ),
            "null_behavior": "Not allowed.",
        },
    )


# -------------------------
# Full prediction configuration
# -------------------------
class PredictionConfig(_PredictionConfigBaseModel):
    """Complete configuration for a BenchRep prediction workflow.

    Every prediction run must be linked through its training manifest to a
    training run with an accepted status. BenchRep uses the manifest and
    recorded resolved training config to locate the selected checkpoint,
    reconstruct config-built components when possible, and inherit applicable
    settings. Requirements for prediction data configuration depend on whether
    an external datamodule is supplied at runtime.

    Use `benchrep.inspect_config(PredictionConfig)` to inspect this configuration.
    Nested configuration types shown in the output can be inspected the same
    way, for example `benchrep.inspect_config(PredictionSourceConfig)` or
    `benchrep.inspect_config(PredictionReconstructionsExportConfig)`. Public
    configuration classes are available from `benchrep.assembly.schemas`.

    Use `benchrep.inspect_registry()` to discover component registries and
    `benchrep.inspect_registry("<registry>", "<component>")` to inspect a
    registered implementation.

    For machine-readable discovery, `PredictionConfig.model_json_schema()`
    returns standard JSON Schema, while `benchrep.list_registries()` and
    `benchrep.list_registered_components()` return structured registry data.
    """

    stage: Literal["prediction"] = Field(
        default="prediction",
        description="Identifies this configuration as a prediction workflow.",
        json_schema_extra={
            "omit_behavior": "Uses `prediction`.",
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

    source: PredictionSourceConfig = Field(
        default_factory=PredictionSourceConfig,
        description="Training-manifest and checkpoint-selection settings.",
        json_schema_extra={
            "omit_behavior": (
                "Uses the defaults defined by `PredictionSourceConfig`. Because "
                "its default does not provide a training-manifest path, that path "
                "must instead be supplied through the prediction entrypoint."
            ),
            "null_behavior": "Not allowed.",
        },
    )

    dataset: SupportedDatasetConfig | None = Field(
        default=None,
        description=(
            "Dataset to build for prediction. Use "
            "`benchrep.inspect_registry(\"dataset\")` to inspect registered "
            "dataset names and aliases. If omitted or null, BenchRep inherits "
            "the recorded training dataset only when training used a "
            "config-built datamodule. This section is ignored when an external "
            "prediction datamodule is supplied."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Inherits the training dataset when it was used by a "
                "config-built training datamodule. Otherwise an explicit "
                "prediction dataset or external prediction datamodule is required."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Inheritance copies the complete training dataset configuration, "
                "including its configured dataset split.",
                "Prediction does not reproduce the training-validation partition; "
                "it predicts over the complete inherited dataset split.",
            ],
        },
    )

    data: PredictionDataConfig = Field(
        default_factory=PredictionDataConfig,
        description=(
            "Prediction batching, data-loading, and batch-limiting settings. "
            "This section is ignored when an external prediction datamodule is "
            "supplied."
        ),
        json_schema_extra={
            "omit_behavior": "Uses the defaults defined by `PredictionDataConfig`.",
            "null_behavior": "Not allowed.",
        },
    )

    inference: PredictionInferenceConfig = Field(
        default_factory=PredictionInferenceConfig,
        description=(
            "Prediction randomness, deterministic execution, float32 "
            "matrix-multiplication precision, and VAE reconstruction behavior."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses the defaults defined by `PredictionInferenceConfig`, "
                "including inheritance from the linked training run where "
                "applicable."
            ),
            "null_behavior": "Not allowed.",
        },
    )

    transform_pipelines: list[PredictionTransformPipelineConfig] | None = Field(
        default=None,
        description="Ordered transform pipelines applied during prediction.",
        json_schema_extra={
            "omit_behavior": (
                "Inherits validation-targeted transform pipelines from training; "
                "otherwise uses no configured transform pipelines."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "An explicit list replaces inherited transform pipelines.",
                "An empty list applies no configured transforms.",
                "Nothing is inherited when training used an external datamodule.",
            ],
        },
    )

    exports: PredictionExportConfig = Field(
        default_factory=PredictionExportConfig,
        description="Prediction embedding and reconstruction artifact settings.",
        json_schema_extra={
            "omit_behavior": "Uses the defaults defined by `PredictionExportConfig`.",
            "null_behavior": "Not allowed.",
        },
    )

    @model_validator(mode="after")
    def validate_prediction_config(
        self,
        info: ValidationInfo,
    ) -> PredictionConfig:
        ctx = info.context or {}

        # Runtime override requirements are checked by the resolver.
        training_manifest_path_overridden = ctx.get(
            "training_manifest_path_overridden",
            False,
        )

        if (
            self.source.training_manifest_path is None
            and not training_manifest_path_overridden
        ):
            raise ValueError(
                "`source.training_manifest_path` is required unless "
                "`training_manifest_path` is passed to the prediction entrypoint."
            )

        return self