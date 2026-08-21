from __future__ import annotations

from pathlib import Path
from typing import Literal, Any, Annotated, TypeAlias

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
    ConfigDict,
    StringConstraints,
    NonNegativeInt,
    ValidationInfo,
    field_validator,
)

RegistrySelection: TypeAlias = Literal["all"] | list[str] | None

NSplits: TypeAlias = Annotated[int, Field(ge=2)]

PositiveFloatOrList: TypeAlias = (
    PositiveFloat
    | Annotated[list[PositiveFloat], Field(min_length=1)]
)
PositiveIntOrList: TypeAlias = (
    PositiveInt
    | Annotated[list[PositiveInt], Field(min_length=1)]
)

KNNWeights: TypeAlias = Literal["uniform", "distance"]
KNNWeightsOrList: TypeAlias = (
    KNNWeights
    | Annotated[list[KNNWeights], Field(min_length=1)]
)
KNNMetric: TypeAlias = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
KNNMetricOrList: TypeAlias = (
    KNNMetric
    | Annotated[list[KNNMetric], Field(min_length=1)]
)

SVMRBFGammaValue: TypeAlias = Literal["scale", "auto"] | PositiveFloat
SVMRBFGammaValueOrList: TypeAlias = (
    SVMRBFGammaValue
    | Annotated[list[SVMRBFGammaValue], Field(min_length=1)]
)
MaxIterWithNoLimitSentinel: TypeAlias = Literal[-1] | PositiveInt

MaxDepthValue: TypeAlias = PositiveInt | None
MaxDepthParam: TypeAlias = (
    MaxDepthValue
    | Annotated[list[MaxDepthValue], Field(min_length=1)]
)

ErrorMapKind: TypeAlias = Literal[
    "absolute",
    "squared",
    "signed",
    "relative",
    "normalized_absolute_global",
    "normalized_absolute_per_channel",
]

HexColor: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^#[0-9A-Fa-f]{6}$"),
]


# -------------------------
# Generic reusable blocks
# -------------------------
class _EvaluationConfigBaseModel(BaseModel):
    """Base model for strict evaluation configuration schemas."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "extra_field_behavior": (
                "Forbidden at this configuration level; unknown fields raise a "
                "validation error."
            ),
        },
    )


class EvalStepConfig(_EvaluationConfigBaseModel):
    """Shared configuration for an optional evaluation step.

    Evaluation steps use tri-state enablement. `True` explicitly enables the
    step, `False` disables it, and `None` delegates the decision to the
    evaluation resolver. Automatic behavior differs between concrete steps and
    may depend on available source artifacts.

    Enabling a step requests its execution but does not guarantee success.
    Recoverable step-local failures are recorded with a `failed` status and the
    pipeline continues, while dependent steps are skipped. Fatal configuration,
    dependency, or unexpected runtime failures terminate the workflow.
    """

    enabled: bool | None = Field(
        default=None,
        description=(
            "Tri-state switch controlling whether the evaluation step runs. "
            "Automatic behavior is defined by the concrete step."
        ),
        json_schema_extra={
            "omit_behavior": "Uses the concrete step's automatic behavior.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Parameters supplied to the evaluation step. Concrete step schemas "
            "replace this mapping with their corresponding typed parameter model."
        ),
        json_schema_extra={
            "omit_behavior": "Uses the concrete step's configured defaults.",
            "null_behavior": "Equivalent to omission.",
        },
    )


class EvalMetricGroupConfig(_EvaluationConfigBaseModel):
    """Shared configuration for a registry-backed metric group.

    Selection is independent of enablement. Once the group runs, `None` uses its
    curated default selection, `"all"` selects every metric registered in the
    process, including custom metrics, and a list selects exactly the provided
    metric names or aliases.

    If some selected metrics fail recoverably, successful results are retained and
    the group completes with warnings. If every selected metric fails recoverably,
    the group is marked failed and the evaluation pipeline continues. Invalid
    configuration and unexpected runtime failures remain fatal.
    """

    enabled: bool | None = Field(
        default=None,
        description=(
            "Tri-state switch controlling whether the metric group runs. "
            "Automatic behavior is defined by the concrete group and may depend "
            "on enabled upstream steps or available source artifacts."
        ),
        json_schema_extra={
            "omit_behavior": "Uses the concrete metric group's automatic behavior.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    selected: RegistrySelection = Field(
        default=None,
        description=(
            "Metrics to compute: curated defaults, every registered metric, or an "
            "explicit list of registered names or aliases."
        ),
        json_schema_extra={
            "omit_behavior": "Uses the concrete metric group's curated defaults.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                '`"all"` selects every registered metric, including custom metrics.',
                "An explicit list selects exactly the provided metric names or aliases.",
                "An empty list is rejected when the metric group runs.",
                "Selection does not enable or disable the metric group.",
            ],
        },
    )

    params: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Per-metric keyword arguments keyed by registered metric name or alias."
        ),
        json_schema_extra={
            "omit_behavior": "Passes no metric-specific parameters.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Nested mappings intentionally permit callable-specific arguments "
                "and are validated against the resolved metric callable."
            ],
        },
    )


# -------------------------
# Source and run config
# -------------------------
class EvaluationSourceConfig(_EvaluationConfigBaseModel):
    """Selects the embedding and reconstruction inputs for evaluation.

    Evaluation requires at least one usable embedding or reconstruction input.
    Either artifact type may be supplied directly or inferred from a prediction
    manifest, allowing embedding-only, reconstruction-only, or combined evaluation.

    A directly configured artifact path takes precedence over the corresponding
    path recorded in the manifest. The manifest still provides upstream
    provenance and may supply evaluation run identity.

    Direct inputs follow BenchRep's public evaluation artifact contracts and do
    not need to have been produced by a BenchRep prediction workflow.

    All configured paths expand `~` and resolve relative paths against the
    current working directory.
    """

    prediction_manifest_path: Path | None = Field(
        default=None,
        description=(
            "Path to a BenchRep prediction manifest used to infer unspecified "
            "input artifacts, upstream provenance, and evaluation run identity."
        ),
        json_schema_extra={
            "omit_behavior": (
                "No configured manifest is loaded. At least one direct artifact path "
                "must be provided unless a manifest is supplied through the evaluation "
                "entrypoint."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "The `prediction_manifest_path` entrypoint argument takes "
                "precedence over this field.",
                "Accepted prediction statuses are `completed`, "
                "`completed_with_warnings`, and `partially_completed`.",
                "A partially completed prediction produces a warning; its "
                "referenced artifacts are still validated before use.",
            ],
        },
    )

    embeddings_path: Path | None = Field(
        default=None,
        description=(
            "Path to an AnnData `.h5ad` file. `adata.X` must be a non-empty, "
            "two-dimensional, finite real numeric matrix with shape "
            "`(n_observations, n_embedding_dimensions)`; dense and sparse matrices "
            "are supported. Each row represents one sample and each column one "
            "embedding dimension."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Infers the embeddings path from the prediction manifest when available; "
                "otherwise embedding-dependent evaluation is unavailable."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "`adata.X` is the representation consumed by dimensionality "
                "reduction, clustering, embedding metrics, and predictability probes.",
                "`adata.obs` may provide per-sample annotations such as external "
                "clustering labels, predictability targets, cross-validation groups, "
                "or metadata columns used to color reduction plots. Required keys "
                "depend on the enabled steps.",
                "No BenchRep-specific `obs`, `obsm`, or `uns` entries are required. "
                "Existing entries are preserved unless an enabled step explicitly "
                "overwrites them.",
                "A directly configured path overrides the embeddings path recorded "
                "in the prediction manifest.",
            ],
        },
    )

    reconstructions_path: Path | None = Field(
        default=None,
        description=(
            "Path to a reconstruction artifact directory used for reconstruction "
            "metrics, error maps, TIFF export, and reconstruction grids. The "
            "directory must contain `input.pt`, `reconstruction.pt`, and `obs.pt`; "
            "`reconstruction_export_metadata.pt` is optional."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses the reconstruction bundle recorded in the prediction manifest "
                "when available; otherwise reconstruction inputs are unavailable."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "`input.pt` must contain a real numeric PyTorch tensor or NumPy array "
                "with shape `(B, H, W)` or `(B, C, H, W)`. All axes must be non-empty "
                "and all values finite.",
                "`reconstruction.pt` must contain the same kind of array, with "
                "exactly the same shape as `input.pt`. The two arrays are compared "
                "by reconstruction metrics and error-map computations.",
                "`obs.pt` must contain an object with an inferable length of exactly "
                "`B`. A mapping of column names to equal-length sequences, or a "
                "table-like object with columns, provides the fullest support.",
                "Optional `obs.pt` fields include `sample_id` for TIFF filenames, "
                "`sample_id` or `source_index` for reconstruction-grid row labels, "
                "and any field selected through reconstruction-grid `stratify_by`.",
                "`reconstruction_export_metadata.pt`, when present, must contain a "
                "mapping. Its optional `channel_names` sequence must contain exactly "
                "`C` names and labels per-channel reconstruction metric results; "
                "otherwise names such as `channel_0` are generated.",
                "A directly configured directory overrides reconstruction paths "
                "recorded in the prediction manifest.",
                "An incomplete directly configured bundle raises an error. An "
                "incomplete manifest-derived bundle produces a warning and is skipped.",
                "PyTorch `.pt` artifacts should only be loaded from trusted sources.",
            ],
        },
    )

    @field_validator(
        "prediction_manifest_path",
        "embeddings_path",
        "reconstructions_path",
    )
    @classmethod
    def resolve_source_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None

        return value.expanduser().resolve()

    @field_validator("prediction_manifest_path")
    @classmethod
    def validate_prediction_manifest_extension(
        cls,
        value: Path | None,
    ) -> Path | None:
        if value is not None and value.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                "source.prediction_manifest_path must point to a YAML file."
            )

        return value

    @field_validator("embeddings_path")
    @classmethod
    def validate_embeddings_extension(
        cls,
        value: Path | None,
    ) -> Path | None:
        if value is not None and value.suffix.lower() != ".h5ad":
            raise ValueError(
                "source.embeddings_path must point to an AnnData .h5ad file."
            )

        return value


class EvaluationRunConfig(_EvaluationConfigBaseModel):
    """Controls the evaluation output location and generated run identity.

    BenchRep writes each evaluation beneath an `evaluation/` stage directory and
    creates a timestamped run directory. When a prediction manifest is available,
    project and model identity are inferred from it when possible.
    """

    output_root: Path | None = Field(
        default=None,
        description=(
            "Base directory beneath which BenchRep creates the evaluation stage "
            "and generated run directories."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Infers the base output root from the prediction manifest when "
                "possible; otherwise uses `outputs/` relative to the working directory."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )

    run_name: str | None = Field(
        default=None,
        description=(
            "Optional stem used to construct the generated evaluation run name. "
            "BenchRep sanitizes the stem and appends a timestamp."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses model identity inferred from the prediction manifest when "
                "available; otherwise uses `evaluation`."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A project name inferred from the manifest is prepended when available.",
                "If the generated directory already exists, BenchRep appends a "
                "short unique suffix.",
            ],
        },
    )

    @field_validator("output_root")
    @classmethod
    def resolve_output_root(cls, value: Path | None) -> Path | None:
        if value is None:
            return None

        return value.expanduser().resolve()


# -------------------------
# Reductions config
# -------------------------
class PCAParams(BaseModel):
    """Controls scikit-learn PCA computation and AnnData storage.

    PCA is fitted to `adata.X`. Coordinates are stored in
    `adata.obsm[key_added]`, with explained-variance summaries and provenance
    stored under `adata.uns["benchrep"]["reductions"][key_added]`.

    Additional non-null fields are forwarded to
    `sklearn.decomposition.PCA`.
    """

    n_components: PositiveInt | None = Field(
        default=None,
        description=(
            "Number of components passed to `sklearn.decomposition.PCA`."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses the smallest of 30, the number of observations, and the "
                "number of embedding dimensions."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )

    key_added: str | None = Field(
        default="X_pca",
        description="Key used by BenchRep to store coordinates in `adata.obsm`.",
        json_schema_extra={
            "omit_behavior": "Uses `X_pca`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    random_state: int | None = Field(
        default=137,
        description=(
            "Random seed passed to `sklearn.decomposition.PCA`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses 137.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    overwrite: bool | None = Field(
        default=False,
        description=(
            "Whether BenchRep may replace an existing "
            "`adata.obsm[key_added]` entry."
        ),
        json_schema_extra={
            "omit_behavior": "Does not overwrite an existing entry.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A key collision without overwrite is a recoverable step failure."
            ],
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null fields are forwarded as keyword "
                "arguments to `sklearn.decomposition.PCA`."
            ),
        },
    )


class PCAConfig(EvalStepConfig):
    """Configures PCA as an evaluation step.

    PCA operates directly on `adata.X`. It is enabled automatically when
    embeddings are available and disabled when they are unavailable.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether PCA is included in the evaluation pipeline.",
        json_schema_extra={
            "omit_behavior": (
                "Enables PCA when embeddings are available; otherwise disables it."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )

    params: PCAParams | None = Field(
        default_factory=PCAParams,
        description="Parameters controlling PCA computation and output storage.",
        json_schema_extra={
            "omit_behavior": "Uses `PCAParams` defaults.",
            "null_behavior": "Equivalent to omission.",
        },
    )


class UMAPParams(_EvaluationConfigBaseModel):
    """Controls Scanpy neighbor-graph construction and UMAP computation.

    Coordinates are stored in `adata.obsm[key_added]`. Neighbor metadata is
    stored in `adata.uns[neighbors_key]`, with distance and connectivity
    matrices stored in `adata.obsp`. BenchRep provenance is stored under
    `adata.uns["benchrep"]["reductions"][key_added]`.

    Unless `use_rep` is set, Scanpy uses `adata.X` below its PCA threshold
    and otherwise reuses or computes `adata.obsm["X_pca"]`.
    """

    n_neighbors: PositiveInt | None = Field(
        default=15,
        description=(
            "Number of kNN neighbors passed to `scanpy.pp.neighbors()`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses 15.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Must be smaller than the number of observations."
            ],
        },
    )

    n_pcs: NonNegativeInt | None = Field(
        default=None,
        description=(
            "Number of principal components passed to "
            "`scanpy.pp.neighbors()`."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Lets Scanpy choose the representation and PCA dimensionality."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Zero forces `adata.X` when `use_rep` is not set."
            ],
        },
    )

    use_rep: str | None = Field(
        default=None,
        description=(
            "Representation key passed to `scanpy.pp.neighbors()`. `X` selects "
            "`adata.X`; other values select `adata.obsm[use_rep]`."
        ),
        json_schema_extra={
            "omit_behavior": "Delegates representation selection to Scanpy.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    metric: str | None = Field(
        default="euclidean",
        description="Distance metric passed to `scanpy.pp.neighbors()`.",
        json_schema_extra={
            "omit_behavior": "Uses `euclidean`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    neighbors_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword arguments passed to `scanpy.pp.neighbors()`."
        ),
        json_schema_extra={
            "omit_behavior": "Passes no additional arguments.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Duplicating arguments represented by dedicated fields raises "
                "`TypeError` during backend invocation.",
            ],
        },
    )

    min_dist: NonNegativeFloat | None = Field(
        default=0.1,
        description="Minimum distance passed to `scanpy.tl.umap()`.",
        json_schema_extra={
            "omit_behavior": "Uses 0.1.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    key_added: str | None = Field(
        default="X_umap",
        description=(
            "Key passed to `scanpy.tl.umap()` for storing coordinates in "
            "`adata.obsm`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `X_umap`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    umap_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword arguments passed to `scanpy.tl.umap()`."
        ),
        json_schema_extra={
            "omit_behavior": "Passes no additional arguments.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Duplicating arguments represented by dedicated fields raises "
                "`TypeError` during backend invocation.",
            ],
        },
    )

    neighbors_key: str | None = Field(
        default="neighbors",
        description=(
            "Namespace used by `scanpy.pp.neighbors()` to store the graph and "
            "by `scanpy.tl.umap()` to retrieve it."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `neighbors`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    random_state: int | None = Field(
        default=137,
        description=(
            "Random seed passed to `scanpy.pp.neighbors()` and "
            "`scanpy.tl.umap()`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses 137.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    overwrite: bool | None = Field(
        default=False,
        description=(
            "Whether BenchRep may replace existing UMAP and neighbor-graph "
            "outputs."
        ),
        json_schema_extra={
            "omit_behavior": "Does not overwrite existing outputs.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A key collision without overwrite is a recoverable step failure."
            ],
        },
    )


class UMAPConfig(EvalStepConfig):
    """Configures the optional Scanpy UMAP evaluation step.

    UMAP requires embeddings and the optional Scanpy dependency. It is disabled
    by default and is skipped with a warning if explicitly enabled without
    embeddings.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether UMAP is included in the evaluation pipeline.",
        json_schema_extra={
            "omit_behavior": "Disables UMAP.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    params: UMAPParams | None = Field(
        default_factory=UMAPParams,
        description=(
            "Parameters passed to neighbor-graph construction and UMAP."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `UMAPParams` defaults.",
            "null_behavior": "Equivalent to omission.",
        },
    )


class TSNEParams(BaseModel):
    """Controls Scanpy t-SNE computation and AnnData storage.

    Coordinates are stored in `adata.obsm[key_added]`, with BenchRep provenance
    stored under `adata.uns["benchrep"]["reductions"][key_added]`.

    Unless `use_rep` is set, Scanpy uses `adata.X` below its PCA threshold and
    otherwise reuses or computes `adata.obsm["X_pca"]`. Additional non-null
    fields are forwarded to `scanpy.tl.tsne()`.
    """

    n_pcs: NonNegativeInt | None = Field(
        default=None,
        description=(
            "Number of principal components passed to `scanpy.tl.tsne()`."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Lets Scanpy choose the representation and PCA dimensionality."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Zero forces `adata.X` when `use_rep` is not set."
            ],
        },
    )

    use_rep: str | None = Field(
        default=None,
        description=(
            "Representation key passed to `scanpy.tl.tsne()`. `X` selects "
            "`adata.X`; any other value names an entry in `adata.obsm`."
        ),
        json_schema_extra={
            "omit_behavior": "Delegates representation selection to Scanpy.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    perplexity: PositiveFloat | None = Field(
        default=30.0,
        description="Perplexity passed to `scanpy.tl.tsne()`.",
        json_schema_extra={
            "omit_behavior": "Uses 30.0.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Must be smaller than the number of observations."
            ],
        },
    )

    key_added: str | None = Field(
        default="X_tsne",
        description=(
            "Key passed to `scanpy.tl.tsne()` for storing coordinates in "
            "`adata.obsm`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `X_tsne`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    random_state: int | None = Field(
        default=137,
        description="Random seed passed to `scanpy.tl.tsne()`.",
        json_schema_extra={
            "omit_behavior": "Uses 137.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    overwrite: bool | None = Field(
        default=False,
        description=(
            "Whether BenchRep may replace an existing "
            "`adata.obsm[key_added]` entry."
        ),
        json_schema_extra={
            "omit_behavior": "Does not overwrite an existing entry.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A key collision without overwrite is a recoverable step failure."
            ],
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null fields are forwarded as keyword "
                "arguments to `scanpy.tl.tsne()`."
            ),
        },
    )


class TSNEConfig(EvalStepConfig):
    """Configures the optional Scanpy t-SNE evaluation step.

    t-SNE requires embeddings. It is disabled by default and raises an error if
    explicitly enabled without embeddings.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether t-SNE is included in the evaluation pipeline.",
        json_schema_extra={
            "omit_behavior": "Disables t-SNE.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    params: TSNEParams | None = Field(
        default_factory=TSNEParams,
        description="Parameters controlling Scanpy t-SNE computation.",
        json_schema_extra={
            "omit_behavior": "Uses `TSNEParams` defaults.",
            "null_behavior": "Equivalent to omission.",
        },
    )


class EvaluationReductionsConfig(_EvaluationConfigBaseModel):
    """Groups dimensionality-reduction steps in execution order.

    PCA, UMAP, and t-SNE are tracked independently, so a recoverable failure in
    one does not prevent later reductions from running. When Scanpy representation
    selection is automatic, UMAP and t-SNE may reuse an available `X_pca` or
    compute one internally.
    """

    pca: PCAConfig = Field(
        default_factory=PCAConfig,
        description="Configuration for the PCA evaluation step.",
        json_schema_extra={
            "omit_behavior": "Uses `PCAConfig` defaults.",
            "null_behavior": "Not allowed; a configuration mapping is required.",
        },
    )

    umap: UMAPConfig = Field(
        default_factory=UMAPConfig,
        description="Configuration for the UMAP evaluation step.",
        json_schema_extra={
            "omit_behavior": "Uses `UMAPConfig` defaults.",
            "null_behavior": "Not allowed; a configuration mapping is required.",
        },
    )

    tsne: TSNEConfig = Field(
        default_factory=TSNEConfig,
        description="Configuration for the t-SNE evaluation step.",
        json_schema_extra={
            "omit_behavior": "Uses `TSNEConfig` defaults.",
            "null_behavior": "Not allowed; a configuration mapping is required.",
        },
    )


# -------------------------
# Clustering config
# -------------------------
class KMeansParams(BaseModel):
    """Controls scikit-learn KMeans clustering and AnnData storage.

    KMeans is fitted directly to `adata.X`. Cluster labels are stored as
    categorical strings in `adata.obs[key_added]`, with clustering provenance
    stored under `adata.uns["benchrep"]["clustering"][key_added]`.

    Additional non-null fields are forwarded to `sklearn.cluster.KMeans`.
    """

    n_clusters: PositiveInt | None = Field(
        default=None,
        description="Number of clusters passed to `sklearn.cluster.KMeans`.",
        json_schema_extra={
            "omit_behavior": (
                "Rejected when KMeans is enabled because `n_clusters` is required; "
                "otherwise leaves the value unset."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Must not exceed the number of observations.",
                "A single cluster is allowed but prevents applicable internal "
                "clustering metrics from being computed.",
            ],
        },
    )

    key_added: str | None = Field(
        default="kmeans",
        description=(
            "Key used by BenchRep to store cluster labels in `adata.obs`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `kmeans`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    random_state: int | None = Field(
        default=137,
        description="Random seed passed to `sklearn.cluster.KMeans`.",
        json_schema_extra={
            "omit_behavior": "Uses 137.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    n_init: Literal["auto"] | PositiveInt | None = Field(
        default="auto",
        description="Initialization count passed to `sklearn.cluster.KMeans`.",
        json_schema_extra={
            "omit_behavior": "Uses `auto`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    overwrite: bool | None = Field(
        default=False,
        description=(
            "Whether BenchRep may replace an existing `adata.obs[key_added]` "
            "entry."
        ),
        json_schema_extra={
            "omit_behavior": "Does not overwrite an existing entry.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A key collision without overwrite is a recoverable step failure."
            ],
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null fields are forwarded as keyword "
                "arguments to `sklearn.cluster.KMeans`."
            ),
        },
    )


class KMeansConfig(EvalStepConfig):
    """Configures the optional scikit-learn KMeans evaluation step.

    KMeans clusters `adata.X` directly and does not consume reduction outputs.
    It is disabled by default and is skipped with a warning if explicitly
    enabled without embeddings.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether KMeans is included in the evaluation pipeline.",
        json_schema_extra={
            "omit_behavior": "Disables KMeans.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    params: KMeansParams | None = Field(
        default_factory=KMeansParams,
        description="Parameters controlling KMeans clustering and output storage.",
        json_schema_extra={
            "omit_behavior": "Uses `KMeansParams` defaults.",
            "null_behavior": (
                "Accepted while KMeans is disabled; rejected when it is enabled."
            ),
        },
    )

    @model_validator(mode="after")
    def validate_kmeans(self) -> KMeansConfig:
        if self.enabled is True and (
            self.params is None or self.params.n_clusters is None
        ):
            raise ValueError(
                "clustering.kmeans.params.n_clusters is required when "
                "KMeans is enabled."
            )

        return self


class LeidenParams(_EvaluationConfigBaseModel):
    """Controls Scanpy neighbor-graph construction and Leiden clustering.

    Cluster labels are stored in `adata.obs[key_added]`. Neighbor metadata is
    stored in `adata.uns[neighbors_key]`, with distance and connectivity
    matrices in `adata.obsp`. BenchRep provenance is stored under
    `adata.uns["benchrep"]["clustering"][key_added]`.

    Unless `use_rep` is set, Scanpy uses `adata.X` below its PCA threshold and
    otherwise reuses or computes `adata.obsm["X_pca"]`.
    """

    n_neighbors: PositiveInt | None = Field(
        default=15,
        description=(
            "Number of kNN neighbors passed to `scanpy.pp.neighbors()`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses 15.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Must be smaller than the number of observations."
            ],
        },
    )

    n_pcs: NonNegativeInt | None = Field(
        default=None,
        description=(
            "Number of principal components passed to "
            "`scanpy.pp.neighbors()`."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Lets Scanpy choose the representation and PCA dimensionality."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Zero forces `adata.X` when `use_rep` is not set."
            ],
        },
    )

    use_rep: str | None = Field(
        default=None,
        description=(
            "Representation key passed to `scanpy.pp.neighbors()`. `X` selects "
            "`adata.X`; any other value names an entry in `adata.obsm`."
        ),
        json_schema_extra={
            "omit_behavior": "Delegates representation selection to Scanpy.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    metric: str | None = Field(
        default="euclidean",
        description="Distance metric passed to `scanpy.pp.neighbors()`.",
        json_schema_extra={
            "omit_behavior": "Uses `euclidean`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    neighbors_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword arguments passed to `scanpy.pp.neighbors()`."
        ),
        json_schema_extra={
            "omit_behavior": "Passes no additional arguments.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Duplicating arguments represented by dedicated fields raises "
                "`TypeError` during backend invocation."
            ],
        },
    )

    resolution: PositiveFloat | None = Field(
        default=1.0,
        description="Resolution passed to `scanpy.tl.leiden()`.",
        json_schema_extra={
            "omit_behavior": "Uses 1.0.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Higher values generally produce more clusters."
            ],
        },
    )

    key_added: str | None = Field(
        default="leiden",
        description=(
            "Key passed to `scanpy.tl.leiden()` for storing labels in "
            "`adata.obs`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `leiden`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    leiden_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword arguments passed to `scanpy.tl.leiden()`."
        ),
        json_schema_extra={
            "omit_behavior": "Passes no additional user arguments.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "BenchRep defaults `flavor='igraph'`, `n_iterations=2`, and "
                "`directed=False` unless overridden here.",
                "Duplicating arguments represented by dedicated fields raises "
                "`TypeError` during backend invocation.",
            ],
        },
    )

    neighbors_key: str | None = Field(
        default="neighbors",
        description=(
            "Namespace used by `scanpy.pp.neighbors()` to store the graph and "
            "by `scanpy.tl.leiden()` to retrieve it."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `neighbors`.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Enabled UMAP and Leiden steps cannot share this key unless "
                "Leiden overwrite is enabled."
            ],
        },
    )

    random_state: int | None = Field(
        default=137,
        description=(
            "Random seed passed to `scanpy.pp.neighbors()` and "
            "`scanpy.tl.leiden()`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses 137.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    overwrite: bool | None = Field(
        default=False,
        description=(
            "Whether BenchRep may replace existing Leiden and neighbor-graph "
            "outputs."
        ),
        json_schema_extra={
            "omit_behavior": "Does not overwrite existing outputs.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A key collision without overwrite is a recoverable step failure."
            ],
        },
    )


class LeidenConfig(EvalStepConfig):
    """Configures the optional Scanpy Leiden evaluation step.

    Leiden constructs its own neighbor graph and does not cluster UMAP
    coordinates. It requires embeddings and the optional Scanpy, igraph, and
    Leiden dependencies. The step is disabled by default and is skipped with a
    warning if explicitly enabled without embeddings.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether Leiden is included in the evaluation pipeline.",
        json_schema_extra={
            "omit_behavior": "Disables Leiden.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    params: LeidenParams | None = Field(
        default_factory=LeidenParams,
        description=(
            "Parameters controlling neighbor construction, Leiden clustering, "
            "and output storage."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `LeidenParams` defaults.",
            "null_behavior": "Equivalent to omission.",
        },
    )


class HDBSCANParams(BaseModel):
    """Controls scikit-learn HDBSCAN clustering and AnnData storage.

    HDBSCAN clusters `adata.X` directly. Labels are stored as categorical
    strings in `adata.obs[key_added]`, where `-1` denotes noise. Membership
    strengths are stored in `adata.obs[f"{key_added}_probability"]`, with
    provenance under `adata.uns["benchrep"]["clustering"][key_added]`.

    Additional non-null fields are forwarded to `sklearn.cluster.HDBSCAN`.
    BenchRep uses `copy=False` unless overridden.
    """

    min_cluster_size: Annotated[int, Field(ge=2)] | None = Field(
        default=5,
        description=(
            "Minimum cluster size passed to `sklearn.cluster.HDBSCAN`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses 5.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    min_samples: PositiveInt | None = Field(
        default=None,
        description=(
            "Core-point neighborhood size passed to "
            "`sklearn.cluster.HDBSCAN`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `min_cluster_size`.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Must not exceed the number of observations."
            ],
        },
    )

    cluster_selection_epsilon: NonNegativeFloat | None = Field(
        default=0.0,
        description=(
            "Cluster-merging distance threshold passed to "
            "`sklearn.cluster.HDBSCAN`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses 0.0.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    metric: str | None = Field(
        default="euclidean",
        description="Distance metric passed to `sklearn.cluster.HDBSCAN`.",
        json_schema_extra={
            "omit_behavior": "Uses `euclidean`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    cluster_selection_method: Literal["eom", "leaf"] | None = Field(
        default="eom",
        description=(
            "Condensed-tree cluster-selection method passed to "
            "`sklearn.cluster.HDBSCAN`."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `eom`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    allow_single_cluster: bool | None = Field(
        default=False,
        description=(
            "Whether `sklearn.cluster.HDBSCAN` may select one non-noise cluster."
        ),
        json_schema_extra={
            "omit_behavior": "Does not allow a single cluster.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    key_added: str | None = Field(
        default="hdbscan",
        description=(
            "Key used by BenchRep to store labels in `adata.obs`; membership "
            "strengths use the corresponding `<key_added>_probability` key."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `hdbscan`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    overwrite: bool | None = Field(
        default=False,
        description=(
            "Whether BenchRep may replace existing label and probability entries."
        ),
        json_schema_extra={
            "omit_behavior": "Does not overwrite existing entries.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A key collision without overwrite is a recoverable step failure."
            ],
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null fields are forwarded as keyword "
                "arguments to `sklearn.cluster.HDBSCAN`."
            ),
        },
    )


class HDBSCANConfig(EvalStepConfig):
    """Configures the optional scikit-learn HDBSCAN evaluation step.

    HDBSCAN clusters `adata.X` directly and does not consume reduction outputs.
    Noise observations are excluded from dependent clustering metrics. Producing
    no non-noise clusters or only one non-noise cluster completes with warnings.

    The step is disabled by default and is skipped with a warning if explicitly
    enabled without embeddings.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether HDBSCAN is included in the evaluation pipeline.",
        json_schema_extra={
            "omit_behavior": "Disables HDBSCAN.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    params: HDBSCANParams | None = Field(
        default_factory=HDBSCANParams,
        description=(
            "Parameters controlling HDBSCAN clustering and output storage."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `HDBSCANParams` defaults.",
            "null_behavior": "Equivalent to omission.",
        },
    )


class EvaluationClusteringConfig(_EvaluationConfigBaseModel):
    """Groups independently tracked clustering evaluation steps.

    KMeans, Leiden, and HDBSCAN run in that order. A recoverable failure in one
    method does not prevent the others from running. KMeans and HDBSCAN cluster
    `adata.X` directly, while Leiden first constructs a Scanpy neighbor graph
    from its configured representation.
    """

    kmeans: KMeansConfig = Field(
        default_factory=KMeansConfig,
        description="Configuration for the KMeans evaluation step.",
        json_schema_extra={
            "omit_behavior": "Uses `KMeansConfig` defaults.",
            "null_behavior": "Not allowed; a configuration mapping is required.",
        },
    )

    leiden: LeidenConfig = Field(
        default_factory=LeidenConfig,
        description="Configuration for the Leiden evaluation step.",
        json_schema_extra={
            "omit_behavior": "Uses `LeidenConfig` defaults.",
            "null_behavior": "Not allowed; a configuration mapping is required.",
        },
    )

    hdbscan: HDBSCANConfig = Field(
        default_factory=HDBSCANConfig,
        description="Configuration for the HDBSCAN evaluation step.",
        json_schema_extra={
            "omit_behavior": "Uses `HDBSCANConfig` defaults.",
            "null_behavior": "Not allowed; a configuration mapping is required.",
        },
    )


# -------------------------
# Reconstruction artifacts config
# -------------------------
class ErrorMapParams(_EvaluationConfigBaseModel):
    """Controls error maps used by TIFF and reconstruction-grid exports.

    This block does not enable error-map output independently. Each
    requested kind is computed separately. Successful kinds are retained
    when another kind fails recoverably.
    """

    kinds: list[ErrorMapKind] = Field(
        default_factory=lambda: ["absolute", "signed", "relative"],
        min_length=1,
        description="Error-map representations passed to `compute_error_maps()`.",
        json_schema_extra={
            "omit_behavior": "Computes `absolute`, `signed`, and `relative` maps.",
            "null_behavior": "Not allowed.",
            "notes": [
                "`signed` returns the residual; `absolute` and `squared` "
                "return its absolute value or square.",
                "`relative` divides absolute error by the absolute input, "
                "bounded below by `denominator_floor`.",
                "`normalized_absolute_global` uses one global input range; "
                "`normalized_absolute_per_channel` infers separate ranges.",
            ],
        },
    )

    denominator_floor: PositiveFloat | None = Field(
        default=None,
        description=(
            "Minimum denominator used by relative and inferred-range "
            "normalization."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `1e-8`.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    data_range: PositiveFloat | None = Field(
        default=None,
        description=(
            "Fixed global intensity range used by "
            "`normalized_absolute_global`."
        ),
        json_schema_extra={
            "omit_behavior": (
                "When global normalization is requested, infers the range "
                "from the input values."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Applies only to `normalized_absolute_global`.",
                "When provided without that kind, it is ignored with a warning.",
            ],
        },
    )


class EvaluationReconstructionConfig(_EvaluationConfigBaseModel):
    """Controls reconstruction TIFF artifacts and shared error-map settings.

    Reconstruction metrics are configured separately. Reconstruction grids use
    their own example selection but reuse the error-map settings defined here.
    """

    export_tiffs: bool = Field(
        default=False,
        description=(
            "Whether to export inputs, reconstructions, and configured error "
            "maps as individual TIFF files."
        ),
        json_schema_extra={
            "omit_behavior": "Does not export reconstruction TIFFs.",
            "null_behavior": "Not allowed.",
            "notes": [
                "If enabled but no reconstruction bundle can be resolved, "
                "configuration resolution raises an error."
            ],
        },
    )

    n_examples: PositiveInt | None = Field(
        default=None,
        description="Maximum number of examples included in TIFF export.",
        json_schema_extra={
            "omit_behavior": "Uses all available reconstruction examples.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Values above the available count are capped.",
                "Does not limit reconstruction metrics or reconstruction grids.",
            ],
        },
    )

    error_maps: ErrorMapParams = Field(
        default_factory=ErrorMapParams,
        description=(
            "Error-map settings shared by TIFF and reconstruction-grid exports."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `ErrorMapParams` defaults.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Does not independently enable error-map computation or export."
            ],
        },
    )


# -------------------------
# Metrics config
# -------------------------

# Clustering ---
class InternalClusteringMetricConfig(EvalMetricGroupConfig):
    """Configures internal metrics for each enabled clustering result.

    Registered metrics receive `adata.X` and the corresponding cluster labels.
    Their returned values are validated and normalized according to their declared
    metric-result contracts.

    Results are stored under
    `adata.uns["benchrep"]["metrics"]["clustering"]["internal"][cluster_key]`.
    Existing results for the same cluster key cause a recoverable step failure
    unless overwriting is enabled. A separate metric step runs for each enabled
    clustering result and depends on that clustering step completing successfully.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether internal clustering metrics are computed.",
        json_schema_extra={
            "omit_behavior": (
                "Enables the group when at least one clustering method is enabled; "
                "otherwise disables it."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "With `enabled=True`, having no enabled clustering method produces "
                "an error instead of disabling the group.",
                "A separate metric step runs for each enabled clustering method.",
            ],
        },
    )

    selected: RegistrySelection = Field(
        default=None,
        description=(
            "Internal clustering metrics to compute using curated defaults, every "
            "registered metric, or an explicit list of names or aliases."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Computes `silhouette`, `calinski_harabasz`, and "
                "`davies_bouldin`."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                '`"all"` selects every registered internal clustering metric, '
                "including custom metrics.",
                "An explicit list selects exactly the provided names or aliases.",
                "An empty list is rejected when the group runs.",
                "Inspect the current canonical names and aliases with "
                '`benchrep.inspect_registry("internal_clustering_metric")`.',
            ],
        },
    )

    overwrite: bool | None = Field(
        default=False,
        description=(
            "Whether BenchRep may replace existing internal metric results for "
            "the same clustering result."
        ),
        json_schema_extra={
            "omit_behavior": "Does not replace existing results.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A result collision without overwrite is a recoverable step failure."
            ],
        },
    )


class ExternalClusteringMetricConfig(EvalMetricGroupConfig):
    """Configures label-referenced metrics for each enabled clustering result.

    Registered metrics receive reference labels from `adata.obs[label_key]` and
    the corresponding cluster assignments. Their returned values are validated
    and normalized according to their declared metric-result contracts.

    Results are stored under
    `adata.uns["benchrep"]["metrics"]["clustering"]["external"][cluster_key]`.
    Existing results for the same cluster key cause a recoverable step failure
    unless overwriting is enabled. A separate metric step runs for each enabled
    clustering result and depends on that clustering step completing successfully.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether external clustering metrics are computed.",
        json_schema_extra={
            "omit_behavior": (
                "Enables the group when clustering is enabled and `label_key` "
                "exists in `adata.obs`; otherwise disables it."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "With `enabled=True`, unavailable clustering or label prerequisites "
                "produce an error instead of disabling the group.",
            ],
        },
    )

    label_key: str = Field(
        default="label",
        description=(
            "Column in `adata.obs` containing the reference labels passed to "
            "external metric callables."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `label`.",
            "null_behavior": "Not allowed.",
        },
    )

    selected: RegistrySelection = Field(
        default=None,
        description=(
            "External clustering metrics to compute using curated defaults, every "
            "registered metric, or an explicit list of names or aliases."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Computes `adjusted_mutual_info`, `adjusted_rand_index`, and "
                "`homogeneity`."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                '`"all"` selects every registered external clustering metric, '
                "including custom metrics.",
                "An explicit list selects exactly the provided names or aliases.",
                "An empty list is rejected when the group runs.",
                "Inspect the current canonical names and aliases with "
                '`benchrep.inspect_registry("external_clustering_metric")`.',
            ],
        },
    )

    overwrite: bool | None = Field(
        default=False,
        description=(
            "Whether BenchRep may replace existing external metric results for "
            "the same clustering result."
        ),
        json_schema_extra={
            "omit_behavior": "Does not replace existing results.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A result collision without overwrite is a recoverable step failure."
            ],
        },
    )


class EvaluationClusteringMetricsConfig(_EvaluationConfigBaseModel):
    """Groups internal and external metrics for clustering outputs.

    Both groups are evaluated independently for every enabled clustering
    method. Observations assigned to HDBSCAN's noise cluster are excluded from
    both internal and external metrics.
    """

    internal: InternalClusteringMetricConfig = Field(
        default_factory=InternalClusteringMetricConfig,
        description="Internal metrics evaluating cluster structure in `adata.X`.",
        json_schema_extra={
            "omit_behavior": "Uses `InternalClusteringMetricConfig` defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    external: ExternalClusteringMetricConfig = Field(
        default_factory=ExternalClusteringMetricConfig,
        description=(
            "External metrics comparing cluster assignments with reference "
            "labels."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `ExternalClusteringMetricConfig` defaults.",
            "null_behavior": "Not allowed.",
        },
    )


# Embedding ---
class EmbeddingMetricConfig(EvalMetricGroupConfig):
    """Configures metrics computed directly from the embedding matrix.

    Registered metrics receive `adata.X`. Their returned values are validated
    and normalized according to their declared metric-result contracts.

    Results are stored under
    `adata.uns["benchrep"]["metrics"]["embedding"]`. Existing embedding metric
    results cause a recoverable step failure unless overwriting is enabled.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether embedding metrics are computed.",
        json_schema_extra={
            "omit_behavior": (
                "Enables the group when embeddings are available; otherwise "
                "disables it."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "With `enabled=True`, unavailable embeddings produce an error "
                "instead of disabling the group.",
            ],
        },
    )

    selected: RegistrySelection = Field(
        default=None,
        description=(
            "Embedding metrics to compute using curated defaults, every "
            "registered metric, or an explicit list of names or aliases."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Computes `mean`, `median`, `standard_deviation`, `minimum`, "
                "`maximum`, and `quantiles`."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                '`"all"` selects every registered embedding metric, including '
                "custom metrics.",
                "An explicit list selects exactly the provided names or aliases.",
                "An empty list is rejected when the group runs.",
                "Inspect the current canonical names and aliases with "
                '`benchrep.inspect_registry("embedding_metric")`.',
            ],
        },
    )

    overwrite: bool | None = Field(
        default=False,
        description=(
            "Whether BenchRep may replace existing embedding metric results."
        ),
        json_schema_extra={
            "omit_behavior": "Does not replace existing results.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A result collision without overwrite is a recoverable step failure."
            ],
        },
    )


# Reconstruction ---
class ReconstructionMetricConfig(EvalMetricGroupConfig):
    """Configures metrics comparing inputs with their reconstructions.

    Registered metrics receive the input and reconstruction arrays. Their
    returned values are validated and normalized according to their declared
    metric-result contracts.

    Metrics may run once over the complete arrays, independently for each
    reconstruction channel, or both.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether reconstruction metrics are computed.",
        json_schema_extra={
            "omit_behavior": (
                "Enables the group when reconstruction inputs are available; "
                "otherwise disables it."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "With `enabled=True`, unavailable reconstruction inputs produce "
                "an error instead of disabling the group.",
            ],
        },
    )

    selected: RegistrySelection = Field(
        default=None,
        description=(
            "Reconstruction metrics to compute using curated defaults, every "
            "registered metric, or an explicit list of names or aliases."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Computes `mae`, `mse`, `rmse`, and `max_absolute_error`."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                '`"all"` selects every registered reconstruction metric, '
                "including custom metrics.",
                "An explicit list selects exactly the provided names or aliases.",
                "An empty list is rejected when the group runs.",
                "Inspect the current canonical names and aliases with "
                '`benchrep.inspect_registry("reconstruction_metric")`.',
            ],
        },
    )

    reduction: Literal["global", "per_channel", "both"] = Field(
        default="global",
        description=(
            "Whether metrics are computed over the complete arrays, separately "
            "for each channel, or both."
        ),
        json_schema_extra={
            "omit_behavior": "Computes each metric globally.",
            "null_behavior": "Not allowed.",
        },
    )


# Predictability ---
class EvaluationCrossValidationConfig(_EvaluationConfigBaseModel):
    """Configures outer cross-validation for predictability probes.

    Outer folds estimate probe performance on held-out observations. When
    hyperparameter tuning is enabled, inner cross-validation uses the same
    strategy family within each outer training fold. Grouped strategies keep
    all observations belonging to the same group within one fold.
    """

    method: Literal[
        "stratified_kfold",
        "kfold",
        "group_kfold",
        "stratified_group_kfold",
    ] | None = Field(
        default=None,
        description="Cross-validation strategy used for predictability evaluation.",
        json_schema_extra={
            "omit_behavior": (
                "Uses `stratified_kfold` for classification and `kfold` for "
                "regression."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "`stratified_kfold` and `stratified_group_kfold` are only valid "
                "for classification.",
                "`group_kfold` and `stratified_group_kfold` require `group_key`.",
                "When tuning is enabled, the inner cross-validation uses the same "
                "strategy family.",
            ],
        },
    )

    n_splits: NSplits = Field(
        default=5,
        description="Number of outer cross-validation folds.",
        json_schema_extra={
            "omit_behavior": "Uses five outer folds.",
            "null_behavior": "Not allowed.",
            "notes": [
                "The available observations, groups, and class counts are "
                "validated when predictability evaluation runs.",
            ],
        },
    )

    group_key: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1),
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "AnnData observation column containing group labels for grouped "
            "cross-validation."
        ),
        json_schema_extra={
            "omit_behavior": "Does not provide grouping labels.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Required by `group_kfold` and `stratified_group_kfold`.",
                "Each group is kept entirely within a single fold.",
            ],
        },
    )

    shuffle: bool = Field(
        default=True,
        description="Whether observations or groups are shuffled before splitting.",
        json_schema_extra={
            "omit_behavior": "Enables shuffling.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Grouped strategies preserve group boundaries while shuffling.",
                "`random_state` is ignored when shuffling is disabled.",
                "When tuning is enabled, the same setting applies to inner "
                "cross-validation.",
            ],
        },
    )

    random_state: int | None = Field(
        default=137,
        description="Random seed used by shuffled cross-validation splitters.",
        json_schema_extra={
            "omit_behavior": "Uses seed 137.",
            "null_behavior": "Does not use a fixed seed.",
            "notes": [
                "Ignored when shuffling is disabled.",
                "When tuning is enabled, the same seed is used for inner "
                "cross-validation.",
            ],
        },
    )

    scoring: Literal[
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "accuracy",
        "r2",
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
    ] | None = Field(
        default=None,
        description=(
            "Scoring function used to evaluate probe performance and, when "
            "tuning is enabled, select hyperparameters."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses `balanced_accuracy` for classification and `r2` for "
                "regression."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "`balanced_accuracy`, `f1_macro`, `f1_weighted`, and `accuracy` "
                "are classification scorers.",
                "`r2`, `neg_mean_absolute_error`, and "
                "`neg_root_mean_squared_error` are regression scorers.",
                "Negated error scorers follow scikit-learn's convention that "
                "higher scores are better.",
            ],
        },
    )

    @model_validator(mode="after")
    def validate_cv(self) -> EvaluationCrossValidationConfig:
        if (
            self.method in {"group_kfold", "stratified_group_kfold"}
            and self.group_key is None
        ):
            raise ValueError(
                "cv.group_key is required when cv.method is `group_kfold` or "
                "`stratified_group_kfold`."
            )

        return self


class TuningInnerCVConfig(_EvaluationConfigBaseModel):
    """Configures inner cross-validation used for hyperparameter selection."""

    n_splits: NSplits = Field(
        default=3,
        description="Number of inner cross-validation folds.",
        json_schema_extra={
            "omit_behavior": "Uses three inner folds.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Inner folds are created separately within each outer training "
                "fold.",
            ],
        },
    )


class EvaluationCVTuningConfig(_EvaluationConfigBaseModel):
    """Configures nested cross-validated hyperparameter tuning.

    When enabled, list-valued parameters for selected probes define search
    grids. Hyperparameters are selected within each outer training fold using
    inner cross-validation, leaving the corresponding outer test fold untouched.
    """

    enabled: bool = Field(
        default=False,
        description="Whether probe hyperparameters are tuned with inner CV.",
        json_schema_extra={
            "omit_behavior": "Disables hyperparameter tuning.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Enabling tuning requires at least one list-valued parameter "
                "for a selected probe.",
                "List-valued probe parameters are rejected while tuning is "
                "disabled.",
            ],
        },
    )

    inner_cv: TuningInnerCVConfig = Field(
        default_factory=TuningInnerCVConfig,
        description="Inner cross-validation settings used during tuning.",
        json_schema_extra={
            "omit_behavior": "Uses three inner folds.",
            "null_behavior": "Not allowed.",
            "notes": [
                "The inner splitter uses the outer CV strategy, shuffle setting, "
                "and random seed.",
            ],
        },
    )


class DummyProbeConfig(_EvaluationConfigBaseModel):
    """Configures a task-appropriate non-informative baseline probe."""

    strategy: Literal[
        "most_frequent",
        "stratified",
        "uniform",
        "mean",
        "median",
    ] | None = Field(
        default=None,
        description="Baseline prediction strategy used by the dummy probe.",
        json_schema_extra={
            "omit_behavior": (
                "Uses `most_frequent` for classification and `mean` for "
                "regression."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "`most_frequent`, `stratified`, and `uniform` are classification "
                "strategies.",
                "`mean` and `median` are regression strategies.",
            ],
        },
    )

    random_state: int | None = Field(
        default=137,
        description="Random seed used by randomized dummy strategies.",
        json_schema_extra={
            "omit_behavior": "Uses seed 137.",
            "null_behavior": "Does not use a fixed seed.",
            "notes": [
                "Only affects the classification strategies `stratified` and "
                "`uniform`.",
            ],
        },
    )


class LogisticRegressionProbeConfig(BaseModel):
    """Configures the linear classification probe.

    Recognized fields configure BenchRep's standard logistic-regression setup.
    Additional fields are accepted and forwarded to scikit-learn's
    `LogisticRegression`. List-valued parameters define hyperparameter grids.
    """

    model: Literal["logistic_regression"] = Field(
        default="logistic_regression",
        description="Discriminator selecting logistic regression.",
        json_schema_extra={
            "omit_behavior": "Uses logistic regression.",
            "null_behavior": "Not allowed.",
        },
    )

    standardize: bool = Field(
        default=True,
        description="Whether embedding features are standardized before fitting.",
        json_schema_extra={
            "omit_behavior": "Standardizes embedding features.",
            "null_behavior": "Not allowed.",
            "notes": [
                "The scaler is fitted within each cross-validation training fold."
            ],
        },
    )

    C: PositiveFloatOrList = Field(
        default=1.0,
        description=(
            "Inverse regularization strength, or candidate values for tuning."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `C=1.0`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Smaller values apply stronger regularization.",
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    class_weight: Literal["balanced"] | None = Field(
        default=None,
        description="Optional automatic weighting of classes by frequency.",
        json_schema_extra={
            "omit_behavior": "Does not apply class weighting.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    max_iter: PositiveInt = Field(
        default=5000,
        description="Maximum number of solver iterations.",
        json_schema_extra={
            "omit_behavior": "Allows up to 5000 iterations.",
            "null_behavior": "Not allowed.",
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null, non-list fields are forwarded as "
                "keyword arguments to `sklearn.linear_model.LogisticRegression`. "
                "List-valued fields define hyperparameter candidates when tuning "
                "is enabled."
            ),
        },
    )


class RidgeProbeConfig(BaseModel):
    """Configures the linear regression probe.

    Recognized fields configure BenchRep's standard ridge-regression setup.
    Additional fields are accepted and interpreted as scikit-learn `Ridge`
    parameters. List-valued parameters define hyperparameter grids.
    """

    model: Literal["ridge"] = Field(
        default="ridge",
        description="Discriminator selecting ridge regression.",
        json_schema_extra={
            "omit_behavior": "Uses ridge regression.",
            "null_behavior": "Not allowed.",
        },
    )

    standardize: bool = Field(
        default=True,
        description="Whether embedding features are standardized before fitting.",
        json_schema_extra={
            "omit_behavior": "Standardizes embedding features.",
            "null_behavior": "Not allowed.",
            "notes": [
                "The scaler is fitted within each cross-validation training fold."
            ],
        },
    )

    alpha: PositiveFloatOrList = Field(
        default=1.0,
        description="L2 regularization strength, or candidate values for tuning.",
        json_schema_extra={
            "omit_behavior": "Uses `alpha=1.0`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Larger values apply stronger regularization.",
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null, non-list fields are forwarded as "
                "keyword arguments to `sklearn.linear_model.Ridge`. List-valued "
                "fields define hyperparameter candidates when tuning is enabled."
            ),
        },
    )


LinearProbeConfig: TypeAlias = Annotated[
    LogisticRegressionProbeConfig | RidgeProbeConfig,
    Field(discriminator="model"),
]


class KNNProbeConfig(BaseModel):
    """Configures the task-dependent nearest-neighbor probe.

    Classification uses scikit-learn's `KNeighborsClassifier`, while regression
    uses `KNeighborsRegressor`. Additional fields are accepted and interpreted
    as parameters for the task-appropriate estimator. List-valued parameters
    define hyperparameter grids.
    """

    standardize: bool = Field(
        default=True,
        description="Whether embedding features are standardized before fitting.",
        json_schema_extra={
            "omit_behavior": "Standardizes embedding features.",
            "null_behavior": "Not allowed.",
            "notes": [
                "The scaler is fitted within each cross-validation training fold."
            ],
        },
    )

    n_neighbors: PositiveIntOrList = Field(
        default=15,
        description="Number of neighbors, or candidate values for tuning.",
        json_schema_extra={
            "omit_behavior": "Uses 15 neighbors.",
            "null_behavior": "Not allowed.",
            "notes": [
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    weights: KNNWeightsOrList = Field(
        default="distance",
        description="Neighbor weighting strategy, or candidates for tuning.",
        json_schema_extra={
            "omit_behavior": "Weights neighbors by inverse distance.",
            "null_behavior": "Not allowed.",
            "notes": [
                "`uniform` weights every neighbor equally.",
                "`distance` gives closer neighbors greater influence.",
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    metric: KNNMetricOrList = Field(
        default="euclidean",
        description="Distance metric, or candidate metric names for tuning.",
        json_schema_extra={
            "omit_behavior": "Uses Euclidean distance.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Metric names are interpreted by scikit-learn.",
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null, non-list fields are forwarded as "
                "keyword arguments to the task-appropriate scikit-learn "
                "KNeighbors estimator. List-valued fields define hyperparameter "
                "candidates when tuning is enabled."
            ),
        },
    )


class RandomForestProbeConfig(BaseModel):
    """Configures the task-dependent random-forest probe.

    Classification uses scikit-learn's `RandomForestClassifier`, while
    regression uses `RandomForestRegressor`. Additional fields are accepted and
    interpreted as parameters for the task-appropriate estimator. List-valued
    parameters define hyperparameter grids.
    """

    n_estimators: PositiveIntOrList = Field(
        default=500,
        description="Number of trees, or candidate values for tuning.",
        json_schema_extra={
            "omit_behavior": "Uses 500 trees.",
            "null_behavior": "Not allowed.",
            "notes": [
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    max_depth: MaxDepthParam = Field(
        default=None,
        description="Maximum tree depth, or candidate values for tuning.",
        json_schema_extra={
            "omit_behavior": "Allows trees to grow without a depth limit.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A tuning list may include `null` as the unlimited-depth "
                "candidate.",
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    class_weight: Literal["balanced"] | None = Field(
        default=None,
        description="Optional automatic weighting of classification classes.",
        json_schema_extra={
            "omit_behavior": "Does not apply class weighting.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Only valid for classification; regression rejects this field.",
            ],
        },
    )

    random_state: int | None = Field(
        default=137,
        description="Random seed used by the random-forest estimator.",
        json_schema_extra={
            "omit_behavior": "Uses seed 137.",
            "null_behavior": "Does not use a fixed seed.",
        },
    )

    n_jobs: int | None = Field(
        default=-1,
        description="Number of parallel worker jobs used when fitting.",
        json_schema_extra={
            "omit_behavior": "Uses all available processors.",
            "null_behavior": "Uses scikit-learn's default job count.",
            "notes": [
                "`-1` uses all available processors.",
            ],
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null, non-list fields are forwarded as "
                "keyword arguments to the task-appropriate scikit-learn random-"
                "forest estimator. List-valued fields define hyperparameter "
                "candidates when tuning is enabled."
            ),
        },
    )


class XGBoostProbeConfig(BaseModel):
    """Configures the task-dependent optional XGBoost probe.

    Classification uses `xgboost.XGBClassifier`, while regression uses
    `xgboost.XGBRegressor`. XGBoost is imported only when this probe runs.
    Additional fields are accepted and interpreted as parameters for the
    task-appropriate estimator. List-valued parameters define hyperparameter
    grids.
    """

    n_estimators: PositiveIntOrList = Field(
        default=300,
        description="Number of boosting rounds, or candidate values for tuning.",
        json_schema_extra={
            "omit_behavior": "Uses 300 boosting rounds.",
            "null_behavior": "Not allowed.",
            "notes": [
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    max_depth: PositiveIntOrList | None = Field(
        default=None,
        description="Maximum tree depth, or candidate values for tuning.",
        json_schema_extra={
            "omit_behavior": "Uses the XGBoost backend default.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    learning_rate: PositiveFloatOrList = Field(
        default=0.01,
        description="Boosting learning rate, or candidate values for tuning.",
        json_schema_extra={
            "omit_behavior": "Uses a learning rate of 0.01.",
            "null_behavior": "Not allowed.",
            "notes": [
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    random_state: int | None = Field(
        default=137,
        description="Random seed used by the XGBoost estimator.",
        json_schema_extra={
            "omit_behavior": "Uses seed 137.",
            "null_behavior": "Does not use a fixed seed.",
        },
    )

    n_jobs: int | None = Field(
        default=None,
        description="Number of parallel worker threads used when fitting.",
        json_schema_extra={
            "omit_behavior": (
                "Uses the XGBoost backend default, which uses all available "
                "threads."
            ),
            "null_behavior": "Equivalent to omission.",
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null, non-list fields are forwarded as "
                "keyword arguments to the task-appropriate XGBoost estimator. "
                "List-valued fields define hyperparameter candidates when tuning "
                "is enabled."
            ),
        },
    )


class SVMRBFProbeConfig(BaseModel):
    """Configures the task-dependent radial-basis-function SVM probe.

    Classification uses scikit-learn's `SVC`, while regression uses `SVR`.
    Embeddings may be standardized before fitting. Additional fields are
    accepted and interpreted as parameters for the task-appropriate estimator.
    List-valued parameters define hyperparameter grids.
    """

    standardize: bool = Field(
        default=True,
        description="Whether embedding features are standardized before fitting.",
        json_schema_extra={
            "omit_behavior": "Standardizes embedding features.",
            "null_behavior": "Not allowed.",
            "notes": [
                "The scaler is fitted within each cross-validation training fold."
            ],
        },
    )

    C: PositiveFloatOrList = Field(
        default=1.0,
        description=(
            "Inverse regularization strength, or candidate values for tuning."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `C=1.0`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Smaller values apply stronger regularization.",
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    gamma: SVMRBFGammaValueOrList = Field(
        default="scale",
        description="RBF kernel coefficient, or candidate values for tuning.",
        json_schema_extra={
            "omit_behavior": "Uses scikit-learn's `scale` heuristic.",
            "null_behavior": "Not allowed.",
            "notes": [
                "`scale` derives gamma from feature count and variance.",
                "`auto` uses the inverse feature count.",
                "A list requires predictability tuning to be enabled.",
            ],
        },
    )

    class_weight: Literal["balanced"] | None = Field(
        default=None,
        description="Optional automatic weighting of classification classes.",
        json_schema_extra={
            "omit_behavior": "Does not apply class weighting.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Only valid for classification; regression rejects this field.",
            ],
        },
    )

    cache_size: PositiveFloat = Field(
        default=200.0,
        description="Kernel cache size in megabytes.",
        json_schema_extra={
            "omit_behavior": "Uses a 200 MB kernel cache.",
            "null_behavior": "Not allowed.",
        },
    )

    max_iter: MaxIterWithNoLimitSentinel = Field(
        default=-1,
        description="Maximum solver iterations.",
        json_schema_extra={
            "omit_behavior": "Runs without an iteration limit.",
            "null_behavior": "Not allowed.",
            "notes": [
                "`-1` disables the iteration limit.",
            ],
        },
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null, non-list fields are forwarded as "
                "keyword arguments to the task-appropriate scikit-learn SVM "
                "estimator. List-valued fields define hyperparameter candidates "
                "when tuning is enabled."
            ),
        },
    )


class EvaluationPredictabilityParamsConfig(_EvaluationConfigBaseModel):
    """Groups parameter configurations by canonical predictability probe name.

    Only parameters belonging to selected probes are used. List-valued estimator
    parameters define search grids and therefore require tuning to be enabled.
    """

    dummy: DummyProbeConfig = Field(
        default_factory=DummyProbeConfig,
        description="Parameters for the task-dependent dummy baseline.",
        json_schema_extra={
            "omit_behavior": "Uses task-appropriate dummy-probe defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    linear: LinearProbeConfig | None = Field(
        default=None,
        description="Parameters for the task-dependent linear probe.",
        json_schema_extra={
            "omit_behavior": (
                "Uses logistic regression defaults for classification and ridge "
                "defaults for regression."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "An explicit mapping must set `model` to `logistic_regression` or "
                "`ridge`.",
            ],
        },
    )

    knn: KNNProbeConfig = Field(
        default_factory=KNNProbeConfig,
        description="Parameters for the task-dependent nearest-neighbor probe.",
        json_schema_extra={
            "omit_behavior": "Uses `KNNProbeConfig` defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    random_forest: RandomForestProbeConfig = Field(
        default_factory=RandomForestProbeConfig,
        description="Parameters for the task-dependent random-forest probe.",
        json_schema_extra={
            "omit_behavior": "Uses `RandomForestProbeConfig` defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    xgboost: XGBoostProbeConfig = Field(
        default_factory=XGBoostProbeConfig,
        description="Parameters for the optional task-dependent XGBoost probe.",
        json_schema_extra={
            "omit_behavior": "Uses `XGBoostProbeConfig` defaults.",
            "null_behavior": "Not allowed.",
            "notes": [
                "These parameters have no effect unless `xgboost` is selected.",
            ],
        },
    )

    svm_rbf: SVMRBFProbeConfig = Field(
        default_factory=SVMRBFProbeConfig,
        description="Parameters for the task-dependent RBF-SVM probe.",
        json_schema_extra={
            "omit_behavior": "Uses `SVMRBFProbeConfig` defaults.",
            "null_behavior": "Not allowed.",
        },
    )


class EvaluationPredictabilityConfig(EvalStepConfig):
    """Configures cross-validated probes of embedding predictability.

    Registered probe builders construct supervised estimators that predict
    `target_key` from the embedding matrix. Results are stored under
    `adata.uns["benchrep"]["metrics"]["predictability"][target_key]`. Existing
    results for the same target cause a recoverable step failure unless
    overwriting is enabled.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether predictability probes are evaluated.",
        json_schema_extra={
            "omit_behavior": "Disables predictability evaluation.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "With `enabled=True`, unavailable embeddings or target data produce "
                "an error instead of disabling the step.",
            ],
        },
    )

    selected: RegistrySelection = Field(
        default=None,
        description=(
            "Predictability probes to evaluate using curated defaults, every "
            "registered probe, or an explicit list of names or aliases."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses `dummy`, `linear`, `knn`, `random_forest`, and `svm_rbf`."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                '`"all"` selects every registered predictability probe.',
                "An explicit list selects exactly the provided names or aliases.",
                "An empty list is rejected when the step runs.",
                "Inspect the current canonical names and aliases with "
                '`benchrep.inspect_registry("predictability_probe")`.',
            ],
        },
    )

    target_key: str = Field(
        default="label",
        description=(
            "AnnData observation column containing the prediction target."
        ),
        json_schema_extra={
            "omit_behavior": "Uses `adata.obs['label']`.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Classification accepts categorical targets.",
                "Regression requires finite numeric targets.",
                "Results are stored under a namespace keyed by `target_key`.",
            ],
        },
    )

    overwrite: bool | None = Field(
        default=False,
        description=(
            "Whether BenchRep may replace existing predictability results for "
            "`target_key`."
        ),
        json_schema_extra={
            "omit_behavior": "Does not replace existing results.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "A result collision without overwrite is a recoverable step failure."
            ],
        },
    )

    task: Literal["classification", "regression"] = Field(
        default="classification",
        description="Supervised prediction task performed by the probes.",
        json_schema_extra={
            "omit_behavior": "Uses classification.",
            "null_behavior": "Not allowed.",
            "notes": [
                "The task determines the estimator family, default CV strategy, "
                "default scorer, and task-dependent probe defaults.",
            ],
        },
    )

    cv: EvaluationCrossValidationConfig = Field(
        default_factory=EvaluationCrossValidationConfig,
        description="Outer cross-validation and scoring configuration.",
        json_schema_extra={
            "omit_behavior": "Uses task-appropriate five-fold CV defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    tuning: EvaluationCVTuningConfig = Field(
        default_factory=EvaluationCVTuningConfig,
        description="Nested cross-validated hyperparameter tuning configuration.",
        json_schema_extra={
            "omit_behavior": "Disables hyperparameter tuning.",
            "null_behavior": "Not allowed.",
        },
    )

    params: EvaluationPredictabilityParamsConfig = Field(
        default_factory=EvaluationPredictabilityParamsConfig,
        description="Per-probe estimator parameters and tuning candidates.",
        json_schema_extra={
            "omit_behavior": "Uses task-appropriate probe defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    @model_validator(mode="after")
    def validate_predictability_config(self) -> EvaluationPredictabilityConfig:
        if self.enabled is not True:
            return self

        if self.selected == []:
            raise ValueError(
                "predictability.selected cannot be empty when predictability is enabled."
            )

        if self.target_key.strip() == "":
            raise ValueError(
                "predictability.target_key must be a non-empty string when "
                "predictability is enabled."
            )

        return self


# Full evaluation metrics config ---
class EvaluationMetricsConfig(_EvaluationConfigBaseModel):
    """Groups evaluation metric and predictability configurations.

    Each nested configuration resolves its own enablement, selection, parameters,
    prerequisites, and result-storage behavior.
    """

    clustering: EvaluationClusteringMetricsConfig = Field(
        default_factory=EvaluationClusteringMetricsConfig,
        description="Internal and external clustering metric configurations.",
        json_schema_extra={
            "omit_behavior": "Uses clustering metric defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    embedding: EmbeddingMetricConfig = Field(
        default_factory=EmbeddingMetricConfig,
        description="Embedding summary metric configuration.",
        json_schema_extra={
            "omit_behavior": "Uses embedding metric defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    predictability: EvaluationPredictabilityConfig = Field(
        default_factory=EvaluationPredictabilityConfig,
        description="Cross-validated embedding predictability configuration.",
        json_schema_extra={
            "omit_behavior": "Disables predictability evaluation.",
            "null_behavior": "Not allowed.",
        },
    )

    reconstruction: ReconstructionMetricConfig = Field(
        default_factory=ReconstructionMetricConfig,
        description="Reconstruction comparison metric configuration.",
        json_schema_extra={
            "omit_behavior": "Uses reconstruction metric defaults.",
            "null_behavior": "Not allowed.",
        },
    )


# -------------------------
# Plots config
# -------------------------
class ReconstructionGridConfig(_EvaluationConfigBaseModel):
    """Configures paginated reconstruction comparison grids.

    Each grid compares original inputs with their reconstructions for one
    selected channel. Without stratification, at most 12 examples are sampled
    into one grid. With stratification, one example is sampled per distinct
    stratum and the results are divided into pages of at most 12 examples.
    """

    include_error_maps: bool = Field(
        default=True,
        description=(
            "Whether reconstruction error maps are included alongside inputs "
            "and reconstructions."
        ),
        json_schema_extra={
            "omit_behavior": "Includes configured reconstruction error maps.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Error maps use the parameters configured under "
                "`reconstruction.error_maps`.",
            ],
        },
    )

    random_state: int = Field(
        default=137,
        description=(
            "Random seed used to select and order reconstruction examples."
        ),
        json_schema_extra={
            "omit_behavior": "Uses seed 137.",
            "null_behavior": "Not allowed.",
        },
    )

    stratify_by: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ] | None = Field(
        default=None,
        description=(
            "Reconstruction observation field used to sample one example per "
            "distinct value."
        ),
        json_schema_extra={
            "omit_behavior": "Randomly samples at most 12 examples without stratification.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "The field must contain one scalar value per reconstruction example.",
                "Stratified selections are divided into pages of at most 12 examples.",
                "Missing values are treated as one distinct stratum.",
            ],
        },
    )

    channel_selection: (
        Literal["all"]
        | NonNegativeInt
        | Annotated[list[NonNegativeInt], Field(min_length=1)]
        | None
    ) = Field(
        default=None,
        description=(
            "Zero-based reconstruction channel index or indices included in grids."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Uses channel 0 and warns when multiple channels are available."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                '`"all"` selects every available channel.',
                "An integer selects one channel.",
                "A list selects the specified channels and removes duplicates.",
                "Each selected channel is exported to separate grid files.",
            ],
        },
    )


class PlotParams(_EvaluationConfigBaseModel):
    """Configures shared evaluation plot styling and file output.

    These settings apply to reduction plots, clustering diagnostics, and
    reconstruction grids where relevant.
    """

    accent_color: HexColor = Field(
        default="#6A3D9A",
        description=(
            "Hexadecimal color used for uncolored projections and diagnostic plots."
        ),
        json_schema_extra={
            "omit_behavior": "Uses dark purple (`#6A3D9A`).",
            "null_behavior": "Not allowed.",
        },
    )

    color_by: list[str] | None = Field(
        default=None,
        description=(
            "Additional AnnData observation columns used to color reduction plots."
        ),
        json_schema_extra={
            "omit_behavior": "Does not request additional coloring fields.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Clustering output keys and the external-metric label key are added "
                "automatically when relevant.",
                "Duplicate and blank names are removed.",
                "Fields unavailable when plots are exported are skipped.",
            ],
        },
    )

    dpi: PositiveInt = Field(
        default=300,
        description="Resolution in dots per inch used for exported plots.",
        json_schema_extra={
            "omit_behavior": "Uses 300 DPI.",
            "null_behavior": "Not allowed.",
        },
    )

    formats: Annotated[
        list[Literal["png", "pdf", "svg"]],
        Field(min_length=1),
    ] = Field(
        default_factory=lambda: ["png"],
        description="File formats written for each evaluation plot.",
        json_schema_extra={
            "omit_behavior": "Exports PNG files.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Multiple formats produce a separate file for each plot.",
                "Duplicate formats are exported only once.",
            ],
        },
    )

    reconstruction_grid: ReconstructionGridConfig = Field(
        default_factory=ReconstructionGridConfig,
        description="Parameters controlling reconstruction comparison grids.",
        json_schema_extra={
            "omit_behavior": "Uses reconstruction-grid defaults.",
            "null_behavior": "Not allowed.",
        },
    )


class EvaluationPlotsConfig(EvalStepConfig):
    """Configures evaluation figure exports.

    Plotting operates on available reduction, clustering, and reconstruction
    outputs. Individual plot categories are exported only when their required
    data was produced or supplied.
    """

    enabled: bool | None = Field(
        default=None,
        description="Whether evaluation plots are exported when applicable.",
        json_schema_extra={
            "omit_behavior": (
                "Enables plots for available reduction, clustering, and "
                "reconstruction outputs."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "Unavailable plot categories are not exported.",
                "Plot export failures are recoverable and recorded in the "
                "evaluation outcome summary.",
            ],
        },
    )

    params: PlotParams | None = Field(
        default_factory=PlotParams,
        description="Shared styling, file-output, and reconstruction-grid parameters.",
        json_schema_extra={
            "omit_behavior": "Uses `PlotParams` defaults.",
            "null_behavior": "Equivalent to omission.",
        },
    )


# -------------------------
# Full evaluation configuration
# -------------------------
class EvaluationConfig(_EvaluationConfigBaseModel):
    """Complete configuration for a BenchRep evaluation workflow.

    Evaluation may consume embeddings, reconstructions, or both. The configured
    pipeline can perform dimensionality reduction, clustering, metric evaluation,
    predictability probing, and artifact or figure export.

    Use `benchrep.inspect_config(EvaluationConfig)` to inspect this configuration.
    Nested configuration types shown in the output can be inspected the same
    way, for example `benchrep.inspect_config(EvaluationSourceConfig)` or
    `benchrep.inspect_config(ReconstructionMetricConfig)`. Public configuration
    classes are available from `benchrep.assembly.schemas`.

    Use `benchrep.inspect_registry()` to discover component registries and
    `benchrep.inspect_registry("<registry>", "<component>")` to inspect a
    registered implementation.

    For machine-readable discovery, `EvaluationConfig.model_json_schema()` returns
    standard JSON Schema, while `benchrep.list_registries()` and
    `benchrep.list_registered_components()` return structured registry data.
    """

    stage: Literal["evaluation"] = Field(
        default="evaluation",
        description="Identifies this configuration as an evaluation workflow.",
        json_schema_extra={
            "omit_behavior": "Uses `evaluation`.",
            "null_behavior": "Not allowed.",
        },
    )

    source: EvaluationSourceConfig = Field(
        default_factory=EvaluationSourceConfig,
        description="Input artifact and prediction-manifest configuration.",
        json_schema_extra={
            "omit_behavior": (
                "Requires a prediction manifest supplied through the evaluation "
                "entrypoint; otherwise no evaluation source is available."
            ),
            "null_behavior": "Not allowed.",
        },
    )

    run: EvaluationRunConfig = Field(
        default_factory=EvaluationRunConfig,
        description="Evaluation output location and generated run identity.",
        json_schema_extra={
            "omit_behavior": "Uses evaluation run defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    reductions: EvaluationReductionsConfig = Field(
        default_factory=EvaluationReductionsConfig,
        description="Embedding dimensionality-reduction configurations.",
        json_schema_extra={
            "omit_behavior": "Uses dimensionality-reduction defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    clustering: EvaluationClusteringConfig = Field(
        default_factory=EvaluationClusteringConfig,
        description="Embedding clustering configurations.",
        json_schema_extra={
            "omit_behavior": "Uses clustering defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    metrics: EvaluationMetricsConfig = Field(
        default_factory=EvaluationMetricsConfig,
        description=(
            "Clustering, embedding, predictability, and reconstruction metric "
            "configurations."
        ),
        json_schema_extra={
            "omit_behavior": "Uses evaluation metric defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    reconstruction: EvaluationReconstructionConfig = Field(
        default_factory=EvaluationReconstructionConfig,
        description="Reconstruction TIFF export and shared error-map configuration.",
        json_schema_extra={
            "omit_behavior": "Uses reconstruction export defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    plots: EvaluationPlotsConfig = Field(
        default_factory=EvaluationPlotsConfig,
        description="Evaluation figure-export configuration.",
        json_schema_extra={
            "omit_behavior": "Uses evaluation plot defaults.",
            "null_behavior": "Not allowed.",
        },
    )

    @model_validator(mode="after")
    def validate_source(self, info: ValidationInfo) -> EvaluationConfig:
        prediction_manifest_path_overridden = (info.context or {}).get(
            "prediction_manifest_path_overridden",
            False,
        )

        if (
            self.source.embeddings_path is None
            and self.source.reconstructions_path is None
            and self.source.prediction_manifest_path is None
            and not prediction_manifest_path_overridden
        ):
            raise ValueError(
                "At least one evaluation source must be provided through "
                "source.embeddings_path, source.reconstructions_path, "
                "source.prediction_manifest_path, or the "
                "prediction_manifest_path workflow argument."
            )

        return self