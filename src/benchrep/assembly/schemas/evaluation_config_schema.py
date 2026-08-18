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

NSplits = Annotated[int, Field(ge=2)]

PositiveFloatOrList: TypeAlias = PositiveFloat | list[PositiveFloat]
PositiveIntOrList: TypeAlias = PositiveInt | list[PositiveInt]

KNNWeights: TypeAlias = Literal["uniform", "distance"]
KNNWeightsOrList: TypeAlias = KNNWeights | list[KNNWeights]

SVMRBFGammaValue: TypeAlias = Literal["scale", "auto"] | PositiveFloat
SVMRBFGammaValueOrList: TypeAlias = SVMRBFGammaValue | list[SVMRBFGammaValue]
MaxIterWithNoLimitSentinel: TypeAlias = Literal[-1] | PositiveInt

MaxDepthValue: TypeAlias = PositiveInt | None
MaxDepthParam: TypeAlias = MaxDepthValue | list[MaxDepthValue]

PredictabilityProbeName = Literal[
    "dummy",
    "linear",
    "knn",
    "random_forest",
    "xgboost",
    "svm_rbf"
]

ErrorMapKind = Literal[
    "absolute",
    "squared",
    "signed",
    "relative",
    "normalized_absolute_global",
    "normalized_absolute_per_channel",
]

HexColor = Annotated[
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
            "null_behavior": (
                "Uses an empty parameter mapping, allowing runtime defaults to apply."
            ),
        },
    )


class EvalMetricGroupConfig(_EvaluationConfigBaseModel):
    """Shared configuration for a registry-backed metric group.

    Each group may compute multiple registered metrics. If some metrics fail
    recoverably, successful results are retained and the group completes with
    warnings. If every selected metric fails recoverably, the group is marked
    failed and the evaluation pipeline continues. Invalid configuration and
    unexpected runtime failures remain fatal.
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
    selected: list[str] | None = Field(
        default=None,
        description="Registered metric names or aliases to compute.",
        json_schema_extra={
            "omit_behavior": (
                "Selects all registered metrics unless the concrete group "
                "defines its own default selection."
            ),
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "An empty list is rejected when the metric group runs."
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

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null fields are forwarded as keyword "
                "arguments to `sklearn.decomposition.PCA`."
            ),
        },
    )

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

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null fields are forwarded as keyword "
                "arguments to `scanpy.tl.tsne()`."
            ),
        },
    )

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


class TSNEConfig(EvalStepConfig):
    """Configures the optional Scanpy t-SNE evaluation step.

    t-SNE requires embeddings and the optional Scanpy dependency. It is disabled
    by default and is skipped with a warning if explicitly enabled without
    embeddings.
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
            "null_behavior": "Rejected; a configuration mapping is required.",
        },
    )
    umap: UMAPConfig = Field(
        default_factory=UMAPConfig,
        description="Configuration for the UMAP evaluation step.",
        json_schema_extra={
            "omit_behavior": "Uses `UMAPConfig` defaults.",
            "null_behavior": "Rejected; a configuration mapping is required.",
        },
    )
    tsne: TSNEConfig = Field(
        default_factory=TSNEConfig,
        description="Configuration for the t-SNE evaluation step.",
        json_schema_extra={
            "omit_behavior": "Uses `TSNEConfig` defaults.",
            "null_behavior": "Rejected; a configuration mapping is required.",
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

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null fields are forwarded as keyword "
                "arguments to `sklearn.cluster.KMeans`."
            ),
        },
    )

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
    def validate_kmeans(self) -> "KMeansConfig":
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

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "extra_field_behavior": (
                "Allowed; additional non-null fields are forwarded as keyword "
                "arguments to `sklearn.cluster.HDBSCAN`."
            ),
        },
    )

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
            "null_behavior": "Rejected; a configuration mapping is required.",
        },
    )
    leiden: LeidenConfig = Field(
        default_factory=LeidenConfig,
        description="Configuration for the Leiden evaluation step.",
        json_schema_extra={
            "omit_behavior": "Uses `LeidenConfig` defaults.",
            "null_behavior": "Rejected; a configuration mapping is required.",
        },
    )
    hdbscan: HDBSCANConfig = Field(
        default_factory=HDBSCANConfig,
        description="Configuration for the HDBSCAN evaluation step.",
        json_schema_extra={
            "omit_behavior": "Uses `HDBSCANConfig` defaults.",
            "null_behavior": "Rejected; a configuration mapping is required.",
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
            "null_behavior": "Rejected.",
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
            "null_behavior": "Rejected.",
            "notes": [
                "When requested without usable reconstruction inputs, the "
                "export is disabled with a warning."
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
            "null_behavior": "Rejected.",
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
    pass


class ExternalClusteringMetricConfig(EvalMetricGroupConfig):
    label_key: str = "label"


class EvaluationClusteringMetricsConfig(_EvaluationConfigBaseModel):
    internal: InternalClusteringMetricConfig = Field(default_factory=InternalClusteringMetricConfig)
    external: ExternalClusteringMetricConfig = Field(default_factory=ExternalClusteringMetricConfig)


# Embedding ---
class EmbeddingMetricConfig(EvalMetricGroupConfig):
    selected: list[str] = Field(
        default_factory=lambda: [
            "mean",
            "median",
            "standard_deviation",
            "minimum",
            "maximum",
            "quantiles",
        ]
    )

    @model_validator(mode="after")
    def validate_embedding_metrics(self) -> "EmbeddingMetricConfig":
        if self.enabled is not False and not self.selected:
            raise ValueError(
                "metrics.embedding.selected cannot be empty when "
                "embedding metrics are enabled."
            )

        return self


# Reconstruction ---
class ReconstructionMetricConfig(EvalMetricGroupConfig):
    reduction: Literal["global", "per_channel", "both"] = "global"


# Predictability ---
class EvaluationCrossValidationConfig(_EvaluationConfigBaseModel):
    method: Literal[
        "stratified_kfold",
        "kfold",
        "group_kfold",
        "stratified_group_kfold",
    ] = "stratified_kfold"
    n_splits: NSplits = 5
    group_key: str | None = None
    shuffle: bool = True
    random_state: int | None = 137
    scoring: Literal[
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "accuracy",
        "r2",
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
    ] | None = None

    @model_validator(mode="after")
    def validate_cv(self) -> EvaluationCrossValidationConfig:
        if self.method in ("group_kfold", "stratified_group_kfold") and self.group_key is None:
            raise ValueError(
                "cv.group_key required when using group_kfold or stratified_group_kfold for cv.method."
            )

        if self.method == "group_kfold" and self.shuffle:
            raise ValueError(
                f"cv.shuffle is not supported for cv.method = {self.method}"
            )

        return self


class TuningInnerCVConfig(_EvaluationConfigBaseModel):
    n_splits: NSplits = 3


class EvaluationCVTuningConfig(_EvaluationConfigBaseModel):
    enabled: bool = False
    inner_cv: TuningInnerCVConfig | None = None

    @model_validator(mode="after")
    def validate_tuning(self) -> EvaluationCVTuningConfig:
        if self.enabled and self.inner_cv is None:
            raise ValueError(
                "tuning.inner_cv is required when predictability.tuning.enabled is true."
            )
        return self


class DummyProbeConfig(_EvaluationConfigBaseModel):
    strategy: Literal[
        "most_frequent",
        "stratified",
        "uniform",
        "mean",
        "median",
    ] = "most_frequent"
    random_state: int | None = 137


class LogisticRegressionProbeConfig(BaseModel):
    model: Literal["logistic_regression"] = "logistic_regression"
    standardize: bool = True
    C: PositiveFloatOrList = 1.0
    class_weight: Literal["balanced"] | None = None
    max_iter: PositiveInt = 5000

    model_config = ConfigDict(extra="allow")


class RidgeProbeConfig(BaseModel):
    model: Literal["ridge"] = "ridge"
    standardize: bool = True
    alpha: PositiveFloatOrList = 1.0

    model_config = ConfigDict(extra="allow")


LinearProbeConfig = Annotated[
    LogisticRegressionProbeConfig | RidgeProbeConfig,
    Field(discriminator="model"),
]


class KNNProbeConfig(BaseModel):
    standardize: bool = True
    n_neighbors: PositiveIntOrList = 15
    weights: KNNWeightsOrList = "distance"
    metric: str | list[str] = "euclidean"

    model_config = ConfigDict(extra="allow")


class RandomForestProbeConfig(BaseModel):
    n_estimators: PositiveIntOrList = 500
    max_depth: MaxDepthParam = None
    class_weight: Literal["balanced"] | None = None
    random_state: int | None = 137
    n_jobs: int | None = -1

    model_config = ConfigDict(extra="allow")


class XGBoostProbeConfig(BaseModel):
    n_estimators: PositiveIntOrList = 300
    max_depth: PositiveIntOrList | None = None
    learning_rate: PositiveFloatOrList = 0.01
    random_state: int | None = 137
    n_jobs: int | None = -1

    model_config = ConfigDict(extra="allow")


class SVMRBFProbeConfig(BaseModel):
    standardize: bool = True
    C: PositiveFloatOrList = 1.0
    gamma: SVMRBFGammaValueOrList = "scale"
    class_weight: Literal["balanced"] | None = None
    cache_size: PositiveFloat = 200.0
    max_iter: MaxIterWithNoLimitSentinel = -1

    model_config = ConfigDict(extra="allow")


class EvaluationPredictabilityParamsConfig(_EvaluationConfigBaseModel):
    dummy: DummyProbeConfig = Field(default_factory=DummyProbeConfig)
    linear: LinearProbeConfig = Field(default_factory=LogisticRegressionProbeConfig)
    knn: KNNProbeConfig = Field(default_factory=KNNProbeConfig)
    random_forest: RandomForestProbeConfig = Field(default_factory=RandomForestProbeConfig)
    xgboost: XGBoostProbeConfig = Field(default_factory=XGBoostProbeConfig)
    svm_rbf: SVMRBFProbeConfig = Field(default_factory=SVMRBFProbeConfig)


class EvaluationPredictabilityConfig(EvalStepConfig):
    selected: list[PredictabilityProbeName] = Field(
        default_factory=lambda: ["dummy", "linear", "knn", "random_forest", "svm_rbf"]
    )
    target_key: str = "label"
    task: Literal["classification", "regression"] = "classification"
    cv: EvaluationCrossValidationConfig = Field(default_factory=EvaluationCrossValidationConfig)
    tuning: EvaluationCVTuningConfig = Field(default_factory=EvaluationCVTuningConfig)
    params: EvaluationPredictabilityParamsConfig = Field(default_factory=EvaluationPredictabilityParamsConfig)

    @model_validator(mode="after")
    def validate_predictability_config(self) -> "EvaluationPredictabilityConfig":
        if self.enabled is not True:
            return self

        if len(self.selected) == 0:
            raise ValueError("predictability.selected cannot be an empty list.")

        if self.target_key.strip() == "":
            raise ValueError(
                "predictability.target_key must be a non-empty string when "
                "predictability is enabled."
            )

        if "dummy" in self.selected:
            resolved_task = self.task

            if resolved_task == "regression":
                if self.params.dummy.strategy not in {"mean", "median"}:
                    raise ValueError(
                        "predictability.params.dummy.strategy must be one of "
                        "['mean', 'median'] when predictability.task is 'regression'."
                    )
            else:
                if self.params.dummy.strategy not in {
                    "most_frequent",
                    "stratified",
                    "uniform",
                }:
                    raise ValueError(
                        "predictability.params.dummy.strategy must be one of "
                        "['most_frequent', 'stratified', 'uniform'] when "
                        "predictability.task is 'classification'."
                    )

        return self



# Full evaluation metrics config ---
class EvaluationMetricsConfig(_EvaluationConfigBaseModel):
    clustering: EvaluationClusteringMetricsConfig = Field(default_factory=EvaluationClusteringMetricsConfig)
    embedding: EmbeddingMetricConfig = Field(default_factory=EmbeddingMetricConfig)
    predictability: EvaluationPredictabilityConfig = Field(default_factory=EvaluationPredictabilityConfig)
    reconstruction: ReconstructionMetricConfig = Field(default_factory=ReconstructionMetricConfig)


# -------------------------
# Plots config
# -------------------------
class ReconstructionGridConfig(_EvaluationConfigBaseModel):
    include_error_maps: bool = True
    random_state: int = 137
    stratify_by: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ] | None = None

    channel_selection: (
        Literal["all"]
        | NonNegativeInt
        | Annotated[list[NonNegativeInt], Field(min_length=1)]
        | None
    ) = None

class PlotParams(_EvaluationConfigBaseModel):
    accent_color: HexColor = "#6A3D9A"
    color_by: list[str] | None = None
    dpi: PositiveInt = 300
    formats: list[Literal["png", "pdf", "svg"]] = Field(default_factory=lambda: ["png"])
    reconstruction_grid: ReconstructionGridConfig = Field(
        default_factory=ReconstructionGridConfig
    )


class EvaluationPlotsConfig(EvalStepConfig):
    params: PlotParams | None = Field(default_factory=PlotParams)


# -------------------------
# Full evaluation configuration
# -------------------------
class EvaluationConfig(_EvaluationConfigBaseModel):
    stage: Literal["evaluation"] = "evaluation"
    source: EvaluationSourceConfig = Field(default_factory=EvaluationSourceConfig)
    run: EvaluationRunConfig = Field(default_factory=EvaluationRunConfig)
    reductions: EvaluationReductionsConfig = Field(default_factory=EvaluationReductionsConfig)
    clustering: EvaluationClusteringConfig = Field(default_factory=EvaluationClusteringConfig)
    metrics: EvaluationMetricsConfig = Field(default_factory=EvaluationMetricsConfig)
    reconstruction: EvaluationReconstructionConfig = Field(default_factory=EvaluationReconstructionConfig)
    plots: EvaluationPlotsConfig = Field(default_factory=EvaluationPlotsConfig)

    @model_validator(mode="after")
    def validate_source(self, info: ValidationInfo) -> "EvaluationConfig":
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
