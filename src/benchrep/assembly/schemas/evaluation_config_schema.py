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
class EvalStepConfig(BaseModel):
    # None means "auto"; the resolver decides based on available inputs.
    enabled: bool | None = None
    params: dict[str, Any] | None = None


class EvalMetricGroupConfig(BaseModel):
    # None means "auto"; the resolver decides based on available inputs.
    enabled: bool | None = None
    selected: list[str] | None = None
    params: dict[str, dict[str, Any]] | None = None


# -------------------------
# Source and run config
# -------------------------
class EvaluationSourceConfig(BaseModel):
    prediction_manifest_path: Path | None = None
    embeddings_path: Path | None = None
    reconstructions_path: Path | None = None


class EvaluationRunConfig(BaseModel):
    output_root: Path | None = None
    run_name: str | None = None


# -------------------------
# Reductions config
# -------------------------
class PCAParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_components: PositiveInt | None = None
    key_added: str | None = "X_pca"
    random_state: int | None = 137
    overwrite: bool | None = False


class UMAPParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_neighbors: PositiveInt | None = 15
    n_pcs: PositiveInt | None = None
    min_dist: NonNegativeFloat | None = 0.1
    metric: str | None = "euclidean"
    key_added: str | None = "X_umap"
    neighbors_key: str | None = "neighbors"
    random_state: int | None = 137
    overwrite: bool | None = False
    neighbors_kwargs: dict[str, Any] | None = None
    umap_kwargs: dict[str, Any] | None = None


class TSNEParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_pcs: PositiveInt | None = None
    perplexity: PositiveFloat | None = 30.0
    key_added: str | None = "X_tsne"
    random_state: int | None = 137
    overwrite: bool | None = False


class PCAConfig(EvalStepConfig):
    params: PCAParams | None = Field(default_factory=PCAParams)


class UMAPConfig(EvalStepConfig):
    params: UMAPParams | None = Field(default_factory=UMAPParams)


class TSNEConfig(EvalStepConfig):
    params: TSNEParams | None = Field(default_factory=TSNEParams)


class EvaluationReductionsConfig(BaseModel):
    pca: PCAConfig = Field(default_factory=PCAConfig)
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    tsne: TSNEConfig = Field(default_factory=TSNEConfig)


# -------------------------
# Clustering config
# -------------------------
class KMeansParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_clusters: PositiveInt | None = None
    key_added: str | None = "kmeans"
    random_state: int | None = 137
    n_init: str | int | None = "auto"
    overwrite: bool | None = False


class LeidenParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    resolution: PositiveFloat | None = 1.0
    n_neighbors: PositiveInt | None = 15
    n_pcs: PositiveInt | None = None
    metric: str | None = "euclidean"
    key_added: str | None = "leiden"
    neighbors_key: str | None = "neighbors"
    random_state: int | None = 137
    overwrite: bool | None = False
    neighbors_kwargs: dict[str, Any] | None = None
    leiden_kwargs: dict[str, Any] | None = None


class KMeansConfig(EvalStepConfig):
    params: KMeansParams | None = Field(default_factory=KMeansParams)

    @model_validator(mode="after")
    def validate_kmeans(self) -> "KMeansConfig":
        if self.enabled is True and (
                self.params is None or self.params.n_clusters is None
        ):
            raise ValueError(
                "clustering.kmeans.params.n_clusters is required when KMeans is enabled."
            )
        return self


class LeidenConfig(EvalStepConfig):
    params: LeidenParams | None = Field(default_factory=LeidenParams)


class EvaluationClusteringConfig(BaseModel):
    kmeans: KMeansConfig = Field(default_factory=KMeansConfig)
    leiden: LeidenConfig = Field(default_factory=LeidenConfig)


# -------------------------
# Reconstruction artifacts config
# -------------------------
class ErrorMapParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kinds: list[ErrorMapKind] = Field(
        default_factory=lambda: ["absolute", "signed", "relative"],
        min_length=1,
    )
    denominator_floor: PositiveFloat | None = None


class EvaluationReconstructionConfig(BaseModel):
    export_tiffs: bool = False
    n_examples: PositiveInt | None = None
    error_maps: ErrorMapParams = Field(default_factory=ErrorMapParams)


# -------------------------
# Metrics config
# -------------------------

# Clustering ---
class InternalClusteringMetricConfig(EvalMetricGroupConfig):
    pass


class ExternalClusteringMetricConfig(EvalMetricGroupConfig):
    label_key: str = "label"


class EvaluationClusteringMetricsConfig(BaseModel):
    internal: InternalClusteringMetricConfig = Field(default_factory=InternalClusteringMetricConfig)
    external: ExternalClusteringMetricConfig = Field(default_factory=ExternalClusteringMetricConfig)


# Reconstruction ---
class ReconstructionMetricConfig(EvalMetricGroupConfig):
    reduction: Literal["global", "per_channel", "both"] = "global"


# Predictability ---
class EvaluationCrossValidationConfig(BaseModel):
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


class TuningInnerCVConfig(BaseModel):
    n_splits: NSplits = 3


class EvaluationCVTuningConfig(BaseModel):
    enabled: bool = False
    inner_cv: TuningInnerCVConfig | None = None

    @model_validator(mode="after")
    def validate_tuning(self) -> EvaluationCVTuningConfig:
        if self.enabled and self.inner_cv is None:
            raise ValueError(
                "tuning.inner_cv is required when predictability.tuning.enabled is true."
            )
        return self


class DummyProbeConfig(BaseModel):
    strategy: Literal["most_frequent", "stratified", "uniform", "mean", "median"] = "most_frequent"


class LogisticRegressionProbeConfig(BaseModel):
    model: Literal["logistic_regression"] = "logistic_regression"
    standardize: bool = True
    C: PositiveFloatOrList = 1.0
    class_weight: Literal["balanced"] | None = None
    max_iter: PositiveInt = 5000


class RidgeProbeConfig(BaseModel):
    model: Literal["ridge"] = "ridge"
    standardize: bool = True
    alpha: PositiveFloatOrList = 1.0


LinearProbeConfig = Annotated[
    LogisticRegressionProbeConfig | RidgeProbeConfig,
    Field(discriminator="model"),
]


class KNNProbeConfig(BaseModel):
    standardize: bool = True
    n_neighbors: PositiveIntOrList = 15
    weights: KNNWeightsOrList = "distance"
    metric: str | list[str] = "euclidean"


class RandomForestProbeConfig(BaseModel):
    n_estimators: PositiveIntOrList = 500
    max_depth: MaxDepthParam = None
    class_weight: Literal["balanced"] | None = None
    random_state: int | None = 137
    n_jobs: int | None = -1


class XGBoostProbeConfig(BaseModel):
    n_estimators: PositiveIntOrList = 300
    max_depth: PositiveIntOrList | None = None
    learning_rate: PositiveFloatOrList = 0.01
    random_state: int | None = 137
    n_jobs: int | None = -1


class SVMRBFProbeConfig(BaseModel):
    standardize: bool = True
    C: PositiveFloatOrList = 1.0
    gamma: SVMRBFGammaValueOrList = "scale"
    class_weight: Literal["balanced"] | None = None
    cache_size: PositiveFloat = 200.0
    max_iter: MaxIterWithNoLimitSentinel = -1


class EvaluationPredictabilityParamsConfig(BaseModel):
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
class EvaluationMetricsConfig(BaseModel):
    clustering: EvaluationClusteringMetricsConfig = Field(default_factory=EvaluationClusteringMetricsConfig)
    embedding: EvalMetricGroupConfig = Field(default_factory=EvalMetricGroupConfig)
    predictability: EvaluationPredictabilityConfig = Field(default_factory=EvaluationPredictabilityConfig)
    reconstruction: ReconstructionMetricConfig = Field(default_factory=ReconstructionMetricConfig)


# -------------------------
# Plots config
# -------------------------
class ReconstructionGridConfig(BaseModel):
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

class PlotParams(BaseModel):
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
class EvaluationConfig(BaseModel):
    stage: Literal["evaluation"] = "evaluation"
    source: EvaluationSourceConfig = Field(default_factory=EvaluationSourceConfig)
    run: EvaluationRunConfig = Field(default_factory=EvaluationRunConfig)
    reductions: EvaluationReductionsConfig = Field(default_factory=EvaluationReductionsConfig)
    clustering: EvaluationClusteringConfig = Field(default_factory=EvaluationClusteringConfig)
    metrics: EvaluationMetricsConfig = Field(default_factory=EvaluationMetricsConfig)
    reconstruction: EvaluationReconstructionConfig = Field(default_factory=EvaluationReconstructionConfig)
    plots: EvaluationPlotsConfig = Field(default_factory=EvaluationPlotsConfig)

    @model_validator(mode="after")
    def validate_source(self) -> "EvaluationConfig":
        if (
            self.source.embeddings_path is None
            and self.source.prediction_manifest_path is None
        ):
            raise ValueError(
                "Either source.embeddings_path or source.prediction_manifest_path "
                "must be provided. If embeddings_path is null, embeddings are "
                "inferred from the prediction manifest."
            )

        if (self.source.embeddings_path is not None
              and self.source.embeddings_path.suffix.lower() != ".h5ad"
        ):
            raise ValueError("source.embeddings_path must point to an AnnData .h5ad file.")

        if self.source.prediction_manifest_path is not None:
            if self.source.prediction_manifest_path.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError("source.prediction_manifest_path must point to a YAML file.")

        return self
