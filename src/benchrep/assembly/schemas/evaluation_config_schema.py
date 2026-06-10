from __future__ import annotations

from pathlib import Path
from typing import Literal, Any

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
    ConfigDict,
)


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
    kind: Literal["absolute", "squared"] | None = "absolute"
    n_examples: PositiveInt | None = None


class ErrorMapConfig(EvalStepConfig):
    params: ErrorMapParams | None = Field(default_factory=ErrorMapParams)


class EvaluationReconstructionConfig(BaseModel):
    error_maps: ErrorMapConfig = Field(default_factory=ErrorMapConfig)


# -------------------------
# Metrics config
# -------------------------
class InternalClusteringMetricConfig(EvalMetricGroupConfig):
    pass


class ExternalClusteringMetricConfig(EvalMetricGroupConfig):
    label_key: str = "label"


class EvaluationClusteringMetricsConfig(BaseModel):
    internal: InternalClusteringMetricConfig = Field(default_factory=InternalClusteringMetricConfig)
    external: ExternalClusteringMetricConfig = Field(default_factory=ExternalClusteringMetricConfig)


class EvaluationMetricsConfig(BaseModel):
    clustering: EvaluationClusteringMetricsConfig = Field(default_factory=EvaluationClusteringMetricsConfig)
    embedding: EvalMetricGroupConfig = Field(default_factory=EvalMetricGroupConfig)
    reconstruction: EvalMetricGroupConfig = Field(default_factory=EvalMetricGroupConfig)


# -------------------------
# Plots config
# -------------------------
class PlotParams(BaseModel):
    color_by: list[str] | None = None


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