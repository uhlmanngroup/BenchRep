from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


CompositeModelInputRole: TypeAlias = Literal[
    "sample_image",
    "positive_image",
    "negative_image",

    "categorical_prediction_target_scalar",
    "categorical_prediction_target_vector",
    "continuous_prediction_target_scalar",
    "continuous_prediction_target_vector",

    "condition_image",
    "categorical_condition_scalar",
    "categorical_condition_vector",
    "continuous_condition_scalar",
    "continuous_condition_vector",
]

CompositeModelBatchMetadataRole: TypeAlias = Literal[
    "index",
    "group",
]

CompositeModelOutputRole: TypeAlias = Literal[
    "embedding_vector",
    "projection_vector",
    "reconstruction_image",

    "categorical_prediction_scalar",
    "categorical_prediction_vector",
    "continuous_prediction_scalar",
    "continuous_prediction_vector",

    "auxiliary_image",
    "categorical_auxiliary_scalar",
    "categorical_auxiliary_vector",
    "continuous_auxiliary_scalar",
    "continuous_auxiliary_vector",
]

CompositeModelComponentKind: TypeAlias = Literal[
    "encoder",
    "decoder",
    "head",
]


# -------------------------
# Generic reusable blocks
# -------------------------
class _CompositeModelConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompositeModelNamedConfig(_CompositeModelConfigBaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Composite model declarations
# -------------------------
# Expected data declarations
class CompositeModelInputConfig(_CompositeModelConfigBaseModel):
    role: CompositeModelInputRole


# Batch metadata declarations
class CompositeModelBatchMetadataConfig(_CompositeModelConfigBaseModel):
    role: CompositeModelBatchMetadataRole


# Produced data declarations
class CompositeModelOutputConfig(_CompositeModelConfigBaseModel):
    role: CompositeModelOutputRole


class CompositeModelDeclarationsConfig(_CompositeModelConfigBaseModel):
    expects: dict[str, CompositeModelInputConfig] = Field(
        min_length=1,
    )
    batch_metadata: dict[str, CompositeModelBatchMetadataConfig] | None = None
    produces: dict[str, CompositeModelOutputConfig] = Field(
        min_length=1,
    )

# -------------------------
# Component configuration
# -------------------------
class CompositeModelComponentConfig(CompositeModelNamedConfig):
    kind: CompositeModelComponentKind


# -------------------------
# Assembly configuration
# -------------------------
class CompositeModelAssemblyStepConfig(_CompositeModelConfigBaseModel):
    component: str

    inputs: dict[str, str] = Field(
        min_length=1,
    )

    outputs: dict[str, str] = Field(
        min_length=1,
    )
