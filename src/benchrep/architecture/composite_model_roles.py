"""Semantic roles for data declared by Composite models."""
from __future__ import annotations

from typing import Literal, TypeAlias


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
CompositeModelTensorRole: TypeAlias = (
    CompositeModelInputRole | CompositeModelOutputRole
)
CompositeModelComponentKind: TypeAlias = Literal[
    "encoder",
    "decoder",
    "head",
]
