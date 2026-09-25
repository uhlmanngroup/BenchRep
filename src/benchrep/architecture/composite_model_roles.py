"""Semantic roles for data declared by Composite models."""
from __future__ import annotations

from typing import Final, Literal, TypeAlias


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
TensorStructure: TypeAlias = Literal[
    "scalar",
    "vector",
    "image",
]
TENSOR_STRUCTURE_BY_ROLE: Final[
    dict[CompositeModelTensorRole, TensorStructure]
] = {
    "sample_image": "image",
    "positive_image": "image",
    "negative_image": "image",
    "categorical_prediction_target_scalar": "scalar",
    "categorical_prediction_target_vector": "vector",
    "continuous_prediction_target_scalar": "scalar",
    "continuous_prediction_target_vector": "vector",
    "condition_image": "image",
    "categorical_condition_scalar": "scalar",
    "categorical_condition_vector": "vector",
    "continuous_condition_scalar": "scalar",
    "continuous_condition_vector": "vector",
    "embedding_vector": "vector",
    "projection_vector": "vector",
    "reconstruction_image": "image",
    "categorical_prediction_scalar": "scalar",
    "categorical_prediction_vector": "vector",
    "continuous_prediction_scalar": "scalar",
    "continuous_prediction_vector": "vector",
    "auxiliary_image": "image",
    "categorical_auxiliary_scalar": "scalar",
    "categorical_auxiliary_vector": "vector",
    "continuous_auxiliary_scalar": "scalar",
    "continuous_auxiliary_vector": "vector",

}
