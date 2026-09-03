from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


CompositeModelInputRole: TypeAlias = Literal[
    "sample",
    "positive",
    "negative",
    "prediction_target",
    "condition",
]

CompositeModelInputKind: TypeAlias = Literal[
    "image",
    "categorical",
    "continuous",
]

CompositeModelBatchMetadataRole: TypeAlias = Literal[
    "index",
    "group",
]

CompositeModelOutputRole: TypeAlias = Literal[
    "embedding",
    "reconstruction",
    "projection",
    "prediction",
    "auxiliary",
]

CompositeModelOutputKind: TypeAlias = Literal[
    "image",
    "vector",
    "scalar",
    "categorical",
    "continuous",
]

CompositeModelComponentKind: TypeAlias = Literal[
    "encoder",
    "decoder",
    "head",
]

_COMPOSITE_INPUT_ROLE_KINDS: dict[
    CompositeModelInputRole,
    frozenset[CompositeModelInputKind],
] = {
    "sample": frozenset[CompositeModelInputKind]({"image"}),
    "positive": frozenset[CompositeModelInputKind]({"image"}),
    "negative": frozenset[CompositeModelInputKind]({"image"}),
    "prediction_target": frozenset[CompositeModelInputKind]({
        "categorical",
        "continuous",
    }),
    "condition": frozenset[CompositeModelInputKind]({
        "image",
        "categorical",
        "continuous",
    }),
}


_COMPOSITE_OUTPUT_ROLE_KINDS: dict[
    CompositeModelOutputRole,
    frozenset[CompositeModelOutputKind],
] = {
    "embedding": frozenset[CompositeModelOutputKind]({"vector"}),
    "reconstruction": frozenset[CompositeModelOutputKind]({"image"}),
    "projection": frozenset[CompositeModelOutputKind]({"vector"}),
    "prediction": frozenset[CompositeModelOutputKind]({
        "categorical",
        "continuous",
    }),
    "auxiliary": frozenset[CompositeModelOutputKind]({
        "image",
        "vector",
        "scalar",
        "categorical",
        "continuous",
    }),
}


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
    kind: CompositeModelInputKind

    @model_validator(mode="after")
    def validate_role_kind(self) -> CompositeModelInputConfig:
        allowed_kinds = _COMPOSITE_INPUT_ROLE_KINDS[self.role]

        if self.kind not in allowed_kinds:
            raise ValueError(
                f"`role: {self.role}` requires `kind` to be one of "
                f"{sorted(allowed_kinds)}."
            )

        return self


# Batch metadata declarations
class CompositeModelBatchMetadataConfig(_CompositeModelConfigBaseModel):
    role: CompositeModelBatchMetadataRole


# Produced data declarations
class CompositeModelOutputConfig(_CompositeModelConfigBaseModel):
    role: CompositeModelOutputRole
    kind: CompositeModelOutputKind

    @model_validator(mode="after")
    def validate_role_kind(self) -> CompositeModelOutputConfig:
        allowed_kinds = _COMPOSITE_OUTPUT_ROLE_KINDS[self.role]

        if self.kind not in allowed_kinds:
            raise ValueError(
                f"`role: {self.role}` requires `kind` to be one of "
                f"{sorted(allowed_kinds)}."
            )

        return self


# Top level of declarations
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
