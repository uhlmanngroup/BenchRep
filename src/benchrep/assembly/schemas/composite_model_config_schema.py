from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchrep.architecture.composite_model_roles import (
    CompositeModelInputRole,
    CompositeModelBatchMetadataRole,
    CompositeModelOutputRole,
    CompositeModelComponentKind,
)


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
