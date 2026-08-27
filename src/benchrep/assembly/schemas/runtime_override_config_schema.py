from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class RuntimeComponentOverrideConfig(BaseModel):
    """Configuration for an externally supplied component."""

    model_config = ConfigDict(extra="forbid")

    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments passed when BenchRep instantiates an externally "
            "supplied component class."
        ),
        json_schema_extra={
            "omit_behavior": "Uses no constructor parameters.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Parameters are passed only to class overrides.",
                "Parameters must be empty when an instantiated object is supplied.",
            ],
        },
    )


class RuntimeOverridesConfig(BaseModel):
    """Records external components required to reconstruct a workflow."""

    model_config = ConfigDict(extra="forbid")

    model: RuntimeComponentOverrideConfig | None = Field(
        default=None,
        description=(
            "When non-null, the workflow requires an externally supplied model."
        ),
        json_schema_extra={
            "omit_behavior": "Does not require an external model.",
            "null_behavior": "Equivalent to omission.",
        },
    )

    datamodule: RuntimeComponentOverrideConfig | None = Field(
        default=None,
        description=(
            "When non-null, the workflow requires an externally supplied "
            "datamodule."
        ),
        json_schema_extra={
            "omit_behavior": "Does not require an external datamodule.",
            "null_behavior": "Equivalent to omission.",
        },
    )