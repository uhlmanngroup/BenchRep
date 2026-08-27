from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class RuntimeComponentOverrideConfig(BaseModel):
    """Records that a workflow requires an externally supplied component."""

    model_config = ConfigDict(extra="forbid")

    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments used when BenchRep instantiates an externally "
            "supplied component class."
        ),
        json_schema_extra={
            "omit_behavior": "Uses no constructor parameters.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Non-empty parameters are reserved for future class overrides.",
                "Current runtime overrides must be instantiated objects.",
            ],
        },
    )

    @field_validator("params")
    @classmethod
    def reject_unsupported_constructor_params(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        if value:
            raise ValueError(
                "External component constructor parameters are not supported "
                "yet because runtime overrides currently require instantiated "
                "objects."
            )

        return value


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