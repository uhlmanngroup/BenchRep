from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class RuntimeComponentOverrideConfig(BaseModel):
    """Declares constructor arguments for an externally supplied component.

    This configuration does not contain or instantiate the Python component.
    Supply the actual model or datamodule through the workflow's
    `TrainingOverrides` or `PredictionOverrides` object.

    When the supplied component is a class, `params` provides its constructor
    keyword arguments. When it is an already constructed instance, `params`
    must be empty.
    """

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
    """Declares external model and datamodule requirements in configuration.

    A non-null component entry declares that the workflow requires that
    component to be supplied through Python. An empty mapping declares the
    requirement without supplying constructor arguments.

    These declarations allow schema validation without requiring the
    corresponding config-built component sections. Workflow resolution
    subsequently checks that the required external components were actually
    supplied.

    This schema is distinct from the `TrainingOverrides` and
    `PredictionOverrides` objects passed to workflow entrypoints. Those
    objects carry actual Python components and configuration overrides.

    Whole-model overrides are supported only by canonical autoencoder and
    VAE workflows. Composite models are assembled from configuration.
    """

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