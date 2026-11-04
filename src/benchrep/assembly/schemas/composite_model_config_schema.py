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
    """Selects a registered component and supplies its constructor arguments."""

    name: str = Field(
        description="Registered component name or alias.",
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "The component's `kind` selects the registry used for lookup.",
            ],
        },
    )

    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments passed to the registered component constructor."
        ),
        json_schema_extra={
            "omit_behavior": "Uses an empty parameter mapping.",
            "null_behavior": "Not allowed; use an empty mapping instead.",
            "notes": [
                "All required constructor arguments must be supplied.",
                "Unlike canonical decoder assembly, Composite assembly does "
                "not automatically supply dimensions from upstream components.",
                "Constructor arguments are separate from runtime tensor "
                "bindings configured in assembly steps.",
            ],
        },
    )


# -------------------------
# Composite model declarations
# -------------------------
class CompositeModelDeclarationsConfig(_CompositeModelConfigBaseModel):
    """Declares named model inputs, contextual metadata, and model outputs.

    Each mapping associates user-defined names with semantic roles. Roles
    determine the expected data structure and inform graph compatibility,
    transform routing, loss wiring, and prediction export.

    Declarations describe the data contract; they do not create tensors or
    architecture components. Assembly steps generate the declared outputs.
    """

    expects: dict[str, CompositeModelInputRole] = Field(
        min_length=1,
        description=(
            "Nonempty mapping from model-input field names to semantic roles."
        ),
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "At least one input must have role `sample_image`.",
                "Inputs must be available in the batch passed to the model, "
                "after any configured dataset transforms.",
                "Assembly steps and ordinary losses reference these fields "
                "using `expects.<name>`.",
                "Multiple inputs may share a role; features relying on a "
                "unique input may then require explicit field selection.",
            ],
        },
    )

    batch_metadata: dict[str, CompositeModelBatchMetadataRole] | None = Field(
        default=None,
        description=(
            "Mapping from contextual batch-field names to `index` or `group` roles."
        ),
        json_schema_extra={
            "omit_behavior": "Declares no batch metadata.",
            "null_behavior": "Equivalent to omission.",
            "notes": [
                "An empty mapping also declares no batch metadata.",
                "At most one field may have role `index`.",
                "For AnnData export, index metadata supplies observation "
                "identifiers and group metadata supplies observation columns.",
                "These declarations are not valid sources for assembly "
                "inputs or ordinary loss wiring; those use `expects` "
                "and `produces` references.",
            ],
        },
    )

    produces: dict[str, CompositeModelOutputRole] = Field(
        min_length=1,
        description=(
            "Nonempty mapping from model-output names to semantic roles."
        ),
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Every declared output must be bound exactly once by the "
                "assembly graph.",
                "Outputs may be consumed by downstream assembly steps and "
                "ordinary losses through `produces.<name>` references.",
                "Intermediate graph values must also be declared here when "
                "they are passed between assembly steps.",
                "Output roles determine eligibility for AnnData and "
                "reconstruction exports.",
            ],
        },
    )

# -------------------------
# Component configuration
# -------------------------
class CompositeModelComponentConfig(CompositeModelNamedConfig):
    """Configures one reusable architecture component.

    The enclosing `composite_model_components` mapping supplies its component
    ID. Each ID is instantiated once and may be invoked by multiple assembly
    steps, sharing the same module and parameters.

    Compatibility is defined by the registered `ArchitectureComponent`
    runtime contract. Composite encoders and decoders do not need to subclass
    the canonical `BaseEncoder` or `BaseDecoder` interfaces.

    Every configured component must be used by at least one assembly step.
    User registrations must be repeated in each process that reconstructs
    the model.
    """

    kind: CompositeModelComponentKind = Field(
        description=(
            "Registry containing the component: `encoder`, `decoder`, or `head`."
        ),
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Use `benchrep.inspect_registry(kind)` to list components.",
                "Pass the component name as the second argument to inspect "
                "its constructor and Composite runtime contract.",
                "Registry kind alone does not establish compatibility; "
                "the registered runtime contract must support the wiring.",
            ],
        },
    )


# -------------------------
# Assembly configuration
# -------------------------
class CompositeModelAssemblyStepConfig(_CompositeModelConfigBaseModel):
    """Invokes one configured component with explicitly wired runtime inputs.

    The enclosing `composite_model_assembly` mapping supplies the step ID.
    This identifies an invocation, whereas `component` identifies the reusable
    module being invoked.

    Execution order follows output dependencies. Mapping order is preserved
    among steps at the same dependency level. Dependency cycles are rejected.

    Input and output bindings must match the selected component's registered
    runtime contract. Tensor shapes and other runtime requirements are also
    checked when the model executes.
    """

    component: str = Field(
        description=(
            "Existing component ID from `composite_model_components`."
        ),
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Use the configured component ID, not its registry name.",
                "Multiple assembly steps may reference the same component.",
            ],
        },
    )

    inputs: dict[str, str] = Field(
        min_length=1,
        description=(
            "Mapping from component runtime-input parameter names to "
            "`expects.<name>` or `produces.<name>` references."
        ),
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "Keys must exactly match the component's registered "
                "runtime-input contract.",
                "Each referenced declaration must have a semantic role "
                "accepted by that runtime input.",
                "A `produces` reference creates a dependency on the step "
                "that generates that output.",
                "Constructor arguments belong in the component's `params`, "
                "not in this mapping.",
            ],
        },
    )

    outputs: str | dict[str, str] = Field(
        min_length=1,
        description=(
            "Bindings from the component's runtime result to declared "
            "`produces.<name>` outputs."
        ),
        json_schema_extra={
            "omit_behavior": "Required; omission raises a validation error.",
            "null_behavior": "Not allowed.",
            "notes": [
                "A component returning one tensor requires a single "
                "`produces.<name>` string.",
                "A component returning a mapping requires a mapping from "
                "runtime-result keys to `produces.<name>` references.",
                "Mapping keys must exactly match the registered result "
                "contract; individual result entries cannot be omitted.",
                "Each destination declaration must have a tensor structure "
                "supported by the corresponding runtime result.",
                "Each declared model output may be bound only once "
                "across the complete assembly graph.",
            ],
        },
    )
