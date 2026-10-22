from __future__ import annotations

from typing import Any

import lightning as L
from pydantic import BaseModel, Field, TypeAdapter

from benchrep.architecture.composite_model_roles import TENSOR_STRUCTURE_BY_ROLE
from benchrep.assembly.config import (
    compose_effective_config,
    normalize_config_components,
)
from benchrep.assembly.registries.builtins import register_builtins
from benchrep.assembly.registries.core import (
    CLASSIFICATION_LOSSES,
    CONTRASTIVE_LOSSES,
    CUSTOM_OBJECTIVE_LOSSES,
    DATASETS,
    DECODERS,
    ENCODERS,
    HEADS,
    LOSS_WEIGHT_SCHEDULES,
    MODELS,
    RECONSTRUCTION_LOSSES,
    REGRESSION_LOSSES,
    REGULARIZATION_LOSSES,
    TRANSFORMS,
)
from benchrep.assembly.registries.utils import normalize_name
from benchrep.assembly.resolvers.training_config_resolver import (
    resolve_training_config,
)
from benchrep.assembly.resolvers.utils import (
    get_component_override_name,
    resolve_component_source,
)
from benchrep.assembly.schemas.composite_model_config_schema import (
    CompositeModelAssemblyStepConfig,
    CompositeModelComponentConfig,
    CompositeModelDeclarationsConfig,
)
from benchrep.assembly.schemas.parsing import parse_training_config
from benchrep.assembly.schemas.training_config_schema import (
    DatasetConfig,
    StrictPositiveInt,
    TrainingConfig,
    TrainingDataModuleConfig,
    TrainingDecoderConfig,
    TrainingEncoderConfig,
    TrainingLossTermConfig,
    TrainingModelConfig,
    TrainingReproducibilityConfig,
    TrainingRunConfig,
    TrainingTransformPipelineConfig,
    _dataset_config_discriminator,
    _loss_weight_schedule_discriminator,
    _normalize_loss_weight_schedule_config,
)
from benchrep.gui.shared.fields import FieldPresentation
from benchrep.gui.shared.forms import SectionSpec
from benchrep.gui.shared.state import (
    WizardDraft,
    editor_spec,
    union_variant_details,
)
from benchrep.interfaces.model_families import (
    COMPOSITE_FAMILY,
    CanonicalModelFamilySpec,
    ModelFamilySpec,
)
from benchrep.utils.workflow_overrides import TrainingOverrides
from benchrep.utils.workflow_utils import TrainingConfigInput


class TrainingModelEditor(BaseModel):
    """UI representation of model family and canonical VAE parameters."""

    name: str = Field(
        description=(
            "Model family selected when launching the wizard. "
            "Launch a new wizard session to select a different family."
        ),
        json_schema_extra={
            "config_path": "model.name",
            "schema_path": "TrainingModelConfig.name",
        },
    )

    latent_dim: StrictPositiveInt = Field(
        description="Number of dimensions in the canonical VAE latent space.",
        json_schema_extra={
            "config_path": "model.params.latent_dim",
            "schema_path": "TrainingModelConfig.params",
            "omit_behavior": "Required for the canonical VAE.",
            "null_behavior": "Not allowed.",
        },
    )


def prepare_training_draft(
    *,
    model_family: ModelFamilySpec,
    config: TrainingConfigInput,
    overrides: TrainingOverrides,
) -> WizardDraft:
    """Prepare wizard values; validate a supplied base before displaying it."""
    register_builtins()
    overrides.validate_for(model_family)

    model_source = (
        resolve_component_source(
            overrides.model,
            expected_base_class=model_family.model_base_class,
            component_name="model",
        )
        if isinstance(model_family, CanonicalModelFamilySpec)
        else "config"
    )
    datamodule_source = resolve_component_source(
        overrides.datamodule,
        expected_base_class=L.LightningDataModule,
        component_name="datamodule",
    )

    components = normalize_config_components(
        schema=TrainingConfig,
        config_components=overrides.config_components,
    )

    if config is not None:
        composition = compose_effective_config(
            schema=TrainingConfig,
            config=config,
            config_components=overrides.config_components,
            external_model=overrides.model is not None,
            external_datamodule=overrides.datamodule is not None,
        )

        # Revalidate typed inputs too: they may have been constructed
        # under a different runtime-override context.
        effective = parse_training_config(
            composition.effective_config.model_dump(mode="python"),
            model_overridden=overrides.model is not None,
            datamodule_overridden=overrides.datamodule is not None,
        )

        # Reuse workflow checks, including model-family compatibility.
        # This resolves configuration; it does not start training.
        resolve_training_config(
            effective,
            model_family=model_family,
            model_source=model_source,
            datamodule_source=datamodule_source,
            model_override_name=(
                get_component_override_name(overrides.model)
                if overrides.model is not None
                else None
            ),
        )

        # Validation uses the effective model; editing starts from the
        # supplied payload so defaults and normalization do not rewrite it.
        source = (
            config.model_dump(mode="python", exclude_unset=True)
            if isinstance(config, BaseModel)
            else composition.original_config_raw
        )
        data = TypeAdapter(dict[str, Any]).dump_python(
            source,
            mode="json",
        )
        data.update(
            TypeAdapter(dict[str, Any]).dump_python(
                components,
                mode="json",
            )
        )
    else:
        # A new draft is intentionally incomplete.
        data = TypeAdapter(dict[str, Any]).dump_python(
            components,
            mode="json",
        )
        data.setdefault("stage", "training")

        if overrides.model is None:
            data.setdefault("model", {"name": model_family.name})
            configured_name = MODELS.resolve_key(
                normalize_name(
                    data["model"]["name"],
                    field_name="model.name",
                )
            )
            if configured_name != model_family.name:
                raise ValueError(
                    "The supplied model config component conflicts with "
                    f"the selected family: {configured_name!r} versus "
                    f"{model_family.name!r}."
                )

    # These are the sections discarded by the training resolver when
    # the corresponding runtime object is supplied.
    omitted_sections: set[str] = set()
    if overrides.model is not None:
        omitted_sections.update(
            {
                "model",
                "encoder",
                "decoder",
                "composite_model_declarations",
                "composite_model_components",
                "composite_model_assembly",
                "losses",
                "optimizer",
            }
        )
    if overrides.datamodule is not None:
        omitted_sections.update(
            {
                "dataset",
                "transform_pipelines",
                "datamodule",
            }
        )

    for section in omitted_sections:
        data.pop(section, None)

    return WizardDraft(
        initial_data=data,
        supplied_sections=frozenset(components) - omitted_sections,
    )


_ARCHITECTURE_SECTIONS = {
    "encoder": (TrainingEncoderConfig, ENCODERS),
    "decoder": (TrainingDecoderConfig, DECODERS),
}


def training_architecture_schemas(
    *,
    model_family: ModelFamilySpec,
    overrides: TrainingOverrides,
) -> dict[str, type[BaseModel]]:
    """Canonical architectures use encoder/decoder unless model is external."""
    if overrides.model is not None or not isinstance(
        model_family, CanonicalModelFamilySpec
    ):
        return {}

    return {key: schema for key, (schema, _) in _ARCHITECTURE_SECTIONS.items()}


def architecture_name_options(
    *,
    section: str,
    current_name: str | None,
) -> list[str]:
    """List registered names while preserving an existing spelling or alias."""
    _, registry = _ARCHITECTURE_SECTIONS[section]
    names = list(registry.canonical_keys())

    if current_name is not None and current_name not in names:
        names.append(current_name)

    return sorted(names)


class TrainingModelNameEditor(BaseModel):
    """Fixed model-family name for canonical AE and Composite workflows."""

    name: str


def training_sections(
    *, model_family: ModelFamilySpec, overrides: TrainingOverrides, initial: WizardDraft
) -> list[SectionSpec]:
    """Declare the implemented sections; callbacks and scalar editors are shared."""

    def standard(
        key: str,
        title: str,
        description: str,
        options=None,
        refresh_on_change=None,
        presentation=None,
        validate_local=None,
    ):
        info = TrainingConfig.model_fields[key]
        return SectionSpec(
            key=key,
            title=title,
            description=description,
            annotation=info.rebuild_annotation(),
            field_info=info,
            options=options,
            refresh_on_change=refresh_on_change,
            presentation=presentation,
            validate_local=validate_local,
            documentation_schema={
                "run": TrainingRunConfig,
                "reproducibility": TrainingReproducibilityConfig,
                "datamodule": TrainingDataModuleConfig,
                "dataset": DatasetConfig,
                "transform_pipelines": TrainingTransformPipelineConfig,
                "encoder": TrainingEncoderConfig,
                "decoder": TrainingDecoderConfig,
                "losses": TrainingLossTermConfig,
                "composite_model_declarations": CompositeModelDeclarationsConfig,
                "composite_model_components": CompositeModelComponentConfig,
                "composite_model_assembly": CompositeModelAssemblyStepConfig,
            }[key],
        )

    sections = [
        standard(
            "run", "Run", "Choose where outputs are saved and an optional project name."
        ),
        standard(
            "reproducibility",
            "Reproducibility",
            "Configure random seeds and numerical precision.",
        ),
    ]
    if overrides.model is None:
        model_data = initial.initial_data.get("model") or {"name": model_family.name}
        family_name = model_data.get("name", model_family.name)
        is_vae = model_family.name == "vae"

        def to_editor(data):
            if data is None:
                return None
            values = {"name": family_name}
            params = data.get("params") or {}
            if is_vae and "latent_dim" in params:
                values["latent_dim"] = params["latent_dim"]
            return values

        def from_editor(values):
            if values is None:
                return None, {}
            data = {"name": family_name}
            params = (
                {"latent_dim": values["latent_dim"]}
                if is_vae and "latent_dim" in values
                else {}
            )
            if params or "params" in model_data:
                data["params"] = params
            return data, {}

        editor = TrainingModelEditor if is_vae else TrainingModelNameEditor
        sections.append(
            SectionSpec(
                key="model",
                title="Model",
                annotation=editor | None,
                field_info=TrainingConfig.model_fields["model"],
                description="The model family is fixed by the launcher."
                + (" Set the VAE latent dimension." if is_vae else ""),
                documentation_schema=TrainingModelConfig,
                locked_paths=frozenset({("model", "name")}),
                to_editor=to_editor,
                from_editor=from_editor,
                error_path=lambda path: (
                    ("params", "latent_dim", *path[1:])
                    if path and path[0] == "latent_dim"
                    else path
                ),
            )
        )
    for key in training_architecture_schemas(
        model_family=model_family, overrides=overrides
    ):

        def choices(path, value, section_data, section=key):
            return (
                architecture_name_options(section=section, current_name=value)
                if path == (section, "name")
                else None
            )

        sections.append(
            standard(
                key,
                key.title(),
                f"Choose a registered {key} and its constructor parameters.",
                choices,
            )
        )
    if model_family == COMPOSITE_FAMILY and overrides.model is None:
        sections.extend(
            [
                standard(
                    "composite_model_declarations",
                    "Declarations",
                    "Name the model's inputs, batch metadata and outputs, and select "
                    "their semantic roles. These names are used in assembly references.",
                ),
                standard(
                    "composite_model_components",
                    "Components",
                    "Give each reusable component an ID. Select its kind (encoder, "
                    "decoder or head) to narrow the registered name choices, then "
                    "provide its constructor parameters. Changing kind preserves "
                    "your existing name and parameters for review.",
                    options=composite_component_name_options,
                    presentation=composite_presentation,
                    validate_local=validate_component_names,
                    refresh_on_change=lambda path: (
                        len(path) == 3 and path[-1] == "kind"
                    ),
                ),
                standard(
                    "composite_model_assembly",
                    "Assembly",
                    "Give each step an ID and reference a component ID. Bind runtime "
                    "input names to expects.<name> or produces.<name>. For outputs, "
                    "use one produces.<name> string for a tensor, or a mapping from "
                    "runtime output names to produces.<name>. Wiring and component "
                    "compatibility are checked during final validation.",
                    presentation=composite_presentation,
                ),
            ]
        )
    if overrides.model is None:
        sections.append(
            standard(
                "losses",
                "Losses",
                "Group weighted loss terms by role. Each term selects a registered loss, "
                "constructor parameters and an optional weight schedule. "
                + (
                    "Ordinary Composite losses wire their runtime inputs to declared "
                    "expects/produces references; custom objectives receive full context automatically."
                    if model_family == COMPOSITE_FAMILY
                    else "Canonical models supply loss inputs automatically; no Composite wiring is needed."
                ),
                options=loss_name_options,
                validate_local=validate_loss_names,
                presentation=lambda path, value, draft: loss_presentation(
                    path, value, draft, composite=model_family == COMPOSITE_FAMILY
                ),
                refresh_on_change=lambda path: (
                    len(path) == 2 or (len(path) == 4 and path[-1] == "name")
                ),
            )
        )
    if overrides.datamodule is None:
        sections.extend(
            [
                standard(
                    "dataset",
                    "Dataset",
                    "Choose a registered dataset. Built-ins expose their typed parameters; "
                    "custom datasets accept constructor parameters as a JSON object.",
                    presentation=dataset_presentation,
                    validate_local=validate_dataset_name,
                ),
                standard(
                    "transform_pipelines",
                    "Transform pipelines",
                    "Pipelines run in list order, and steps run in order within each pipeline. "
                    "Choose which splits each step applies to. Routes use sample field names; "
                    "route compatibility is checked during final validation.",
                    options=transform_name_options,
                    presentation=lambda path, value, draft: transform_presentation(
                        path, value, draft, composite=model_family == COMPOSITE_FAMILY
                    ),
                    validate_local=validate_transform_names,
                ),
            ]
        )
        sections.append(
            standard(
                "datamodule",
                "Datamodule",
                "Configure batching, validation splitting, and data loading.",
            )
        )
    return sections


_COMPOSITE_COMPONENT_REGISTRIES = {
    "encoder": ENCODERS,
    "decoder": DECODERS,
    "head": HEADS,
}


def composite_component_name_options(
    path: tuple, current_name: Any, section_data: Any
) -> list[str] | None:
    """Offer registry names using the kind of this component, without rewriting input."""
    if len(path) != 3 or path[0] != "composite_model_components" or path[-1] != "name":
        return None
    component = section_data.get(path[1], {}) if isinstance(section_data, dict) else {}
    kind = component.get("kind") if isinstance(component, dict) else None
    registry = _COMPOSITE_COMPONENT_REGISTRIES.get(kind)
    registries = (registry,) if registry is not None else ()
    names = {name for item in registries for name in item.canonical_keys()}
    # Preserve loaded aliases and previous names for local validation.
    if isinstance(current_name, str):
        names.add(current_name)
    return sorted(names)


_LOSS_REGISTRIES = {
    "reconstruction": RECONSTRUCTION_LOSSES,
    "regularization": REGULARIZATION_LOSSES,
    "contrastive": CONTRASTIVE_LOSSES,
    "classification": CLASSIFICATION_LOSSES,
    "regression": REGRESSION_LOSSES,
    "custom_objective": CUSTOM_OBJECTIVE_LOSSES,
}


def _at(data: Any, path: tuple, default=None) -> Any:
    """Read a draft path without requiring intermediate values to be valid."""
    for key in path:
        if (
            isinstance(data, dict)
            and key in data
            or isinstance(data, list)
            and isinstance(key, int)
            and 0 <= key < len(data)
        ):
            data = data[key]
        else:
            return default
    return data


def _registered_entry(registry, name):
    if registry is None or not isinstance(name, str):
        return None
    try:
        return registry.get(registry.resolve_key(name))
    except (KeyError, ValueError, TypeError):
        return None


def _reference_suggestions(draft: dict, *, outputs_only=False) -> tuple[str, ...]:
    groups = ("produces",) if outputs_only else ("expects", "produces")
    references = []
    for group in groups:
        names = _at(draft, ("composite_model_declarations", group), {})
        if isinstance(names, dict):
            references.extend(
                f"{group}.{name}" for name in names if isinstance(name, str)
            )
    return tuple(references)


def validate_component_names(data: Any) -> dict[tuple, str]:
    """Check only the same-section kind/name dropdown relationship."""
    errors = {}
    if not isinstance(data, dict):
        return errors
    for key, component in data.items():
        if not isinstance(component, dict):
            continue
        kind, name = component.get("kind"), component.get("name")
        registry = (
            _COMPOSITE_COMPONENT_REGISTRIES.get(kind) if isinstance(kind, str) else None
        )
        if registry is not None and isinstance(name, str):
            error = _registry_name_error(registry, name, kind)
            if error:
                errors[(key, "name")] = error
    return errors


def composite_presentation(path: tuple, value: Any, draft: dict) -> FieldPresentation:
    if path[0] == "composite_model_components":
        if len(path) == 2:
            return FieldPresentation(order=("kind", "name", "params"))
        if len(path) == 3 and path[-1] == "name":
            kind = _at(draft, (*path[:-1], "kind"))
            if not isinstance(kind, str) or kind not in _COMPOSITE_COMPONENT_REGISTRIES:
                return FieldPresentation(
                    disabled_reason="Select a component kind first."
                )
    if path[0] == "composite_model_assembly" and len(path) >= 3:
        if path[-1] == "component" and len(path) == 3:
            components = draft.get("composite_model_components")
            return FieldPresentation(
                suggestions=tuple(components) if isinstance(components, dict) else ()
            )
        if path[2] in {"inputs", "outputs"}:
            step = _at(draft, path[:2], {})
            component_id = step.get("component") if isinstance(step, dict) else None
            component = (
                _at(draft, ("composite_model_components", component_id), {})
                if isinstance(component_id, str)
                else {}
            )
            kind = component.get("kind") if isinstance(component, dict) else None
            registry = (
                _COMPOSITE_COMPONENT_REGISTRIES.get(kind)
                if isinstance(kind, str)
                else None
            )
            entry = _registered_entry(
                registry, component.get("name") if isinstance(component, dict) else None
            )
            if path[2] == "inputs":
                ports = getattr(entry, "runtime_inputs", ()) or ()
            else:
                ports = (
                    getattr(getattr(entry, "runtime_result", None), "outputs", ()) or ()
                )
            return FieldPresentation(
                suggestions=_reference_suggestions(
                    draft, outputs_only=path[2] == "outputs"
                ),
                key_suggestions=tuple(port.name for port in ports),
            )
    return FieldPresentation()


def loss_name_options(
    path: tuple, current_name: Any, section_data: Any
) -> list[str] | None:
    if len(path) != 4 or path[0] != "losses" or path[-1] != "name":
        return None
    registry = _LOSS_REGISTRIES.get(path[1])
    names = set(registry.canonical_keys()) if registry is not None else set()
    if isinstance(current_name, str):
        names.add(current_name)
    return sorted(names)


def loss_presentation(
    path: tuple, value: Any, draft: dict, *, composite: bool
) -> FieldPresentation:
    if len(path) == 4 and path[0] == "losses" and path[-1] == "weight_schedule":
        field = TrainingLossTermConfig.model_fields["weight_schedule"]
        spec = editor_spec(field.rebuild_annotation(), field)
        branches = {
            union_variant_details(spec, index)[0]: index
            for index in range(len(spec.variants))
        }

        names = set(LOSS_WEIGHT_SCHEDULES.canonical_keys())
        current_name = _at(draft, (*path, "name"))
        if isinstance(current_name, str):
            names.add(current_name)

        return FieldPresentation(
            named_variants=tuple(
                (
                    name,
                    branches[
                        _loss_weight_schedule_discriminator(
                            _normalize_loss_weight_schedule_config({"name": name})
                        )
                    ],
                )
                for name in sorted(names)
            ),
        )
    # Loss terms within a role have no user-defined execution order.
    if len(path) == 2 and path[0] == "losses":
        return FieldPresentation(reorderable=False)

    if len(path) == 4 and path[-1] == "name" and path[1] not in _LOSS_REGISTRIES:
        return FieldPresentation(
            disabled_reason="Select a loss role in the mapping key first."
        )
    if len(path) >= 4 and path[3] == "composite_wiring":
        applicable = composite and path[1] != "custom_objective"
        if not applicable:
            # Hide unused fields, but never hide an explicitly configured value.
            return FieldPresentation(hidden=True)
        name = _at(draft, (*path[:3], "name"))
        entry = _registered_entry(_LOSS_REGISTRIES.get(path[1]), name)
        ports = getattr(entry, "runtime_inputs", ()) or ()
        return FieldPresentation(
            suggestions=_reference_suggestions(draft),
            key_suggestions=tuple(port.name for port in ports),
        )
    return FieldPresentation()


def _registry_name_error(registry, name: str, label: str) -> str | None:
    try:
        registry.resolve_key(name)
    except (KeyError, ValueError, TypeError):
        return (
            f"{name!r} is not registered in the {label} registry. "
            "Choose a name from this registry; existing parameters have been preserved."
        )
    return None


def validate_loss_names(data: Any) -> dict[tuple, str]:
    """Apply the same dependent-dropdown check to role/name pairs."""
    errors = {}
    if not isinstance(data, dict):
        return errors
    for role, terms in data.items():
        registry = _LOSS_REGISTRIES.get(role)
        if registry is None or not isinstance(terms, list):
            continue
        for index, term in enumerate(terms):
            name = term.get("name") if isinstance(term, dict) else None
            if isinstance(name, str):
                error = _registry_name_error(registry, name, f"{role} loss")
                if error:
                    errors[(role, index, "name")] = error
    return errors


def dataset_presentation(path: tuple, value: Any, draft: dict) -> FieldPresentation:
    if path != ("dataset",):
        return FieldPresentation()
    field = TrainingConfig.model_fields["dataset"]
    spec = editor_spec(field.rebuild_annotation(), field)
    branches = {
        union_variant_details(spec, index)[0]: index
        for index in range(len(spec.variants))
    }
    names = set(DATASETS.canonical_keys())
    current_name = _at(draft, ("dataset", "name"))
    if isinstance(current_name, str):
        names.add(current_name)
    return FieldPresentation(
        named_variants=tuple(
            (name, branches[_dataset_config_discriminator({"name": name})])
            for name in sorted(names)
        ),
    )


def validate_dataset_name(data: Any) -> dict[tuple, str]:
    name = data.get("name") if isinstance(data, dict) else None
    if isinstance(name, str):
        error = _registry_name_error(DATASETS, name, "dataset")
        if error:
            return {("name",): error}
    return {}


def transform_name_options(
    path: tuple, current_name: Any, section_data: Any
) -> list[str] | None:
    if not (
        len(path) == 5
        and path[0] == "transform_pipelines"
        and path[2] == "steps"
        and path[-1] == "name"
    ):
        return None
    names = set(TRANSFORMS.canonical_keys())
    if isinstance(current_name, str):
        names.add(current_name)
    return sorted(names)


def transform_presentation(
    path: tuple, value: Any, draft: dict, *, composite: bool
) -> FieldPresentation:
    if len(path) == 5 and path[2] == "steps" and path[-1] == "apply_to":
        return FieldPresentation(reorderable=False)
    if len(path) == 3 and path[-1] in {"input", "output"}:
        suggestions = ("x",)
        if composite:
            declarations = _at(draft, ("composite_model_declarations", "expects"), {})
            suggestions = (
                tuple(
                    name
                    for name, role in declarations.items()
                    if isinstance(name, str)
                    and isinstance(role, str)
                    and TENSOR_STRUCTURE_BY_ROLE.get(role) == "image"
                )
                if isinstance(declarations, dict)
                else ()
            )
        return FieldPresentation(suggestions=suggestions)
    return FieldPresentation()


def validate_transform_names(data: Any) -> dict[tuple, str]:
    errors = {}
    if not isinstance(data, list):
        return errors
    for index, pipeline in enumerate(data):
        steps = pipeline.get("steps") if isinstance(pipeline, dict) else None
        if not isinstance(steps, list):
            continue
        for step_index, step in enumerate(steps):
            name = step.get("name") if isinstance(step, dict) else None
            if isinstance(name, str):
                error = _registry_name_error(TRANSFORMS, name, "transform")
                if error:
                    errors[(index, "steps", step_index, "name")] = error
    return errors
