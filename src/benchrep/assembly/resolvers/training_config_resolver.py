from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from benchrep.assembly.schemas import (
    TrainingCheckpointConfig,
    TrainingConfig,
    TrainingDataModuleConfig,
    CompositeModelDeclarationsConfig,
    TrainingTransformPipelineConfig,
)
from benchrep.interfaces.model_families import (
    CanonicalModelFamilySpec,
    ModelFamilySpec,
)
from benchrep.assembly.registries.core import MODELS
from benchrep.assembly.registries.utils import normalize_name
from benchrep.assembly.resolvers.composite_model_resolver import (
    CompositeModelSpec,
    resolve_composite_model_config,
)
from benchrep.assembly.resolvers.utils import (
    ComponentSource,
    RunIdentitySpec,
    resolve_runtime_override_config,
)
from benchrep.architecture.composite_model_roles import (
    TENSOR_STRUCTURE_BY_ROLE,
)


@dataclass(frozen=True)
class TrainingRunSpec:
    stage: Literal["training"]
    training_config: TrainingConfig
    model_family: ModelFamilySpec
    model_source: ComponentSource
    datamodule_source: ComponentSource
    compatibility_policy: Literal["error", "warn"]
    run_identity: RunIdentitySpec
    composite_model_spec: CompositeModelSpec | None = None


def resolve_training_config(
    training_config: TrainingConfig,
    *,
    model_family: ModelFamilySpec,
    model_source: ComponentSource = "config",
    datamodule_source: ComponentSource = "config",
    model_override_name: str | None = None,
    compatibility_policy: Literal["error", "warn"] = "error",
) -> TrainingRunSpec:
    """Resolve a parsed training config into its pre-execution RunSpec."""

    if compatibility_policy not in {"error", "warn"}:
        raise ValueError(
            "`compatibility_policy` must be either 'error' or 'warn'."
        )

    model_is_external = model_source != "config"
    datamodule_is_external = datamodule_source != "config"

    resolved_transform_pipelines = _resolve_transform_pipelines(
        training_config.transform_pipelines,
        model_family=model_family,
        declarations_config=training_config.composite_model_declarations,
        datamodule_overridden=datamodule_is_external,
    )

    if (
        model_is_external
        and not isinstance(model_family, CanonicalModelFamilySpec)
    ):
        raise ValueError(
            "Whole-model overrides are supported only for the canonical "
            "`autoencoder` and `vae` model families; they are not supported "
            "for `composite`."
        )

    resolved_datamodule = _resolve_datamodule_config(
        training_config.datamodule,
        datamodule_overridden=datamodule_is_external,
    )

    resolved_checkpointing = _resolve_checkpointing_config(
        training_config.checkpointing,
    )

    resolved_model_override = resolve_runtime_override_config(
        training_config.overrides.model,
        source=model_source,
        config_path="overrides.model",
    )

    resolved_datamodule_override = resolve_runtime_override_config(
        training_config.overrides.datamodule,
        source=datamodule_source,
        config_path="overrides.datamodule",
    )

    resolved_overrides = training_config.overrides.model_copy(
        update={
            "model": resolved_model_override,
            "datamodule": resolved_datamodule_override,
        },
    )

    resolved_updates: dict[str, Any] = {
        "overrides": resolved_overrides,
        "datamodule": resolved_datamodule,
        "checkpointing": resolved_checkpointing,
        "transform_pipelines": resolved_transform_pipelines,
    }

    if model_is_external:
        resolved_updates.update(
            {
                "model": None,
                "encoder": None,
                "decoder": None,
                "composite_model_declarations": None,
                "composite_model_components": None,
                "composite_model_assembly": None,
                "losses": None,
                "optimizer": None,
            }
        )

    if datamodule_is_external:
        resolved_updates.update(
            {
                "dataset": None,
                "transform_pipelines": [],
                "datamodule": None,
            }
        )

    resolved_config = training_config.model_copy(
        update=resolved_updates,
    )

    # Composite model config resolution.
    composite_model_spec: CompositeModelSpec | None = None

    if not model_is_external:
        assert resolved_config.model is not None

        configured_model_name = MODELS.resolve_key(
            resolved_config.model.name
        )

        if configured_model_name == "composite":
            assert resolved_config.composite_model_declarations is not None
            assert resolved_config.composite_model_components is not None
            assert resolved_config.composite_model_assembly is not None
            assert resolved_config.losses is not None

            composite_model_spec = resolve_composite_model_config(
                declarations_config=(
                    resolved_config.composite_model_declarations
                ),
                components_config=(
                    resolved_config.composite_model_components
                ),
                assembly_config=(
                    resolved_config.composite_model_assembly
                ),
                losses_config=resolved_config.losses,
            )

    model_name = _resolve_training_model_name(
        training_config=resolved_config,
        model_family=model_family,
        model_overridden=model_is_external,
        model_override_name=model_override_name,
    )

    return TrainingRunSpec(
        stage=resolved_config.stage,
        training_config=resolved_config,
        model_family=model_family,
        model_source=model_source,
        datamodule_source=datamodule_source,
        compatibility_policy=compatibility_policy,
        run_identity=RunIdentitySpec(
            output_root=resolved_config.run.output_root,
            project_name=resolved_config.run.project_name,
            model_name=model_name,
        ),
        composite_model_spec=composite_model_spec,
    )


def _resolve_datamodule_config(
    config: TrainingDataModuleConfig | None,
    *,
    datamodule_overridden: bool,
) -> TrainingDataModuleConfig | None:
    if (
        config is None
        or datamodule_overridden
        or config.pin_memory != "auto"
    ):
        return config

    return config.model_copy(
        update={"pin_memory": torch.cuda.is_available()},
    )


def _resolve_checkpointing_config(
    config: TrainingCheckpointConfig,
) -> TrainingCheckpointConfig:
    if config.monitor is not None or config.save_top_k == 0:
        return config

    return config.model_copy(
        update={"save_top_k": 0},
    )


def _resolve_training_model_name(
    *,
    training_config: TrainingConfig,
    model_family: ModelFamilySpec,
    model_overridden: bool,
    model_override_name: str | None,
) -> str:
    if model_overridden:
        if model_override_name is None:
            raise ValueError(
                "`model_override_name` is required when the training "
                "model is overridden."
            )

        return f"{model_family.name}_external_{model_override_name}"

    assert training_config.model is not None

    configured_model_name = MODELS.resolve_key(
        normalize_name(
            training_config.model.name,
            field_name="model.name",
        )
    )

    if configured_model_name != model_family.name:
        raise ValueError(
            "Configured model is incompatible with the selected training "
            "model family: "
            f"family={model_family.name!r}, "
            f"configured_model={configured_model_name!r}, "
            f"expected={model_family.name!r}."
        )

    if configured_model_name == "composite":
        return configured_model_name

    # Canonical models path
    assert training_config.encoder is not None

    model_name = (
        f"{training_config.model.name}_"
        f"{training_config.encoder.name}"
    )

    if training_config.decoder is not None:
        model_name = f"{model_name}_{training_config.decoder.name}"

    return model_name


def _resolve_transform_pipelines(
    config: list[TrainingTransformPipelineConfig],
    *,
    model_family: ModelFamilySpec,
    declarations_config: CompositeModelDeclarationsConfig | None,
    datamodule_overridden: bool,
) -> list[TrainingTransformPipelineConfig]:
    if datamodule_overridden or not config:
        return []

    # For canonical models, only in-place augmentations are supported.
    if isinstance(model_family, CanonicalModelFamilySpec):
        for index, pipeline in enumerate(config):
            route = (pipeline.input, pipeline.output)

            if route not in {
                (None, None),
                ("x", "x"),
            }:
                raise ValueError(
                    "Canonical model transform pipelines use the fixed route "
                    "`x` to `x`; omit both fields or provide that resolved "
                    f"route: transform_pipelines[{index}]."
                )

        return [
            pipeline.model_copy(
                update={
                    "input": "x",
                    "output": "x",
                }
            )
            for pipeline in config
        ]

    # Composite models require declarations.
    assert declarations_config is not None

    input_roles = declarations_config.expects
    sample_image_name = next(
        name
        for name, role in input_roles.items()
        if role == "sample_image"
    )

    resolved: list[TrainingTransformPipelineConfig] = []

    for index, pipeline in enumerate(config):
        # Default fallback is in-place augmentation of whatever is declared
        # under the "sample_image" role, which can only have one assignment,
        # or whatever is under the input field.
        input_name = pipeline.input or sample_image_name
        output_name = pipeline.output or input_name

        for field_name, declared_name in (
            ("input", input_name),
            ("output", output_name),
        ):
            if declared_name not in input_roles:
                raise ValueError(
                    f"`transform_pipelines[{index}].{field_name}` references "
                    f"{declared_name!r}, which is not declared under "
                    "`composite_model_declarations.expects`."
                )

            role = input_roles[declared_name]

            if TENSOR_STRUCTURE_BY_ROLE[role] != "image":
                raise ValueError(
                    f"`transform_pipelines[{index}].{field_name}` references "
                    f"{declared_name!r} with role {role!r}; transform pipeline "
                    "routing supports only image-valued declarations."
                )

        resolved.append(
            pipeline.model_copy(
                update={
                    "input": input_name,
                    "output": output_name,
                }
            )
        )

    return resolved
