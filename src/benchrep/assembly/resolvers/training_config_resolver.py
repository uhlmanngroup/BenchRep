from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from benchrep.assembly.schemas import (
    RuntimeComponentOverrideConfig,
    TrainingCheckpointConfig,
    TrainingConfig,
    TrainingDataModuleConfig,
)
from benchrep.interfaces.model_families import ModelFamilySpec
from benchrep.assembly.registries.utils import normalize_name
from benchrep.assembly.resolvers.utils import ComponentSource, RunIdentitySpec


@dataclass(frozen=True)
class TrainingRunSpec:
    stage: Literal["training"]
    training_config: TrainingConfig
    model_family: ModelFamilySpec
    model_source: ComponentSource
    datamodule_source: ComponentSource
    compatibility_policy: Literal["error", "warn"]
    run_identity: RunIdentitySpec


def resolve_training_config(
    training_config: TrainingConfig,
    *,
    model_family: ModelFamilySpec,
    model_overridden: bool = False,
    datamodule_overridden: bool = False,
    model_override_name: str | None = None,
    compatibility_policy: Literal["error", "warn"] = "error",
) -> TrainingRunSpec:
    """Resolve a parsed training config into its pre-execution RunSpec."""

    if compatibility_policy not in {"error", "warn"}:
        raise ValueError(
            "`compatibility_policy` must be either 'error' or 'warn'."
        )

    resolved_datamodule = _resolve_datamodule_config(
        training_config.datamodule,
        datamodule_overridden=datamodule_overridden,
    )

    resolved_checkpointing = _resolve_checkpointing_config(
        training_config.checkpointing,
    )

    resolved_model_override = (
        training_config.overrides.model
        if model_overridden
        else None
    )
    if model_overridden and resolved_model_override is None:
        resolved_model_override = RuntimeComponentOverrideConfig()

    resolved_datamodule_override = (
        training_config.overrides.datamodule
        if datamodule_overridden
        else None
    )
    if (
            datamodule_overridden
            and resolved_datamodule_override is None
    ):
        resolved_datamodule_override = RuntimeComponentOverrideConfig()

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
    }

    if model_overridden:
        resolved_updates.update(
            {
                "model": None,
                "encoder": None,
                "decoder": None,
                "losses": None,
                "optimizer": None,
            }
        )

    if datamodule_overridden:
        resolved_updates.update(
            {
                "dataset": None,
                "transforms": [],
                "datamodule": None,
            }
        )

    resolved_config = training_config.model_copy(
        update=resolved_updates,
    )

    model_name = _resolve_training_model_name(
        training_config=resolved_config,
        model_family=model_family,
        model_overridden=model_overridden,
        model_override_name=model_override_name,
    )

    model_source: ComponentSource = (
        "external_object" if model_overridden else "config"
    )

    datamodule_source: ComponentSource = (
        "external_object" if datamodule_overridden else "config"
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
    assert training_config.encoder is not None

    configured_model_name = normalize_name(
        training_config.model.name,
        field_name="config.model.name",
    )

    if configured_model_name not in model_family.config_model_names:
        raise ValueError(
            "Configured model is incompatible with the selected training "
            "model family: "
            f"family={model_family.name!r}, "
            f"configured_model={configured_model_name!r}, "
            f"expected one of {model_family.config_model_names!r}."
        )

    model_name = (
        f"{training_config.model.name}_"
        f"{training_config.encoder.name}"
    )

    if training_config.decoder is not None:
        model_name = f"{model_name}_{training_config.decoder.name}"

    return model_name
