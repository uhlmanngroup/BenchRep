from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from benchrep.assembly.schemas import (
    TrainingCheckpointConfig,
    TrainingConfig,
    TrainingDataModuleConfig,
)
from benchrep.interfaces.model_families import ModelFamilySpec
from benchrep.assembly.registries.utils import normalize_name


TrainingComponentSource = Literal["config", "external_object"]


@dataclass(frozen=True)
class TrainingRunIdentitySpec:
    output_root: Path
    project_name: str | None
    model_name: str


@dataclass(frozen=True)
class TrainingRunSpec:
    stage: Literal["training"]
    training_config: TrainingConfig
    model_family: ModelFamilySpec
    model_source: TrainingComponentSource
    datamodule_source: TrainingComponentSource
    compatibility_policy: Literal["error", "warn"]
    run_identity: TrainingRunIdentitySpec


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

    resolved_config = training_config.model_copy(
        update={
            "datamodule": resolved_datamodule,
            "checkpointing": resolved_checkpointing,
        },
    )

    model_name = _resolve_training_model_name(
        training_config=resolved_config,
        model_family=model_family,
        model_overridden=model_overridden,
        model_override_name=model_override_name,
    )

    model_source: TrainingComponentSource = (
        "external_object" if model_overridden else "config"
    )

    datamodule_source: TrainingComponentSource = (
        "external_object" if datamodule_overridden else "config"
    )

    return TrainingRunSpec(
        stage=resolved_config.stage,
        training_config=resolved_config,
        model_family=model_family,
        model_source=model_source,
        datamodule_source=datamodule_source,
        compatibility_policy=compatibility_policy,
        run_identity=TrainingRunIdentitySpec(
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
