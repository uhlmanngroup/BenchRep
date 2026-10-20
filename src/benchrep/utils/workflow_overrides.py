from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import lightning as L

from benchrep.assembly.config import (
    SupportedTrainingConfigComponent,
    SupportedPredictionConfigComponent,
    SupportedEvaluationConfigComponent,
)
from benchrep.assembly.resolvers.utils import resolve_component_source
from benchrep.interfaces.model_families import (
    SupportedModel,
    SupportedModelBaseClass,
    ModelFamilySpec,
    CanonicalModelFamilySpec,
    COMPOSITE_FAMILY,
)


@dataclass(frozen=True, kw_only=True)
class TrainingOverrides:
    """Overrides supplied to a training entrypoint.

    config_components
        Mapping of TrainingConfig field names to typed config sections.
        Each component replaces the entire matching top-level section of
        the supplied YAML or config object. With no base config, components
        and schema defaults must provide enough information for validation.
    model
        Model instance or class compatible with the selected AE or VAE
        entrypoint. Whole-model overrides are not supported for composite
        models. None uses the configured model.
    datamodule
        LightningDataModule instance or class. None uses the configured
        datamodule.

    Supplying a class allows BenchRep to construct it using configured
    constructor parameters. A supplied instance is used directly.
    """

    config_components: Mapping[str, SupportedTrainingConfigComponent] | None = None
    model: SupportedModel | SupportedModelBaseClass | None = None
    datamodule: L.LightningDataModule | type[L.LightningDataModule] | None = None

    def validate_for(self, model_family: ModelFamilySpec) -> None:
        """Validate the model override against the selected model family."""
        _validate_model_override(self.model, model_family)


@dataclass(frozen=True, kw_only=True)
class PredictionOverrides:
    """Overrides supplied to a prediction entrypoint.

    config_components
        Mapping of PredictionConfig field names to typed config sections.
        Each component replaces the entire matching top-level section of
        the supplied YAML or config object. With no base config, components
        and schema defaults must provide enough information for validation.
    model
        Model instance or class compatible with the selected AE or VAE
        entrypoint. Whole-model overrides are not supported for composite
        models. None follows the normal model reconstruction path.
    datamodule
        LightningDataModule instance or class. None follows the normal
        datamodule construction path.
    training_manifest_path
        Training manifest YAML path, taking precedence over the path in
        the effective prediction config. None uses the configured path.

    External models, datamodules, and custom registrations used during
    training are not automatically restored as Python objects from the
    manifest; supply or register them again where required.
    """

    config_components: Mapping[str, SupportedPredictionConfigComponent] | None = None
    model: SupportedModel | SupportedModelBaseClass | None = None
    datamodule: L.LightningDataModule | type[L.LightningDataModule] | None = None
    training_manifest_path: Path | str | None = None

    def validate_for(self, model_family: ModelFamilySpec) -> None:
        """Validate the model override against the selected model family."""
        _validate_model_override(self.model, model_family)


@dataclass(frozen=True, kw_only=True)
class EvaluationOverrides:
    """Overrides supplied to the evaluation entrypoint.

    config_components
        Mapping of EvaluationConfig field names to typed config sections.
        Each component replaces the entire matching top-level section of
        the supplied YAML or config object. With no base config, components
        and schema defaults must provide enough information for validation.
    prediction_manifest_path
        Prediction manifest YAML path, taking precedence over the path in
        the effective evaluation config. None uses the configured source,
        which may instead specify standalone evaluation artifacts.
    """

    config_components: Mapping[str, SupportedEvaluationConfigComponent] | None = None
    prediction_manifest_path: Path | str | None = None


def _validate_model_override(
    model: SupportedModel | SupportedModelBaseClass | None,
    model_family: ModelFamilySpec,
) -> None:
    if isinstance(model_family, CanonicalModelFamilySpec):
        resolve_component_source(
            model,
            expected_base_class=model_family.model_base_class,
            component_name="model",
        )
    elif model_family == COMPOSITE_FAMILY:
        if model is not None:
            raise TypeError(
                "Whole-model overrides are supported only for the canonical "
                "`autoencoder` and `vae` model families; they are not supported "
                "for `composite`."
            )
    else:
        raise ValueError(f"Unsupported model family: {model_family!r}")