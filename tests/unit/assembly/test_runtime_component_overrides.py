from __future__ import annotations

import lightning as L
import pytest

from benchrep.assembly.builders import build_runtime_component
from benchrep.assembly.resolvers.utils import (
    ComponentSource,
    resolve_component_source,
    resolve_runtime_override_config,
)
from benchrep.assembly.schemas import RuntimeComponentOverrideConfig
from benchrep.interfaces.models import BenchRepAutoencoderModel
from tests.fixtures.datamodules import ParameterizedExternalDataModule
from tests.fixtures.models import CompatibleExternalAutoencoder


@pytest.mark.parametrize(
    ("component", "expected_source"),
    [
        pytest.param(None, "config", id="config"),
        pytest.param(
            CompatibleExternalAutoencoder(),
            "external_instance",
            id="external-instance",
        ),
        pytest.param(
            CompatibleExternalAutoencoder,
            "external_class",
            id="external-class",
        ),
    ],
)
def test_resolve_component_source_classifies_model_override(
    component: (
        CompatibleExternalAutoencoder
        | type[CompatibleExternalAutoencoder]
        | None
    ),
    expected_source: ComponentSource,
) -> None:
    assert resolve_component_source(
        component,
        expected_base_class=BenchRepAutoencoderModel,
        component_name="model",
    ) == expected_source


@pytest.mark.parametrize(
    "component",
    [
        pytest.param(object(), id="invalid-instance"),
        pytest.param(L.LightningDataModule, id="invalid-class"),
    ],
)
def test_resolve_component_source_rejects_wrong_component_type(
    component: object,
) -> None:
    with pytest.raises(TypeError, match="BenchRepAutoencoderModel"):
        resolve_component_source(
            component,
            expected_base_class=BenchRepAutoencoderModel,
            component_name="model",
        )


def test_external_class_override_preserves_constructor_params() -> None:
    config = RuntimeComponentOverrideConfig(
        params={"latent_dim": 6, "lr": 2e-3},
    )

    resolved = resolve_runtime_override_config(
        config,
        source="external_class",
        config_path="overrides.model",
    )

    assert resolved is config


def test_external_class_without_config_gets_empty_constructor_params() -> None:
    resolved = resolve_runtime_override_config(
        None,
        source="external_class",
        config_path="overrides.model",
    )

    assert resolved == RuntimeComponentOverrideConfig()


def test_external_instance_override_rejects_constructor_params() -> None:
    config = RuntimeComponentOverrideConfig(
        params={"latent_dim": 6},
    )

    with pytest.raises(
        ValueError,
        match=r"overrides\.model\.params.*instantiated component",
    ):
        resolve_runtime_override_config(
            config,
            source="external_instance",
            config_path="overrides.model",
        )


def test_config_built_component_rejects_external_override_config() -> None:
    with pytest.raises(
        ValueError,
        match=r"overrides\.model.*external component override",
    ):
        resolve_runtime_override_config(
            RuntimeComponentOverrideConfig(),
            source="config",
            config_path="overrides.model",
        )


def test_external_class_is_instantiated_with_resolved_params() -> None:
    component = build_runtime_component(
        CompatibleExternalAutoencoder,
        override_config=RuntimeComponentOverrideConfig(
            params={"latent_dim": 6, "lr": 2e-3},
        ),
        component_name="model",
    )

    assert isinstance(component, CompatibleExternalAutoencoder)
    assert component.latent_dim == 6
    assert component.lr == 2e-3


def test_external_instance_is_returned_unchanged() -> None:
    instance = ParameterizedExternalDataModule(seed=200)

    component = build_runtime_component(
        instance,
        override_config=RuntimeComponentOverrideConfig(),
        component_name="datamodule",
    )

    assert component is instance


@pytest.mark.parametrize(
    ("component", "override_config", "message"),
    [
        pytest.param(
            None,
            RuntimeComponentOverrideConfig(),
            "was not supplied",
            id="config-without-component",
        ),
        pytest.param(
            CompatibleExternalAutoencoder,
            None,
            "configuration is missing",
            id="class-without-config",
        ),
    ],
)
def test_builder_rejects_component_and_config_disagreement(
    component: (
        CompatibleExternalAutoencoder
        | type[CompatibleExternalAutoencoder]
        | None
    ),
    override_config: RuntimeComponentOverrideConfig | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_runtime_component(
            component,
            override_config=override_config,
            component_name="model",
        )
