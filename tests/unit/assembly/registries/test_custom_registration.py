from __future__ import annotations

from typing import Any

import pytest

from benchrep.assembly.registries.core import Registry
from benchrep.discovery.registry import _COMPONENT_REGISTRIES


EXPECTED_CUSTOM_REGISTRATION_POLICY = {
    "dataset": True,
    "transform": True,
    "encoder": True,
    "decoder": True,
    "model": False,
    "reconstruction_loss": True,
    "regularization_loss": True,
    "optimizer": True,
    "logger": True,
    "callback": True,
    "reduction": False,
    "clustering_method": False,
    "internal_clustering_metric": True,
    "external_clustering_metric": True,
    "embedding_metric": True,
    "predictability_probe": False,
    "reconstruction_metric": True,
}


def test_custom_registration_policy_is_complete() -> None:
    actual_policy = {
        registry_name: registry_info.registry.custom_registration_supported
        for registry_name, registry_info
        in _COMPONENT_REGISTRIES.items()
    }

    assert actual_policy == EXPECTED_CUSTOM_REGISTRATION_POLICY


@pytest.mark.parametrize(
    ("registry_name", "registry"),
    [
        pytest.param(
            registry_name,
            registry_info.registry,
            id=registry_name,
        )
        for registry_name, registry_info
        in _COMPONENT_REGISTRIES.items()
        if not registry_info.registry.custom_registration_supported
    ],
)
def test_unsupported_registry_rejects_custom_registration(
    registry_name: str,
    registry: Registry,
) -> None:
    class CustomComponent:
        pass

    expected_message = (
        "Custom registration is not supported for the "
        f"{registry.name} registry. This registry exposes BenchRep's "
        "built-in components for discovery and configuration only."
    )

    with pytest.raises(RuntimeError) as exc_info:
        registry.register(
            f"test_custom_{registry_name}",
            CustomComponent,
        )

    assert str(exc_info.value) == expected_message
    assert f"test_custom_{registry_name}" not in registry.keys()