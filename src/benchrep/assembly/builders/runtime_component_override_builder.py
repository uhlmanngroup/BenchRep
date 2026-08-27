from __future__ import annotations

from typing import TypeVar

import lightning as L

from benchrep.assembly.schemas import RuntimeComponentOverrideConfig


ComponentT = TypeVar(
    "ComponentT",
    bound=L.LightningModule | L.LightningDataModule,
)


def build_runtime_component(
    component: ComponentT | type[ComponentT] | None,
    *,
    override_config: RuntimeComponentOverrideConfig | None,
    component_name: str,
) -> ComponentT | None:
    """Produce the final config-built or externally supplied component.

    Expected inputs:

    - ``component`` is the raw class, instance, or ``None`` received by the
      workflow entrypoint.
    - ``override_config`` is the corresponding ``overrides.model`` or
      ``overrides.datamodule`` value produced by the workflow resolver.

    When both are ``None``, this returns ``None`` so BenchRep's normal builder can
    construct the component. An external instance is returned unchanged, while an
    external class is instantiated using ``override_config.params``.
    """
    if component is None:
        if override_config is not None:
            raise ValueError(
                f"`{component_name}` was not supplied, but its resolved "
                "override configuration is present."
            )

        return None

    if override_config is None:
        raise ValueError(
            f"`{component_name}` was supplied, but its resolved override "
            "configuration is missing."
        )

    if isinstance(component, type):
        return component(**override_config.params)

    if override_config.params:
        raise ValueError(
            f"`overrides.{component_name}.params` cannot be used with an "
            "instantiated component. Pass the component class instead."
        )

    return component