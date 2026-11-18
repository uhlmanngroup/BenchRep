"""Prepare authored configuration data for the workflow wizards."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from benchrep.assembly.config import (
    SupportedConfigComponent,
    load_yaml,
    normalize_config_components,
    prepare_authored,
)
from benchrep.utils.workflow_utils import ConfigT, route_config_input


def wizard_draft_inputs(
    config: Path | str | ConfigT | None,
    schema: type[ConfigT],
    config_components: Mapping[str, SupportedConfigComponent] | None,
) -> tuple[dict[str, Any], frozenset[str]]:
    """Load a draft and return the names of sections supplied through overrides.

    YAML may be incomplete while the user is editing. Typed inputs preserve
    explicit defaults, nulls, and the fields needed to identify their types.
    Caller-supplied parameter objects retain their Pydantic serialization.
    """
    path, typed = route_config_input(config, schema)
    if path is not None:
        data = load_yaml(path)
    elif typed is not None:
        prepared, _ = prepare_authored(typed)
        data = prepared.model_dump(mode="json", exclude_unset=True)
    else:
        data = {}

    if config_components:
        normalized = normalize_config_components(
            schema=schema, config_components=config_components
        )
        _, included = prepare_authored(config_components)
        data.update(TypeAdapter(Any).dump_python(normalized, mode="json", include=included))

    return data, frozenset(config_components or ())
