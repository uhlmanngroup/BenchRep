from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, Literal

from benchrep.assembly.schemas.runtime_override_config_schema import (
    RuntimeComponentOverrideConfig,
)


ResolvedT = TypeVar("ResolvedT")

ComponentSource = Literal[
    "config",
    "external_instance",
    "external_class",
]


@dataclass(frozen=True)
class RunIdentitySpec:
    output_root: Path
    project_name: str | None
    model_name: str


def resolve_component_source(
    component: object | None,
    *,
    expected_base_class: type[Any],
    component_name: str,
) -> ComponentSource:
    """Validate and classify an optional runtime component override."""
    if component is None:
        return "config"

    if isinstance(component, type):
        if not issubclass(component, expected_base_class):
            raise TypeError(
                f"`{component_name}` class must inherit from "
                f"`{expected_base_class.__name__}`."
            )

        return "external_class"

    if not isinstance(component, expected_base_class):
        raise TypeError(
            f"`{component_name}` must be an instance or subclass of "
            f"`{expected_base_class.__name__}`."
        )

    return "external_instance"


def get_component_override_name(component: object) -> str:
    """Return the class name for an external instance or class."""
    if isinstance(component, type):
        return component.__name__

    return type(component).__name__


def resolve_runtime_override_config(
    config: RuntimeComponentOverrideConfig | None,
    *,
    source: ComponentSource,
    config_path: str,
) -> RuntimeComponentOverrideConfig | None:
    """Resolve configuration for an external instance or class override."""
    if source == "config":
        if config is not None:
            raise ValueError(
                f"`{config_path}` requires an external component override."
            )

        return None

    resolved_config = config or RuntimeComponentOverrideConfig()

    if source == "external_instance" and resolved_config.params:
        raise ValueError(
            f"`{config_path}.params` cannot be used with an instantiated "
            "component. Pass the component class instead."
        )

    return resolved_config

def resolve_optional(
    override_value: ResolvedT | None,
    fallback_value: ResolvedT | None,
    *,
    field_name: str,
) -> ResolvedT:
    if override_value is not None:
        return override_value

    if fallback_value is None:
        raise ValueError(
            f"Could not resolve {field_name}: prediction config value is null, "
            "but the corresponding training config fallback is also null."
        )

    return fallback_value


def get_required_nested_str(
    data: dict[str, Any],
    section: str,
    key: str,
    *extra_keys: str,
) -> str:
    value = get_required_nested_value(data, section, key, *extra_keys)
    dotted_path = ".".join((section, key, *extra_keys))

    if not isinstance(value, str):
        raise TypeError(
            f"Manifest field '{dotted_path}' must be a string, "
            f"got {type(value).__name__}."
        )

    return value


def get_required_nested_path(
    data: dict[str, Any],
    section: str,
    key: str,
    *extra_keys: str,
    base_dir: Path,
) -> Path:
    value = get_required_nested_value(data, section, key, *extra_keys)
    dotted_path = ".".join((section, key, *extra_keys))

    if not isinstance(value, str | Path):
        raise TypeError(
            f"Manifest field '{dotted_path}' must be a path-like string, "
            f"got {type(value).__name__}."
        )

    path = Path(value)

    if not path.is_absolute():
        path = base_dir / path

    return path.resolve()


def get_required_nested_value(
    data: dict[str, Any],
    section: str,
    key: str,
    *extra_keys: str,
) -> Any:
    keys = (section, key, *extra_keys)
    current: Any = data

    for depth, current_key in enumerate(keys):
        dotted_path = ".".join(keys[: depth + 1])

        if not isinstance(current, dict):
            parent_path = ".".join(keys[:depth])
            raise TypeError(
                f"Manifest field '{parent_path}' must be a mapping/dictionary, "
                f"got {type(current).__name__}."
            )

        if current_key not in current:
            if depth == 0:
                raise KeyError(f"Required manifest section is missing: '{current_key}'")

            raise KeyError(f"Required manifest field is missing: '{dotted_path}'")

        current = current[current_key]

    if current is None:
        raise ValueError(f"Required manifest field is null: '{'.'.join(keys)}'")

    return current


def get_optional_nested_path(
    data: dict[str, Any],
    section: str,
    key: str,
    *extra_keys: str,
    base_dir: Path,
) -> Path | None:
    keys = (section, key, *extra_keys)
    dotted_path = ".".join(keys)

    try:
        value = get_required_nested_value(data, section, key, *extra_keys)
    except (KeyError, ValueError):
        return None

    if not isinstance(value, str | Path):
        raise TypeError(
            f"Manifest field '{dotted_path}' must be a path-like string or null, "
            f"got {type(value).__name__}."
        )

    path = Path(value)

    if not path.is_absolute():
        path = base_dir / path

    return path.resolve()


def get_optional_nested_value(
    data: dict[str, Any],
    section: str,
    key: str,
    *extra_keys: str,
) -> Any:
    keys = (section, key, *extra_keys)
    current: Any = data

    for depth, current_key in enumerate(keys):
        if not isinstance(current, dict):
            parent_path = ".".join(keys[:depth])
            raise TypeError(
                f"Manifest field '{parent_path}' must be a mapping/dictionary, "
                f"got {type(current).__name__}."
            )

        if current_key not in current:
            return None

        current = current[current_key]

    return current


def params_to_dict(params: Any) -> dict[str, Any]:
    if params is None:
        return {}

    if hasattr(params, "model_dump"):
        return params.model_dump(exclude_none=True)

    if isinstance(params, dict):
        return params

    raise TypeError(
        f"Expected params to be a mapping or Pydantic model, got {type(params).__name__}."
    )