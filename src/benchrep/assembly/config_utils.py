from __future__ import annotations

from typing import Any


def require_mapping(config: Any, name: str) -> dict[str, Any]:
    """Require a config object to be a dictionary-like mapping."""
    if not isinstance(config, dict):
        raise TypeError(f"{name} must be a dictionary, got {type(config).__name__}.")

    return config


def get_required_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Get a required nested config section as a dictionary."""
    if key not in config:
        raise KeyError(f"Config must contain section {key!r}.")

    section = config[key]

    if not isinstance(section, dict):
        raise TypeError(
            f"Config section {key!r} must be a dictionary, "
            f"got {type(section).__name__}."
        )

    return section


def get_optional_section(
    config: dict[str, Any],
    key: str,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get an optional nested config section as a dictionary."""
    if key not in config:
        return {} if default is None else default

    section = config[key]

    if not isinstance(section, dict):
        raise TypeError(
            f"Config section {key!r} must be a dictionary if provided, "
            f"got {type(section).__name__}."
        )

    return section


def get_required_value(config: dict[str, Any], key: str) -> Any:
    """Get a required scalar or object value from a config dictionary."""
    if key not in config:
        raise KeyError(f"Config must contain key {key!r}.")

    return config[key]


def normalize_name(name: Any, field_name: str = "name") -> str:
    """Normalize a config name field for registry lookup."""
    if not isinstance(name, str):
        raise TypeError(f"{field_name} must be a string, got {type(name).__name__}.")

    name = name.lower().strip()

    if not name:
        raise ValueError(f"{field_name} must be a non-empty string.")

    return name