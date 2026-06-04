from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_optional(
    override_value: Any,
    fallback_value: Any,
    *,
    field_name: str,
) -> Any:
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
) -> str:
    value = get_required_nested_value(data, section, key)

    if not isinstance(value, str):
        raise TypeError(
            f"Manifest field '{section}.{key}' must be a string, "
            f"got {type(value).__name__}."
        )

    return value


def get_required_nested_path(
    data: dict[str, Any],
    section: str,
    key: str,
    *,
    base_dir: Path,
) -> Path:
    value = get_required_nested_value(data, section, key)

    if not isinstance(value, str | Path):
        raise TypeError(
            f"Manifest field '{section}.{key}' must be a path-like string, "
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
) -> Any:
    if section not in data:
        raise KeyError(f"Required manifest section is missing: '{section}'")

    section_data = data[section]

    if not isinstance(section_data, dict):
        raise TypeError(
            f"Manifest section '{section}' must be a mapping/dictionary, "
            f"got {type(section_data).__name__}."
        )

    if key not in section_data:
        raise KeyError(f"Required manifest field is missing: '{section}.{key}'")

    value = section_data[key]

    if value is None:
        raise ValueError(f"Required manifest field is null: '{section}.{key}'")

    return value