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
        dotted_path = ".".join(keys[: depth + 1])

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