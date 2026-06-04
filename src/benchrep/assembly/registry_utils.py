from __future__ import annotations

from typing import Any


def normalize_name(name: Any, field_name: str = "name") -> str:
    """Normalize a config name field for registry lookup."""
    if not isinstance(name, str):
        raise TypeError(f"{field_name} must be a string, got {type(name).__name__}.")

    name = name.lower().strip()

    if not name:
        raise ValueError(f"{field_name} must be a non-empty string.")

    return name