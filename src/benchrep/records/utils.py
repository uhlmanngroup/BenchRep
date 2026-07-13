from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def paths_to_strings(value: Any) -> Any:
    """Recursively convert Path values to strings."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            key: paths_to_strings(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [paths_to_strings(item) for item in value]

    return value


def count_paths(value: Any) -> int:
    """Count Path objects in a nested structure."""
    if isinstance(value, Path):
        return 1

    if isinstance(value, Mapping):
        return sum(count_paths(item) for item in value.values())

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return sum(count_paths(item) for item in value)

    return 0