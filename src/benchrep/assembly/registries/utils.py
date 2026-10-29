from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias
from collections.abc import Mapping, Sequence

from pydantic import AfterValidator

def normalize_name(name: Any, field_name: str = "name") -> str:
    """Normalize a config name field for registry lookup."""
    if not isinstance(name, str):
        raise TypeError(f"{field_name} must be a string, got {type(name).__name__}.")

    name = name.lower().strip()

    if not name:
        raise ValueError(f"{field_name} must be a non-empty string.")

    return name


def normalize_path(value: Path | str) -> Path:
    """Reject blank strings and expand the home directory without rebasing."""
    if isinstance(value, str) and not value.strip():
        raise ValueError("Path must not be blank; use '.' explicitly if intended.")
    return Path(value).expanduser()


PathInput: TypeAlias = Annotated[Path | str, AfterValidator(normalize_path)]


# Resolving helpers
def resolve_registry_keys(
        selected: Sequence[str] | None,
        registry: Any,
        *,
        none_policy: Literal["preserve", "all"] = "preserve",
) -> list[str] | None:
    """Validate registry keys and resolve aliases to canonical names.

    Parameters
    ----------
    selected:
        Explicit keys to resolve. If ``None``, behavior depends on
        ``none_policy``.
    registry:
        Registry with ``resolve_key`` and ``canonical_keys`` methods.
    none_policy:
        ``"preserve"`` returns ``None`` when ``selected`` is ``None``.
        ``"all"`` returns all canonical registry keys when ``selected`` is
        ``None``.
    """
    if selected is None:
        if none_policy == "preserve":
            return None
        if none_policy == "all":
            return list(registry.canonical_keys())
        raise ValueError(
            "none_policy must be either 'preserve' or 'all', "
            f"got {none_policy!r}."
        )

    if isinstance(selected, str):
        raise TypeError(
            "selected must be a sequence of registry keys or None, not a string."
        )

    resolved: list[str] = []
    seen: set[str] = set()

    for key in selected:
        canonical_key = registry.resolve_key(key)

        if canonical_key in seen:
            continue

        resolved.append(canonical_key)
        seen.add(canonical_key)

    return resolved


def resolve_registry_param_keys(
        params: Mapping[str, Mapping[str, Any]] | None,
        registry: Any,
) -> dict[str, dict[str, Any]]:
    """Validate registry-param keys and resolve aliases to canonical names."""

    if params is None:
        return {}

    resolved: dict[str, dict[str, Any]] = {}

    for key, value in params.items():
        canonical_key = registry.resolve_key(key)

        if canonical_key in resolved:
            raise ValueError(
                f"Duplicate parameter entries resolve to the same canonical "
                f"{registry.name} key {canonical_key!r}."
            )

        resolved[canonical_key] = dict(value)

    return resolved