from __future__ import annotations

from typing import Any

from benchrep.architecture.heads import BaseHead
from benchrep.assembly.registries.core import HEADS
from benchrep.assembly.registries.utils import normalize_name


def build_head(
    name: str,
    *,
    params: dict[str, Any] | None = None,
) -> BaseHead:
    """Build a registered model head."""

    head_name = normalize_name(
        name,
        field_name="head.name",
    )

    head = HEADS.create(
        head_name,
        **(params or {}),
    )

    if not isinstance(head, BaseHead):
        raise TypeError(
            f"Registered head {head_name!r} produced "
            f"{type(head).__name__}, expected a BaseHead instance."
        )

    return head