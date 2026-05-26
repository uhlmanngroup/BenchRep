from __future__ import annotations

from typing import Any

from benchrep.assembly.schemas.config_schema import BenchRepConfig


def parse_config(raw_config: dict[str, Any]) -> BenchRepConfig:
    """Validate a raw config dictionary and return a typed BenchRep config."""
    return BenchRepConfig.model_validate(raw_config)