from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from benchrep.utils.workflow_overrides import (
    EvaluationOverrides,
    PredictionOverrides,
    TrainingOverrides,
)
from benchrep.utils.workflow_utils import ConfigT

OverridesT = TypeVar(
    "OverridesT",
    TrainingOverrides,
    PredictionOverrides,
    EvaluationOverrides,
)


@dataclass(frozen=True, kw_only=True)
class ConfiguredWorkflow(Generic[ConfigT, OverridesT]):
    """Configuration and execution overrides returned by the wizard.

    Attributes
    ----------
    config : ConfigT
        Composed, validated workflow config, including supplied config components.
    overrides : OverridesT
        Overrides still needed at execution. Components already incorporated into ``config`` are removed; the caller's original overrides are not modified.
    config_path : Path | None
        Saved authored YAML path, or ``None`` when unavailable.
    resolved_config_path : Path | None
        Saved resolved YAML path, or ``None`` when unavailable.
    authored_config : dict[str, Any] | None
        Authored mapping preserving omitted keys and explicit nulls, or ``None`` when unavailable.
    resolved_config : ConfigT | None
        Environment-dependent config produced during review, or ``None`` when unavailable. Use ``config`` or ``config_path`` for execution so settings are resolved again.
    """

    config: ConfigT
    overrides: OverridesT

    config_path: Path | None = None
    resolved_config_path: Path | None = None
    authored_config: dict[str, Any] | None = None
    resolved_config: ConfigT | None = None
