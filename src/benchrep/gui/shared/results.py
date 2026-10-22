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
    """Configuration and execution overrides produced by the wizard.

    config
        Composed, validated workflow config, including supplied config
        components.
    overrides
        Overrides still required for execution. Config components already
        incorporated into config are removed from this returned bundle.
        The original caller-supplied overrides object is not modified.
    config_path
        Path to the authored YAML saved by the wizard.
    resolved_config_path
        Path to the resolved YAML saved by the wizard.
    authored_config
        Raw composed configuration, retaining omitted keys and explicit nulls.
    resolved_config
        Environment-dependent settings produced during review, for inspection.
        Execution should use config or config_path and resolve again.
    """

    config: ConfigT
    overrides: OverridesT

    config_path: Path | None = None
    resolved_config_path: Path | None = None
    authored_config: dict[str, Any] | None = None
    resolved_config: ConfigT | None = None
