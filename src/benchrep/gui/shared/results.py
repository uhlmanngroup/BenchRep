from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

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
    """

    config: ConfigT
    overrides: OverridesT
