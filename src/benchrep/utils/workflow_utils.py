from pathlib import Path
from typing import TypeAlias, TypeVar

from benchrep.assembly.schemas.training_config_schema import TrainingConfig
from benchrep.assembly.schemas.prediction_config_schema import PredictionConfig
from benchrep.assembly.schemas.evaluation_config_schema import EvaluationConfig


TrainingConfigInput: TypeAlias = Path | str | TrainingConfig | None
PredictionConfigInput: TypeAlias = Path | str | PredictionConfig | None
EvaluationConfigInput: TypeAlias = Path | str | EvaluationConfig | None

ConfigT = TypeVar(
    "ConfigT",
    TrainingConfig,
    PredictionConfig,
    EvaluationConfig,
)


def route_config_input(
    config: Path | str | ConfigT | None,
    schema: type[ConfigT],
) -> tuple[Path | str | None, ConfigT | None]:
    if config is None or isinstance(config, (Path, str)):
        return config, None
    if isinstance(config, schema):
        return None, config
    raise TypeError(
        f"config must be a path, {schema.__name__}, or None; "
        f"got {type(config).__name__}."
    )