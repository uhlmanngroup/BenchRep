"""Read-only tutorial resources shared by source checkouts and pip installations."""

from pathlib import Path
from typing import Literal, overload

from benchrep.assembly.config import load_yaml
from benchrep.assembly.schemas import EvaluationConfig, PredictionConfig, TrainingConfig

Stage = Literal["training", "prediction", "evaluation"]
_ROOT = Path(__file__).resolve().parent
_SCHEMAS = {
    "training": TrainingConfig,
    "prediction": PredictionConfig,
    "evaluation": EvaluationConfig,
}


def _resource(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Expected a path relative to {root.name}: {relative!r}")
    if not path.exists() or path == root:
        raise FileNotFoundError(f"No packaged example resource: {relative!r}")
    return path


def example_config_path(name: str, *, stage: Stage | None = None) -> Path:
    """Locate a shipped YAML. Use a bundle name and stage, or a relative YAML filename.

    Pass the path to a workflow or wizard to preserve the original YAML and
    allow incomplete templates to be completed by overrides.
    """
    if stage is not None and stage not in _SCHEMAS:
        raise ValueError(f"Unknown workflow stage: {stage!r}")
    filename = name if stage is None else f"{name}/{stage}.yaml"
    path = _resource(_ROOT / "configs", filename)
    if path.suffix != ".yaml" or not path.is_file():
        raise ValueError(f"Expected an example YAML file: {filename!r}")
    return path


def example_asset_path(relative: str) -> Path:
    """Locate a shipped example input file or directory under examples/assets."""
    return _resource(_ROOT / "assets", relative)


@overload
def load_example_config(name: str, *, stage: Literal["training"]) -> TrainingConfig: ...


@overload
def load_example_config(name: str, *, stage: Literal["prediction"]) -> PredictionConfig: ...


@overload
def load_example_config(name: str, *, stage: Literal["evaluation"]) -> EvaluationConfig: ...


def load_example_config(
    name: str, *, stage: Stage
) -> TrainingConfig | PredictionConfig | EvaluationConfig:
    """Validate a complete stage YAML and return a fresh config object."""
    path = example_config_path(name, stage=stage)
    return _SCHEMAS[stage].model_validate(load_yaml(path))
