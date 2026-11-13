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
    """Locate a packaged YAML without loading or validating it.

    Parameters
    ----------
    name
        Example bundle name when ``stage`` is supplied, such as
        ``"01_yaml_pipeline"``. Otherwise a YAML filename relative to the packaged
        ``configs`` directory, including any subdirectory.
    stage
        ``"training"``, ``"prediction"``, or ``"evaluation"``. When supplied, locate
        ``<name>/<stage>.yaml``.

    Returns
    -------
    Path
        Absolute path to the packaged YAML. Pass it to a workflow or wizard;
        incomplete templates can be completed through overrides.

    Raises
    ------
    ValueError
        Invalid stage, absolute or escaping resource path, or non-YAML resource.
    FileNotFoundError
        The requested resource does not exist.

    Notes
    -----
    Treat packaged files as read-only. Copy a YAML before editing it. Relative
    paths inside it retain the workflow's usual path semantics; this helper does
    not rewrite them relative to the package directory.
    """
    if stage is not None and stage not in _SCHEMAS:
        raise ValueError(f"Unknown workflow stage: {stage!r}")
    filename = name if stage is None else f"{name}/{stage}.yaml"
    path = _resource(_ROOT / "configs", filename)
    if path.suffix != ".yaml" or not path.is_file():
        raise ValueError(f"Expected an example YAML file: {filename!r}")
    return path


def example_asset_path(relative: str) -> Path:
    """Locate a packaged example input file or directory.

    Parameters
    ----------
    relative
        Path relative to the packaged ``examples/assets`` directory.

    Returns
    -------
    Path
        Absolute path to an existing resource. Treat packaged inputs as read-only.

    Raises
    ------
    ValueError
        The requested path is absolute or escapes the assets directory.
    FileNotFoundError
        The resource is missing or the request names the assets root itself.
    """
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
    """Load a complete packaged stage YAML into a fresh config object.

    Parameters
    ----------
    name
        Example bundle name, such as ``"01_yaml_pipeline"``.
    stage
        Stage selecting both the YAML and its workflow configuration class.

    Returns
    -------
    TrainingConfig or PredictionConfig or EvaluationConfig
        Schema-validated configuration matching the selected stage.

    Raises
    ------
    FileNotFoundError
        The requested packaged YAML is unavailable.
    ValueError
        The stage or resource path is invalid.
    pydantic.ValidationError
        The YAML does not satisfy the selected workflow schema.

    Notes
    -----
    This performs schema validation, not workflow resolution. Use
    ``update_config()`` to create edited copies. For an incomplete YAML that needs
    config overrides before validation, use ``example_config_path()`` instead.
    """
    path = example_config_path(name, stage=stage)
    return _SCHEMAS[stage].model_validate(load_yaml(path))
