"""Utilities for saving config records for a BenchRep run.

This module writes the config-related records for a run:

- the original user-provided config file, if one exists
- the resolved/validated config object actually used by BenchRep

Config records are written to `RunContext.config_dir` by default, but an
explicit output directory can also be provided.
"""

from pathlib import Path
from shutil import copy2
from typing import Any

import yaml
from pydantic import BaseModel

from benchrep.assembly.schemas import BenchRepConfig
from benchrep.runtime import RunContext


def save_config_records(
    *,
    config_out_dir: RunContext | Path | str,
    resolved_config: BenchRepConfig | dict[str, Any],
    original_config_path: Path | str | None = None,
) -> None:
    """Save config records for a run.

    Parameters
    ----------
    config_out_dir:
        RunContext or explicit config output directory.
    resolved_config:
        Validated config object, or a plain config dictionary.
    original_config_path:
        Optional path to the original user-provided YAML config file.
        This may be absent when BenchRep is used without a config file.
    """
    if isinstance(config_out_dir, RunContext):
        config_out_dir = config_out_dir.config_dir
    else:
        config_out_dir = Path(config_out_dir).expanduser().resolve()

    if original_config_path is not None:
        original_config_path = Path(original_config_path).expanduser().resolve()

        save_original_config(
            original_config_path=original_config_path,
            out_dir=config_out_dir,
        )

    save_resolved_config(
        resolved_config=resolved_config,
        out_dir=config_out_dir,
    )


def save_original_config(
    *,
    original_config_path: Path,
    out_dir: Path,
    filename: str = "original_config.yaml",
) -> Path:
    """Copy the original user-provided config file into the run config directory."""

    if not original_config_path.exists():
        raise FileNotFoundError(
            f"Original config file does not exist: {original_config_path}"
        )

    if not original_config_path.is_file():
        raise ValueError(
            f"Original config path is not a file: {original_config_path}"
        )

    output_path = out_dir / filename
    copy2(original_config_path, output_path)

    return output_path


def save_resolved_config(
    *,
    resolved_config: BaseModel | dict[str, Any],
    out_dir: Path,
    filename: str = "resolved_config.yaml",
) -> Path:
    """Save the resolved config object into the run config directory."""

    output_path = out_dir / filename
    config_dict = _as_serializable_dict(resolved_config)

    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config_dict, file, sort_keys=False)

    return output_path


def _as_serializable_dict(config: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic config object or plain dictionary into a YAML-safe dict."""

    if isinstance(config, BaseModel):
        return config.model_dump(mode="json")

    if isinstance(config, dict):
        return config

    raise TypeError(
        "`resolved_config` must be a Pydantic model or a plain dictionary, "
        f"got {type(config).__name__}."
    )