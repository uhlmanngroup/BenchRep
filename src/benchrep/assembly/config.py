from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a dictionary.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.

    Returns
    -------
    dict[str, Any]
        Parsed config dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    if not config_path.is_file():
        raise ValueError(f"Config path must point to a file, got: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(config, dict):
        raise TypeError(
            f"Config file must define a YAML mapping/dictionary at the top level, "
            f"got {type(config).__name__}."
        )

    return config