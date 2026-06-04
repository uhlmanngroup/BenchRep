from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """Load a YAML config or manifest file as a dictionary.

    Parameters
    ----------
    yaml_path:
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed dictionary.
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")

    if not yaml_path.is_file():
        raise ValueError(f"YAML path must point to a file, got: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        yaml_file = yaml.safe_load(file)

    if yaml_file is None:
        raise ValueError(f"YAML file is empty: {yaml_path}")

    if not isinstance(yaml_file, dict):
        raise TypeError(
            f"YAML file must define a YAML mapping/dictionary at the top level, "
            f"got {type(yaml_file).__name__}."
        )

    return yaml_file