from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distributions
from importlib.metadata import version as distribution_version
from pathlib import Path
import platform
from typing import Any, Literal

import yaml


RuntimeEnvironmentStage = Literal[
    "training",
    "prediction",
    "evaluation",
]

RUNTIME_ENVIRONMENT_FILENAME = "runtime_environment.yaml"
RUNTIME_ENVIRONMENT_FORMAT_VERSION = 1


def collect_runtime_environment(
    *,
    stage: RuntimeEnvironmentStage,
    run_name: str,
) -> dict[str, Any]:
    """Collect the runtime environment shared by all BenchRep workflows."""
    return {
        "format_version": RUNTIME_ENVIRONMENT_FORMAT_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "run": {
            "stage": stage,
            "run_name": run_name,
        },
        "benchrep": {
            "version": _get_distribution_version("benchrep"),
        },
        "platform": {
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "operating_system_version": platform.version(),
            "architecture": platform.machine(),
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler(),
            },
        },
        "software": {
            "python_packages": _collect_python_packages(),
        },
    }


def write_runtime_environment(
    *,
    output_path: Path | str,
    stage: RuntimeEnvironmentStage,
    run_name: str,
) -> Path:
    """Collect and write a BenchRep runtime-environment record."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    environment = collect_runtime_environment(
        stage=stage,
        run_name=run_name,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(environment, handle, sort_keys=False)

    return output_path


def _get_distribution_version(distribution_name: str) -> str:
    try:
        return distribution_version(distribution_name)
    except PackageNotFoundError:
        return "unknown"


def _collect_python_packages() -> list[dict[str, str]]:
    packages = []

    for distribution in distributions():
        name = distribution.metadata.get("Name")

        if not name:
            continue

        packages.append(
            {
                "name": name,
                "version": distribution.version,
            }
        )

    return sorted(
        packages,
        key=lambda package: (
            package["name"].casefold(),
            package["version"],
        ),
    )