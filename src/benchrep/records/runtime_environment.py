from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distributions
from importlib.metadata import version as distribution_version
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Literal, TYPE_CHECKING

import torch
import yaml
from yaml.nodes import ScalarNode

from benchrep.records.utils import now_isoformat
from benchrep.interfaces.model_families import VAE_FAMILY

if TYPE_CHECKING:
    import lightning as L

    from benchrep.assembly.resolvers import (
        TrainingRunSpec,
        PredictionRunSpec,
    )
    from benchrep.assembly.resolvers.evaluation_config_resolver import (
        EvaluationRunSpec,
        PredictabilityTargetSpec,
    )


RuntimeEnvironmentStage = Literal[
    "training",
    "prediction",
    "evaluation",
]

RUNTIME_ENVIRONMENT_FORMAT_VERSION = 1


class _LiteralString(str):
    """String serialized using YAML literal-block style."""


class _RuntimeEnvironmentDumper(yaml.SafeDumper):
    """Safe YAML dumper for runtime-environment records."""


def _represent_literal_string(
    dumper: yaml.SafeDumper,
    value: _LiteralString,
) -> ScalarNode:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str",
        str(value),
        style="|",
    )


_RuntimeEnvironmentDumper.add_representer(
    _LiteralString,
    _represent_literal_string,
)


def get_runtime_environment_filename(
    stage: RuntimeEnvironmentStage,
) -> str:
    return f"{stage}_runtime_environment.yaml"


def collect_runtime_environment(
    *,
    stage: RuntimeEnvironmentStage,
    run_name: str,
    workflow_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect the runtime environment shared by all BenchRep workflows."""
    git_provenance, collection_warnings = _collect_git_provenance()
    hardware, hardware_warnings = _collect_hardware()
    collection_warnings.extend(hardware_warnings)
    accelerator_stack, accelerator_warnings = (
        _collect_accelerator_stack()
    )
    collection_warnings.extend(accelerator_warnings)

    return {
        "format_version": RUNTIME_ENVIRONMENT_FORMAT_VERSION,
        "captured_at": now_isoformat(),
        "run": {
            "stage": stage,
            "run_name": run_name,
        },
        "workflow": dict(workflow_context),
        "benchrep": {
            "version": _get_distribution_version("benchrep"),
        },
        "platform": {
            "operating_system": platform.system(),
            "distribution": _get_os_distribution(),
            "operating_system_release": platform.release(),
            "operating_system_version": platform.version(),
            "architecture": platform.machine(),
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler(),
            },
        },
        "hardware": hardware,
        "source": {
            "git": git_provenance,
        },
        "software": {
            "accelerator_stack": accelerator_stack,
            "python_packages": _collect_python_packages(),
        },
        "collection_warnings": collection_warnings,
    }


def collect_training_environment_context(
    *,
    run_spec: TrainingRunSpec,
    trainer: L.Trainer,
) -> dict[str, Any]:
    """Collect runtime and reproducibility context for training."""
    resolved_training_config = run_spec.training_config
    datamodule_config = resolved_training_config.datamodule

    configured_num_workers = (
        datamodule_config.num_workers
        if run_spec.datamodule_source == "config"
        and datamodule_config is not None
        else None
    )

    return {
        "execution": {
            "configured": {
                "accelerator": resolved_training_config.trainer.accelerator,
                "devices": resolved_training_config.trainer.devices,
                "strategy": getattr(
                    resolved_training_config.trainer,
                    "strategy",
                    None,
                ),
                "num_nodes": getattr(
                    resolved_training_config.trainer,
                    "num_nodes",
                    None,
                ),
                "precision": resolved_training_config.trainer.precision,
            },
            "resolved": _collect_trainer_runtime_state(trainer),
        },
        "reproducibility": {
            "configured": {
                "global_seed": resolved_training_config.reproducibility.seed,
                "seed_workers": (
                    resolved_training_config.reproducibility.seed_workers
                ),
                "float32_matmul_precision": (
                    resolved_training_config.reproducibility
                    .float32_matmul_precision
                ),
                "deterministic": (
                    resolved_training_config.trainer.deterministic
                ),
                "benchmark": resolved_training_config.trainer.benchmark,
            },
            "data_loading": {
                "datamodule_source": run_spec.datamodule_source,
                "num_workers": configured_num_workers,
            },
            "runtime_state": _collect_torch_runtime_state(),
        },
    }

def collect_prediction_environment_context(
        *,
        run_spec: PredictionRunSpec,
        trainer: L.Trainer,
) -> dict[str, Any]:
    """Collect runtime and reproducibility context for prediction."""
    prediction_config = run_spec.prediction_config
    model_family = run_spec.model_family
    model_source = run_spec.model_source
    datamodule_source = run_spec.datamodule_source
    reconstruction_spec = run_spec.export_spec.reconstructions

    reconstruction_export_uses_randomness = (
        reconstruction_spec.enabled
        and reconstruction_spec.selection == "random"
        and reconstruction_spec.n_examples != "all"
    )

    vae_reconstruction_applicable = model_family == VAE_FAMILY

    vae_reconstruction_uses_randomness = (
        run_spec.reconstruction_latent_source == "sample"
        if vae_reconstruction_applicable and model_source == "config"
        else None
    )

    return {
        "execution": {
            "configured": {
                "accelerator": run_spec.trainer_config.accelerator,
                "devices": run_spec.trainer_config.devices,
                "strategy": getattr(
                    run_spec.trainer_config,
                    "strategy",
                    None,
                ),
                "num_nodes": getattr(
                    run_spec.trainer_config,
                    "num_nodes",
                    None,
                ),
                "precision": run_spec.trainer_config.precision,
                "max_batches": run_spec.max_batches,
            },
            "resolved": {
                **_collect_trainer_runtime_state(trainer),
                "limit_predict_batches": (
                    trainer.limit_predict_batches
                ),
            },
        },
        "reproducibility": {
            "resolved_config": {
                "global_seed": prediction_config.inference.seed,
                "seed_workers": (
                    prediction_config.inference.seed_workers
                ),
                "deterministic": (
                    prediction_config.inference.deterministic
                ),
                "float32_matmul_precision": (
                    prediction_config.inference
                    .float32_matmul_precision
                ),
                "reconstruction_latent_source": (
                    prediction_config.inference.reconstruction_latent_source
                ),
            },
            "resolved": {
                "global_seed": run_spec.seed,
                "seed_workers": run_spec.seed_workers,
                "deterministic": (
                    run_spec.trainer_config.deterministic
                ),
                "benchmark": run_spec.trainer_config.benchmark,
                "float32_matmul_precision": (
                    run_spec.float32_matmul_precision
                ),
                "reconstruction_latent_source": (
                    run_spec.reconstruction_latent_source
                ),
            },
            "components": {
                "reconstruction_export_selection": {
                    "enabled": reconstruction_spec.enabled,
                    "selection": reconstruction_spec.selection,
                    "uses_randomness": (
                        reconstruction_export_uses_randomness
                    ),
                    "seed": (
                        reconstruction_spec.seed
                        if reconstruction_export_uses_randomness
                        else None
                    ),
                },
                "vae_reconstruction": {
                    "applicable": vae_reconstruction_applicable,
                    "model_source": (
                        model_source
                        if vae_reconstruction_applicable
                        else None
                    ),
                    "latent_source": (
                        run_spec.reconstruction_latent_source
                        if vae_reconstruction_applicable
                        else None
                    ),
                    "uses_randomness": (
                        vae_reconstruction_uses_randomness
                    ),
                    "seed": (
                        run_spec.seed
                        if vae_reconstruction_uses_randomness
                        else None
                    ),
                },
            },
            "data_loading": {
                "datamodule_source": datamodule_source,
                "resolved_config_batch_size": (
                    prediction_config.data.batch_size
                ),
                "effective_batch_size": run_spec.batch_size,
                "resolved_config_num_workers": (
                    prediction_config.data.num_workers
                ),
                "effective_num_workers": run_spec.num_workers,
            },
            "runtime_state": _collect_torch_runtime_state(),
        },
    }


def _collect_predictability_target_environment_context(
    *,
    target_spec: PredictabilityTargetSpec,
    enabled: bool,
) -> dict[str, Any]:
    """Collect resolved reproducibility context for one predictability target."""

    cv_params = dict(target_spec.cv_params)
    tuning_params = dict(target_spec.tuning_params)
    cv_shuffle = bool(cv_params.get("shuffle", False))
    cv_uses_randomness = enabled and cv_shuffle

    tuning_enabled = (
        enabled
        and bool(tuning_params.get("enabled", False))
    )
    tuning_uses_randomness = tuning_enabled and cv_shuffle

    probes: dict[str, dict[str, Any]] = {}

    for probe_name in target_spec.probes:
        params = dict(target_spec.probe_params.get(probe_name, {}))

        if probe_name == "dummy":
            uses_randomness = (
                enabled
                and params.get("strategy") in {"stratified", "uniform"}
            )
        else:
            uses_randomness = (
                enabled
                and probe_name in {"random_forest", "xgboost"}
            )

        probes[probe_name] = {
            **params,
            "uses_randomness": uses_randomness,
        }

    return {
        "task": target_spec.task,
        "selected_probes": list(target_spec.probes),
        "cross_validation": {
            **cv_params,
            "uses_randomness": cv_uses_randomness,
        },
        "tuning": {
            **tuning_params,
            "uses_randomness": tuning_uses_randomness,
            "random_state": (
                cv_params.get("random_state")
                if tuning_uses_randomness
                else None
            ),
        },
        "probes": probes,
    }


def collect_evaluation_environment_context(
    *,
    run_spec: EvaluationRunSpec,
) -> dict[str, Any]:
    """Collect resolved reproducibility context for evaluation."""
    step_spec = run_spec.step_spec

    predictability_enabled = step_spec.predictability_enabled
    predictability_targets = {
        target_spec.target_key: (
            _collect_predictability_target_environment_context(
                target_spec=target_spec,
                enabled=predictability_enabled,
            )
        )
        for target_spec in step_spec.predictability_targets
    }

    reconstruction_grid_params = (
        step_spec.plot_params.get(
            "reconstruction_grid",
            {},
        )
    )
    reconstruction_grid_enabled = (
        step_spec.plots_enabled
        and run_spec.input_spec.reconstructions is not None
    )

    return {
        "execution": {
            "lightning_trainer_used": False,
        },
        "reproducibility": {
            "global_seed_applied": False,
            "components": {
                "reductions": {
                    "pca": _collect_enabled_random_state(
                        enabled=step_spec.pca_enabled,
                        params=step_spec.pca_params,
                    ),
                    "umap": _collect_enabled_random_state(
                        enabled=step_spec.umap_enabled,
                        params=step_spec.umap_params,
                    ),
                    "tsne": _collect_enabled_random_state(
                        enabled=step_spec.tsne_enabled,
                        params=step_spec.tsne_params,
                    ),
                },
                "clustering": {
                    "kmeans": _collect_enabled_random_state(
                        enabled=step_spec.kmeans_enabled,
                        params=step_spec.kmeans_params,
                    ),
                    "leiden": _collect_enabled_random_state(
                        enabled=step_spec.leiden_enabled,
                        params=step_spec.leiden_params,
                    ),
                },
                "predictability": {
                    "enabled": predictability_enabled,
                    "targets": predictability_targets,
                },
                "reconstruction_grid": {
                    "enabled": reconstruction_grid_enabled,
                    "uses_randomness": (
                        reconstruction_grid_enabled
                    ),
                    "random_state": (
                        reconstruction_grid_params.get(
                            "random_state"
                        )
                        if reconstruction_grid_enabled
                        else None
                    ),
                },
            },
        },
    }


def write_runtime_environment(
    *,
    output_path: Path | str,
    stage: RuntimeEnvironmentStage,
    run_name: str,
    workflow_context: Mapping[str, Any],
) -> Path:
    """Collect and write a BenchRep runtime-environment record."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    environment = collect_runtime_environment(
        stage=stage,
        run_name=run_name,
        workflow_context=workflow_context,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(
            environment,
            handle,
            Dumper=_RuntimeEnvironmentDumper,
            sort_keys=False,
        )

    return output_path


def _get_distribution_version(distribution_name: str) -> str:
    try:
        return distribution_version(distribution_name)
    except PackageNotFoundError:
        return "unknown"


def _get_os_distribution() -> dict[str, str | None]:
    try:
        distribution = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        distribution = {}

    return {
        "name": distribution.get("NAME"),
        "version_id": distribution.get("VERSION_ID"),
        "pretty_name": distribution.get("PRETTY_NAME"),
    }


def _collect_python_packages() -> list[dict[str, str]]:
    packages: dict[tuple[str, str], dict[str, str]] = {}

    for distribution in distributions():
        name = distribution.metadata.get("Name")

        if not name:
            continue

        normalized_name = re.sub(
            r"[-_.]+",
            "-",
            name,
        ).casefold()

        package = {
            "name": normalized_name,
            "version": distribution.version,
        }

        packages[(normalized_name, distribution.version)] = package

    return sorted(
        packages.values(),
        key=lambda package: (
            package["name"],
            package["version"],
        ),
    )


def _collect_git_provenance() -> tuple[dict[str, Any], list[str]]:
    repository_root = _run_git_command(
        "rev-parse",
        "--show-toplevel",
    )

    if repository_root is None:
        return (
            {
                "available": False,
                "commit": None,
                "branch": None,
                "tag": None,
                "dirty": None,
            },
            ["BenchRep Git repository metadata was not available."],
        )

    commit = _run_git_command("rev-parse", "HEAD")
    branch = _run_git_command(
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
    )
    tag = _run_git_command(
        "describe",
        "--tags",
        "--exact-match",
        "HEAD",
    )
    status = _run_git_command(
        "status",
        "--porcelain",
        "--untracked-files=normal",
    )

    warnings = []

    if commit is None:
        warnings.append("The BenchRep Git commit could not be determined.")

    if status is None:
        warnings.append("The BenchRep Git working-tree status could not be determined.")

    return (
        {
            "available": True,
            "commit": commit,
            "branch": branch or None,
            "tag": tag or None,
            "dirty": None if status is None else bool(status),
        },
        warnings,
    )


def _run_git_command(*arguments: str) -> str | None:
    source_dir = Path(__file__).resolve().parent

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(source_dir),
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def _collect_hardware() -> tuple[dict[str, Any], list[str]]:
    warnings = []

    cuda_available = torch.cuda.is_available()
    cuda_device_count = (
        torch.cuda.device_count()
        if cuda_available
        else 0
    )
    cuda_devices = []

    if cuda_available:
        for device_index in range(cuda_device_count):
            try:
                properties = torch.cuda.get_device_properties(
                    device_index
                )

                cuda_devices.append(
                    {
                        "index": device_index,
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "compute_capability": (
                            f"{properties.major}.{properties.minor}"
                        ),
                    }
                )
            except Exception as error:
                warnings.append(
                    "CUDA device "
                    f"{device_index} properties could not be collected: "
                    f"{type(error).__name__}: {error}"
                )

    mps_backend = getattr(torch.backends, "mps", None)

    return (
        {
            "cpu": {
                "model": _get_cpu_model(),
                "logical_cores_system": os.cpu_count(),
                "logical_cores_available_to_process": (
                    _get_available_logical_cpu_count()
                ),
            },
            "memory": {
                "total_system_memory_bytes": (
                    _get_total_system_memory_bytes()
                ),
            },
            "accelerators": {
                "cuda": {
                    "available": cuda_available,
                    "device_count": cuda_device_count,
                    "devices": cuda_devices,
                },
                "mps": {
                    "built": (
                        mps_backend.is_built()
                        if mps_backend is not None
                        else False
                    ),
                    "available": (
                        mps_backend.is_available()
                        if mps_backend is not None
                        else False
                    ),
                },
            },
        },
        warnings,
    )


def _get_cpu_model() -> str | None:
    cpuinfo_path = Path("/proc/cpuinfo")

    if cpuinfo_path.is_file():
        try:
            for line in cpuinfo_path.read_text(
                encoding="utf-8"
            ).splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", maxsplit=1)[1].strip()
        except (OSError, UnicodeError, IndexError):
            pass

    return platform.processor() or None


def _get_available_logical_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


def _get_total_system_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None

    return int(page_size * physical_pages)


def _collect_accelerator_stack() -> tuple[dict[str, Any], list[str]]:
    warnings = []

    driver_versions = _get_nvidia_driver_versions()

    if torch.cuda.is_available() and not driver_versions:
        warnings.append(
            "CUDA is available through PyTorch, but the NVIDIA driver "
            "version could not be collected."
        )

    cudnn_version_number = torch.backends.cudnn.version()
    cudnn_version = _format_cudnn_version(
        cudnn_version_number
    )

    return (
        {
            "pytorch": {
                "version": str(torch.__version__),
                "git_version": _string_or_none(
                    getattr(torch.version, "git_version", None)
                ),
                "cuda_build_version": _string_or_none(
                    torch.version.cuda
                ),
                "rocm_build_version": _string_or_none(
                    getattr(torch.version, "hip", None)
                ),
                "build_configuration": _LiteralString(
                    _get_pytorch_build_configuration(
                        cudnn_version=cudnn_version,
                    )
                ),
            },
            "cudnn": {
                "available": torch.backends.cudnn.is_available(),
                "version": cudnn_version,
                "version_number": cudnn_version_number,
            },
            "nvidia": {
                "driver_versions": driver_versions,
                "cuda_toolkit_version_on_path": (
                    _get_cuda_toolkit_version_on_path()
                ),
            },
        },
        warnings,
    )


def _get_nvidia_driver_versions() -> list[str]:
    output = _run_system_command(
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    )

    if output is None:
        return []

    return sorted(
        {
            line.strip()
            for line in output.splitlines()
            if line.strip()
        }
    )


def _get_cuda_toolkit_version_on_path() -> str | None:
    output = _run_system_command("nvcc", "--version")

    if output is None:
        return None

    match = re.search(
        r"\brelease\s+([0-9.]+)",
        output,
    )

    return match.group(1) if match is not None else None


def _run_system_command(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            list(arguments),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _format_cudnn_version(
    version_number: int | None,
) -> str | None:
    if version_number is None:
        return None

    if version_number >= 90000:
        major = version_number // 10000
        minor = (version_number % 10000) // 100
    else:
        major = version_number // 1000
        minor = (version_number % 1000) // 100

    patch = version_number % 100

    return f"{major}.{minor}.{patch}"


def _get_pytorch_build_configuration(
    *,
    cudnn_version: str | None,
) -> str:
    configuration = torch.__config__.show().strip()

    if cudnn_version is None:
        return configuration

    return re.sub(
        r"(?m)^(\s*-\s*CuDNN)\s+\S+\s*$",
        lambda match: (
            f"{match.group(1)} {cudnn_version}"
        ),
        configuration,
    )


def _collect_torch_runtime_state() -> dict[str, Any]:
    return {
        "initial_seed": torch.initial_seed(),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
        ),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "default_dtype": str(torch.get_default_dtype()),
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }


def _collect_trainer_runtime_state(
    trainer: L.Trainer,
) -> dict[str, Any]:
    return {
        "accelerator": type(trainer.accelerator).__name__,
        "strategy": type(trainer.strategy).__name__,
        "root_device": str(trainer.strategy.root_device),
        "device_ids": list(trainer.device_ids),
        "num_devices": trainer.num_devices,
        "num_nodes": trainer.num_nodes,
        "world_size_at_capture": trainer.world_size,
        "precision": str(trainer.precision),
    }


def _collect_enabled_random_state(
    *,
    enabled: bool,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "random_state": (
            params.get("random_state")
            if enabled
            else None
        ),
    }
