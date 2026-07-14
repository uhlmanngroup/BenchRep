from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any

import yaml

from lightning.pytorch.callbacks import ModelCheckpoint

from benchrep.assembly.schemas import TrainingConfig, PredictionConfig, EvaluationConfig
from benchrep.assembly.config import ConfigCompositionResult
from benchrep.assembly.resolvers import PredictionRunSpec, EvaluationRunSpec
from benchrep.runtime import RunContext
from benchrep.records.prediction_exports import PredictionExportPaths
from benchrep.records.evaluation_exports import EvaluationExportPaths
from benchrep.records.logs import (
    RUN_LOG_FILENAME,
    STDERR_LOG_FILENAME,
    STDOUT_LOG_FILENAME,
)
from benchrep.records.utils import (
    paths_to_strings,
    count_paths,
)


def write_training_manifest(
    *,
    config_composition_result: ConfigCompositionResult[TrainingConfig],
    output_path: Path,
    run_context: RunContext,
    checkpoint_callback: ModelCheckpoint,
    torchview_graph_path: Path | None = None,
    created_at: str,
    completed_at: str,
    status: str = "completed",
    model_source: str = "config",
    model_class_name: str,
    datamodule_source: str = "config",
    datamodule_class_name: str,
) -> None:
    # Source flags
    model_is_external = model_source != "config"
    datamodule_is_external = datamodule_source != "config"

    config = config_composition_result.effective_config

    configured_model = config.model.name if config.model is not None else None
    configured_encoder = config.encoder.name if config.encoder is not None else None
    configured_decoder = config.decoder.name if config.decoder is not None else None

    configured_dataset = config.dataset.name if config.dataset is not None else None
    configured_transform = (
        config.dataset.transform.name
        if config.dataset is not None and config.dataset.transform is not None
        else None
    )
    configured_batch_size = (
        config.datamodule.batch_size
        if not datamodule_is_external and config.datamodule is not None
        else None
    )

    records = _build_common_records(
        config_composition_result,
        run_context,
    )
    records["architecture"] = {
        "torchview_graph_path": paths_to_strings(torchview_graph_path),
    }

    summary = {
        "model_source": model_source,
        "datamodule_source": datamodule_source,
        "model": model_class_name if model_is_external else configured_model,
        "encoder": None if model_is_external else configured_encoder,
        "decoder": None if model_is_external else configured_decoder,
        "dataset": None if datamodule_is_external else configured_dataset,
        "datamodule": datamodule_class_name if datamodule_is_external else None,
        "transform": None if datamodule_is_external else configured_transform,
        "batch_size": None if datamodule_is_external else configured_batch_size,
        "losses": (
            None
            if model_is_external or config.losses is None
            else {
                role: list(loss_terms)
                for role, loss_terms in config.losses.items()
            }
        ),
        "optimizer": (
            None
            if model_is_external or config.optimizer is None
            else config.optimizer.name
        ),
        "seed": config.reproducibility.seed,
        "float32_matmul_precision": config.reproducibility.float32_matmul_precision,
        "trainer": {
            "max_epochs": getattr(config.trainer, "max_epochs", None),
            "deterministic": getattr(config.trainer, "deterministic", None),
        },
        "logger": {
            "name": config.logger.name if config.logger is not None else None,
        },
    }

    checkpoints = {
        "checkpoint_dir": (
            str(Path(checkpoint_callback.dirpath))
            if checkpoint_callback.dirpath is not None
            else str(run_context.training_checkpoint_dir)
        ),
        "monitor": config.checkpointing.monitor,
        "mode": config.checkpointing.mode,
        "best_checkpoint_path": checkpoint_callback.best_model_path or None,
        "best_checkpoint_score": (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None
        ),
        "last_checkpoint_path": checkpoint_callback.last_model_path or None,
        "best_k_models": [
            {
                "path": path,
                "score": float(score),
            }
            for path, score in checkpoint_callback.best_k_models.items()
        ],
    }

    manifest = {
        "stage": config.stage,
        "status": status,
        "created_at": created_at,
        "completed_at": completed_at,
        "run": {
            "run_name": run_context.run_name,
            "output_dir": str(run_context.output_dir),
        },
        "provenance": {
            "config": {
                "run_reconstructable_from_resolved_config": (
                    not model_is_external and not datamodule_is_external
                ),
                "effective_source": config_composition_result.effective_source,
                "yaml_supplied": config_composition_result.yaml_supplied,
                "yaml_used_as_base": config_composition_result.yaml_used_as_base,
                "original_config_path": (
                    str(config_composition_result.original_config_path)
                    if config_composition_result.original_config_path is not None
                    else None
                ),
            },
            "model": {
                "source": model_source,
                "class_name": model_class_name,
                "config_reconstructable": not model_is_external,
                "configured_model": None if model_is_external else configured_model,
                "configured_encoder": None if model_is_external else configured_encoder,
                "configured_decoder": None if model_is_external else configured_decoder,
            },
            "datamodule": {
                "source": datamodule_source,
                "class_name": datamodule_class_name,
                "config_reconstructable": not datamodule_is_external,
                "configured_dataset": None if datamodule_is_external else configured_dataset,
                "configured_transform": None if datamodule_is_external else configured_transform,
                "configured_batch_size": None if datamodule_is_external else configured_batch_size,
            },
        },
        "records": records,
        "checkpoints": checkpoints,
        "summary": summary,
    }

    write_yaml_record(manifest, output_path)


def write_prediction_manifest(
    *,
    config_composition_result: ConfigCompositionResult[PredictionConfig],
    output_path: Path,
    run_spec: PredictionRunSpec,
    run_context: RunContext,
    export_paths: PredictionExportPaths,
    created_at: str,
    completed_at: str,
    status: str = "completed",
    model_source: str = "config",
    model_class_name: str,
    datamodule_source: str = "config",
    datamodule_class_name: str,
) -> None:
    training_provenance = run_spec.training_manifest.get("provenance", {})
    training_config_provenance = training_provenance.get("config", {})
    training_run_reconstructable = bool(
        training_config_provenance.get(
            "run_reconstructable_from_resolved_config",
            False,
        )
    )

    model_is_external = model_source != "config"
    datamodule_is_external = datamodule_source != "config"

    configured_model = (
        run_spec.training_config.model.name
        if run_spec.training_config.model is not None
        else None
    )
    configured_encoder = (
        run_spec.training_config.encoder.name
        if run_spec.training_config.encoder is not None
        else None
    )
    configured_decoder = (
        run_spec.training_config.decoder.name
        if run_spec.training_config.decoder is not None
        else None
    )

    configured_dataset = (
        run_spec.dataset_config.name
        if run_spec.dataset_config is not None
        else None
    )
    configured_transform = (
        run_spec.dataset_config.transform.name
        if (
            run_spec.dataset_config is not None
            and run_spec.dataset_config.transform is not None
        )
        else None
    )
    configured_batch_size = (
        run_spec.batch_size if not datamodule_is_external else None
    )
    configured_split = (
        run_spec.split if not datamodule_is_external else None
    )

    embedding_spec = run_spec.export_spec.embeddings
    reconstruction_spec = run_spec.export_spec.reconstructions
    embedding_export = export_paths.embedding_export
    reconstruction_paths = export_paths.reconstruction_paths

    records = _build_common_records(
        config_composition_result,
        run_context,
    )

    summary = {
        "project_name": run_spec.training_config.run.project_name,
        "model_source": model_source,
        "datamodule_source": datamodule_source,
        "model": model_class_name if model_is_external else configured_model,
        "encoder": None if model_is_external else configured_encoder,
        "decoder": None if model_is_external else configured_decoder,
        "dataset": None if datamodule_is_external else configured_dataset,
        "datamodule": datamodule_class_name if datamodule_is_external else None,
        "data_split": configured_split,
        "batch_size": configured_batch_size,
        "max_batches": run_spec.max_batches,
    }

    manifest = {
        "stage": run_spec.stage,
        "status": status,
        "created_at": created_at,
        "completed_at": completed_at,
        "run": {
            "run_name": run_context.run_name,
            "output_dir": str(run_context.output_dir),
        },
        "source": {
            "training_manifest_path": str(run_spec.training_manifest_path),
            "training_run_name": run_spec.training_run_name,
            "training_output_dir": str(run_spec.training_output_dir),
            "resolved_training_config_path": str(run_spec.resolved_training_config_path),
            "checkpoint_path": str(run_spec.checkpoint_path),
        },
        "provenance": {
            "training": training_provenance,
            "prediction": {
                "config": {
                    "run_reconstructable_from_resolved_config": (
                        training_run_reconstructable
                        and not model_is_external
                        and not datamodule_is_external
                    ),
                    "effective_source": config_composition_result.effective_source,
                    "yaml_supplied": config_composition_result.yaml_supplied,
                    "yaml_used_as_base": config_composition_result.yaml_used_as_base,
                    "original_config_path": (
                        str(config_composition_result.original_config_path)
                        if config_composition_result.original_config_path is not None
                        else None
                    ),
                },
                "model": {
                    "source": model_source,
                    "class_name": model_class_name,
                    "config_reconstructable": (
                            not model_is_external and configured_model is not None
                    ),
                    "configured_model": None if model_is_external else configured_model,
                    "configured_encoder": None if model_is_external else configured_encoder,
                    "configured_decoder": None if model_is_external else configured_decoder,
                },
                "datamodule": {
                    "source": datamodule_source,
                    "class_name": datamodule_class_name,
                    "config_reconstructable": (
                            not datamodule_is_external
                            and configured_dataset is not None
                            and run_spec.datamodule_config is not None
                    ),
                    "configured_dataset": None if datamodule_is_external else configured_dataset,
                    "configured_transform": None if datamodule_is_external else configured_transform,
                    "configured_batch_size": None if datamodule_is_external else configured_batch_size,
                    "configured_split": configured_split,
                },
            },
        },
        "records": records,
        "exports": {
            "mode": run_spec.export_spec.mode,
            "embeddings": {
                "enabled": embedding_spec.enabled,
                "requested_keys": embedding_spec.keys,
                "requested_primary_key": embedding_spec.primary_key,
                "path": (
                    paths_to_strings(embedding_export.embeddings_h5ad_path)
                    if embedding_export is not None
                    else None
                ),
                "resolved_keys": (
                    embedding_export.resolved_keys
                    if embedding_export is not None
                    else None
                ),
                "resolved_primary_key": (
                    embedding_export.resolved_primary_key
                    if embedding_export is not None
                    else None
                ),
            },
            "reconstructions": {
                "enabled": reconstruction_spec.enabled,
                "n_examples_requested": reconstruction_spec.n_examples,
                "n_examples_exported": (
                    reconstruction_paths.n_examples_exported
                    if reconstruction_paths is not None
                    else None
                ),
                "selection": reconstruction_spec.selection,
                "stratify_by": reconstruction_spec.stratify_by,
                "seed": reconstruction_spec.seed,
                "include_input": reconstruction_spec.include_input,
                "include_prediction": reconstruction_spec.include_prediction,
                "paths": {
                    "input": (
                        paths_to_strings(reconstruction_paths.input_path)
                        if reconstruction_paths is not None
                        else None
                    ),
                    "reconstruction": (
                        paths_to_strings(reconstruction_paths.reconstruction_path)
                        if reconstruction_paths is not None
                        else None
                    ),
                    "obs": (
                        paths_to_strings(reconstruction_paths.obs_path)
                        if reconstruction_paths is not None
                        else None
                    ),
                    "metadata": (
                        paths_to_strings(reconstruction_paths.metadata_path)
                        if reconstruction_paths is not None
                        else None
                    ),
                },
            },
        },
        "summary": summary,
    }

    write_yaml_record(manifest, output_path)


def write_evaluation_manifest(
    *,
    config_composition_result: ConfigCompositionResult[EvaluationConfig],
    output_path: Path,
    run_spec: EvaluationRunSpec,
    run_context: RunContext,
    export_paths: EvaluationExportPaths,
    created_at: str,
    completed_at: str,
    status: str = "completed",
) -> None:
    """Write the evaluation workflow manifest."""
    config = run_spec.evaluation_config
    step_spec = run_spec.step_spec
    reconstruction_spec = run_spec.input_spec.reconstructions

    prediction_manifest = run_spec.prediction_manifest

    prediction_run: Mapping[str, Any] = {}
    prediction_provenance: Mapping[str, Any] | None = None
    prediction_summary: Mapping[str, Any] = {}

    if prediction_manifest is not None:
        run_value = prediction_manifest.get("run", {})
        if isinstance(run_value, Mapping):
            prediction_run = run_value

        provenance_value = prediction_manifest.get("provenance")
        if isinstance(provenance_value, Mapping):
            prediction_provenance = provenance_value

        summary_value = prediction_manifest.get("summary", {})
        if isinstance(summary_value, Mapping):
            prediction_summary = summary_value

    embeddings_source = (
        "direct_path"
        if config.source.embeddings_path is not None
        else "prediction_manifest"
    )

    if reconstruction_spec is None:
        reconstructions_source = None
    elif config.source.reconstructions_path is not None:
        reconstructions_source = "direct_path"
    else:
        reconstructions_source = "prediction_manifest"

    if run_spec.input_spec.prediction_manifest_path is None:
        source_mode = "direct"
    elif (
        embeddings_source == "direct_path"
        or reconstructions_source == "direct_path"
    ):
        source_mode = "mixed"
    else:
        source_mode = "prediction_manifest"

    grid_params = step_spec.plot_params.get("reconstruction_grid", {})
    if not isinstance(grid_params, Mapping):
        grid_params = {}

    tiff_paths = export_paths.reconstruction_tiff_paths
    grid_paths = export_paths.reconstruction_grid_paths

    provenance = (
        dict(prediction_provenance)
        if prediction_provenance is not None
        else {}
    )

    provenance["evaluation"] = {
        "config": {
            "run_reconstructable_from_resolved_config": True,
            "effective_source": config_composition_result.effective_source,
            "yaml_supplied": config_composition_result.yaml_supplied,
            "yaml_used_as_base": config_composition_result.yaml_used_as_base,
            "original_config_path": paths_to_strings(
                config_composition_result.original_config_path
            ),
        },
    }

    records = _build_common_records(
        config_composition_result,
        run_context,
    )

    manifest = {
        "stage": run_spec.stage,
        "status": status,
        "created_at": created_at,
        "completed_at": completed_at,
        "run": {
            "run_name": run_context.run_name,
            "output_dir": str(run_context.output_dir),
        },
        "source": {
            "mode": source_mode,
            "prediction_manifest_path": paths_to_strings(
                run_spec.input_spec.prediction_manifest_path
            ),
            "prediction_run_name": prediction_run.get("run_name"),
            "prediction_output_dir": prediction_run.get("output_dir"),
            "embeddings": {
                "source": embeddings_source,
                "path": str(run_spec.input_spec.embeddings_path),
            },
            "reconstructions": {
                "source": reconstructions_source,
                "available": reconstruction_spec is not None,
                "n_examples_limit": (
                    reconstruction_spec.n_examples
                    if reconstruction_spec is not None
                    else None
                ),
                "paths": {
                    "input": (
                        paths_to_strings(reconstruction_spec.input_path)
                        if reconstruction_spec is not None
                        else None
                    ),
                    "reconstruction": (
                        paths_to_strings(
                            reconstruction_spec.reconstruction_path
                        )
                        if reconstruction_spec is not None
                        else None
                    ),
                    "obs": (
                        paths_to_strings(reconstruction_spec.obs_path)
                        if reconstruction_spec is not None
                        else None
                    ),
                    "metadata": (
                        paths_to_strings(reconstruction_spec.metadata_path)
                        if reconstruction_spec is not None
                        else None
                    ),
                },
            },
        },
        "provenance": provenance,
        "records": records,
        "exports": {
            "embeddings": {
                "path": str(export_paths.evaluated_embeddings_path),
            },
            "metrics": {
                "path": str(export_paths.metrics_json_path),
            },
            "figures": {
                "reductions": {
                    "n_files": count_paths(
                        export_paths.reduction_plot_paths
                    ),
                    "output_dir": common_parent_path(
                        export_paths.reduction_plot_paths
                    ),
                },
                "cluster_sizes": {
                    "n_files": count_paths(
                        export_paths.cluster_size_plot_paths
                    ),
                    "output_dir": common_parent_path(
                        export_paths.cluster_size_plot_paths
                    ),
                },
            },
            "reconstructions": {
                "error_map_params": step_spec.error_map_params,
                "tiffs": {
                    "enabled": step_spec.reconstruction_tiffs_enabled,
                    "n_examples_requested": (
                        config.reconstruction.n_examples
                    ),
                    "n_examples_limit": (
                        reconstruction_spec.n_examples
                        if reconstruction_spec is not None
                        else None
                    ),
                    "n_examples_exported": (
                        _infer_reconstruction_examples_exported(tiff_paths)
                    ),
                    "n_files": count_paths(tiff_paths),
                    "output_dir": common_parent_path(tiff_paths),
                },
                "grids": {
                    "enabled": (
                        reconstruction_spec is not None
                        and step_spec.plots_enabled
                    ),
                    "include_error_maps": grid_params.get(
                        "include_error_maps",
                        True,
                    ),
                    "stratify_by": grid_params.get("stratify_by"),
                    "channel_selection": grid_params.get(
                        "channel_selection"
                    ),
                    "n_files": count_paths(grid_paths),
                    "output_dir": common_parent_path(grid_paths),
                },
            },
        },
        "summary": {
            "project_name": prediction_summary.get("project_name"),
            "model": prediction_summary.get("model"),
            "encoder": prediction_summary.get("encoder"),
            "decoder": prediction_summary.get("decoder"),
            "source_mode": source_mode,
            "has_reconstructions": reconstruction_spec is not None,
            "reductions": {
                "pca": step_spec.pca_enabled,
                "umap": step_spec.umap_enabled,
                "tsne": step_spec.tsne_enabled,
            },
            "clustering": {
                "kmeans": step_spec.kmeans_enabled,
                "leiden": step_spec.leiden_enabled,
            },
            "metrics": {
                "clustering_internal": (
                    step_spec.internal_clustering_metrics_enabled
                ),
                "clustering_external": (
                    step_spec.external_clustering_metrics_enabled
                ),
                "embedding": step_spec.embedding_metrics_enabled,
                "predictability": step_spec.predictability_enabled,
                "reconstruction": (
                    step_spec.reconstruction_metrics_enabled
                ),
            },
            "plots_enabled": step_spec.plots_enabled,
            "reconstruction_tiffs_enabled": (
                step_spec.reconstruction_tiffs_enabled
            ),
        },
    }

    write_yaml_record(manifest, output_path)


def _build_common_records(
    config_composition_result: ConfigCompositionResult[Any],
    run_context: RunContext,
) -> dict[str, Any]:
    has_original_config = (
        config_composition_result.original_config_path is not None
    )

    return {
        "input_config_path": paths_to_strings(
            config_composition_result.original_config_path
        ),
        "original_config_record_path": (
            str(run_context.config_dir / "original_config.yaml")
            if has_original_config
            else None
        ),
        "resolved_config_path": str(
            run_context.config_dir / "resolved_config.yaml"
        ),
        "log_dir": str(run_context.log_dir),
        "metadata_dir": str(run_context.metadata_dir),
        "run_log_path": str(run_context.log_dir / RUN_LOG_FILENAME),
        "console_stderr_path": str(
            run_context.log_dir / STDERR_LOG_FILENAME
        ),
        "console_stdout_path": (
            f"{run_context.log_dir / STDOUT_LOG_FILENAME} "
            "[optional; only written when stdout capture is enabled]"
        ),
    }


def write_yaml_record(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _infer_reconstruction_examples_exported(
    paths: Mapping[str, Any] | None,
) -> int | None:
    """Infer the TIFF example count from an input or prediction path list."""
    if paths is None:
        return None

    for key in ("inputs", "predictions"):
        values = paths.get(key)

        if isinstance(values, Sequence) and not isinstance(
            values,
            str | bytes,
        ):
            return len(values)

    return None


def common_parent_path(value: Any) -> str | None:
    """Return the common parent directory of one or more nested paths."""
    paths = _collect_paths(value)

    if not paths:
        return None

    parent_paths = [str(path.parent) for path in paths]

    return str(Path(os.path.commonpath(parent_paths)))


def _collect_paths(value: Any) -> list[Path]:
    """Collect Path objects recursively from nested export-path structures."""
    if isinstance(value, Path):
        return [value]

    if isinstance(value, Mapping):
        paths: list[Path] = []

        for item in value.values():
            paths.extend(_collect_paths(item))

        return paths

    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes,
    ):
        paths = []

        for item in value:
            paths.extend(_collect_paths(item))

        return paths

    return []