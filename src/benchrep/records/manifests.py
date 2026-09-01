from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

from lightning.pytorch.callbacks import ModelCheckpoint

from benchrep.assembly.schemas import TrainingConfig, PredictionConfig, EvaluationConfig
from benchrep.assembly.config import ConfigCompositionResult
from benchrep.assembly.resolvers import (
    TrainingRunSpec,
    PredictionRunSpec,
    EvaluationRunSpec,
)
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
from benchrep.runtime import RunContext
from benchrep.records.runtime_environment import (
    get_runtime_environment_filename,
)
from benchrep.runtime.status import (
    EarlyStoppingRecord,
    TrainingStatusReport,
    PredictionStatusReport,
    build_outcome_summary,
    EvaluationSectionStatus,
    EvaluationStatusReport,
)


if TYPE_CHECKING:
    import anndata as ad


def write_training_manifest(
    *,
    config_composition_result: ConfigCompositionResult[TrainingConfig],
    output_path: Path,
    run_spec: TrainingRunSpec,
    run_context: RunContext,
    checkpoint_callback: ModelCheckpoint,
    early_stopping_record: EarlyStoppingRecord | None,
    torchview_graph_path: Path | None = None,
    created_at: str,
    completed_at: str,
    status_report: TrainingStatusReport,
    model_class_name: str,
    datamodule_class_name: str,
) -> dict[str, Any]:
    # Source flags
    model_is_external = run_spec.model_source != "config"
    datamodule_is_external = run_spec.datamodule_source != "config"

    config = run_spec.training_config

    configured_model = config.model.name if config.model is not None else None
    configured_encoder = config.encoder.name if config.encoder is not None else None
    configured_decoder = config.decoder.name if config.decoder is not None else None

    configured_dataset = (
        config.dataset.model_dump(mode="json")
        if not datamodule_is_external and config.dataset is not None
        else None
    )

    configured_transforms = (
        [
            transform.model_dump(mode="json")
            for transform in config.transforms
        ]
        if not datamodule_is_external
        else None
    )

    transform_names_by_split = (
        {
            "training": [
                transform.name
                for transform in config.transforms
                if "training" in transform.apply_to
            ],
            "validation": [
                transform.name
                for transform in config.transforms
                if "validation" in transform.apply_to
            ],
        }
        if not datamodule_is_external
        else None
    )

    configured_datamodule = (
        config.datamodule.model_dump(mode="json")
        if not datamodule_is_external and config.datamodule is not None
        else None
    )

    records = _build_common_records(
        config_composition_result,
        run_context,
    )
    records["architecture"] = {
        "torchview_requested": config.inspection.torchview.enabled,
        "torchview_graph_path": paths_to_strings(torchview_graph_path),
    }

    early_stopping = (
        {
            "triggered": early_stopping_record.triggered,
            "reason": early_stopping_record.reason,
            "monitor": early_stopping_record.monitor,
            "stopped_epoch": early_stopping_record.stopped_epoch,
            "best_score": early_stopping_record.best_score,
            "wait_count": early_stopping_record.wait_count,
        }
        if early_stopping_record is not None
        else None
    )

    summary = {
        "model_source": run_spec.model_source,
        "datamodule_source": run_spec.datamodule_source,
        "model": model_class_name if model_is_external else configured_model,
        "model_family": run_spec.model_family.name,
        "encoder": None if model_is_external else configured_encoder,
        "decoder": None if model_is_external else configured_decoder,
        "dataset": (
            configured_dataset["name"]
            if configured_dataset is not None
            else None
        ),
        "transforms": transform_names_by_split,
        "datamodule": datamodule_class_name if datamodule_is_external else None,
        "batch_size": (
            configured_datamodule.get("batch_size")
            if configured_datamodule is not None
            else None
        ),
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
        "save_top_k": config.checkpointing.save_top_k,
        "save_last": config.checkpointing.save_last,
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

    outcome_summary = build_outcome_summary(
        (status_report.status,)
    )

    manifest = {
        "stage": config.stage,
        "status": status_report.status,
        "status_report": {
            "training": {
                "status": status_report.status,
                "issues": list(status_report.issues),
                "interruption_signal": status_report.interruption_signal,
            },
        },
        "outcome_summary": outcome_summary,
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
                "source": run_spec.model_source,
                "family": run_spec.model_family.name,
                "class_name": model_class_name,
                "config_reconstructable": not model_is_external,
                "configured_model": None if model_is_external else configured_model,
                "configured_encoder": None if model_is_external else configured_encoder,
                "configured_decoder": None if model_is_external else configured_decoder,
            },
            "dataset": configured_dataset,
            "transforms": configured_transforms,
            "datamodule": {
                "source": run_spec.datamodule_source,
                "class_name": datamodule_class_name,
                "config_reconstructable": (
                        not datamodule_is_external
                        and configured_dataset is not None
                        and configured_datamodule is not None
                ),
                "configured_datamodule": configured_datamodule,
            },
        },
        "records": records,
        "early_stopping": early_stopping,
        "checkpoints": checkpoints,
        "summary": summary,
    }

    write_yaml_record(manifest, output_path)

    return manifest


def write_prediction_manifest(
    *,
    config_composition_result: ConfigCompositionResult[PredictionConfig],
    output_path: Path,
    run_spec: PredictionRunSpec,
    run_context: RunContext,
    export_paths: PredictionExportPaths,
    created_at: str,
    completed_at: str,
    status_report: PredictionStatusReport,
    model_class_name: str,
    datamodule_class_name: str,
    n_batches: int,
    n_observations: int | None,
) -> dict[str, Any]:
    model_family = run_spec.model_family
    model_source = run_spec.model_source
    datamodule_source = run_spec.datamodule_source
    training_provenance = run_spec.training_manifest.get("provenance", {})
    training_status = run_spec.training_manifest.get("status")
    training_status_report = run_spec.training_manifest.get("status_report")
    training_status_record = (
        training_status_report.get("training")
        if isinstance(training_status_report, Mapping)
        else None
    )
    training_interruption_signal = (
        training_status_record.get("interruption_signal")
        if isinstance(training_status_record, Mapping)
        else None
    )
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
        run_spec.dataset_config.model_dump(mode="json")
        if not datamodule_is_external and run_spec.dataset_config is not None
        else None
    )

    configured_transforms = (
        [
            transform.model_dump(mode="json")
            for transform in run_spec.transform_configs
        ]
        if not datamodule_is_external
        and run_spec.transform_configs is not None
        else None
    )

    configured_datamodule = (
        run_spec.datamodule_config.model_dump(mode="json")
        if not datamodule_is_external and run_spec.datamodule_config is not None
        else None
    )

    if configured_datamodule is not None:
        configured_datamodule["batch_size"] = run_spec.batch_size


    embedding_spec = run_spec.export_spec.embeddings
    reconstruction_spec = run_spec.export_spec.reconstructions
    embedding_export = export_paths.embedding_export
    reconstruction_paths = export_paths.reconstruction_paths

    records = _build_common_records(
        config_composition_result,
        run_context,
    )

    summary = {
        "project_name": run_spec.run_identity.project_name,
        "model_source": model_source,
        "datamodule_source": datamodule_source,
        "model": model_class_name if model_is_external else configured_model,
        "model_family": model_family.name,
        "encoder": None if model_is_external else configured_encoder,
        "decoder": None if model_is_external else configured_decoder,
        "dataset": (
            configured_dataset["name"] if configured_dataset is not None else None
        ),
        "transforms": (
            {
                "prediction": [
                    transform["name"]
                    for transform in configured_transforms
                ],
            }
            if configured_transforms is not None
            else None
        ),
        "datamodule": datamodule_class_name if datamodule_is_external else None,
        "batch_size": (
            configured_datamodule.get("batch_size") if configured_datamodule is not None else None
        ),
        "max_batches": run_spec.max_batches,
    }

    outcome_summary = build_outcome_summary(
        outcome.status
        for outcome in (
            status_report.inference,
            status_report.embeddings_export,
            status_report.reconstructions_export,
        )
    )

    manifest = {
        "stage": run_spec.stage,
        "status": status_report.status,
        "status_report": {
            "inference": {
                "status": status_report.inference.status,
                "issues": list(status_report.inference.issues),
                "n_batches": n_batches,
                "n_observations": n_observations,
            },
            "exports": {
                "embeddings": {
                    "status": status_report.embeddings_export.status,
                    "issues": list(
                        status_report.embeddings_export.issues
                    ),
                },
                "reconstructions": {
                    "status": status_report.reconstructions_export.status,
                    "issues": list(
                        status_report.reconstructions_export.issues
                    ),
                    "n_strata": (
                        reconstruction_paths.n_strata
                        if reconstruction_paths is not None
                        else None
                    ),
                    "n_represented_strata": (
                        reconstruction_paths.n_represented_strata
                        if reconstruction_paths is not None
                        else None
                    ),
                    "n_omitted_strata": (
                        reconstruction_paths.n_omitted_strata
                        if reconstruction_paths is not None
                        else None
                    ),
                },
            },
        },
        "outcome_summary": outcome_summary,
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
            "training_status": training_status,
            "training_interruption_signal": training_interruption_signal,
            "resolved_training_config_path": str(run_spec.resolved_training_config_path),
            "checkpoint_selection": str(run_spec.checkpoint_selection),
            "checkpoint_source": run_spec.checkpoint_source,
            "checkpoint_path": str(run_spec.checkpoint_path),
        },
        "config_inheritance": {
            "inherited_from_training": sorted(
                run_spec.inherited_config_fields,
            ),
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
                    "family": model_family.name,
                    "class_name": model_class_name,
                    "config_reconstructable": (
                            not model_is_external and configured_model is not None
                    ),
                    "configured_model": None if model_is_external else configured_model,
                    "configured_encoder": None if model_is_external else configured_encoder,
                    "configured_decoder": None if model_is_external else configured_decoder,
                },
                "inference": (
                    run_spec.prediction_config.inference.model_dump(
                        mode="json",
                    )
                ),
                "dataset": configured_dataset,
                "transforms": configured_transforms,
                "datamodule": {
                    "source": datamodule_source,
                    "class_name": datamodule_class_name,
                    "config_reconstructable": (
                            not datamodule_is_external
                            and configured_dataset is not None
                            and configured_datamodule is not None
                    ),
                    "configured_datamodule": configured_datamodule,
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
                "configured_enabled": (
                    run_spec.prediction_config.exports.reconstructions.enabled
                ),
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

    return manifest


def write_evaluation_manifest(
    *,
    config_composition_result: ConfigCompositionResult[EvaluationConfig],
    output_path: Path,
    run_spec: EvaluationRunSpec,
    run_context: RunContext,
    adata: ad.AnnData | None,
    export_paths: EvaluationExportPaths,
    status_report: EvaluationStatusReport,
    created_at: str,
    completed_at: str,
) -> dict[str, Any]:
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

    if run_spec.input_spec.embeddings_path is None:
        embeddings_source = None
    elif config.source.embeddings_path is not None:
        embeddings_source = "direct_path"
    else:
        embeddings_source = "prediction_manifest"

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

    anndata_output_locations = (
        _build_evaluation_anndata_output_locations(
            adata=adata,
            run_spec=run_spec,
        )
        if adata is not None
        else {}
    )

    outcome_statuses = [
        outcome.status
        for section in (
            status_report.embeddings,
            status_report.reconstructions,
            status_report.exports,
        )
        for outcome in section.outcomes
    ]

    if status_report.fatal_issue is not None:
        outcome_statuses.append("failed")

    manifest = {
        "stage": run_spec.stage,
        "status": status_report.status,
        "status_report": _evaluation_status_report_to_manifest(
            status_report,
            anndata_output_locations=anndata_output_locations,
        ),
        "outcome_summary": build_outcome_summary(outcome_statuses),
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
                "path": paths_to_strings(
                    run_spec.input_spec.embeddings_path
                ),
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
                "path": paths_to_strings(
                    export_paths.evaluated_embeddings_path
                ),
                "n_obs": (
                    int(adata.n_obs)
                    if export_paths.evaluated_embeddings_path is not None
                       and adata is not None
                    else None
                ),
                "n_vars": (
                    int(adata.n_vars)
                    if export_paths.evaluated_embeddings_path is not None
                       and adata is not None
                    else None
                ),
            },
            "metrics": {
                "path": paths_to_strings(export_paths.metrics_json_path),
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
            "has_embeddings": run_spec.input_spec.embeddings_path is not None,
            "has_reconstructions": reconstruction_spec is not None,
        },
    }

    write_yaml_record(manifest, output_path)

    return manifest


def _evaluation_status_report_to_manifest(
    status_report: EvaluationStatusReport,
    *,
    anndata_output_locations: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Convert active evaluation outcomes to a YAML-safe mapping."""

    result: dict[str, Any] = {}

    section_specs = (
        (
            "embeddings",
            status_report.embeddings,
            anndata_output_locations,
        ),
        (
            "reconstructions",
            status_report.reconstructions,
            None,
        ),
        (
            "exports",
            status_report.exports,
            None,
        ),
    )

    for section_name, section, output_locations in section_specs:
        section_record = _status_section_to_manifest(
            section,
            output_locations=output_locations,
        )

        if section_record is not None:
            result[section_name] = section_record

    if status_report.fatal_issue is not None:
        result["fatal_issue"] = status_report.fatal_issue

    return result


def _status_section_to_manifest(
    section: EvaluationSectionStatus,
    *,
    output_locations: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Convert one active evaluation status section to a mapping."""

    outcomes: dict[str, Any] = {}
    seen_names: set[str] = set()

    for outcome in section.outcomes:
        if outcome.name in seen_names:
            raise ValueError(
                "Evaluation status section contains duplicate outcome "
                f"name {outcome.name!r}."
            )

        seen_names.add(outcome.name)

        if outcome.status == "disabled":
            continue

        outcome_record: dict[str, Any] = {
            "status": outcome.status,
            "issues": list(outcome.issues),
        }

        if outcome.status in {
            "completed",
            "completed_with_warnings",
        }:
            outputs = (
                output_locations.get(outcome.name)
                if output_locations is not None
                else None
            )

            if outputs:
                outcome_record["outputs"] = dict(outputs)

        outcomes[outcome.name] = outcome_record

    if not outcomes:
        return None

    return {
        "status": section.status,
        "outcomes": outcomes,
    }


def _build_evaluation_anndata_output_locations(
    *,
    adata: ad.AnnData,
    run_spec: EvaluationRunSpec,
) -> dict[str, dict[str, str]]:
    """Map evaluation outcomes to outputs present in the final AnnData."""

    step_spec = run_spec.step_spec
    reduction_keys = {
        "pca": step_spec.pca_params.get("key_added", "X_pca"),
        "umap": step_spec.umap_params.get("key_added", "X_umap"),
        "tsne": step_spec.tsne_params.get("key_added", "X_tsne"),
    }
    clustering_keys = {
        "kmeans": step_spec.kmeans_params.get("key_added", "kmeans"),
        "leiden": step_spec.leiden_params.get("key_added", "leiden"),
        "hdbscan": step_spec.hdbscan_params.get(
            "key_added",
            "hdbscan",
        ),
    }
    current_cluster_keys = set(clustering_keys.values())

    benchrep = adata.uns.get("benchrep", {})
    if not isinstance(benchrep, Mapping):
        return {}

    output_locations: dict[str, dict[str, str]] = {}

    reductions = benchrep.get("reductions", {})
    if isinstance(reductions, Mapping):
        for key_added, metadata in reductions.items():
            if (
                not isinstance(key_added, str)
                or not isinstance(metadata, Mapping)
                or key_added not in adata.obsm
            ):
                continue

            method = metadata.get("method")
            if (
                not isinstance(method, str)
                or reduction_keys.get(method) != key_added
            ):
                continue

            outputs = {
                "coordinates": _anndata_location("obsm", key_added),
                "metadata": _anndata_location(
                    "uns",
                    "benchrep",
                    "reductions",
                    key_added,
                ),
            }

            neighbors_key = metadata.get("neighbors_key")
            if isinstance(neighbors_key, str):
                outputs.update(
                    _neighbor_output_locations(adata, neighbors_key)
                )

            output_locations[method] = outputs

    clustering = benchrep.get("clustering", {})
    if isinstance(clustering, Mapping):
        for cluster_key, metadata in clustering.items():
            if (
                not isinstance(cluster_key, str)
                or not isinstance(metadata, Mapping)
                or cluster_key not in adata.obs
            ):
                continue

            method = metadata.get("method")
            if (
                not isinstance(method, str)
                or clustering_keys.get(method) != cluster_key
            ):
                continue

            outputs = {
                "clusters": _anndata_location("obs", cluster_key),
                "metadata": _anndata_location(
                    "uns",
                    "benchrep",
                    "clustering",
                    cluster_key,
                ),
            }

            probability_key = metadata.get("probability_key")
            if (
                isinstance(probability_key, str)
                and probability_key in adata.obs
            ):
                outputs["probabilities"] = _anndata_location(
                    "obs",
                    probability_key,
                )

            neighbors_key = metadata.get("neighbors_key")
            if isinstance(neighbors_key, str):
                outputs.update(
                    _neighbor_output_locations(adata, neighbors_key)
                )

            output_locations[method] = outputs

    metrics = benchrep.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return output_locations

    if "embedding" in metrics:
        output_locations["embedding_metrics"] = {
            "metrics": _anndata_location(
                "uns",
                "benchrep",
                "metrics",
                "embedding",
            )
        }

    predictability = metrics.get("predictability", {})
    if isinstance(predictability, Mapping):
        for target_spec in step_spec.predictability_targets:
            target_key = target_spec.target_key

            if target_key not in predictability:
                continue

            output_locations[
                f"predictability_metrics_{target_key}"
            ] = {
                "metrics": _anndata_location(
                    "uns",
                    "benchrep",
                    "metrics",
                    "predictability",
                    target_key,
                )
            }

    clustering_metrics = metrics.get("clustering", {})
    if isinstance(clustering_metrics, Mapping):
        for metric_group in ("internal", "external"):
            group_results = clustering_metrics.get(metric_group, {})
            if not isinstance(group_results, Mapping):
                continue

            for cluster_key in group_results:
                if cluster_key in current_cluster_keys:
                    output_locations[
                        f"{metric_group}_clustering_metrics_{cluster_key}"
                    ] = {
                        "metrics": _anndata_location(
                            "uns",
                            "benchrep",
                            "metrics",
                            "clustering",
                            metric_group,
                            cluster_key,
                        )
                    }

    return output_locations


def _neighbor_output_locations(
    adata: ad.AnnData,
    neighbors_key: str,
) -> dict[str, str]:
    outputs: dict[str, str] = {}

    if neighbors_key in adata.uns:
        outputs["neighbor_metadata"] = _anndata_location(
            "uns",
            neighbors_key,
        )

    prefix = "" if neighbors_key == "neighbors" else f"{neighbors_key}_"

    for output_name in ("distances", "connectivities"):
        key = f"{prefix}{output_name}"

        if key in adata.obsp:
            outputs[output_name] = _anndata_location("obsp", key)

    return outputs


def _anndata_location(attribute: str, *keys: str) -> str:
    return f"adata.{attribute}" + "".join(
        f"[{key!r}]"
        for key in keys
    )


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
        "runtime_environment_path": str(
            run_context.metadata_dir
            / get_runtime_environment_filename(run_context.stage)
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