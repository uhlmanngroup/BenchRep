from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd
from benchrep.records.logs import get_run_logger
from benchrep.evaluation.reconstructions.data import (
    ReconstructionEvaluationInput,
    load_reconstruction_evaluation_input,
)
from benchrep.evaluation.utils import validate_adata_x
from benchrep.assembly.config import load_yaml, ConfigCompositionResult
from benchrep.runtime.run_context import RunContext
from benchrep.runtime.utils import (
    AuditItem,
    audit_existing_file,
    audit_existing_dir,
    log_audit_summary,
    audit_config_records,
    audit_resolved_config_reconstructability,
)


if TYPE_CHECKING:
    from benchrep.assembly.resolvers.evaluation_config_resolver import (
        EvaluationRunSpec,
    )
    from benchrep.records.evaluation_exports import EvaluationExportPaths


@dataclass(frozen=True, slots=True)
class EvaluateSourceInputsResult:
    """Validated evaluation source inputs needed by the evaluation runner."""

    adata_input: ad.AnnData
    reconstruction_input: ReconstructionEvaluationInput | None


def prepare_evaluate_source_inputs(
    run_spec: EvaluationRunSpec,
) -> EvaluateSourceInputsResult:
    """Load and validate source inputs required by the evaluation runner."""

    adata = _load_embeddings_adata(run_spec.input_spec.embeddings_path)
    _validate_embedding_adata_basic_contract(adata)
    _validate_enabled_step_preconditions(
        adata=adata,
        run_spec=run_spec,
    )

    reconstruction_input = None
    recon_spec = run_spec.input_spec.reconstructions

    if recon_spec is not None:
        if recon_spec.input_path is None:
            raise ValueError("Resolved reconstruction input path is None.")

        if recon_spec.reconstruction_path is None:
            raise ValueError("Resolved reconstruction prediction path is None.")

        if recon_spec.obs_path is None:
            raise ValueError("Resolved reconstruction obs path is None.")

        reconstruction_input = load_reconstruction_evaluation_input(
            input_path=recon_spec.input_path,
            reconstruction_path=recon_spec.reconstruction_path,
            obs_path=recon_spec.obs_path,
            metadata_path=recon_spec.metadata_path,
            n_examples=recon_spec.n_examples,
        )

        _validate_reconstruction_obs_length(reconstruction_input)

    return EvaluateSourceInputsResult(
        adata_input=adata,
        reconstruction_input=reconstruction_input,
    )


def log_clustering_count_warnings(
    adata: ad.AnnData,
    *,
    max_clusters_warn: int | None,
) -> None:
    """Log warnings for clustering outputs with high cluster count.

    Expects clustering metadata at:

        adata.uns["benchrep"]["clustering"][key_added]

    Each metadata record should contain ``n_clusters``. If missing, the count is
    inferred from ``adata.obs[key_added]`` when possible.
    """

    if max_clusters_warn is None:
        return

    run_log = get_run_logger()

    benchrep_uns = adata.uns.get("benchrep")

    if benchrep_uns is None:
        return

    if not isinstance(benchrep_uns, Mapping):
        run_log.warning(
            "Expected adata.uns['benchrep'] to be a mapping, but found %s. "
            "Skipping clustering count warnings.",
            type(benchrep_uns).__name__,
        )
        return

    clustering_uns = benchrep_uns.get("clustering")

    if clustering_uns is None:
        return

    if not isinstance(clustering_uns, Mapping):
        run_log.warning(
            "Expected adata.uns['benchrep']['clustering'] to be a mapping, "
            "but found %s. Skipping clustering count warnings.",
            type(clustering_uns).__name__,
        )
        return

    if not clustering_uns:
        return

    for key_added, metadata in clustering_uns.items():
        if not isinstance(metadata, Mapping):
            run_log.warning(
                "Expected clustering metadata for key '%s' to be a mapping, "
                "but found %s. Skipping this clustering result.",
                key_added,
                type(metadata).__name__,
            )
            continue

        n_clusters = metadata.get("n_clusters")

        if n_clusters is None:
            if key_added in adata.obs_keys():
                n_clusters = int(adata.obs[key_added].nunique(dropna=True))
                run_log.warning(
                    "Clustering metadata for key '%s' is missing required field "
                    "'n_clusters'. Computed it from adata.obs instead.",
                    key_added,
                )
            else:
                run_log.warning(
                    "Clustering metadata for key '%s' is missing required field "
                    "'n_clusters'. Cannot check whether this clustering result has "
                    "too many clusters.",
                    key_added,
                )
                continue

        try:
            n_clusters = int(n_clusters)
        except (TypeError, ValueError):
            run_log.warning(
                "Clustering metadata for key '%s' has invalid n_clusters=%r. "
                "Cannot check whether this clustering result has too many clusters.",
                key_added,
                n_clusters,
            )
            continue

        if n_clusters > max_clusters_warn:
            run_log.warning(
                "Clustering key '%s' produced %d clusters, exceeding the "
                "configured warning threshold of %d. Cluster-colored reduction "
                "plots and cluster-size plots may be difficult to interpret.",
                key_added,
                n_clusters,
                max_clusters_warn,
            )


def audit_evaluate_outputs(
    *,
    run_context: RunContext,
    run_spec: EvaluationRunSpec,
    adata: ad.AnnData,
    reconstruction_outputs: Mapping[str, Any] | None,
    export_paths: EvaluationExportPaths,
    config_composition_result: ConfigCompositionResult[Any],
    resolved_config_path: Path | str,
    evaluation_manifest_path: Path | str,
) -> list[AuditItem]:
    """Audit evaluation runtime results and exported artifacts."""
    audit_items: list[AuditItem] = []

    evaluation_manifest_path = Path(evaluation_manifest_path)
    evaluation_manifest: dict[str, Any] | None = None

    # -------------------------
    # Config records
    # -------------------------
    audit_config_records(
        audit_items=audit_items,
        config_composition_result=config_composition_result,
        resolved_config_path=resolved_config_path,
    )

    # -------------------------
    # Evaluation manifest
    # -------------------------
    manifest_path_is_valid = audit_existing_file(
        audit_items=audit_items,
        name="evaluation manifest",
        path=evaluation_manifest_path,
        expected_suffixes={".yaml", ".yml"},
    )

    if manifest_path_is_valid:
        try:
            loaded_manifest = load_yaml(evaluation_manifest_path)

            if not isinstance(loaded_manifest, dict):
                audit_items.append(
                    AuditItem(
                        name="evaluation manifest load",
                        status="error",
                        message=(
                            f"loaded YAML object from '{evaluation_manifest_path}', "
                            f"but expected a mapping and got "
                            f"{type(loaded_manifest).__name__}"
                        ),
                    )
                )
            else:
                evaluation_manifest = loaded_manifest
                audit_items.append(
                    AuditItem(
                        name="evaluation manifest load",
                        status="ok",
                        message=(
                            f"loaded YAML mapping from "
                            f"'{evaluation_manifest_path}'"
                        ),
                    )
                )

        except Exception as exc:
            audit_items.append(
                AuditItem(
                    name="evaluation manifest load",
                    status="error",
                    message=(
                        f"could not load YAML from "
                        f"'{evaluation_manifest_path}'. "
                        f"Original error ({type(exc).__name__}): {exc}"
                    ),
                )
            )

    # -------------------------
    # Resolved-config reconstructability
    # -------------------------
    if evaluation_manifest is None:
        audit_items.append(
            AuditItem(
                name="run reconstructability from resolved config",
                status="skipped",
                message=(
                    "could not be checked because the evaluation manifest "
                    "is unavailable"
                ),
            )
        )
    else:
        provenance_section = evaluation_manifest.get("provenance")
        evaluation_provenance = (
            provenance_section.get("evaluation")
            if isinstance(provenance_section, Mapping)
            else None
        )
        config_provenance = (
            evaluation_provenance.get("config")
            if isinstance(evaluation_provenance, Mapping)
            else None
        )

        run_reconstructable = (
            config_provenance.get(
                "run_reconstructable_from_resolved_config"
            )
            if isinstance(config_provenance, Mapping)
            else None
        )

        audit_resolved_config_reconstructability(
            audit_items=audit_items,
            config_composition_result=config_composition_result,
            resolved_config_path=resolved_config_path,
            run_reconstructable_from_resolved_config=run_reconstructable,
        )

    # -------------------------
    # Manifest status and output directory
    # -------------------------
    if evaluation_manifest is None:
        audit_items.append(
            AuditItem(
                name="manifest status",
                status="skipped",
                message=(
                    "could not be checked because the evaluation manifest "
                    "is unavailable"
                ),
            )
        )
        audit_items.append(
            AuditItem(
                name="manifest run output dir",
                status="skipped",
                message=(
                    "could not be checked because the evaluation manifest "
                    "is unavailable"
                ),
            )
        )

    else:
        manifest_stage = evaluation_manifest.get("stage")

        audit_items.append(
            AuditItem(
                name="manifest stage",
                status="ok" if manifest_stage == "evaluation" else "error",
                message=(
                    "stage is 'evaluation'"
                    if manifest_stage == "evaluation"
                    else (
                        f"stage is {manifest_stage!r}, "
                        "expected 'evaluation'"
                    )
                ),
            )
        )

        manifest_status = evaluation_manifest.get("status")

        audit_items.append(
            AuditItem(
                name="manifest status",
                status="ok" if manifest_status == "completed" else "error",
                message=(
                    "status is 'completed'"
                    if manifest_status == "completed"
                    else (
                        f"status is {manifest_status!r}, "
                        "expected 'completed'"
                    )
                ),
            )
        )

        run_section = evaluation_manifest.get("run")

        if not isinstance(run_section, Mapping):
            audit_items.append(
                AuditItem(
                    name="manifest run output dir",
                    status="error",
                    message="manifest is missing mapping section `run`",
                )
            )

        else:
            output_dir_value = run_section.get("output_dir")

            if not isinstance(output_dir_value, str):
                audit_items.append(
                    AuditItem(
                        name="manifest run output dir",
                        status="error",
                        message=(
                            "`run.output_dir` must be a string path, "
                            f"got {type(output_dir_value).__name__}"
                        ),
                    )
                )

            else:
                manifest_output_dir = Path(output_dir_value)

                audit_existing_dir(
                    audit_items=audit_items,
                    name="manifest run output dir",
                    path=manifest_output_dir,
                )

                if manifest_output_dir != run_context.output_dir:
                    audit_items.append(
                        AuditItem(
                            name="manifest run output dir consistency",
                            status="warning",
                            message=(
                                f"`run.output_dir` is "
                                f"'{manifest_output_dir}', but current run "
                                f"output directory is "
                                f"'{run_context.output_dir}'"
                            ),
                        )
                    )

    # -------------------------
    # Evaluation source provenance
    # -------------------------
    audit_existing_file(
        audit_items=audit_items,
        name="source embeddings",
        path=run_spec.input_spec.embeddings_path,
        expected_suffixes={".h5ad"},
    )

    prediction_manifest_path = (
        run_spec.input_spec.prediction_manifest_path
    )

    if prediction_manifest_path is None:
        audit_items.append(
            AuditItem(
                name="source prediction manifest",
                status="skipped",
                message=(
                    "evaluation used direct source paths and no prediction "
                    "manifest was required"
                ),
            )
        )
    else:
        audit_existing_file(
            audit_items=audit_items,
            name="source prediction manifest",
            path=prediction_manifest_path,
            expected_suffixes={".yaml", ".yml"},
        )

    reconstruction_spec = run_spec.input_spec.reconstructions

    if reconstruction_spec is None:
        audit_items.append(
            AuditItem(
                name="reconstruction source",
                status="skipped",
                message="no reconstruction inputs were configured",
            )
        )

    else:
        _audit_optional_expected_file(
            audit_items=audit_items,
            name="reconstruction input source",
            path=reconstruction_spec.input_path,
            expected_suffixes={".pt"},
            required=True,
        )
        _audit_optional_expected_file(
            audit_items=audit_items,
            name="reconstruction prediction source",
            path=reconstruction_spec.reconstruction_path,
            expected_suffixes={".pt"},
            required=True,
        )
        _audit_optional_expected_file(
            audit_items=audit_items,
            name="reconstruction obs source",
            path=reconstruction_spec.obs_path,
            expected_suffixes={".pt"},
            required=True,
        )
        _audit_optional_expected_file(
            audit_items=audit_items,
            name="reconstruction metadata source",
            path=reconstruction_spec.metadata_path,
            expected_suffixes={".pt"},
            required=False,
        )

    # -------------------------
    # Evaluated AnnData runtime outputs
    # -------------------------
    if adata.n_obs <= 0 or adata.n_vars <= 0:
        audit_items.append(
            AuditItem(
                name="evaluated AnnData",
                status="error",
                message=(
                    f"evaluated AnnData has invalid shape {adata.shape}"
                ),
            )
        )
    else:
        audit_items.append(
            AuditItem(
                name="evaluated AnnData",
                status="ok",
                message=f"evaluated AnnData has shape {adata.shape}",
            )
        )

    step_spec = run_spec.step_spec

    _audit_adata_output_key(
        audit_items=audit_items,
        name="PCA output",
        enabled=step_spec.pca_enabled,
        key=step_spec.pca_params.get("key_added", "X_pca"),
        available_keys=adata.obsm,
        location="adata.obsm",
    )
    _audit_adata_output_key(
        audit_items=audit_items,
        name="UMAP output",
        enabled=step_spec.umap_enabled,
        key=step_spec.umap_params.get("key_added", "X_umap"),
        available_keys=adata.obsm,
        location="adata.obsm",
    )
    _audit_adata_output_key(
        audit_items=audit_items,
        name="t-SNE output",
        enabled=step_spec.tsne_enabled,
        key=step_spec.tsne_params.get("key_added", "X_tsne"),
        available_keys=adata.obsm,
        location="adata.obsm",
    )
    _audit_adata_output_key(
        audit_items=audit_items,
        name="K-means output",
        enabled=step_spec.kmeans_enabled,
        key=step_spec.kmeans_params.get("key_added", "kmeans"),
        available_keys=adata.obs,
        location="adata.obs",
    )
    _audit_adata_output_key(
        audit_items=audit_items,
        name="Leiden output",
        enabled=step_spec.leiden_enabled,
        key=step_spec.leiden_params.get("key_added", "leiden"),
        available_keys=adata.obs,
        location="adata.obs",
    )

    # -------------------------
    # Reconstruction runtime outputs
    # -------------------------
    expected_reconstruction_output_keys: set[str] = set()

    if step_spec.reconstruction_metrics_enabled:
        expected_reconstruction_output_keys.add(
            "reconstruction_metrics"
        )

    if step_spec.reconstruction_tiffs_enabled:
        expected_reconstruction_output_keys.add("error_maps")

    if reconstruction_outputs is None:
        if expected_reconstruction_output_keys:
            audit_items.append(
                AuditItem(
                    name="reconstruction pipeline outputs",
                    status="error",
                    message=(
                        "reconstruction pipeline returned no outputs, but "
                        f"expected keys "
                        f"{sorted(expected_reconstruction_output_keys)}"
                    ),
                )
            )
        else:
            audit_items.append(
                AuditItem(
                    name="reconstruction pipeline outputs",
                    status="skipped",
                    message=(
                        "no reconstruction pipeline outputs were expected"
                    ),
                )
            )

    elif not isinstance(reconstruction_outputs, Mapping):
        audit_items.append(
            AuditItem(
                name="reconstruction pipeline outputs",
                status="error",
                message=(
                    "expected reconstruction outputs to be a mapping, "
                    f"got {type(reconstruction_outputs).__name__}"
                ),
            )
        )

    else:
        missing_output_keys = (
            expected_reconstruction_output_keys
            - set(reconstruction_outputs)
        )

        audit_items.append(
            AuditItem(
                name="reconstruction pipeline outputs",
                status="error" if missing_output_keys else "ok",
                message=(
                    "reconstruction pipeline is missing expected output "
                    f"keys: {sorted(missing_output_keys)}"
                    if missing_output_keys
                    else (
                        "reconstruction pipeline returned expected output "
                        f"keys: "
                        f"{sorted(expected_reconstruction_output_keys)}"
                    )
                ),
            )
        )

    # -------------------------
    # Always-written artifacts
    # -------------------------
    audit_existing_file(
        audit_items=audit_items,
        name="evaluated embeddings artifact",
        path=export_paths.evaluated_embeddings_path,
        expected_suffixes={".h5ad"},
    )

    metrics_path_is_valid = audit_existing_file(
        audit_items=audit_items,
        name="evaluation metrics artifact",
        path=export_paths.metrics_json_path,
        expected_suffixes={".json"},
    )

    metrics: Mapping[str, Any] | None = None

    if metrics_path_is_valid:
        metrics = _load_metrics_json(
            path=export_paths.metrics_json_path,
            audit_items=audit_items,
        )

    # -------------------------
    # Actual exported metric groups
    # -------------------------
    if metrics is not None:
        _audit_metric_group(
            audit_items=audit_items,
            name="internal clustering metrics",
            metrics=metrics,
            path=("clustering", "internal"),
            expected=(
                step_spec.internal_clustering_metrics_enabled
            ),
        )

        external_clustering_metrics_expected = (
            step_spec.external_clustering_metrics_enabled
        )

        if external_clustering_metrics_expected is None:
            external_clustering_metrics_expected = (
                step_spec.external_clustering_label_key
                in adata.obs.columns
            )
        _audit_metric_group(
            audit_items=audit_items,
            name="external clustering metrics",
            metrics=metrics,
            path=("clustering", "external"),
            expected=external_clustering_metrics_expected,
        )
        _audit_metric_group(
            audit_items=audit_items,
            name="embedding metrics",
            metrics=metrics,
            path=("embedding",),
            expected=step_spec.embedding_metrics_enabled,
        )
        _audit_metric_group(
            audit_items=audit_items,
            name="predictability metrics",
            metrics=metrics,
            path=("predictability",),
            expected=step_spec.predictability_enabled,
        )
        _audit_metric_group(
            audit_items=audit_items,
            name="reconstruction metrics",
            metrics=metrics,
            path=("reconstruction",),
            expected=step_spec.reconstruction_metrics_enabled,
        )

    # -------------------------
    # Optional artifact groups
    # -------------------------
    reductions_expected = (
        step_spec.plots_enabled
        and (
            step_spec.pca_enabled
            or step_spec.umap_enabled
            or step_spec.tsne_enabled
        )
    )

    clustering_plots_expected = (
        step_spec.plots_enabled
        and (
            step_spec.kmeans_enabled
            or step_spec.leiden_enabled
        )
    )

    reconstruction_grids_expected = (
        reconstruction_spec is not None
        and step_spec.plots_enabled
    )

    _audit_artifact_group(
        audit_items=audit_items,
        name="embedding reduction figures",
        value=export_paths.reduction_plot_paths,
        expected=reductions_expected,
        expected_suffixes={".png", ".pdf", ".svg"},
    )
    _audit_artifact_group(
        audit_items=audit_items,
        name="cluster-size figures",
        value=export_paths.cluster_size_plot_paths,
        expected=clustering_plots_expected,
        expected_suffixes={".png", ".pdf", ".svg"},
    )
    _audit_artifact_group(
        audit_items=audit_items,
        name="reconstruction TIFF artifacts",
        value=export_paths.reconstruction_tiff_paths,
        expected=step_spec.reconstruction_tiffs_enabled,
        expected_suffixes={".tif", ".tiff"},
    )
    _audit_artifact_group(
        audit_items=audit_items,
        name="reconstruction grid figures",
        value=export_paths.reconstruction_grid_paths,
        expected=reconstruction_grids_expected,
        expected_suffixes={".png", ".pdf", ".svg"},
    )

    # -------------------------
    # Final audit summary
    # -------------------------
    log_audit_summary(
        stage="evaluation",
        audit_items=audit_items,
    )

    return audit_items


def _load_embeddings_adata(path: Path | str) -> ad.AnnData:
    embeddings_path = Path(path)

    if embeddings_path.suffix.lower() != ".h5ad":
        raise ValueError(
            "Evaluation embeddings input must be an AnnData `.h5ad` file. "
            f"Got: {embeddings_path}"
        )

    if not embeddings_path.is_file():
        raise FileNotFoundError(
            f"Embeddings AnnData file does not exist: {embeddings_path}"
        )

    try:
        adata = ad.read_h5ad(embeddings_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load embeddings AnnData from '{embeddings_path}'. "
            f"Original error ({type(exc).__name__}): {exc}"
        ) from exc

    if not isinstance(adata, ad.AnnData):
        raise TypeError(
            "Loaded embeddings object must be an AnnData object, "
            f"got {type(adata).__name__}."
        )

    return adata


def _validate_embedding_adata_basic_contract(adata: ad.AnnData) -> None:
    """Validate only the basic AnnData contract needed to start evaluation."""

    validate_adata_x(adata)

    if adata.n_obs < 1:
        raise ValueError(
            "Evaluation embeddings AnnData must contain at least one observation."
        )

    if adata.n_vars < 1:
        raise ValueError(
            "Evaluation embeddings AnnData must contain at least one variable."
        )

    if adata.X.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            "adata.X shape does not match AnnData dimensions. "
            f"Got X.shape={adata.X.shape}, "
            f"expected ({adata.n_obs}, {adata.n_vars})."
        )

    if not np.issubdtype(adata.X.dtype, np.number):
        raise TypeError(
            "adata.X must be numeric for evaluation, "
            f"got dtype {adata.X.dtype}."
        )


def _validate_enabled_step_preconditions(
    *,
    adata: ad.AnnData,
    run_spec: EvaluationRunSpec,
) -> None:
    """Fail early on obvious enabled-step issues before mutating AnnData."""

    step_spec = run_spec.step_spec

    if step_spec.pca_enabled:
        params = step_spec.pca_params

        _validate_obsm_key_available(
            adata=adata,
            key=_param(params, "key_added", "X_pca"),
            overwrite=_param(params, "overwrite", False),
            step_name="PCA",
        )

        n_components = _param(params, "n_components", 2)
        max_components = min(adata.n_obs, adata.n_vars)

        if n_components > max_components:
            raise ValueError(
                "PCA n_components cannot exceed min(adata.n_obs, adata.n_vars). "
                f"Got n_components={n_components}, max={max_components}."
            )

    if step_spec.umap_enabled:
        params = step_spec.umap_params

        _validate_obsm_key_available(
            adata=adata,
            key=_param(params, "key_added", "X_umap"),
            overwrite=_param(params, "overwrite", False),
            step_name="UMAP",
        )
        _validate_uns_key_available(
            adata=adata,
            key=_param(params, "neighbors_key", "neighbors"),
            overwrite=_param(params, "overwrite", False),
            step_name="UMAP neighbors",
        )
        _validate_n_neighbors(
            n_neighbors=_param(params, "n_neighbors", 15),
            n_obs=adata.n_obs,
            step_name="UMAP",
        )

    if step_spec.tsne_enabled:
        params = step_spec.tsne_params

        _validate_obsm_key_available(
            adata=adata,
            key=_param(params, "key_added", "X_tsne"),
            overwrite=_param(params, "overwrite", False),
            step_name="t-SNE",
        )

        perplexity = _param(params, "perplexity", 30.0)
        if perplexity >= adata.n_obs:
            raise ValueError(
                "t-SNE perplexity must be smaller than adata.n_obs. "
                f"Got perplexity={perplexity}, n_obs={adata.n_obs}."
            )

    if step_spec.kmeans_enabled:
        params = step_spec.kmeans_params

        _validate_obs_key_available_for_write(
            adata=adata,
            key=_param(params, "key_added", "kmeans"),
            overwrite=_param(params, "overwrite", False),
            step_name="KMeans",
        )

        n_clusters = params.get("n_clusters")
        if n_clusters is None:
            raise ValueError("KMeans n_clusters is required when KMeans is enabled.")

        if n_clusters > adata.n_obs:
            raise ValueError(
                "KMeans n_clusters cannot exceed adata.n_obs. "
                f"Got n_clusters={n_clusters}, n_obs={adata.n_obs}."
            )

    if step_spec.leiden_enabled:
        params = step_spec.leiden_params

        _validate_obs_key_available_for_write(
            adata=adata,
            key=_param(params, "key_added", "leiden"),
            overwrite=_param(params, "overwrite", False),
            step_name="Leiden",
        )
        _validate_uns_key_available(
            adata=adata,
            key=_param(params, "neighbors_key", "neighbors"),
            overwrite=_param(params, "overwrite", False),
            step_name="Leiden neighbors",
        )
        _validate_n_neighbors(
            n_neighbors=_param(params, "n_neighbors", 15),
            n_obs=adata.n_obs,
            step_name="Leiden",
        )

    if (
        step_spec.external_clustering_metrics_enabled is True
        and step_spec.external_clustering_label_key not in adata.obs.columns
    ):
        raise KeyError(
            "External clustering metrics were explicitly enabled, but "
            "the loaded AnnData object does not contain "
            f"adata.obs[{step_spec.external_clustering_label_key!r}]."
        )

    if step_spec.predictability_enabled:
        _validate_predictability_preconditions(
            adata=adata,
            target_key=step_spec.predictability_target_key,
            task=step_spec.predictability_task,
            probes=step_spec.predictability_probes,
            cv_params=step_spec.predictability_cv_params,
            tuning_params=step_spec.predictability_tuning_params,
        )


def _validate_predictability_preconditions(
    *,
    adata: ad.AnnData,
    target_key: str,
    task: str,
    probes: list[str],
    cv_params: Mapping[str, Any],
    tuning_params: Mapping[str, Any],
) -> None:
    if target_key not in adata.obs.columns:
        raise KeyError(
            "Predictability metrics were enabled, but the loaded AnnData object "
            f"does not contain adata.obs[{target_key!r}]."
        )

    target = adata.obs[target_key]

    if target.isna().any():
        raise ValueError(
            f"Predictability target adata.obs[{target_key!r}] contains missing values."
        )

    class_counts = None

    if task == "classification":
        class_counts = target.value_counts(dropna=False)

        if class_counts.shape[0] < 2:
            raise ValueError(
                "Classification predictability requires at least two target classes. "
                f"Got {class_counts.shape[0]} class(es)."
            )

    elif task == "regression":
        if not pd.api.types.is_numeric_dtype(target):
            raise TypeError(
                "Regression predictability requires a numeric target column. "
                f"adata.obs[{target_key!r}] has dtype {target.dtype}."
            )

    else:
        raise ValueError(
            "Predictability task must be either 'classification' or 'regression', "
            f"got {task!r}."
        )

    method = _param(cv_params, "method", "stratified_kfold")
    n_splits = _param(cv_params, "n_splits", 5)

    if n_splits > adata.n_obs:
        raise ValueError(
            "Predictability cv.n_splits cannot exceed adata.n_obs. "
            f"Got n_splits={n_splits}, n_obs={adata.n_obs}."
        )

    if method in {"group_kfold", "stratified_group_kfold"}:
        group_key = cv_params.get("group_key")

        if group_key not in adata.obs.columns:
            raise KeyError(
                "Predictability grouped CV was enabled, but the loaded AnnData "
                f"object does not contain adata.obs[{group_key!r}]."
            )

        groups = adata.obs[group_key]

        if groups.isna().any():
            raise ValueError(
                f"Predictability group column adata.obs[{group_key!r}] "
                "contains missing values."
            )

        n_groups = int(groups.nunique(dropna=False))
        if n_splits > n_groups:
            raise ValueError(
                "Predictability grouped CV n_splits cannot exceed the number "
                f"of groups. Got n_splits={n_splits}, n_groups={n_groups}."
            )

    if method in {"stratified_kfold", "stratified_group_kfold"}:
        if task != "classification":
            raise ValueError(
                f"Predictability cv.method={method!r} is only valid for classification."
            )

        min_class_count = int(class_counts.min())
        if n_splits > min_class_count:
            raise ValueError(
                "Stratified predictability CV n_splits cannot exceed the "
                "smallest class count. "
                f"Got n_splits={n_splits}, min_class_count={min_class_count}."
            )

    if tuning_params.get("enabled", False):
        inner_cv = tuning_params.get("inner_cv", {})
        inner_n_splits = _param(inner_cv, "n_splits", 3)

        if inner_n_splits > adata.n_obs:
            raise ValueError(
                "Predictability tuning inner_cv.n_splits cannot exceed adata.n_obs. "
                f"Got inner_n_splits={inner_n_splits}, n_obs={adata.n_obs}."
            )

    if "xgboost" in probes and find_spec("xgboost") is None:
        raise ImportError(
            "The xgboost predictability probe was selected, but xgboost is not "
            "installed. Install xgboost or remove 'xgboost' from "
            "metrics.predictability.selected."
        )


def _validate_reconstruction_obs_length(
    reconstruction_input: ReconstructionEvaluationInput,
) -> None:
    """Lightly validate reconstruction obs length.

    Array shape/type validation is already handled by
    load_reconstruction_evaluation_input().
    """

    obs_length = _infer_obs_length(reconstruction_input.obs)

    if obs_length is None:
        raise TypeError(
            "Could not infer the number of reconstruction observations "
            f"from object of type "
            f"{type(reconstruction_input.obs).__name__}."
        )

    n_available = int(reconstruction_input.inputs.shape[0])

    if obs_length != n_available:
        raise ValueError(
            "Reconstruction obs must contain exactly one row/item per "
            "reconstruction example. "
            f"Got obs length={obs_length}, reconstruction examples={n_available}."
        )


def _infer_obs_length(obs: Any) -> int | None:
    if isinstance(obs, Mapping):
        lengths: dict[str, int] = {}

        for key, value in obs.items():
            if isinstance(value, str | bytes):
                continue

            try:
                lengths[str(key)] = len(value)
            except TypeError:
                continue

        if not lengths:
            return None

        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                "Reconstruction obs mapping contains columns with inconsistent "
                f"lengths: {lengths}."
            )

        return unique_lengths.pop()

    if hasattr(obs, "shape") and len(obs.shape) > 0:
        return int(obs.shape[0])

    try:
        return len(obs)
    except TypeError:
        return None


def _validate_n_neighbors(
    *,
    n_neighbors: int,
    n_obs: int,
    step_name: str,
) -> None:
    if n_neighbors >= n_obs:
        raise ValueError(
            f"{step_name} n_neighbors must be smaller than adata.n_obs. "
            f"Got n_neighbors={n_neighbors}, n_obs={n_obs}."
        )


def _validate_obsm_key_available(
    *,
    adata: ad.AnnData,
    key: str,
    overwrite: bool,
    step_name: str,
) -> None:
    if key in adata.obsm and not overwrite:
        raise KeyError(
            f"{step_name} output key adata.obsm[{key!r}] already exists. "
            "Set overwrite=True to replace it."
        )


def _validate_obs_key_available_for_write(
    *,
    adata: ad.AnnData,
    key: str,
    overwrite: bool,
    step_name: str,
) -> None:
    if key in adata.obs.columns and not overwrite:
        raise KeyError(
            f"{step_name} output key adata.obs[{key!r}] already exists. "
            "Set overwrite=True to replace it."
        )


def _validate_uns_key_available(
    *,
    adata: ad.AnnData,
    key: str,
    overwrite: bool,
    step_name: str,
) -> None:
    if key in adata.uns and not overwrite:
        raise KeyError(
            f"{step_name} output key adata.uns[{key!r}] already exists. "
            "Set overwrite=True to replace it."
        )


def _param(
    params: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    value = params.get(key, default)
    return default if value is None else value


def _audit_optional_expected_file(
    *,
    audit_items: list[AuditItem],
    name: str,
    path: Path | str | None,
    expected_suffixes: set[str],
    required: bool,
) -> bool:
    if path is None:
        audit_items.append(
            AuditItem(
                name=name,
                status="error" if required else "skipped",
                message=(
                    "expected artifact path was not resolved"
                    if required
                    else "optional artifact path was not provided"
                ),
            )
        )
        return False

    return audit_existing_file(
        audit_items=audit_items,
        name=name,
        path=path,
        expected_suffixes=expected_suffixes,
    )


def _audit_adata_output_key(
    *,
    audit_items: list[AuditItem],
    name: str,
    enabled: bool,
    key: Any,
    available_keys: Any,
    location: str,
) -> None:
    if not enabled:
        audit_items.append(
            AuditItem(
                name=name,
                status="skipped",
                message="step was not enabled",
            )
        )
        return

    if not isinstance(key, str):
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=(
                    f"resolved output key must be a string, "
                    f"got {type(key).__name__}"
                ),
            )
        )
        return

    if key not in available_keys:
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=f"expected output `{location}[{key!r}]` was not found",
            )
        )
        return

    audit_items.append(
        AuditItem(
            name=name,
            status="ok",
            message=f"found output at `{location}[{key!r}]`",
        )
    )


def _load_metrics_json(
    *,
    path: Path | str,
    audit_items: list[AuditItem],
) -> Mapping[str, Any] | None:
    path = Path(path)

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)

    except Exception as exc:
        audit_items.append(
            AuditItem(
                name="evaluation metrics load",
                status="error",
                message=(
                    f"could not load metrics JSON from '{path}'. "
                    f"Original error ({type(exc).__name__}): {exc}"
                ),
            )
        )
        return None

    if not isinstance(value, Mapping):
        audit_items.append(
            AuditItem(
                name="evaluation metrics load",
                status="error",
                message=(
                    "loaded metrics JSON must contain a mapping, "
                    f"got {type(value).__name__}"
                ),
            )
        )
        return None

    audit_items.append(
        AuditItem(
            name="evaluation metrics load",
            status="ok",
            message=f"loaded metrics mapping from '{path}'",
        )
    )

    return value


def _audit_metric_group(
    *,
    audit_items: list[AuditItem],
    name: str,
    metrics: Mapping[str, Any],
    path: tuple[str, ...],
    expected: bool | None,
) -> None:
    value: Any = metrics

    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            value = None
            break
        value = value[key]

    present = (
        isinstance(value, Mapping)
        and bool(value)
    )

    dotted_path = ".".join(path)

    if expected is True:
        audit_items.append(
            AuditItem(
                name=name,
                status="ok" if present else "error",
                message=(
                    f"found exported metric group `{dotted_path}`"
                    if present
                    else (
                        f"metric group `{dotted_path}` was enabled "
                        "but was not exported"
                    )
                ),
            )
        )
        return

    if expected is False:
        audit_items.append(
            AuditItem(
                name=name,
                status="warning" if present else "skipped",
                message=(
                    f"metric group `{dotted_path}` was exported even "
                    "though it was disabled"
                    if present
                    else f"metric group `{dotted_path}` was not enabled"
                ),
            )
        )
        return

    # None means runtime auto-resolution.
    audit_items.append(
        AuditItem(
            name=name,
            status="ok" if present else "skipped",
            message=(
                f"auto-resolved metric group `{dotted_path}` was exported"
                if present
                else (
                    f"auto-resolved metric group `{dotted_path}` "
                    "was not produced"
                )
            ),
        )
    )


def _audit_artifact_group(
    *,
    audit_items: list[AuditItem],
    name: str,
    value: Any,
    expected: bool,
    expected_suffixes: set[str],
) -> None:
    paths = _collect_artifact_paths(value)

    if not expected:
        audit_items.append(
            AuditItem(
                name=name,
                status="warning" if paths else "skipped",
                message=(
                    f"found {len(paths)} artifact file(s) even though "
                    "the export was not expected"
                    if paths
                    else "artifact export was not expected"
                ),
            )
        )
        return

    if not paths:
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=(
                    "artifact export was expected, but no paths "
                    "were returned"
                ),
            )
        )
        return

    normalized_suffixes = {
        suffix.lower()
        for suffix in expected_suffixes
    }

    missing = [
        path
        for path in paths
        if not path.exists() or not path.is_file()
    ]
    invalid_suffixes = [
        path
        for path in paths
        if path.suffix.lower() not in normalized_suffixes
    ]

    if missing or invalid_suffixes:
        details = []

        if missing:
            details.append(
                f"{len(missing)} missing/non-file path(s)"
            )
        if invalid_suffixes:
            details.append(
                f"{len(invalid_suffixes)} path(s) with unexpected suffixes"
            )

        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message="; ".join(details),
            )
        )
        return

    audit_items.append(
        AuditItem(
            name=name,
            status="ok",
            message=f"found {len(paths)} exported artifact file(s)",
        )
    )


def _collect_artifact_paths(value: Any) -> list[Path]:
    if isinstance(value, Path):
        return [value]

    if isinstance(value, Mapping):
        paths: list[Path] = []

        for item in value.values():
            paths.extend(_collect_artifact_paths(item))

        return paths

    if isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes,
    ):
        paths = []

        for item in value:
            paths.extend(_collect_artifact_paths(item))

        return paths

    return []