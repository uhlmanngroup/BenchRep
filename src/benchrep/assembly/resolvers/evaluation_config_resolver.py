from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import warnings

from benchrep.assembly.config import load_yaml
from benchrep.assembly.registries.core import (
    EVAL_INTERNAL_CLUSTERING_METRICS,
    EVAL_EXTERNAL_CLUSTERING_METRICS,
    EVAL_EMBEDDING_METRICS,
    EVAL_RECONSTRUCTION_METRICS,
    EVAL_PREDICTABILITY_PROBES,
)
from benchrep.assembly.registries.utils import (
    resolve_registry_keys,
    resolve_registry_param_keys,
)
from benchrep.assembly.resolvers.utils import (
    get_optional_nested_path,
    get_required_nested_path,
    get_optional_nested_value,
    params_to_dict,
)
from benchrep.assembly.schemas.evaluation_config_schema import (
    EvaluationConfig,
    EvaluationRunConfig,
)


# -------------------------
# Resolved specs
# -------------------------
@dataclass(frozen=True)
class EvaluationReconstructionInputSpec:
    input_path: Path | None
    reconstruction_path: Path | None
    obs_path: Path | None
    metadata_path: Path | None
    n_examples: int | None


@dataclass(frozen=True)
class EvaluationInputSpec:
    embeddings_path: Path
    reconstructions: EvaluationReconstructionInputSpec | None
    prediction_manifest_path: Path | None


PredictabilityTask = Literal["classification", "regression"]


@dataclass(frozen=True)
class EvaluationStepSpec:
    pca_enabled: bool
    pca_params: dict[str, Any]

    umap_enabled: bool
    umap_params: dict[str, Any]

    tsne_enabled: bool
    tsne_params: dict[str, Any]

    kmeans_enabled: bool
    kmeans_params: dict[str, Any]

    leiden_enabled: bool
    leiden_params: dict[str, Any]

    internal_clustering_metrics_enabled: bool
    internal_clustering_metrics: list[str] | None
    internal_clustering_metric_params: dict[str, dict[str, Any]]

    external_clustering_metrics_enabled: bool | None
    external_clustering_label_key: str
    external_clustering_metrics: list[str] | None
    external_clustering_metric_params: dict[str, dict[str, Any]]

    embedding_metrics_enabled: bool
    embedding_metrics: list[str]
    embedding_metric_params: dict[str, dict[str, Any]]

    predictability_enabled: bool
    predictability_target_key: str
    predictability_task: PredictabilityTask
    predictability_probes: list[str]
    predictability_probe_params: dict[str, dict[str, Any]]
    predictability_cv_params: dict[str, Any]
    predictability_tuning_params: dict[str, Any]

    reconstruction_metrics_enabled: bool
    reconstruction_metrics: list[str] | None
    reconstruction_metric_params: dict[str, dict[str, Any]]
    reconstruction_metrics_reduction: str

    reconstruction_tiffs_enabled: bool
    error_map_params: dict[str, Any]

    plots_enabled: bool
    plot_params: dict[str, Any]


@dataclass(frozen=True)
class EvaluationRunIdentitySpec:
    output_root: Path
    run_name_stem: str | None
    project_name: str | None
    model_name: str | None


@dataclass(frozen=True)
class EvaluationRunSpec:
    stage: Literal["evaluation"]
    evaluation_config: EvaluationConfig
    prediction_manifest: dict[str, Any] | None
    run_identity: EvaluationRunIdentitySpec
    input_spec: EvaluationInputSpec
    step_spec: EvaluationStepSpec


def resolve_evaluation_config(
    evaluation_config: EvaluationConfig,
    prediction_manifest_path_override: Path | str | None = None,
) -> EvaluationRunSpec:
    """Resolve a parsed evaluation config into an executable run spec.

    This function is the top-level evaluation resolver. It performs only
    workflow-level orchestration: loading an optional prediction manifest,
    resolving input paths, resolving reconstruction artifacts, deriving run
    identity information, resolving step defaults, and packaging everything
    into an immutable ``EvaluationRunSpec``.

    Context-dependent decisions are delegated to smaller helpers. In particular,
    reconstruction availability is resolved before step defaults, because
    reconstruction metrics and error maps depend on whether reconstruction
    artifacts are available.

    Parameters
    ----------
    evaluation_config
        Parsed evaluation configuration.

    Returns
    -------
    EvaluationRunSpec
        Fully resolved evaluation runtime specification.
    """
    # Resolve prediction manifest
    evaluation_config, prediction_manifest_path = _resolve_prediction_manifest_path(
        evaluation_config=evaluation_config,
        prediction_manifest_path_override=prediction_manifest_path_override,
    )

    if prediction_manifest_path is not None:
        prediction_manifest = _load_prediction_manifest(prediction_manifest_path)
        manifest_base_dir = prediction_manifest_path.parent
    else:
        prediction_manifest = None
        manifest_base_dir = None

    # Resolve embedding input: manual path overrides manifest
    embeddings_path = resolve_embeddings_path(
        embeddings_path=evaluation_config.source.embeddings_path,
        prediction_manifest=prediction_manifest,
        manifest_base_dir=manifest_base_dir,
    )

    # Resolve reconstructions: manual path overrides manifest
    reconstructions = resolve_reconstructions(
        reconstructions_path=evaluation_config.source.reconstructions_path,
        prediction_manifest=prediction_manifest,
        manifest_base_dir=manifest_base_dir,
        n_examples=evaluation_config.reconstruction.n_examples,
    )

    has_reconstructions = reconstructions is not None

    # Resolve run identity. RunContext is handled by the entrypoint workflow script
    run_identity = resolve_run_identity(
        run_config=evaluation_config.run,
        prediction_manifest=prediction_manifest,
        manifest_base_dir=manifest_base_dir,
    )

    # Resolve step spec (some configs need further downstream resolution)
    step_spec = resolve_step_spec(
        evaluation_config=evaluation_config,
        has_reconstructions=has_reconstructions,
    )

    input_spec = EvaluationInputSpec(
        embeddings_path=embeddings_path,
        reconstructions=reconstructions,
        prediction_manifest_path=prediction_manifest_path,
    )

    return EvaluationRunSpec(
        stage=evaluation_config.stage,
        evaluation_config=evaluation_config,
        prediction_manifest=prediction_manifest,
        run_identity=run_identity,
        input_spec=input_spec,
        step_spec=step_spec,
    )


def resolve_embeddings_path(
        embeddings_path: Path | None = None,
        prediction_manifest: dict[str, Any] | None = None,
        manifest_base_dir: Path | None = None,
) -> Path:
    """Resolve the AnnData embeddings input for evaluation.

    Manual ``source.embeddings_path`` takes precedence over embeddings inferred
    from a prediction manifest.
    """
    if embeddings_path is not None:
        resolved_embeddings_path = Path(embeddings_path).resolve()

    else:
        if prediction_manifest is None:
            raise ValueError(
                "source.embeddings_path is required when no prediction manifest "
                "is provided."
            )

        if manifest_base_dir is None:
            raise ValueError(
                "manifest_base_dir is required when resolving embeddings "
                "from a prediction manifest."
            )

        resolved_embeddings_path = get_required_nested_path(
            prediction_manifest,
            "exports",
            "embeddings",
            "path",
            base_dir=manifest_base_dir,
        )

    return resolved_embeddings_path


def resolve_reconstructions(
        reconstructions_path: Path | None = None,
        prediction_manifest: dict[str, Any] | None = None,
        manifest_base_dir: Path | None = None,
        n_examples: int | None = None,
) -> EvaluationReconstructionInputSpec | None:
    """Resolve reconstruction artifact inputs for evaluation.

    Manual ``source.reconstructions_path`` takes precedence over reconstruction
    paths inferred from a prediction manifest. Manual reconstruction input is treated
    as an explicit user request and must point to a complete reconstruction artifact
    bundle containing ``input.pt``, ``reconstruction.pt``, and ``obs.pt``.

    When reconstruction paths are inferred from a prediction manifest, an absent
    bundle is treated as unavailable reconstruction input. An incomplete manifest
    bundle is skipped with a warning, because embeddings-only evaluation can still
    proceed.

    Returns
    -------
    EvaluationReconstructionInputSpec | None
        Resolved reconstruction artifact paths, or ``None`` if no usable
        reconstruction bundle is available.
    """
    # Resolve reconstructions: manual path overrides manifest
    if reconstructions_path is not None:
        _recon_root = Path(reconstructions_path).resolve()
        if not _recon_root.is_dir():
            raise NotADirectoryError(
                "source.reconstructions_path must point to a directory containing "
                "the reconstruction artifact bundle. "
                f"Got: {_recon_root}"
            )

        _recon_input_path = _recon_root / "input.pt"
        _recon_path = _recon_root / "reconstruction.pt"
        _recon_obs_path = _recon_root / "obs.pt"
        _recon_metadata_path = _recon_root / "reconstruction_export_metadata.pt"

        missing_required_recon_paths = [
            path
            for path in (_recon_input_path, _recon_path, _recon_obs_path)
            if not path.is_file()
        ]
        if missing_required_recon_paths:
            raise FileNotFoundError(
                f"Expected reconstruction files do not exist in provided "
                f"source.reconstructions_path: {missing_required_recon_paths}"
            )

        reconstructions = EvaluationReconstructionInputSpec(
            input_path=_recon_input_path,
            reconstruction_path=_recon_path,
            obs_path=_recon_obs_path,
            metadata_path=_recon_metadata_path if _recon_metadata_path.is_file() else None,
            n_examples=n_examples, # manual branch only respects explicit eval config
        )

    elif prediction_manifest is not None:
        manifest_n_examples = get_optional_nested_value(
            prediction_manifest,
            "exports",
            "reconstructions",
            "n_examples_exported",
        )

        if (
                manifest_n_examples is not None
                and (
                not isinstance(manifest_n_examples, int)
                or isinstance(manifest_n_examples, bool)
                or manifest_n_examples < 1
        )
        ):
            warnings.warn(
                "Prediction manifest field "
                "'exports.reconstructions.n_examples_exported' is not a positive "
                "integer. Ignoring it for reconstruction artifact count resolution.",
                stacklevel=2,
            )
            manifest_n_examples = None

        resolved_n_examples = (
            n_examples
            if n_examples is not None
            else manifest_n_examples
        )

        if manifest_base_dir is None:
            raise ValueError(
                "manifest_base_dir is required when resolving reconstructions "
                "from a prediction manifest."
            )

        _recon_input_path = get_optional_nested_path(
            prediction_manifest,
            "exports",
            "reconstructions",
            "paths",
            "input",
            base_dir=manifest_base_dir,
        )
        _recon_path = get_optional_nested_path(
            prediction_manifest,
            "exports",
            "reconstructions",
            "paths",
            "reconstruction",
            base_dir=manifest_base_dir,
        )
        _recon_obs_path = get_optional_nested_path(
            prediction_manifest,
            "exports",
            "reconstructions",
            "paths",
            "obs",
            base_dir=manifest_base_dir,
        )
        _recon_metadata_path = get_optional_nested_path(
            prediction_manifest,
            "exports",
            "reconstructions",
            "paths",
            "metadata",
            base_dir=manifest_base_dir,
        )

        required_recon_paths = {
            "input": _recon_input_path,
            "reconstruction": _recon_path,
            "obs": _recon_obs_path,
        }

        # If all required path missing, abort reconstruction evaluation
        if all(path is None for path in required_recon_paths.values()):
            reconstructions = None
        # If only subset missing, abort but warn
        else:
            missing_required_recon_paths = [
                name
                for name, path in required_recon_paths.items()
                if path is None or not path.is_file()
            ]

            if missing_required_recon_paths:
                warnings.warn(
                    "Prediction manifest contains an incomplete reconstruction "
                    "artifact bundle. Reconstruction inputs will be skipped. "
                    f"Missing required files: {missing_required_recon_paths}",
                    stacklevel=2,
                )
                reconstructions = None
            else:
                reconstructions = EvaluationReconstructionInputSpec(
                    input_path=_recon_input_path,
                    reconstruction_path=_recon_path,
                    obs_path=_recon_obs_path,
                    metadata_path=(
                        _recon_metadata_path
                        if _recon_metadata_path is not None
                           and _recon_metadata_path.is_file()
                        else None
                    ),
                    n_examples=resolved_n_examples,
                )
    else:
        reconstructions = None

    return reconstructions


def resolve_run_identity(
        run_config: EvaluationRunConfig,
        prediction_manifest: dict[str, Any] | None = None,
        manifest_base_dir: Path | None = None,
) -> EvaluationRunIdentitySpec:
    """Resolve output-root and run-name identity hints for an evaluation run.

    ``RunContext`` creation is intentionally left to the workflow entrypoint.
    This helper only resolves the pieces needed to construct a sensible
    evaluation run identity: output root, optional explicit run-name stem,
    project name, and model name.

    If a prediction manifest is available, the output root is inferred from the
    parent workflow output directory when possible. Project/model identity is
    inferred from manifest summary fields when they are valid strings.

    The derived ``model_name`` follows a dependency rule: decoder is included
    only if both model and encoder are valid strings. This avoids names such as
    ``vae_mlp_decoder`` when the encoder identity is missing or invalid.
    """
    # Resolve output_root
    if run_config.output_root is not None:
        output_root = run_config.output_root.resolve()

    elif prediction_manifest is not None:
        prediction_output_dir = get_optional_nested_path(
            prediction_manifest,
            "run",
            "output_dir",
            base_dir=manifest_base_dir,
        )

        if prediction_output_dir is None:
            warnings.warn(
                "Could not infer evaluation output root from prediction manifest "
                "because 'run.output_dir' is missing or null. Falling back to 'outputs/'.",
                stacklevel=2,
            )
            output_root = Path("outputs").resolve()
        else:
            output_root = prediction_output_dir.parent.parent

    else:
        output_root = Path("outputs").resolve()

    # Resolve run_name
    if run_config.run_name is not None:
        run_name_stem = run_config.run_name
    else:
        run_name_stem = None

    if prediction_manifest is not None:
        project_name_value = get_optional_nested_value(
            prediction_manifest,
            "summary",
            "project_name",
        )
        project_name = (
            project_name_value
            if isinstance(project_name_value, str)
            else None
        )

        model = get_optional_nested_value(
            prediction_manifest,
            "summary",
            "model",
        )
        encoder = get_optional_nested_value(
            prediction_manifest,
            "summary",
            "encoder",
        )
        decoder = get_optional_nested_value(
            prediction_manifest,
            "summary",
            "decoder",
        )

        if not isinstance(model, str):
            model_name = None
        elif not isinstance(encoder, str):
            model_name = model
        elif not isinstance(decoder, str):
            model_name = f"{model}_{encoder}"
        else:
            model_name = f"{model}_{encoder}_{decoder}"
    else:
        project_name = None
        model_name = None

    return EvaluationRunIdentitySpec(
        output_root=output_root,
        run_name_stem=run_name_stem,
        project_name=project_name,
        model_name=model_name,
    )


def resolve_step_spec(
        evaluation_config: EvaluationConfig,
        *,
        has_reconstructions: bool,
) -> EvaluationStepSpec:
    """Resolve evaluation step switches and backend parameter dictionaries.

    Evaluation ``enabled`` fields use tri-state semantics:

    - ``True`` explicitly enables a step.
    - ``False`` explicitly disables a step.
    - ``None`` means automatic/default behavior, resolved here unless later
      runtime information is required.

    Most static defaults are resolved immediately. PCA, UMAP, Leiden, internal
    clustering metrics, embedding metrics, and plots default to enabled. t-SNE
    and KMeans default to disabled.

    Reconstruction metrics default to enabled when reconstruction artifacts are
    available. Reconstruction TIFF export requires explicit enablement and available
    reconstructions. Error-map parameters are shared by TIFF and reconstruction-grid
    consumers and do not have an independent enablement switch.

    External clustering metrics remain partially unresolved because they depend
    on the loaded AnnData object: ``None`` is preserved so the workflow can later
    decide based on whether ``label_key`` exists in ``adata.obs``.

    Clustering metrics also require at least one clustering method to be enabled.
    If all clustering methods are disabled, both internal and external clustering
    metrics are forced off during resolution. The availability check currently
    reflects the explicitly modeled clustering methods in the config schema
    (``kmeans`` and ``leiden``); if additional clustering methods are added to the
    schema, they should be included in this check as well.

    Selected metric names are validated against the evaluation registries and
    resolved to canonical names. Parameter objects are converted to dictionaries for
    downstream/backend use.
    ``None`` parameters are omitted by ``params_to_dict`` so backend defaults can
    apply.
    """
    # Resolve step switches and params needed by multiple downstream decisions.
    pca_enabled = enabled_by_default(evaluation_config.reductions.pca.enabled)
    pca_params = params_to_dict(evaluation_config.reductions.pca.params)

    umap_enabled = enabled_by_default(evaluation_config.reductions.umap.enabled)
    umap_params = params_to_dict(evaluation_config.reductions.umap.params)

    tsne_enabled = disabled_by_default(evaluation_config.reductions.tsne.enabled)
    tsne_params = params_to_dict(evaluation_config.reductions.tsne.params)

    kmeans_enabled = disabled_by_default(evaluation_config.clustering.kmeans.enabled)
    kmeans_params = params_to_dict(evaluation_config.clustering.kmeans.params)

    leiden_enabled = enabled_by_default(evaluation_config.clustering.leiden.enabled)
    leiden_params = params_to_dict(evaluation_config.clustering.leiden.params)

    # Prep for guard to prevent clustering metric computation if no clustering is enabled.
    clustering_enabled = kmeans_enabled or leiden_enabled

    external_clustering_metrics_enabled = (
        evaluation_config.metrics.clustering.external.enabled
        if clustering_enabled
        else False
    )

    embedding_metrics = resolve_registry_keys(
        selected=evaluation_config.metrics.embedding.selected,
        registry=EVAL_EMBEDDING_METRICS,
        none_policy="preserve",
    )

    if embedding_metrics is None:
        raise ValueError(
            "metrics.embedding.selected cannot be None."
        )

    plot_params = params_to_dict(evaluation_config.plots.params)
    plot_params = _resolve_plot_params(
        plot_params=plot_params,
        kmeans_enabled=kmeans_enabled,
        kmeans_params=kmeans_params,
        leiden_enabled=leiden_enabled,
        leiden_params=leiden_params,
        external_clustering_metrics_enabled=external_clustering_metrics_enabled,
        external_clustering_label_key=evaluation_config.metrics.clustering.external.label_key,
    )

    # Guard against key collision for kNNs of UMAP and Leiden (as current implementation is independent kNN ownership)
    if (
            umap_enabled
            and leiden_enabled
            and umap_params["neighbors_key"] == leiden_params["neighbors_key"]
            and not leiden_params.get("overwrite", False)
    ):
        raise ValueError(
            "UMAP and Leiden are both enabled and both resolve to the same "
            f"neighbors_key: {umap_params['neighbors_key']!r}. "
            "BenchRep evaluation steps currently own their own kNN graphs. "
            "Use distinct neighbors_key values, or set "
            "clustering.leiden.params.overwrite=True if you intentionally want "
            "Leiden to overwrite the existing graph."
        )

    predictability_config = evaluation_config.metrics.predictability
    predictability_enabled = disabled_by_default(predictability_config.enabled)
    predictability_task: PredictabilityTask = predictability_config.task

    predictability_probes = resolve_registry_keys(
        selected=predictability_config.selected,
        registry=EVAL_PREDICTABILITY_PROBES,
        none_policy="preserve",
    )
    if predictability_probes is None:
        raise ValueError("metrics.predictability.selected cannot be None.")

    predictability_probe_params = resolve_registry_param_keys(
        params=params_to_dict(predictability_config.params),
        registry=EVAL_PREDICTABILITY_PROBES,
    )
    predictability_cv_params = params_to_dict(predictability_config.cv)
    predictability_tuning_params = params_to_dict(predictability_config.tuning)

    predictability_cv_params["scoring"] = _resolve_predictability_scoring(
        task=predictability_task,
        cv_params=predictability_cv_params,
    )

    if predictability_enabled:
        _validate_predictability_linear_model(
            task=predictability_task,
            probes=predictability_probes,
            probe_params=predictability_probe_params,
        )
        _validate_predictability_tuning_grid(
            probes=predictability_probes,
            tuning_params=predictability_tuning_params,
            probe_params=predictability_probe_params,
        )

    reconstruction_metrics_enabled = resolve_enabled_if_available(
        configured=evaluation_config.metrics.reconstruction.enabled,
        available=has_reconstructions,
        name="Reconstruction metrics",
    )

    reconstruction_tiffs_enabled = resolve_enabled_if_explicit_and_available(
        configured=evaluation_config.reconstruction.export_tiffs,
        available=has_reconstructions,
        name="Reconstruction TIFF export",
    )

    if (
            evaluation_config.reconstruction.n_examples is not None
            and not reconstruction_tiffs_enabled
    ):
        warnings.warn(
            "reconstruction.n_examples is set, but reconstruction TIFF export is "
            "disabled. The value will have no effect; reconstruction metrics use "
            "all available reconstructions and reconstruction grids use their own "
            "sampling configuration.",
            UserWarning,
            stacklevel=2,
        )

    return EvaluationStepSpec(
        # None = True
        pca_enabled=pca_enabled,
        pca_params=pca_params,

        # None = True
        umap_enabled=umap_enabled,
        umap_params=umap_params,

        # None = False
        tsne_enabled=tsne_enabled,
        tsne_params=tsne_params,

        # None = False
        kmeans_enabled=kmeans_enabled,
        kmeans_params=kmeans_params,

        # None = True
        leiden_enabled=leiden_enabled,
        leiden_params=leiden_params,

        # None = True
        # Force disable if not clustering_enabled
        internal_clustering_metrics_enabled=(
            enabled_by_default(evaluation_config.metrics.clustering.internal.enabled)
            if clustering_enabled
            else False
        ),
        internal_clustering_metrics=resolve_registry_keys(
            selected=evaluation_config.metrics.clustering.internal.selected,
            registry=EVAL_INTERNAL_CLUSTERING_METRICS,
            none_policy="preserve",
        ),
        internal_clustering_metric_params=resolve_registry_param_keys(
            params=evaluation_config.metrics.clustering.internal.params,
            registry=EVAL_INTERNAL_CLUSTERING_METRICS,
        ),

        # None is resolved later after loading AnnData and checking adata.obs
        # Force disable if not clustering_enabled
        external_clustering_metrics_enabled=external_clustering_metrics_enabled,
        external_clustering_label_key=evaluation_config.metrics.clustering.external.label_key,
        external_clustering_metrics=resolve_registry_keys(
            selected=evaluation_config.metrics.clustering.external.selected,
            registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
            none_policy="preserve",
        ),
        external_clustering_metric_params=resolve_registry_param_keys(
            params=evaluation_config.metrics.clustering.external.params,
            registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
        ),

        # None = True
        embedding_metrics_enabled=enabled_by_default(
            evaluation_config.metrics.embedding.enabled
        ),
        embedding_metrics=embedding_metrics,
        embedding_metric_params=resolve_registry_param_keys(
            params=evaluation_config.metrics.embedding.params,
            registry=EVAL_EMBEDDING_METRICS,
        ),

        # None = False
        predictability_enabled=predictability_enabled,
        predictability_target_key=predictability_config.target_key,
        predictability_task=predictability_task,
        predictability_probes=predictability_probes,
        predictability_probe_params=predictability_probe_params,
        predictability_cv_params=predictability_cv_params,
        predictability_tuning_params=predictability_tuning_params,

        # True if not disabled and reconstructions are available
        reconstruction_metrics_enabled=reconstruction_metrics_enabled,
        reconstruction_metrics=resolve_registry_keys(
            selected=evaluation_config.metrics.reconstruction.selected,
            registry=EVAL_RECONSTRUCTION_METRICS,
            none_policy="preserve",
        ),
        reconstruction_metric_params=resolve_registry_param_keys(
            params=evaluation_config.metrics.reconstruction.params,
            registry=EVAL_RECONSTRUCTION_METRICS,
        ),
        reconstruction_metrics_reduction=(
            evaluation_config.metrics.reconstruction.reduction
        ),

        # True only if explicitly enabled and reconstructions are available
        reconstruction_tiffs_enabled=reconstruction_tiffs_enabled,

        # Shared parameters used by TIFF and reconstruction-grid consumers
        error_map_params=params_to_dict(
            evaluation_config.reconstruction.error_maps
        ),

        # None = True
        plots_enabled=enabled_by_default(evaluation_config.plots.enabled),
        plot_params=plot_params,
    )


def _resolve_plot_params(
    *,
    plot_params: dict[str, Any],
    kmeans_enabled: bool,
    kmeans_params: dict[str, Any],
    leiden_enabled: bool,
    leiden_params: dict[str, Any],
    external_clustering_metrics_enabled: bool | None,
    external_clustering_label_key: str,
) -> dict[str, Any]:
    """Resolve effective plot parameters from user config and default outputs."""

    resolved = dict(plot_params)
    color_by = list(resolved.get("color_by") or [])

    if kmeans_enabled:
        color_by.append(kmeans_params.get("key_added", "kmeans"))

    if leiden_enabled:
        color_by.append(leiden_params.get("key_added", "leiden"))

    if external_clustering_metrics_enabled is not False:
        color_by.append(external_clustering_label_key)

    resolved["color_by"] = _deduplicate_strings(color_by)

    return resolved


def _deduplicate_strings(values: list[Any]) -> list[str]:
    """Return non-empty strings once, preserving first occurrence order."""

    seen: set[str] = set()
    deduplicated: list[str] = []

    for value in values:
        if not isinstance(value, str):
            continue

        value = value.strip()
        if value == "" or value in seen:
            continue

        seen.add(value)
        deduplicated.append(value)

    return deduplicated


# Logical helpers
def enabled_by_default(value: bool | None) -> bool:
    """Resolve a tri-state switch where ``None`` means enabled."""
    return value is not False


def disabled_by_default(value: bool | None) -> bool:
    """Resolve a tri-state switch where ``None`` means disabled."""
    return value is True


def resolve_enabled_if_available(
    configured: bool | None,
    *,
    available: bool,
    name: str,
) -> bool:
    """Resolve a tri-state switch that defaults to enabled if input exists.

    ``True`` and ``None`` both enable the step when the required input is
    available. ``False`` always disables it. If the user explicitly enables the
    step but the required input is unavailable, the step is skipped with a
    warning rather than failing the full evaluation run.
    """
    if available:
        return configured is not False

    if configured:
        warnings.warn(
            f"{name} were explicitly enabled, but required inputs are unavailable. "
            f"They will be skipped.",
            UserWarning,
            stacklevel=2,
        )

    return False


def resolve_enabled_if_explicit_and_available(
    configured: bool | None,
    *,
    available: bool,
    name: str,
) -> bool:
    """Resolve a switch that requires both explicit enablement and input data.

    This is used for optional artifact-producing steps such as reconstruction
    error maps. ``None`` behaves like ``False``. If the user explicitly enables
    the step but the required input is unavailable, the step is skipped with a
    warning.
    """
    if not configured:
        return False

    if not available:
        warnings.warn(
            f"{name} were explicitly enabled, but required inputs are unavailable. "
            f"They will be skipped.",
            UserWarning,
            stacklevel=2,
        )
        return False

    return True


# Predictability validation helpers
CLASSIFICATION_SCORERS = {
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "accuracy",
}

REGRESSION_SCORERS = {
    "r2",
    "neg_mean_absolute_error",
    "neg_root_mean_squared_error",
}


def has_tunable_param_grid(value: Any) -> bool:
    """Return True if a nested parameter dictionary contains any list-valued parameter."""
    if isinstance(value, list):
        return True

    if isinstance(value, dict):
        return any(has_tunable_param_grid(v) for v in value.values())

    return False


def _resolve_predictability_scoring(
    *,
    task: PredictabilityTask,
    cv_params: dict[str, Any],
) -> str:
    scoring = cv_params.get("scoring")

    if scoring is None:
        if task == "classification":
            return "balanced_accuracy"

        if task == "regression":
            return "r2"

        raise ValueError(
            "task must be either 'classification' or 'regression', "
            f"got {task!r}."
        )

    valid_scorers = (
        CLASSIFICATION_SCORERS
        if task == "classification"
        else REGRESSION_SCORERS
    )

    if scoring not in valid_scorers:
        raise ValueError(
            f"metrics.predictability.cv.scoring={scoring!r} "
            f"is not valid for task={task!r}."
        )

    return scoring


def _validate_predictability_linear_model(
    *,
    task: PredictabilityTask,
    probes: list[str],
    probe_params: dict[str, dict[str, Any]],
) -> None:
    if "linear" not in probes:
        return

    linear_model = probe_params.get("linear", {}).get("model")

    if task == "classification" and linear_model != "logistic_regression":
        raise ValueError(
            "metrics.predictability.params.linear.model must be "
            "'logistic_regression' when task='classification'."
        )

    if task == "regression" and linear_model != "ridge":
        raise ValueError(
            "metrics.predictability.params.linear.model must be "
            "'ridge' when task='regression'."
        )


def _validate_predictability_tuning_grid(
    *,
    probes: list[str],
    tuning_params: dict[str, Any],
    probe_params: dict[str, dict[str, Any]],
) -> None:
    selected_params = {
        name: probe_params[name]
        for name in probes
        if name in probe_params
    }

    has_grid_params = has_tunable_param_grid(selected_params)
    tuning_enabled = tuning_params.get("enabled", False)

    if tuning_enabled and not has_grid_params:
        raise ValueError(
            "metrics.predictability.tuning.enabled=True, but no list-valued "
            "hyperparameters were found for the selected probes."
        )

    if not tuning_enabled and has_grid_params:
        raise ValueError(
            "metrics.predictability.tuning.enabled=False, but list-valued "
            "hyperparameters were found for the selected probes."
        )


def _resolve_prediction_manifest_path(
    *,
    evaluation_config: EvaluationConfig,
    prediction_manifest_path_override: Path | str | None,
) -> tuple[EvaluationConfig, Path | None]:
    if prediction_manifest_path_override is not None:
        prediction_manifest_path = Path(prediction_manifest_path_override).resolve()

        evaluation_config = evaluation_config.model_copy(
            update={
                "source": evaluation_config.source.model_copy(
                    update={"prediction_manifest_path": prediction_manifest_path}
                )
            }
        )
    else:
        prediction_manifest_path = evaluation_config.source.prediction_manifest_path
        if prediction_manifest_path is not None:
            prediction_manifest_path = prediction_manifest_path.resolve()

    if prediction_manifest_path is None:
        return evaluation_config, None

    if prediction_manifest_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            "source.prediction_manifest_path must point to a YAML file. "
            f"Got: {prediction_manifest_path}"
        )

    if not prediction_manifest_path.is_file():
        raise FileNotFoundError(
            f"Prediction manifest file does not exist: {prediction_manifest_path}"
        )

    return evaluation_config, prediction_manifest_path


def _load_prediction_manifest(path: Path) -> dict[str, Any]:
    prediction_manifest = load_yaml(path)

    if not isinstance(prediction_manifest, dict):
        raise TypeError(
            "Prediction manifest must load as a mapping, "
            f"got {type(prediction_manifest).__name__}."
        )

    manifest_stage = prediction_manifest.get("stage")
    if manifest_stage != "prediction":
        raise ValueError(
            "Evaluation requires a prediction manifest, "
            f"but manifest stage is {manifest_stage!r}."
        )

    manifest_status = prediction_manifest.get("status")
    if manifest_status != "completed":
        raise ValueError(
            "Evaluation requires a completed prediction manifest, "
            f"but manifest status is {manifest_status!r}."
        )

    return prediction_manifest
