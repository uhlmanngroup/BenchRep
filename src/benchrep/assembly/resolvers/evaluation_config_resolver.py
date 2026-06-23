from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import warnings

from benchrep.assembly.config import load_yaml
from benchrep.assembly.registry import (
    EVAL_INTERNAL_CLUSTERING_METRICS,
    EVAL_EXTERNAL_CLUSTERING_METRICS,
    EVAL_EMBEDDING_METRICS,
    EVAL_RECONSTRUCTION_METRICS,
    EVAL_PREDICTABILITY_PROBES,
)
from benchrep.assembly.registry_utils import (
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
    embedding_metrics: list[str] | None
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

    error_maps_enabled: bool
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
    # Resolve manifest
    prediction_manifest_path = evaluation_config.source.prediction_manifest_path

    if prediction_manifest_path is not None:
        prediction_manifest_path = prediction_manifest_path.resolve()
        prediction_manifest = load_yaml(prediction_manifest_path)
        manifest_base_dir = prediction_manifest_path.parent
    else:
        prediction_manifest = None
        manifest_base_dir = None

    # Resolve embedding input: manual path overrides manifest
    if evaluation_config.source.embeddings_path is not None:
        embeddings_path = evaluation_config.source.embeddings_path.resolve()
    else:
        if prediction_manifest is None:
            raise ValueError(
                "source.embeddings_path is required when no prediction manifest "
                "is provided."
            )

        embeddings_path = get_required_nested_path(
            prediction_manifest,
            "exports",
            "embeddings",
            "path",
            base_dir=manifest_base_dir,
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
    manifest_n_examples = None

    if prediction_manifest is not None:
        manifest_n_examples = get_optional_nested_value(
            prediction_manifest,
            "exports",
            "reconstructions",
            "n_examples_exported",
        )

        if manifest_n_examples is not None and not isinstance(manifest_n_examples, int):
            warnings.warn(
                "Prediction manifest field "
                "'exports.reconstructions.n_examples_exported' is not an integer. "
                "Ignoring it for reconstruction artifact count resolution.",
                stacklevel=2,
            )
            manifest_n_examples = None

    resolved_n_examples = n_examples if n_examples is not None else manifest_n_examples

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
            n_examples=resolved_n_examples,
        )

    elif prediction_manifest is not None:
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
    clustering metrics, and plots default to enabled. t-SNE, KMeans, and
    embedding metrics default to disabled. Reconstruction metrics default to
    enabled only when reconstruction artifacts are available. Error maps require
    explicit enablement and available reconstructions.

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
    # Prep for guard to prevent clustering metric computation if no clustering is enabled.
    kmeans_enabled = disabled_by_default(evaluation_config.clustering.kmeans.enabled)
    leiden_enabled = enabled_by_default(evaluation_config.clustering.leiden.enabled)
    clustering_enabled = kmeans_enabled or leiden_enabled

    # Guard against key collision for kNNs of UMAP and Leiden (as current implementation is independent kNN ownership)
    umap_enabled = enabled_by_default(evaluation_config.reductions.umap.enabled)
    umap_params = params_to_dict(evaluation_config.reductions.umap.params)
    leiden_params = params_to_dict(evaluation_config.clustering.leiden.params)
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

    if predictability_enabled:
        _validate_predictability_task_and_scoring(
            task=predictability_task,
            tuning_params=predictability_tuning_params,
        )
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

    return EvaluationStepSpec(
        # None = True
        pca_enabled=enabled_by_default(evaluation_config.reductions.pca.enabled),
        pca_params=params_to_dict(evaluation_config.reductions.pca.params),

        # None = True
        umap_enabled=umap_enabled,
        umap_params=umap_params,

        # None = False
        tsne_enabled=disabled_by_default(evaluation_config.reductions.tsne.enabled),
        tsne_params=params_to_dict(evaluation_config.reductions.tsne.params),

        # None = False
        kmeans_enabled=kmeans_enabled,
        kmeans_params=params_to_dict(evaluation_config.clustering.kmeans.params),

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
        external_clustering_metrics_enabled=(
            evaluation_config.metrics.clustering.external.enabled
            if clustering_enabled
            else False
        ),
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

        # None = False
        embedding_metrics_enabled=disabled_by_default(evaluation_config.metrics.embedding.enabled),
        embedding_metrics=resolve_registry_keys(
            selected=evaluation_config.metrics.embedding.selected,
            registry=EVAL_EMBEDDING_METRICS,
            none_policy="preserve",
        ),
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

        # True if not disabled and has_reconstructions
        reconstruction_metrics_enabled=resolve_enabled_if_available(
            configured=evaluation_config.metrics.reconstruction.enabled,
            available=has_reconstructions,
            name="Reconstruction metrics",
        ),
        reconstruction_metrics=resolve_registry_keys(
            selected=evaluation_config.metrics.reconstruction.selected,
            registry=EVAL_RECONSTRUCTION_METRICS,
            none_policy="preserve",
        ),
        reconstruction_metric_params=resolve_registry_param_keys(
            params=evaluation_config.metrics.reconstruction.params,
            registry=EVAL_RECONSTRUCTION_METRICS,
        ),
        reconstruction_metrics_reduction=evaluation_config.metrics.reconstruction.reduction,

        # True only if explicitly enabled and reconstructions are available
        error_maps_enabled=resolve_enabled_if_explicit_and_available(
            configured=evaluation_config.reconstruction.error_maps.enabled,
            available=has_reconstructions,
            name="Reconstruction error maps",
        ),
        error_map_params=params_to_dict(evaluation_config.reconstruction.error_maps.params),

        # None = True
        plots_enabled=enabled_by_default(evaluation_config.plots.enabled),
        plot_params=params_to_dict(evaluation_config.plots.params),
    )


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


def _validate_predictability_task_and_scoring(
    *,
    task: PredictabilityTask,
    tuning_params: dict[str, Any],
) -> None:
    if not tuning_params.get("enabled", False):
        return

    inner_cv = tuning_params.get("inner_cv", {})
    scoring = inner_cv.get("scoring")

    valid_scorers = (
        CLASSIFICATION_SCORERS
        if task == "classification"
        else REGRESSION_SCORERS
    )

    if scoring not in valid_scorers:
        raise ValueError(
            f"metrics.predictability.tuning.inner_cv.scoring={scoring!r} "
            f"is not valid for task={task!r}."
        )


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