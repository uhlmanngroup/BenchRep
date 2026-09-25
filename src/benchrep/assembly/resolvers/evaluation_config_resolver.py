from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal
import warnings

from benchrep.assembly.config import load_yaml
from benchrep.assembly.registries.core import (
    Registry,
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
    get_optional_nested_value,
    params_to_dict,
)
from benchrep.assembly.schemas.evaluation_config_schema import (
    EvaluationConfig,
    EvaluationPredictabilityTargetConfig,
    EvaluationRunConfig,
    EvaluationLogisticRegressionProbeConfig,
    EvaluationRidgeProbeConfig,
)
from benchrep.runtime.status import ACCEPTABLE_PREDICTION_STATUSES


# -------------------------
# Registry selection defaults
# -------------------------
DEFAULT_INTERNAL_CLUSTERING_METRICS: tuple[str, ...] = (
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
)

DEFAULT_EXTERNAL_CLUSTERING_METRICS: tuple[str, ...] = (
    "adjusted_mutual_info",
    "adjusted_rand_index",
    "homogeneity",
)

DEFAULT_EMBEDDING_METRICS: tuple[str, ...] = (
    "mean",
    "median",
    "standard_deviation",
    "minimum",
    "maximum",
    "quantiles",
)

DEFAULT_PREDICTABILITY_PROBES: tuple[str, ...] = (
    "dummy",
    "linear",
    "knn",
    "random_forest",
    "svm_rbf",
)

DEFAULT_RECONSTRUCTION_METRICS: tuple[str, ...] = (
    "mae",
    "mse",
    "rmse",
    "max_absolute_error",
)


def _resolve_registry_selection(
    *,
    selected: Literal["all"] | list[str] | None,
    default_selection: tuple[str, ...],
    registry: Registry,
) -> list[str]:
    """Resolve an evaluation selection to unique canonical registry keys."""

    if selected == "all":
        return list(registry.canonical_keys())

    requested_selection = (
        default_selection
        if selected is None
        else selected
    )

    resolved_selection = resolve_registry_keys(
        selected=requested_selection,
        registry=registry,
        none_policy="preserve",
    )

    if resolved_selection is None:
        raise RuntimeError(
            f"{registry.name} selection unexpectedly resolved to None."
        )

    return resolved_selection


# -------------------------
# Resolved specs
# -------------------------
EvaluationInheritableField = Literal[
    "run.output_root",
    "source.anndata_path",
    "source.reconstructions_path",
    "reconstruction.n_examples",
]

EvaluationArtifactSource = Literal[
    "direct_path",
    "prediction_manifest",
]


@dataclass(frozen=True)
class EvaluationAnnDataInputSpec:
    """Resolved AnnData artifact selected for evaluation."""

    path: Path
    source: EvaluationArtifactSource


@dataclass(frozen=True)
class EvaluationReconstructionInputSpec:
    """Resolved reconstruction bundle selected for evaluation."""

    bundle_dir: Path
    source: EvaluationArtifactSource
    pair_id: str | None
    input_path: Path
    reconstruction_path: Path
    observations_path: Path
    metadata_path: Path | None
    n_examples: int | None


@dataclass(frozen=True)
class EvaluationInputSpec:
    """Resolved artifact inputs consumed by one evaluation run."""

    anndata: EvaluationAnnDataInputSpec | None
    reconstructions: EvaluationReconstructionInputSpec | None
    prediction_manifest_path: Path | None


PredictabilityTask = Literal["classification", "regression"]


@dataclass(frozen=True)
class PredictabilityTargetSpec:
    target_key: str
    task: PredictabilityTask
    probes: list[str]
    probe_params: dict[str, dict[str, Any]]
    cv_params: dict[str, Any]
    tuning_params: dict[str, Any]
    overwrite: bool


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

    hdbscan_enabled: bool
    hdbscan_params: dict[str, Any]

    internal_clustering_metrics_enabled: bool
    internal_clustering_metrics: list[str]
    internal_clustering_metric_params: dict[str, dict[str, Any]]
    internal_clustering_metrics_overwrite: bool

    external_clustering_metrics_enabled: bool | None
    external_clustering_label_key: str
    external_clustering_metrics: list[str]
    external_clustering_metric_params: dict[str, dict[str, Any]]
    external_clustering_metrics_overwrite: bool


    embedding_metrics_enabled: bool
    embedding_metrics: list[str]
    embedding_metric_params: dict[str, dict[str, Any]]
    embedding_metrics_overwrite: bool

    predictability_enabled: bool
    predictability_targets: tuple[PredictabilityTargetSpec, ...]

    reconstruction_metrics_enabled: bool
    reconstruction_metrics: list[str]
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
    inherited_config_fields: frozenset[EvaluationInheritableField]
    input_spec: EvaluationInputSpec
    step_spec: EvaluationStepSpec


def resolve_evaluation_config(
    evaluation_config: EvaluationConfig,
    prediction_manifest_path_override: Path | str | None = None,
) -> EvaluationRunSpec:
    """Resolve an evaluation config into its executable runtime specification.

    The resolver loads an optional prediction manifest, resolves at most one
    AnnData artifact and one reconstruction bundle, derives run identity and
    step availability, and materializes the concrete artifact paths used by
    the evaluation run.
    """

    inherited_config_fields: set[EvaluationInheritableField] = set()

    (
        evaluation_config,
        prediction_manifest_path,
    ) = _resolve_prediction_manifest_path(
        evaluation_config=evaluation_config,
        prediction_manifest_path_override=(
            prediction_manifest_path_override
        ),
    )

    if prediction_manifest_path is not None:
        prediction_manifest = _load_prediction_manifest(
            prediction_manifest_path
        )
        manifest_base_dir = prediction_manifest_path.parent
    else:
        prediction_manifest = None
        manifest_base_dir = None

    anndata_input = resolve_anndata_input(
        anndata_path=evaluation_config.source.anndata_path,
        prediction_manifest=prediction_manifest,
        manifest_base_dir=manifest_base_dir,
    )

    reconstruction_input = resolve_reconstructions(
        reconstructions_path=(
            evaluation_config.source.reconstructions_path
        ),
        reconstruction_pair_id=(
            evaluation_config.source.reconstruction_pair_id
        ),
        prediction_manifest=prediction_manifest,
        manifest_base_dir=manifest_base_dir,
        n_examples=evaluation_config.reconstruction.n_examples,
    )

    if (
        anndata_input is not None
        and anndata_input.source == "prediction_manifest"
    ):
        inherited_config_fields.add("source.anndata_path")

    if (
        reconstruction_input is not None
        and reconstruction_input.source == "prediction_manifest"
    ):
        inherited_config_fields.add(
            "source.reconstructions_path"
        )

        if (
            evaluation_config.reconstruction.n_examples is None
            and reconstruction_input.n_examples is not None
        ):
            inherited_config_fields.add(
                "reconstruction.n_examples"
            )

    has_embeddings = anndata_input is not None
    has_reconstructions = reconstruction_input is not None

    if not has_embeddings and not has_reconstructions:
        raise ValueError(
            "Evaluation could not resolve a usable AnnData artifact or "
            "reconstruction bundle from the configured sources."
        )

    run_identity = resolve_run_identity(
        run_config=evaluation_config.run,
        prediction_manifest=prediction_manifest,
        manifest_base_dir=manifest_base_dir,
    )

    if (
        evaluation_config.run.output_root is None
        and prediction_manifest is not None
    ):
        assert manifest_base_dir is not None

        prediction_output_dir = get_optional_nested_path(
            prediction_manifest,
            "run",
            "output_dir",
            base_dir=manifest_base_dir,
        )

        if prediction_output_dir is not None:
            inherited_config_fields.add("run.output_root")

    step_spec = resolve_step_spec(
        evaluation_config=evaluation_config,
        has_embeddings=has_embeddings,
        has_reconstructions=has_reconstructions,
    )

    resolved_anndata_path = (
        anndata_input.path
        if anndata_input is not None
        else evaluation_config.source.anndata_path
    )
    resolved_reconstructions_path = (
        reconstruction_input.bundle_dir
        if reconstruction_input is not None
        else evaluation_config.source.reconstructions_path
    )

    resolved_source_config = evaluation_config.source.model_copy(
        update={
            "prediction_manifest_path": prediction_manifest_path,
            "anndata_path": resolved_anndata_path,
            "reconstructions_path": resolved_reconstructions_path,
            "reconstruction_pair_id": None,
        },
    )

    resolved_run_config = evaluation_config.run

    if "run.output_root" in inherited_config_fields:
        resolved_run_config = evaluation_config.run.model_copy(
            update={"output_root": run_identity.output_root},
        )

    resolved_reconstruction_config = (
        evaluation_config.reconstruction
    )

    if "reconstruction.n_examples" in inherited_config_fields:
        assert reconstruction_input is not None
        assert reconstruction_input.n_examples is not None

        resolved_reconstruction_config = (
            evaluation_config.reconstruction.model_copy(
                update={
                    "n_examples": reconstruction_input.n_examples,
                },
            )
        )

    evaluation_config = evaluation_config.model_copy(
        update={
            "source": resolved_source_config,
            "run": resolved_run_config,
            "reconstruction": resolved_reconstruction_config,
        },
    )

    input_spec = EvaluationInputSpec(
        anndata=anndata_input,
        reconstructions=reconstruction_input,
        prediction_manifest_path=prediction_manifest_path,
    )

    return EvaluationRunSpec(
        stage=evaluation_config.stage,
        evaluation_config=evaluation_config,
        prediction_manifest=prediction_manifest,
        run_identity=run_identity,
        inherited_config_fields=frozenset[EvaluationInheritableField](
            inherited_config_fields
        ),
        input_spec=input_spec,
        step_spec=step_spec,
    )


def resolve_anndata_input(
    *,
    anndata_path: Path | None,
    prediction_manifest: dict[str, Any] | None,
    manifest_base_dir: Path | None,
) -> EvaluationAnnDataInputSpec | None:
    """Resolve the AnnData artifact whose ``X`` is evaluated.

    A directly configured path takes precedence over the AnnData artifact
    recorded in the prediction manifest.
    """

    if anndata_path is not None:
        return EvaluationAnnDataInputSpec(
            path=anndata_path.expanduser().resolve(),
            source="direct_path",
        )

    if prediction_manifest is None:
        return None

    assert manifest_base_dir is not None

    resolved_path = get_optional_nested_path(
        prediction_manifest,
        "exports",
        "anndata",
        "path",
        base_dir=manifest_base_dir,
    )

    if resolved_path is None:
        return None

    return EvaluationAnnDataInputSpec(
        path=resolved_path,
        source="prediction_manifest",
    )


def resolve_reconstructions(
    *,
    reconstructions_path: Path | None,
    reconstruction_pair_id: str | None,
    prediction_manifest: dict[str, Any] | None,
    manifest_base_dir: Path | None,
    n_examples: int | None,
) -> EvaluationReconstructionInputSpec | None:
    """Resolve one complete reconstruction bundle for evaluation.

    A directly configured bundle takes precedence over manifest-derived
    reconstruction artifacts and must contain all required files.

    For manifest-derived input, an explicit pair identifier selects that pair.
    Otherwise, the pair is inferred only when exactly one complete pair is
    available. An ambiguous set of usable pairs leaves reconstruction input
    unavailable until the user selects one.
    """

    if reconstructions_path is not None:
        bundle_dir = reconstructions_path.expanduser().resolve()

        input_path, reconstruction_path, observations_path, metadata_path = (
            _validate_reconstruction_bundle_dir(bundle_dir)
        )

        return EvaluationReconstructionInputSpec(
            bundle_dir=bundle_dir,
            source="direct_path",
            pair_id=None,
            input_path=input_path,
            reconstruction_path=reconstruction_path,
            observations_path=observations_path,
            metadata_path=metadata_path,
            n_examples=n_examples,
        )

    if prediction_manifest is None:
        if reconstruction_pair_id is not None:
            raise ValueError(
                "source.reconstruction_pair_id requires a prediction manifest."
            )

        return None

    assert manifest_base_dir is not None

    pair_records = get_optional_nested_value(
        prediction_manifest,
        "exports",
        "reconstructions",
        "pairs",
    )

    if pair_records is None:
        if reconstruction_pair_id is not None:
            raise ValueError(
                "source.reconstruction_pair_id was configured, but the "
                "prediction manifest does not contain reconstruction-pair "
                "records."
            )

        return None

    if not isinstance(pair_records, Mapping):
        raise TypeError(
            "Prediction manifest field "
            "`exports.reconstructions.pairs` must be a mapping."
        )

    usable_pairs: dict[
        str,
        tuple[Path, Path, Path, Path, Path | None],
    ] = {}
    incomplete_pairs: dict[str, str] = {}

    for candidate_id, pair_record in pair_records.items():
        if not isinstance(candidate_id, str):
            raise TypeError(
                "Prediction manifest reconstruction pair identifiers must "
                "be strings."
            )

        if not isinstance(pair_record, Mapping):
            raise TypeError(
                "Prediction manifest reconstruction pair "
                f"{candidate_id!r} must be a mapping."
            )

        bundle_dir = get_optional_nested_path(
            prediction_manifest,
            "exports",
            "reconstructions",
            "pairs",
            candidate_id,
            "paths",
            "bundle_dir",
            base_dir=manifest_base_dir,
        )

        if bundle_dir is None:
            incomplete_pairs[candidate_id] = (
                "the manifest does not record `paths.bundle_dir`"
            )
            continue

        try:
            (
                input_path,
                reconstruction_path,
                observations_path,
                metadata_path,
            ) = _validate_reconstruction_bundle_dir(bundle_dir)

        except (NotADirectoryError, FileNotFoundError) as exc:
            incomplete_pairs[candidate_id] = str(exc)
            continue

        usable_pairs[candidate_id] = (
            bundle_dir,
            input_path,
            reconstruction_path,
            observations_path,
            metadata_path,
        )

    if reconstruction_pair_id is not None:
        if reconstruction_pair_id not in pair_records:
            raise ValueError(
                "Evaluation reconstruction pair "
                f"{reconstruction_pair_id!r} is not recorded in the "
                "prediction manifest. Available pairs: "
                f"{list(pair_records)}."
            )

        if reconstruction_pair_id not in usable_pairs:
            raise FileNotFoundError(
                "Evaluation reconstruction pair "
                f"{reconstruction_pair_id!r} does not contain a complete "
                "artifact bundle. "
                f"{incomplete_pairs.get(reconstruction_pair_id)}"
            )

        selected_pair_id = reconstruction_pair_id

    elif len(usable_pairs) == 1:
        selected_pair_id = next(iter(usable_pairs))

    elif len(usable_pairs) > 1:
        warnings.warn(
            "Prediction manifest contains multiple usable reconstruction "
            "pairs, but source.reconstruction_pair_id was not configured. "
            "Reconstruction-dependent evaluation will remain unavailable. "
            f"Available pairs: {list(usable_pairs)}.",
            UserWarning,
            stacklevel=2,
        )
        return None

    else:
        if incomplete_pairs:
            warnings.warn(
                "Prediction manifest does not contain a complete "
                "reconstruction artifact bundle. Reconstruction-dependent "
                "evaluation will remain unavailable. Incomplete pairs: "
                f"{incomplete_pairs}.",
                UserWarning,
                stacklevel=2,
            )

        return None

    (
        bundle_dir,
        input_path,
        reconstruction_path,
        observations_path,
        metadata_path,
    ) = usable_pairs[selected_pair_id]

    pair_record = pair_records[selected_pair_id]
    manifest_n_examples = pair_record.get("n_examples_exported")

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
            f"`exports.reconstructions.pairs.{selected_pair_id}."
            "n_examples_exported` is not a positive integer. Ignoring it.",
            UserWarning,
            stacklevel=2,
        )
        manifest_n_examples = None

    resolved_n_examples = (
        n_examples
        if n_examples is not None
        else manifest_n_examples
    )

    return EvaluationReconstructionInputSpec(
        bundle_dir=bundle_dir,
        source="prediction_manifest",
        pair_id=selected_pair_id,
        input_path=input_path,
        reconstruction_path=reconstruction_path,
        observations_path=observations_path,
        metadata_path=metadata_path,
        n_examples=resolved_n_examples,
    )


def _validate_reconstruction_bundle_dir(
    bundle_dir: Path,
) -> tuple[Path, Path, Path, Path | None]:
    """Validate a reconstruction bundle and return its artifact paths."""

    bundle_dir = bundle_dir.expanduser().resolve()

    if not bundle_dir.is_dir():
        raise NotADirectoryError(
            "Reconstruction bundle path must point to a directory. "
            f"Got: {bundle_dir}"
        )

    input_path = bundle_dir / "input.pt"
    reconstruction_path = bundle_dir / "reconstruction.pt"
    observations_path = bundle_dir / "obs.pt"
    metadata_path = (
        bundle_dir / "reconstruction_export_metadata.pt"
    )

    required_paths = {
        "input": input_path,
        "reconstruction": reconstruction_path,
        "observations": observations_path,
    }
    missing = [
        name
        for name, path in required_paths.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"Reconstruction bundle '{bundle_dir}' is incomplete. "
            f"Missing required artifacts: {missing}."
        )

    return (
        input_path,
        reconstruction_path,
        observations_path,
        metadata_path if metadata_path.is_file() else None,
    )


def resolve_run_identity(
    run_config: EvaluationRunConfig,
    prediction_manifest: dict[str, Any] | None = None,
    manifest_base_dir: Path | None = None,
) -> EvaluationRunIdentitySpec:
    """Resolve output location and identity hints for an evaluation run.

    Output-root, project, and model identity are inferred from the linked
    prediction lineage when not configured directly. ``RunContext`` remains
    responsible for constructing the final timestamped run name.
    """

    if run_config.output_root is not None:
        output_root = run_config.output_root.resolve()

    elif prediction_manifest is not None:
        assert manifest_base_dir is not None

        prediction_output_dir = get_optional_nested_path(
            prediction_manifest,
            "run",
            "output_dir",
            base_dir=manifest_base_dir,
        )

        if prediction_output_dir is None:
            warnings.warn(
                "Could not infer the evaluation output root from the "
                "prediction manifest because `run.output_dir` is missing. "
                "Falling back to `outputs/`.",
                UserWarning,
                stacklevel=2,
            )
            output_root = Path("outputs").resolve()
        else:
            output_root = prediction_output_dir.parent.parent

    else:
        output_root = Path("outputs").resolve()

    run_name_stem = run_config.run_name

    if prediction_manifest is None:
        project_name = None
        model_name = None

    else:
        project_name_value = get_optional_nested_value(
            prediction_manifest,
            "appendix",
            "training_manifest",
            "appendix",
            "resolved_config",
            "run",
            "project_name",
        )
        project_name = (
            project_name_value
            if isinstance(project_name_value, str)
            else None
        )

        model_name_value = get_optional_nested_value(
            prediction_manifest,
            "summary",
            "model_name",
        )
        model_class_value = get_optional_nested_value(
            prediction_manifest,
            "summary",
            "model_class",
        )

        if isinstance(model_name_value, str):
            model_name = model_name_value
        elif isinstance(model_class_value, str):
            model_name = model_class_value
        else:
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
        has_embeddings: bool,
        has_reconstructions: bool,
) -> EvaluationStepSpec:
    """Resolve evaluation step switches, selections, and backend parameters.

    Evaluation `enabled` fields use tri-state semantics. `True` explicitly
    requests a step and raises when its required source inputs are unavailable.
    `False` disables it. `None` uses the concrete step's automatic behavior and
    does not raise merely because optional inputs are unavailable.

    PCA, embedding metrics, and reconstruction metrics default to enabled when
    their respective inputs exist. UMAP, t-SNE, KMeans, Leiden, HDBSCAN,
    predictability, and reconstruction TIFF export require explicit enablement.
    Plots default to enabled.

    Internal and external clustering metrics require at least one enabled
    clustering method. Explicitly enabling either group without one raises an
    error; automatic enablement disables the group. External metric enablement
    may remain `None` until the loaded AnnData object can be checked for
    `label_key`.

    Registry selections resolve to canonical names. `None` uses the group's
    curated defaults, `"all"` uses every currently registered component, and an
    explicit list resolves exactly the provided names or aliases.

    Parameter objects are converted to dictionaries for downstream use. `None`
    parameters are omitted so callable or backend defaults can apply.
    """
    # Resolve step switches and params needed by multiple downstream decisions.
    pca_enabled = resolve_enabled_if_available(
        configured=evaluation_config.reductions.pca.enabled,
        available=has_embeddings,
        name="PCA",
    )
    pca_params = params_to_dict(evaluation_config.reductions.pca.params)

    umap_enabled = resolve_enabled_if_explicit_and_available(
        configured=evaluation_config.reductions.umap.enabled,
        available=has_embeddings,
        name="UMAP",
    )
    umap_params = params_to_dict(evaluation_config.reductions.umap.params)

    tsne_enabled = resolve_enabled_if_explicit_and_available(
        configured=evaluation_config.reductions.tsne.enabled,
        available=has_embeddings,
        name="t-SNE",
    )
    tsne_params = params_to_dict(evaluation_config.reductions.tsne.params)

    kmeans_enabled = resolve_enabled_if_explicit_and_available(
        configured=evaluation_config.clustering.kmeans.enabled,
        available=has_embeddings,
        name="KMeans",
    )
    kmeans_params = params_to_dict(evaluation_config.clustering.kmeans.params)

    leiden_enabled = resolve_enabled_if_explicit_and_available(
        configured=evaluation_config.clustering.leiden.enabled,
        available=has_embeddings,
        name="Leiden",
    )
    leiden_params = params_to_dict(evaluation_config.clustering.leiden.params)

    hdbscan_enabled = resolve_enabled_if_explicit_and_available(
        configured=evaluation_config.clustering.hdbscan.enabled,
        available=has_embeddings,
        name="HDBSCAN",
    )
    hdbscan_params = params_to_dict(evaluation_config.clustering.hdbscan.params)

    # Prep for guard to prevent clustering metric computation if no clustering is enabled.
    clustering_enabled = kmeans_enabled or leiden_enabled or hdbscan_enabled

    internal_metrics_config = (
        evaluation_config.metrics.clustering.internal
    )

    if (
            internal_metrics_config.enabled is True
            and not clustering_enabled
    ):
        raise ValueError(
            "Internal clustering metrics were explicitly enabled, but no "
            "clustering method is enabled."
        )

    internal_clustering_metrics_enabled = (
            clustering_enabled
            and internal_metrics_config.enabled is not False
    )

    internal_clustering_metrics = _resolve_registry_selection(
        selected=internal_metrics_config.selected,
        default_selection=DEFAULT_INTERNAL_CLUSTERING_METRICS,
        registry=EVAL_INTERNAL_CLUSTERING_METRICS,
    )

    if (
            internal_clustering_metrics_enabled
            and not internal_clustering_metrics
    ):
        raise ValueError(
            "metrics.clustering.internal.selected cannot be empty when internal "
            "clustering metrics are enabled."
        )

    external_metrics_config = (
        evaluation_config.metrics.clustering.external
    )

    if (
            external_metrics_config.enabled is True
            and not clustering_enabled
    ):
        raise ValueError(
            "External clustering metrics were explicitly enabled, but no "
            "clustering method is enabled."
        )

    external_clustering_metrics_enabled = (
        external_metrics_config.enabled
        if clustering_enabled
        else False
    )

    external_clustering_metrics = _resolve_registry_selection(
        selected=external_metrics_config.selected,
        default_selection=DEFAULT_EXTERNAL_CLUSTERING_METRICS,
        registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
    )

    if (
            external_clustering_metrics_enabled is True
            and not external_clustering_metrics
    ):
        raise ValueError(
            "metrics.clustering.external.selected cannot be empty when external "
            "clustering metrics are enabled."
        )

    embedding_metrics_enabled = resolve_enabled_if_available(
        configured=evaluation_config.metrics.embedding.enabled,
        available=has_embeddings,
        name="Embedding metrics",
    )

    embedding_metrics = _resolve_registry_selection(
        selected=evaluation_config.metrics.embedding.selected,
        default_selection=DEFAULT_EMBEDDING_METRICS,
        registry=EVAL_EMBEDDING_METRICS,
    )

    if embedding_metrics_enabled and not embedding_metrics:
        raise ValueError(
            "metrics.embedding.selected cannot be empty when embedding metrics "
            "are enabled."
        )

    plot_params = params_to_dict(evaluation_config.plots.params)
    plot_params = _resolve_plot_params(
        plot_params=plot_params,
        kmeans_enabled=kmeans_enabled,
        kmeans_params=kmeans_params,
        leiden_enabled=leiden_enabled,
        leiden_params=leiden_params,
        hdbscan_enabled=hdbscan_enabled,
        hdbscan_params=hdbscan_params,
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
    predictability_enabled = resolve_enabled_if_explicit_and_available(
        configured=predictability_config.enabled,
        available=has_embeddings,
        name="Predictability evaluation",
    )
    resolved_predictability_targets: list[
        PredictabilityTargetSpec
    ] = []

    for target_key, target_config in predictability_config.targets.items():
        try:
            target_spec = _resolve_predictability_target(
                target_key=target_key,
                target_config=target_config,
                enabled=predictability_enabled,
            )
        except (KeyError, TypeError, ValueError) as error:
            target_path = (
                f"metrics.predictability.targets[{target_key!r}]"
            )
            raise ValueError(
                f"Failed to resolve {target_path}. Original error "
                f"({type(error).__name__}): {error}"
            ) from error

        resolved_predictability_targets.append(target_spec)

    predictability_targets = tuple(resolved_predictability_targets)

    reconstruction_metrics_enabled = resolve_enabled_if_available(
        configured=evaluation_config.metrics.reconstruction.enabled,
        available=has_reconstructions,
        name="Reconstruction metrics",
    )

    reconstruction_metrics = _resolve_registry_selection(
        selected=evaluation_config.metrics.reconstruction.selected,
        default_selection=DEFAULT_RECONSTRUCTION_METRICS,
        registry=EVAL_RECONSTRUCTION_METRICS,
    )

    if reconstruction_metrics_enabled and not reconstruction_metrics:
        raise ValueError(
            "metrics.reconstruction.selected cannot be empty when reconstruction "
            "metrics are enabled."
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

    step_spec = EvaluationStepSpec(
        # None = True
        pca_enabled=pca_enabled,
        pca_params=pca_params,

        # None = False
        umap_enabled=umap_enabled,
        umap_params=umap_params,

        # None = False
        tsne_enabled=tsne_enabled,
        tsne_params=tsne_params,

        # None = False
        kmeans_enabled=kmeans_enabled,
        kmeans_params=kmeans_params,

        # None = False
        leiden_enabled=leiden_enabled,
        leiden_params=leiden_params,

        # None = False
        hdbscan_enabled=hdbscan_enabled,
        hdbscan_params=hdbscan_params,

        # None = True
        # Force disable if not clustering_enabled
        internal_clustering_metrics_enabled=internal_clustering_metrics_enabled,
        internal_clustering_metrics=internal_clustering_metrics,
        internal_clustering_metric_params=resolve_registry_param_keys(
            params=evaluation_config.metrics.clustering.internal.params,
            registry=EVAL_INTERNAL_CLUSTERING_METRICS,
        ),
        internal_clustering_metrics_overwrite=(
                internal_metrics_config.overwrite is True
        ),

        # None is resolved later after loading AnnData and checking adata.obs
        # Force disable if not clustering_enabled
        external_clustering_metrics_enabled=external_clustering_metrics_enabled,
        external_clustering_label_key=evaluation_config.metrics.clustering.external.label_key,
        external_clustering_metrics=external_clustering_metrics,
        external_clustering_metric_params=resolve_registry_param_keys(
            params=evaluation_config.metrics.clustering.external.params,
            registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
        ),
        external_clustering_metrics_overwrite=(
                external_metrics_config.overwrite is True
        ),

        # None = True
        embedding_metrics_enabled=embedding_metrics_enabled,
        embedding_metrics=embedding_metrics,
        embedding_metric_params=resolve_registry_param_keys(
            params=evaluation_config.metrics.embedding.params,
            registry=EVAL_EMBEDDING_METRICS,
        ),
        embedding_metrics_overwrite=(
                evaluation_config.metrics.embedding.overwrite is True
        ),

        # None = False
        predictability_enabled=predictability_enabled,
        predictability_targets=predictability_targets,

        # True if not disabled and reconstructions are available
        reconstruction_metrics_enabled=reconstruction_metrics_enabled,
        reconstruction_metrics=reconstruction_metrics,
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

    _validate_has_enabled_evaluation_work(
        step_spec=step_spec,
        has_reconstructions=has_reconstructions,
    )

    return step_spec


def _resolve_plot_params(
    *,
    plot_params: dict[str, Any],
    kmeans_enabled: bool,
    kmeans_params: dict[str, Any],
    leiden_enabled: bool,
    leiden_params: dict[str, Any],
    hdbscan_enabled: bool,
    hdbscan_params: dict[str, Any],
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

    if hdbscan_enabled:
        color_by.append(
            hdbscan_params.get("key_added", "hdbscan")
        )

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


def resolve_enabled_if_available(
    configured: bool | None,
    *,
    available: bool,
    name: str,
) -> bool:
    """Resolve a step that defaults to enabled when its inputs are available.

    `False` always disables the step. `None` enables it only when the required
    inputs are available. `True` requires those inputs and raises when they are
    unavailable.
    """

    if configured is True and not available:
        raise ValueError(
            f"{name} was explicitly enabled, but its required inputs are "
            "unavailable."
        )

    if not available:
        return False

    return configured is not False


def resolve_enabled_if_explicit_and_available(
    configured: bool | None,
    *,
    available: bool,
    name: str,
) -> bool:
    """Resolve a step that runs only when explicitly enabled.

    `False` and `None` disable the step. `True` enables it when its required
    inputs are available and raises when they are unavailable.
    """

    if configured is not True:
        return False

    if not available:
        raise ValueError(
            f"{name} was explicitly enabled, but its required inputs are "
            "unavailable."
        )

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


def _resolve_predictability_cv_method(
    *,
    task: PredictabilityTask,
    cv_params: Mapping[str, Any],
) -> str:
    """Resolve and validate the predictability cross-validation strategy."""

    method = cv_params.get("method")

    if method is not None and not isinstance(method, str):
        raise TypeError(
            "cv.method must be a string or None, "
            f"got {type(method).__name__}."
        )

    if method is None:
        if task == "classification":
            return "stratified_kfold"

        if task == "regression":
            return "kfold"

        raise ValueError(
            "task must be either 'classification' or 'regression', "
            f"got {task!r}."
        )

    if (
        task == "regression"
        and method in {"stratified_kfold", "stratified_group_kfold"}
    ):
        raise ValueError(
            f"cv.method={method!r} is not valid for "
            "task='regression'."
        )

    return method


def _resolve_predictability_scoring(
    *,
    task: PredictabilityTask,
    cv_params: dict[str, Any],
) -> str:
    scoring = cv_params.get("scoring")

    if scoring is not None and not isinstance(scoring, str):
        raise TypeError(
            "cv.scoring must be a string or None, "
            f"got {type(scoring).__name__}."
        )

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
            f"cv.scoring={scoring!r} "
            f"is not valid for task={task!r}."
        )

    return scoring


def _resolve_predictability_probe_params(
    *,
    task: PredictabilityTask,
    probe_params: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve task-dependent defaults for predictability probes."""

    resolved_params = {
        probe_name: dict(params)
        for probe_name, params in probe_params.items()
    }

    dummy_params = resolved_params.setdefault("dummy", {})

    if "strategy" not in dummy_params:
        if task == "classification":
            dummy_params["strategy"] = "most_frequent"
        elif task == "regression":
            dummy_params["strategy"] = "mean"
        else:
            raise ValueError(
                "task must be either 'classification' or 'regression', "
                f"got {task!r}."
            )

    if "linear" not in resolved_params:
        if task == "classification":
            linear_config = EvaluationLogisticRegressionProbeConfig()
        elif task == "regression":
            linear_config = EvaluationRidgeProbeConfig()
        else:
            raise ValueError(
                "task must be either 'classification' or 'regression', "
                f"got {task!r}."
            )

        resolved_params["linear"] = params_to_dict(linear_config)

    return resolved_params


def _validate_predictability_dummy_strategy(
    *,
    task: PredictabilityTask,
    probes: list[str],
    probe_params: dict[str, dict[str, Any]],
) -> None:
    """Validate the dummy strategy when the dummy probe is selected."""

    if "dummy" not in probes:
        return

    strategy = probe_params.get("dummy", {}).get("strategy")

    valid_strategies = (
        {"most_frequent", "stratified", "uniform"}
        if task == "classification"
        else {"mean", "median"}
    )

    if strategy not in valid_strategies:
        raise ValueError(
            "params.dummy.strategy must be one of "
            f"{sorted(valid_strategies)} when task={task!r}."
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
            "params.linear.model must be "
            "'logistic_regression' when task='classification'."
        )

    if task == "regression" and linear_model != "ridge":
        raise ValueError(
            "params.linear.model must be "
            "'ridge' when task='regression'."
        )


def _validate_predictability_class_weight(
    *,
    task: PredictabilityTask,
    probes: list[str],
    probe_params: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject classification-only class weighting for regression probes."""

    if task != "regression":
        return

    incompatible_probes = [
        probe_name
        for probe_name in ("random_forest", "svm_rbf")
        if (
            probe_name in probes
            and "class_weight" in probe_params.get(probe_name, {})
        )
    ]

    if incompatible_probes:
        raise ValueError(
            "`class_weight` is only valid for classification predictability "
            "probes, but it was configured for regression probes "
            f"{incompatible_probes}."
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
            "tuning.enabled=True, but no list-valued "
            "hyperparameters were found for the selected probes."
        )

    if not tuning_enabled and has_grid_params:
        raise ValueError(
            "tuning.enabled=False, but list-valued "
            "hyperparameters were found for the selected probes."
        )


def _resolve_predictability_target(
    *,
    target_key: str,
    target_config: EvaluationPredictabilityTargetConfig,
    enabled: bool,
) -> PredictabilityTargetSpec:
    """Resolve one predictability target and its task-dependent settings."""

    task: PredictabilityTask = target_config.task

    probes = _resolve_registry_selection(
        selected=target_config.selected,
        default_selection=DEFAULT_PREDICTABILITY_PROBES,
        registry=EVAL_PREDICTABILITY_PROBES,
    )

    if enabled and not probes:
        raise ValueError(
            "metrics.predictability.targets"
            f"[{target_key!r}].selected cannot be empty when predictability "
            "is enabled."
        )

    probe_params = resolve_registry_param_keys(
        params=params_to_dict(target_config.params),
        registry=EVAL_PREDICTABILITY_PROBES,
    )
    probe_params = _resolve_predictability_probe_params(
        task=task,
        probe_params=probe_params,
    )

    cv_params = params_to_dict(target_config.cv)
    cv_params["method"] = _resolve_predictability_cv_method(
        task=task,
        cv_params=cv_params,
    )
    cv_params["scoring"] = _resolve_predictability_scoring(
        task=task,
        cv_params=cv_params,
    )

    tuning_params = params_to_dict(target_config.tuning)

    if enabled:
        _validate_predictability_dummy_strategy(
            task=task,
            probes=probes,
            probe_params=probe_params,
        )
        _validate_predictability_linear_model(
            task=task,
            probes=probes,
            probe_params=probe_params,
        )
        _validate_predictability_class_weight(
            task=task,
            probes=probes,
            probe_params=probe_params,
        )
        _validate_predictability_tuning_grid(
            probes=probes,
            tuning_params=tuning_params,
            probe_params=probe_params,
        )

    return PredictabilityTargetSpec(
        target_key=target_key,
        task=task,
        probes=probes,
        probe_params=probe_params,
        cv_params=cv_params,
        tuning_params=tuning_params,
        overwrite=target_config.overwrite is True,
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
    if manifest_status not in ACCEPTABLE_PREDICTION_STATUSES:
        raise ValueError(
            "Evaluation accepts a provided prediction manifest only when its "
            f"status is in {sorted(ACCEPTABLE_PREDICTION_STATUSES)}, "
            f"but manifest status is {manifest_status!r}."
        )

    if manifest_status == "partially_completed":
        warnings.warn(
            "Prediction manifest is partially completed. Evaluation will proceed "
            "using available artifacts. Missing anndata will still prevent "
            "evaluation, while unavailable reconstruction artifacts will disable "
            "reconstruction-dependent steps.",
            UserWarning,
            stacklevel=2,
        )

    return prediction_manifest


def _validate_has_enabled_evaluation_work(
    *,
    step_spec: EvaluationStepSpec,
    has_reconstructions: bool,
) -> None:
    """Reject evaluation runs that resolve to no meaningful work.

    Resolved evaluation-step switches use the ``_enabled`` suffix. Plotting is
    handled separately because enabling plots alone does not guarantee that any
    artifact can be produced.
    """
    has_enabled_evaluation_step = any(
        getattr(step_spec, field.name) is not False
        for field in fields(step_spec)
        if field.name.endswith("_enabled")
        and field.name != "plots_enabled"
    )

    has_reconstruction_grid_export = (
        step_spec.plots_enabled
        and has_reconstructions
    )

    if (
        has_enabled_evaluation_step
        or has_reconstruction_grid_export
    ):
        return

    raise ValueError(
        "Evaluation configuration resolves to no enabled evaluation "
        "or artifact-producing steps."
    )
