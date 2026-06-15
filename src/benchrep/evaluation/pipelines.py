from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import anndata as ad

from benchrep.assembly.registry import (
    EVAL_CLUSTERING_METHODS,
    EVAL_REDUCTIONS,
)
from benchrep.evaluation.embeddings.clustering_metrics import (
    compute_external_clustering_metrics,
    compute_internal_clustering_metrics,
)
from benchrep.evaluation.reconstructions.error_maps import compute_error_maps
from benchrep.evaluation.reconstructions.reconstruction_metrics import (
    compute_reconstruction_metrics,
)

if TYPE_CHECKING:
    from benchrep.assembly.resolvers.evaluation_config_resolver import (
        EvaluationRunSpec,
    )


# -------------------------
# Step specs
# -------------------------
@dataclass(frozen=True)
class AnnDataEvaluationStep:
    """A single AnnData-based evaluation step.

    The wrapped function must follow the BenchRep AnnData evaluation contract:

        AnnData -> AnnData

    The function may mutate the input AnnData in place, but it must still return
    the updated AnnData so steps can be chained by the pipeline.
    """

    name: str
    fn: Callable[..., ad.AnnData]
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def run(self, adata: ad.AnnData) -> ad.AnnData:
        """Run the step on an AnnData object."""

        if not self.enabled:
            return adata

        result = self.fn(adata, **dict(self.params))

        if not isinstance(result, ad.AnnData):
            raise TypeError(
                f"AnnData evaluation step {self.name!r} must return an AnnData "
                f"object, got {type(result).__name__}."
            )

        return result


@dataclass(frozen=True)
class ReconstructionEvaluationStep:
    """A single reconstruction-based evaluation step.

    Reconstruction steps use a looser contract than AnnData steps because
    reconstruction artifacts are not naturally stored in AnnData. The wrapped
    function receives the reconstruction input object plus step params and
    returns a mapping of outputs, metrics, or artifact references.
    """

    name: str
    fn: Callable[..., Mapping[str, Any]]
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def run(self, reconstruction_input: Any) -> dict[str, Any]:
        """Run the step on reconstruction input data."""

        if not self.enabled:
            return {}

        result = self.fn(reconstruction_input, **dict(self.params))

        if not isinstance(result, Mapping):
            raise TypeError(
                f"Reconstruction evaluation step {self.name!r} must return a "
                f"mapping, got {type(result).__name__}."
            )

        return dict(result)


# -------------------------
# Pipelines
# -------------------------
class AnnDataEvaluationPipeline:
    """Sequential pipeline for AnnData-based evaluation steps.

    The pipeline owns only orchestration: it runs enabled steps in order and
    passes the updated AnnData object from one step to the next. Actual
    evaluation logic should live in the step functions.
    """

    def __init__(
        self,
        steps: Sequence[AnnDataEvaluationStep],
    ) -> None:
        self.steps = list(steps)

    def run(self, adata: ad.AnnData) -> ad.AnnData:
        """Run all enabled AnnData evaluation steps."""

        for step in self.steps:
            adata = step.run(adata)

        return adata


class ReconstructionEvaluationPipeline:
    """Sequential pipeline for reconstruction-based evaluation steps.

    Each enabled step receives the same reconstruction input object. Step outputs
    are collected into a nested dictionary keyed by step name.
    """

    def __init__(
        self,
        steps: Sequence[ReconstructionEvaluationStep],
    ) -> None:
        self.steps = list(steps)

    def run(self, reconstruction_input: Any | None) -> dict[str, Any]:
        """Run all enabled reconstruction evaluation steps."""

        if reconstruction_input is None:
            return {}

        outputs: dict[str, Any] = {}

        for step in self.steps:
            step_output = step.run(reconstruction_input)

            if not step_output:
                continue

            if step.name in outputs:
                raise KeyError(
                    f"Reconstruction pipeline output already contains step "
                    f"{step.name!r}."
                )

            outputs[step.name] = step_output

        return outputs


# -------------------------
# Pipeline creation
# -------------------------
def create_anndata_evaluation_pipeline(
    run_spec: "EvaluationRunSpec",
) -> AnnDataEvaluationPipeline:
    """Create the AnnData evaluation pipeline from a resolved run spec.

    This function translates the resolved evaluation step spec into ordered
    ``AnnDataEvaluationStep`` objects. It owns only workflow wiring: step order,
    registry lookup for reduction/clustering callables, and metric-runner setup.
    Actual computation remains in the low-level evaluation functions.
    """

    step_spec = run_spec.step_spec

    steps: list[AnnDataEvaluationStep] = [
        AnnDataEvaluationStep(
            name="pca",
            fn=EVAL_REDUCTIONS.get("pca"),
            params=step_spec.pca_params,
            enabled=step_spec.pca_enabled,
        ),
        AnnDataEvaluationStep(
            name="umap",
            fn=EVAL_REDUCTIONS.get("umap"),
            params=step_spec.umap_params,
            enabled=step_spec.umap_enabled,
        ),
        AnnDataEvaluationStep(
            name="tsne",
            fn=EVAL_REDUCTIONS.get("tsne"),
            params=step_spec.tsne_params,
            enabled=step_spec.tsne_enabled,
        ),
        AnnDataEvaluationStep(
            name="kmeans",
            fn=EVAL_CLUSTERING_METHODS.get("kmeans"),
            params=step_spec.kmeans_params,
            enabled=step_spec.kmeans_enabled,
        ),
        AnnDataEvaluationStep(
            name="leiden",
            fn=EVAL_CLUSTERING_METHODS.get("leiden"),
            params=step_spec.leiden_params,
            enabled=step_spec.leiden_enabled,
        ),
    ]

    cluster_keys = _resolve_enabled_cluster_keys(run_spec)

    for cluster_key in cluster_keys:
        steps.append(
            AnnDataEvaluationStep(
                name=f"internal_clustering_metrics_{cluster_key}",
                fn=compute_internal_clustering_metrics,
                params={
                    "cluster_key": cluster_key,
                    "selected": step_spec.internal_clustering_metrics,
                    "metric_params": step_spec.internal_clustering_metric_params,
                },
                enabled=step_spec.internal_clustering_metrics_enabled,
            )
        )

        steps.append(
            AnnDataEvaluationStep(
                name=f"external_clustering_metrics_{cluster_key}",
                fn=_compute_external_clustering_metrics_if_possible,
                params={
                    "label_key": step_spec.external_clustering_label_key,
                    "cluster_key": cluster_key,
                    "selected": step_spec.external_clustering_metrics,
                    "metric_params": step_spec.external_clustering_metric_params,
                    "external_metrics_enabled": (
                        step_spec.external_clustering_metrics_enabled
                    ),
                },
                enabled=step_spec.external_clustering_metrics_enabled is not False,
            )
        )

    return AnnDataEvaluationPipeline(steps=steps)


def create_reconstruction_evaluation_pipeline(
    run_spec: "EvaluationRunSpec",
) -> ReconstructionEvaluationPipeline:
    """Create the reconstruction evaluation pipeline from a resolved run spec.

    This function only wires reconstruction-side evaluation steps from the
    resolved run spec. Actual metric computation and error-map generation remain
    in ``reconstructions.reconstruction_metrics`` and
    ``reconstructions.error_maps``.
    """

    step_spec = run_spec.step_spec

    steps: list[ReconstructionEvaluationStep] = [
        ReconstructionEvaluationStep(
            name="reconstruction_metrics",
            fn=compute_reconstruction_metrics,
            params={
                "selected": step_spec.reconstruction_metrics,
                "metric_params": step_spec.reconstruction_metric_params,
                "reduction": step_spec.reconstruction_metrics_reduction,
            },
            enabled=step_spec.reconstruction_metrics_enabled,
        ),
        ReconstructionEvaluationStep(
            name="error_maps",
            fn=compute_error_maps,
            params=step_spec.error_map_params,
            enabled=step_spec.error_maps_enabled,
        ),
    ]

    return ReconstructionEvaluationPipeline(steps=steps)


def _resolve_enabled_cluster_keys(
    run_spec: "EvaluationRunSpec",
) -> list[str]:
    """Return cluster-label obs keys for enabled clustering steps."""

    step_spec = run_spec.step_spec
    cluster_keys: list[str] = []

    if step_spec.kmeans_enabled:
        cluster_keys.append(step_spec.kmeans_params.get("key_added", "kmeans"))

    if step_spec.leiden_enabled:
        cluster_keys.append(step_spec.leiden_params.get("key_added", "leiden"))

    return cluster_keys


def _compute_external_clustering_metrics_if_possible(
    adata: ad.AnnData,
    *,
    label_key: str,
    cluster_key: str,
    selected: Sequence[str] | None,
    metric_params: Mapping[str, Mapping[str, Any]] | None,
    external_metrics_enabled: bool | None,
) -> ad.AnnData:
    """Compute external clustering metrics when labels are available.

    This is a small runtime wrapper around ``compute_external_clustering_metrics``.
    The resolver cannot fully decide whether external metrics should run because
    ``enabled=None`` means auto-detect labels, and label availability is only known
    after the evaluation AnnData object has been loaded.

    ``external_metrics_enabled=None`` means auto mode: run only if ``label_key``
    exists in ``adata.obs``. ``external_metrics_enabled=True`` means the user
    explicitly requested external metrics, so missing labels should fail loudly.
    """

    if label_key not in adata.obs.columns:
        if external_metrics_enabled is True:
            raise KeyError(
                f"External clustering metrics were explicitly enabled, but "
                f"adata.obs does not contain label_key={label_key!r}."
            )

        return adata

    return compute_external_clustering_metrics(
        adata,
        label_key=label_key,
        cluster_key=cluster_key,
        selected=selected,
        metric_params=metric_params,
    )