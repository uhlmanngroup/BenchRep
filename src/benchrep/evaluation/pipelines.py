from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING, TypeVar
import warnings

import anndata as ad

from benchrep.assembly.registries.core import (
    EVAL_CLUSTERING_METHODS,
    EVAL_REDUCTIONS,
)
from benchrep.evaluation.utils import RecoverableEvaluationStepError
from benchrep.evaluation.embeddings.clustering_metrics import (
    compute_external_clustering_metrics,
    compute_internal_clustering_metrics,
)
from benchrep.evaluation.embeddings.embedding_metrics import (
    compute_embedding_metrics,
)
from benchrep.evaluation.embeddings.predictability import compute_predictability_metrics
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
EvaluationStepStatus = Literal[
    "pending",
    "running",
    "disabled",
    "completed",
    "completed_with_warnings",
    "skipped",
    "failed",
]

StepResultT = TypeVar("StepResultT")


@dataclass
class EvaluationStep:
    """Shared configuration and runtime state for an evaluation step."""

    name: str
    fn: Callable[..., Any]
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    dependencies: tuple["EvaluationStep", ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    status: EvaluationStepStatus = field(
        default="pending",
        init=False,
    )
    issues: list[str] = field(
        default_factory=list,
        init=False,
    )

    def mark_skipped(self, reason: str) -> None:
        """Mark the step as skipped without executing its callable."""

        self.issues.clear()
        self.status = "skipped"
        self.issues.append(f"Skipped: {reason}")

    def _run_with_status(
        self,
        *,
        operation: Callable[[], StepResultT],
        inactive_result: StepResultT,
    ) -> StepResultT:
        """Run an operation while maintaining step status and issues."""

        self.issues.clear()

        if not self.enabled:
            self.status = "disabled"
            return inactive_result

        if not self._dependencies_satisfied():
            return inactive_result

        self.status = "running"

        try:
            with warnings.catch_warnings(record=True) as captured_warnings:
                result = operation()

        except RecoverableEvaluationStepError as error:
            self.status = "failed"
            self._record_warnings(captured_warnings)
            self.issues.append(
                f"Error ({type(error).__name__}): {error}"
            )
            return inactive_result

        except Exception as error:
            self.status = "failed"
            self._record_warnings(captured_warnings)
            self.issues.append(
                f"Error ({type(error).__name__}): {error}"
            )
            raise

        self._record_warnings(captured_warnings)
        self.status = (
            "completed_with_warnings"
            if self.issues
            else "completed"
        )

        return result

    def _dependencies_satisfied(self) -> bool:
        """Check whether all dependency steps completed successfully."""

        unfinished_dependencies = [
            dependency
            for dependency in self.dependencies
            if dependency.status in {"pending", "running"}
        ]

        if unfinished_dependencies:
            dependency_states = self._format_dependency_states(
                unfinished_dependencies
            )
            message = (
                f"Evaluation step {self.name!r} was reached before its "
                f"dependency step(s) finished: {dependency_states}."
            )

            self.status = "failed"
            self.issues.append(f"Error (RuntimeError): {message}")
            raise RuntimeError(message)

        unsuccessful_dependencies = [
            dependency
            for dependency in self.dependencies
            if dependency.status in {"disabled", "skipped", "failed"}
        ]

        if unsuccessful_dependencies:
            dependency_states = self._format_dependency_states(
                unsuccessful_dependencies
            )
            self.mark_skipped(
                "required dependency step(s) did not complete successfully: "
                f"{dependency_states}"
            )
            return False

        return True

    def _record_warnings(
        self,
        captured_warnings: Sequence[warnings.WarningMessage],
    ) -> None:
        """Append captured Python warnings to the step issues."""

        self.issues.extend(
            f"Warning ({warning.category.__name__}): {warning.message}"
            for warning in captured_warnings
        )

    @staticmethod
    def _format_dependency_states(
        dependencies: Sequence["EvaluationStep"],
    ) -> str:
        return ", ".join(
            f"{dependency.name} ({dependency.status})"
            for dependency in dependencies
        )


@dataclass
class AnnDataEvaluationStep(EvaluationStep):
    """A single AnnData-based evaluation step."""

    def run(self, adata: ad.AnnData) -> ad.AnnData:
        """Run the step and enforce the AnnData output contract."""

        def operation() -> ad.AnnData:
            result = self.fn(
                adata,
                **dict(self.params),
            )

            if not isinstance(result, ad.AnnData):
                raise TypeError(
                    f"AnnData evaluation step {self.name!r} must return an "
                    f"AnnData object, got {type(result).__name__}."
                )

            return result

        return self._run_with_status(
            operation=operation,
            inactive_result=adata,
        )


@dataclass
class ReconstructionEvaluationStep(EvaluationStep):
    """A single reconstruction-based evaluation step."""

    def run(self, reconstruction_input: Any) -> dict[str, Any]:
        """Run the step and enforce the reconstruction output contract."""

        def operation() -> dict[str, Any]:
            result = self.fn(
                reconstruction_input,
                **dict(self.params),
            )

            if not isinstance(result, Mapping):
                raise TypeError(
                    f"Reconstruction evaluation step {self.name!r} must return "
                    f"a mapping, got {type(result).__name__}."
                )

            return dict(result)

        return self._run_with_status(
            operation=operation,
            inactive_result={},
        )


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

    kmeans_step = AnnDataEvaluationStep(
        name="kmeans",
        fn=EVAL_CLUSTERING_METHODS.get("kmeans"),
        params=step_spec.kmeans_params,
        enabled=step_spec.kmeans_enabled,
    )

    leiden_step = AnnDataEvaluationStep(
        name="leiden",
        fn=EVAL_CLUSTERING_METHODS.get("leiden"),
        params=step_spec.leiden_params,
        enabled=step_spec.leiden_enabled,
    )

    hdbscan_step = AnnDataEvaluationStep(
        name="hdbscan",
        fn=EVAL_CLUSTERING_METHODS.get("hdbscan"),
        params=step_spec.hdbscan_params,
        enabled=step_spec.hdbscan_enabled,
    )

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
        kmeans_step,
        leiden_step,
        hdbscan_step,
    ]

    clustering_steps = (
        (
            kmeans_step,
            step_spec.kmeans_params.get("key_added", "kmeans"),
        ),
        (
            leiden_step,
            step_spec.leiden_params.get("key_added", "leiden"),
        ),
        (
            hdbscan_step,
            step_spec.hdbscan_params.get("key_added", "hdbscan"),
        ),
    )

    for clustering_step, cluster_key in clustering_steps:
        if not clustering_step.enabled:
            continue

        steps.append(
            AnnDataEvaluationStep(
                name=f"internal_clustering_metrics_{cluster_key}",
                fn=compute_internal_clustering_metrics,
                dependencies=(clustering_step,),
                params={
                    "cluster_key": cluster_key,
                    "selected": step_spec.internal_clustering_metrics,
                    "metric_params": (
                        step_spec.internal_clustering_metric_params
                    ),
                },
                enabled=step_spec.internal_clustering_metrics_enabled,
            )
        )

        steps.append(
            AnnDataEvaluationStep(
                name=f"external_clustering_metrics_{cluster_key}",
                fn=_compute_external_clustering_metrics_if_possible,
                dependencies=(clustering_step,),
                params={
                    "label_key": step_spec.external_clustering_label_key,
                    "cluster_key": cluster_key,
                    "selected": step_spec.external_clustering_metrics,
                    "metric_params": (
                        step_spec.external_clustering_metric_params
                    ),
                    "external_metrics_enabled": (
                        step_spec.external_clustering_metrics_enabled
                    ),
                },
                enabled=(
                    step_spec.external_clustering_metrics_enabled
                    is not False
                ),
            )
        )

    steps.append(
        AnnDataEvaluationStep(
            name="embedding_metrics",
            fn=compute_embedding_metrics,
            params={
                "selected": step_spec.embedding_metrics,
                "metric_params": step_spec.embedding_metric_params,
            },
            enabled=step_spec.embedding_metrics_enabled,
        )
    )

    predictability_target_key = step_spec.predictability_target_key
    steps.append(
        AnnDataEvaluationStep(
            name=f"predictability_metrics_{predictability_target_key}",
            fn=compute_predictability_metrics,
            params={
                "target_key": predictability_target_key,
                "task": step_spec.predictability_task,
                "selected": step_spec.predictability_probes,
                "probe_params": step_spec.predictability_probe_params,
                "cv_params": step_spec.predictability_cv_params,
                "tuning_params": step_spec.predictability_tuning_params,
            },
            enabled=step_spec.predictability_enabled,
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
            params={
                **step_spec.error_map_params,
                "n_examples": (
                    run_spec.input_spec.reconstructions.n_examples
                    if run_spec.input_spec.reconstructions is not None
                    else None
                ),
            },
            enabled=step_spec.reconstruction_tiffs_enabled,
        ),
    ]

    return ReconstructionEvaluationPipeline(steps=steps)


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