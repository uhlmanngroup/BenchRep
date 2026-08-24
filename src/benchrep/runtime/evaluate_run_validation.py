from __future__ import annotations

from collections.abc import Mapping, Collection
from dataclasses import dataclass, replace
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
from scipy import sparse

from benchrep.assembly.resolvers.evaluation_config_resolver import EvaluationStepSpec
from benchrep.evaluation.reconstructions.data import (
    ReconstructionEvaluationInput,
    load_reconstruction_evaluation_input,
)
from benchrep.evaluation.utils import validate_adata_x, load_scanpy_backend


if TYPE_CHECKING:
    from benchrep.assembly.resolvers.evaluation_config_resolver import (
        EvaluationRunSpec,
    )


@dataclass(frozen=True, slots=True)
class EvaluateSourceInputsResult:
    """Validated evaluation source inputs needed by the evaluation runner."""

    adata_input: ad.AnnData | None
    reconstruction_input: ReconstructionEvaluationInput | None


def prepare_evaluate_source_inputs(
    run_spec: EvaluationRunSpec,
) -> EvaluateSourceInputsResult:
    """Load and validate source inputs required by the evaluation runner."""

    adata = None
    embeddings_path = run_spec.input_spec.embeddings_path

    if embeddings_path is not None:
        adata = _load_embeddings_adata(embeddings_path)
        _validate_embedding_adata_basic_contract(adata)

    _validate_enabled_step_preconditions(
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

    if adata is None and reconstruction_input is None:
        raise RuntimeError(
            "Evaluation source resolution produced no loadable inputs."
        )

    return EvaluateSourceInputsResult(
        adata_input=adata,
        reconstruction_input=reconstruction_input,
    )


def finalize_evaluation_step_spec(
    step_spec: EvaluationStepSpec,
    *,
    available_obs_columns: Collection[str],
) -> EvaluationStepSpec:
    """Finalize automatic step settings that depend on loaded evaluation inputs."""

    if step_spec.external_clustering_metrics_enabled is not None:
        return step_spec

    return replace(
        step_spec,
        external_clustering_metrics_enabled=(
            step_spec.external_clustering_label_key
            in available_obs_columns
        ),
    )


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

    if (
        not np.issubdtype(adata.X.dtype, np.number)
        or np.issubdtype(adata.X.dtype, np.complexfloating)
    ):
        raise TypeError(
            "adata.X must contain real numeric values for evaluation, "
            f"got dtype {adata.X.dtype}."
        )

    _validate_finite_embedding_values(adata)


def _validate_finite_embedding_values(adata: ad.AnnData) -> None:
    """Require every value in the evaluation embedding matrix to be finite."""

    if sparse.issparse(adata.X):
        matrix = adata.X.tocoo()
        values = matrix.data

        nan_mask = np.isnan(values)
        infinite_mask = np.isinf(values)
        nonfinite_mask = nan_mask | infinite_mask

        if not nonfinite_mask.any():
            return

        affected_rows = np.unique(matrix.row[nonfinite_mask])

    else:
        values = np.asarray(adata.X)

        nan_mask = np.isnan(values)
        infinite_mask = np.isinf(values)
        nonfinite_mask = nan_mask | infinite_mask

        if not nonfinite_mask.any():
            return

        affected_rows = np.flatnonzero(
            nonfinite_mask.any(axis=1)
        )

    preview_rows = affected_rows[:10].tolist()

    raise ValueError(
        "Evaluation embeddings must contain only finite values. "
        f"Found {int(nan_mask.sum())} NaN values and "
        f"{int(infinite_mask.sum())} infinite values across "
        f"{len(affected_rows)} observations. First affected observation "
        f"positions: {preview_rows}."
    )


def _validate_enabled_step_preconditions(
    *,
    run_spec: EvaluationRunSpec,
) -> None:
    """Fail early on obvious enabled-step issues before mutating AnnData."""

    step_spec = run_spec.step_spec

    scanpy_steps = []

    if step_spec.umap_enabled:
        scanpy_steps.append("UMAP")

    if step_spec.tsne_enabled:
        scanpy_steps.append("t-SNE")

    if step_spec.leiden_enabled:
        scanpy_steps.append("Leiden clustering")

    if scanpy_steps:
        load_scanpy_backend(
            feature=", ".join(scanpy_steps),
            require_leiden=step_spec.leiden_enabled,
        )

    if step_spec.kmeans_enabled:
        n_clusters = step_spec.kmeans_params.get("n_clusters")

        if n_clusters is None:
            raise ValueError(
                "KMeans n_clusters is required when KMeans is enabled."
            )

    if step_spec.predictability_enabled:
        for target_spec in step_spec.predictability_targets:
            _validate_predictability_configuration_preconditions(
                target_key=target_spec.target_key,
                task=target_spec.task,
                probes=target_spec.probes,
                cv_params=target_spec.cv_params,
            )


def _validate_predictability_configuration_preconditions(
    *,
    target_key: str,
    task: str,
    probes: list[str],
    cv_params: Mapping[str, Any],
) -> None:
    """Validate data-independent requirements for one predictability target."""

    target_path = f"metrics.predictability.targets[{target_key!r}]"
    method = cv_params.get("method", "stratified_kfold")

    if (
        method in {"stratified_kfold", "stratified_group_kfold"}
        and task != "classification"
    ):
        raise ValueError(
            f"{target_path}.cv.method={method!r} is only valid for "
            "classification."
        )

    if "xgboost" in probes and find_spec("xgboost") is None:
        raise ImportError(
            "The xgboost predictability probe was selected for "
            f"{target_path}, but xgboost is not installed. Install xgboost or "
            f"remove 'xgboost' from {target_path}.selected."
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
