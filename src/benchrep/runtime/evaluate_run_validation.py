from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
from pandas.api.types import is_numeric_dtype

from benchrep.evaluation.reconstructions.data import (
    ReconstructionEvaluationInput,
    load_reconstruction_evaluation_input,
)
from benchrep.evaluation.utils import validate_adata_x

if TYPE_CHECKING:
    from benchrep.assembly.resolvers.evaluation_config_resolver import (
        EvaluationRunSpec,
    )


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
        if not is_numeric_dtype(target):
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
        return

    n_examples = reconstruction_input.n_examples
    if n_examples is None:
        n_examples = int(reconstruction_input.inputs.shape[0])

    if obs_length < n_examples:
        raise ValueError(
            "Reconstruction obs must contain at least as many rows/items as the "
            "selected reconstruction examples. "
            f"Got obs length={obs_length}, selected examples={n_examples}."
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
