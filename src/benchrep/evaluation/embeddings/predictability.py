from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dataclasses import dataclass, asdict

import numpy as np

import anndata as ad

from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from benchrep.evaluation.utils import (
    PredictabilityTask,
    validate_adata_x,
    validate_obs_key,
)
from benchrep.evaluation.embeddings.predictability_probes import PredictabilityProbeSpec
from benchrep.assembly.registries.core import EVAL_PREDICTABILITY_PROBES


@dataclass(frozen=True)
class PredictabilityCVSpec:
    outer_cv: Any
    inner_cv: Any | None
    use_groups: bool


@dataclass(frozen=True)
class PredictabilityInputSpec:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray | None


@dataclass(frozen=True)
class PredictabilityProbeResult:
    fold_scores: list[float]
    mean_score: float
    std_score: float
    best_params_by_fold: list[dict[str, Any]] | None
    tuned: bool


def build_predictability_cv(
    *,
    task: PredictabilityTask,
    cv_params: Mapping[str, Any],
    tuning_enabled: bool,
    tuning_params: Mapping[str, Any],
) -> PredictabilityCVSpec:
    """Build cross-validation splitters for predictability evaluation.

    The outer CV splitter is always created and is used to estimate final probe
    performance. If hyperparameter tuning is enabled, an inner CV splitter is
    also created and should be passed to ``GridSearchCV`` within each outer
    training fold.

    The inner splitter follows the same CV family as the outer splitter:
    non-grouped methods use KFold/StratifiedKFold, while grouped methods use
    GroupKFold/StratifiedGroupKFold. When ``use_groups`` is True, downstream
    fitting code must pass the corresponding training-fold groups to both outer
    splitting and inner tuning.

    Splits share method, shuffle, and random_state.

    Parameters
    ----------
    task
        Predictability task, either ``"classification"`` or ``"regression"``.
    cv_params
        Resolved outer CV parameters, including method, split count, optional
        group key, shuffle behavior, random state, and scoring.
    tuning_enabled
        Whether to construct an inner CV splitter for hyperparameter tuning.
    tuning_params
        Resolved tuning parameters, including inner CV split count when tuning
        is enabled.

    Returns
    -------
    PredictabilityCVSpec
        Resolved outer CV splitter, optional inner CV splitter, and whether the
        selected CV strategy requires group labels.
    """
    if task not in ("classification", "regression"):
        raise ValueError(
            "task must be either 'classification' or 'regression', "
            f"got {task!r}."
        )

    method = cv_params.get("method", "stratified_kfold")
    outer_n_splits = int(cv_params.get("n_splits", 5))
    shuffle = bool(cv_params.get("shuffle", True))
    random_state = cv_params.get("random_state", None)

    if tuning_enabled:
        inner_cv_params = tuning_params.get("inner_cv", {})
        inner_n_splits = int(inner_cv_params.get("n_splits", 3))
    else:
        inner_n_splits = None


    if method == "kfold":
        return PredictabilityCVSpec(
            outer_cv = KFold(
                n_splits=outer_n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            ),
            inner_cv = KFold(
                n_splits=inner_n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            ) if tuning_enabled else None,
            use_groups = False,
        )

    if method == "stratified_kfold":
        if task != "classification":
            raise ValueError(
                "cv.method='stratified_kfold' is only valid for classification."
            )
        return PredictabilityCVSpec(
            outer_cv=StratifiedKFold(
                n_splits=outer_n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            ),
            inner_cv=StratifiedKFold(
                n_splits=inner_n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            ) if tuning_enabled else None,
            use_groups=False,
        )

    if method == "group_kfold":
        return PredictabilityCVSpec(
            outer_cv=GroupKFold(n_splits=outer_n_splits),
            inner_cv=GroupKFold(n_splits=inner_n_splits) if tuning_enabled else None,
            use_groups=True,
        )

    if method == "stratified_group_kfold":
        if task != "classification":
            raise ValueError(
                "cv.method='stratified_group_kfold' is only valid for classification."
            )
        return PredictabilityCVSpec(
            outer_cv=StratifiedGroupKFold(
                n_splits=outer_n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            ),
            inner_cv=StratifiedGroupKFold(
                n_splits=inner_n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )  if tuning_enabled else None,
            use_groups=True,
        )

    raise ValueError(
        "cv.method must be one of 'kfold', 'stratified_kfold', "
        "'group_kfold', or 'stratified_group_kfold'. "
        f"Got {method!r}."
    )


def resolve_predictability_inputs(
    adata: ad.AnnData,
    *,
    target_key: str,
    group_key: str | None,
    use_groups: bool,
) -> PredictabilityInputSpec:
    """Resolve AnnData arrays used by predictability evaluation."""
    validate_adata_x(adata)
    validate_obs_key(adata, target_key)

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.asarray(X)

    if X.shape[0] != adata.n_obs:
        raise ValueError(
            "Predictability embedding matrix row count does not match adata.n_obs. "
            f"Got X.shape[0]={X.shape[0]} and adata.n_obs={adata.n_obs}."
        )

    y = adata.obs[target_key].to_numpy()

    if use_groups:
        if group_key is None:
            raise ValueError(
                "Predictability CV uses groups, but cv.group_key is None."
            )

        validate_obs_key(adata, group_key)
        groups = adata.obs[group_key].to_numpy()
    else:
        groups = None

    return PredictabilityInputSpec(
        X=X,
        y=y,
        groups=groups,
    )


def evaluate_predictability_probe(
    *,
    input_spec: PredictabilityInputSpec,
    cv_spec: PredictabilityCVSpec,
    probe_spec: PredictabilityProbeSpec,
    scoring: str,
    tuning_enabled: bool,
) -> PredictabilityProbeResult:
    """Evaluate one predictability probe across outer CV folds."""
    X = input_spec.X
    y = input_spec.y
    groups = input_spec.groups

    scorer = get_scorer(scoring)

    if cv_spec.use_groups:
        if groups is None:
            raise ValueError(
                "Grouped predictability CV requires group labels, but groups=None."
            )
        outer_splits = cv_spec.outer_cv.split(X, y, groups)
    else:
        outer_splits = cv_spec.outer_cv.split(X, y)

    fold_scores: list[float] = []
    best_params_by_fold: list[dict[str, Any]] = []

    for train_idx, test_idx in outer_splits:
        estimator = clone(probe_spec.estimator)

        if tuning_enabled:
            if cv_spec.inner_cv is None:
                raise ValueError(
                    "Predictability tuning is enabled, but inner_cv is None."
                )

            search = GridSearchCV(
                estimator=estimator,
                param_grid=probe_spec.param_grid,
                scoring=scoring,
                cv=cv_spec.inner_cv,
            )

            if cv_spec.use_groups:
                search.fit(
                    X[train_idx],
                    y[train_idx],
                    groups=groups[train_idx],
                )
            else:
                search.fit(
                    X[train_idx],
                    y[train_idx],
                )

            fitted_estimator = search.best_estimator_
            best_params_by_fold.append(dict(search.best_params_))

        else:
            fitted_estimator = estimator.fit(
                X[train_idx],
                y[train_idx],
            )

        fold_score = scorer(
            fitted_estimator,
            X[test_idx],
            y[test_idx],
        )
        fold_scores.append(float(fold_score))

    return PredictabilityProbeResult(
        fold_scores=fold_scores,
        mean_score=float(np.mean(fold_scores)),
        std_score=float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        best_params_by_fold=best_params_by_fold if tuning_enabled else None,
        tuned=tuning_enabled,
    )


def compute_predictability_metrics(
        adata: ad.AnnData,
        *,
        target_key: str,
        task: PredictabilityTask,
        selected: Sequence[str],
        probe_params: Mapping[str, Mapping[str, Any]],
        cv_params: Mapping[str, Any],
        tuning_params: Mapping[str, Any],
) -> ad.AnnData:
    """Compute supervised predictability metrics from embeddings in ``adata.X``."""
    if len(selected) == 0:
        raise ValueError("selected must contain at least one predictability probe.")

    cv_params = dict(cv_params)
    tuning_params = dict(tuning_params)
    probe_params = {
        probe_name: dict(params)
        for probe_name, params in probe_params.items()
    }

    scoring = cv_params.get("scoring")
    if scoring is None:
        raise ValueError("cv_params must contain a resolved predictability scoring.")

    tuning_enabled = bool(tuning_params.get("enabled", False))

    cv_spec = build_predictability_cv(
        task=task,
        cv_params=cv_params,
        tuning_enabled=tuning_enabled,
        tuning_params=tuning_params,
    )

    input_spec = resolve_predictability_inputs(
        adata,
        target_key=target_key,
        group_key=cv_params.get("group_key"),
        use_groups=cv_spec.use_groups,
    )

    probe_results: dict[str, PredictabilityProbeResult] = {}
    for probe_name in selected:
        probe_builder = EVAL_PREDICTABILITY_PROBES.get(probe_name)
        probe_spec = probe_builder(
            task=task,
            params=probe_params[probe_name],
        )
        probe_results[probe_name] = evaluate_predictability_probe(
            input_spec=input_spec,
            cv_spec=cv_spec,
            probe_spec=probe_spec,
            scoring=scoring,
            tuning_enabled=tuning_enabled,
        )

    result = {
        "target_key": target_key,
        "task": task,
        "scoring": scoring,
        "cv": dict(cv_params),
        "tuning": dict(tuning_params),
        "probes": {
            probe_name: asdict(probe_result)
            for probe_name, probe_result in probe_results.items()
        },
    }

    _store_predictability_metric_result(
        adata,
        target_key=target_key,
        result=result,
    )

    return adata


def _store_predictability_metric_result(
    adata: ad.AnnData,
    *,
    target_key: str,
    result: Mapping[str, Any],
) -> None:
    """Store predictability metric results under the BenchRep namespace.

    Results are written to:

        adata.uns["benchrep"]["metrics"]["predictability"][target_key]

    The resulting structure is:

        adata.uns["benchrep"] = {
            "metrics": {
                "predictability": {
                    target_key: {
                        "target_key": target_key,
                        "task": ...,
                        "scoring": ...,
                        "cv": ...,
                        "tuning": ...,
                        "probes": {
                            "dummy": {...},
                            "linear": {...},
                            "knn": {...},
                        },
                    },
                },
            },
        }
    """

    benchrep_uns = adata.uns.setdefault("benchrep", {})
    metrics_uns = benchrep_uns.setdefault("metrics", {})
    predictability_uns = metrics_uns.setdefault("predictability", {})

    predictability_uns[target_key] = dict(result)