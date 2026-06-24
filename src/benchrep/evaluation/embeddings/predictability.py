from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dataclasses import dataclass

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from benchrep.evaluation.utils import PredictabilityTask


@dataclass(frozen=True)
class PredictabilityCVSpec:
    outer_cv: Any
    inner_cv: Any | None
    use_groups: bool


def build_predictability_cv(
    *,
    task: PredictabilityTask,
    cv_params: Mapping[str, Any],
    tuning_enabled: bool,
    tuning_params: Mapping[str, Any],
) -> PredictabilityCVSpec:
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