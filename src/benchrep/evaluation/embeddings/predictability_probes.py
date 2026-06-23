from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from benchrep.evaluation.utils import PredictabilityTask


@dataclass(frozen=True)
class PredictabilityProbeSpec:
    """Resolved sklearn estimator and optional tuning grid for one probe."""

    estimator: Any
    param_grid: dict[str, list[Any]]


def build_dummy_predictability_probe(
    *,
    task: PredictabilityTask,
    params: Mapping[str, Any],
) -> PredictabilityProbeSpec:
    """Build a dummy baseline predictability probe.

    For classification, this returns ``DummyClassifier``.
    For regression, this returns ``DummyRegressor``.

    Dummy probes do not expose tunable parameters in the current BenchRep
    predictability workflow, so ``param_grid`` is always empty.
    """
    params = dict(params)
    strategy = params.pop("strategy")

    # Dummy probe takes just one parameter
    if params:
        raise ValueError(
            "Unsupported dummy predictability parameters: "
            f"{sorted(params)}."
        )

    if task == "classification":
        estimator = DummyClassifier(strategy=strategy)

    elif task == "regression":
        estimator = DummyRegressor(strategy=strategy)

    else:
        raise ValueError(
            "task must be either 'classification' or 'regression', "
            f"got {task!r}."
        )

    return PredictabilityProbeSpec(
        estimator=estimator,
        param_grid={},
    )


def build_linear_predictability_probe(
    *,
    task: PredictabilityTask,
    params: Mapping[str, Any],
) -> PredictabilityProbeSpec:
    """Build a linear predictability probe.

    Classification uses ``LogisticRegression``.
    Regression uses ``Ridge``.

    If ``standardize`` is true, the estimator is wrapped in a sklearn
    ``Pipeline`` with ``StandardScaler``.
    """
    params = dict(params)

    model_name = params.pop("model")
    standardize = params.pop("standardize", True)

    fixed_params, param_grid = split_fixed_and_grid_params(params)

    if task == "classification":
        if model_name != "logistic_regression":
            raise ValueError(
                "Linear classification predictability requires "
                "model='logistic_regression'."
            )
        estimator = LogisticRegression(**fixed_params)

    elif task == "regression":
        if model_name != "ridge":
            raise ValueError(
                "Linear regression predictability requires model='ridge'."
            )
        estimator = Ridge(**fixed_params)

    else:
        raise ValueError(
            "task must be either 'classification' or 'regression', "
            f"got {task!r}."
        )

    if standardize:
        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("estimator", estimator),
            ]
        )
        param_grid = {
            f"estimator__{key}": value
            for key, value in param_grid.items()
        }

    return PredictabilityProbeSpec(
        estimator=estimator,
        param_grid=param_grid,
    )


def build_knn_predictability_probe(
        *,
        task: PredictabilityTask,
        params: Mapping[str, Any],
) -> PredictabilityProbeSpec:
    """Build a k-nearest-neighbors predictability probe.

    Classification uses ``KNeighborsClassifier``.
    Regression uses ``KNeighborsRegressor``.

    If ``standardize`` is true, the estimator is wrapped in a sklearn
    ``Pipeline`` with ``StandardScaler``.
    """
    params = dict(params)

    standardize = params.pop("standardize", True)

    fixed_params, param_grid = split_fixed_and_grid_params(params)

    if task == "classification":
        estimator = KNeighborsClassifier(**fixed_params)

    elif task == "regression":
        estimator = KNeighborsRegressor(**fixed_params)

    else:
        raise ValueError(
            "task must be either 'classification' or 'regression', "
            f"got {task!r}."
        )

    if standardize:
        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("estimator", estimator),
            ]
        )
        param_grid = {
            f"estimator__{key}": value
            for key, value in param_grid.items()
        }

    return PredictabilityProbeSpec(
        estimator=estimator,
        param_grid=param_grid,
    )


def build_random_forest_predictability_probe(
        *,
        task: PredictabilityTask,
        params: Mapping[str, Any],
) -> PredictabilityProbeSpec:
    """Build a random forest predictability probe.

    Classification uses ``RandomForestClassifier``.
    Regression uses ``RandomForestRegressor``.
    """
    params = dict(params)

    fixed_params, param_grid = split_fixed_and_grid_params(params)

    if task == "classification":
        estimator = RandomForestClassifier(**fixed_params)

    elif task == "regression":
        fixed_params.pop("class_weight", None)
        param_grid.pop("class_weight", None)
        estimator = RandomForestRegressor(**fixed_params)

    else:
        raise ValueError(
            "task must be either 'classification' or 'regression', "
            f"got {task!r}."
        )

    return PredictabilityProbeSpec(
        estimator=estimator,
        param_grid=param_grid,
    )


def build_xgboost_predictability_probe(
    *,
    task: PredictabilityTask,
    params: Mapping[str, Any],
) -> PredictabilityProbeSpec:
    """Build an optional XGBoost predictability probe.

    Classification uses ``XGBClassifier``.
    Regression uses ``XGBRegressor``.

    XGBoost is imported lazily so it does not become a core BenchRep dependency.
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as error:
        raise ImportError(
            "The xgboost predictability probe was selected, but xgboost is not "
            "installed. Install the optional xgboost dependency or remove "
            "'xgboost' from metrics.predictability.selected."
        ) from error

    params = dict(params)
    fixed_params, param_grid = split_fixed_and_grid_params(params)

    if task == "classification":
        estimator = XGBClassifier(**fixed_params)

    elif task == "regression":
        estimator = XGBRegressor(**fixed_params)

    else:
        raise ValueError(
            "task must be either 'classification' or 'regression', "
            f"got {task!r}."
        )

    return PredictabilityProbeSpec(
        estimator=estimator,
        param_grid=param_grid,
    )


def split_fixed_and_grid_params(
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Split scalar estimator params from list-valued tuning grid params.

    Scalar values are passed directly to the estimator constructor.
    List-valued params are treated as hyperparameter grid candidates.
    """
    fixed_params: dict[str, Any] = {}
    param_grid: dict[str, list[Any]] = {}

    for key, value in params.items():
        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError(
                    f"Tunable predictability parameter {key!r} cannot be an empty list."
                )
            param_grid[key] = value
        else:
            fixed_params[key] = value

    return fixed_params, param_grid