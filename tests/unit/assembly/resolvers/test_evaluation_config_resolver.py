from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from benchrep.assembly.resolvers.evaluation_config_resolver import (
    _load_prediction_manifest,
    resolve_step_spec,
)
from benchrep.assembly.schemas import EvaluationConfig

@pytest.mark.parametrize(
    "status",
    [
        "completed",
        "completed_with_warnings",
    ],
)
def test_load_prediction_manifest_accepts_successful_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    manifest_path = tmp_path / "prediction_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "stage": "prediction",
                "status": status,
            }
        )
    )

    manifest = _load_prediction_manifest(manifest_path)

    assert manifest["status"] == status


def test_resolve_step_spec_uses_default_predictability_target() -> None:
    config = EvaluationConfig.model_validate(
        {
            "source": {
                "embeddings_path": "embeddings.h5ad",
            },
            "metrics": {
                "predictability": {
                    "enabled": True,
                },
            },
        }
    )

    step_spec = resolve_step_spec(
        evaluation_config=config,
        has_embeddings=True,
        has_reconstructions=False,
    )

    assert step_spec.predictability_enabled is True
    assert len(step_spec.predictability_targets) == 1

    target_spec = step_spec.predictability_targets[0]

    assert target_spec.target_key == "label"
    assert target_spec.task == "classification"
    assert target_spec.probes == [
        "dummy",
        "linear",
        "knn",
        "random_forest",
        "svm_rbf",
    ]
    assert target_spec.cv_params["method"] == "stratified_kfold"
    assert target_spec.cv_params["scoring"] == "balanced_accuracy"
    assert target_spec.probe_params["dummy"]["strategy"] == "most_frequent"
    assert (
        target_spec.probe_params["linear"]["model"]
        == "logistic_regression"
    )
    assert target_spec.overwrite is False


def test_resolve_step_spec_resolves_predictability_targets_independently() -> None:
    config = EvaluationConfig.model_validate(
        {
            "source": {
                "embeddings_path": "embeddings.h5ad",
            },
            "metrics": {
                "predictability": {
                    "enabled": True,
                    "targets": {
                        "label": {
                            "selected": ["svm_rbf"],
                            "task": "classification",
                            "cv": {
                                "scoring": "balanced_accuracy",
                            },
                            "tuning": {
                                "enabled": False,
                            },
                            "params": {
                                "svm_rbf": {
                                    "C": 2.0,
                                },
                            },
                        },
                        "intensity": {
                            "selected": [
                                "linear",
                                "random_forest",
                            ],
                            "overwrite": True,
                            "task": "regression",
                            "cv": {
                                "n_splits": 3,
                            },
                            "tuning": {
                                "enabled": True,
                                "inner_cv": {
                                    "n_splits": 2,
                                },
                            },
                            "params": {
                                "linear": {
                                    "model": "ridge",
                                    "alpha": [0.1, 1.0],
                                },
                            },
                        },
                    },
                },
            },
        }
    )

    step_spec = resolve_step_spec(
        evaluation_config=config,
        has_embeddings=True,
        has_reconstructions=False,
    )

    assert tuple(
        target.target_key
        for target in step_spec.predictability_targets
    ) == ("label", "intensity")

    classification_target, regression_target = (
        step_spec.predictability_targets
    )

    assert classification_target.task == "classification"
    assert classification_target.probes == ["svm_rbf"]
    assert (
        classification_target.cv_params["method"]
        == "stratified_kfold"
    )
    assert (
        classification_target.cv_params["scoring"]
        == "balanced_accuracy"
    )
    assert classification_target.probe_params["svm_rbf"]["C"] == 2.0
    assert classification_target.tuning_params["enabled"] is False
    assert classification_target.overwrite is False

    assert regression_target.task == "regression"
    assert regression_target.probes == ["linear", "random_forest"]
    assert regression_target.cv_params["method"] == "kfold"
    assert regression_target.cv_params["scoring"] == "r2"
    assert regression_target.cv_params["n_splits"] == 3
    assert regression_target.probe_params["linear"]["model"] == "ridge"
    assert regression_target.probe_params["linear"]["alpha"] == [0.1, 1.0]
    assert regression_target.tuning_params["enabled"] is True
    assert regression_target.tuning_params["inner_cv"]["n_splits"] == 2
    assert regression_target.overwrite is True


@pytest.mark.parametrize(
    ("targets", "error_match"),
    [
        (
            {},
            "predictability.targets cannot be empty",
        ),
        (
            {
                "label": {
                    "selected": [],
                },
            },
            "predictability target selections cannot be empty",
        ),
    ],
    ids=[
        "empty_targets",
        "empty_target_selection",
    ],
)
def test_enabled_predictability_rejects_empty_targets_or_selections(
    targets: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match=error_match,
    ):
        EvaluationConfig.model_validate(
            {
                "source": {
                    "embeddings_path": "embeddings.h5ad",
                },
                "metrics": {
                    "predictability": {
                        "enabled": True,
                        "targets": targets,
                    },
                },
            }
        )


def test_predictability_resolution_error_identifies_target() -> None:
    config = EvaluationConfig.model_validate(
        {
            "source": {
                "embeddings_path": "embeddings.h5ad",
            },
            "metrics": {
                "predictability": {
                    "enabled": True,
                    "targets": {
                        "intensity": {
                            "selected": ["linear"],
                            "task": "regression",
                            "params": {
                                "linear": {
                                    "model": "logistic_regression",
                                },
                            },
                        },
                    },
                },
            },
        }
    )

    with pytest.raises(ValueError) as exc_info:
        resolve_step_spec(
            evaluation_config=config,
            has_embeddings=True,
            has_reconstructions=False,
        )

    message = str(exc_info.value)

    assert (
        "metrics.predictability.targets['intensity']"
        in message
    )
    assert "model must be 'ridge'" in message


def test_resolve_step_spec_rejects_no_enabled_work() -> None:
    config = EvaluationConfig.model_validate(
        {
            "source": {
                "embeddings_path": "embeddings.h5ad",
            },
            "reductions": {
                "pca": {"enabled": False},
                "umap": {"enabled": False},
                "tsne": {"enabled": False},
            },
            "clustering": {
                "kmeans": {"enabled": False},
                "leiden": {"enabled": False},
                "hdbscan": {"enabled": False},
            },
            "metrics": {
                "clustering": {
                    "internal": {"enabled": False},
                    "external": {"enabled": False},
                },
                "embedding": {"enabled": False},
                "predictability": {"enabled": False},
                "reconstruction": {"enabled": False},
            },
            "reconstruction": {
                "export_tiffs": False,
            },
            "plots": {
                "enabled": False,
            },
        }
    )

    with pytest.raises(
        ValueError,
        match="resolves to no enabled evaluation",
    ):
        resolve_step_spec(
            evaluation_config=config,
            has_embeddings=True,
            has_reconstructions=False,
        )