from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from benchrep.assembly.config import compose_effective_config
from benchrep.assembly.schemas import (
    EvaluationConfig,
    PredictionConfig,
    TrainingConfig,
)
from tests.fixtures.configs.configs import (
    make_evaluation_clustering_config,
    make_evaluation_config,
    make_evaluation_metrics_config,
    make_evaluation_plots_config,
    make_evaluation_reconstruction_config,
    make_evaluation_reductions_config,
    make_evaluation_run_config,
    make_evaluation_source_config,
    make_prediction_config,
    make_prediction_data_config,
    make_prediction_dataset_config,
    make_prediction_exports_config,
    make_prediction_inference_config,
    make_prediction_source_config,
    make_training_checkpoint_config,
    make_training_config,
    make_training_custom_dataset_config,
    make_training_datamodule_config,
    make_training_decoder_config,
    make_training_encoder_config,
    make_training_inspection_config,
    make_training_logger_config,
    make_training_losses_config,
    make_training_mnist_dataset_config,
    make_training_model_config,
    make_training_optimizer_config,
    make_training_reproducibility_config,
    make_training_run_config,
    make_training_trainer_config,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "configs"


@pytest.mark.parametrize(
    (
        "schema",
        "config_name",
        "section_name",
        "component_factory",
        "composition_kwargs",
    ),
    [
        # Training
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "run",
            make_training_run_config,
            {},
            id="training-run",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "reproducibility",
            make_training_reproducibility_config,
            {},
            id="training-reproducibility",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "model",
            make_training_model_config,
            {},
            id="training-model",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "encoder",
            make_training_encoder_config,
            {},
            id="training-encoder",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "decoder",
            make_training_decoder_config,
            {},
            id="training-decoder",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "losses",
            make_training_losses_config,
            {},
            id="training-losses",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "optimizer",
            make_training_optimizer_config,
            {},
            id="training-optimizer",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "dataset",
            make_training_custom_dataset_config,
            {},
            id="training-custom-dataset",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "dataset",
            make_training_mnist_dataset_config,
            {},
            id="training-mnist-dataset",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "datamodule",
            make_training_datamodule_config,
            {},
            id="training-datamodule",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "trainer",
            make_training_trainer_config,
            {},
            id="training-trainer",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "logger",
            make_training_logger_config,
            {},
            id="training-logger",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "checkpointing",
            make_training_checkpoint_config,
            {},
            id="training-checkpointing",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "inspection",
            make_training_inspection_config,
            {},
            id="training-inspection",
        ),

        # Prediction
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "source",
            make_prediction_source_config,
            {"training_manifest_path_overridden": True},
            id="prediction-source",
        ),
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "dataset",
            make_prediction_dataset_config,
            {"training_manifest_path_overridden": True},
            id="prediction-dataset",
        ),
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "data",
            make_prediction_data_config,
            {"training_manifest_path_overridden": True},
            id="prediction-data",
        ),
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "inference",
            make_prediction_inference_config,
            {"training_manifest_path_overridden": True},
            id="prediction-inference",
        ),
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "exports",
            make_prediction_exports_config,
            {"training_manifest_path_overridden": True},
            id="prediction-exports",
        ),

        # Evaluation
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "source",
            make_evaluation_source_config,
            {"prediction_manifest_path_overridden": True},
            id="evaluation-source",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "run",
            make_evaluation_run_config,
            {"prediction_manifest_path_overridden": True},
            id="evaluation-run",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "reductions",
            make_evaluation_reductions_config,
            {"prediction_manifest_path_overridden": True},
            id="evaluation-reductions",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "clustering",
            make_evaluation_clustering_config,
            {"prediction_manifest_path_overridden": True},
            id="evaluation-clustering",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "metrics",
            make_evaluation_metrics_config,
            {"prediction_manifest_path_overridden": True},
            id="evaluation-metrics",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "reconstruction",
            make_evaluation_reconstruction_config,
            {"prediction_manifest_path_overridden": True},
            id="evaluation-reconstruction",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "plots",
            make_evaluation_plots_config,
            {"prediction_manifest_path_overridden": True},
            id="evaluation-plots",
        ),
    ],
)
def test_yaml_component_replaces_top_level_section(
    schema: Any,
    config_name: str,
    section_name: str,
    component_factory: Callable[[], Any],
    composition_kwargs: dict[str, Any],
) -> None:
    """A supplied component replaces only its matching YAML section."""
    config_path = CONFIG_DIR / config_name
    replacement_component = component_factory()

    baseline_result = compose_effective_config(
        schema=schema,
        config_path=config_path,
        **composition_kwargs,
    )

    composition_result = compose_effective_config(
        schema=schema,
        config_path=config_path,
        config_components={
            section_name: replacement_component,
        },
        **composition_kwargs,
    )

    assert composition_result.effective_source == "yaml_with_components"
    assert composition_result.yaml_supplied is True
    assert composition_result.yaml_used_as_base is True
    assert composition_result.original_config_path == config_path.resolve()
    assert composition_result.original_config_raw is not None

    assert (
        getattr(composition_result.effective_config, section_name)
        == replacement_component
    )

    assert(
        getattr(baseline_result.effective_config, section_name)
        != replacement_component
    )

    for field_name in schema.model_fields:
        if field_name == section_name:
            continue

        assert (
            getattr(composition_result.effective_config, field_name)
            == getattr(baseline_result.effective_config, field_name)
        )


@pytest.mark.parametrize(
    (
        "schema",
        "config_name",
        "full_config_factory",
        "ignored_component_name",
        "ignored_component_factory",
    ),
    [
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            make_training_config,
            "run",
            lambda: make_training_run_config().model_copy(
                update={"project_name": "ignored_component"}
            ),
            id="training",
        ),
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            make_prediction_config,
            "data",
            lambda: make_prediction_data_config().model_copy(
                update={"batch_size": 7}
            ),
            id="prediction",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            make_evaluation_config,
            "run",
            lambda: make_evaluation_run_config().model_copy(
                update={"run_name": "ignored_component"}
            ),
            id="evaluation",
        ),
    ],
)
def test_full_config_object_takes_precedence(
    schema: Any,
    config_name: str,
    full_config_factory: Callable[[], Any],
    ignored_component_name: str,
    ignored_component_factory: Callable[[], Any],
) -> None:
    """A full config object takes precedence over YAML and components."""
    config_path = CONFIG_DIR / config_name
    full_config = full_config_factory()
    ignored_component = ignored_component_factory()

    assert (
            getattr(full_config, ignored_component_name)
            != ignored_component
    )

    composition_result = compose_effective_config(
        schema=schema,
        config_path=config_path,
        full_config_object=full_config,
        config_components={
            ignored_component_name: ignored_component,
        },
    )

    assert composition_result.effective_config is full_config
    assert composition_result.effective_source == "config_object"
    assert composition_result.yaml_supplied is True
    assert composition_result.yaml_used_as_base is False
    assert composition_result.original_config_path == config_path.resolve()
    assert composition_result.original_config_raw is not None

    assert len(composition_result.composition_warnings) == 2
    assert any(
        "config YAML will be ignored"
        in warning
        for warning in composition_result.composition_warnings
    )
    assert any(
        "config_components will be ignored"
        in warning
        for warning in composition_result.composition_warnings
    )