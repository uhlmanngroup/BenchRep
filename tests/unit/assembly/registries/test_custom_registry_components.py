from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from benchrep import evaluate, predict_vae, train_vae
from benchrep.assembly.schemas import (
    CustomDatasetConfig,
    TrainingDecoderConfig,
    EvaluationEmbeddingMetricConfig,
    TrainingEncoderConfig,
    EvaluationClusteringMetricsConfig,
    EvaluationMetricsConfig,
    EvaluationPlotsConfig,
    EvaluationPredictabilityConfig,
    EvaluationExternalClusteringMetricConfig,
    EvaluationInternalClusteringMetricConfig,
    TrainingLoggerConfig,
    TrainingAdditionalCallbackConfig,
    TrainingLossTermConfig,
    TrainingOptimizerConfig,
    EvaluationReconstructionMetricConfig,
    TrainingTransformConfig,
)
from benchrep.discovery.registry import _COMPONENT_REGISTRIES
from tests.fixtures.registered_components import (
    COMPONENT_CALLS,
    register_custom_test_components,
    reset_component_calls,
    CustomRegisteredCallback,
)


CONFIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "configs"
)


@pytest.fixture
def isolated_custom_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    registries = tuple(
        registry_info.registry
        for registry_info in _COMPONENT_REGISTRIES.values()
        if registry_info.registry.custom_registration_supported
    )

    for registry in registries:
        # Ensure the original dictionaries already contain all built-ins.
        registry.keys()

        monkeypatch.setattr(
            registry,
            "_items",
            dict(getattr(registry, "_items")),
        )
        monkeypatch.setattr(
            registry,
            "_canonical_keys",
            dict(getattr(registry, "_canonical_keys")),
        )

    yield


def test_custom_registered_components_work_end_to_end(
    isolated_custom_registries: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_custom_test_components()
    reset_component_calls()
    monkeypatch.chdir(tmp_path)

    dataset_config = CustomDatasetConfig(
        name="custom_test_dataset",
        params={
            "n_samples": 32,
            "image_shape": [1, 28, 28],
            "n_classes": 4,
            "n_groups": 2,
            "signal_strength": 0.8,
            "noise_std": 0.05,
            "seed": 137,
        },
    )

    print("\n=== Training with custom registered components ===")
    training_result = train_vae(
        config_path=(
            CONFIG_DIR
            / "training_tiny_synthetic_vae.yaml"
        ),
        config_components={
            "dataset": dataset_config,
            "transforms": [
                TrainingTransformConfig(
                    name="custom_test_transform",
                    apply_to=["validation"],
                    params={"scale": 1.0},
                ),
            ],
            "encoder": TrainingEncoderConfig(
                name="custom_test_encoder",
                params={
                    "input_shape": [1, 28, 28],
                    "output_dim": 8,
                },
            ),
            "decoder": TrainingDecoderConfig(
                name="custom_test_decoder",
                params={
                    "output_shape": [1, 28, 28],
                },
            ),
            "losses": {
                "reconstruction": {
                    "custom_test_reconstruction_loss": TrainingLossTermConfig(
                        weight=1.0,
                    ),
                },
                "regularization": {
                    "custom_test_regularization_loss": TrainingLossTermConfig(
                        weight=0.0001,
                    ),
                },
                "custom_objective": {
                    "custom_test_objective_loss": TrainingLossTermConfig(
                        weight=0.1,
                        params={
                            "regularization_weight": 0.0001,
                        },
                    ),
                },
            },
            "optimizer": TrainingOptimizerConfig(
                name="custom_test_optimizer",
                params={"lr": 0.001},
            ),
            "logger": TrainingLoggerConfig(
                name="custom_test_logger",
                params={},
            ),
            "additional_callbacks": [
                TrainingAdditionalCallbackConfig(
                    name="custom_test_callback",
                    params={
                        "marker": "configured_from_test",
                    },
                ),
            ],
        },
    )

    custom_callbacks = [
        callback
        for callback in training_result.trainer.callbacks
        if isinstance(callback, CustomRegisteredCallback)
    ]

    assert len(custom_callbacks) == 1
    assert custom_callbacks[0].marker == "configured_from_test"

    assert training_result.manifest_path.is_file()

    print("\n=== Prediction with custom registered components ===")
    prediction_result = predict_vae(
        config_path=(
            CONFIG_DIR
            / "prediction_tiny_synthetic.yaml"
        ),
        training_manifest_path=training_result.manifest_path,
        config_components={
            "dataset": dataset_config,
        },
    )

    assert prediction_result.manifest_path.is_file()

    print("\n=== Evaluation with custom registered components ===")
    evaluation_result = evaluate(
        config_path=(
            CONFIG_DIR
            / "evaluation_tiny_synthetic.yaml"
        ),
        prediction_manifest_path=prediction_result.manifest_path,
        config_components={
            "metrics": EvaluationMetricsConfig(
                clustering=EvaluationClusteringMetricsConfig(
                    internal=EvaluationInternalClusteringMetricConfig(
                        enabled=True,
                        selected=["custom_test_internal_metric"],
                        params={
                            "custom_test_internal_metric": {
                                "offset": 0.0,
                            },
                        },
                    ),
                    external=EvaluationExternalClusteringMetricConfig(
                        enabled=True,
                        label_key="label",
                        selected=["custom_test_external_metric"],
                        params={
                            "custom_test_external_metric": {
                                "offset": 0.0,
                            },
                        },
                    ),
                ),
                embedding=EvaluationEmbeddingMetricConfig(
                    enabled=True,
                    selected=["custom_test_embedding_metric"],
                    params={
                        "custom_test_embedding_metric": {
                            "scale": 1.0,
                        },
                    },
                ),
                predictability=EvaluationPredictabilityConfig(
                    enabled=False,
                ),
                reconstruction=EvaluationReconstructionMetricConfig(
                    enabled=True,
                    selected=[
                        "custom_test_reconstruction_metric",
                    ],
                    params={
                        "custom_test_reconstruction_metric": {
                            "offset": 0.0,
                        },
                    },
                    reduction="global",
                ),
            ),
            "plots": EvaluationPlotsConfig(
                enabled=False,
            ),
        },
    )

    assert evaluation_result.manifest_path.is_file()

    expected_calls = {
        "dataset_init",
        "dataset_getitem",
        "transform_factory",
        "transform_call",
        "encoder_init",
        "encoder_forward",
        "decoder_init",
        "decoder_forward",
        "reconstruction_loss",
        "regularization_loss",
        "custom_objective_loss",
        "optimizer_factory",
        "logger_init",
        "logger_metrics",
        "callback_init",
        "callback_train_start",
        "internal_clustering_metric",
        "external_clustering_metric",
        "embedding_metric",
        "reconstruction_metric",
    }

    missing_calls = {
        call_name
        for call_name in expected_calls
        if COMPONENT_CALLS[call_name] == 0
    }

    assert not missing_calls, (
        "Some custom registered components were never exercised: "
        f"{sorted(missing_calls)}. "
        f"Observed calls: {dict(COMPONENT_CALLS)}"
    )

    metrics = evaluation_result.adata.uns["benchrep"]["metrics"]

    assert (
        "custom_test_internal_metric"
        in metrics["clustering"]["internal"]["kmeans"]["metrics"]
    )
    assert (
        "custom_test_external_metric"
        in metrics["clustering"]["external"]["kmeans"]["metrics"]
    )
    assert (
        "custom_test_embedding_metric"
        in metrics["embedding"]["metrics"]
    )

    reconstruction_outputs = evaluation_result.reconstruction_outputs
    assert reconstruction_outputs is not None

    global_reconstruction_metrics = (
        reconstruction_outputs["reconstruction_metrics"]
        ["metrics"]["global"]
    )

    assert (
        "custom_test_reconstruction_metric"
        in global_reconstruction_metrics
    )