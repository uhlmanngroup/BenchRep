from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchrep.assembly.config import compose_effective_config, update_config
from benchrep.assembly.schemas import (
    EvaluationConfig,
    PredictionConfig,
    TrainingConfig,
    LinearScheduleConfig,
    MNISTDatasetConfig,
    MNISTDatasetParams,
    PredictionDataConfig,
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
    make_prediction_transform_pipelines_config,
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
    make_training_transform_pipelines_config,
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
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "transform_pipelines",
            make_training_transform_pipelines_config,
            {},
            id="training-transforms",
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
            "transform_pipelines",
            make_prediction_transform_pipelines_config,
            {"training_manifest_path_overridden": True},
            id="prediction-transforms",
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
        config=config_path,
        **composition_kwargs,
    )

    composition_result = compose_effective_config(
        schema=schema,
        config=config_path,
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
    ("schema", "full_config_factory", "section_name", "component_factory"),
    [
        pytest.param(
            TrainingConfig,
            make_training_config,
            "run",
            lambda: make_training_run_config().model_copy(
                update={"run_name_prefix": "replacement_component"}
            ),
            id="training",
        ),
        pytest.param(
            PredictionConfig,
            make_prediction_config,
            "data",
            lambda: make_prediction_data_config().model_copy(
                update={"batch_size": 7}
            ),
            id="prediction",
        ),
        pytest.param(
            EvaluationConfig,
            make_evaluation_config,
            "run",
            lambda: make_evaluation_run_config().model_copy(
                update={"run_name_stem": "replacement_component"}
            ),
            id="evaluation",
        ),
    ],
)
def test_config_object_component_replaces_top_level_section(
    schema: Any,
    full_config_factory: Callable[[], Any],
    section_name: str,
    component_factory: Callable[[], Any],
) -> None:
    """A component replaces its section of a supplied config object."""
    full_config = full_config_factory()
    replacement_component = component_factory()

    assert getattr(full_config, section_name) != replacement_component

    result = compose_effective_config(
        schema=schema,
        config=full_config,
        config_components={section_name: replacement_component},
    )

    assert result.effective_config is not full_config
    assert getattr(result.effective_config, section_name) == replacement_component

    for field_name in schema.model_fields:
        if field_name != section_name:
            assert (
                getattr(result.effective_config, field_name)
                == getattr(full_config, field_name)
            )

    assert getattr(full_config, section_name) != replacement_component
    assert result.effective_source == "config_object_with_components"
    assert result.yaml_supplied is False
    assert result.yaml_used_as_base is False
    assert result.original_config_path is None
    assert result.original_config_raw is None
    assert result.composition_warnings == ()

@pytest.mark.parametrize(
    "checkpointing",
    [{}, {"monitor": "losses/val/total_weighted_loss"}, {"monitor": None}],
    ids=["omitted", "explicit-default-metric", "explicit-null"],
)
def test_update_config_preserves_checkpoint_monitor_presence(checkpointing):
    """An unrelated trainer edit preserves exactly how the monitor was supplied."""
    base = TrainingConfig.model_validate({
        "overrides": {"model": {}, "datamodule": {}},
        "checkpointing": checkpointing,
    })

    updated = update_config(base, {"trainer.max_epochs": 5})

    assert updated.trainer.max_epochs == 5
    assert updated.model_fields_set == {"overrides", "checkpointing", "trainer"}
    assert updated.trainer.model_fields_set == {"max_epochs"}
    assert updated.checkpointing.monitor == base.checkpointing.monitor
    assert updated.checkpointing.model_fields_set == set(checkpointing)
    assert "trainer" not in base.model_fields_set


def test_update_config_preserves_presence_in_replacements_and_list_edits():
    """A whole-section replacement owns its fields. A list edit affects one item."""
    base = PredictionConfig.model_validate({
        "source": {"training_manifest_path": "training.yaml"},
        "data": {"batch_size": 32},
        "transform_pipelines": [
            {"steps": [{"name": "normalize"}, {"name": "normalize"}]},
        ],
    })

    updated = update_config(base, {
        "data": {"num_workers": 0},
        "transform_pipelines.0.steps.1.random_apply": 1.0,
    })

    assert updated.data.num_workers == 0
    assert updated.data.batch_size is None
    assert updated.data.model_fields_set == {"num_workers"}

    first, second = updated.transform_pipelines[0].steps
    assert first.model_fields_set == {"name"}
    assert second.model_fields_set == {"name", "random_apply"}
    assert base.data.batch_size == 32
    assert base.transform_pipelines[0].steps[1].model_fields_set == {"name"}


@pytest.mark.parametrize("base_kind", ["typed", "yaml", "components-only"])
def test_composition_preserves_base_and_replacement_field_presence(base_kind, tmp_path):
    """Each composition path preserves explicit nulls and the selected dataset type."""
    raw = {
        "source": {"training_manifest_path": "training.yaml"},
        "data": {"num_workers": 2},
    }
    typed = PredictionConfig.model_validate(raw)
    components = {
        "data": PredictionDataConfig(batch_size=None),
        "dataset": MNISTDatasetConfig(params=MNISTDatasetParams(root=tmp_path)),
    }

    if base_kind == "typed":
        base = typed
    elif base_kind == "yaml":
        base = tmp_path / "prediction.yaml"
        base.write_text(yaml.safe_dump(raw), encoding="utf-8")
    else:
        base = None
        components["source"] = typed.source

    effective = compose_effective_config(
        schema=PredictionConfig,
        config=base,
        config_components=components,
    ).effective_config

    assert effective.model_fields_set == {"source", "data", "dataset"}
    assert effective.source.model_fields_set == {"training_manifest_path"}
    assert effective.data.model_fields_set == {"batch_size"}
    assert effective.data.batch_size is None
    assert effective.data.num_workers is None
    assert isinstance(effective.dataset, MNISTDatasetConfig)
    assert effective.dataset.model_fields_set == {"params"}
    assert effective.dataset.params.model_fields_set == {"root"}


def test_dataset_name_edit_preserves_omitted_params(tmp_path):
    """Switching dataset class preserves the shared parameters' field presence."""
    base = PredictionConfig.model_validate({
        "source": {"training_manifest_path": "training.yaml"},
        "dataset": MNISTDatasetConfig(params=MNISTDatasetParams(root=tmp_path)),
    })

    updated = update_config(base, {"dataset.name": "cifar10"})

    assert updated.dataset.name == "cifar10"
    assert updated.dataset.model_fields_set == {"name", "params"}
    assert updated.dataset.params.model_fields_set == {"root"}


def test_composition_preserves_omitted_monitor_on_training_config_subclass():
    """A subclass used as a base keeps its checkpoint choices during composition."""
    class ExperimentConfig(TrainingConfig):
        pass

    base = ExperimentConfig.model_validate({"overrides": {"model": {}, "datamodule": {}}})
    effective = compose_effective_config(
        schema=TrainingConfig,
        config=base,
        config_components={"run": make_training_run_config()},
    ).effective_config

    assert effective.model_fields_set == {"overrides", "run"}
    assert "monitor" not in effective.checkpointing.model_fields_set


def test_raw_loss_replacement_preserves_nested_typed_schedule_presence():
    """The schedule's validator can reparse it while a whole loss term is replaced."""
    base = TrainingConfig.model_validate({
        "overrides": {"model": {}, "datamodule": {}},
        "losses": {"reconstruction": [{"name": "mse"}]},
    })
    schedule = LinearScheduleConfig(name="linear", params={"end": 5})

    updated = update_config(base, {
        "losses.reconstruction.0": {"name": "mse", "weight_schedule": schedule},
    })

    result = updated.losses["reconstruction"][0].weight_schedule
    assert result.model_fields_set == {"name", "params"}
    assert result.params.model_fields_set == {"end"}
