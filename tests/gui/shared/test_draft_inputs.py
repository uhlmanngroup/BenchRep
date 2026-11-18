import pytest
import yaml
from pydantic import BaseModel, Field, RootModel, field_serializer

pytest.importorskip("dash_mantine_components")

from benchrep.assembly import schemas
from benchrep.gui.rules.evaluate import prepare_evaluation_draft
from benchrep.gui.rules.predict import prepare_prediction_draft
from benchrep.gui.rules.train import prepare_training_draft
from benchrep.interfaces.model_families import AUTOENCODER_FAMILY
from benchrep.utils.workflow_overrides import (
    EvaluationOverrides,
    PredictionOverrides,
    TrainingOverrides,
)
from tests.fixtures.configs.configs import make_training_config


def _reload_authored_yaml(draft):
    return yaml.safe_load(yaml.safe_dump(draft.initial_data))


def test_training_draft_preserves_checkpoint_omission_and_gradient_kind():
    config = make_training_config()
    config.inspection.gradient_norms.append(
        schemas.OptimizerGradientNormConfig(target="whole_model")
    )

    draft = prepare_training_draft(
        model_family=AUTOENCODER_FAMILY,
        config=config,
        overrides=TrainingOverrides(
            config_components={
                "checkpointing": schemas.TrainingCheckpointConfig(save_last=True),
            }
        ),
    )
    raw = _reload_authored_yaml(draft)

    assert draft.supplied_sections == {"checkpointing"}
    assert raw["checkpointing"] == {"save_last": True}
    assert raw["inspection"]["gradient_norms"] == [
        {"kind": "optimizer_gradients", "target": "whole_model"}
    ]

    reloaded = schemas.TrainingConfig.model_validate(raw)
    assert reloaded.checkpointing.model_fields_set == {"save_last"}
    assert isinstance(
        reloaded.inspection.gradient_norms[0], schemas.OptimizerGradientNormConfig
    )


def test_prediction_draft_preserves_dataset_type_and_explicit_null(tmp_path):
    config = schemas.PredictionConfig.model_validate({
        "source": {"training_manifest_path": tmp_path / "training.yaml"},
        "dataset": schemas.CIFAR10DatasetConfig(
            params=schemas.CIFAR10DatasetParams(root=tmp_path)
        ),
    })

    draft = prepare_prediction_draft(
        model_family=AUTOENCODER_FAMILY,
        config=config,
        overrides=PredictionOverrides(
            config_components={"data": schemas.PredictionDataConfig(batch_size=None)}
        ),
    )
    raw = _reload_authored_yaml(draft)

    assert draft.supplied_sections == {"data"}
    assert raw["dataset"] == {"name": "cifar10", "params": {"root": str(tmp_path)}}
    assert raw["data"] == {"batch_size": None}

    reloaded = schemas.PredictionConfig.model_validate(raw)
    assert isinstance(reloaded.dataset, schemas.CIFAR10DatasetConfig)
    assert reloaded.data.model_fields_set == {"batch_size"}


@pytest.mark.parametrize(
    ("probe_class", "task", "model_name"),
    [
        (schemas.EvaluationLogisticRegressionProbeConfig, "classification", "logistic_regression"),
        (schemas.EvaluationRidgeProbeConfig, "regression", "ridge"),
    ],
)
def test_evaluation_draft_preserves_edits_under_default_sections(
    tmp_path, probe_class, task, model_name
):
    config = schemas.EvaluationConfig.model_validate({
        "source": {"anndata_path": tmp_path / "embeddings.h5ad"},
    })
    target = config.metrics.predictability.targets["label"]
    target.task = task
    target.params.linear = probe_class()

    draft = prepare_evaluation_draft(config=config, overrides=EvaluationOverrides())
    raw = _reload_authored_yaml(draft)

    assert set(raw) == {"source", "metrics"}
    assert raw["metrics"]["predictability"]["targets"]["label"] == {
        "task": task,
        "params": {"linear": {"model": model_name}},
    }

    reloaded = schemas.EvaluationConfig.model_validate(raw)
    assert isinstance(reloaded.metrics.predictability.targets["label"].params.linear, probe_class)


@pytest.mark.parametrize("component_override", [False, True], ids=["typed", "component"])
def test_wizard_respects_parameter_serialization(component_override):
    """The wizard respects exclusions, custom serializers, and root models in dataset params."""
    class Options(BaseModel):
        label: str
        private: str = Field(exclude=True)
        limit: int = 4

        @field_serializer("label")
        def serialize_label(self, value, info):
            return f"{info.mode}-{value.upper()}"

    dataset = schemas.CustomDatasetConfig(name="custom", params={
        "options": Options(label="test", private="hidden"),
        "ids": RootModel[list[int]]([1, 2]),
    })
    source = schemas.PredictionSourceConfig(training_manifest_path="train.yaml")
    if component_override:
        config = schemas.PredictionConfig(source=source)
        overrides = PredictionOverrides(config_components={"dataset": dataset})
        expected = {"label": "python-TEST", "limit": 4}
    else:
        config = schemas.PredictionConfig(source=source, dataset=dataset)
        overrides = PredictionOverrides()
        expected = {"label": "json-TEST"}

    draft = prepare_prediction_draft(
        model_family=AUTOENCODER_FAMILY, config=config, overrides=overrides
    )
    raw = _reload_authored_yaml(draft)
    assert raw["dataset"]["params"] == {"options": expected, "ids": [1, 2]}


@pytest.mark.parametrize("component_override", [False, True], ids=["typed", "component"])
def test_wizard_keeps_omitted_custom_dataset_params(component_override):
    """The default empty kwargs dict remains omitted for either typed input route."""
    dataset = schemas.CustomDatasetConfig(name="custom")
    source = schemas.PredictionSourceConfig(training_manifest_path="train.yaml")
    if component_override:
        config = schemas.PredictionConfig(source=source)
        overrides = PredictionOverrides(config_components={"dataset": dataset})
    else:
        config = schemas.PredictionConfig(source=source, dataset=dataset)
        overrides = PredictionOverrides()

    draft = prepare_prediction_draft(
        model_family=AUTOENCODER_FAMILY, config=config, overrides=overrides
    )
    assert _reload_authored_yaml(draft)["dataset"] == {"name": "custom"}
