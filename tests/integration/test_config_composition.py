from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from benchrep.assembly.config import compose_effective_config, update_config
from benchrep.assembly.schemas import (
    EvaluationConfig,
    ConfigContext,
    TrainingConfigContext,
    PredictionConfigContext,
    EvaluationConfigContext,
    PredictionConfig,
    TrainingConfig,
    LinearScheduleConfig,
    CosineScheduleConfig,
    SigmoidScheduleConfig,
    ExponentialScheduleConfig,
    PiecewiseLinearScheduleConfig,
    PiecewiseConstantScheduleConfig,
    CustomScheduleConfig,
    TrainingLossTermConfig,
    parse_training_config,
    parse_prediction_config,
    parse_evaluation_config,
    MNISTDatasetConfig,
    MNISTDatasetParams,
    PredictionDataConfig,
)
from benchrep.interfaces.model_families import (
    AUTOENCODER_FAMILY,
    VAE_FAMILY,
    COMPOSITE_FAMILY,
    YOYO_FAMILY,
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

_CONTEXTS = {
    TrainingConfig: TrainingConfigContext(external_model=False, external_datamodule=False),
    PredictionConfig: PredictionConfigContext(
        model_family=AUTOENCODER_FAMILY,
        training_manifest_path_overridden=True,
    ),
    EvaluationConfig: EvaluationConfigContext(prediction_manifest_path_overridden=True),
}


@pytest.mark.parametrize(
    (
        "schema",
        "config_name",
        "section_name",
        "component_factory",
        "context",
    ),
    [
        # Training
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "run",
            make_training_run_config,
            _CONTEXTS[TrainingConfig],
            id="training-run",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "reproducibility",
            make_training_reproducibility_config,
            _CONTEXTS[TrainingConfig],
            id="training-reproducibility",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "model",
            make_training_model_config,
            _CONTEXTS[TrainingConfig],
            id="training-model",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "encoder",
            make_training_encoder_config,
            _CONTEXTS[TrainingConfig],
            id="training-encoder",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "decoder",
            make_training_decoder_config,
            _CONTEXTS[TrainingConfig],
            id="training-decoder",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "losses",
            make_training_losses_config,
            _CONTEXTS[TrainingConfig],
            id="training-losses",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "optimizer",
            make_training_optimizer_config,
            _CONTEXTS[TrainingConfig],
            id="training-optimizer",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "dataset",
            make_training_custom_dataset_config,
            _CONTEXTS[TrainingConfig],
            id="training-custom-dataset",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "dataset",
            make_training_mnist_dataset_config,
            _CONTEXTS[TrainingConfig],
            id="training-mnist-dataset",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "datamodule",
            make_training_datamodule_config,
            _CONTEXTS[TrainingConfig],
            id="training-datamodule",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "trainer",
            make_training_trainer_config,
            _CONTEXTS[TrainingConfig],
            id="training-trainer",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "logger",
            make_training_logger_config,
            _CONTEXTS[TrainingConfig],
            id="training-logger",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "checkpointing",
            make_training_checkpoint_config,
            _CONTEXTS[TrainingConfig],
            id="training-checkpointing",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "inspection",
            make_training_inspection_config,
            _CONTEXTS[TrainingConfig],
            id="training-inspection",
        ),
        pytest.param(
            TrainingConfig,
            "training_tiny_synthetic_ae.yaml",
            "transform_pipelines",
            make_training_transform_pipelines_config,
            _CONTEXTS[TrainingConfig],
            id="training-transforms",
        ),

        # Prediction
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "source",
            make_prediction_source_config,
            _CONTEXTS[PredictionConfig],
            id="prediction-source",
        ),
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "dataset",
            make_prediction_dataset_config,
            _CONTEXTS[PredictionConfig],
            id="prediction-dataset",
        ),
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "data",
            make_prediction_data_config,
            _CONTEXTS[PredictionConfig],
            id="prediction-data",
        ),
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "transform_pipelines",
            make_prediction_transform_pipelines_config,
            _CONTEXTS[PredictionConfig],
            id="prediction-transforms",
        ),
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "inference",
            make_prediction_inference_config,
            _CONTEXTS[PredictionConfig],
            id="prediction-inference",
        ),
        pytest.param(
            PredictionConfig,
            "prediction_tiny_synthetic.yaml",
            "exports",
            make_prediction_exports_config,
            _CONTEXTS[PredictionConfig],
            id="prediction-exports",
        ),

        # Evaluation
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "source",
            make_evaluation_source_config,
            _CONTEXTS[EvaluationConfig],
            id="evaluation-source",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "run",
            make_evaluation_run_config,
            _CONTEXTS[EvaluationConfig],
            id="evaluation-run",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "reductions",
            make_evaluation_reductions_config,
            _CONTEXTS[EvaluationConfig],
            id="evaluation-reductions",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "clustering",
            make_evaluation_clustering_config,
            _CONTEXTS[EvaluationConfig],
            id="evaluation-clustering",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "metrics",
            make_evaluation_metrics_config,
            _CONTEXTS[EvaluationConfig],
            id="evaluation-metrics",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "reconstruction",
            make_evaluation_reconstruction_config,
            _CONTEXTS[EvaluationConfig],
            id="evaluation-reconstruction",
        ),
        pytest.param(
            EvaluationConfig,
            "evaluation_tiny_synthetic.yaml",
            "plots",
            make_evaluation_plots_config,
            _CONTEXTS[EvaluationConfig],
            id="evaluation-plots",
        ),
    ],
)
def test_yaml_component_replaces_top_level_section(
    schema: Any,
    config_name: str,
    section_name: str,
    component_factory: Callable[[], Any],
    context: ConfigContext,
) -> None:
    """A supplied component replaces only its matching YAML section."""
    config_path = CONFIG_DIR / config_name
    replacement_component = component_factory()

    baseline_result = compose_effective_config(
        schema=schema,
        config=config_path,
        context=context,
    )

    composition_result = compose_effective_config(
        schema=schema,
        config=config_path,
        config_components={
            section_name: replacement_component,
        },
        context=context,
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
        context=_CONTEXTS[schema],
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

    updated = update_config(base, {"trainer.max_epochs": 5}, context=_CONTEXTS[TrainingConfig])

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
    }, context=_CONTEXTS[PredictionConfig])

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
        context=_CONTEXTS[PredictionConfig],
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

    updated = update_config(base, {"dataset.name": "cifar10"}, context=_CONTEXTS[PredictionConfig])

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
        context=_CONTEXTS[TrainingConfig],
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
    }, context=_CONTEXTS[TrainingConfig])

    result = updated.losses["reconstruction"][0].weight_schedule
    assert result.model_fields_set == {"name", "params"}
    assert result.params.model_fields_set == {"end"}


@pytest.mark.parametrize(
    ("parser", "raw", "context", "path", "value"),
    [
        pytest.param(
            parse_training_config,
            {},
            TrainingConfigContext(external_model=True, external_datamodule=True),
            "trainer.max_epochs",
            5,
            id="external-training-components",
        ),
        pytest.param(
            parse_prediction_config,
            {},
            PredictionConfigContext(
                model_family=YOYO_FAMILY, training_manifest_path_overridden=False
            ),
            "data.max_batches",
            1,
            id="standalone-yoyo",
        ),
        pytest.param(
            parse_prediction_config,
            {"source": {"checkpoint": None}},
            PredictionConfigContext(
                model_family=YOYO_FAMILY, training_manifest_path_overridden=False
            ),
            "data.max_batches",
            1,
            id="standalone-yoyo-current-weights",
        ),
        pytest.param(
            parse_prediction_config,
            {"source": {"training_manifest_path": "training.yaml", "checkpoint": None}},
            PredictionConfigContext(
                model_family=YOYO_FAMILY, training_manifest_path_overridden=False
            ),
            "data.max_batches",
            1,
            id="linked-yoyo-current-weights",
        ),
        pytest.param(
            parse_prediction_config,
            {},
            PredictionConfigContext(
                model_family=AUTOENCODER_FAMILY, training_manifest_path_overridden=True
            ),
            "data.max_batches",
            1,
            id="prediction-manifest-override",
        ),
        pytest.param(
            parse_evaluation_config,
            {},
            EvaluationConfigContext(prediction_manifest_path_overridden=True),
            "run.run_name_stem",
            "edited",
            id="evaluation-manifest-override",
        ),
    ],
)
def test_parser_context_can_be_reused_for_updates(parser, raw, context, path, value):
    """Context-dependent configs stay editable while keeping validation and omissions."""
    base = parser(raw, context=context)
    before = base.model_dump()

    updated = update_config(base, {path: value}, context=context)

    section, field = path.split(".")
    assert getattr(getattr(updated, section), field) == value
    assert updated.model_fields_set == base.model_fields_set | {section}
    assert base.model_dump() == before
    if isinstance(base, PredictionConfig):
        assert updated.source.model_fields_set == base.source.model_fields_set
        assert updated.source.checkpoint == base.source.checkpoint

    with pytest.raises(ValidationError):
        update_config(base, {"stage": "invalid"}, context=context)


@pytest.mark.parametrize(
    ("parser", "schema", "raw", "context"),
    [
        (
            parse_training_config,
            TrainingConfig,
            {"overrides": {"model": {}, "datamodule": {}}},
            _CONTEXTS[TrainingConfig],
        ),
        (parse_prediction_config, PredictionConfig, {}, _CONTEXTS[PredictionConfig]),
        (parse_evaluation_config, EvaluationConfig, {}, _CONTEXTS[EvaluationConfig]),
    ],
)
def test_config_helpers_require_matching_context(parser, schema, raw, context):
    """The public helpers reject missing or mismatched contexts, including typed bases."""
    base = parser(raw, context=context)
    with pytest.raises(TypeError, match="context"):
        parser(raw)
    with pytest.raises(TypeError, match="context"):
        update_config(base, {})
    with pytest.raises(TypeError, match="context"):
        compose_effective_config(schema=schema, config=base)

    for wrong in [
        None,
        {},
        *[c for c in _CONTEXTS.values() if type(c) is not type(context)],
    ]:
        with pytest.raises(TypeError, match=type(context).__name__):
            parser(raw, context=wrong)
        with pytest.raises(TypeError, match=type(context).__name__):
            update_config(base, {}, context=wrong)
        with pytest.raises(TypeError, match=type(context).__name__):
            compose_effective_config(schema=schema, config=base, context=wrong)


@pytest.mark.parametrize("family", [AUTOENCODER_FAMILY, VAE_FAMILY, COMPOSITE_FAMILY])
@pytest.mark.parametrize("path", ["source.checkpoint", "source.training_manifest_path"])
def test_prediction_context_keeps_other_families_source_requirements(family, path):
    """The context API preserves the existing non-YOYO source requirements."""
    context = PredictionConfigContext(
        model_family=family,
        training_manifest_path_overridden=False,
    )
    base = parse_prediction_config(
        {"source": {"training_manifest_path": "training.yaml"}},
        context=context,
    )
    with pytest.raises(ValidationError):
        update_config(base, {path: None}, context=context)


@pytest.mark.parametrize("base_kind", ["yaml", "typed-with-components"])
def test_yoyo_composition_preserves_context_and_checkpoint_omission(
    base_kind, tmp_path
):
    """The family context reaches parsing through both composition paths."""
    context = PredictionConfigContext(
        model_family=YOYO_FAMILY,
        training_manifest_path_overridden=False,
    )
    if base_kind == "yaml":
        base = tmp_path / "prediction.yaml"
        base.write_text("data:\n  max_batches: 1\n", encoding="utf-8")
        components = None
    else:
        base = parse_prediction_config({}, context=context)
        components = {"data": PredictionDataConfig(max_batches=1)}

    effective = compose_effective_config(
        schema=PredictionConfig,
        config=base,
        config_components=components,
        context=context,
    ).effective_config

    assert effective.data.max_batches == 1
    assert effective.source.training_manifest_path is None
    assert effective.source.model_fields_set == set()


def test_update_context_preserves_config_subclass():
    """Updating with context retains the concrete config class and its validators."""

    class ExperimentConfig(TrainingConfig):
        pass

    context = TrainingConfigContext(external_model=True, external_datamodule=True)
    base = ExperimentConfig.model_validate(
        {},
        context={
            "model_is_external": True,
            "datamodule_overridden": True,
        },
    )
    result = update_config(base, {"trainer.max_epochs": 5}, context=context)
    assert type(result) is ExperimentConfig
    assert result.trainer.max_epochs == 5
    assert "monitor" not in result.checkpointing.model_fields_set


@pytest.mark.parametrize(
    ("schedule_type", "name", "params"),
    [
        (LinearScheduleConfig, "linear", {"end": 5}),
        (CosineScheduleConfig, "cosine", {"end": 5}),
        (SigmoidScheduleConfig, "sigmoid", {"end": 5}),
        (ExponentialScheduleConfig, "exponential", {"end": 5, "start_factor": 0.1}),
        (
            PiecewiseLinearScheduleConfig,
            "piecewise_linear",
            {"milestones": [(0, 0.0), (5, 1.0)]},
        ),
        (
            PiecewiseConstantScheduleConfig,
            "piecewise_constant",
            {"milestones": [(0, 0.0), (5, 1.0)]},
        ),
        (CustomScheduleConfig, "custom_schedule_without_params", None),
    ],
)
@pytest.mark.parametrize("explicit_defaults", [False, True])
def test_typed_schedule_preserves_authored_fields(
    schedule_type, name, params, explicit_defaults
):
    """Nesting typed schedules preserves omissions, explicit defaults, and all values."""
    raw = {"name": name}
    if params is not None:
        raw["params"] = params
    schedule = schedule_type.model_validate(raw)
    if explicit_defaults:
        raw = schedule.model_dump()
        schedule = schedule_type.model_validate(raw)

    result = TrainingLossTermConfig(
        name="mse", weight_schedule=schedule
    ).weight_schedule

    assert result.model_dump() == schedule.model_dump()
    assert result.model_dump(exclude_unset=True) == raw
    assert result.model_fields_set == schedule.model_fields_set
    if isinstance(result.params, BaseModel):
        assert result.params.model_fields_set == schedule.params.model_fields_set


def test_typed_schedule_keeps_custom_parameter_payload():
    """Filtering schedule fields must not strip defaults inside custom payloads."""

    class Payload(BaseModel):
        enabled: bool = True

    schedule = CustomScheduleConfig(
        name="custom_payload_schedule",
        params={"payload": Payload()},
    )
    result = TrainingLossTermConfig(
        name="mse", weight_schedule=schedule
    ).weight_schedule
    assert result.params == {"payload": {"enabled": True}}
    assert result.model_fields_set == {"name", "params"}


def test_typed_schedule_still_revalidates_parameters():
    """Invalid edits to a typed schedule are rejected when it is nested."""
    schedule = LinearScheduleConfig(name="linear", params={"end": 5})
    schedule.params.start_factor = -1.0
    with pytest.raises(ValidationError):
        TrainingLossTermConfig(name="mse", weight_schedule=schedule)


def test_typed_schedule_subclass_keeps_defaulted_discriminator():
    """A subclass may default its name, but normalization still needs that tag."""
    from typing import Literal

    class DefaultLinearSchedule(LinearScheduleConfig):
        name: Literal["linear"] = "linear"

    schedule = DefaultLinearSchedule(params={"end": 5})
    result = TrainingLossTermConfig(name="mse", weight_schedule=schedule).weight_schedule
    assert result.name == "linear"
    assert result.model_fields_set == {"params"}
    assert result.params.model_fields_set == {"end"}


def test_typed_schedule_keeps_in_place_custom_parameter_edits():
    """Editing a default parameter mapping in place must survive normalization."""
    from benchrep.assembly.config import prepare_authored

    schedule = CustomScheduleConfig(name="custom_schedule")
    schedule.params["factor"] = 0.5

    result = TrainingLossTermConfig(name="mse", weight_schedule=schedule).weight_schedule

    assert result.params == {"factor": 0.5}
    assert result.model_fields_set == schedule.model_fields_set == {"name"}
    _, included = prepare_authored(result)
    assert result.model_dump(include=included) == {
        "name": "custom_schedule", "params": {"factor": 0.5}
    }


def test_typed_schedule_keeps_subclass_default_values():
    """Normalization retains actual values when a caller changes schema defaults."""
    from typing import Literal

    class StepSchedule(LinearScheduleConfig):
        interval: Literal["epoch", "step"] = "step"

    schedule = StepSchedule(name="linear", params={"end": 5})
    result = TrainingLossTermConfig(name="mse", weight_schedule=schedule).weight_schedule

    assert result.interval == "step"
    assert result.params.end == 5


def test_typed_schedule_original_record_preserves_omissions(tmp_path):
    """The original YAML records only authored schedule fields; resolved YAML is dense."""
    from benchrep.records.configs import save_config_records

    schedule = LinearScheduleConfig(
        name="linear", params={"end": 5}, cycle={"length": 10}
    )
    config = parse_training_config(
        {
            "overrides": {"model": {}, "datamodule": {}},
            "losses": {
                "reconstruction": [
                    TrainingLossTermConfig(name="mse", weight_schedule=schedule)
                ]
            },
        },
        context=_CONTEXTS[TrainingConfig],
    )
    save_config_records(
        config_out_dir=tmp_path, original_config=config, resolved_config=config
    )

    original = yaml.safe_load((tmp_path / "original_config.yaml").read_text())
    resolved = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    assert original["losses"]["reconstruction"][0]["weight_schedule"] == {
        "name": "linear",
        "params": {"end": 5},
        "cycle": {"length": 10},
    }
    assert (
        resolved["losses"]["reconstruction"][0]["weight_schedule"]
        == schedule.model_dump()
    )
