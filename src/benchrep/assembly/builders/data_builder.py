from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel

import torch

from benchrep.records import get_run_logger
from benchrep.architecture.data import (
    BaseDataset,
    BenchRepDataModule,
    TransformPipeline,
    TransformStep,
)
from benchrep.assembly.schemas import (
    PredictionTransformPipelineConfig,
    PredictionTransformStepConfig,
    TrainingDataModuleConfig,
    TrainingTransformPipelineConfig,
    TrainingTransformStepConfig,
    SupportedDatasetConfig,
)
from benchrep.assembly.schemas.training_config_schema import NamedConfig
from benchrep.assembly.registries.utils import normalize_name
from benchrep.assembly.registries.core import DATASETS, TRANSFORMS


SupportedTransformStepConfig: TypeAlias = (
    TrainingTransformStepConfig | PredictionTransformStepConfig
)

SupportedTransformPipelineConfig: TypeAlias = (
    TrainingTransformPipelineConfig | PredictionTransformPipelineConfig
)


@dataclass(frozen=True)
class TransformPipelineBundle:
    """Split-specific ordered transform pipelines."""

    training: tuple[TransformPipeline, ...]
    validation: tuple[TransformPipeline, ...]


def build_datamodule(
    *,
    dataset: BaseDataset,
    datamodule_config: TrainingDataModuleConfig,
    seed: int | None = None,
    stage: Literal["training", "prediction"],
    training_pipelines: Sequence[TransformPipeline] | None = None,
    validation_pipelines: Sequence[TransformPipeline] | None = None,
    prediction_pipelines: Sequence[TransformPipeline] | None = None,
) -> BenchRepDataModule:
    """Build a BenchRep datamodule around an instantiated dataset.

    During training, ``dataset`` becomes the training dataset and may be
    divided into training and validation subsets according to
    ``datamodule_config.val_fraction``. The resulting subsets receive
    ``training_pipelines`` and ``validation_pipelines`` respectively.

    During prediction, ``dataset`` becomes the prediction dataset and receives
    ``prediction_pipelines``. If no explicit prediction pipeline is supplied,
    ``BenchRepDataModule`` falls back to ``validation_pipelines``.

    Parameters
    ----------
    dataset:
        Instantiated dataset assigned according to ``stage``.
    datamodule_config:
        Batching, data-loading, and optional validation-splitting settings.
    seed:
        Optional seed used for reproducible train-validation splitting.
    stage:
        Whether the dataset is assigned for training or prediction.
    training_pipelines:
        Ordered transforms applied to training samples.
    validation_pipelines:
        Ordered transforms applied to validation samples. Also used for test
        samples and as the prediction fallback when no explicit prediction
        pipeline is supplied.
    prediction_pipelines:
        Optional explicit transforms applied to prediction samples.

    Returns
    -------
    BenchRepDataModule
        Configured BenchRep datamodule.

    Raises
    ------
    ValueError
        If ``stage`` is neither ``"training"`` nor ``"prediction"``.
    """
    run_log = get_run_logger()

    if stage not in {"training", "prediction"}:
        raise ValueError(
            "stage must be either 'training' or 'prediction', "
            f"got {stage!r}."
        )

    dm = _instantiate_datamodule(
        datamodule_config=datamodule_config,
        train_dataset=dataset if stage == "training" else None,
        predict_dataset=dataset if stage == "prediction" else None,
        training_pipelines=training_pipelines,
        validation_pipelines=validation_pipelines,
        prediction_pipelines=prediction_pipelines,
        seed=seed,
    )

    run_log.info(
        "Built datamodule: stage=%s, dataset=%s, datamodule=%s",
        stage,
        type(dataset).__name__,
        type(dm).__name__,
    )

    return dm


def build_dataset(
    *,
    dataset_config: SupportedDatasetConfig,
) -> BaseDataset:
    """Build a registered BenchRep dataset from validated configuration.

    The dataset name is resolved through the dataset registry. Typed built-in
    parameters or arbitrary custom parameters are converted to constructor
    keyword arguments before the registered dataset callable is invoked.

    Parameters
    ----------
    dataset_config:
      Validated built-in or custom dataset configuration containing the
      registered dataset name and its constructor parameters.

    Returns
    -------
    BaseDataset
      Instantiated BenchRep-compatible dataset.

    Raises
    ------
    KeyError
      If the configured dataset is not registered.
    TypeError
      If the registered dataset does not produce a ``BaseDataset`` instance.
    """
    run_log = get_run_logger()

    dataset_name = normalize_name(
        dataset_config.name,
        field_name="config.dataset.name",
    )
    dataset_factory = DATASETS.get(dataset_name)

    raw_params = dataset_config.params
    if isinstance(raw_params, BaseModel):
        dataset_params = raw_params.model_dump(mode="python")
    else:
        dataset_params = dict(raw_params)

    run_log.info("Building dataset: dataset=%s", dataset_name)

    dataset = dataset_factory(**dataset_params)

    if not isinstance(dataset, BaseDataset):
        raise TypeError(
            f"Registered dataset {dataset_name!r} must produce a BaseDataset "
            f"instance, got {type(dataset).__name__}."
        )

    run_log.info(
        "Built dataset: dataset=%s, class=%s, samples=%d",
        dataset_name,
        type(dataset).__name__,
        len(dataset),
    )

    return dataset


def build_transform_pipelines_bundle(
    transform_pipeline_configs: Sequence[TrainingTransformPipelineConfig],
) -> TransformPipelineBundle:
    """Build training and validation transform-pipeline sequences."""

    return TransformPipelineBundle(
        training=build_transform_pipeline_sequence(
            transform_pipeline_configs,
            apply_to="training",
        ),
        validation=build_transform_pipeline_sequence(
            transform_pipeline_configs,
            apply_to="validation",
        ),
    )


def build_transform_pipeline_sequence(
    transform_pipeline_configs: Sequence[
        SupportedTransformPipelineConfig
    ],
    *,
    apply_to: Literal["training", "validation"] | None = None,
) -> tuple[TransformPipeline, ...]:
    """Build one ordered sequence of routed transform pipelines."""

    pipelines: list[TransformPipeline] = []

    for pipeline_index, pipeline_config in enumerate(
        transform_pipeline_configs
    ):
        assert pipeline_config.input is not None
        assert pipeline_config.output is not None

        pipeline = build_transform_pipeline(
            pipeline_config.steps,
            input_key=pipeline_config.input,
            output_key=pipeline_config.output,
            config_path=(
                f"config.transform_pipelines[{pipeline_index}].steps"
            ),
            apply_to=apply_to,
        )

        if len(pipeline) > 0:
            pipelines.append(pipeline)

    return tuple(pipelines)


def build_transform_pipeline(
    transform_step_configs: Sequence[SupportedTransformStepConfig],
    *,
    input_key: str,
    output_key: str,
    config_path: str,
    apply_to: Literal["training", "validation"] | None = None,
) -> TransformPipeline:
    """Build one routed, ordered transform pipeline."""

    steps: list[TransformStep] = []

    for index, transform_step_config in enumerate(
        transform_step_configs
    ):
        if apply_to is not None:
            if not isinstance(
                transform_step_config,
                TrainingTransformStepConfig,
            ):
                raise TypeError(
                    "`apply_to` filtering requires training transform steps."
                )

            if apply_to not in transform_step_config.apply_to:
                continue

        steps.append(
            _build_transform_step(
                transform_step_config,
                config_path=f"{config_path}[{index}]",
            )
        )

    return TransformPipeline(
        input_key=input_key,
        output_key=output_key,
        steps=steps,
    )


def _build_transform_step(
    transform_step_config: NamedConfig,
    *,
    config_path: str,
) -> TransformStep:
    """Resolve and instantiate one registered transform step."""

    transform_name = normalize_name(
        transform_step_config.name,
        field_name=f"{config_path}.name",
    )

    transform = TRANSFORMS.create(
        transform_name,
        **transform_step_config.params,
    )

    return TransformStep(
        name=transform_name,
        transform=transform,
    )


def _instantiate_datamodule(
    *,
    datamodule_config: TrainingDataModuleConfig,
    seed: int | None = None,
    train_dataset: Any | None = None,
    val_dataset: Any | None = None,
    test_dataset: Any | None = None,
    predict_dataset: Any | None = None,
    training_pipelines: Sequence[TransformPipeline] | None = None,
    validation_pipelines: Sequence[TransformPipeline] | None = None,
    prediction_pipelines: Sequence[TransformPipeline] | None = None,
) -> BenchRepDataModule:
    datamodule_params = datamodule_config.model_dump()

    # Resolve "auto" to pin CPU memory only when CUDA is available.
    if datamodule_params.get("pin_memory") == "auto":
        datamodule_params["pin_memory"] = torch.cuda.is_available()

    return BenchRepDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        predict_dataset=predict_dataset,
        training_pipelines=training_pipelines,
        validation_pipelines=validation_pipelines,
        prediction_pipelines=prediction_pipelines,
        seed=seed,
        **datamodule_params,
    )
