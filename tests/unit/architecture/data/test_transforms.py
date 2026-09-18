import torch
from torch.utils.data import DataLoader

from benchrep.architecture.data import (
    TransformPipeline,
    TransformStep,
    TransformedDataset,
    BenchRepDataModule,
)
from benchrep.assembly.builders.data_builder import (
    build_transform_pipelines_bundle,
)
from benchrep.assembly.schemas import (
    TrainingTransformPipelineConfig,
    TrainingTransformStepConfig,
)
from tests.fixtures.datasets import TinySyntheticDataset


class _AddRandomNoise:
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.rand_like(value)


def test_transform_pipeline_applies_steps_in_order() -> None:
    pipeline = TransformPipeline(
        input_key="x",
        output_key="x",
        steps=[
            TransformStep(
                name="add_two",
                transform=lambda value: value + 2,
            ),
            TransformStep(
                name="multiply_by_three",
                transform=lambda value: value * 3,
            ),
        ]
    )

    result = pipeline(torch.tensor([1.0]))

    assert torch.equal(result, torch.tensor([9.0]))


def test_empty_transform_pipeline_is_identity() -> None:
    value = torch.tensor([1.0])
    pipeline = TransformPipeline(
        input_key="x",
        output_key="x",
    )

    result = pipeline(value)

    assert result is value


def test_transformed_dataset_replaces_only_x() -> None:
    source_dataset = TinySyntheticDataset(n_samples=4)
    source_sample = source_dataset[0]
    source_x = source_sample["x"].clone()

    pipeline = TransformPipeline(
        input_key="x",
        output_key="x",
        steps=[
            TransformStep(
                name="add_five",
                transform=lambda value: value + 5,
            )
        ]
    )
    dataset = TransformedDataset(
        dataset=source_dataset,
        pipelines=[pipeline],
    )

    transformed_sample = dataset[0]

    assert torch.equal(
        transformed_sample["x"],
        source_x + 5,
    )
    assert transformed_sample["label"] == source_sample["label"]
    assert transformed_sample["sample_id"] == source_sample["sample_id"]
    assert transformed_sample["metadata"] == source_sample["metadata"]
    assert torch.equal(source_dataset[0]["x"], source_x)


def test_builder_filters_steps_by_training_and_validation_applicability() -> None:
    pipeline_config = TrainingTransformPipelineConfig(
        input="x",
        output="x",
        steps=[
            TrainingTransformStepConfig(
                name="to_dtype",
                apply_to=["training"],
                params={"dtype": "float32", "scale": False},
            ),
            TrainingTransformStepConfig(
                name="to_dtype",
                apply_to=["validation"],
                params={"dtype": "float64", "scale": False},
            ),
        ],
    )

    bundle = build_transform_pipelines_bundle([pipeline_config])
    value = torch.tensor([0, 255], dtype=torch.uint8)

    assert len(bundle.training) == len(bundle.validation) == 1
    assert bundle.training[0](value).dtype == torch.float32
    assert bundle.validation[0](value).dtype == torch.float64


def test_datamodule_applies_training_validation_and_prediction_pipelines() -> None:
    source_dataset = TinySyntheticDataset(n_samples=8)

    validation_pipeline = TransformPipeline(
        input_key="x",
        output_key="x",
        steps=[
            TransformStep(
                name="add_one",
                transform=lambda value: value + 1,
            ),
            TransformStep(
                name="add_three",
                transform=lambda value: value + 3,
            ),
        ],
    )
    training_pipeline = TransformPipeline(
        input_key="x",
        output_key="x",
        steps=[
            TransformStep(
                name="add_one",
                transform=lambda value: value + 1,
            ),
            TransformStep(
                name="multiply_by_two",
                transform=lambda value: value * 2,
            ),
            TransformStep(
                name="add_three",
                transform=lambda value: value + 3,
            ),
        ],
    )
    prediction_pipeline = TransformPipeline(
        input_key="x",
        output_key="x",
        steps=[
            TransformStep(
                name="add_ten",
                transform=lambda value: value + 10,
            ),
        ],
    )

    datamodule = BenchRepDataModule(
        train_dataset=source_dataset,
        predict_dataset=source_dataset,
        training_pipelines=(training_pipeline,),
        validation_pipelines=(validation_pipeline,),
        prediction_pipelines=(prediction_pipeline,),
        batch_size=2,
        val_fraction=0.25,
        num_workers=0,
        seed=137,
    )
    datamodule.setup("fit")

    assert isinstance(datamodule.train_dataset, TransformedDataset)
    assert isinstance(datamodule.val_dataset, TransformedDataset)
    assert isinstance(datamodule.predict_dataset, TransformedDataset)

    raw_train_sample = datamodule.train_dataset.dataset[0]
    transformed_train_sample = datamodule.train_dataset[0]

    raw_val_sample = datamodule.val_dataset.dataset[0]
    transformed_val_sample = datamodule.val_dataset[0]

    raw_prediction_sample = datamodule.predict_dataset.dataset[0]
    transformed_prediction_sample = datamodule.predict_dataset[0]

    assert torch.equal(
        transformed_train_sample["x"],
        (raw_train_sample["x"] + 1) * 2 + 3,
    )
    assert torch.equal(
        transformed_val_sample["x"],
        raw_val_sample["x"] + 1 + 3,
    )
    assert torch.equal(
        transformed_prediction_sample["x"],
        raw_prediction_sample["x"] + 10,
    )

def test_transformed_dataset_clones_for_different_output_key() -> None:
    source_dataset = TinySyntheticDataset(n_samples=4)
    source_x = source_dataset[0]["x"].clone()

    pipeline = TransformPipeline(
        input_key="x",
        output_key="positive_x",
        steps=[
            TransformStep(
                name="mutating_add_five",
                transform=lambda value: value.add_(5),
            )
        ],
    )

    dataset = TransformedDataset(
        dataset=source_dataset,
        pipelines=(pipeline,),
    )

    transformed_sample = dataset[0]

    assert torch.equal(transformed_sample["x"], source_x)
    assert torch.equal(
        transformed_sample["positive_x"],
        source_x + 5,
    )


def test_materialized_branch_survives_later_in_place_input_mutation() -> None:
    source_dataset = TinySyntheticDataset(n_samples=4)
    source_x = source_dataset[0]["x"].clone()

    dataset = TransformedDataset(
        dataset=source_dataset,
        pipelines=(
            TransformPipeline(
                input_key="x",
                output_key="positive_x",
                steps=[
                    TransformStep(
                        name="mutating_add_one",
                        transform=lambda value: value.add_(1),
                    ),
                ],
            ),
            TransformPipeline(
                input_key="x",
                output_key="x",
                steps=[
                    TransformStep(
                        name="mutating_multiply_by_two",
                        transform=lambda value: value.mul_(2),
                    ),
                ],
            ),
        ),
    )

    transformed_sample = dataset[0]

    assert torch.equal(
        transformed_sample["positive_x"],
        source_x + 1,
    )
    assert torch.equal(
        transformed_sample["x"],
        source_x * 2,
    )
    assert (
        transformed_sample["positive_x"].data_ptr()
        != transformed_sample["x"].data_ptr()
    )


def test_stochastic_views_are_independent_and_worker_seed_reproducible() -> None:
    first = _load_stochastic_views(seed=137)
    replay = _load_stochastic_views(seed=137)
    different_seed = _load_stochastic_views(seed=138)

    assert not torch.equal(first["positive_x"], first["x"])
    assert torch.equal(replay["positive_x"], first["positive_x"])
    assert torch.equal(replay["x"], first["x"])
    assert not torch.equal(
        different_seed["positive_x"],
        first["positive_x"],
    )
    assert not torch.equal(different_seed["x"], first["x"])


def test_transformed_dataset_applies_pipelines_in_order() -> None:
    source_dataset = TinySyntheticDataset(n_samples=4)
    source_x = source_dataset[0]["x"].clone()

    dataset = TransformedDataset(
        dataset=source_dataset,
        pipelines=(
            TransformPipeline(
                input_key="x",
                output_key="positive_x",
                steps=[
                    TransformStep(
                        name="add_one",
                        transform=lambda value: value + 1,
                    )
                ],
            ),
            TransformPipeline(
                input_key="positive_x",
                output_key="positive_x",
                steps=[
                    TransformStep(
                        name="multiply_by_two",
                        transform=lambda value: value * 2,
                    )
                ],
            ),
        ),
    )

    transformed_sample = dataset[0]

    assert torch.equal(transformed_sample["x"], source_x)
    assert torch.equal(
        transformed_sample["positive_x"],
        (source_x + 1) * 2,
    )


def _load_stochastic_views(*, seed: int) -> dict[str, torch.Tensor]:
    dataset = TransformedDataset(
        dataset=TinySyntheticDataset(n_samples=8),
        pipelines=(
            TransformPipeline(
                input_key="x",
                output_key="positive_x",
                steps=[
                    TransformStep(
                        name="positive_random_noise",
                        transform=_AddRandomNoise(),
                    ),
                ],
            ),
            TransformPipeline(
                input_key="x",
                output_key="x",
                steps=[
                    TransformStep(
                        name="anchor_random_noise",
                        transform=_AddRandomNoise(),
                    ),
                ],
            ),
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=2,
        generator=torch.Generator().manual_seed(seed),
    )

    return next(iter(loader))
