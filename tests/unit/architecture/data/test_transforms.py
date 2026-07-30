import torch

from benchrep.architecture.data import (
    TransformPipeline,
    TransformStep,
    TransformedDataset,
    DataModule,
)
from tests.fixtures.datasets import TinySyntheticDataset


def test_transform_pipeline_applies_steps_in_order() -> None:
    pipeline = TransformPipeline(
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
    pipeline = TransformPipeline()

    result = pipeline(value)

    assert result is value


def test_transformed_dataset_replaces_only_x() -> None:
    source_dataset = TinySyntheticDataset(n_samples=4)
    source_sample = source_dataset[0]
    source_x = source_sample["x"].clone()

    pipeline = TransformPipeline(
        steps=[
            TransformStep(
                name="add_five",
                transform=lambda value: value + 5,
            )
        ]
    )
    dataset = TransformedDataset(
        dataset=source_dataset,
        pipeline=pipeline,
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


def test_datamodule_applies_split_specific_pipelines() -> None:
    source_dataset = TinySyntheticDataset(n_samples=8)

    preprocessing_pipeline = TransformPipeline(
        steps=[
            TransformStep(
                name="add_one",
                transform=lambda value: value + 1,
            ),
            TransformStep(
                name="add_three",
                transform=lambda value: value + 3,
            ),
        ]
    )
    training_pipeline = TransformPipeline(
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
        ]
    )

    datamodule = DataModule(
        train_dataset=source_dataset,
        training_pipeline=training_pipeline,
        preprocessing_pipeline=preprocessing_pipeline,
        batch_size=2,
        val_fraction=0.25,
        num_workers=0,
        seed=137,
    )
    datamodule.setup("fit")

    assert isinstance(datamodule.train_dataset, TransformedDataset)
    assert isinstance(datamodule.val_dataset, TransformedDataset)

    raw_train_sample = datamodule.train_dataset.dataset[0]
    transformed_train_sample = datamodule.train_dataset[0]

    raw_val_sample = datamodule.val_dataset.dataset[0]
    transformed_val_sample = datamodule.val_dataset[0]

    assert torch.equal(
        transformed_train_sample["x"],
        (raw_train_sample["x"] + 1) * 2 + 3,
    )
    assert torch.equal(
        transformed_val_sample["x"],
        raw_val_sample["x"] + 1 + 3,
    )