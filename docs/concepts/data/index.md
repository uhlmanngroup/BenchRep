# Data

BenchRep separates reading samples, transforming their contents, and organizing them into batches. This lets you reuse a dataset across models while changing preprocessing, augmentation, or loading behavior independently.

| Layer | Responsibility | Config location |
| --- | --- | --- |
| Dataset | Read one sample as a dictionary of tensors and annotations | `dataset` |
| Transform pipelines | Apply ordered transforms to selected sample fields | `transform_pipelines` |
| Datamodule | Prepare subsets and provide dataloaders | Training: `datamodule`; prediction: `data` and inherited training settings |

The consuming model determines which fields a batch must contain. Canonical models expect `x`; composite models use declared input names and roles. A [datamodule override](../workflows/overrides.md#supply-an-implementation) can replace this data preparation and loading machinery while still supplying batches that satisfy the model's contract.

The sequence below shows how these layers turn your configuration into batches for training.

!!! abstract "From configuration to batches"
    For training without a datamodule runtime override:

    ### I. You configure the inputs

    1. Select a registered dataset by name and supply its parameters. For your own dataset, first register a [`BaseDataset`][benchrep.architecture.data.datasets.BaseDataset] subclass.
    2. Configure transform pipelines and datamodule settings.

    ### II. BenchRep prepares the data

    1. Instantiates the dataset and builds [`TransformPipeline`][benchrep.architecture.data.transforms.TransformPipeline] objects from [`TransformStep`][benchrep.architecture.data.transforms.TransformStep] objects.
    2. [`BenchRepDataModule`][benchrep.architecture.data.datamodule.BenchRepDataModule] prepares the training and optional validation subsets.
    3. Where transforms are configured, it wraps each subset in [`TransformedDataset`][benchrep.architecture.data.datasets.TransformedDataset] with the corresponding pipeline sequence.
    4. The datamodule creates the dataloaders.

    ### III. During loading

    1. A dataloader requests samples.
    2. If wrapped, [`TransformedDataset`][benchrep.architecture.data.datasets.TransformedDataset] retrieves each sample dictionary, applies its pipelines, and returns the transformed sample. Otherwise, samples are read directly.
    3. The dataloader collates samples into a batch for the model.

Explore each layer in more detail:

- **[Datasets and sample structure](datasets.md):** dataset selection, sample fields, batching, and the distinction between model inputs and metadata.
- **[Transform pipelines](transforms.md):** field routing, execution order, split-specific steps, and prediction inheritance.
- **[Splits and dataloaders](datamodules.md):** validation partitioning, loading settings, and when to supply a datamodule yourself.

These pages describe data supplied to training and prediction. Evaluation consumes exported or external artifacts; see [Inputs and evaluation pipelines](../evaluation/inputs.md).