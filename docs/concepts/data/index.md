# Data

BenchRep separates reading samples, transforming their contents, and organizing them into batches. This lets you reuse a dataset across models while changing preprocessing, augmentation, or loading behavior independently.

| Layer | Responsibility | Config location |
| --- | --- | --- |
| Dataset | Read one sample as a dictionary of tensors and annotations | `dataset` |
| Transform pipelines | Apply ordered transforms to selected sample fields | `transform_pipelines` |
| Datamodule | Prepare subsets and provide dataloaders | Training: `datamodule`; prediction: `data` and inherited training settings |

The consuming model determines which fields a batch must contain. Canonical models expect `x`; composite models use declared input names and roles. A datamodule override can replace the config-built data path while still supplying batches that satisfy the model's contract.

- **[Datasets and sample structure](datasets.md):** dataset selection, sample fields, batching, and the distinction between model inputs and metadata.
- **[Transform pipelines](transforms.md):** field routing, execution order, split-specific steps, and prediction inheritance.
- **[Splits and dataloaders](datamodules.md):** validation partitioning, loading settings, and when to supply a datamodule yourself.

These pages describe data supplied to training and prediction. Evaluation consumes exported or external artifacts; see [Inputs and evaluation pipelines](../evaluation/inputs.md).
