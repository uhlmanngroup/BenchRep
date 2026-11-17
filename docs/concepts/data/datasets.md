# Datasets and sample structure

A dataset connects a data source to BenchRep by returning one sample at a time. BenchRep uses dictionaries so that images, additional model inputs, labels, and identifiers can travel together without imposing one fixed tuple layout on every experiment.

## Select a dataset

The `dataset` section selects a registered implementation and supplies its constructor parameters:

```yaml
dataset:
  name: mnist
  params:
    root: data/mnist
    split: train
    download: true
```

Here, `split: train` selects the dataset's own training set. Reserving part of that set for validation is a separate [datamodule operation](datamodules.md#dataset-splits-and-validation-subsets).

BenchRep includes adapters for [MNIST][benchrep.architecture.data.datasets.MNISTDataset], [CIFAR-10][benchrep.architecture.data.datasets.CIFAR10Dataset], and [STL-10][benchrep.architecture.data.datasets.STL10Dataset]. Their samples contain `x`, `label`, and `sample_id`. Images are converted to tensors while retaining their original unsigned 8-bit values; conversion to floating point, scaling, and normalization belong in [transform pipelines](transforms.md).

Built-in dataset parameters have dedicated schemas. Other registered datasets use [`CustomDatasetConfig`][benchrep.assembly.schemas.training_config_schema.CustomDatasetConfig], which passes its `params` mapping to the registered constructor or factory. The config shape remains `name` plus `params`; the factory must return a [`BaseDataset`][benchrep.architecture.data.datasets.BaseDataset] instance.

See the [dataset catalogue](../../reference/components/datasets.md) for built-ins and [Registering components](../extensibility/registration.md) for external datasets. [`inspect_registry("dataset")`][benchrep.inspect_registry] lists available registrations.

## A sample becomes a batch

A canonical image sample might contain:

```python
sample = {
    "x": image_tensor,
    "label": 7,
    "sample_id": "sample-0042",
    "metadata": {"patient": "patient-A"},
}
```

The dataset returns an individual sample, without a batch dimension. With the default dataloader collation, image tensors of shape `(C, H, W)` become a tensor of shape `(B, C, H, W)`. Numeric labels become a batch of labels, strings are collected into lists, and nested dictionaries are collated by key.

Samples therefore need compatible fields and values for batching. Variable image sizes, missing fields, or arbitrary Python objects may require preprocessing or a custom collation function. A valid sample dictionary alone does not guarantee a valid batch.

## Fields have different roles

| Consumer | Required input | Annotations and other fields |
| --- | --- | --- |
| Canonical AE/VAE | A tensor under `x`, used as the input and reconstruction target | `label`, `sample_id`, and a `metadata` mapping are supported by the standard prediction contract; custom objectives can access additional batch fields |
| Composite model | Tensor fields named under `composite_model_declarations.expects` | Fields declared under `batch_metadata` are also required; their roles identify observation annotations for prediction |

For canonical models, providing `label` does not automatically add a supervised objective. For composite models, a target used by a loss belongs among the declared inputs, with the appropriate role and loss wiring. Metadata declarations describe annotations rather than component inputs. See [Composite models](../models/composite.md) and [Losses and objectives](../models/losses.md).

Composite input names describe the sample **after transforms**. A pipeline may create a required image field from another field, so the underlying dataset need not supply every derived view itself. All declared inputs and metadata must be present by the time the model receives the batch.

Only fields carried into the prediction output are available for export. Adding an arbitrary field to the dataset does not automatically make it an AnnData observation column; see [Selecting prediction exports](../prediction/exports.md).

!!! note "Sample identity"
    Built-in datasets use the sample's index within the selected dataset split as `sample_id`. These identifiers are not globally unique across datasets or splits. Custom datasets can provide identifiers that remain meaningful across your own runs and exports.

## Where validation happens

[`BaseDataset`][benchrep.architecture.data.datasets.BaseDataset] defines [`__len__()`][benchrep.architecture.data.datasets.BaseDataset.__len__] and [`__getitem__()`][benchrep.architecture.data.datasets.BaseDataset.__getitem__]; its [`validate_sample()`][benchrep.architecture.data.datasets.BaseDataset.validate_sample] helper checks that a sample is a dictionary. More specific checks belong to the consumers: routed transforms require tensor inputs and outputs, canonical models require tensor-valued `x`, and composite execution checks declared fields and component tensor contracts.

Configuration validation cannot inspect every real sample. A schema-valid dataset configuration can still fail when data is read, collated, or passed to a model. The [extension reference](../../reference/interfaces.md#datasets-and-transforms) documents the dataset interface; [Architecture component contracts](../models/component-contracts.md) explains model-side expectations.
