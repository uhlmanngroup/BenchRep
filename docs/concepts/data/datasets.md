# Datasets and sample structure

BenchRep datasets follow the standard PyTorch `Dataset` interface with `__len__()` and `__getitem__()` methods. [`BaseDataset`][benchrep.architecture.data.datasets.BaseDataset] subclasses PyTorch’s `Dataset` and requires dictionary samples, with a static [`validate_sample()`][benchrep.architecture.data.datasets.BaseDataset.validate_sample] helper to check this format.

To use your own data through configuration, implement or adapt your dataset as a `BaseDataset` subclass, register it under a name of your choice, and select that name in your config. The built-in datasets use the same mechanism. See [Register external components](../../guides/register-components.md#adapt-an-external-dataset) for an example.

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

The dataset returns an individual sample, without a batch dimension. PyTorch’s default dataloader collation combines samples field by field: image tensors of shape `(C, H, W)` become a tensor of shape `(B, C, H, W)`, numeric labels become a tensor, and strings become a list. Dictionary keys are preserved, and nested dictionaries follow the same rules.

Samples therefore need compatible fields and values for batching. Variable image sizes, missing fields, or arbitrary Python objects may require preprocessing or a custom collation function. A valid sample dictionary alone does not guarantee a valid batch.

## Fields have different roles

| Consumer | Required input | Annotations and other fields |
| --- | --- | --- |
| Canonical AE/VAE | A tensor under `x`, used as the input and reconstruction target | `label`, `sample_id`, and a `metadata` mapping are supported by the standard prediction contract; custom objectives can access additional batch fields |
| Composite model | Tensor fields named under `composite_model_declarations.expects` |Optional `batch_metadata` declarations identify observation annotations for prediction. Any fields declared there must be present in each batch. |

For composite models, distinguish values used for computation from annotations. For example, a class label used by a classification loss must be declared under `expects` with the appropriate target role and connected to that loss. A patient identifier retained for grouping or annotating predictions belongs under `batch_metadata`. See [Composite models](../models/composite.md) and [Losses and objectives](../models/losses.md).

!!! note "Roles also guide behavior"
    Roles describe what a field represents, but can also affect validation, defaults, and exports. Some distinctions are descriptive for a particular operation; others change its behavior. For example, a unique `sample_image` input becomes the default transform input, while metadata with the `index` role supplies the AnnData observation index. See [Composite models](../models/composite.md) for the individual roles and their effects.

The model receives samples **after transformation**. Your dataset therefore does not need to supply fields that a transform pipeline creates, such as a second augmented image derived from the original. By the time samples reach the model, however, all fields declared under `expects` and `batch_metadata` must be present.

Prediction exports contain the fields supported by the model's prediction interface. For example, canonical models support `label`, `sample_id`, and `metadata` annotations. Adding another field to your dataset does not automatically include it in the exported AnnData; see [Selecting prediction exports](../prediction/exports.md).

!!! note "Sample identity"
    Built-in datasets number samples within each split. For example, MNIST's training and test sets both contain a sample with `sample_id=0`, but these are different images. When combining outputs, keep track of which dataset and split they came from. For your own dataset, you can use identifiers such as filenames or cell IDs instead.

## Where validation happens

[`BaseDataset`][benchrep.architecture.data.datasets.BaseDataset] defines [`__len__()`][benchrep.architecture.data.datasets.BaseDataset.__len__] and [`__getitem__()`][benchrep.architecture.data.datasets.BaseDataset.__getitem__]; its [`validate_sample()`][benchrep.architecture.data.datasets.BaseDataset.validate_sample] helper checks that a sample is a dictionary. More specific checks belong to the consumers: routed transforms require tensor inputs and outputs, canonical models require tensor-valued `x`, and composite execution checks declared fields and component tensor contracts.

Configuration validation cannot inspect every real sample. A schema-valid dataset configuration can still fail when data is read, collated, or passed to a model. The [extension reference](../../reference/interfaces.md#datasets-and-transforms) documents the dataset interface; [Architecture component contracts](../models/component-contracts.md) explains model-side expectations.
