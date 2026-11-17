# Splits and dataloaders

A dataset supplies samples; a datamodule decides how they are divided, transformed, and loaded for a workflow. BenchRep's config-built path uses [`BenchRepDataModule`][benchrep.architecture.data.datamodule.BenchRepDataModule], a Lightning datamodule that prepares training and validation subsets and exposes the corresponding dataloaders.

## Dataset splits and validation subsets

The dataset's named split and BenchRep's validation partition are separate choices:

```yaml
dataset:
  name: mnist
  params:
    root: data/mnist
    split: train
    download: true

datamodule:
  batch_size: 128
  val_fraction: 0.1
  num_workers: 0
```

This selects MNIST's training set, then reserves 10% of that set for validation. It does not use MNIST's test set for validation.

During [`setup()`][benchrep.architecture.data.datamodule.BenchRepDataModule.setup], [`BenchRepDataModule`][benchrep.architecture.data.datamodule.BenchRepDataModule] creates random, disjoint index subsets using the workflow's `reproducibility.seed`. It then wraps each subset with its corresponding transform pipelines. The split is prepared once per datamodule instance; repeated [`setup()`][benchrep.architecture.data.datamodule.BenchRepDataModule.setup] calls do not repartition it.

The default partitioning is not stratified by class or grouped by patient, slide, or acquisition. For predefined, stratified, or grouped partitions, construct the datasets yourself and supply a datamodule override.

??? note "Validation size and reproducibility"
    The validation size is `int(len(dataset) * val_fraction)`; the remainder is used for training. A positive fraction that produces 0 validation samples raises an error. Setting `val_fraction: 0` keeps the entire dataset for training without a validation subset.

    Reproducing the subset requires the same dataset length and sample ordering as well as the same seed. The split seed controls partitioning; it does not by itself guarantee identical random transforms or training results. See [Reproducibility](../results/reproducibility.md).

## Dataloader behavior

The built-in datamodule uses the following loading policy:

| Loader | Shuffle samples | Drop the final incomplete batch | Transform sequence |
| --- | --- | --- | --- |
| Training | Yes | Controlled by `drop_last` | Training |
| Validation | No | No | Validation |
| Test | No | No | Validation |
| Prediction | No | No | Prediction; falls back to validation pipelines when constructed directly without explicit prediction pipelines |

Test and explicit validation datasets are available through the Python datamodule interface. The standard training config builds 1 dataset and optionally partitions it; it does not automatically construct a separate test dataset or run a test stage.

Training's `datamodule` section controls batch size, worker processes, pinned memory, and prefetching. `num_workers: 0` loads data in the main process. `persistent_workers: true` and an explicit `prefetch_factor` require workers greater than 0. Configured `pin_memory: auto` enables pinned CPU memory when CUDA is available.

`dataloader_kwargs` forwards additional arguments accepted by the installed PyTorch `DataLoader`. BenchRep reserves the arguments it controls, including `dataset`, `batch_size`, `shuffle`, `sampler`, and `batch_sampler`. Use a datamodule override for a custom sampling policy. The complete settings and reserved arguments are documented in the [training schema reference](../../reference/schemas/training.md).

## Prediction data

Prediction constructs a dataset independently of the training datamodule. An explicit prediction `dataset` selects what to read; if omitted or null, its configuration is inherited from training when training used a config-built datamodule.

**Inheriting the dataset config does not recover the validation subset.** If training selected MNIST's `train` split and held out 10%, prediction with an inherited dataset config reads the full MNIST training set. To predict on the test set, explicitly select `split: test`. To reuse the actual validation subset in the same process, supply [`training_result.datamodule.val_dataset`][benchrep.architecture.data.datamodule.BenchRepDataModule] through a prediction datamodule override, as in [Tune with Optuna](../../guides/optuna.md).

Prediction's `data.batch_size` and `data.num_workers` replace the corresponding training settings when supplied; otherwise they inherit them where available. Other loader settings come from the recorded training datamodule config, or defaults if training used an external datamodule. Prediction disables validation splitting and incomplete-batch dropping. With 0 workers, it also disables persistent workers and clears the prefetch factor.

`data.max_batches` limits the number of prediction batches processed. It remains active with a datamodule override. Transform inheritance is handled separately, as described in [Transform pipelines](transforms.md#prediction-inherits-validation-processing).

## Supply your own partitions or loaders

You can retain BenchRep's loading behavior while providing explicit training and validation datasets:

```python
from benchrep import TrainingOverrides
from benchrep.architecture.data import BenchRepDataModule

datamodule = BenchRepDataModule(
    train_dataset=train_dataset,
    val_dataset=validation_dataset,
    val_fraction=0.0,
    batch_size=128,
    num_workers=0,
)
training_overrides = TrainingOverrides(datamodule=datamodule)
```

Here, `train_dataset` and `validation_dataset` are datasets you have already constructed. Pass `training_overrides` to the training entrypoint's `overrides` argument. Explicit validation data and automatic validation splitting are mutually exclusive, hence `val_fraction=0.0`.

The override takes responsibility for preprocessing too: the workflow's `dataset`, `transform_pipelines`, and `datamodule` sections are bypassed. Supply pipeline objects to the datamodule or use datasets that already perform the desired transforms. Avoid wrapping an already transformed validation subset with the same transforms again.

For more specialized loading, supply another `LightningDataModule` subclass or instance. It must provide the loaders required by the workflow and batches compatible with the model; it need not inherit [`BenchRepDataModule`][benchrep.architecture.data.datamodule.BenchRepDataModule]. See [Python overrides](../workflows/overrides.md#supply-an-implementation), [External models and datamodules](../extensibility/runtime-implementations.md), and the [datamodule reference](../../reference/interfaces.md#datamodules).
