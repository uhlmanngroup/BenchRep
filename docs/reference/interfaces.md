# Extension interfaces

These contracts define what an external component must accept or return. The [registry reference](registries.md) identifies which entry type each registry accepts; the [component catalogue](components.md) lists built-ins and configuration locations.

| Extension | Registration or runtime interface |
| --- | --- |
| Dataset | Constructor or factory returning `BaseDataset` |
| Transform | Constructor or factory returning a tensor-transform callable |
| Encoder, decoder, or head | `ArchitectureComponent`; canonical encoders and decoders also implement their base interfaces |
| Loss | `LossComponent`; custom objectives additionally inherit `BaseCustomObjectiveLoss` |
| Loss-weight schedule | Constructor returning an object satisfying `LossWeightSchedule` |
| Evaluation metric | `EvaluationMetric` |
| Whole canonical model | Runtime class or instance inheriting the matching model base |
| Datamodule | Runtime `LightningDataModule` class or instance; `BenchRepDataModule` is the supplied implementation |

Optimizers, learning-rate schedulers, loggers, and callbacks follow the corresponding PyTorch or Lightning interfaces. Their registry-specific calling conventions appear in the [catalogue](components.md).

## Datasets and transforms

Registered dataset factories must return a `BaseDataset`. Samples are dictionaries; required fields depend on the consuming model. A transform registration supplies a constructor or factory, whose result operates on a routed sample tensor. `TransformStep` and `TransformPipeline` are useful when assembling datamodules directly in Python.

::: benchrep.architecture.data.datasets.BaseDataset
    options:
      heading_level: 3
      members:
        - __len__
        - __getitem__
        - validate_sample

::: benchrep.architecture.data.transforms.TransformStep
    options:
      heading_level: 3
      members:
        - run

::: benchrep.architecture.data.transforms.TransformPipeline
    options:
      heading_level: 3
      members:
        - run

::: benchrep.architecture.data.datasets.TransformedDataset
    options:
      heading_level: 3

## Datamodules

A datamodule override supplies the data-loading implementation for training or prediction. It need not inherit `BenchRepDataModule`, but it must inherit `LightningDataModule` and provide the loaders required by that workflow. Constructor parameters for a class override belong in the config; an instance has already been constructed. See [Overrides and config helpers](configuration.md).

::: benchrep.architecture.data.datamodule.BenchRepDataModule
    options:
      heading_level: 3
      members:
        - setup
        - train_dataloader
        - val_dataloader
        - test_dataloader
        - predict_dataloader

## Architecture components

`ArchitectureComponent` separates constructor parameters from the tensors passed to `forward()`. Input ports name forward arguments; result contracts describe either one tensor or a mapping of named tensors. The declared structures describe each sample, excluding the batch dimension.

::: benchrep.architecture.composite_model_component_contracts.ArchitectureComponent
    options:
      heading_level: 3

::: benchrep.architecture.composite_model_component_contracts.ComponentPort
    options:
      heading_level: 3

::: benchrep.architecture.composite_model_component_contracts.ComponentTensorResult
    options:
      heading_level: 3

::: benchrep.architecture.composite_model_component_contracts.ComponentMappingResult
    options:
      heading_level: 3

### Canonical encoders and decoders

Components used by canonical models also expose the dimensions needed to connect the encoder, variational head, and decoder. Composite components can instead be ordinary `torch.nn.Module` classes with declared runtime contracts.

::: benchrep.architecture.encoders.base.BaseEncoder
    options:
      heading_level: 4
      members:
        - input_shape
        - output_dim
        - feature_shape
        - forward

::: benchrep.architecture.decoders.base.BaseDecoder
    options:
      heading_level: 4
      members:
        - input_dim
        - output_shape
        - forward

## Losses and weight schedules

Ordinary composite losses declare semantic roles for each tensor argument. Canonical reconstruction and regularization losses can use their registry's fixed calling convention without declaring composite ports. Custom objectives use a separate fixed interface that receives complete batch and model-output mappings.

::: benchrep.architecture.losses.composite_model_contracts.LossComponent
    options:
      heading_level: 3

::: benchrep.architecture.losses.composite_model_contracts.LossTensorPort
    options:
      heading_level: 3

::: benchrep.architecture.losses.custom_objective.BaseCustomObjectiveLoss
    options:
      heading_level: 3
      members:
        - forward

Register a custom objective as `LossComponent(MyObjective)` in `CUSTOM_OBJECTIVE_LOSSES`, with no `runtime_inputs`. The complete context consists of the batch and output mappings; it is not the model instance.

::: benchrep.architecture.losses.weight_schedules.LossWeightSchedule
    options:
      heading_level: 3
      members:
        - factor

## Evaluation metrics

Register the wrapper, not its bare `fn`. The selected registry determines the metric's input arguments; the wrapper declares its output shape. A vector axis must be supported by the consuming evaluator, not merely be an arbitrary label.

::: benchrep.evaluation.metrics.contracts.EvaluationMetric
    options:
      heading_level: 3

## Whole-model overrides

Whole-model overrides are supported by the canonical autoencoder and VAE training and prediction entrypoints. Composite models are assembled from configuration. The base classes below establish the prediction contract; your model must also implement the Lightning behavior needed by the requested workflow.

::: benchrep.interfaces.models.BenchRepAutoencoderModel
    options:
      heading_level: 3
      members:
        - predict_step

::: benchrep.interfaces.models.BenchRepVAEModel
    options:
      heading_level: 3
      members:
        - predict_step

## Batch and output contracts

The canonical batch and forward contracts are dictionaries. Prediction contracts are dataclasses, with tensors and observation annotations aligned by sample. `PredictionObsValue` accepts a tensor, a list of integers, or a list of strings; `PredictionMetadata` maps annotation names to those values.

::: benchrep.interfaces.contracts.AutoencoderBatch
    options:
      heading_level: 3

::: benchrep.interfaces.contracts.AutoencoderForwardOutput
    options:
      heading_level: 3

::: benchrep.interfaces.contracts.VAEForwardOutput
    options:
      heading_level: 3

::: benchrep.interfaces.contracts.AutoencoderPredictionOutput
    options:
      heading_level: 3

::: benchrep.interfaces.contracts.VAEPredictionOutput
    options:
      heading_level: 3

::: benchrep.interfaces.contracts.CompositePredictionOutput
    options:
      heading_level: 3

## Model families

Import these constants from `benchrep.interfaces.model_families`, for example when launching a wizard. They select existing family contracts; constructing a family spec does not register a new model family.

| Constant | Name | Whole-model override base |
| --- | --- | --- |
| `AUTOENCODER_FAMILY` | `autoencoder` | `BenchRepAutoencoderModel` |
| `VAE_FAMILY` | `vae` | `BenchRepVAEModel` |
| `COMPOSITE_FAMILY` | `composite` | Not supported |

::: benchrep.interfaces.model_families.ModelFamilySpec
    options:
      heading_level: 3

::: benchrep.interfaces.model_families.CanonicalModelFamilySpec
    options:
      heading_level: 3
