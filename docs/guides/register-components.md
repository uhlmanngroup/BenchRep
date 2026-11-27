# Register external components

<strong class="highlight">How can I bring my own components into BenchRep?</strong> Register an implementation under a name, then select it in a config alongside built-in components.

This tutorial walks through three separate examples. We adapt Fashion-MNIST as a dataset, add two custom losses to a VAE on MNIST, and fine-tune a pretrained ConvNeXt encoder in an autoencoder on STL-10. The snippets show how each component is registered and selected in the config. Each example links to its runnable script and configs.

<strong class="highlight">Requirements.</strong> The dataset and loss examples use the `scverse` extra for UMAP. The pretrained encoder example needs only the core installation. If you need to add `scverse`, run the following command.

```bash
python -m pip install "benchrep[scverse] @ git+https://github.com/uhlmanngroup/BenchRep.git"
```

## Choose the registration interface

| Component | Registered entry | Selected through |
| --- | --- | --- |
| Dataset, transform, optimizer, LR scheduler, logger, callback, or loss-weight schedule | Constructor or factory | The corresponding named config section or list entry |
| Encoder, decoder, or head | [`ArchitectureComponent`][benchrep.architecture.composite_model_component_contracts.ArchitectureComponent] containing the module class and its input and result contracts | `encoder` or `decoder` for canonical models, or `composite_model_components` for encoders, decoders, and heads in composite models |
| Loss | [`LossComponent`][benchrep.architecture.losses.composite_model_contracts.LossComponent] containing the loss class and any required input contract | A named term within a loss role, including [`custom_objective`](#add-a-full-context-objective) |
| Evaluation metric | [`EvaluationMetric`][benchrep.evaluation.metrics.contracts.EvaluationMetric] containing the metric function and its result contract | A metric group's `selected` list, with per-metric `params` |

See [Registering components](../concepts/extensibility/registration.md) for the full registry map and which registries accept external implementations.

!!! warning "Register before running"
    Register your implementation before resolving a config that refers to it. Registrations apply to the current Python process, so repeat the registration if you reconstruct the model for prediction in a separate process.

## Adapt an external dataset

BenchRep datasets return each sample as a dictionary. The [canonical VAE](../concepts/models/canonical.md#variational-autoencoder) expects the image tensor under `x` and uses it as the reconstruction target. Prediction can also export `label` and `sample_id` as annotations. See [Datasets and sample structure](../concepts/data/datasets.md#fields-have-different-roles) for the fields each model family expects.

The dataset example subclasses [`BaseDataset`][benchrep.architecture.data.datasets.BaseDataset] to adapt [torchvision’s Fashion-MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html). It puts each image under `x`, its class under `label`, and its index within the split under `sample_id`.

[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/04_custom_dataset_pipeline.py) · [View configs](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/configs/04_custom_dataset_pipeline)

```python
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

from benchrep.architecture.data import BaseDataset


class FashionMNISTDataset(BaseDataset):
    def __init__(self, root, split="train", download=False):
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        self.dataset = FashionMNIST(
            root=root,
            train=split == "train",
            transform=v2.ToImage(),
            download=download,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        sample = {"x": image, "label": label, "sample_id": index}
        return self.validate_sample(sample)
```

`ToImage()` converts the image to a tensor and preserves its pixel values. The configured [transform pipeline](../concepts/data/transforms.md) converts those values to floating point in `[0, 1]`.

Register the dataset class as `fashion_mnist`, with `fashionmnist` as an optional alias.

```python
from benchrep.assembly.registries import DATASETS

DATASETS.register("fashion_mnist", FashionMNISTDataset, "fashionmnist")
```

Set `dataset.name` to the registered name and pass the constructor arguments in `dataset.params`.

```yaml
dataset:
  name: fashion_mnist
  params:
    root: data/fashion_mnist
    split: train
    download: true
```

Run the full training, prediction, and evaluation example.

```bash
python -m benchrep.examples.usage.04_custom_dataset_pipeline
```

## Add custom losses

The loss example trains a canonical VAE on MNIST. It adds a reconstruction loss that compares neighboring-pixel differences and a custom objective that uses digit labels to supervise the latent representation.

[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/05_custom_loss_pipeline.py) · [View configs](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/configs/05_custom_loss_pipeline)

### Extend a loss role

A canonical reconstruction loss is a `torch.nn.Module` whose `forward(reconstruction, target)` returns a scalar tensor. Here, `target` is the input image.

The training config uses a built-in pixelwise MSE and an external `GradientDifferenceLoss` as separate reconstruction terms. The snippet below implements `GradientDifferenceLoss`, which penalizes differences in horizontal and vertical image gradients.

```python
import torch


class GradientDifferenceLoss(torch.nn.Module):
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_dx = (
            reconstruction[..., 1:] - reconstruction[..., :-1]
        ).abs()
        target_dx = (target[..., 1:] - target[..., :-1]).abs()

        reconstruction_dy = (
            reconstruction[..., 1:, :] - reconstruction[..., :-1, :]
        ).abs()
        target_dy = (target[..., 1:, :] - target[..., :-1, :]).abs()

        horizontal_loss = (reconstruction_dx - target_dx).abs().mean()
        vertical_loss = (reconstruction_dy - target_dy).abs().mean()
        return horizontal_loss + vertical_loss
```

Register the `LossComponent` in `RECONSTRUCTION_LOSSES`.

```python
from benchrep.architecture.losses import LossComponent
from benchrep.assembly.registries import RECONSTRUCTION_LOSSES

RECONSTRUCTION_LOSSES.register(
    "gradient_difference",
    LossComponent(GradientDifferenceLoss),
    "gdl",
)
```

Here, the canonical VAE supplies both forward arguments automatically. To use this loss in a composite model, you would have to declare its input ports in `LossComponent.runtime_inputs` and connect them to model fields in `composite_wiring`. See [Wire composite losses](../concepts/models/losses.md#wire-composite-losses).

### Add a full-context objective

Use the `custom_objective` role when a loss falls outside the standard roles your model family supports. For a canonical VAE, those roles are `reconstruction` and `regularization`. A custom objective receives the full `batch` and `model_output` mappings in `forward()`. See [Custom objectives](../concepts/models/losses.md#custom-objectives).

Here we subclass [`BaseCustomObjectiveLoss`][benchrep.architecture.losses.custom_objective.BaseCustomObjectiveLoss] to predict digit labels from the VAE’s latent embeddings.

```python
from collections.abc import Mapping
from typing import Any

import torch

from benchrep.architecture.losses import BaseCustomObjectiveLoss


class AuxiliaryLatentClassificationLoss(BaseCustomObjectiveLoss):
    def __init__(self, embedding_dim: int, n_classes: int) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(embedding_dim, n_classes)

    def forward(
        self,
        *,
        batch: Mapping[str, Any],
        model_output: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        labels = batch["label"]
        embeddings = model_output["embedding"]
        logits = self.classifier(embeddings)
        return torch.nn.functional.cross_entropy(logits, labels.long())
```

The classifier predicts digit labels from the VAE's latent means, exposed as `model_output["embedding"]`. Its gradients also update the encoder and the mean projection of the variational head. BenchRep attaches the loss module to the VAE, so its classifier parameters are optimized with the rest of the model and saved in checkpoints.

Register the loss in `CUSTOM_OBJECTIVE_LOSSES`, the registry for the `custom_objective` role. This registry has no built-in entries. BenchRep supplies both mappings automatically in canonical and composite models, so `LossComponent` only needs the loss class.

```python
from benchrep.architecture.losses import LossComponent
from benchrep.assembly.registries import CUSTOM_OBJECTIVE_LOSSES

CUSTOM_OBJECTIVE_LOSSES.register(
    "auxiliary_latent_classification",
    LossComponent(AuxiliaryLatentClassificationLoss),
    "latent_classification",
)
```

### Inspect and configure the losses

If you want to confirm that both losses were registered, you can look them up with `inspect_registry()`.

```python
from benchrep import inspect_registry

inspect_registry("reconstruction_loss", "gradient_difference")
inspect_registry("custom_objective_loss", "auxiliary_latent_classification")
```

The training config selects the registered names under `losses.reconstruction` and `losses.custom_objective`, alongside built-in MSE and Gaussian KL terms.

```yaml
losses:
  reconstruction:
    - name: mse
      weight: 0.8
      params:
        reduction: mean
    - name: gradient_difference
      weight: 0.2
      params: {}

  regularization:
    - name: gaussian_kld
      weight: 0.0001
      params:
        reduction: mean

  custom_objective:
    - name: auxiliary_latent_classification
      weight: 0.1
      params:
        embedding_dim: 32
        n_classes: 10
```

Each `params` mapping supplies constructor arguments. Here, `embedding_dim` matches `model.params.latent_dim`, and `n_classes` matches MNIST's 10 digit classes. BenchRep scales each scalar loss by its `weight` before summing the terms. See [Losses and objectives](../concepts/models/losses.md) for weighting and logging.

Run the full example to train the VAE, export its latent representations, and evaluate them.

```bash
python -m benchrep.examples.usage.05_custom_loss_pipeline
```

## Register a pretrained encoder

The pretrained example wraps [torchvision’s ConvNeXt-Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html) in a BenchRep encoder. Its constructor loads the weights specified in the config. The autoencoder fine-tunes the encoder alongside a newly initialized decoder on  [STL-10](https://cs.stanford.edu/~acoates/stl10/). For a built-in pretrained encoder, see the ResNet example in [Learn from multiple views](learn-from-multiple-views.md).

[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/11_pretrained_encoder_pipeline.py) · [View configs](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/configs/11_pretrained_encoder_pipeline)

### Adapt the encoder interface

`PretrainedConvNeXtEncoder` subclasses [`BaseEncoder`][benchrep.architecture.encoders.BaseEncoder] and implements `forward()` and `output_dim`. It also exposes `input_shape` for shape checks and `feature_shape` so BenchRep can infer the upsampling decoder's initial feature-map shape. See [Architecture component contracts](../concepts/models/component-contracts.md#canonical-components-also-expose-dimensions).

The wrapper loads the ImageNet weights and replaces the final classification layer with an identity. ConvNeXt's pooling and output normalization yield a 768-element feature vector per image.

```python
self.backbone = convnext_tiny(weights=weights)
self._output_dim = self.backbone.classifier[-1].in_features
self.backbone.classifier[-1] = nn.Identity()
```

The transform pipeline scales pixel values to `[0, 1]`, matching the sigmoid decoder's output range. The encoder then applies per-channel ImageNet normalization inside `forward()`. The reconstruction loss compares the decoder output with the `[0, 1]` input image.

### Register and configure it

Register the encoder with an `ArchitectureComponent` that declares an image input named `x` and a vector result. This lets a composite model check the encoder's connections during assembly. Canonical models use the `BaseEncoder` interface.

```python
from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
    ComponentPort,
    ComponentTensorResult,
)
from benchrep.assembly.registries import ENCODERS

ENCODERS.register(
    "pretrained_convnext",
    ArchitectureComponent(
        component=PretrainedConvNeXtEncoder,
        runtime_inputs=(
            ComponentPort(name="x", supported_structures=("image",)),
        ),
        runtime_result=ComponentTensorResult(supported_structures=("vector",)),
    ),
)
```

Select the encoder by name in the corresponding training config section.

=== "Canonical AE / VAE"

    ```yaml
    encoder:
      name: pretrained_convnext
      params:
        input_shape: [3, 96, 96]
        weights: IMAGENET1K_V1
    ```

=== "Composite"

    ```yaml
    composite_model_components:
      my_encoder:
        kind: encoder
        name: pretrained_convnext
        params:
          input_shape: [3, 96, 96]
          weights: IMAGENET1K_V1
    ```

    In `composite_model_assembly`, connect the encoder's `x` argument to a declared image input and bind its result to a declared vector output. See [Describe values, components, and connections](../concepts/models/composite.md#describe-values-components-and-connections).

### Run the example

The script fine-tunes the autoencoder on [STL-10](https://cs.stanford.edu/~acoates/stl10/) and loads the best checkpoint for test-set prediction. Evaluation produces PCA plots, clustering and reconstruction metrics, and input, reconstruction, and error grids for each RGB channel.

```bash
python -m benchrep.examples.usage.11_pretrained_encoder_pipeline
```

???+ warning "Runtime and disk space"
    The STL-10 archive and extracted files take about 5.4 GiB together. Removing the archive after extraction frees roughly half that space.

    The example trains for 10 epochs at batch size 32. Lower `trainer.max_epochs` for a shorter run or `datamodule.batch_size` to reduce GPU memory use during training. Prediction uses `data.batch_size`.

## Register other component types

- [Registering components](../concepts/extensibility/registration.md) lists which registries accept external implementations and what each expects.
- [Metrics and their contracts](../concepts/evaluation/metrics.md) explains how to select evaluation metrics and declare their result types.
- [Registries and discovery](../concepts/extensibility/discovery.md) shows how to look up registered names, aliases, parameters, and contracts.
- [Configure workflows](configure-workflows/index.md) shows how to use or override these settings from Python.
