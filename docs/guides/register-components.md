# Register external components

Register an implementation under a name, then select it in a workflow config. This tutorial uses 2 existing examples: adapting Fashion-MNIST as a custom dataset, and adding reconstruction and full-context losses to a VAE on MNIST.

## Choose the registration interface

The required entry depends on what you register:

| Component | Registered entry | Selected through |
|---|---|---|
| Dataset, transform, optimizer, LR scheduler, logger, callback, or loss-weight schedule | A constructor or factory directly | The corresponding named config section or list entry |
| Encoder, decoder, or head | `ArchitectureComponent` containing the module class and its port/result contract | A canonical architecture section or a composite component definition |
| Loss | `LossComponent` containing the module class and any required input contract | A term within an existing loss role |
| Evaluation metric | `EvaluationMetric` containing the executed function and its result contract | A metric group's `selected` list, with per-name `params` |

See [Registering components](../understanding/extensibility/registration.md) for the complete registry map. Model families, reductions, clustering methods, and predictability probes currently do not accept custom registrations.

<strong class="highlight">Requirements.</strong> The complete dataset and loss examples use `scverse` for evaluation:

```bash
python -m pip install "benchrep[scverse] @ git+https://github.com/uhlmanngroup/BenchRep.git"
```

!!! note "Register before running"
    Registration applies to the current Python process. Define or import your implementation and register it before resolving a config that uses it. Repeat the registration in another process, including when reconstructing a model for prediction.

## Adapt an external dataset

BenchRep datasets return sample dictionaries. For the canonical VAE, the image is supplied as `x`; optional `label` and `sample_id` fields support prediction exports and evaluation.

The complete dataset example adapts torchvision's Fashion-MNIST. Its interface can be written as:

```python
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

from benchrep.architecture.data import BaseDataset


class FashionMNISTDataset(BaseDataset):
    def __init__(self, root, split="train", download=False):
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
        return {"x": image, "label": label, "sample_id": index}
```

This adapter converts images to tensors; the example's configured transform pipeline handles conversion to scaled floating-point values. See [Datasets and sample structure](../understanding/data/datasets.md) and [Transform pipelines](../understanding/data/transforms.md) for the contracts.

Register the class directly, with an optional alias:

```python
from benchrep.assembly.registries import DATASETS

DATASETS.register("fashion_mnist", FashionMNISTDataset, "fashionmnist")
```

The existing `dataset` section then selects it and supplies constructor arguments:

```yaml
dataset:
  name: fashion_mnist
  params:
    root: data/fashion_mnist
    split: train
    download: true
```

Run the complete dataset example, which trains, predicts, and evaluates:

```bash
python -m benchrep.examples.usage.04_custom_dataset_pipeline
```

[View the script](https://github.com/uhlmanngroup/BenchRep/blob/main/examples/usage/04_custom_dataset_pipeline.py).

## Extend a loss role

Canonical models pass `reconstruction` and `target` to reconstruction losses. Implement a `torch.nn.Module` whose `forward()` accepts those arguments and returns a scalar tensor:

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

This term compares neighboring-pixel changes along the last 2 dimensions. It complements MSE by penalizing discrepancies in local edge strength. The complete example includes the input checks.

Register the class under a name that the config can select:

```python
from benchrep.architecture.losses import LossComponent
from benchrep.assembly.registries import RECONSTRUCTION_LOSSES

RECONSTRUCTION_LOSSES.register(
    "gradient_difference",
    LossComponent(GradientDifferenceLoss),
    "gdl",
)
```

`gradient_difference` is the registered name; `gdl` is an optional alias. `LossComponent` wraps the loss module class. Canonical models supply its fixed inputs; composite use additionally requires a declared input contract. See [Losses and objectives](../understanding/models/losses.md).

## Add a full-context objective

A reconstruction loss receives only its reconstruction and target. For objectives that need labels or other model outputs, subclass `BaseCustomObjectiveLoss`. Its fixed interface receives the complete `batch` and `model_output` mappings:

```python
from collections.abc import Mapping
from typing import Any

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

For this VAE, `model_output["embedding"]` contains the latent means. The classifier learns to predict digit labels, and its gradients also update the encoder and variational head. BenchRep attaches loss modules to the model, so the classifier’s parameters are included in optimization and checkpoints.

Register it in the custom-objective registry:

```python
from benchrep.assembly.registries import CUSTOM_OBJECTIVE_LOSSES

CUSTOM_OBJECTIVE_LOSSES.register(
    "auxiliary_latent_classification",
    LossComponent(AuxiliaryLatentClassificationLoss),
    "latent_classification",
)
```

This existing loss role has no built-in entries. Its fixed interface needs no individual input-port declarations; this implementation expects `batch["label"]` and `model_output["embedding"]`.

## Inspect and configure the losses

Once registered, both losses are available through the discovery interface:

```python
from benchrep import inspect_registry

inspect_registry("reconstruction_loss", "gradient_difference")
inspect_registry("custom_objective_loss", "auxiliary_latent_classification")
```

The supplied training YAML selects them by name alongside the built-in terms:

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

Each `params` mapping supplies constructor arguments; `weight` scales the returned scalar before BenchRep sums the terms. Here, `embedding_dim` must match the VAE’s latent dimension, and `n_classes` matches MNIST’s 10 digit classes. Loss weights do not need to sum to 1. See [Losses and objectives](../understanding/models/losses.md) for the weighting and wiring rules.

Run the complete loss example, including registration, training, prediction, and evaluation:

```bash
python -m benchrep.examples.usage.05_custom_loss_pipeline
```

[View the script](https://github.com/uhlmanngroup/BenchRep/blob/main/examples/usage/05_custom_loss_pipeline.py). It combines the 2 custom terms with MSE reconstruction and Gaussian KL regularization. The auxiliary classifier uses labels during training, so this is a supervised extension of the VAE objective.

Training logs the raw and weighted terms separately. Evaluation still uses its own metrics: registering a training loss does not add an evaluation metric.

## Register other component types

The same register-and-select pattern applies elsewhere, but the interface must match the component's role:

- [Architecture component contracts](../understanding/models/component-contracts.md) describe input ports and tensor or mapping results.
- [Metrics and their contracts](../understanding/evaluation/metrics.md) describe executed metric functions and scalar, vector, or mapping results.
- [Registries and discovery](../understanding/extensibility/discovery.md) explains how to inspect names, aliases, parameters, and contracts before using a component.

To vary the settings in either example, see [Configure workflows](configure-workflows/index.md).
