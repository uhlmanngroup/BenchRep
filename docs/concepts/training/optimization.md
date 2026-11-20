# Training and optimization

The objective determines what training minimizes; the optimizer determines how parameter values change. BenchRep's config-built models use Lightning's automatic optimization: the model returns a scalar training loss, and Lightning manages backward computation and optimizer updates.

## Configure the optimizer

The `optimizer` section selects a registered optimizer and supplies its constructor arguments:

```yaml
optimizer:
  name: adamw
  params:
    lr: 0.001
    weight_decay: 0.01
```

The built-in choices include Adam, AdamW, and SGD. Their `params` follow the selected PyTorch constructor. [`inspect_registry("optimizer", "adamw")`][benchrep.inspect_registry] exposes that signature; the [component catalogue](../../reference/components.md#catalogue-optimizer) lists registered names and aliases.

The concrete BenchRep models pass `self.parameters()` to the optimizer. This includes registered architecture modules and any learnable modules inside their loss terms. A whole-model [runtime override](../workflows/overrides.md#supply-an-implementation) owns its optimizer setup instead; the ordinary `optimizer` section is bypassed along with model construction.

## Schedule the learning rate

An optional `optimizer.lr_scheduler` changes the optimizer's learning rate during training:

```yaml
optimizer:
  name: adam
  params:
    lr: 0.001
  lr_scheduler:
    name: cosine_annealing
    params:
      T_max: 50
    interval: epoch
    frequency: 1
```

`params` configures the scheduler itself. `interval` and `frequency` tell Lightning when to advance it: every configured number of epochs or optimizer steps. With gradient accumulation, optimizer steps are less frequent than training batches.

A metric-driven scheduler such as `reduce_on_plateau` also requires `monitor`, for example `losses/val/total_weighted_loss`. The metric must exist when the scheduler runs; coordinate its frequency with validation. Its monitor is independent of checkpointing and early stopping.

Learning-rate scheduling and [loss-weight scheduling](../models/loss-weight-schedules.md) solve different problems. The former changes parameter-update behavior; the latter changes each objective's contribution to the total loss. See the [training schema](../../reference/schemas/training.md) and [scheduler catalogue](../../reference/components.md#catalogue-lr_scheduler) for supported settings and implementations.

## Configure the training loop

The `trainer` section forwards declared settings and additional non-null arguments to Lightning's Trainer:

```yaml
trainer:
  max_epochs: 50
  accelerator: auto
  devices: 1
  precision: 32-true
  accumulate_grad_batches: 2
  gradient_clip_val: 1.0
```

BenchRep reserves `default_root_dir`, `logger`, `callbacks`, and `enable_checkpointing`. Configure their supported equivalents through `run`, `logger`, `checkpointing`, `early_stopping`, and `additional_callbacks`. These controls let BenchRep keep artifacts and callbacks consistent with the run's records.

The training workflow calls `Trainer.fit()`, which performs training and any configured validation. It does not run a separate test loop. For held-out assessment, run [prediction](../prediction/index.md) on the intended dataset and then [evaluation](../evaluation/index.md).

## Implementation: constructing optimizers at the right time

An optimizer needs actual parameter objects, so configuration alone is not enough to instantiate it. BenchRep handles that dependency with delayed factories:

1. The [optimizer builder](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/optimizer_builder.py) resolves the registered constructor and captures its parameters in an optimizer factory.
2. Model construction stores that factory on the model. If a learning-rate scheduler is configured, it also stores a scheduler factory.
3. Lightning calls the model's `configure_optimizers()`. The model passes its parameters to the first factory, then passes the resulting optimizer to the scheduler factory.
4. The model returns the optimizer and scheduler settings to Lightning, which advances them during fitting.

The scheduler builder checks its constructor signature before instantiation, then checks that the created scheduler belongs to the supplied optimizer. It also detects `ReduceLROnPlateau` and requires its monitored metric. This keeps configuration, model construction, and optimization setup separate without creating placeholder parameters or optimizers.

During each training step, the built-in model executes its forward computation, evaluates the configured losses, and returns their weighted sum. PyTorch builds the differentiation graph from those tensor operations; Lightning performs backward computation, accumulation, clipping, and optimizer stepping according to the Trainer configuration. The [model implementation sections](../models/index.md) explain the forward and loss mechanisms in detail.
