# Losses and objectives

BenchRep treats an objective as a collection of loss terms. Each term selects a registered implementation, computes a scalar value, and contributes that value multiplied by its weight. This lets you change a reconstruction measure, combine several tasks, or add a custom objective without rewriting the model's training loop.

## Loss roles select registries

The parent key under `losses` identifies a **role** and selects its registry. The term's `name` selects an implementation within that registry; `params` supplies its constructor arguments.

| Role | Purpose | Built-in choices                                                                                                                                       |
| --- | --- |--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `reconstruction` | Compare reconstructed and target images | [`mse`][benchrep.architecture.losses.reconstruction.MSEReconstructionLoss], [`mae`][benchrep.architecture.losses.reconstruction.MAEReconstructionLoss] |
| `regularization` | Constrain the latent distribution | [`gaussian_kl`][benchrep.architecture.losses.regularization.GaussianKLDivergenceLoss]                                                                  |
| `contrastive` | Bring related representations closer than contrasting ones | [`triplet_margin`][benchrep.architecture.losses.contrastive.TripletMarginContrastiveLoss], , [`nt_xent`][benchrep.architecture.losses.contrastive.NTXentContrastiveLoss]                                                            |
| `classification` | Learn categorical predictions | [`cross_entropy`][benchrep.architecture.losses.classification.CrossEntropyClassificationLoss]                                                          |
| `regression` | Learn continuous predictions | [`mse`][benchrep.architecture.losses.regression.MSERegressionLoss]                                                                                     |
| `custom_objective` | Compute an objective from the full batch and model outputs | No built-ins; register your own                                                                                                                        |

The same name can mean different implementations in different roles: reconstruction `mse` compares images, while regression `mse` compares continuous predictions and targets. Registered aliases work too. [`inspect_registry("reconstruction_loss", "mse")`][benchrep.inspect_registry] shows the constructor and supported composite inputs; the [catalogue](../../reference/components.md) lists all loss registries.

### Which losses can a family use?

| Family | Supported roles | Minimum objective for config-built models |
| --- | --- | --- |
| Autoencoder | `reconstruction`, `custom_objective` | At least one term in either role |
| VAE | `reconstruction`, `regularization`, `custom_objective` | Reconstruction **and** regularization, or at least one custom objective |
| Composite | All 6 roles | At least one configured term |

A custom objective can accompany ordinary terms or replace them. If it replaces reconstruction or KL regularization, it must implement any such behavior you still want; BenchRep does not add it implicitly. Whole-model runtime overrides own their loss computation instead of using this config-built loss system.

## Configure and combine terms

For a canonical VAE, this fragment combines 2 reconstruction measures and a KL term:

```yaml
losses:
  reconstruction:
    - name: mse
      weight: 0.8
    - name: mae
      weight: 0.2
  regularization:
    - name: gaussian_kl
      weight: 0.001
```

The objective is `0.8 * mse + 0.2 * mae + 0.001 * gaussian_kl`. Weights default to `1.0` and are not normalized, either within roles or across them. An optional [weight schedule](loss-weight-schedules.md) multiplies a term's base weight.

Each loss must return a scalar tensor. Its reduction determines that scalar's scale: the built-in reconstruction losses default to a mean over image elements, whereas Gaussian KL sums over latent dimensions and then averages over samples. Consequently, equal coefficients do not imply equal contributions or gradients. Choose reductions and weights together when comparing objectives.

A zero weight removes a term's contribution to the total, but does not skip its computation or logging.

## Canonical models supply loss inputs

Canonical models use fixed calling conventions:

| Role | Arguments supplied to the loss |
| --- | --- |
| `reconstruction` | `reconstruction` from the model and `target=batch["x"]` |
| `regularization` | VAE posterior `z_mu` and `z_logvar` |
| `custom_objective` | Complete `batch` and `model_output` mappings |

There is no `composite_wiring` in a canonical loss configuration. To register another loss for an ordinary role, its forward method must accept that role's arguments. See [Register external components](../../guides/register-components.md) for an example extending reconstruction losses.

## Wire composite losses

Composite models do not know which of their declared values should enter a loss. Every ordinary loss therefore needs `composite_wiring`, mapping its forward arguments to `expects.*` or `produces.*` references.

For example, suppose a model declares `label: categorical_prediction_target_scalar` under `expects`, and `class_logits: categorical_prediction_vector` under `produces`. A classification term connects them as follows:

```yaml
losses:
  classification:
    - name: cross_entropy
      composite_wiring:
        prediction: produces.class_logits
        target: expects.label
```

Here, `prediction` and `target` are the loss's argument names; `class_logits` and `label` are model declaration names. The built-in cross-entropy registration expects a vector of unnormalized logits and scalar class targets. The output component should not apply softmax before this loss.

Ordinary composite losses declare their argument contracts using [`LossTensorPort`][benchrep.architecture.losses.composite_model_contracts.LossTensorPort]. Resolution checks that every wired field has a supported role. For example, reconstruction MSE accepts a `reconstruction_image` prediction and a `sample_image`, `positive_image`, or `negative_image` target. A tensor being image-shaped is not sufficient if its declared role is incompatible.

These checks do not change the tensors' encodings or dimensions. The selected loss still requires appropriately shaped and typed values at runtime. See [Declared roles](composite.md#declared-roles) and the [loss implementation reference](../../reference/components/losses.md).

## Custom objectives

A custom objective is useful when a loss needs several parts of the batch or model output together, or when its computation does not fit an ordinary role's inputs. It subclasses [`BaseCustomObjectiveLoss`][benchrep.architecture.losses.custom_objective.BaseCustomObjectiveLoss] and implements a keyword-only `forward(*, batch, model_output)` returning a scalar tensor.

The `batch` mapping includes annotations and additional fields supplied by the dataloader. The `model_output` mapping contains the canonical forward outputs or, for composite models, the declared `produces` values. It is not the model instance itself.

For example, a custom objective can contain a classifier that takes `model_output["embedding"]` and compares its logits with `batch["label"]`. This adds label supervision to a canonical model without changing its predefined forward graph. The [:material-script-text-outline: custom-loss example](https://github.com/uhlmanngroup/BenchRep/blob/main/examples/usage/05_custom_loss_pipeline.py) demonstrates this alongside an external reconstruction loss.

Register the class as a [`LossComponent`][benchrep.architecture.losses.composite_model_contracts.LossComponent] in `CUSTOM_OBJECTIVE_LOSSES`, then select it under `losses.custom_objective`. Custom objectives do not declare `runtime_inputs` or use `composite_wiring`; BenchRep supplies the 2 mappings automatically.

Loss modules are attached to the model. Their trainable parameters are included in the built-in models' optimizer parameters and checkpoints. A classifier contained inside a loss nevertheless remains part of the objective, not a declared prediction output; use an architecture head when its predictions should be exposed through the model's ordinary prediction/export interface.

## Identify and inspect contributions

Each term has an identity used in module keys and metric names. Without an `id`, it is the canonical registered loss name. With `id: view_a`, a loss named `mse` becomes `view_a_mse`. Identities must be unique within each role, so give repeated instances distinct IDs. Aliases resolve to the same canonical name.

For a reconstruction term whose identity is `mse`, validation logs include:

| Metric | Meaning |
| --- | --- |
| `losses/val/reconstruction/mse` | Raw scalar loss |
| `losses/val/reconstruction/mse_weighted` | Loss multiplied by its effective weight |
| `loss_weights/val/reconstruction/mse` | Effective weight |
| `losses/val/total_weighted_loss` | Sum of all weighted terms |

Training uses `train` in place of `val`. These values help distinguish improvement in a raw objective from a change caused by its weight. They can also be monitored by [checkpointing and callbacks](../training/callbacks.md).

## Implementation

The loss system separates the reusable implementation from a particular configured use of it, and from the value computed for a batch:

| Object | What it represents |
| --- | --- |
| [`LossComponent`][benchrep.architecture.losses.composite_model_contracts.LossComponent] | A registry entry containing a loss class and its calling contract |
| [`TrainingLossTermConfig`][benchrep.assembly.schemas.training_config_schema.TrainingLossTermConfig] | One requested use of a loss, including its parameters, identity, weight, schedule, and optional wiring |
| `LossSpec` | The resolved instructions for constructing and calling that configured loss |
| [`LossTerm`][benchrep.architecture.losses.base.LossTerm] | The instantiated loss module, its base weight, and optional schedule |
| Scalar loss tensor | The result of calling the loss on one batch; retained in the differentiation graph |

Registering MSE once can therefore support many configured MSE terms, each with its own instance, identity, settings, and input connections.

### Resolve names, identities, and inputs

The [loss resolver](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/loss_resolver.py) resolves aliases, checks constructor arguments, derives term identities, and rejects identity or generated metric-name collisions. It produces a flat tuple of `LossSpec` objects across the configured roles.

Canonical loss resolution checks that each implementation supports the model's fixed calling convention. Its specifications do not need tensor routes: the model already knows which values to pass for reconstruction, regularization, or a custom objective.

For a composite loss, resolution additionally checks the declared roles and converts dotted references into argument-to-field mappings. A reconstruction term wired to `produces.x_hat` and `expects.x`, for example, records:

| Specification field | Resolved value |
| --- | --- |
| `inputs_from_model_outputs` | `{"reconstruction": "x_hat"}` |
| `inputs_from_model_inputs` | `{"target": "x"}` |

A composite custom objective instead records `context_inputs={"batch": "batch", "model_output": "model_output"}`. This tells the executor to supply whole mappings rather than selecting individual tensors.

### Instantiate and retain each loss

The [loss builder](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/loss_builder.py) constructs each registered loss from the specification's constructor parameters. It constructs any schedule, wraps the loss in a `LossTerm`, and groups the terms by role and identity.

The concrete model stores those groups in PyTorch `ModuleDict` containers. `LossTerm` itself inherits `nn.Module`, and its `loss` attribute is the child loss module. This ownership is why a learnable auxiliary classifier inside a custom objective participates in parameter discovery, device movement, and checkpointing without a separate optimizer-registration mechanism.

`LossTerm` does not implement a loss calculation in `forward()`. Its two responsibilities remain separate: `loss_term.loss(...)` computes the raw value, while [`effective_weight()`][benchrep.architecture.losses.base.LossTerm.effective_weight] obtains the coefficient for the current position.

### Evaluate the objective during a model step

Each model first runs its forward computation. Canonical models prepare one argument mapping per loss role; the composite model uses each `LossSpec` to gather arguments from model inputs, model outputs, or full context. After that difference, the calculation follows the same pattern:

```python
raw_loss = loss_term.loss(**loss_inputs)
effective_weight = loss_term.effective_weight(
    epoch=self.current_epoch,
    step=self.global_step,
)
weighted_loss = effective_weight * raw_loss
```

The models require `raw_loss` to be a zero-dimensional tensor, log the raw and weighted values, and sum all weighted terms. The training hook returns that total to Lightning for backpropagation and optimization. Validation uses the same objective calculation with validation logging.

No loss is automatically detached from the forward graph. If several objectives depend on the same encoder, their weighted gradients accumulate into that encoder's parameters. An objective affects only the parameters on which its tensor computation depends; merely registering or configuring it cannot create a missing differentiable connection.

The canonical model code contains the fixed loss argument groups; the [composite executor](composite.md#implementation) interprets resolved routes. Both use the same `LossTerm` and [weight-schedule mechanism](loss-weight-schedules.md#implementation).
