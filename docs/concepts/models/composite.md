# Composite models

A composite model lets you describe the model's computation in configuration: which components to construct, which values they consume, and where their outputs go. Unlike a canonical AE/VAE, it does not assume an encoder–decoder chain or automatically connect outputs to losses.

Use it for multiple image inputs or augmented views, shared encoders, several prediction heads, or combined reconstruction, classification, regression, and contrastive objectives. A composite model need not reconstruct its input or contain a variational head. Its current input contract does, however, require at least one image field with the `sample_image` role.

## Describe values, components, and connections

The configuration separates 3 questions:

| Section | Question it answers |
| --- | --- |
| `composite_model_declarations` | Which batch fields does the model expect, and which values will its computation produce? |
| `composite_model_components` | Which registered modules should BenchRep instantiate, with which constructor parameters? |
| `composite_model_assembly` | How are those modules called, and how do their inputs and outputs connect? |

This training-config fragment builds a simple composite autoencoder. Dataset, optimization, and loss settings are configured separately:

```yaml
model:
  name: composite

composite_model_declarations:
  expects:
    x: sample_image
  batch_metadata:
    sample_id: index
    label: group
  produces:
    z: embedding_vector
    x_hat: reconstruction_image

composite_model_components:
  image_encoder:
    kind: encoder
    name: mlp
    params:
      input_shape: [1, 28, 28]
      output_dim: 32
      hidden_dims: [128]
  image_decoder:
    kind: decoder
    name: mlp
    params:
      input_dim: 32
      output_shape: [1, 28, 28]
      hidden_dims: [128]
      output_activation: sigmoid

composite_model_assembly:
  encode:
    component: image_encoder
    inputs:
      x: expects.x
    outputs: produces.z
  reconstruct:
    component: image_decoder
    inputs:
      z: produces.z
    outputs: produces.x_hat
```

In `expects`, `x` is the batch field's name and `sample_image` describes its role. In `components`, `image_encoder` is an identifier you choose; `kind` selects the registry and `name` selects a registered implementation or alias. In `assembly`, `encode` identifies a call to that component.

An input connection such as `x: expects.x` means “pass the batch field `x` to this component's `forward(x=...)` argument.” `z: produces.z` passes a previously computed value to the decoder's `z` argument. Constructor settings remain in `params`; they are not runtime input connections.

`produces` includes intermediate values as well as final outputs. For example, a composite VAE declares encoder features, Gaussian parameters, a sample, and a reconstruction. Components returning several tensors bind them by key; see [Architecture component contracts](component-contracts.md#one-tensor-or-a-mapping).

### Execution order and shared weights

BenchRep derives execution order from the connections, rather than relying on the order in which assembly steps were written. The [implementation](#implementation) below explains how those dependencies become an executable plan.

<strong class="highlight">A component identifier refers to one instantiated module.</strong> Multiple assembly steps can call the same component on different inputs, sharing its weights. For example, 2 augmented images can pass through the same encoder. To use independently trained encoders instead, declare 2 component identifiers, even if they select the same registered implementation.

Composite construction does not infer matching feature widths between components. In the example, the encoder's `output_dim` and decoder's `input_dim` are both explicitly `32`. Structure checks distinguish images from vectors; they do not prove that every numerical dimension is compatible.

## Declared roles

Names such as `x`, `z`, or `class_logits` are yours to choose. Their **roles** describe what the values represent and let BenchRep check or guide their use. A role also determines the expected [tensor structure](component-contracts.md#tensor-structure-is-not-tensor-size).

### Inputs: images, targets, and conditions

All fields declared under `expects` must be available in the batch after [transforms](../data/transforms.md). A field can be consumed by an architecture component, a loss, or both.

| Role | Meaning |
| --- | --- |
| `sample_image` | A primary image input. At least one is required; the first establishes the runtime batch size. If there is exactly one, it is also the default input for routed transforms. |
| `positive_image`, `negative_image` | Related and contrasting image inputs, for example in a triplet objective. The dataset or transforms must supply them; these roles do not select pairs or negatives. |
| `condition_image` | An additional image used to condition a computation. It must be connected to a component that accepts it. |
| `categorical_prediction_target_scalar` | One categorical target per sample, such as an integer class label. |
| `categorical_prediction_target_vector` | A vector-valued categorical target, such as a class-distribution encoding. The selected loss must support this role and encoding. |
| `continuous_prediction_target_scalar`, `continuous_prediction_target_vector` | One continuous target or a vector of continuous targets per sample. |
| `categorical_condition_scalar`, `categorical_condition_vector` | Categorical information supplied as a conditioning input, represented by a scalar or vector. |
| `continuous_condition_scalar`, `continuous_condition_vector` | Continuous conditioning information, represented by a scalar or vector. |

A target is a value an objective asks the model to predict; a condition is information made available to the computation. Declaring either does not create a loss or connect it automatically. Categorical roles do not perform one-hot encoding, choose a dtype, or apply softmax: those requirements belong to the selected component or loss.

### Outputs: representations, predictions, and auxiliary values

| Role | Meaning |
| --- | --- |
| `embedding_vector` | A learned representation intended for downstream use. |
| `projection_vector` | A projected or intermediate representation, for example features passed to another head or a contrastive objective. |
| `reconstruction_image` | An image reconstruction, eligible for reconstruction loss wiring and reconstruction export. |
| `categorical_prediction_scalar`, `categorical_prediction_vector` | A categorical prediction represented by one value or a vector, such as a class index or class logits. The component determines its actual encoding. |
| `continuous_prediction_scalar`, `continuous_prediction_vector` | A scalar or vector of predicted continuous values. |
| `auxiliary_image` | An intermediate or supporting image-valued result. |
| `categorical_auxiliary_scalar`, `categorical_auxiliary_vector` | Supporting categorical values that are not designated prediction outputs. |
| `continuous_auxiliary_scalar`, `continuous_auxiliary_vector` | Supporting continuous values, such as a VAE's posterior log-variance vector. |

The distinction between roles can be descriptive for one operation and functional for another. For example, both `embedding_vector` and `projection_vector` can connect to a vector-valued component input, enter the built-in triplet loss, or serve as an AnnData `primary_key`. But the built-in Gaussian KL loss accepts `embedding_vector` or `continuous_auxiliary_vector` for its mean input, not `projection_vector`.

Architecture ports check **structure**, ordinary loss ports check **supported roles**, and prediction exports use both roles and structures. Vector outputs can be selected for AnnData; declared scalar inputs and selected scalar outputs become observation columns. Reconstruction exports specifically select `reconstruction_image` outputs. See [Loss wiring](losses.md#wire-composite-losses) and [Selecting prediction exports](../prediction/exports.md).

### Batch metadata

`batch_metadata` is optional. When present, it declares annotations to retain for prediction:

| Role | Meaning and effect |
| --- | --- |
| `index` | Sample identifiers used as the AnnData observation index. At most one field can have this role. |
| `group` | Observation annotations, such as labels or experimental groups. It does not cause grouped splitting or sampling. |

Every declared metadata field must be present in the batch. Metadata is separate from `expects`: it cannot be wired into ordinary component or loss inputs. If a label is used for classification, declare it as a target under `expects`; if it only annotates predictions, `batch_metadata` is appropriate. Custom objectives receive the complete batch and can access either.

## Attach the objective

Each ordinary composite loss explicitly selects its inputs. For the autoencoder above:

```yaml
losses:
  reconstruction:
    - name: mse
      composite_wiring:
        reconstruction: produces.x_hat
        target: expects.x
```

This compares the decoder output with the input image. Wiring a different declared image target supports other arrangements, such as reconstructing a clean image from a corrupted input. Every composite model needs at least one configured loss, but no particular role is mandatory. [Losses and objectives](losses.md) explains how terms are checked and combined.

## Implementation

<strong class="highlight">Composite assembly connects tensor values, not module objects.</strong> BenchRep instantiates the modules, then follows a resolved plan that says which tensors to pass to each call and where to store its results. The model's `forward()` method interprets that plan using ordinary PyTorch operations.

### Resolve the configuration into a plan

The [composite resolver](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/composite_model_resolver.py) produces a `CompositeModelSpec`. This is a collection of dataclasses describing the computation; it contains no instantiated neural-network modules or batch tensors.

| Config information | Resolved representation |
| --- | --- |
| `expects`, `produces`, and `batch_metadata` | Names mapped to their roles |
| Each component ID, kind, name, and `params` | A component specification with its canonical registry name and constructor arguments |
| Each assembly step | A step specification naming its component, its input sources, and its output destinations |
| Dependencies between steps | Assembly steps grouped into execution levels |

For example, the decoder's `z: produces.z` connection becomes `inputs_from_model_outputs={"z": "z"}`. Its `outputs: produces.x_hat` becomes the destination name `x_hat`. References to `expects.*` go into a separate input-source mapping, so execution does not repeatedly parse dotted strings or rediscover their meaning.

The resolver checks these connections against registered [component contracts](component-contracts.md). It also records which step produces each declared output, then repeatedly selects steps whose producers have already been ordered. If steps remain but none can run next, the graph contains a dependency cycle. Missing references, unused components, and outputs without exactly one producer are rejected.

Loss configuration is resolved separately into `LossSpec` objects using the same declarations. The resulting training run specification therefore holds both an **architecture plan** and a **loss plan**. This separation lets the architecture describe computation without embedding a particular objective in each component. The [wizard](../workflows/wizards.md) can preview the resolved architecture before modules are constructed.

### Construct the model and its modules

The [model builder](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/model_builder.py) creates one module for each component ID by looking up its registry entry and passing its constructor parameters. It also builds the loss modules and optimizer/scheduler factories, then passes everything to [`CompositeModel`][benchrep.architecture.models.composite.CompositeModel].

The model keeps component instances in `self.components_by_id`, a PyTorch `ModuleDict`, and loss terms in nested `ModuleDict` containers indexed by role and loss ID. These containers register the modules as children of the model: their parameters and buffers participate in device movement, parameter discovery, and checkpointing.

Assembly steps hold component IDs rather than their own copies of modules. This is why calling `image_encoder` in 2 steps shares weights, while declaring 2 component IDs creates separate instances.

### Execute one batch

The training hook checks the batch's declared fields and extracts the `expects` tensors into a `model_inputs` dictionary. [`forward()`][benchrep.architecture.models.composite.CompositeModel.forward] then creates an empty `model_output` dictionary and visits assembly steps in dependency order.

For the autoencoder above, the effective calls are:

```python
model_output["z"] = components_by_id["image_encoder"](
    x=model_inputs["x"],
)
model_output["x_hat"] = components_by_id["image_decoder"](
    z=model_output["z"],
)
```

The actual executor performs these calls in a loop. For each step, it gathers the specified tensors into keyword arguments, calls the selected module, checks the returned tensor or mapping, and stores its values under the declared output names. Checks include tensor structure, batch size, and exact mapping keys. Independent steps at the same dependency level still execute sequentially.

This process supports branches and joins without special model classes. Several steps can read the same tensor; a component accepting several arguments can consume values from different branches. The returned dictionary contains every declared output, including intermediate values.

### Compute losses and propagate gradients

After the forward pass, the model follows each loss specification to gather its inputs. The reconstruction term above receives `model_output["x_hat"]` and `model_inputs["x"]`. Custom objectives instead receive the original batch and the complete output dictionary. The model evaluates each loss module, applies its effective weight, and returns their scalar sum from `training_step()`; see [Loss implementation](losses.md#implementation).

<strong class="highlight">The resolved execution plan and PyTorch's autograd graph are different things.</strong> The plan fixes which calls to make. During each forward pass, the tensor operations inside those calls build the autograd graph that records how values depend on trainable parameters.

Routing tensors through dictionaries does not detach or copy them. Lightning can therefore backpropagate the summed loss through every contributing branch. Repeated calls to a shared component contribute gradients to the same parameters; an output with no path to a loss does not acquire a training signal merely because it was declared. BenchRep supplies the call plan and objective, while PyTorch handles differentiation.

## Prediction behavior

Prediction reconstructs the composite model from the training run's recorded configuration and checkpoint. Its declared inputs are still required, including targets declared under `expects`; prediction does not automatically remove unused inputs or graph branches. Export selection chooses which results to save, not which assembly steps to execute.

For a composite VAE, a prediction config can reroute the decoder from the sampled latent to the mean:

```yaml
inference:
  composite_model_assembly_input_overrides:
    reconstruct:
      inputs:
        z: produces.z_mu
```

This fragment assumes a `reconstruct` step whose `z` input originally uses a sample produced by the same variational-head step as `z_mu`. These overrides only replace existing `produces.*` input connections with outputs from the same producer; they cannot replace components or output bindings. See the [prediction inference schema](../../reference/schemas/prediction.md).

The prediction hook runs the forward graph without computing training losses, then returns [`CompositePredictionOutput`][benchrep.interfaces.contracts.CompositePredictionOutput], containing declared inputs, outputs, and batch metadata. These are the tensors and annotations from which the export layer selects artifacts.

Whole-model runtime overrides are <strong class="highlight">currently supported only by the canonical families</strong>. For composite models, extend registered architecture components and losses, then assemble them through configuration. Datamodule overrides remain available. The [:material-script-text-outline: multitask composite example](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/08A_dual_input_multitask_composite_vae_yaml.py) demonstrates several of these mechanisms together.
