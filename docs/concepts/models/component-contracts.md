# Architecture component contracts

A component's constructor tells BenchRep how to create it. Its **runtime contract** tells BenchRep how it can be connected to other components: which arguments `forward()` accepts, and whether it returns an image, a vector, or a mapping of tensors.

These contracts make configuration-driven assembly possible without hard-coding every encoder, decoder, or head into the model executor. They check the interface, not whether a component is scientifically appropriate for a task.

## Construction and computation are separate

An encoder configuration might pass `input_shape`, `hidden_dims`, and `output_dim` to its constructor. An assembly step then calls that instance with `x=...`. Constructor arguments belong in `params`; tensors passed during execution belong in the step's `inputs` mapping.

The encoder, decoder, and head registries store [`ArchitectureComponent`][benchrep.architecture.composite_model_component_contracts.ArchitectureComponent] entries, each describing:

| Part | Purpose |
| --- | --- |
| `component` | The `torch.nn.Module` class to instantiate |
| `runtime_inputs` | Named [`ComponentPort`][benchrep.architecture.composite_model_component_contracts.ComponentPort] entries matching the component's forward arguments |
| `runtime_result` | A single-tensor or mapping result contract |

For example, the registered MLP encoder accepts an image through `x` and returns a vector. The MLP decoder accepts a vector through `z` and returns an image. `kind: encoder`, `decoder`, or `head` chooses the registry; it is the entry's contract that describes its callable interface.

Use [`inspect_registry()`][benchrep.inspect_registry] to inspect both the constructor and contract:

```python
from benchrep import inspect_registry

inspect_registry("encoder", "mlp")
inspect_registry("head", "gaussian_variational")
```

## Tensor structure is not tensor size

[`TensorStructure`][benchrep.architecture.composite_model_roles.TensorStructure] describes each sample's structure. Composite runtime checks expect these batched shapes:

| Structure | Batched shape | Example |
| --- | --- | --- |
| `scalar` | `(B,)` or `(B, 1)` | One regression value per sample |
| `vector` | `(B, F)` | An embedding or class-logit vector |
| `image` | `(B, C, H, W)` | A batch of 2D multichannel images |

A port can support several structures. Semantic [roles](composite.md#declared-roles) map to these structures: `embedding_vector` and `projection_vector`, for example, are both vectors. Architecture connections are checked at this structural level; loss connections can impose narrower semantic requirements.

These declarations do not encode exact feature widths, channel counts, dtypes, or numerical ranges. Connecting a 32-dimensional vector to a layer expecting 64 features is still invalid, even though both sides are vectors. Configure compatible dimensions and preprocessing; actual module execution catches requirements beyond the declared contract.

## One tensor or a mapping

[`ComponentTensorResult`][benchrep.architecture.composite_model_component_contracts.ComponentTensorResult] describes one unnamed tensor. Assembly binds it directly to a declared output:

```yaml
outputs: produces.embedding
```

[`ComponentMappingResult`][benchrep.architecture.composite_model_component_contracts.ComponentMappingResult] instead declares the exact keys of a returned mapping. For example, [`GaussianVariationalHead`][benchrep.architecture.heads.variational.GaussianVariationalHead] returns `z_sample`, `z_mu`, and `z_logvar`:

```yaml
outputs:
  z_sample: produces.latent_sample
  z_mu: produces.latent_mean
  z_logvar: produces.latent_logvar
```

The left-hand names are fixed by the component; the right-hand names are your model declarations. All contracted outputs must be bound, even if only some are used by later steps or selected for export. A tuple of tensors is not a substitute for either supported result form.

## Canonical components also expose dimensions

Components used by a canonical model must implement [`BaseEncoder`][benchrep.architecture.encoders.base.BaseEncoder] or [`BaseDecoder`][benchrep.architecture.decoders.base.BaseDecoder]. These interfaces expose information needed to connect the predefined architecture:

| Interface | Information supplied |
| --- | --- |
| Encoder | `input_shape`, vector `output_dim`, and optional spatial `feature_shape` |
| Decoder | Latent `input_dim` and image `output_shape` |

BenchRep uses this information to check input/output agreement, supply compatible decoder dimensions, and construct a VAE's variational head. Where supported, an encoder's spatial feature shape can also supply an upsampling decoder's initial shape.

Composite components can be ordinary `torch.nn.Module` classes with the declared contract; they need not inherit the canonical encoder/decoder bases. There is no corresponding mandatory head base class. This allows components with several inputs or outputs while keeping the graph executor independent of their internal architecture.

## Checks happen at different stages

The contract objects are dataclasses. Constructing an `ArchitectureComponent` checks that its declared input names can bind to the module's forward signature using placeholder arguments; it does not instantiate the module or run tensors through it. Composite resolution then uses the entry to check the configured connections before construction.

| Stage | What is checked |
| --- | --- |
| Registration | The entry describes a module class and valid ports; the declared inputs can call its forward signature |
| Configuration resolution | Constructor arguments match the selected signature; references, port connections, structures, and graph dependencies are valid |
| Construction | Constructor arguments create the component; canonical dimension checks also connect its base-interface information |
| Composite execution | Actual tensor structures, batch sizes, and result keys match the resolved declarations and bindings; module operations enforce their own requirements |

At runtime, the [composite executor](composite.md#implementation) uses the resolved field roles and result bindings to check actual values. Contracts describe acceptable inputs and outputs; they do not reshape tensors or insert adapters. The [contract definitions](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/architecture/composite_model_component_contracts.py) and [composite resolver](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/composite_model_resolver.py) implement these two stages.

When extending BenchRep, the registration contract and the implementation must agree. See [Registering components](../extensibility/registration.md), the [architecture interface reference](../../reference/interfaces.md#architecture-components), and the [built-in implementations](../../reference/components/architectures.md).
