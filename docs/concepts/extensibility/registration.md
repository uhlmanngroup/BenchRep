# Registering components

Register an implementation when you want to select it by name while retaining BenchRep's existing workflow machinery. The important decisions are **which registry consumes it** and **which entry type that registry requires**. Registration stores the implementation; construction or execution happens later.

Import registry objects from `benchrep.assembly.registries` and call their [`register()`][benchrep.assembly.registries.core.Registry.register] method with a canonical name, an entry, and optional aliases.

## Four entry types

| Registered entry | Used for | What it describes |
| --- | --- | --- |
| Constructor or factory | Datasets, transforms, optimizers, LR schedulers, loggers, callbacks, loss-weight schedules | How to create the object the consumer needs |
| [`ArchitectureComponent`][benchrep.architecture.composite_model_component_contracts.ArchitectureComponent] | Encoders, decoders, heads | A module class plus its forward-input and result contracts |
| [`LossComponent`][benchrep.architecture.losses.composite_model_contracts.LossComponent] | All loss roles | A loss module class and, where required, its tensor-input contract |
| [`EvaluationMetric`][benchrep.evaluation.metrics.contracts.EvaluationMetric] | Evaluation metrics | The function to execute and the shape of its result |

### Constructors and factories

A directly registered callable generally creates another object. For example, a transform constructor returns the callable that will process a sample tensor:

```python
from benchrep.assembly.registries import TRANSFORMS


class MultiplyIntensity:
    def __init__(self, factor=1.0):
        self.factor = factor

    def __call__(self, image):
        return image * self.factor


TRANSFORMS.register("multiply_intensity", MultiplyIntensity, "intensity_factor")
```

The corresponding transform-step fragment is:

```yaml
name: multiply_intensity
params:
  factor: 0.5
```

The registry stores `MultiplyIntensity`; construction produces `MultiplyIntensity(factor=0.5)`; the [transform pipeline](../data/transforms.md) later calls that object on a routed tensor. A bare function that directly transforms an image is not the same as a factory returning that function.

### Architecture contracts

For a module class `MyProjectionHead` whose `forward(x)` consumes and returns vectors, registration would take this form:

```python
from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
    ComponentPort,
    ComponentTensorResult,
)
from benchrep.assembly.registries import HEADS

HEADS.register(
    "my_projection",
    ArchitectureComponent(
        component=MyProjectionHead,
        runtime_inputs=(ComponentPort("x", ("vector",)),),
        runtime_result=ComponentTensorResult(("vector",)),
    ),
)
```

The contract makes its connections checkable before tensors are available. Composite components can be ordinary `torch.nn.Module` classes. Canonical encoders and decoders additionally implement their respective base interfaces. See [Architecture component contracts](../models/component-contracts.md) for shape categories, mapping results, and dimension requirements.

### Loss contracts

An ordinary canonical reconstruction loss can be registered as `LossComponent(MyLoss)` if its forward method accepts `reconstruction` and `target`. Composite use additionally requires [`LossTensorPort`][benchrep.architecture.losses.composite_model_contracts.LossTensorPort] declarations, and configuration wires those arguments to named inputs or outputs.

The `custom_objective` role follows a different rule: its class inherits [`BaseCustomObjectiveLoss`][benchrep.architecture.losses.custom_objective.BaseCustomObjectiveLoss] and receives `batch` and `model_output` automatically. Register it in `CUSTOM_OBJECTIVE_LOSSES` without individual runtime-input declarations. This role is supported by all 3 model families and has no built-in entries.

The [loss system](../models/losses.md) explains these calling conventions; [Register external components](../../guides/register-components.md) demonstrates both extending reconstruction losses and registering a full-context objective.

### Evaluation functions

Evaluation metrics register the function that computes the result, rather than a constructor. For example:

```python
from benchrep.assembly.registries import EVAL_EMBEDDING_METRICS
from benchrep.evaluation.metrics import EvaluationMetric


def mean_feature_variance(embedding, ddof=0):
    return float(embedding.var(axis=0, ddof=ddof).mean())


EVAL_EMBEDDING_METRICS.register(
    "mean_feature_variance",
    EvaluationMetric(mean_feature_variance, result_kind="scalar"),
)
```

Select it in an evaluation-config fragment:

```yaml
metrics:
  embedding:
    selected: [mean_feature_variance]
    params:
      mean_feature_variance:
        ddof: 0
```

The evaluator supplies the embedding matrix, passes `ddof` to the function, and checks the returned scalar. Other metric registries supply different inputs. Vector and mapping results need the corresponding result declaration; see [Metrics and their contracts](../evaluation/metrics.md).

## Where registered names are consumed

Registration makes an implementation available within an existing configuration mechanism. It does not create a new workflow section or loss role.

| Extension | Configuration and consumer |
| --- | --- |
| Dataset | `dataset.name` and `params`; constructor/factory returns a [`BaseDataset`][benchrep.architecture.data.datasets.BaseDataset] |
| Transform | Each pipeline step's `name` and `params`; factory returns a tensor-transform callable |
| Encoder, decoder, head | Canonical `encoder`/`decoder`, or a composite component's `kind`, `name`, and `params` |
| Loss | A named term under `losses.<role>`; the role selects the registry, and the model supplies or routes its inputs |
| Loss-weight schedule | A loss term's `weight_schedule`; constructor returns an object with [`factor(position)`][benchrep.architecture.losses.weight_schedules.LossWeightSchedule.factor] |
| Optimizer | `optimizer`; factory receives model parameters and configured keyword arguments |
| LR scheduler | `optimizer.lr_scheduler`; factory receives the constructed optimizer and configured parameters |
| Logger | `logger`; class follows Lightning's logger interface |
| Callback | `additional_callbacks`; constructor/factory returns a Lightning callback |
| Evaluation metric | A metric group's `selected` names and per-name `params`; the evaluator executes the registered function |

Custom datasets use the same `name`/`params` shape as built-ins. Their settings are represented by [`CustomDatasetConfig`][benchrep.assembly.schemas.training_config_schema.CustomDatasetConfig]; there is no special `name: custom` requirement. Similarly, the empty custom-objective registry is selected through the existing `losses.custom_objective` role, not a new top-level configuration key.

Evaluation has separate extensible registries for internal clustering metrics, external clustering metrics, embedding metrics, and reconstruction metrics. Their function arguments respectively receive embeddings and cluster labels; reference and cluster labels; embeddings; or input and reconstruction arrays. The [registry reference](../../reference/registries.md) and runtime [`inspect_registry()`][benchrep.inspect_registry] show each full contract.

Model families, evaluation reductions, clustering methods, and predictability probes are currently **closed to external registration**. Their registry entries support selection and discovery, but adding one requires changes to the surrounding implementation. Whole-model customization instead uses [runtime overrides](runtime-implementations.md).

## Registration lifetime and lookup

Registrations belong to the current Python process. Import or execute your registration code before resolving a configuration or launching a wizard that uses it. Another process must import those registrations again; a manifest records names and settings, not the implementation's Python code.

Names and aliases share a namespace within each registry. They are normalized by trimming whitespace, lowercasing, and replacing hyphens with underscores. Existing names, including built-ins, cannot be overwritten through `register()`. In a notebook, re-executing a registration cell with the same name therefore raises an error.

Internally, [`Registry`][benchrep.assembly.registries.core.Registry] stores entries and a mapping from every accepted name to its canonical name. [`get()`][benchrep.assembly.registries.core.Registry.get] retrieves the entry unchanged; [`resolve_key()`][benchrep.assembly.registries.core.Registry.resolve_key] retrieves its canonical name. [`create()`][benchrep.assembly.registries.core.Registry.create] calls a constructor/factory, unwrapping architecture and loss entries to their component class first. Evaluation metrics are retrieved and executed by evaluators rather than passed to `create()`.

Registration validates the entry's declared interface. Configuration resolution then checks its use in a particular experiment, and execution checks actual values. These stages complement one another: a registered component is available for use, not guaranteed to work with every configuration or input.
