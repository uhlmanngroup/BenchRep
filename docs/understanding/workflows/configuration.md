# Configuration semantics

A configuration describes an experiment: which components to use, how to connect them, and how to run and evaluate them. BenchRep checks that description and derives the settings needed for execution. Understanding that distinction helps explain why a valid YAML file is not necessarily a valid experiment.

## One configuration, several ways to author it

Workflow entrypoints accept a YAML path or the matching Python config object through `config`. Editing a file, building a config in Python, and using the wizard are ways to author that input; they do not create different kinds of workflows.

Each stage has its own schema: [`TrainingConfig`][benchrep.assembly.schemas.training_config_schema.TrainingConfig], [`PredictionConfig`][benchrep.assembly.schemas.prediction_config_schema.PredictionConfig], or [`EvaluationConfig`][benchrep.assembly.schemas.evaluation_config_schema.EvaluationConfig]. See [Configure workflows](../../guides/configure-workflows/index.md) for the authoring routes in practice.

## Sections, components, and parameters

Top-level sections group related settings, such as the training `encoder`, `optimizer`, and `dataset`. Nested fields describe each section. Many component sections use a registered `name` and constructor arguments under `params`:

```yaml
encoder:
  name: mlp
  params:
    input_shape: [1, 28, 28]
    output_dim: 32
    hidden_dims: [256, 128]
```

Here `mlp` selects an encoder implementation, while `params` configures that implementation. A registered alias selects the same implementation under another name. External components use this interface where the relevant registry permits registration.

The schema determines which sections and fields are allowed; the component contract determines how an implementation fits into the model or workflow. Not every section follows the same `name`/`params` structure. Use the [schema reference](../../reference/schemas/index.md) for configuration fields and the [component catalogue](../../reference/components.md) for names and constructor details.

## From authored settings to execution

| Form | What it represents |
| --- | --- |
| Authored configuration | The choices you supplied, including omitted fields and explicit nulls in a YAML or wizard draft |
| Effective configuration | The typed configuration obtained after composing the base input with section overrides and applying schema defaults |
| Resolved configuration | Settings interpreted in context, including applicable inheritance, derived values, and runtime override decisions |

Composition determines **which settings are supplied**. Schema validation checks their types, allowed values, and schema-level relationships. Resolution checks broader compatibility and determines **how those settings will be used**.

For example, a VAE latent dimension must be positive at schema validation. Resolution can then set the config-built canonical decoder's input dimension from it. Prediction can resolve `checkpoint: best` to a particular checkpoint using the training manifest.

<!-- Figure slot: configuration lifecycle. Suggested assets:
../../assets/understanding/workflows/configuration-lifecycle-light.svg
../../assets/understanding/workflows/configuration-lifecycle-dark.svg
Show authored input and section overrides feeding composition/validation,
then resolution using upstream records and runtime context, then execution.
A preconstructed config object has already undergone schema validation;
do not imply every route performs exactly the same internal calls.
-->

A resolved config remains configuration data. A **run specification** contains the resulting execution instructions; a **run context** identifies the run and its output directories. Builders and workflow execution use these objects to construct components and perform the work. Their implementation belongs in [Anatomy of a workflow](../internals/workflow-anatomy.md).

## Omitted, null, and empty are different inputs

These states express different choices; their meaning depends on the field. An empty collection explicitly supplies no entries. `null` supplies a null value, which a field may reject or interpret as inheritance, disabling, or another documented behavior. Omission uses the field's default or omission rule.

For prediction's `transform_pipelines`, when using a config-built datamodule:

| Supplied value | Interpretation |
| --- | --- |
| Field omitted | Inherit applicable validation-targeted transforms from training, if available |
| `null` | Same behavior as omission for this field |
| `[]` | Apply no configured transform pipelines |
| A nonempty list | Use the supplied pipelines instead of inherited ones |

This equivalence between omission and null is specific to this field. The Reference documents **If omitted** and **If null** separately for each field.

## Defaults and inherited settings

A **schema default** supplies an omitted value without consulting another run. A **resolved value** can depend on other sections, the selected family, or upstream records.

Prediction illustrates the difference: its config controls inference and exports, while selected settings can inherit from training. This is explicit field-level behavior, not a general merge of the training and prediction YAMLs. See [Loading models for prediction](../prediction/loading-models.md).

## Editing a config in Python

[`update_config()`][benchrep.update_config] returns a validated copy after changing existing field paths:

```python
from benchrep import update_config
from benchrep.examples import load_example_config

base_config = load_example_config("01_yaml_pipeline", stage="training")
training_config = update_config(
    base_config,
    {"model.params.latent_dim": 64},
)
```

The base remains reusable. This updates a field within the existing configuration; [section overrides](overrides.md#replace-a-config-section) instead replace a complete section. Neither operation replaces workflow resolution.

## Paths follow their source

Ordinary relative paths supplied in configuration are interpreted from the process's **working directory**, not automatically from the YAML's directory. For example, `root: data/mnist` still refers to your working directory when the YAML came from a packaged example. Fields declared as paths also expand `~` to the home directory.

Artifact paths read from manifests have separate handling: relative recorded paths are resolved against the relevant manifest directory. Checkpoint selectors also have their own rules. See [Manifests and provenance](../results/provenance.md) and [Prediction source fields](../../reference/schemas/prediction.md).

??? note "A valid config is not a completed runtime check"
    Schema validation does not execute your component. Resolution can check declared interfaces and compatibility, but actual data shapes, component execution, and runtime dependencies can still fail later. A config object returned by `load_example_config()` or `update_config()` has passed schema validation, not an entire workflow run.
