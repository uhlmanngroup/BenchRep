# Configuration wizards

`configure_workflow()` opens a local browser editor, validates and resolves the draft during review, and returns to the calling script after saving. It requires the `gui` extra. For a walkthrough, see [Configure workflows](../guides/configure-workflows/index.md).

## Launch a wizard

Import the launcher from `benchrep`. With no arguments it opens a training wizard for the composite family. Select a canonical family using the [family constants](interfaces.md#model-families); evaluation does not take a model family.

::: benchrep.configure_workflow
    options:
      heading_level: 3

## Returned configuration

The return type follows the stage:

| Stage | `config` | `overrides` |
| --- | --- | --- |
| Training | `TrainingConfig` | `TrainingOverrides` |
| Prediction | `PredictionConfig` | `PredictionOverrides` |
| Evaluation | `EvaluationConfig` | `EvaluationOverrides` |

Pass both returned objects to the matching [workflow entrypoint](workflows.md). Overrides already incorporated into the authored config are removed from the returned override bundle; runtime objects still needed for execution are retained.

```python
from benchrep import configure_workflow, train_vae
from benchrep.interfaces.model_families import VAE_FAMILY

wizard_result = configure_workflow(
    stage="training",
    model_family=VAE_FAMILY,
)

if wizard_result is not None:
    training_result = train_vae(
        config=wizard_result.config,
        overrides=wizard_result.overrides,
    )
```

::: benchrep.gui.shared.results.ConfiguredWorkflow
    options:
      heading_level: 3

!!! note "Authored and resolved configs"
    Use `config` or the saved authored YAML for subsequent runs. `resolved_config` records the settings derived during review; those settings may depend on the current environment or upstream artifacts. A new workflow resolves its inputs again.
