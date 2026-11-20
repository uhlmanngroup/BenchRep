# Overrides and config helpers

Import these classes and functions from `benchrep`. Overrides can replace complete config sections, supply model or datamodule classes or instances where supported, and select upstream manifests. [`update_config()`][benchrep.update_config] replaces individual existing fields in a config object. For examples, see [Configure workflows](../guides/configure-workflows/index.md).

::: benchrep.TrainingOverrides
    options:
      heading_level: 2

::: benchrep.PredictionOverrides
    options:
      heading_level: 2

::: benchrep.EvaluationOverrides
    options:
      heading_level: 2

## Validation context

[`update_config()`][benchrep.update_config], the three `parse_*_config()` helpers, and
`compose_effective_config()`
require a `context` matching the workflow config type. Context describes runtime
information that may be absent from the configuration itself. All context fields
are required keyword arguments.

| Config type | Context | Required fields |
| --- | --- | --- |
| `TrainingConfig` | `TrainingConfigContext` | `external_model`, `external_datamodule` |
| `PredictionConfig` | `PredictionConfigContext` | `model_family`, `training_manifest_path_overridden` |
| `EvaluationConfig` | `EvaluationConfigContext` | `prediction_manifest_path_overridden` |

Use false for components built from configuration or manifest paths provided in
the config's `source` section. A manifest override flag is true when the matching
workflow overrides object will supply that path separately. For standalone YOYO
prediction, `training_manifest_path_overridden` is false.

For example, reuse one context while parsing and editing a training config:

```python
from benchrep import TrainingConfigContext, update_config
from benchrep.assembly.config import load_yaml
from benchrep.assembly.schemas import parse_training_config

context = TrainingConfigContext(external_model=False, external_datamodule=False)
config = parse_training_config(load_yaml("training.yaml"), context=context)
edited = update_config(config, {"trainer.max_epochs": 20}, context=context)
```

Canonical autoencoders and VAEs use `AUTOENCODER_FAMILY` and `VAE_FAMILY` for
prediction context. Composite models use `COMPOSITE_FAMILY`; YOYO models use
`YOYO_FAMILY`. Import these constants from `benchrep.interfaces.model_families`.

```python
from benchrep import PredictionConfigContext, update_config
from benchrep.assembly.schemas import parse_prediction_config
from benchrep.interfaces.model_families import YOYO_FAMILY

context = PredictionConfigContext(
    model_family=YOYO_FAMILY,
    training_manifest_path_overridden=False,
)
config = parse_prediction_config({}, context=context)
edited = update_config(config, {"data.max_batches": 1}, context=context)
```

These examples perform schema validation only. Context supplies no model,
datamodule, or manifest, and does not replace runtime overrides or resolution
checks. Non-null training `overrides` declarations can also establish external
component intent during schema validation.

Workflow entrypoints and wizards construct context from the actual family and
overrides. Their public calling conventions are unchanged. Context is not a
config field, is not saved in YAML, and is not retained automatically by Pydantic.
Pass these dataclasses to the BenchRep helpers, which translate them to Pydantic's
internal validation context. A context of the wrong stage raises `TypeError`.

::: benchrep.TrainingConfigContext
    options:
      heading_level: 3

::: benchrep.PredictionConfigContext
    options:
      heading_level: 3

::: benchrep.EvaluationConfigContext
    options:
      heading_level: 3

`ConfigContext` is the union of those three types; construct the stage-specific
class, not the union itself.

::: benchrep.assembly.schemas.parsing.ConfigContext
    options:
      heading_level: 3

## Editing individual fields

::: benchrep.update_config
    options:
      heading_level: 2

## Read and save YAML

[`load_yaml()`][benchrep.assembly.config.load_yaml] returns a mapping, not a workflow config object. To obtain one with runtime context, pass the mapping and the matching context to `parse_training_config()`, `parse_prediction_config()`, or `parse_evaluation_config()` from `benchrep.assembly.schemas`. A self-contained config can also be constructed with its schema's `model_validate()` method. Workflow entrypoints can read a YAML path directly.

::: benchrep.assembly.config.load_yaml
    options:
      heading_level: 3

[`save_config_yaml()`][benchrep.records.configs.save_config_yaml] writes a config object or mapping to an existing directory. It serializes the supplied values; it does not preserve the original YAML's formatting or which default-valued fields were omitted. Unsupported runtime objects become descriptive markers, not reconstructible Python objects.

::: benchrep.records.configs.save_config_yaml
    options:
      heading_level: 3

## Accepted input types

These aliases describe the values accepted by workflow entrypoints and override bundles. A config-component union lists possible section values; the mapping key must still identify the corresponding workflow section.

::: benchrep.utils.workflow_utils.TrainingConfigInput
    options:
      heading_level: 3

::: benchrep.utils.workflow_utils.PredictionConfigInput
    options:
      heading_level: 3

::: benchrep.utils.workflow_utils.EvaluationConfigInput
    options:
      heading_level: 3

::: benchrep.assembly.config.SupportedTrainingConfigComponent
    options:
      heading_level: 3

::: benchrep.assembly.config.SupportedPredictionConfigComponent
    options:
      heading_level: 3

::: benchrep.assembly.config.SupportedEvaluationConfigComponent
    options:
      heading_level: 3

### Section collection types

These aliases describe the list and mapping values used by section overrides.

::: benchrep.assembly.config.LossesConfig
    options:
      heading_level: 3

::: benchrep.assembly.config.CompositeModelComponentsConfig
    options:
      heading_level: 3

::: benchrep.assembly.config.CompositeModelAssemblyConfig
    options:
      heading_level: 3

::: benchrep.assembly.config.TrainingTransformPipelinesConfig
    options:
      heading_level: 3

::: benchrep.assembly.config.PredictionTransformPipelinesConfig
    options:
      heading_level: 3

::: benchrep.assembly.config.AdditionalCallbacksConfig
    options:
      heading_level: 3

