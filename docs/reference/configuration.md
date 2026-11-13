# Overrides and config helpers

Import these classes and functions from `benchrep`. Overrides can replace complete config sections, supply model or datamodule classes or instances where supported, and select upstream manifests. `update_config()` replaces individual existing fields in a config object. For examples, see [Configure workflows](../guides/configure-workflows/index.md).

::: benchrep.TrainingOverrides
    options:
      heading_level: 2

::: benchrep.PredictionOverrides
    options:
      heading_level: 2

::: benchrep.EvaluationOverrides
    options:
      heading_level: 2

::: benchrep.update_config
    options:
      heading_level: 2

## Read and save YAML

`load_yaml()` returns a mapping, not a workflow config object. To obtain one, pass that mapping to the matching schema's `model_validate()` method. Workflow entrypoints can also read a YAML path directly.

::: benchrep.assembly.config.load_yaml
    options:
      heading_level: 3

`save_config_yaml()` writes a config object or mapping to an existing directory. It serializes the supplied values; it does not preserve the original YAML's formatting or which default-valued fields were omitted. Unsupported runtime objects become descriptive markers, not reconstructible Python objects.

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

