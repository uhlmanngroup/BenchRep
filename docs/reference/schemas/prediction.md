# Prediction schemas

Fields for `PredictionConfig`, including checkpoint selection, inference, transforms, and exports. Prediction reuses the [dataset and runtime override schemas](shared.md).

Expand a field to see its type, description, constraints, and omission/null behavior. Defaults shown here are schema declarations; workflow resolution can derive additional settings. See [Reading this reference](index.md#reading-this-reference).

## Workflow configuration

::: benchrep.assembly.schemas.prediction_config_schema.PredictionConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Source and data loading

::: benchrep.assembly.schemas.prediction_config_schema.PredictionSourceConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.prediction_config_schema.PredictionDataConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Inference and composite rewiring

::: benchrep.assembly.schemas.prediction_config_schema.PredictionInferenceConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.prediction_config_schema.CompositeModelAssemblyInputOverrideConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Transforms

::: benchrep.assembly.schemas.prediction_config_schema.PredictionTransformPipelineConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.prediction_config_schema.PredictionTransformStepConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Exports

::: benchrep.assembly.schemas.prediction_config_schema.PredictionExportConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.prediction_config_schema.PredictionAnnDataExportConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.prediction_config_schema.PredictionReconstructionsExportConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.prediction_config_schema.PredictionReconstructionPairConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Field types

These aliases define allowed values, constrained types, or alternative schemas. They are types, not additional YAML sections. Validator names in `Annotated` declarations describe validation hooks, not configuration arguments.

::: benchrep.assembly.schemas.prediction_config_schema.PredictionOutputKey
    options:
      heading_level: 3

::: benchrep.assembly.schemas.prediction_config_schema.PredictionReconstructionPairId
    options:
      heading_level: 3

::: benchrep.assembly.schemas.prediction_config_schema.PredictionReconstructionFieldName
    options:
      heading_level: 3
