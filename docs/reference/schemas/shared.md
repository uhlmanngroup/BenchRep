# Shared and composite schemas

Reusable schema contracts, datasets, runtime override requirements, and composite model configuration. Concrete schemas include inherited fields; you do not need to follow the inheritance tree to find them.

Expand a field to see its type, description, constraints, and omission/null behavior. Defaults shown here are schema declarations; workflow resolution can derive additional settings. See [Reading this reference](index.md#reading-this-reference).

## Named components

`NamedConfig` is defined in the module shown below rather than re-exported from `benchrep.assembly.schemas`.

::: benchrep.assembly.schemas.training_config_schema.NamedConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Runtime override requirements

::: benchrep.assembly.schemas.runtime_override_config_schema.RuntimeOverridesConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.runtime_override_config_schema.RuntimeComponentOverrideConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Datasets

::: benchrep.assembly.schemas.training_config_schema.DatasetConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.MNISTDatasetConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.MNISTDatasetParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.CIFAR10DatasetConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.CIFAR10DatasetParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.STL10DatasetConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.STL10DatasetParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.CustomDatasetConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Composite model declarations and assembly

::: benchrep.assembly.schemas.composite_model_config_schema.CompositeModelDeclarationsConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.composite_model_config_schema.CompositeModelNamedConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.composite_model_config_schema.CompositeModelComponentConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.composite_model_config_schema.CompositeModelAssemblyStepConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Evaluation base schemas

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationStepConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationMetricGroupConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Field types

These aliases define allowed values, constrained types, or alternative schemas. They are types, not additional YAML sections. Validator names in `Annotated` declarations describe validation hooks, not configuration arguments.

::: benchrep.assembly.schemas.training_config_schema.Float32MatmulPrecision
    options:
      heading_level: 3

::: benchrep.assembly.schemas.training_config_schema.SupportedDatasetConfig
    options:
      heading_level: 3

::: benchrep.assembly.registries.utils.PathInput
    options:
      heading_level: 3

::: benchrep.architecture.composite_model_roles.CompositeModelInputRole
    options:
      heading_level: 3

::: benchrep.architecture.composite_model_roles.CompositeModelBatchMetadataRole
    options:
      heading_level: 3

::: benchrep.architecture.composite_model_roles.CompositeModelOutputRole
    options:
      heading_level: 3

::: benchrep.architecture.composite_model_roles.CompositeModelComponentKind
    options:
      heading_level: 3
