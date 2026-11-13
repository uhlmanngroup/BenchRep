# Training schemas

Fields for `TrainingConfig` and its training-specific sections. Dataset schemas, runtime override requirements, and composite declarations/components/assembly are documented under [Shared and composite schemas](shared.md).

Expand a field to see its type, description, constraints, and omission/null behavior. Defaults shown here are schema declarations; workflow resolution can derive additional settings. See [Reading this reference](index.md#reading-this-reference).

## Workflow configuration

::: benchrep.assembly.schemas.training_config_schema.TrainingConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Run and reproducibility

::: benchrep.assembly.schemas.training_config_schema.TrainingRunConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingReproducibilityConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Canonical model and architecture

::: benchrep.assembly.schemas.training_config_schema.TrainingModelConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingEncoderConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingDecoderConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Data and transforms

::: benchrep.assembly.schemas.training_config_schema.TrainingDataModuleConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingTransformPipelineConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingTransformStepConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Losses

::: benchrep.assembly.schemas.training_config_schema.TrainingLossTermConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Loss-weight schedules

::: benchrep.assembly.schemas.training_config_schema.TrainingLossWeightScheduleConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.LossWeightScheduleCycleConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.RampScheduleParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.LinearScheduleConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.CosineScheduleConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.SigmoidScheduleParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.SigmoidScheduleConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.ExponentialScheduleParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.ExponentialScheduleConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.PiecewiseScheduleParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.PiecewiseLinearScheduleConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.PiecewiseConstantScheduleConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.CustomScheduleConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Optimization and trainer

::: benchrep.assembly.schemas.training_config_schema.TrainingOptimizerConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingLRSchedulerConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingTrainerConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Logging and callbacks

::: benchrep.assembly.schemas.training_config_schema.TrainingLoggerConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingCheckpointConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingEarlyStoppingConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingAdditionalCallbackConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Inspection

::: benchrep.assembly.schemas.training_config_schema.TrainingInspectionConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.TrainingTorchviewConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.PerLossGradientNormConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.training_config_schema.OptimizerGradientNormConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Field types

These aliases define allowed values, constrained types, or alternative schemas. They are types, not additional YAML sections. Validator names in `Annotated` declarations describe validation hooks, not configuration arguments.

::: benchrep.assembly.schemas.training_config_schema.SupportedLossRole
    options:
      heading_level: 3

::: benchrep.assembly.schemas.training_config_schema.LossTermId
    options:
      heading_level: 3

::: benchrep.assembly.schemas.training_config_schema.StrictPositiveInt
    options:
      heading_level: 3

::: benchrep.assembly.schemas.training_config_schema.SchedulePosition
    options:
      heading_level: 3

::: benchrep.assembly.schemas.training_config_schema.ScheduleFactor
    options:
      heading_level: 3

::: benchrep.assembly.schemas.training_config_schema.SupportedLossWeightScheduleConfig
    options:
      heading_level: 3

::: benchrep.assembly.schemas.training_config_schema.LossRoleTerms
    options:
      heading_level: 3

::: benchrep.assembly.schemas.training_config_schema.GradientNormConfig
    options:
      heading_level: 3
