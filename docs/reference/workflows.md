# Workflow entrypoints

Import these functions from `benchrep`. Each accepts a YAML path or a workflow config object and returns a workflow-specific result. Choose the model family through the training or prediction function; evaluation operates on exported artifacts.

Use [overrides](configuration.md) to replace config sections or supply supported runtime objects.

## Training

::: benchrep.train_ae
    options:
      heading_level: 3

::: benchrep.train_vae
    options:
      heading_level: 3

::: benchrep.train_composite
    options:
      heading_level: 3

## Prediction

::: benchrep.predict_ae
    options:
      heading_level: 3

::: benchrep.predict_vae
    options:
      heading_level: 3

::: benchrep.predict_composite
    options:
      heading_level: 3

## Evaluation

::: benchrep.evaluate
    options:
      heading_level: 3

## Compatibility policy

Training and prediction accept this policy for supported runtime compatibility checks.

::: benchrep.runtime.utils.CompatibilityPolicy
    options:
      heading_level: 3

