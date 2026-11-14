# Python overrides

Overrides let you reuse an experiment without repeatedly editing its YAML. You can replace a section for another trial or pass the manifest produced by the preceding workflow directly into the next call. They also provide an **escape hatch for Python implementations** that cannot be fully described by configuration.

These uses share an argument, but act at different points: configuration overrides change the description BenchRep will interpret; runtime overrides supply something it would otherwise build.

## What an override changes

| Mechanism | Purpose | What BenchRep does |
| --- | --- | --- |
| `config_components` | Reuse a base config with selected sections replaced | Compose the replacement sections into the effective config |
| Upstream manifest path | Connect runs programmatically | Use the supplied path instead of the configured source path |
| Model or datamodule | Supply a runtime implementation | Use the supplied instance, or instantiate the supplied class, instead of the corresponding config-built implementation |

Use [`TrainingOverrides`][benchrep.TrainingOverrides], [`PredictionOverrides`][benchrep.PredictionOverrides], or [`EvaluationOverrides`][benchrep.EvaluationOverrides] to match the workflow. Their Reference entries list the accepted fields.

## Replace a config section

`config_components` maps a top-level section name to its replacement. The base can be a YAML path or a Python config object:

```python
from benchrep import TrainingOverrides, train_vae
from benchrep.assembly.schemas import TrainingOptimizerConfig
from benchrep.examples import example_config_path

training_result = train_vae(
    config=example_config_path("01_yaml_pipeline", stage="training"),
    overrides=TrainingOverrides(
        config_components={
            "optimizer": TrainingOptimizerConfig(
                name="adam",
                params={"lr": 3e-4},
            ),
        },
    ),
)
```

The supplied optimizer section **replaces the whole original section**. It is not recursively merged with it. Other top-level sections remain supplied by the base config.

Suppose the original optimizer parameters contain both `lr` and `weight_decay`:

| Operation | Resulting optimizer parameters |
| --- | --- |
| Base config | `lr: 0.001`, `weight_decay: 0.01` |
| `update_config()` changes `optimizer.params.lr` | `lr: 0.0003`, `weight_decay: 0.01` |
| Section override supplies only `params={"lr": 0.0003}` | `lr: 0.0003`; the previous `weight_decay` is not carried over |

Omitted fields in the replacement follow that section's schema defaults; required fields must still be supplied. Component constructor defaults may apply to omitted constructor arguments. Use [`update_config()`][benchrep.update_config] when your intent is to retain a section and change individual existing fields.

<!-- Figure slot: section replacement versus field update. Suggested assets:
../../assets/understanding/workflows/section-replacement-light.svg
../../assets/understanding/workflows/section-replacement-dark.svg
Use the optimizer comparison above: highlight retained weight_decay for a
field update and its absence after replacing the complete optimizer section.
-->

A base `config` is optional when `config_components` and schema defaults provide enough information to construct a valid workflow config. Overrides do not make missing requirements disappear.

## Connect runs without rewriting paths

The result of one workflow supplies the manifest path needed by the next:

```python
from benchrep import EvaluationOverrides, PredictionOverrides

prediction_overrides = PredictionOverrides(
    training_manifest_path=training_result.manifest_path,
)

evaluation_overrides = EvaluationOverrides(
    prediction_manifest_path=prediction_result.manifest_path,
)
```

The result objects come from the preceding workflow calls. Pass each bundle as `overrides` to its matching entrypoint.

A supplied manifest path takes precedence over the source path in the effective config, including one supplied through a replacement source section. The base YAML can therefore remain reusable across runs. See [the workflow chain tutorial](../../guides/first-workflow-chain.md#how-the-workflows-connect).

## Supply an implementation

A runtime override changes **who provides the model or data-loading behavior**. This is broader than selecting another registered component.

| Runtime override | Supported use | Required interface |
| --- | --- | --- |
| Model | Canonical AE/VAE training and prediction | `BenchRepAutoencoderModel` or `BenchRepVAEModel` |
| Datamodule | Training and prediction for all 3 families | `LightningDataModule` |

A whole-model override supplies the Lightning model, including its training and optimizer behavior where applicable. BenchRep does not rebuild that model's architecture or losses from the normal training sections. A datamodule override supplies the workflow's loaders and data preparation instead of the configured dataset/transform/datamodule construction.

The surrounding workflow still handles tasks such as trainer setup, prediction exports, and records. In particular, prediction still selects a checkpoint through its training manifest and loads its weights into the model.

**Pass a class** when BenchRep should construct it using `overrides.model.params` or `overrides.datamodule.params` in the config. **Pass an instance** when you have already constructed it; its constructor `params` must be empty. The Python override object supplies the implementation; the config's `overrides` section supplies constructor requirements and parameters.

See [External models and datamodules](../extensibility/runtime-implementations.md) for their runtime responsibilities, and [Extension interfaces](../../reference/interfaces.md) for the base classes and contracts.

??? note "What stays active with a prediction datamodule override?"
    The supplied datamodule owns its dataset, transforms, batch size, and worker settings. Prediction's inference and export settings still apply, and `data.max_batches` still limits execution. Treat the override as replacing data construction, not the entire prediction configuration.

## Runtime callbacks are separate

Training accepts callback instances through `runtime_callbacks`, independently of its overrides object. This supports callbacks that need live Python objects, such as an Optuna trial. They are added alongside callbacks constructed from configuration. See [Callbacks and checkpoints](../training/callbacks.md).

Registrations and runtime objects belong to the current Python process. Recorded names and settings do not recreate arbitrary Python implementations: import registrations and supply required runtime overrides again when starting another process.
