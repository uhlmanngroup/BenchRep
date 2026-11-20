# Loading models for prediction

Prediction needs both a model implementation and trained parameter values. BenchRep obtains the construction settings from a training manifest, recreates the model or accepts a runtime override, and loads the selected checkpoint into it.

## Select the training run

The prediction entrypoint must match the family recorded by training. For example:

```python
from benchrep import PredictionOverrides, predict_vae

prediction_result = predict_vae(
    config="prediction.yaml",
    overrides=PredictionOverrides(
        training_manifest_path=training_result.manifest_path,
    ),
)
```

The manifest path override takes precedence over the config's source path. The resolver reads the resolved training configuration embedded in `appendix.resolved_config`; the separately saved training YAML is retained as a record, rather than reread as the source of construction settings.

For config-built models, the recorded architecture and losses are reconstructed so the new model has the structure needed to load its checkpoint. Loss modules may themselves have parameters, even though prediction does not evaluate the losses. External component registrations must therefore be available in the prediction process as well. See [Registries and discovery](../extensibility/discovery.md).

## Select a checkpoint

```yaml
source:
  training_manifest_path: outputs/training/my-run/records/metadata/training_manifest.yaml
  checkpoint: best
```

The path above is illustrative; use the actual manifest path returned or printed by training.

| `checkpoint` | Selection |
| --- | --- |
| `best` | The best checkpoint recorded in the training manifest; the default |
| `last` | The recorded last checkpoint |
| A bare filename ending in `.ckpt` | That file within the training run's checkpoint directory |
| An absolute `.ckpt` path | That explicit checkpoint file |

The selected file must exist. There is no automatic fallback from a missing best checkpoint to last. An explicit checkpoint path still uses the linked training run for configuration and family compatibility; it does not turn prediction into a standalone model loader.

## Supply runtime implementations

Training with an external canonical model requires a compatible model override for prediction. BenchRep cannot recreate its Python implementation from a class name recorded in YAML. Supply its class or instance, with any needed class-constructor parameters under the config's `overrides.model.params`.

<strong class="highlight">A supplied model instance still receives the selected checkpoint's weights.</strong> Prediction does not simply use whatever weights the instance already holds. Its state structure must be compatible with the checkpoint. Whole-model overrides are currently supported only for the canonical families; datamodule overrides are supported for all 3 families.

An external training datamodule does not necessarily require an external prediction datamodule: explicit prediction dataset settings can provide a config-built data path instead. A prediction datamodule override bypasses dataset construction, transforms, and ordinary loader settings, while inference/export settings and `data.max_batches` still apply. See [Python overrides](../workflows/overrides.md#supply-an-implementation).

## Choose inference data and behavior

When not replaced, applicable settings are inherited from the recorded training config:

| Setting | Prediction behavior |
| --- | --- |
| Dataset | An explicit prediction dataset replaces the training selection; otherwise the config can be inherited |
| Transforms | Omitted/null inherits training's validation-applicable processing; `[]` disables it; an explicit list replaces it |
| Loader settings | Batch size and worker count can be overridden; prediction disables validation splitting and incomplete-batch dropping |
| Reproducibility | Seed, worker seeding, determinism, and matrix-multiplication precision inherit unless explicitly set |
| Trainer | General training Trainer settings are reused, with prediction logging/checkpointing disabled and training callbacks omitted |

Inheriting a dataset config does not recover the exact validation subset. Prediction constructs a dataset independently and reads its selected split. See [Prediction data](../data/datamodules.md#prediction-data) and [Transform inheritance](../data/transforms.md#prediction-inherits-validation-processing).

Internally built VAEs decode the posterior mean by default during prediction; `inference.canonical_vae_reconstruction_latent_source: sample` selects sampled latents instead. Composite models can reroute selected assembly inputs, such as a decoder's input from a latent sample to its mean. These are family-specific controls, described in [Canonical models](../models/canonical.md) and [Composite models](../models/composite.md).

## Implementation: reconstruct, load, predict

The [prediction resolver](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/prediction_config_resolver.py) combines the source manifest and prediction settings into a `PredictionRunSpec`. This includes the concrete checkpoint path, effective data/inference settings, optional composite specification, and an export plan. Inherited values are recorded before they are materialized into the resolved config.

The workflow then:

1. Constructs the datamodule and model, or instantiates the supplied override classes.
2. Checks applicable runtime contracts and loads the checkpoint's `state_dict` with `model.load_state_dict(...)`.
3. Sets the model to evaluation mode and builds a prediction Trainer.
4. Calls `Trainer.predict(..., return_predictions=True)`, which invokes the model's `predict_step()` for each batch.
5. Validates the collected outputs and passes them to the [exporter](exports.md#implementation-from-prediction-batches-to-artifacts).

Checkpoint loading restores model state, not a running training loop: optimizer progress is not resumed. Prediction uses the model's prediction hook and output contract, which can differ from calling `forward()` directly.

!!! warning "Export selection does not limit inference"
    The workflow collects prediction batches before exporting. Selecting fewer exported keys or reconstruction examples does not skip forward branches or reduce how many batches are inferred. Use `data.max_batches` for a diagnostic limit, and account for the memory needed to retain predictions.
