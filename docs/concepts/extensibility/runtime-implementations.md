# External models and datamodules

Runtime overrides let you supply Python implementations where selecting registered components is insufficient. For example, you may need a custom sampling policy, a private batch format, or a model with its own training logic. BenchRep retains the surrounding workflow while handing responsibility for the replaced part to your implementation.

The override arguments, class/instance distinction, and bypassed configuration sections are summarized in [Python overrides](../workflows/overrides.md#supply-an-implementation). This page explains the interfaces and execution behavior behind them.

## Supply a model

Whole-model overrides are currently supported by the canonical autoencoder and VAE entrypoints. The implementation must inherit the corresponding [`BenchRepAutoencoderModel`][benchrep.interfaces.models.BenchRepAutoencoderModel] or [`BenchRepVAEModel`][benchrep.interfaces.models.BenchRepVAEModel], both of which extend Lightning's `LightningModule`.

Your implementation owns the Lightning hooks needed for the run: training and validation behavior, optimizer configuration, and prediction. BenchRep does not attach the configured encoder, decoder, losses, or optimizer to an externally supplied model. You can also subclass the concrete [`Autoencoder`][benchrep.architecture.models.autoencoder.Autoencoder] or [`VAE`][benchrep.architecture.models.vae.VAE] to reuse their implementation.

The model's `predict_step()` must return the family's prediction output: [`AutoencoderPredictionOutput`][benchrep.interfaces.contracts.AutoencoderPredictionOutput] or [`VAEPredictionOutput`][benchrep.interfaces.contracts.VAEPredictionOutput], or a structurally compatible dataclass. These carry the representation, input and reconstruction tensors, family-specific latent values, and optional annotations. They are distinct from the dictionary returned by a built-in model's `forward()`.

Because workflow exports depend on that output interface, even training with an external model checks the declared return annotation of `predict_step()`. Matching a family base class alone does not guarantee that the actual outputs are compatible.

## Supply a datamodule

All 3 families accept a `LightningDataModule` subclass or instance. It need not inherit [`BenchRepDataModule`][benchrep.architecture.data.datamodule.BenchRepDataModule]. It owns dataset preparation, splitting, transforms, sampling, collation, and dataloader construction.

Implement the loaders required by the requested workflow and any `prepare_data()` or `setup()` behavior they need. Training needs a training loader; validation is needed when your training setup uses it. Prediction needs a prediction loader. Lightning calls the datamodule hooks during workflow execution.

If BenchRep constructs the model, the datamodule must supply compatible batches: canonical models expect the `x` field, while composite models require their declared inputs and metadata. If you supply both a canonical model and datamodule, their internal batch format can be private to your implementation; the prediction output must still satisfy the family's contract.

See [Datasets and sample structure](../data/datasets.md) and [Splits and dataloaders](../data/datamodules.md) for the config-built data path and its interfaces.

## Classes and instances

An override instance has already been constructed. BenchRep uses that object directly, so leave its constructor `params` empty. A class is instantiated using the matching config section:

```yaml
overrides:
  model:
    params:
      latent_dim: 32
```

This fragment assumes the supplied model class accepts `latent_dim`. The class itself is passed through [`TrainingOverrides(model=MyModel)`][benchrep.TrainingOverrides] or [`PredictionOverrides(model=MyModel)`][benchrep.PredictionOverrides]; the YAML only supplies constructor arguments. Datamodule classes use `overrides.datamodule.params` in the same way.

For an already constructed pair, the call has this form:

```python
from benchrep import TrainingOverrides, train_ae

training_result = train_ae(
    config=training_config,
    overrides=TrainingOverrides(
        model=model,
        datamodule=datamodule,
    ),
)
```

Here, `model`, `datamodule`, and `training_config` are prepared by your code. The workflow still uses the applicable trainer, logging, checkpoint, callback, and record settings. An external model must log whichever metrics its checkpointing and early-stopping settings monitor.

## Compatibility checks and runtime checks

BenchRep distinguishes Python interface checks from checking real prediction outputs:

| Situation | Compatibility checks |
| --- | --- |
| External canonical model | Correct family base class; compatible `predict_step()` return annotation |
| External model with config-built data | Additionally checks its batch annotations: `training_step()` and `predict_step()` for training, or `predict_step()` for prediction |
| External model and datamodule together | Their private batch format is not checked against the canonical batch annotation |
| External datamodule with an internal canonical model | Model execution must accept the supplied batches; failures are reported with the expected batch contract |

The canonical entrypoints accept `compatibility_policy="error"` by default. Setting it to `"warn"` lets supported **annotation sanity-check failures** become recorded warnings. It does not waive the required model base class, configuration validation, or checks on actual prediction results.

Composite models check their declared batch fields and tensor structures during model execution rather than using these canonical annotation checks.

Annotation checks compare the declared fields and types; they do not execute a representative batch. After prediction, BenchRep checks the returned objects structurally, then export code checks the values it needs. Subclassing the provided prediction dataclasses is convenient, but a separate dataclass with the required compatible fields is also accepted. See the [batch and output reference](../../reference/interfaces.md#batch-and-output-contracts).

## Reconstruct the implementation for prediction

An external model is not serialized as executable Python code in a training manifest. Supply a compatible model class or instance again when predicting from that run. For a class, constructor parameters can be inherited from the recorded training override settings when prediction does not supply its own.

The current prediction workflow still requires the training manifest, selects its checkpoint, and loads the checkpoint's state dictionary into the supplied model. That model must therefore have compatible parameter names and shapes. Supplying an instance does not skip checkpoint loading or turn prediction into an unrelated pretrained-model inference call.

An external training datamodule is likewise not recreated from its recorded class name. Prediction can use a new datamodule override, or explicit dataset and transform settings that provide compatible data. The same training split is not automatically recovered. See [Loading models for prediction](../prediction/loading-models.md) and [Manifests and provenance](../results/provenance.md).

## Implementation boundary

Workflow composition records whether each implementation comes from configuration, a Python class, or an instance. Resolution removes the construction settings bypassed by the override and resolves any constructor parameters. The runtime-component builder instantiates a class or returns an instance unchanged; the normal model/data builders are used only for config-built components.

The workflow then performs applicable compatibility checks and passes the resulting model and datamodule to Lightning. This keeps run management independent of who built them. The shared sequence is described in [Workflow implementation](../workflows/workflow-anatomy.md); the relevant construction code is in the [runtime override builder](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/runtime_component_override_builder.py).
