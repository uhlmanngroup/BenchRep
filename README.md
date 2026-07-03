## Framework Structure

BenchRep is an early-stage framework for benchmarking representation-learning models on microscopy-like image data. It keeps simple experiments runnable from config files, while still allowing advanced users to bypass parts of the configuration system and provide instantiated models, datamodules, or components directly.

High-level workflow:

```text
YAML config and/or Pydantic config objects (under dev) and/or instantiated components
    ↓
raw dict validation
    ↓
Pydantic workflow config objects
    ↓
registries, resolvers, and builders
    ↓
train / predict / evaluate workflows
    ↓
checkpoints, manifests, embeddings, reconstructions, metrics, and plots
```

Current scope includes config-driven and Python-driven training, prediction, and evaluation workflows for autoencoder-style representation learning. BenchRep currently supports Autoencoder and VAE models; MLP, CNN, and torchvision-ResNet encoders; MLP and upsample-conv decoders; weighted reconstruction and regularization losses; checkpoint and manifest bookkeeping; embedding and reconstruction export; and downstream embedding/reconstruction evaluation. The framework is under active development, and public APIs should be treated as unstable.

---

## Configuration Layer

Experiments can be defined through a YAML config file.

```text
config.yaml
    ↓
load_yaml_config(...)
    ↓
dict
    ↓
Nested workflow-specific config objects
(e.g. TrainingConfig, PredictionConfig, EvaluationConfig)
```

Everything is then resolved through the registry and the validated config objects are then passed into the builder layer. Currently, the YAML route is the most stable.

---

## Builder Layer

The builder layer task: validated config objects → model and training component objects.

Current major builders:

```text
Data builder
Optimizer builder
Model builder
Trainer builder
```

### Data builder

The built-in data builder currently includes MNIST support only. Workflows can also receive an external Lightning datamodule directly, in which case the dataset/datamodule all their related config sections are treated as overridden/ignored.

```text
DatasetConfig + DataModuleConfig
    ↓
build_datamodule(...): calls builders for Dataset, Transforms, DataModule 
    ↓
LightningDataModule
```

---

### Optimizer factory builder

The optimizer builder creates an optimizer factory (callable) rather than an optimizer instance.

```text
OptimizerConfig
    ↓
build_optimizer_factory(...)
    ↓
callable: model.parameters() → optimizer
```

Used in Lightning model's optimizer configuration step.

---

### Model builder

This is the top level of the hierarchy. Based on the model name (aliases allowed, see `registries/builtins.py`) from the config, it calls a model-specific builder. The currently implemented model families are `Autoencoder` and `VAE`; encoder/decoder components are selected separately through their own registries.

```text
TrainingConfig
    ↓
build_model(...)
    ↓
resolve model name / alias through registry
dispatch by config.model.name
pass encoder / decoder / optimizer / loss configs, or any pre-built components
    ↓
build_autoencoder(...) / build_vae(...): calls Encoder, Decoder, Optimizer, Losses builders and/or assembles if pre-built
    ↓
model-specific LightningModule instance
```

The model-specific builder is responsible for assembling the full model from its components. Builders are intended to support mixed usage: using validated config objects, already-instantiated Python objects, or any combination thereof. Using the builders directly is essentially medium-level API.

---

## Component Layer

Model-specific builders construct reusable components such as encoders, decoders, and losses.

For example, the relevant components for an autoencoder are:

```text
Encoder
Decoder
Loss terms
Weighted loss container
Full Autoencoder LightningModule
```

The loss system supports multiple weighted loss terms, even under a given loss role (e.g. MSE + MAE for reconstruction, and KLD for regularization with global weight coefficients). Loss modules are looked up through the respective registry, instantiated, wrapped in lightweight `LossTerm` dataclasses with their weights, packed in role-specific dictionaries of `LossTerm` objects, and passed to the model. 

Loss building as an example (VAE):

```text
LossTermConfig
    ↓
loss registry: 
    RECONSTRUCTION_LOSSES.create(loss_name, ...)
    REGULARIZATION_LOSSES.create(loss_name, ...)
    ↓
indiv. instantiated loss modules
    ↓
LossTerm(loss, weight), LossTerm(loss, weight)
    ↓
dict[str, LossTerm] passed to the model as reconstruction_losses
dict[str, LossTerm] passed to the model as regularization_losses
```

---

## Trainer Layer

The trainer builder constructs the Lightning trainer, i.e. the orchestrator of the entire Lightning run. Unlike the model
and datamodule, the trainer cannot be passed as a pre-instantiated component (to maintain some measure of reproducibility).

It creates the configured logger and passes it into the Lightning trainer, returning a ready-to-use trainer object.

```text
TrainerConfig + LoggerConfig + RunContext
    ↓
build_trainer(): calls _build_logger() and (not yet implemented) sets up callbacks
    ↓
Trainer(...)
```

Note: BenchRep automatically handles training vs prediction, with the appropriate settings in the trainer.

---

## Miscellaneous

- In addition to the Lightning logger, which is used for training metrics and monitoring, BenchRep writes local run records under `outputs/<run_id>/records/logs/`:
  1. Console stream logs: `stderr.log` is saved by default; `stdout.log` is optional.
  2. Run log: `benchrep_run.log` is written by the `benchrep.run` logger and contains BenchRep-controlled status messages, such as config export, component construction, warnings about various potential issues downstream, workflow updates, and basic sanity checks (e.g. actual instantiated object types).

---

## Usage Modes

BenchRep is intended to support more than one level of usage.

### 1. Config-driven usage

This is the simplest, most stable path: define experiments in YAML and run the workflow entrypoints from Python.

```python
from benchrep.workflows.train import train_vae
from benchrep.workflows.predict import predict_vae
from benchrep.workflows.evaluate import evaluate

train_result = train_vae("examples/configs/training_mnist_vae.yaml")
pred_result = predict_vae(
    "examples/configs/prediction_mnist.yaml",
    training_manifest_path=train_result.manifest_path,
)
eval_result = evaluate(
    "examples/configs/evaluation.yaml",
    prediction_manifest_path=pred_result.manifest_path,
)
```

Automated underlying pipeline:

```text
# Train
1. Load training config → validate dict → compose config → parse
2. Build or accept datamodule
3. Build optimizer factory
4. Build or accept model
   * resolve model name or alias through the registry
   * dispatch to the model-specific builder
   * build encoder
   * build decoder
   * build role-specific loss dicts
   * instantiate the full model
5. Build trainer/logger/checkpointing and run `trainer.fit(model, datamodule)`
6. Write checkpoints, config records, run logs, and training manifest, and audit

# Predict
7. Load prediction config → validate dict → compose config → parse → resolve into a complete run_spec
8. Build or accept prediction datamodule/model
9. Run prediction/inference
10. Export embeddings (AnnData .h5ad), optional reconstructions (.pt), prediction records, and prediction manifest, and audit

# Evaluate
11. Load evaluation config → validate dict → compose config → parse → resolve into a complete run_spec
13. Run downstream evaluation in modular pipeline format
    * reductions/clustering/embedding metrics
    * reconstruction metrics/examples/error maps where available
14. Export evaluation metrics (.json), plots/artifacts, evaluated AnnData (.h5ad), and evaluation records, and audit

```

Useful for reproducible experiments and quick comparisons.

---

### 2. Builder-driven usage

More advanced users can call builders directly from Python: User constructs some components from config, while manually
instantiating other components and passing them to the builders.

Example:

```text
- build the datamodule from config
- manually instantiate a custom encoder, inheriting from BaseEncoder
- build the decoder from config
- build the loss object from config
- pass all components into the model builder
```

Useful for more control without fully leaving the BenchRep framework.

---

### 3. Component-driven usage

The lowest-level usage mode is to import and instantiate components directly, or write them entirely from scratch.
Completely custom models do need to subclass from the BenchRep base models, though those only enforce a predict_step
contract. The train_step and predict_step inputs and outputs are validated only structurally and not nominally. Basically,
any LightningModule and LightningDataModule can work with very few contracts (especially if both are custom).

Caveat is that this route does not allow for the writing of a resolved_config that can be used to reconstruct the run,
at least not without the custom models being passed manually again.

For example, a user may directly use/write:

```text
- an encoder class
- a decoder class
- a loss class
- a model class
- a datamodule
```

Useful for expert users who want BenchRep components (e.g. embedding evaluation) but do not want the full config or builder system. It would also be possible to add the components to the registry to enable later config-driven usage.


NOTE: for users who prefer to code in Python and not YAML, it will soon be possible to pass individual config components
as pydantic objects to the entrypoint workflows, and they would be harmonized with a base YAML if provided to produce a
resolved_config that allows for a fully reconstructable run (as long as no custom model/datamodule were used.) 