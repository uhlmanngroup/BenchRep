## Framework Structure

BenchRep (very tentative) is a layered builder-based workflow that allows simple experiments to be runnable from config files, while still allowing expert users to bypass parts of the configuration system and provide instantiated components directly.

High-level workflow:

```text
YAML config
    ↓
raw dict validation
    ↓
Pydantic config objects
    ↓
builders
    ↓
dataset / datamodule / model / loss / optimizer / trainer
    ↓
Lightning training run
```

Currently, vertical slice is nearly complete for MLP autoencoder runs on MNIST with two reconstruction losses. The same structure is intended to generalize to VAEs, contrastive models, supervised models, more complex encoder/decoder archs, additional dataset types (individual samples, and OME-Zarr), other losses, and thorough downstream evaluation.

---

## Configuration Layer

Experiments can be defined through a YAML config file.

The YAML is loaded with `pyyaml`, checked to be a `dict`, and then parsed into validated `Pydantic` config objects.

Conceptually:

```text
config.yaml
    ↓
load_yaml_config(...)
    ↓
dict
    ↓
Nested objects under BenchRepConfig
```

For example, the config may specify:

```text
reproducibility: ...

run: ...

dataset:
  name: ...
  transform: ...

datamodule:
  batch_size: ...
  val_fraction: ...

model:
  name: autoencoder

encoder:
  name: mlp
  params: ...

decoder:
  name: mlp
  params: ...

losses:
  reconstruction:
    mse: ...
    mae: ...

optimizer: ...

trainer: ...

logger: ...
```

The validated config objects are then passed into the builder layer.

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

The data builder currently supports MNIST. It creates the dataset object internally and uses it to instantiate the Lightning datamodule.

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

The model receives this optimizer factory and uses it during Lightning’s optimizer configuration step.

---

### Model builder

The model builder is in the top level of the hierarchy. Based on the model name from the config, it calls a model-specific builder (currently only the autoencoder is supported).

```text
BenchRepConfig
    ↓
build_model(...)
    ↓
dispatch by config.model.name
pass encoder / decoder / optimizer / loss configs, or any pre-built components
    ↓
build_autoencoder(...): calls Encoder, Decoder, Optimizer, Losses builders and/or assembles if pre-built
    ↓
Autoencoder model instance
```

The model-specific builder is responsible for assembling the full model from its components.

---

## Component Layer

Model-specific builders construct reusable components such as encoders, decoders, and losses.

For the current autoencoder slice, the relevant components are:

```text
Encoder
Decoder
Loss terms
Weighted loss container
Full Autoencoder LightningModule
```

The encoder and decoder are selected and constructed based on the config and registry.

The loss system supports multiple weighted loss terms. Loss terms are looked up through the registry, and packed into a higher-level loss object that the model can call during training.

Loss building as an example:

```text
LossesConfig
    ↓
loss registry: RECONSTRUCTION_LOSSES.create(loss_name, ...)
    ↓
indiv. loss functions/modules
    ↓
LossTerm objects: LossTerm(loss, weight)
    ↓
combined weighted loss object
```

Note: it's possible to have more than one loss under the same model role (e.g. reconstruction), each with its own weight.

---

## Trainer Layer

The trainer builder constructs the Lightning trainer, i.e. the orchestrator of the entire Lightning run.

It creates the configured logger and passes it into the Lightning trainer, returning a ready-to-use trainer object.

```text
TrainerConfig + LoggerConfig + RunContext
    ↓
build_trainer(): calls _build_logger() and (not yet implemented) sets up callbacks
    ↓
Trainer(...)
```

The final training call then looks like:

```text
trainer.fit(model, datamodule)
```

Note: Trainer would also be used for prediction or testing (not yet implemented)

---

## Usage Modes

BenchRep is intended to support more than one level of usage.

### 1. Config-driven usage

This is the simplest path: User defines an experiment in YAML and runs it through the main entrypoint (CLI).

```bash
python -m benchrep.workflows.training --config examples/configs/mnist_autoencoder.yaml
```

Automated underlying workflow:

```text
1. Load YAML config
2. Validate raw config as a dictionary
3. Parse into Pydantic config objects
4. Build the datamodule
5. Build the optimizer factory
6. Build the model
    - dispatch to the autoencoder builder
    - build encoder
    - build decoder
    - build weighted loss object
    - instantiate the full autoencoder model
7. Build the trainer and logger
8. Run trainer.fit(model, datamodule)
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

The lowest-level usage mode is to import and instantiate components directly.

For example, a user may directly use:

```text
- an encoder class
- a decoder class
- a loss class
- a model class
- a datamodule
```

Useful for expert users who want BenchRep components (e.g. embedding evaluation) but do not want the full config or builder system. It would also be possible to add the components to the registry to enable later config-driven usage.


