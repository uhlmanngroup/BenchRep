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

For users who prefer to code in Python and not YAML, it's also
possible to pass a fully constructed top-level Pydantic config object (e.g. TrainingConfig, PredictionConfig) to the train and predict workflows (evaluate workflow in progress) to bypass the YAML-route entirely. BenchRep also allows for passing specific config components (corresponding to YAML sections under the top level Pydantic object) as a mapping together with a base YAML file, which are then harmonized into  a single parsed config object. Precedence rules: full top-level config object > config components > YAML. In all cases, when using a fully config-based approach, whether using YAML or config objects, a resolved config YAML is saved to allow for reconstructing the run easily.

```python
from benchrep.assembly.schemas.training_config_schema import ReproducibilityConfig
from benchrep.workflows.train import train_vae
from benchrep.workflows.predict import predict_vae
from benchrep.workflows.evaluate import evaluate

reproducibility_config = ReproducibilityConfig(
    seed=3407,
    seed_workers=True,
    float32_matmul_precision="medium",
)

train_result = train_vae(
    "examples/configs/training_mnist_vae.yaml",
    config_components={"reproducibility": reproducibility_config},
)
pred_result = predict_vae(
    "examples/configs/prediction_mnist.yaml",
    training_manifest_path=train_result.manifest_path,
)
eval_result = evaluate(
    "examples/configs/evaluation.yaml",
    prediction_manifest_path=pred_result.manifest_path,
)
```

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

Useful for more control without fully leaving the BenchRep framework. However, currently, the export of a resolved YAML
config file that can be used to reconstruct the run is not currently supported for this route. 

---

### 3. Component-driven usage

The lowest-level usage mode is to provide custom Python objects directly to the workflow entrypoints.

For autoencoder-style workflows, custom models should subclass the appropriate BenchRep base model, but the required contract is intentionally tiny. Custom datamodules can be ordinary Lightning datamodules. This route is useful when users want BenchRep’s training, prediction, export, manifest, and evaluation machinery without using the full config/builder system.

Caveat: runs using external Python objects are not fully reconstructable from the resolved config alone. Reproducing the run requires passing the custom model/datamodule code again. In theory, external Python objects can be added to BenchRep's registry to allow for
reconstructable fully config-driven usage later; however, this route is yet fully implemented and would likely result
in auditor errors regardless of runtime success.

Example:

```python
import lightning as L
import torch

from benchrep.interfaces.models import BenchRepAutoencoderModel
from benchrep.workflows.train import train_ae
from benchrep.workflows.predict import predict_ae
from benchrep.workflows.evaluate import evaluate

# predict_step must have structurally compatible output dataclass types, that's all
# but if using a BenchRep DM, predict_step and training_step must also have compatible batch (input) dataclass types.
class MyModel(BenchRepAutoencoderModel):
    ...


class MyDataset(torch.utils.data.Dataset):
    ...


class MyDataModule(L.LightningDataModule):
    ...


model = MyModel(latent_dim=8)
datamodule = MyDataModule(batch_size=64)

train_result = train_ae(
    "examples/configs/training_external_ae.yaml",
    model=model,
    datamodule=datamodule,
    compatibility_policy="warn",
)

pred_result = predict_ae(
    "examples/configs/prediction_external_ae.yaml",
    training_manifest_path=train_result.manifest_path,
    model=model,
    datamodule=datamodule,
    compatibility_policy="warn",
)

eval_result = evaluate(
    "examples/configs/evaluation.yaml",
    prediction_manifest_path=pred_result.manifest_path,
)
```