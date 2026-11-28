# Configure workflows

<strong class="highlight">What are the different ways to create or edit a workflow configuration?</strong> BenchRep supports editing YAML, constructing configs in Python, replacing sections through overrides, and configuring workflows interactively in the browser.

The snippets show four ways to change the [first tutorial’s](../first-workflow-chain.md) VAE latent dimension from 32 to 64. Each route also links to a complete runnable example with its own settings.

| Route | Configure the run by…                               |
|---|-----------------------------------------------------|
| [YAML](#yaml-files) | Editing a file and passing its path                 |
| [Python config objects](#python-config-objects) | Constructing a config or updating an existing one   |
| [Python overrides](#python-overrides) | Replacing whole sections in a base config           |
| [Browser wizard](#browser-wizard) | Editing fields interactively, reviewing, and saving |

Workflow entrypoint functions, such as [`train_vae()`][benchrep.train_vae], accept a YAML path or config object through their `config` argument.

<strong class="highlight">Requirements.</strong> The YAML and Python snippets need the core installation; the wizard needs `gui`. The complete example chains also use `scverse`. See [Installation](../installation.md) to choose extras. The full composite wizard example also needs `artifact_generation` for its custom dataset and `xgboost` for an evaluation probe. On macOS, see [Known issues](../../known-issues.md#xgboost-on-macos) for the workaround.

## YAML files

The complete YAML-based example uses the original 32-dimensional latent space.

[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/01_yaml_pipeline.py){ .md-button }
[:material-file-cog-outline: View configs](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/configs/01_yaml_pipeline){ .md-button }
{: .br-tutorial-resources }

Copy the packaged training YAML to your working directory:

```python
from pathlib import Path
from shutil import copyfile

from benchrep.examples import example_config_path

training_yaml = Path("configs/mnist-vae-64/training.yaml")
training_yaml.parent.mkdir(parents=True, exist_ok=True)
copyfile(
    example_config_path("01_yaml_pipeline", stage="training"),
    training_yaml,
)
```

Change the existing `model` section in that file:

```yaml
model:
  name: vae
  params:
    latent_dim: 64
```

Then pass its path to training:

```python
from benchrep import train_vae

training_result = train_vae(config=training_yaml)
```

??? note "Encoder output and latent dimensions"
    Keep `encoder.params.output_dim` at `32`. The variational head maps those 32 features to the mean and log variance of the 64-dimensional latent distribution. BenchRep sets the decoder input dimension to `64` automatically.

Run the complete example.

```bash
python -m benchrep.examples.usage.01_yaml_pipeline
```

## Python config objects

The Python-only example constructs all 3 workflow configs in full, using its original 32-dimensional latent space.

[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/03_config_object_pipeline.py){ .md-button }
[:material-file-cog-outline: View configs](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/03_config_object_pipeline.py#L79-L374){ .md-button }
{: .br-tutorial-resources }

[`load_example_config()`][benchrep.examples.load_example_config] reads a packaged YAML into its workflow config class. Use [`update_config()`][benchrep.update_config] to replace individual fields and return a validated copy:

```python
from benchrep import TrainingConfigContext, train_vae, update_config
from benchrep.examples import load_example_config

base = load_example_config("01_yaml_pipeline", stage="training")
context = TrainingConfigContext(external_model=False, external_datamodule=False)
config = update_config(base, {"model.params.latent_dim": 64}, context=context)

print(base.model.params["latent_dim"])    # 32
print(config.model.params["latent_dim"])  # 64

training_result = train_vae(config=config)
```

`update_config()` requires the matching stage-specific context even for a
self-contained config. Both flags are false here because the model and
datamodule are built from configuration. Reuse the same context when making
further edits. Prediction and evaluation have their own context classes; see
[Validation context](../../reference/configuration.md#validation-context).

??? tip "Local YAML?"
    To start from your own training YAML, replace the `base = load_example_config(...)` line with:

    ```python
    import yaml
    from benchrep.assembly.schemas import TrainingConfig

    with open("configs/mnist-vae-64/training.yaml", encoding="utf-8") as stream:
        base = TrainingConfig.model_validate(yaml.safe_load(stream))
    ```

You can also construct [`TrainingConfig`][benchrep.assembly.schemas.training_config_schema.TrainingConfig], [`PredictionConfig`][benchrep.assembly.schemas.prediction_config_schema.PredictionConfig], or [`EvaluationConfig`][benchrep.assembly.schemas.evaluation_config_schema.EvaluationConfig] entirely in Python. For example, a typed model section is constructed as:

```python
from benchrep.assembly.schemas import TrainingModelConfig

model = TrainingModelConfig(name="vae", params={"latent_dim": 64})
```

Pass that section as `model=model` when constructing [`TrainingConfig`][benchrep.assembly.schemas.training_config_schema.TrainingConfig], together with the other required sections.

Run the complete example.

```bash
python -m benchrep.examples.usage.03_config_object_pipeline
```

See [Configuration semantics](../../concepts/workflows/configuration.md) for field updates, defaults, and validation.

## Python overrides

The complete overrides example adapts the [MNIST](http://yann.lecun.com/exdb/mnist/) pipeline to [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) by replacing the dataset and architecture sections.

[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/02_yaml_override_pipeline.py){ .md-button }
[:material-file-cog-outline: View configs](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/configs/01_yaml_pipeline){ .md-button }
{: .br-tutorial-resources }

To replace an entire section while keeping the base YAML, pass a workflow-specific overrides object:

```python
from benchrep import TrainingOverrides, train_vae
from benchrep.assembly.schemas import TrainingModelConfig
from benchrep.examples import example_config_path

model = TrainingModelConfig(
    name="gaussian_vae",
    params={"latent_dim": 64},
)

training_result = train_vae(
    config=example_config_path("01_yaml_pipeline", stage="training"),
    overrides=TrainingOverrides(config_components={"model": model}),
)
```

The supplied section replaces the <strong class="highlight">entire `model` section</strong>. Omitted fields take their schema defaults; required fields still need values. Here, `gaussian_vae` is an alias for `vae`, so it matches the family expected by [`train_vae()`][benchrep.train_vae].

The base can also be a config object. You can omit `config` if `config_components`, together with defaults, supplies a valid workflow config. Manifest paths and supported runtime model/datamodule implementations can also be supplied through overrides; see [Python overrides](../../concepts/workflows/overrides.md).

Run the complete example.

```bash
python -m benchrep.examples.usage.02_yaml_override_pipeline
```

## Browser wizard

The complete composite example configures the workflow chain through 3 wizard sessions. It uses a multi-input model and a custom dataset.

[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/08B_dual_input_multitask_composite_vae_wizard.py){ .md-button }
[:material-file-cog-outline: View configs](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/configs/08_dual_input_multitask_composite_vae_pipeline){ .md-button }
{: .br-tutorial-resources }

Launch the training wizard with the packaged YAML as a starting point:

```python
from benchrep import configure_workflow, train_vae
from benchrep.examples import example_config_path
from benchrep.interfaces.model_families import VAE_FAMILY

wizard_result = configure_workflow(
    stage="training",
    model_family=VAE_FAMILY,
    config=example_config_path("01_yaml_pipeline", stage="training"),
    export_directory="wizard_configs/mnist-vae-64",
)

if wizard_result is not None:
    training_result = train_vae(
        config=wizard_result.config,
        # Forward any overrides still needed at runtime; empty in this example.
        overrides=wizard_result.overrides,
    )
```

??? tip "Starting without a base config"
    Omit `config` to start from scratch. With no arguments, `configure_workflow()` opens the Composite training wizard. Set `stage` to configure prediction or evaluation, and use `model_family` to choose a model family for training or prediction.

The launcher opens your browser at <http://127.0.0.1:8050> by default; pass `port=...` to change it. The Python call waits while you edit and returns when you save or cancel. The launcher's `model_family` fixes the family, even if the base YAML names another one.

Select **Model**, change **Latent Dim** to `64`, and inspect **Preview YAML draft**. The **Documentation** panel shows schemas and registered components. The screenshot shows the starting configuration, before the edit.

<figure>
  <a class="br-image-light" href="../../assets/configuration-wizard-light.png">
    <img src="../../assets/configuration-wizard-light.png" alt="Training wizard showing the VAE model section and documentation panel.">
  </a>
  <a class="br-image-dark" href="../../assets/configuration-wizard-dark.png">
    <img src="../../assets/configuration-wizard-dark.png" alt="Training wizard showing the VAE model section and documentation panel.">
  </a>
  <figcaption>Click the screenshot to view it at full resolution.</figcaption>
</figure>

Open **Review & finish**, click **Check configuration**, then **Save & finish**. Review performs workflow-level validation and resolution, including known dependency checks. Saving writes authored and resolved YAMLs and returns the config to your script; **Cancel** returns `None`.

The wizard accepts an `overrides` argument and returns the overrides still needed for execution. This also lets you pass upstream manifest paths when configuring prediction or evaluation. See [Configuration wizards](../../concepts/workflows/wizards.md) for editing states, local validation, review, and returned values.

Run the complete example.

```bash
python -m benchrep.examples.usage.08B_dual_input_multitask_composite_vae_wizard
```

## Check the effective settings

After any of the training snippets, inspect the saved resolved config:

```python
print(training_result.run_context.config_dir / "resolved_config.yaml")
```

Confirm that `model.params.latent_dim` is `64`. Prediction and evaluation can reuse the packaged YAMLs and manifest-linking code from the [first workflow chain](../first-workflow-chain.md), regardless of how you configured training.
