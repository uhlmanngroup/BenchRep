# Supply overrides from Python

The [Python config objects guide](python-configs.md) changed the MNIST VAE’s latent dimension from 32 to 64 with `update_config()`. Here you’ll make the same change by passing a replacement `model` section alongside the original training YAML. This is useful when you want to reuse a base config and choose a whole section in Python for each run.

<strong class="highlight">Requirements.</strong> This training example uses the core BenchRep installation.

## Replace a config section

Build the replacement section and pass it to `train_vae()` through `TrainingOverrides`:

```python
from benchrep import TrainingOverrides, train_vae
from benchrep.assembly.schemas import TrainingModelConfig
from benchrep.examples import example_config_path

model = TrainingModelConfig(
    name="gaussian_vae",
    params={"latent_dim": 64},
)

overrides = TrainingOverrides(config_components={"model": model})

training = train_vae(
    config=example_config_path("01_yaml_pipeline", stage="training"),
    overrides=overrides,
)
```

The `"model"` key in `config_components` tells BenchRep which section to replace. The supplied `TrainingModelConfig` replaces the <strong class="highlight">entire `model` section</strong> in the YAML. Omitted fields take their schema defaults, even if the YAML specified a different value. You must supply all required fields.

Here, `gaussian_vae` is a registered alias for `vae`. Either name works because both select the family expected by `train_vae()`.

!!! tip align-left "Discover names and fields"
    To find aliases such as `gaussian_vae`, use `inspect_registry("model", "vae")`. See [Registries and discovery](../usage/discovery.md).

    To check which fields `TrainingModelConfig` accepts and which are required, use `inspect_config(TrainingModelConfig)`. See the [Schema reference](../usage/schema-reference.md).

    Both helpers are available from `benchrep`.

## Choose the base config

The same `overrides` object also works with a typed config as the base. For example, you can pass the `TrainingConfig` returned by `load_example_config()` in place of the YAML path. You can even omit `config` entirely if the sections in `config_components`, together with schema defaults, supply everything the workflow requires.

You’ve already used manifest overrides in the [first workflow chain](first-workflow-chain.md). You passed the training manifest to prediction to select the run to load, then passed the prediction manifest to evaluation to select the outputs to score.


!!! note "Supplying a model or datamodule"
    The `model` config section above describes the model BenchRep will build. To supply your own model or datamodule, pass a class or instance through `TrainingOverrides` or `PredictionOverrides`.

    - `model` is available for the 2 canonical families. Use `BenchRepAutoencoderModel` as the base class for an autoencoder or `BenchRepVAEModel` for a VAE.
    - `datamodule` accepts a `LightningDataModule` subclass or instance for all 3 families.

    When passing a class, you can supply its constructor arguments in the config. See [Overrides and runtime callbacks](../usage/runtime-inputs.md) for details.
Next: [Use the configuration wizard](wizard.md).
