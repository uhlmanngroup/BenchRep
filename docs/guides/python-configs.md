# Python config objects

The [YAML guide](yaml.md) changed the MNIST VAE’s latent dimension from 32 to 64 by editing a file. Here you’ll make the same change in Python and pass the resulting `TrainingConfig` directly to `train_vae()`. This is useful when choosing settings programmatically, such as during a hyperparameter search.

<strong class="highlight">Requirements.</strong> Loading the config and running training require only the core installation. To run the full chain below, install the `scverse` extra for UMAP if you haven’t already:

```bash
python -m pip install "benchrep[scverse] @ git+https://github.com/uhlmanngroup/BenchRep.git"
```

## Load and edit the training config

`example_config_path()` locates a packaged YAML. `load_example_config()` reads it into the corresponding Python config object, so you can inspect its fields or use it as the starting point for another run.

```python
from benchrep import update_config
from benchrep.examples import load_example_config

example = "01_yaml_pipeline"
base = load_example_config(example, stage="training")

config = update_config(base, {"model.params.latent_dim": 64})

print(base.model.params["latent_dim"])    # 32
print(config.model.params["latent_dim"])  # 64
```

`update_config()` takes a mapping of existing field paths to replacement values and returns a validated copy. You can reuse `base` to try another latent dimension.

!!! note "Local YAML?"
    To start from your own training YAML, replace the `base = load_example_config(...)` line above with:

    ```python
    import yaml
    from benchrep.assembly.schemas import TrainingConfig

    with open("path/to/training.yaml", encoding="utf-8") as stream:
        base = TrainingConfig.model_validate(yaml.safe_load(stream))
    ```

## Run the chain

Pass the edited object as `config` for training. Prediction and evaluation can still use the packaged YAMLs from the first tutorial. Each workflow accepts its configuration independently.

```python
from benchrep import (
    EvaluationOverrides,
    PredictionOverrides,
    evaluate,
    predict_vae,
    train_vae,
)
from benchrep.examples import example_config_path

training = train_vae(config=config)

prediction = predict_vae(
    config=example_config_path(example, stage="prediction"),
    overrides=PredictionOverrides(
        training_manifest_path=training.manifest_path,
    ),
)

evaluation = evaluate(
    config=example_config_path(example, stage="evaluation"),
    overrides=EvaluationOverrides(
        prediction_manifest_path=prediction.manifest_path,
    ),
)
```

## Build configs entirely in Python

You can also construct `TrainingConfig`, `PredictionConfig`, and `EvaluationConfig` directly in Python. For example, pass a `TrainingModelConfig` as the `model` section of a `TrainingConfig`:

```python
from benchrep.assembly.schemas import TrainingConfig, TrainingModelConfig

model = TrainingModelConfig(name="vae", params={"latent_dim": 64})

config = TrainingConfig(
    model=model,
    # Supply the remaining required sections here.
)
```

The [Python-only MNIST example](https://github.com/uhlmanngroup/BenchRep/blob/main/examples/usage/03_config_object_pipeline.py) shows how to build all 3 workflow configs in full.

To run that separate example with its own settings, including its original 32-dimensional latent space:

```bash
python -m benchrep.examples.usage.03_config_object_pipeline
```

Next: [Supply overrides from Python](overrides.md).
