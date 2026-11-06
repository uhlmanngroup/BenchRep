# Configure a workflow with YAML

In the [first workflow chain](first-workflow-chain.md), the MNIST VAE learned a 32-dimensional embedding. Here you’ll change the latent dimension to 64 by editing the training YAML, then run prediction and evaluation with the same settings as before.

<strong class="highlight">Requirements.</strong> The evaluation step uses the `scverse` extra for UMAP. If you ran the first tutorial or installed the full or development dependencies, you’re ready. Otherwise, run:

```bash
python -m pip install "benchrep[scverse] @ git+https://github.com/uhlmanngroup/BenchRep.git"
```

## Copy the example configs

From the working directory you used for the first tutorial, run this Python snippet once. It copies the 3 packaged YAMLs into a new directory so you can edit them and reuse the downloaded MNIST data.

```python
from pathlib import Path
from shutil import copyfile

from benchrep.examples import example_config_path

example = "01_yaml_pipeline"
configs = Path("configs/mnist-vae-64")
configs.mkdir(parents=True, exist_ok=True)

training_yaml = configs / "training.yaml"
copyfile(example_config_path(example, stage="training"), training_yaml)
```

## Change the latent dimension

In `configs/mnist-vae-64/training.yaml`, change `latent_dim` from `32` to `64` in the existing `model` section:

```yaml
model:
  name: vae
  params:
    latent_dim: 64
```

!!! note "Encoder output and latent dimensions"
    Keep `encoder.params.output_dim` at `32`. The variational head maps those 32 features to the mean and log variance of the 64-dimensional latent distribution. BenchRep sets the decoder input dimension to `64` automatically.

## Run the modified chain

Run this code from the same working directory. Training uses your edited YAML. Prediction and evaluation reuse the packaged YAMLs from the [first workflow chain](first-workflow-chain.md), with each manifest linking the next workflow to its inputs.

```python
from pathlib import Path

from benchrep import (
    EvaluationOverrides,
    PredictionOverrides,
    evaluate,
    predict_vae,
    train_vae,
)
from benchrep.examples import example_config_path

example = "01_yaml_pipeline"
training_yaml = Path("configs/mnist-vae-64/training.yaml")

training = train_vae(config=training_yaml)

# Reuse the prediction and evaluation YAMLs from the first workflow chain.
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

## Check what changed

Open the resolved config in the training run directory and confirm that `model.params.latent_dim` is `64`. To locate the file from Python:

```python
print(training.run_context.output_dir / "records/configs/resolved_config.yaml")
```
!!! note "RunContext"
    Each workflow result exposes a `run_context` with the run’s identity and
    the paths to its output directories. Here, `training.run_context.output_dir`
    gives the root directory for this training run.
    See [Run outputs and provenance](../internals/run-records.md) for the full layout.

You can now compare the 32- and 64-dimensional runs using the same clustering and reconstruction metrics from the [first tutorial](first-workflow-chain.md).

Next: [Configure the same run with Python objects](python-configs.md).
