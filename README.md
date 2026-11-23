<h1>
  <img src="docs/assets/logo-wordmark-light.svg#gh-light-mode-only"
       alt="BenchRep" width="340">
  <img src="docs/assets/logo-wordmark-dark.svg#gh-dark-mode-only"
       alt="BenchRep" width="340">
</h1>

BenchRep is a configuration-driven, Lightning-based, extensible **Python workbench for representation learning**. It connects model training, representation export, and standardized evaluation through reproducible, linked workflows.

> [!NOTE]
> BenchRep is an early, pre-1.0 research project under active development. APIs, schemas, contracts, and supported components may still change.

## Why BenchRep?

**Benchmarking.** Compare representations from different models using shared evaluation settings, including outputs produced outside BenchRep. Export metrics and figures with records that trace results back to their inputs and configuration.

**Experimentation.** Swap components and objectives, assemble multi-input or multitask architectures, or bring your own model implementation. Use pretrained weights, inspect training behavior, and automate hyperparameter tuning with Optuna.

**Education.** Explore representation learning with less coding overhead. Configure experiments through YAML, Python, or browser wizards, and inspect model structure through architecture diagrams.

## How it works

### Workflows

**Training** fits a model and saves checkpoints. **Prediction** exports representations and reconstructions. **Evaluation** measures reconstruction quality, clustering, predictability, and batch effects, producing metrics, figures, and AnnData-integrated outputs.

Connect workflows by passing their manifest paths, or evaluate external embeddings and reconstructions directly. Each workflow returns a Python result object and records its configuration, runtime environment, and status, including warnings and failures.

![BenchRep workflow chain](docs/assets/workflow-chain-light.svg#gh-light-mode-only)
![BenchRep workflow chain](docs/assets/workflow-chain-dark.svg#gh-dark-mode-only)

<sub>*Dash-dot arrows mark optional connections. Wands mark configuration wizards.*</sub>

### Model families

A **model family** defines how models are constructed or supplied and the interfaces they follow. Choosing a different encoder, for example, changes the model without changing its family.

- **Canonical families — Autoencoder and VAE:** predefined component connections and model behavior, with configurable encoders, decoders, and objectives.
- **Non-canonical families — Composite and YOYO:** Composite builds models from selected components and explicitly declared connections and objectives. YOYO ("You're On Your Own") accepts a supplied LightningModule implementation for maximum modeling flexibility.

![BenchRep model families](docs/assets/model-families-vconcise-light.svg#gh-light-mode-only)
![BenchRep model families](docs/assets/model-families-vconcise-dark.svg#gh-dark-mode-only)

Think of the families as different ways to lay the tracks: **canonical families** provide a predefined route with configurable parts; **Composite** lets you lay the tracks and checks their connections; **YOYO** leaves you to drive off-road, with control and responsibility for your model's behavior.

Registries let you select built-in or externally registered components by name, including datasets, transforms, encoders, decoders, heads, and losses. All four families support training with pretrained weights; YOYO also supports prediction from supplied weights without a prior BenchRep training run.

## Installation

BenchRep currently targets Python 3.11 or newer. Install directly from the repository:

```bash
python -m pip install "benchrep @ git+https://github.com/uhlmanngroup/BenchRep.git"
```

To include all user-facing optional features, including browser wizards, Optuna, scverse evaluation, external loggers, model graphs, and XGBoost probes:

```bash
python -m pip install "benchrep[full] @ git+https://github.com/uhlmanngroup/BenchRep.git"
```

Architecture diagrams additionally need the [Graphviz system executable](https://graphviz.org/download/). For individual extras and editable installations, see the [installation guide](docs/guides/installation.md) in the draft documentation.

## Quick start

Run a complete VAE training, prediction, and evaluation chain using the packaged example configurations. This example uses UMAP, so install the `scverse` extra if you have not already installed `full`:

```bash
python -m pip install "benchrep[scverse] @ git+https://github.com/uhlmanngroup/BenchRep.git"
```

Run from the directory where you want to keep downloaded data and outputs:

```python
from benchrep import (
    train_vae,
    predict_vae,
    evaluate,
    PredictionOverrides,
    EvaluationOverrides,
)
from benchrep.examples import example_config_path

example = "01_yaml_pipeline"

training_result = train_vae(
    config=example_config_path(example, stage="training"),
)

prediction_result = predict_vae(
    config=example_config_path(example, stage="prediction"),
    overrides=PredictionOverrides(
        training_manifest_path=training_result.manifest_path,
    ),
)

evaluation_result = evaluate(
    config=example_config_path(example, stage="evaluation"),
    overrides=EvaluationOverrides(
        prediction_manifest_path=prediction_result.manifest_path,
    ),
)
```

Each result exposes its `manifest_path` and `run_context.output_dir`. To change the experiment, copy the [example YAMLs](examples/configs/01_yaml_pipeline/) into your working directory, edit them, and pass their paths as `config`.

## Configuration wizards

Create or edit workflow configurations in your browser, with field documentation, component discovery, validation, and YAML export. With the `gui` extra installed (also included in `full`), launch a wizard using a packaged configuration:

```python
from benchrep import configure_workflow
from benchrep.examples import example_config_path
from benchrep.interfaces.model_families import VAE_FAMILY

configured = configure_workflow(
    stage="training",
    model_family=VAE_FAMILY,
    config=example_config_path("01_yaml_pipeline", stage="training"),
)
```

The wizard opens automatically in your browser. Omit `config` to start fresh. Wizards are available for training, prediction, and evaluation; they configure workflows without running them.

![BenchRep training configuration wizard](docs/assets/configuration-wizard-light.png#gh-light-mode-only)
![BenchRep training configuration wizard](docs/assets/configuration-wizard-dark.png#gh-dark-mode-only)

## Documentation and examples

The documentation is **still being written and reviewed** and is not yet published as a website. For now, start with the runnable examples.

- [Usage examples](examples/usage/): workflow configuration, external components, pretrained encoders, Composite assembly, training monitoring, contrastive learning, and Optuna tuning.
- [Example configurations](examples/configs/): YAML settings accompanying the scripts and starting points for your own experiments.
- [Draft documentation](docs/index.md): tutorials, Concepts & design, and API reference material under development.
