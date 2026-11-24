# Installation

BenchRep requires <strong class="highlight">Python 3.12 or newer</strong>. Activate the Python environment where you want to run it. The commands below install from GitHub.

!!! warning "XGBoost on macOS"
    XGBoost evaluation can hang on macOS. Use the core package plus the extras you need, excluding `xgboost`; both `full` and `dev` include it. See [Known issues](../known-issues.md#xgboost-on-macos) for details and a development install command.

## Install BenchRep

=== "Core"

    Install the core package, then add optional features as needed:

    ```bash
    python -m pip install "benchrep @ git+https://github.com/uhlmanngroup/BenchRep.git"
    ```

=== "All features"

    Install all user-facing features and the dependencies used across the tutorials with the `full` extra:

    ```bash
    python -m pip install "benchrep[full] @ git+https://github.com/uhlmanngroup/BenchRep.git"
    ```

Both options include the example scripts and configs, so you can run them from your own working directory.

## Optional dependencies

Choose extras for the features you need, e.g. `benchrep[gui,scverse,optuna]`.

| To use | Install extra |
| --- | --- |
| Browser-based configuration wizards | `gui` |
| Scanpy-backed evaluation, including UMAP and Leiden clustering | `scverse` |
| Hyperparameter tuning with Optuna and Lightning pruning | `optuna` |
| Weights & Biases, TensorBoard, or MLflow logging | `wandb`, `tensorboard`, or `mlflow` individually, or `logging` for all three |
| Model graph export with Torchview | `model_graph` |
| XGBoost predictability probes | `xgboost` |
| Example data preparation that uses scikit-image | `artifact_generation` |

!!! info "Graphviz installation"
    The `model_graph` extra installs the Python packages for architecture diagrams, but rendering also needs the [Graphviz system executable](https://graphviz.org/download/). Without it, workflows still run but skip diagram exports with warnings.

## Install for development

If you plan to edit BenchRep itself, clone the repository and install it in editable mode. The `dev` extra includes the optional features above, the test dependencies, and the documentation tools.

```bash
git clone https://github.com/uhlmanngroup/BenchRep.git
cd BenchRep
python -m pip install -e ".[dev]"
```

## Verify the installation

From any directory, ask BenchRep for the first tutorial’s training config:

```bash
python -c "from benchrep.examples import example_config_path; print(example_config_path('01_yaml_pipeline', stage='training'))"
```

The command should print a path ending in `configs/01_yaml_pipeline/training.yaml`. This checks that Python can import BenchRep and locate its packaged example files.

Next: [Run your first workflow chain](first-workflow-chain.md).
