# Installation

BenchRep requires <strong class="highlight">Python 3.11 or newer</strong>. Activate the Python environment where you want to run it. The commands below install from GitHub.

## Install BenchRep

For the <strong class="highlight">core</strong> package, run:

```bash
python -m pip install "benchrep @ git+https://github.com/uhlmanngroup/BenchRep.git"
```

To install <strong class="highlight">all user-facing features</strong> and the dependencies used across the tutorials, use the `full` extra:

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
