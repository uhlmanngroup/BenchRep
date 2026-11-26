# Installation

BenchRep requires <strong class="highlight">Python 3.12 or newer</strong>. Activate the Python environment where you want to run it. The commands below install from GitHub and require Git.

**Tested environments.** CI runs CPU tests with Python 3.12–3.14 on Ubuntu (24.04 and 26.04), macOS (15, 26, and 27 via a preview runner), and Windows Server 2025. See the [CI workflow](https://github.com/uhlmanngroup/BenchRep/blob/main/.github/workflows/ci.yml) for platform-specific test exclusions.

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
| UMAP, t-SNE, and Leiden clustering | `scverse` |
| Hyperparameter tuning, Lightning pruning, and the Optuna dashboard | `optuna` |
| Weights & Biases, TensorBoard, or MLflow logging | `wandb`, `tensorboard`, or `mlflow` individually, or `logging` for all three |
| Torchview graphs and Composite model diagrams, including wizard previews | `model_graph` |
| XGBoost predictability probes | `xgboost` |
| Example data preparation that uses scikit-image | `artifact_generation` |

!!! info "Graphviz installation"
    The `model_graph` extra installs the Python packages for architecture diagrams. Rendering also requires the [Graphviz](https://graphviz.org/download/) executable `dot` on your `PATH`. Run `dot -V` to check. Without it, training continues with a warning and wizard graph previews are unavailable.

On Windows, XGBoost requires the Microsoft Visual C++ Redistributable. See the [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html#python).

## Install for development

If you plan to edit BenchRep itself, clone the repository and install it in editable mode. The `dev` extra includes the optional features above, the test dependencies, and the documentation tools.

```bash
git clone https://github.com/uhlmanngroup/BenchRep.git
cd BenchRep
python -m pip install -e ".[dev]"
```

To run the wizard’s automated browser tests, install Chromium with `python -m playwright install chromium`. On Linux, use `python -m playwright install --with-deps chromium` if you also need the system libraries. For ordinary wizard use, your regular browser is enough.

From the repository root, run `python -m pytest` to run the test suite. Usage-example tests are opt-in via `--run-usage-examples`. See the [usage-example test guide](https://github.com/uhlmanngroup/BenchRep/blob/main/tests/usage/README.md) for commands and options.

## Verify the installation

See which encoders BenchRep has ready to use.

```bash
python -c "from benchrep import inspect_registry; inspect_registry('encoder')"
```

You should see built-in encoders such as `mlp` and `conv2d`, listed by the names you can use in a config.

Then [run your first workflow chain](first-workflow-chain.md).
