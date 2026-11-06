# Configuring workflows

There are several ways to configure a BenchRep workflow. You can write your own YAML, start from a packaged example, use the browser wizard, or build a config in Python. In each case, the workflow function you call (e.g. `train_vae()`) accepts either a YAML path or a config object through its `config` argument.

Below is a summary of the configuration options available from each starting point.

| Starting point | YAML path | Config object |
|---|---|---|
| Your own YAML | Use the file’s path | Read the file with `yaml.safe_load()`, then pass the resulting mapping to the workflow config class’s `model_validate()` |
| Packaged example | `example_config_path()` | `load_example_config()` |
| Browser wizard | Use the exported YAML path | Use the returned `config` together with its `overrides` |
| Python from scratch | — | Construct a `TrainingConfig`, `PredictionConfig`, or `EvaluationConfig` |

To change <strong class="highlight">individual fields</strong>, use `update_config()` on a config object before passing it to the workflow. To replace <strong class="highlight">whole sections</strong>, pass a workflow-specific overrides object alongside the base config, e.g. `train_vae(config=path, overrides=TrainingOverrides(...))`. You can also omit `config` if `overrides.config_components` supplies all the sections to build a valid full workflow configuration.

## Guides

- [Configure a workflow with YAML](../yaml.md)
- [Build and update config objects in Python](../python-configs.md)
- [Supply overrides from Python](../overrides.md)
- [Use the configuration wizard](../wizard.md)
