# Example helpers

Import these helpers from `benchrep.examples`. They locate packaged resources in both source checkouts and installed distributions, so tutorials can run from your own working directory.

| Helper | Returns | Use |
| --- | --- | --- |
| `example_config_path()` | `Path` | Pass a packaged YAML to a workflow or wizard, or copy it before editing |
| `load_example_config()` | Workflow config object | Inspect settings or edit a validated copy with `update_config()` |
| `example_asset_path()` | `Path` | Locate a packaged input file or directory |

Packaged resources should be treated as read-only. These helpers do not run an example, generate missing assets, or redirect output paths. Run example scripts with `python -m benchrep.examples.usage.<module>`.

## Configuration resources

```python
from benchrep import update_config
from benchrep.examples import example_config_path, load_example_config

training_path = example_config_path("01_yaml_pipeline", stage="training")
base = load_example_config("01_yaml_pipeline", stage="training")
training_config = update_config(base, {"model.params.latent_dim": 64})
```

The path helper leaves validation to its consumer. The object helper immediately validates against the selected workflow schema; it does not run the workflow resolver. Use the path helper for templates that require overrides before schema validation.

::: benchrep.examples.example_config_path
    options:
      heading_level: 3

::: benchrep.examples.load_example_config
    options:
      heading_level: 3

## Packaged assets

::: benchrep.examples.example_asset_path
    options:
      heading_level: 3
