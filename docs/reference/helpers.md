# Example and artifact helpers

Import the packaged-example helpers below from `benchrep.examples`. They locate packaged resources in both source checkouts and installed distributions, so tutorials can run from your own working directory.

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

## AnnData artifacts

Import these functions from `benchrep.records`. `package_matrix_as_anndata()` places a sample-by-feature matrix in `adata.X` and sample annotations in `adata.obs`, ready to save for [standalone evaluation](../guides/standalone-evaluation.md). It does not normalize the matrix or run evaluation.

::: benchrep.records.anndata_io.package_matrix_as_anndata
    options:
      heading_level: 3

::: benchrep.records.anndata_io.read_h5ad
    options:
      heading_level: 3

::: benchrep.records.anndata_io.write_h5ad
    options:
      heading_level: 3

## Model graphs

Import these helpers from `benchrep.records`. Torchview traces a model with example inputs. The composite specification graph instead shows declared components, assembly connections, and loss wiring; it requires the resolved model and loss specifications. These are also available through a completed training result's `run_spec`.

Graph generation requires the appropriate optional Python packages and the Graphviz executable. The workflow and wizard configuration interfaces normally handle these exports for you.

::: benchrep.records.architecture.export_torchview_graph
    options:
      heading_level: 3

::: benchrep.records.architecture.build_composite_model_spec_graph
    options:
      heading_level: 3

::: benchrep.records.architecture.export_composite_model_spec_graph
    options:
      heading_level: 3

::: benchrep.records.architecture.ModelGraphDependencyError
    options:
      heading_level: 3
