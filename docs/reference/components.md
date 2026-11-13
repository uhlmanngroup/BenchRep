# Component catalogue

Built-in names, aliases, implementations, and configuration locations, grouped by registry. This catalogue is generated from BenchRep's registry declarations and built-in registrations at documentation build time. Names registered by your own code appear in runtime discovery instead.

Choose a component by its canonical name or any listed alias. Parameter handling depends on the configuration section: use the [schema reference](schemas/index.md) for fields and the [discovery functions](registries.md) for constructor signatures, descriptions, and runtime contracts:

```python
from benchrep import inspect_registry

inspect_registry("encoder", "conv2d")
inspect_registry("custom_objective_loss")
```

Documented BenchRep implementations link to their API entries; separate source links open their definitions. Third-party implementations are identified by their Python path; their installed signatures can be inspected with the same discovery call. Availability in this catalogue does not imply that optional dependencies are installed.

## Implementation reference

- **[Models](components/models.md):** autoencoder, VAE, and composite model behavior and Lightning hooks.
- **[Encoders, decoders, and heads](components/architectures.md):** constructor parameters, dimensions, and forward methods.
- **[Datasets](components/datasets.md):** sample formats and dataset access.
- **[Losses](components/losses.md):** raw objectives and their tensor inputs.
- **[Loss-weight schedules](components/schedules.md):** position-based multipliers and schedule parameters.

::: benchrep.assembly.registries.core
    options:
      heading_level: 2
      extra:
        component_catalogue: true
      show_root_heading: false
      show_root_toc_entry: false
      skip_local_inventory: true
