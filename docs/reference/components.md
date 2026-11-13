# Component catalogue

Built-in names, aliases, implementations, and configuration locations, grouped by registry. This catalogue is generated from BenchRep's registry declarations and built-in registrations at documentation build time. Names registered by your own code appear in runtime discovery instead.

Choose a component by its canonical name or any listed alias. Parameter handling depends on the configuration section: use the [schema reference](schemas/index.md) for fields and the [discovery functions](registries.md) for constructor signatures, descriptions, and runtime contracts:

```python
from benchrep import inspect_registry

inspect_registry("encoder", "conv2d")
inspect_registry("custom_objective_loss")
```

Implementation links open BenchRep's source. Third-party implementations are identified by their Python path; their installed signatures can be inspected with the same discovery call. Availability in this catalogue does not imply that optional dependencies are installed.

::: benchrep.assembly.registries.core
    options:
      heading_level: 2
      extra:
        component_catalogue: true
      show_root_heading: false
      show_root_toc_entry: false
      skip_local_inventory: true
