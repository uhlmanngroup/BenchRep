# Registries and discovery

Registries associate configuration names with implementations. Some store a constructor or factory directly; others require an [entry with an explicit contract](interfaces.md). External registration is supported only where the registry allows it.

For the available built-ins and their aliases, see the [Component catalogue](components.md). Registration affects the current Python process: import or execute your registrations before resolving a config that uses them.

## Inspect and list components

Import these discovery functions from `benchrep`. Inspection prints human-readable information; the listing functions return values for use in Python.

```python
from benchrep import inspect_registry, list_registered_components

inspect_registry()                   # Registries and extension policies
inspect_registry("encoder")          # Available encoders
inspect_registry("encoder", "mlp")   # Implementation and runtime contract

encoders = list_registered_components("encoder", include_aliases=True)
```

::: benchrep.inspect_registry
    options:
      heading_level: 3

::: benchrep.list_registries
    options:
      heading_level: 3

::: benchrep.list_registered_components
    options:
      heading_level: 3

### Structured discovery

Use this helper when building a component browser or processing the discovery metadata programmatically.

::: benchrep.discovery.registry.registry_documentation
    options:
      heading_level: 4

## Inspect configuration fields

`inspect_config()` describes a schema class and its fields. Pass the class rather than an instance; nested schema types can be inspected in the same way. The [Configuration schemas](schemas/index.md) pages provide the corresponding browsable reference.

```python
from benchrep import inspect_config
from benchrep.assembly.schemas import TrainingConfig

inspect_config(TrainingConfig)
```

::: benchrep.inspect_config
    options:
      heading_level: 3

## Registry objects

Import the objects below from `benchrep.assembly.registries`. The discovery name is the string accepted by the inspection and listing functions; the registry object exposes registration and lookup methods.

::: benchrep.assembly.registries.core
    options:
      heading_level: 2
      extra:
        registry_index: true
      show_root_heading: false
      show_root_toc_entry: false
      skip_local_inventory: true

## Registration and lookup methods

Use methods on the existing registry objects; there is usually no need to construct a new `Registry`. Names and aliases are normalized by stripping surrounding whitespace, converting to lowercase, and replacing `-` with `_`. Registration rejects names already in use.

::: benchrep.assembly.registries.core.Registry
    options:
      heading_level: 3
      members:
        - register
        - get
        - create
        - resolve_key
        - keys
        - canonical_keys
        - aliases_by_canonical
        - entry_type
        - custom_registration_supported

### Loss registry policies

Loss registries additionally define which runtime arguments are supplied automatically and which entry contracts are accepted. Their normal registration and lookup operations are inherited from `Registry`.

::: benchrep.assembly.registries.core.LossRegistry
    options:
      heading_level: 4
      members:
        - canonical_runtime_input_names
        - required_component_base
        - explicit_composite_runtime_inputs_supported
