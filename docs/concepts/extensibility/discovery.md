# Registries and discovery

A registry associates a name with an implementation. Configuration can then select `mlp`, for example, without importing its Python class. Each registry has its own namespace and calling rules: an encoder named `mlp` and a head named `mlp` are separate entries.

Discovery exposes those choices together with their contracts, so you can inspect what is available before constructing a configuration or implementing an extension.

## Inspect components and configuration separately

[`inspect_registry()`][benchrep.inspect_registry] answers questions about implementations:

```python
from benchrep import inspect_registry

inspect_registry()
inspect_registry("encoder")
inspect_registry("encoder", "mlp")
```

The first call lists registries and whether they support external registration or runtime overrides. The second lists a registry's entries; the third shows a particular implementation, including its signature and any declared runtime contract. Component aliases can be used in place of canonical names.

[`inspect_config()`][benchrep.inspect_config] answers questions about the schema that selects and configures those implementations:

```python
from benchrep import inspect_config
from benchrep.assembly.schemas import TrainingConfig, TrainingEncoderConfig

inspect_config(TrainingConfig)
inspect_config(TrainingEncoderConfig)
```

Pass a schema **class**, not a configuration instance. The output describes fields, types, defaults, constraints, and available field documentation. It does not resolve a particular experiment or test its runtime behavior.

For example, the encoder schema explains `name` and `params`, while inspecting the selected encoder explains which constructor parameters belong inside `params`. This distinction also applies to custom registrations: adding a constructor to a registry does not add new fields to the workflow schema.

## Use discovery from Python

The inspection functions print readable descriptions. For values you can process programmatically, use [`list_registries()`][benchrep.list_registries] and [`list_registered_components()`][benchrep.list_registered_components]:

```python
from benchrep import list_registries, list_registered_components

registries = list_registries()
encoder_names = list_registered_components("encoder")
encoder_aliases = list_registered_components("encoder", include_aliases=True)
```

The first result maps discovery names to registry-object paths. Component listing returns sorted canonical names, or a mapping from canonical names to their aliases when `include_aliases=True`.

[`registry_documentation()`][benchrep.discovery.registry.registry_documentation] provides structured descriptions, signatures, extension policies, and contracts for tools that need more than a list. See the [discovery reference](../../reference/registries.md).

## Names, aliases, and availability

Registry names are normalized by trimming surrounding whitespace, converting to lowercase, and replacing hyphens with underscores. Aliases resolve to one canonical entry; they do not create separate implementations. Resolution uses canonical names where stable identifiers are needed, such as loss-term identities.

Discovery reflects the current Python process. Built-ins are initialized when needed, and imported custom registrations appear alongside them. The [component catalogue](../../reference/components.md) documents the built-in set; runtime discovery also includes your additions.

An entry's presence does not establish that every optional dependency or runtime requirement is satisfied. Some implementations load optional dependencies only when constructed or executed. Likewise, a valid signature cannot establish that a particular tensor or dataset will work.

## How discovery gets its information

Registries retain the actual callable or contract wrapper, together with the canonical-name and alias mappings. Discovery combines these entries with registry-specific metadata describing configuration locations, extension policies, and calling conventions. It inspects the underlying implementation's signature and docstring, and reads architecture or loss contracts from their wrappers.

This makes registration immediately visible to discovery without constructing the component. The [wizard's documentation panel](../workflows/wizards.md) uses this information alongside schema documentation. If you add a component, a useful constructor docstring and accurate contract therefore improve both Python inspection and the graphical editor.
