# Extensibility

BenchRep separates selecting a component from implementing it. A registry connects a configuration name to Python code, so a custom dataset or loss can participate in the same workflow as a built-in one. Runtime overrides provide a broader escape hatch when you need to supply the model or data-loading implementation itself.

## Choose the extension point

| Intent | Mechanism |
| --- | --- |
| Reuse your implementation by name in configuration | Register a component in a supported registry |
| Change how the complete model trains or predicts | Supply a compatible canonical model class or instance through an override |
| Control dataset construction, sampling, or batching yourself | Supply a `LightningDataModule` class or instance through an override |
| Attach a callback that needs a live Python object | Pass a callback instance through training's `runtime_callbacks` argument |

Registration preserves BenchRep's surrounding construction and execution machinery. A runtime override replaces the relevant construction path while keeping the workflow's trainer, records, and supported exports. The implementation must still satisfy the interface expected by its consumer.

- **[Registries and discovery](discovery.md):** find implementations, aliases, constructor parameters, and extension policies.
- **[Registering components](registration.md):** choose the correct entry type and understand how configuration selects it.
- **[External models and datamodules](runtime-implementations.md):** implement workflow contracts and understand what remains BenchRep's responsibility.

For complete dataset and loss examples, see [Register external components](../../guides/register-components.md). Callback behavior is covered under [Callbacks and checkpoints](../training/callbacks.md).
