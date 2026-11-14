# Workflows & configuration

A BenchRep workflow turns an experiment description into a run. You choose what to build and execute through configuration, supply Python objects where configuration alone is insufficient, and receive results with records of how they were produced.

These pages explain how those inputs fit together and which rules govern their interpretation.

- **[Workflows and model families](workflows.md):** what each workflow owns, how runs connect, and what choosing a model family determines.
- **[Configuration semantics](configuration.md):** how fields, defaults, component names, and paths become an executable configuration.
- **[Python overrides](overrides.md):** reuse a base experiment while replacing sections, connecting runs, or supplying runtime implementations.
- **[Configuration wizards](wizards.md):** edit and inspect a configuration interactively, understand its validation, and continue execution in Python.

For runnable examples of the different configuration routes, see [Configure workflows](../../guides/configure-workflows/index.md). For the internal sequence of resolvers, builders, and execution, see [Anatomy of a workflow](../internals/workflow-anatomy.md).
