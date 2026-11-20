# Concepts & design

Explore how BenchRep interprets configurations, builds models, runs workflows, and records their results. Each section opens with an introductory **overview**. The **section pages** then explain how to use a feature and how it works. For complex mechanisms with a far-reaching span across the codebase, dedicated **Implementation** sections trace the steps from configuration to execution.

For runnable examples, see [Tutorials](../guides/index.md). To find the code behind a feature, open the [Codebase map](codebase-map.md).

- **[Workflows & configuration](workflows/index.md):** How workflow inputs are assembled, interpreted, and edited.
- **[Data](data/index.md):** How datasets, transforms, and datamodules supply data to a workflow.
- **[Models & objectives](models/index.md):** How canonical and non-canonical models connect components and learning objectives.
- **[Training](training/index.md):** How training is controlled, monitored, and recorded.
- **[Prediction](prediction/index.md):** How training runs become inference inputs and exported representations.
- **[Evaluation](evaluation/index.md):** How model outputs are analyzed, scored, and visualized.
- **[Results & provenance](results/index.md):** Which outputs are produced, where they live, and what they mean.
- **[Extensibility](extensibility/index.md):** How to discover registered components and supply your own implementations.
- **[Codebase map](codebase-map.md):** Browse the repository by responsibility, follow a feature across packages, and find the source and tests for a change.
