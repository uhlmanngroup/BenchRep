# Workflow implementation

A workflow coordinates configuration, component construction, execution, and recording. Its entrypoint does not implement the model or evaluation algorithm itself: it prepares the objects and instructions those implementations need, calls them, and records their outcomes.

The 3 workflows share this organization, but have separate orchestration functions. Training and prediction use Lightning; evaluation builds and runs artifact-processing pipelines. Understanding the shared boundaries makes it easier to follow either a normal run or a change to the implementation.

## Configuration, specifications, and live objects

Several objects describe the same run at different points. They serve different purposes:

| Object | What it contains | Who uses it |
| --- | --- | --- |
| Workflow config | Typed settings, first effective and then resolved | Schemas, resolvers, builders, and records |
| Run specification | Execution decisions derived from the config and context | Builders, workflow orchestration, exporters |
| Run context | Run identity and output-directory paths | Logging, callbacks, artifact writers, manifest writers |
| Runtime objects | Instantiated models, datamodules, trainers, or evaluation pipelines | Execution |
| Workflow result | Resolved settings, outcomes, paths, and stage-specific runtime results | The calling Python code |

A run specification does not contain a trained model. For example, `TrainingRunSpec` contains resolved loss specifications and, for composite models, a graph specification. Builders turn those descriptions into actual modules. Prediction specifications additionally identify the checkpoint and export selections; evaluation specifications identify artifacts and the analysis steps to perform.

## I. Compose and validate the inputs

Each public entrypoint fixes the stage and, for training or prediction, the model family. It identifies which inputs come from configuration and which are supplied as runtime overrides.

The shared composition layer then:

1. Reads the base YAML, accepts the matching config object, or starts from supplied config components.
2. Replaces any top-level sections supplied through `config_components`.
3. Parses the resulting mapping with the workflow schema, using override information as validation context.

This context matters: a supplied model or datamodule can remove configuration requirements for the construction it replaces. The override object itself remains outside the configuration; it is not converted into YAML. A matching typed config object with no section replacements is already validated and is passed through directly.

Composition also retains the source path and original YAML mapping for provenance. See [Configuration semantics](configuration.md) and [Python overrides](overrides.md) for the user-visible rules. The shared implementation is in [`assembly/config.py`](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/config.py).

## II. Resolve the intended execution

The stage's resolver interprets the effective config in context. It resolves names and aliases, checks compatibility, follows supported upstream inheritance, and produces a run specification containing the resolved config.

Consider prediction with `checkpoint: best`. Schema validation can establish that `best` is an allowed selector. Resolution reads the training manifest, finds the recorded checkpoint, and supplies its concrete path to execution. It also determines which training settings are inherited and how selected outputs will be exported. Model construction and checkpoint tensor loading happen later.

Resolution therefore goes beyond checking individual fields, but does not execute the experiment. A composite graph can have valid references and ports while a component still fails on real tensor dimensions. Evaluation also finalizes some automatic choices after loading its inputs—for example, choices that depend on available AnnData observation columns.

The stage resolvers live in [`assembly/resolvers`](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/assembly/resolvers). Model-specific resolution is explained alongside [composite assembly](../models/composite.md) and [losses](../models/losses.md).

## III. Construct and execute

After resolution, the workflow creates its run context and initializes logging and configuration records. It then prepares the required runtime objects:

| Workflow | Construction and execution |
| --- | --- |
| Training | Builds or accepts a datamodule and model, configures a Lightning trainer and callbacks, and calls `trainer.fit()` |
| Prediction | Builds or accepts the model and datamodule, loads the selected checkpoint weights, calls `trainer.predict()`, and interprets the returned batches for export |
| Evaluation | Loads the selected artifacts, constructs the applicable analysis pipelines, and runs their configured steps |

Builders instantiate registered classes and factories using resolved instructions. This is also where construction-dependent information becomes available: for example, the canonical model builder can use an encoder's declared dimensions to construct a compatible decoder or variational head.

A runtime class override is instantiated with its configured constructor arguments; an instance is used directly. The applicable construction branch is bypassed, while the surrounding workflow still manages execution and records. See [External models and datamodules](../extensibility/runtime-implementations.md).

Within training, the model owns its forward computation, objective, and optimizer hooks; Lightning owns the training loop. Within evaluation, pipeline steps own the individual algorithms. The workflow coordinates these systems rather than duplicating their implementations. Stage-specific details are covered under [Training](../training/index.md), [Prediction](../prediction/index.md), and [Evaluation](../evaluation/index.md).

## IV. Record outcomes and return

The workflow gathers stage outcomes, writes supported artifacts and a manifest, and returns its result object when execution reaches that return path. The result exposes live Python objects where appropriate; the manifest records serializable settings, artifact locations, and provenance for later processes.

This is why chaining workflows normally passes a manifest path: the next workflow resolves the saved run's records instead of depending on the previous process's live objects. Required external implementations still need to be supplied again.

<strong class="highlight">Finalization is not a universal exception handler.</strong> Invalid configuration can fail before a run directory exists. Construction failures and unhandled runtime errors can interrupt execution before a manifest is written. Other failures are deliberately recoverable: prediction can report failed exports, evaluation can retain successful steps, and training handles certain interruptions and optional inspection failures. Training can also write a failure manifest and then raise when post-training checkpoint validation fails.

See [Statuses and failures](../results/statuses.md) for these distinctions. For source navigation, start with the stage functions in [`workflows`](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/workflows) and the [Codebase map](../codebase-map.md).
