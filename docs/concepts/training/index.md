# Training

Training combines a model, its objective, and a datamodule with Lightning's training loop. BenchRep prepares these inputs, configures the Trainer and callbacks, and records the run and its checkpoints. The model defines what happens to each batch; the training workflow coordinates execution around it.

Use [`train_ae()`][benchrep.train_ae], [`train_vae()`][benchrep.train_vae], or [`train_composite()`][benchrep.train_composite] for the corresponding [model family](../models/index.md). Each accepts a training config and optional workflow overrides; training also accepts callback instances through `runtime_callbacks`.

The main controls are separated by responsibility:

| Control | Configuration |
| --- | --- |
| Objective and relative loss contributions | `losses` |
| Parameter updates and learning-rate scheduling | `optimizer` |
| Epochs, devices, precision, and loop behavior | `trainer` |
| Checkpoint selection and stopping criteria | `checkpointing`, `early_stopping` |
| Experiment tracking and diagnostics | `logger`, `inspection`, `additional_callbacks` |

The result exposes the model and datamodule along with the run's records. To run prediction, use its manifest path so prediction can reconstruct the model and select a saved checkpoint. The returned in-memory model is the model at the end of fitting; training does not automatically replace its weights with the best checkpoint.

- **[Training and optimization](optimization.md):** optimizer construction, learning-rate scheduling, Trainer settings, and the execution loop.
- **[Callbacks and checkpoints](callbacks.md):** callback sources, checkpoint policies, early stopping, and interruption handling.
- **[Monitoring and inspection](inspection.md):** logs, experiment tracking, gradient measurements, and architecture graphs.

For the shared configuration-to-run lifecycle, see [Workflow implementation](../workflows/workflow-anatomy.md). Model-specific computations and loss aggregation are explained under [Models & objectives](../models/index.md).
