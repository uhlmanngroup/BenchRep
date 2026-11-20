# Tutorials

Run a first workflow chain, then explore a specific task. Each tutorial focuses on the code needed for that task and links to complete usage examples where available. For the underlying rules and interfaces, see [Concepts & design](../concepts/index.md).

## Getting started

[Install BenchRep and run your first workflow chain](getting-started/index.md) to train a VAE on MNIST, export its outputs, and evaluate them.

## Choose a task

- [Configure workflows](configure-workflows/index.md) through YAML, Python config objects, overrides, or the browser wizard.
- [Register external components](register-components.md) and select them in a workflow config.
- [Build a composite model](multitask-composite.md).
- [Learn from multiple views](learn-from-multiple-views.md) and compare representations learned from random or pretrained initialization, with guidance for naturally paired images.
- [Schedule and monitor training](schedule-and-monitor-training.md) with loss-weight and learning-rate schedules, gradient norms, and experiment tracking.
- [Bring your own model or datamodule](bring-your-own-model-or-datamodule.md), including a model initialized with pretrained weights.
- [Evaluate external outputs](standalone-evaluation.md) and probe what their embeddings can predict.
- [Tune with Optuna](optuna.md).
- [Trace an evaluation result](trace-an-evaluation-result.md) through linked manifests, status reports, and runtime environment records.
