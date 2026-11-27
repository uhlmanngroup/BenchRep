# Tutorials

Start with your first workflow chain or jump to a specific task. A tutorial may follow one runnable example, show variations on a main example, or walk through several separate examples. The opening explains how the snippets relate to the linked scripts and configs.

For the underlying rules, interfaces, and implementation mechanics, see [Concepts & design](../concepts/index.md).

## Getting started

[Install BenchRep and run your first workflow chain](getting-started/index.md) to train a VAE on MNIST, export its outputs, and evaluate them.

## Choose a task

- [Configure workflows](configure-workflows/index.md) through YAML, Python config objects, overrides, or the browser wizard.
- [Register external components](register-components.md) and select them in a workflow config.
- [Build a composite model](multitask-composite.md) with two inputs and multiple objectives defined in YAML.
- [Learn from multiple views](learn-from-multiple-views.md) and compare representations from randomly initialized and pretrained encoders.
- [Schedule and monitor training](schedule-and-monitor-training.md) with loss-weight and learning-rate schedules, gradient norms, and experiment tracking.
- [Bring your own model or datamodule](bring-your-own-model-or-datamodule.md), including a model initialized with pretrained weights.
- [Evaluate external outputs](standalone-evaluation.md) and probe what their embeddings can predict.
- [Tune with Optuna](optuna.md) to search training hyperparameters and score trials by embedding quality.
- [Trace an evaluation result](trace-an-evaluation-result.md) through linked manifests, status reports, and runtime environment records.
