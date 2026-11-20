# Run directories and result objects

Each workflow creates a new run directory and, on a normal return, a Python result object. The directory preserves files for later use; the result also gives you access to objects still available in the current process.

## Run identity and layout

[`RunContext`][benchrep.runtime.run_context.RunContext] centralizes paths under:

```text
<output_root>/<stage>/<run_name>/
```

Training and prediction can supply `run.run_name_prefix`; evaluation can supply `run.run_name_stem`. BenchRep combines the resolved name with a timestamp. If that directory already exists, a short UUID suffix avoids overwriting it. Name components are sanitized for filesystem use.

All stages use the same locations for core records, with additional directories for their own outputs:

| Relative location | Contents |
| --- | --- |
| `records/configs/` | Original configuration, when supplied, and resolved configuration |
| `records/logs/` | BenchRep run log and captured console output |
| `records/metadata/` | Stage manifest and runtime-environment record |
| Training: `checkpoints/`, `architecture/` | Lightning checkpoints and available model inspection outputs |
| Prediction: `anndata/`, `reconstructions/`, `architecture/` | Exported representations, reconstruction bundles, and available model graph |
| Evaluation: `records/metrics/`, `artifacts/`, `figures/` | Metrics, evaluated data, and visualizations |

The context creates directories and exposes conventional paths; artifact writers save their contents. A directory can therefore exist without containing an output. Check the export result or manifest for the files actually written.

## Use the returned objects

Every workflow result exposes `config`, `run_spec`, `run_context`, `status_report`, and `manifest_path`. These have distinct purposes:

- **`config`:** the resolved configuration used by the workflow.
- **`run_spec`:** execution instructions derived during resolution, useful when inspecting how settings were interpreted.
- **`run_context`:** the run's identity and conventional output paths.
- **`status_report`:** completion states and recorded issues.
- **`manifest_path`:** the saved record used to inspect or connect the run.

The workflow-specific result adds the objects relevant to its work:

| Result | Additional objects |
| --- | --- |
| [`TrainingWorkflowResult`][benchrep.workflows.train.TrainingWorkflowResult] | Model, datamodule, trainer, checkpoint and early-stopping callbacks, and graph paths |
| [`PredictionWorkflowResult`][benchrep.workflows.predict.PredictionWorkflowResult] | Model, datamodule, trainer, per-batch predictions, and export results |
| [`EvaluationWorkflowResult`][benchrep.workflows.evaluate.EvaluationWorkflowResult] | Evaluated AnnData, reconstruction evaluation outputs, and export paths |

For example, after training:

```python
print(training_result.run_context.output_dir)
print(training_result.status_report.status)
print(training_result.checkpoint_callback.best_model_path)
```

Pass `training_result.manifest_path` to prediction to select a checkpoint from that run. The returned training model is the instance that finished training; it is not automatically replaced with the best checkpoint's weights. See [Loading models for prediction](../prediction/loading-models.md).

## Read a metric without reopening a file

[`EvaluationWorkflowResult.get_metric()`][benchrep.workflows.evaluate.EvaluationWorkflowResult.get_metric] follows the same nesting as the exported `metrics.json`:

```python
ami = evaluation_result.get_metric(
    "clustering", "external", "kmeans", "metrics",
    "adjusted_mutual_info", "value",
)
```

This reads the in-memory evaluation outputs. Calling it without arguments returns the complete metrics mapping. A missing metric raises a lookup error; consult the status report to distinguish a disabled metric from one that failed or was skipped.

Result objects retain runtime objects rather than serializing the whole process. A later process uses saved configurations, manifests, and artifacts instead. See [Manifests and provenance](provenance.md) for what those records can reconstruct, and [Statuses and failures](statuses.md) for cases where a workflow raises instead of returning a result.
