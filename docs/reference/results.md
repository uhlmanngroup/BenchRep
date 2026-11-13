# Results and statuses

Each [workflow entrypoint](workflows.md) returns a result containing its resolved config, run context, status report, and manifest path, together with stage-specific objects and outputs. These are returned objects to inspect; you normally do not construct them yourself.

A returned result is not a guarantee that every requested operation succeeded. Inspect `status_report` for warnings, interruptions, and incomplete exports. Errors raised before a result can be assembled still propagate to the caller.

## Workflow results

::: benchrep.workflows.train.TrainingWorkflowResult
    options:
      heading_level: 3

::: benchrep.workflows.predict.PredictionWorkflowResult
    options:
      heading_level: 3

::: benchrep.workflows.evaluate.EvaluationWorkflowResult
    options:
      heading_level: 3
      members:
        - get_metric

## Run identity and directories

`run_context` centralizes the run name and conventional output paths. Paths describe where outputs belong; use the workflow's export results to determine which artifacts were actually written. The `run_spec` field exposes the resolved execution plan for inspection, rather than a separate configuration interface.

::: benchrep.runtime.run_context.RunContext
    options:
      heading_level: 3

## Prediction export results

Start with `prediction_result.export_result`. The AnnData branch exposes its path and outcome; the reconstruction branch exposes one result per input–reconstruction pair. Disabled, skipped, or unsuccessful exports may have no written file.

::: benchrep.records.prediction_exports.PredictionExportResult
    options:
      heading_level: 3
      members:
        - outcomes

::: benchrep.records.prediction_exports.PredictionAnnDataExportResult
    options:
      heading_level: 3

::: benchrep.records.prediction_exports.PredictionReconstructionsExportResult
    options:
      heading_level: 3

::: benchrep.records.prediction_exports.PredictionReconstructionPairExportResult
    options:
      heading_level: 3

::: benchrep.records.prediction_exports.PredictionReconstructionBundlePaths
    options:
      heading_level: 3

## Evaluation export paths

Use `evaluation_result.export_paths` for saved artifacts and `evaluation_result.get_metric(...)` for in-memory metrics. The latter follows the same nested keys shown in `metrics.json`.

::: benchrep.records.evaluation_exports.EvaluationExportPaths
    options:
      heading_level: 3

## Status reports

Statuses distinguish the whole run from individual operations. `disabled` means an operation was not requested; `skipped` means it was not executed. `partially_completed` reports a mixture of usable results and incomplete work. The associated issues explain the specific outcome.

### Training

A handled SIGINT or SIGTERM can produce `completed_after_interruption`; this does not mean the requested training duration was reached. Recorded errors take precedence and produce `failed`.

::: benchrep.runtime.status.training.TrainingStatusReport
    options:
      heading_level: 4

### Prediction

Inference and the 2 export branches have separate outcomes. Inference failure makes the overall run fail; disabled exports do not count against completion.

::: benchrep.runtime.status.prediction.PredictionStatusReport
    options:
      heading_level: 4

::: benchrep.runtime.status.prediction.PredictionOutcome
    options:
      heading_level: 4

### Evaluation

AnnData analyses, reconstruction analyses, and exports each aggregate their own operation outcomes. `pending` and `running` are intermediate operation states, not completed reports. A section with no active operations is `disabled`.

::: benchrep.runtime.status.evaluation.EvaluationStatusReport
    options:
      heading_level: 4

::: benchrep.runtime.status.evaluation.EvaluationSectionStatus
    options:
      heading_level: 4

::: benchrep.runtime.status.evaluation.EvaluationOutcome
    options:
      heading_level: 4
