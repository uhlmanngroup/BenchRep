# Statuses and failures

BenchRep distinguishes an operation's outcome from the whole workflow's status. This allows independent work to survive some failures—for example, an evaluation metric may fail while another metric and its figures remain useful. Recovery is deliberate and workflow-specific; arbitrary exceptions are not universally swallowed.

## Read the outcome, not just the file list

| Status | Meaning |
| --- | --- |
| `completed` | The operation completed without recorded issues |
| `completed_with_warnings` | It completed, with issues worth inspecting |
| `partially_completed` | An aggregate contains both usable work and incomplete work |
| `failed` | The operation failed, or an aggregate has no successful active work sufficient for completion |
| `disabled` | The operation was not requested; it does not count against completion |
| `skipped` | The operation was not executed, for example because a prerequisite failed |

Not every stage uses every status. Evaluation steps also have internal `pending` and `running` states, which must be finalized before aggregation. Training additionally uses `completed_after_interruption`.

Issues accompany the states and explain their causes. A failed plot export and a failed metric calculation are different outcomes even if both leave no corresponding file. Start with the overall status, then inspect the affected branch or operation.

## Training

[`TrainingStatusReport`][benchrep.runtime.status.training.TrainingStatusReport] contains a status, issues, and any handled interruption signal. Errors take precedence over interruption, which takes precedence over warnings.

Ordinary training exceptions propagate. After training returns, BenchRep checks its checkpoint outputs and performs finalization. If those checks mark the run as failed, it writes a failure manifest and then raises `RuntimeError` rather than returning a training result.

Handled SIGINT or SIGTERM can allow finalization with `completed_after_interruption`. This means the run stopped early and reached finalization without recorded errors; it does not mean the requested training duration completed. A hard process termination cannot guarantee final records. See [Training and optimization](../training/optimization.md).

## Prediction

[`PredictionStatusReport`][benchrep.runtime.status.prediction.PredictionStatusReport] separates inference, AnnData export, and reconstruction export. Reconstruction pairs also have individual export outcomes.

Inference execution errors can propagate immediately. If inference returns but its outputs fail validation, BenchRep marks inference as failed and skips exports. Export branches can otherwise complete independently, retaining successful artifacts when another export fails.

<strong class="highlight">Prediction writes its manifest and then raises for both `failed` and `partially_completed` final statuses.</strong> Successful files may therefore exist even though the call did not return a result object. The exception identifies the manifest to inspect. Disabled exports do not count as incomplete work.

## Evaluation

[`EvaluationStatusReport`][benchrep.runtime.status.evaluation.EvaluationStatusReport] aggregates 3 sections: AnnData analyses, reconstruction analyses, and exports. Each section retains its individual operation outcomes.

Evaluation steps catch explicitly recoverable evaluation errors, record a failed outcome, and allow independent steps to continue. A dependent step is skipped when its prerequisite did not complete successfully. Unexpected exceptions propagate instead of being treated as ordinary analysis outcomes. Export operations have their own recovery handling.

When evaluation reaches finalization, it can return an evaluation result whose status is `partially_completed` or `failed`. Inspect that status before interpreting metrics; a returned object alone does not establish that all requested analyses succeeded.

For example, a missing clustering result can cause its clustering metrics to be skipped while an independent embedding analysis completes. Missing entries in `metrics.json` or a `get_metric()` lookup failure should therefore be interpreted together with the operation issues.

## How statuses are built

Stage-specific status builders aggregate outcomes rather than deriving success from directory contents. Prediction excludes disabled export branches and treats failed inference as a workflow failure. Evaluation summarizes operations into sections, then sections into the overall status; mixed successful and incomplete active work becomes `partially_completed`.

The workflow writes the resulting status report into its manifest and logs an outcome summary. Early failures in configuration, source loading, construction, or execution can occur before this finalization point, so a final manifest is not guaranteed for every exception. Existing logs and partial records may still help locate the failure.

See the [status reference](../../reference/results.md#status-reports) for the report types, and the workflow implementation pages for the exact execution boundaries.
