# Results & provenance

An experiment produces more than a trained model or a score. BenchRep records the settings used, the artifacts written, and the operations that succeeded or failed. These records let you inspect a run in Python, connect it to another workflow, and understand its results later.

Each workflow has its own run directory. Its **result object** exposes live Python objects and output paths; its **manifest** records the run on disk and connects downstream workflows to it. Neither a directory nor a manifest's existence alone proves that every requested operation succeeded: the associated status records explain what happened.

| To answer… | Start with… |
| --- | --- |
| Where did this run write its outputs? | The result's `run_context` and manifest |
| Which settings actually took effect? | `resolved_config.yaml` and manifest inheritance records |
| Which checkpoint or exported data produced this result? | The manifest's source and artifact records |
| What completed, failed, or was skipped? | `status_report` and operation issues |
| Can I reproduce the run elsewhere? | Configurations, upstream artifacts, runtime inputs, and the environment record together |

- **[Run directories and result objects](runs.md):** output layout, naming, and working with returned objects.
- **[Artifact formats](artifacts.md):** checkpoints, AnnData, reconstruction bundles, metrics, and figures.
- **[Manifests and provenance](provenance.md):** how runs are linked and what configuration can reconstruct.
- **[Reproducibility](reproducibility.md):** seeds, execution settings, and recorded environmental context.
- **[Statuses and failures](statuses.md):** recoverable work, exceptions, and workflow-specific completion rules.
