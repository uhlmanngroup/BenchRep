# Evaluation

Evaluation asks what a representation preserves, how observations organize within it, and how well reconstructed images match their inputs. It operates on saved artifacts, so you can compare outputs from BenchRep models and external tools through the same workflow.

The [`evaluate()`][benchrep.evaluate] entrypoint accepts an evaluation config and optionally a prediction manifest through [`EvaluationOverrides`][benchrep.EvaluationOverrides]. It does not load a model or run inference. Its two input branches can run independently or together:

| Input | Analyses | Main outputs |
| --- | --- | --- |
| AnnData file containing a sample-by-feature matrix | Dimensionality reduction, clustering, embedding and clustering metrics, supervised predictability probes | Evaluated AnnData, metrics, diagnostic figures |
| Bundle of input images and their reconstructions | Reconstruction metrics and pixelwise errors | Metrics, comparison grids, optional TIFFs |

An evaluation config selects the analyses and their parameters. The workflow resolves artifact paths, checks the available inputs, runs the applicable steps, and exports results into its own run directory. A returned [`EvaluationWorkflowResult`][benchrep.workflows.evaluate.EvaluationWorkflowResult] gives access to those results and their status; successful analyses can remain available when another step fails recoverably.

<strong class="highlight">An evaluation score answers a particular question, not whether a model is universally better.</strong> Clustering agreement with labels, linear predictability, and reconstruction accuracy measure different properties. Comparing runs meaningfully also requires comparable observations, preprocessing, and evaluation settings.

- **[Inputs and evaluation pipelines](inputs.md):** artifact selection, automatic step activation, execution order, and recovery from failures.
- **[Reductions and clustering](reductions-clustering.md):** which representation each method uses and where it stores its results.
- **[Metrics and their contracts](metrics.md):** metric selection, inputs, result formats, and retrieving scores.
- **[Predictability probes](predictability.md):** learning supervised probes on embeddings, cross-validation, and nested hyperparameter tuning.
- **[Reconstruction evaluation](reconstructions.md):** aligned image comparisons, channel reductions, and error maps.
- **[Figures and visualization](figures.md):** projections, annotation-based coloring, reconstruction grids, and export behavior.

Field-level defaults are listed in the [evaluation schema reference](../../reference/schemas/evaluation.md). For the upstream choice of which representations and images to save, see [Selecting prediction exports](../prediction/exports.md).
