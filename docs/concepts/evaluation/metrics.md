# Metrics and their contracts

Evaluation metrics turn saved representations or reconstruction pairs into measurements. They are separate from training losses: they do not contribute gradients, and their results may be scalars, vectors, or named collections of either.

## What each metric group measures

| Group | Inputs passed to the metric | Question | Workflow defaults |
| --- | --- | --- | --- |
| Internal clustering | `adata.X`, cluster assignments | How separated and compact are the groups in the embedding space? | Silhouette, Calinski–Harabasz, Davies–Bouldin |
| External clustering | Reference labels, cluster assignments | How well do groups agree with known labels? | Adjusted mutual information, adjusted Rand index, homogeneity |
| Embedding | `adata.X` | How are values distributed across embedding dimensions? | Mean, median, standard deviation, minimum, maximum, quantiles |
| Reconstruction | Input and reconstructed image arrays | How closely do the saved images match? | MAE, MSE, RMSE, maximum absolute error |

External clustering metrics use `metrics.clustering.external.label_key`, which defaults to `label` in `adata.obs`. Those labels score the resulting clusters; they are not supplied to the clustering algorithm. Internal metrics need suitable cluster assignments and can be undefined, for example when only one cluster remains. HDBSCAN noise handling is described under [Reductions and clustering](reductions-clustering.md#interpret-and-retain-the-outputs).

Embedding statistics describe the representation's scale and distribution, rather than its task performance. For a supervised question such as whether an embedding preserves a particular annotation, use [Predictability probes](predictability.md).

## Select metrics and parameters

Within a metric group, `selected` accepts registered names or aliases:

```yaml
metrics:
  embedding:
    selected: [mean, standard_deviation]
  clustering:
    external:
      label_key: label
      selected: [adjusted_mutual_info, adjusted_rand_index]
  reconstruction:
    selected: [mse, mae]
    reduction: both
```

This is a config fragment; its groups still follow their [activation rules](inputs.md#select-the-work-to-perform). Omitting `selected`, or setting it to `null`, uses the curated workflow defaults. An explicit list selects exactly those metrics. `selected: all` includes every currently registered metric in the group, including external registrations, so its contents can change when the process registers more components.

Group `params` maps metric names to keyword arguments for their functions. Use [`inspect_registry()`][benchrep.inspect_registry] to inspect the accepted arguments, for example `inspect_registry("internal_clustering_metric", "silhouette")`. The [component catalogue](../../reference/components.md) lists metric registries and built-ins.

## Result contracts

An evaluation registry entry is an [`EvaluationMetric`][benchrep.evaluation.metrics.contracts.EvaluationMetric] wrapper. It associates a callable with the kind of result it promises:

| `result_kind` | Meaning |
| --- | --- |
| `scalar` | One finite real number |
| `scalar_mapping` | A nonempty mapping of names to finite real numbers |
| `vector` | A one-dimensional vector associated with a declared axis |
| `vector_mapping` | A nonempty mapping of names to vectors on that axis |

Vector metrics declare `vector_axis`. The evaluator supplies the actual axis labels and checks the vector length against them. Embedding evaluation supports `embedding_dimension`; reconstruction evaluation supports `reconstruction_channel`. A metric cannot invent an axis that its consuming evaluator does not provide.

This contract gives the exporter a predictable structure while allowing richer measurements than a single score. Scalars are stored with `kind` and `value`; vectors retain their values and labeled axis. For external registration, see [Registering components](../extensibility/registration.md) and the [metric interface](../../reference/interfaces.md#evaluation-metrics).

## Read the results

AnnData metrics are stored under `adata.uns["benchrep"]["metrics"]`. Reconstruction metrics remain in the reconstruction pipeline's output mapping. The exporter combines both into `metrics.json`, and [`get_metric()`][benchrep.workflows.evaluate.EvaluationWorkflowResult.get_metric] exposes that same structure from the returned result:

```python
ami = evaluation_result.get_metric(
    "clustering", "external", "kmeans",
    "metrics", "adjusted_mutual_info", "value",
)
```

Use `evaluation_result.get_metric()` without arguments to inspect the complete mapping. A missing path raises an error; it does not imply a score of zero. Check the [status report](../results/statuses.md) when a requested measurement is absent.

## Implementation

The [metric executor](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/metrics/execution.py) resolves each wrapper, checks keyword arguments against its callable, supplies the group's inputs, and normalizes the result according to its contract. Normalization rejects incompatible shapes, non-real values, and non-finite results instead of writing ambiguous values into JSON or AnnData.

Metric groups distinguish expected data-dependent failures from implementation errors. A metric that raises `RecoverableEvaluationStepError` is recorded as failed while successful metrics are retained. If every selected metric fails recoverably, the group fails recoverably. Invalid registrations, incompatible calling conventions, and unexpected exceptions propagate. This keeps partial evaluation useful without hiding mistakes in an external metric implementation.
