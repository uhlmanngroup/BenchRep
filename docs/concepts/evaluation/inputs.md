# Inputs and evaluation pipelines

Evaluation starts from artifacts, rather than a dataset or model. This makes it possible to revisit a prediction run with different analyses, or evaluate outputs generated outside BenchRep.

## Choose the source

Supply a prediction manifest to discover its exported artifacts, or give artifact paths directly:

```yaml
stage: evaluation
source:
  anndata_path: outputs/external/embeddings.h5ad
```

For a linked workflow, pass `overrides=EvaluationOverrides(prediction_manifest_path=prediction_result.manifest_path)` to [`evaluate()`][benchrep.evaluate]. This replaces the configured manifest path without editing the base YAML; see [Python overrides](../workflows/overrides.md#connect-runs-without-rewriting-paths).

| Source field | Supplies |
| --- | --- |
| `prediction_manifest_path` | Paths to prediction exports and upstream provenance |
| `anndata_path` | One `.h5ad` file for representation analysis |
| `reconstructions_path` | One directory containing a reconstruction bundle |
| `reconstruction_pair_id` | A particular reconstruction pair from a prediction manifest |

A direct artifact path takes precedence over the corresponding manifest entry. You can therefore replace the AnnData input while retaining the manifest's reconstruction bundle, or vice versa. The two branches are analyzed separately; evaluation does not join their observations or infer correspondence between them.

When a manifest contains one usable reconstruction pair, BenchRep selects it automatically. With several pairs, select one using `reconstruction_pair_id`; a single evaluation run does not iterate over every pair. A direct `reconstructions_path` already identifies the bundle and cannot be combined with a pair ID.

??? note "Partially available prediction outputs"
    Evaluation accepts a prediction manifest with status `completed`, `completed_with_warnings`, or `partially_completed`. The latter produces a warning and still requires usable artifacts. An incomplete manifest-derived reconstruction bundle is warned about and skipped; an explicitly supplied incomplete bundle raises an error. Explicitly enabling an analysis whose input is unavailable also raises an error.

## AnnData as the representation interface

<strong class="highlight">`adata.X` is the primary matrix being evaluated.</strong> Rows represent observations and columns represent embedding dimensions or other features. `adata.obs` holds aligned annotations such as class labels, regression targets, grouping identifiers, and plot colors. The matrix must be two-dimensional, nonempty, real-valued, and finite.

Prediction chooses the contents of `X` through its AnnData `primary_key`. Other representations may be retained in `adata.obsm`, but evaluation does not automatically score them all. PCA, KMeans, HDBSCAN, embedding metrics, clustering metrics, and predictability probes use `X`. UMAP, t-SNE, and Leiden can select a different representation through `use_rep`; that choice affects only their own computation.

To compare two representations consistently across analyses, provide separate AnnData inputs with each representation in `X`. [`package_matrix_as_anndata()`][benchrep.records.anndata_io.package_matrix_as_anndata] and [`write_h5ad()`][benchrep.records.anndata_io.write_h5ad] help prepare external matrices and annotations. Neither normalizes the values for you.

Reconstruction inputs follow a separate [artifact contract](reconstructions.md#supply-aligned-images). Their observation annotations need not come from the AnnData file.

## Select the work to perform

Most analysis groups have an `enabled` field. `true` requests the analysis and requires its inputs; `false` disables it. An omitted or `null` value follows the group's automatic behavior:

| Group | Automatic behavior |
| --- | --- |
| PCA and embedding metrics | Enabled when AnnData is available |
| UMAP, t-SNE, clustering, and predictability | Disabled until explicitly enabled |
| Internal clustering metrics | Enabled when a clustering method is enabled |
| External clustering metrics | Enabled when clustering is enabled and the reference-label column is available |
| Reconstruction metrics | Enabled when a reconstruction bundle is available |
| Plots | Enabled for applicable available outputs |
| Reconstruction TIFFs | Disabled; controlled separately by `reconstruction.export_tiffs` |

These defaults make a minimal config useful without requesting every potentially expensive analysis. They are not a dependency solver that silently enables unrequested clustering or dimensionality reduction. For example, explicitly enabling clustering metrics without an enabled clustering method is rejected.

## Implementation

The [evaluation resolver](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/evaluation_config_resolver.py) creates a run specification containing the resolved config, artifact inputs, selected analyses, and upstream information. Runtime preparation then loads the artifacts, checks their contents and required dependencies, and completes decisions that require actual data, such as whether reference labels exist.

The workflow builds two ordered pipelines using the step and pipeline classes in [`evaluation/pipelines.py`](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/pipelines.py):

1. The **AnnData pipeline** runs reductions, clustering, clustering metrics, embedding metrics, and then predictability targets. Each successful step passes its updated AnnData to the next.
2. The **reconstruction pipeline** passes the same aligned images to reconstruction metrics and, when needed for TIFF export, error-map computation. Each step returns a separate output mapping.

An AnnData step works on a copy. Its changes are adopted only after it succeeds, so a recoverable failure does not leave half-written results for subsequent steps. Clustering metrics also declare dependencies on their clustering step: if clustering fails, its dependent metrics are skipped, while unrelated analyses can continue.

<strong class="highlight">Recovery is explicit.</strong> A step may raise `RecoverableEvaluationStepError` for an expected analysis failure, such as an unsuitable clustering result. The pipeline records that failure and continues. Unexpected exceptions propagate instead of being converted indiscriminately into warnings. Individual metric and probe groups can also retain successful members when another member fails recoverably.

Export follows computation. An evaluated AnnData file is written when at least one AnnData step succeeds; failure to write that central artifact stops export. Metrics JSON, figures, and reconstruction TIFFs are attempted as separate optional export groups. See [Figures and visualization](figures.md) and [Statuses and failures](../results/statuses.md) for interpreting incomplete output sets.

The original inputs are not rewritten. Step-level `overwrite` settings permit replacing existing results inside the loaded AnnData; output files belong to the new evaluation run. Inspect [`evaluation_result.status_report`](../../reference/results.md) alongside its [export paths](../../reference/results.md#evaluation-export-paths).
