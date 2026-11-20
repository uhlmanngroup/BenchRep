# Figures and visualization

Evaluation figures make computed results inspectable: projections show how observations organize, cluster-size plots reveal uneven partitions, and reconstruction grids put predictions beside their inputs. Figures complement metrics; they do not replace the quantitative analyses or change their results.

## Plot available results

`plots.enabled` defaults to enabled for available data. BenchRep exports plots for enabled reductions and clustering methods whose results are available, alongside applicable PCA diagnostics and reconstruction grids. Enabling plots does not automatically enable UMAP, clustering, or another analysis.

```yaml
plots:
  enabled: true
  params:
    formats: [png, svg]
    dpi: 300
    color_by:
      - label
      - key: measured_value
        kind: continuous
        cmap: viridis
```

Formats apply to the generated plots; TIFF images use the separate [reconstruction export settings](reconstructions.md#locate-errors-within-images). Set `plots.enabled: false` to run numerical analyses without figure export.

Reduction plots read coordinates already present in `adata.obsm` and use their first two dimensions. They require at least two coordinates per observation. Clustering assignments and labels supply colors, rather than determining the plotted coordinates.

## Color by observation annotations

`color_by` names columns of `adata.obs`. A string uses automatic interpretation; an expanded entry can set `kind` and a Matplotlib colormap. Available clustering keys and the reference-label key are added automatically when relevant. Explicit entries take priority, so you can customize a column's appearance without duplicating its plot request.

Automatic coloring treats strings, categorical values, and booleans as categories. Integer-like numeric columns with at most 30 distinct values are also treated as categorical; other numeric columns generally receive continuous coloring. Choose an explicit kind when the numerical encoding does not reflect the annotation's meaning.

Categorical plots use category legends; continuous plots use a color scale. Missing requested columns are skipped. Plot annotations must already exist in the evaluated artifact: adding a field to a training dataset does not by itself carry it into prediction exports. See [Selecting prediction exports](../prediction/exports.md).

## Reconstruction grids

Each grid compares input and reconstruction images for one channel, optionally adding configured error maps. By default, BenchRep samples at most 12 examples and displays channel 0, warning if other channels are available.

```yaml
plots:
  params:
    reconstruction_grid:
      stratify_by: label
      channel_selection: all
      include_error_maps: true
      random_state: 137
```

With `stratify_by`, BenchRep samples one example per distinct value of that reconstruction observation field and divides the selection into pages of at most 12 examples. Channels are exported separately. This is useful for inspecting coverage across labels without relying on a purely random selection.

The grid uses paired intensity limits for input and reconstruction comparisons and includes identifiers when available. Error-map kinds come from `reconstruction.error_maps`. Grid selection is independent of `reconstruction.n_examples`, which only limits TIFF output; neither setting limits reconstruction metric computation.

## Implementation and export outcomes

The [evaluation exporter](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/records/evaluation_exports.py) determines which figures are applicable, selects observation colors or reconstruction examples, calls the plotting functions, and writes the requested formats. AnnData plots are implemented in [`embeddings/plotting.py`](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/embeddings/plotting.py); reconstruction grids in [`reconstructions/plotting.py`](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/reconstructions/plotting.py).

Figure export groups are recoverable: a plotting or file-writing failure is recorded without preventing the remaining optional export groups from being attempted. This differs from a failed analysis, which may leave no result to plot at all. Use [`evaluation_result.export_paths`][benchrep.records.evaluation_exports.EvaluationExportPaths] to locate saved figures, and its [status report](../results/statuses.md) to distinguish disabled, unavailable, and failed outputs.
