# Evaluate external outputs

<strong class="highlight">How can I evaluate embeddings and reconstructions produced outside BenchRep?</strong> Pass an AnnData file, a reconstruction bundle, or both directly to evaluation.

One runnable example uses embeddings and reconstructions from a VAE trained on an enriched version of [MNIST](http://yann.lecun.com/exdb/mnist/). The snippets show how the script supplies these artifacts, how the YAML configures the analyses, and how to prepare your own inputs.


[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/09_standalone_evaluation_workflow.py){ .md-button }
[:material-file-cog-outline: View config](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/configs/09_standalone_evaluation_workflow/evaluation.yaml){ .md-button }
{: .br-tutorial-resources }

The [artifact generation scripts](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/artifact_generation/09_standalone_evaluation_workflow) use a separate `EnrichedMNISTDataset` implementation with the same enrichment procedure as the one in [Build a composite model](multitask-composite.md).

## Run the example

The evaluation YAML uses the `scverse` extra for UMAP, t-SNE, and Leiden, and the `xgboost` extra for one regression probe.

!!! warning "XGBoost on macOS"
    Install with `benchrep[scverse]` in the command below. In a copy of the evaluation YAML, remove `xgboost` from `metrics.predictability.targets.mean_stroke_width.selected`. Run the Python call in the next section with your edited YAML as `config`. See [XGBoost on macOS](../known-issues.md#xgboost-on-macos).

```bash
python -m pip install "benchrep[scverse,xgboost] @ git+https://github.com/uhlmanngroup/BenchRep.git"
python -m benchrep.examples.usage.09_standalone_evaluation_workflow
```

The example reads packaged inputs without training a model or downloading MNIST. It prints the input shapes and annotations, then the evaluation status, output directory, and manifest path.

???+ warning "Runtime and resource requirements"
    The YAML runs multiple probes with five-fold cross-validation, including a nested search for the acquisition-batch target. Several estimators use `n_jobs: -1` to use all available CPU cores. To shorten the run, copy the YAML and trim the `selected` lists for the digit-label and stroke-width probes. For a quicker check of the other analyses, set `metrics.predictability.enabled: false`. Pass the edited YAML as `config` in the Python call below.

## Supply artifacts directly

The example leaves the source paths open in YAML. The script locates the packaged artifacts with [`example_asset_path()`][benchrep.examples.example_asset_path] and passes them to [`evaluate()`][benchrep.evaluate] through [`EvaluationOverrides`][benchrep.EvaluationOverrides].

```python
from benchrep import EvaluationOverrides, evaluate
from benchrep.assembly.schemas import EvaluationSourceConfig
from benchrep.examples import example_asset_path, example_config_path

example = "09_standalone_evaluation_workflow"
assets = example_asset_path(example)

evaluation_result = evaluate(
    config=example_config_path(example, stage="evaluation"),
    overrides=EvaluationOverrides(
        config_components={
            "source": EvaluationSourceConfig(
                anndata_path=assets / "anndata.h5ad",
                reconstructions_path=assets / "reconstruction_bundle",
            ),
        },
    ),
)
```

### Use your own artifacts

Start with a minimal `evaluation.yaml` and set either path or both under `source`. Call `evaluate(config="evaluation.yaml")` from your working directory. Relative artifact paths are resolved from that directory.

```yaml
stage: evaluation
source:
  anndata_path: outputs/external/embeddings.h5ad
  reconstructions_path: outputs/external/reconstruction_bundle
```

With this config, BenchRep selects default analyses for the inputs you supply. If you adapt the example YAML, also disable any explicitly enabled analysis or export that requires an input you omit. The two pipelines process their observations independently. See [Inputs and evaluation pipelines](../concepts/evaluation/inputs.md).

#### Prepare an AnnData file

Place the sample-by-feature matrix in `adata.X` and row-aligned annotations in `adata.obs`. If you have a matrix, sample IDs, and labels, use [`package_matrix_as_anndata()`][benchrep.records.anndata_io.package_matrix_as_anndata] and [`write_h5ad()`][benchrep.records.anndata_io.write_h5ad].

```python
from benchrep.records import package_matrix_as_anndata, write_h5ad

adata = package_matrix_as_anndata(
    embeddings,
    sample_ids=sample_ids,
    labels=labels,
    label_key="label",
)
write_h5ad(adata, "outputs/external/embeddings.h5ad")
```

This puts the sample IDs in `adata.obs.index` and the labels in `adata.obs["label"]`. Pass additional row-aligned columns through `metadata`. When adapting the example YAML, update target names, `cv.group_key`, the external clustering metric's `label_key`, and `plots.params.color_by` to match your annotations.

#### Prepare a reconstruction bundle

Save aligned inputs and reconstructions as tensors or NumPy arrays of shape `(B, H, W)` or `(B, C, H, W)`, with identical shapes and sample and channel order. Save the image annotations alongside them, with one value per image in each field.

```python
from pathlib import Path

import torch

bundle = Path("outputs/external/reconstruction_bundle")
bundle.mkdir(parents=True, exist_ok=True)
torch.save(inputs, bundle / "input.pt")
torch.save(reconstructions, bundle / "reconstruction.pt")
torch.save({"label": labels}, bundle / "obs.pt")
```

Here, `labels` follows the order of the saved image pairs. Use the bundle's annotation names when configuring grid stratification. For optional channel names and the full bundle format, see [Reconstruction evaluation](../concepts/evaluation/reconstructions.md#supply-aligned-images).

## Analyze the embeddings

The example's AnnData has digit labels, estimated mean stroke widths, and simulated acquisition-batch labels. The YAML projects the embeddings with PCA, UMAP, and t-SNE, and colors the plots by those annotations. It computes per-feature statistics and scores the K-means, Leiden, and HDBSCAN assignments for cluster quality and agreement with digit labels.

Most analyses use `adata.X`. In this YAML, UMAP and Leiden explicitly use `X`, while t-SNE reads the PCA coordinates from `adata.obsm["X_pca"]` through `use_rep: X_pca`. UMAP and Leiden each build a neighborhood graph, and distinct `neighbors_key` values give each graph its own storage key. To compare two representations across the same analyses, run evaluation once with each representation in `X`, keeping the observations and settings fixed. See [Reductions and clustering](../concepts/evaluation/reductions-clustering.md).

### Probe digit identity, stroke width, and acquisition batch

A predictability probe fits a supervised estimator on the saved embeddings. Each key under `metrics.predictability.targets` names a column in `adata.obs`. For the digit and stroke-width targets, `cv.group_key: acquisition_batch` holds out entire batches when testing whether the embeddings retain those properties. The batch probe tests whether the embeddings carry information about the simulated acquisition conditions.

| Target | Task and score | Five-fold split |
| --- | --- | --- |
| `digit_label` | Classification, balanced accuracy | `stratified_group_kfold` keeps batches together while aiming to balance digit labels across folds |
| `mean_stroke_width` | Regression, R² | `group_kfold` keeps batches together |
| `acquisition_batch` | Classification, balanced accuracy | `stratified_kfold` distributes examples of each batch across folds |

Each target compares its fitted probes with a dummy baseline. The digit target uses logistic regression, k-nearest neighbors, random forest, and RBF-SVM. The stroke-width target uses their regression counterparts and adds XGBoost.

The acquisition-batch target searches over logistic regression's `C`. These fields come from `metrics.predictability.targets.acquisition_batch`.

```yaml
tuning:
  enabled: true
  inner_cv:
    n_splits: 3
params:
  linear:
    model: logistic_regression
    C: [0.1, 1.0, 10.0]
```

For each outer fold, the three-fold inner search chooses `C` using the outer training partition. The selected estimator is scored on that fold's held-out observations. To turn off the search, set `tuning.enabled: false` for this target and give `params.linear.C` a scalar value such as `1.0`. See [Predictability probes](../concepts/evaluation/predictability.md) for split strategies and tuning options.

## Measure and visualize reconstruction error

The bundle contains selected input–reconstruction pairs with two channels, morphology and skeleton. The example computes MAE, MSE, RMSE, and maximum absolute error on **all pairs in the bundle**. It sets `metrics.reconstruction.reduction: both` to report global and per-channel values.

For visual inspection, `reconstruction.export_tiffs: true` writes the first 12 pairs and their error maps, as set by `reconstruction.n_examples`. The grid settings sample one saved pair per digit and plot each channel separately.

```yaml
plots:
  params:
    reconstruction_grid:
      include_error_maps: true
      random_state: 137
      stratify_by: digit_label
      channel_selection: all
```

Both exports use the error-map kinds selected under `reconstruction.error_maps.kinds`. The example saves both embedding figures and reconstruction grids in PNG, PDF, and SVG. See [Figures and visualization](../concepts/evaluation/figures.md) for selection and coloring options.

## Inspect the outputs

Open the output directory printed by the script.

| Location | Contents |
| --- | --- |
| `records/metrics/metrics.json` | Consolidated metrics and probe results |
| `artifacts/anndata/evaluated_anndata.h5ad` | AnnData with the added reductions, cluster assignments, and metrics |
| `figures/anndata/` | Embedding projections, PCA diagnostics, and clustering figures |
| `figures/reconstructions/` | Input–reconstruction grids with error maps |
| `artifacts/reconstructions/` | TIFFs of selected inputs, reconstructions, and error maps |
| `records/metadata/evaluation_manifest.yaml` | Input paths, run status, and output locations |

Check `evaluation_result.status_report.status` and the issues recorded in the manifest when an expected result is missing. Independent analyses can still produce results after a recoverable failure.

To retrieve the digit-label linear probe's mean balanced accuracy from Python, use [`get_metric()`][benchrep.workflows.evaluate.EvaluationWorkflowResult.get_metric] with the same nesting as `metrics.json`.

```python
score = evaluation_result.get_metric(
    "predictability", "digit_label", "probes", "linear", "mean_score"
)
print(f"Linear-probe balanced accuracy: {score:.1%}")
```

For more on reading the run records, see [Trace an evaluation result](trace-an-evaluation-result.md).
