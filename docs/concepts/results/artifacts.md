# Artifact formats

BenchRep saves model weights, numerical outputs, annotations, and figures separately. The manifest records their locations; the [run context](runs.md) supplies the directory convention. Which artifacts exist depends on the requested operations and their outcomes.

## Training checkpoints

Training writes Lightning `.ckpt` files under `checkpoints/`. They contain model state and Lightning checkpoint information; the manifest records the best and last checkpoint paths and selection settings.

Prediction constructs a compatible model, reads the selected checkpoint's `state_dict`, and loads those weights. This is inference from saved weights, not resumption of the training loop or optimizer state. A checkpoint does not supply arbitrary Python implementations that were passed as runtime overrides. See [Callbacks and checkpoints](../training/callbacks.md) and [Loading models for prediction](../prediction/loading-models.md).

## AnnData representations

Prediction's `anndata/anndata.h5ad` stores a sample-by-feature matrix with aligned annotations:

| AnnData location | Contents |
| --- | --- |
| `X` | The selected primary vector output |
| `obsm` | Other selected vector outputs, keyed by output name |
| `obs` | Sample annotations and applicable scalar outputs |
| `obs_names` | Sample identifiers |
| `uns["benchrep"]["prediction_export"]` | Export selection and output interpretation |

Canonical models normally use `embedding` as the primary key. Composite models require an explicit primary key and derive annotation behavior from their declarations. See [Selecting prediction exports](../prediction/exports.md).

Evaluation writes a separate `artifacts/anndata/evaluated_anndata.h5ad` when that export succeeds. It retains the input data and adds analysis results such as reduction coordinates in `obsm`, cluster assignments in `obs`, and metric and analysis metadata under `uns["benchrep"]`. The evaluation manifest identifies where its results were stored.

For external embeddings, [`package_matrix_as_anndata()`][benchrep.records.anndata_io.package_matrix_as_anndata] creates the same basic sample-by-feature representation. [`read_h5ad()`][benchrep.records.anndata_io.read_h5ad] and [`write_h5ad()`][benchrep.records.anndata_io.write_h5ad] read and write these files without running an evaluation workflow.

## Reconstruction bundles

Prediction exports each selected input–reconstruction pair into its own directory under `reconstructions/`. A bundle consists of:

| File | Contents |
| --- | --- |
| `input.pt` | Selected input tensor, if requested |
| `reconstruction.pt` | Corresponding reconstructed tensor, if requested |
| `obs.pt` | Dictionary of selected observation values and their `source_index` |
| `reconstruction_export_metadata.pt` | Pair names, selection settings, and exported sample counts |

The `.pt` files are written with `torch.save()`. The exporter chooses sample indices once and applies the same selection to tensors and annotations, preserving their alignment. `source_index` refers to the concatenated prediction output before selection; it is not necessarily a dataset's persistent sample identifier.

Prediction may save only inputs or only reconstructions, but reconstruction evaluation needs a compatible pair of both. For evaluation inputs and external bundles, see [Reconstruction evaluation](../evaluation/reconstructions.md).

## Metrics and figures

Evaluation writes its consolidated metric mapping to `records/metrics/metrics.json`. The file contains computed results and their available metadata, not just a flat list of scores. Non-finite numeric values are serialized as JSON `null`; inspect the surrounding metric record and status issues when interpreting them.

Figures live under `figures/anndata/` or `figures/reconstructions/`. Optional reconstruction TIFFs and numerical error maps are stored under `artifacts/reconstructions/`, separately from presentation figures. These exports can succeed or fail independently of the underlying analysis.

Use [`EvaluationExportPaths`][benchrep.records.evaluation_exports.EvaluationExportPaths] and [`PredictionExportResult`][benchrep.records.prediction_exports.PredictionExportResult] to find written artifacts programmatically. Paths for unsuccessful or disabled exports may be `None`; the matching outcomes explain why.

## Configuration and log records

An original YAML is copied to `records/configs/original_config.yaml`; an original config object is serialized there instead. A run constructed from section overrides without a base config has no original-config record. `resolved_config.yaml` records the interpreted settings in every case.

`benchrep.run.log` contains BenchRep's workflow messages. Console capture writes `stderr.log` during the wrapped execution calls; pass `capture_stdout=True` to also write `stdout.log`. These files complement the structured status records rather than replacing them.
