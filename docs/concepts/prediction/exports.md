# Selecting prediction exports

Prediction produces batch-level model outputs; exports turn selected parts of those outputs into reusable artifacts. AnnData stores representations and annotations, while reconstruction bundles retain image tensors and their associated observations. The 2 exports can be enabled independently.

## Select representations for AnnData

AnnData export is enabled by default. For a canonical VAE, this fragment saves the posterior mean as the primary representation and also retains sampled latents:

```yaml
exports:
  anndata:
    enabled: true
    mode: custom
    keys: [embedding, z_sample]
    primary_key: embedding
```

| Selected value | AnnData location |
| --- | --- |
| Primary vector-valued output | `adata.X` |
| Other selected vector-valued outputs | `adata.obsm[output_name]` |
| Selected scalar outputs | Columns in `adata.obs` |
| Contextual annotations | Observation index or columns in `adata.obs` |

Image outputs are not selected through AnnData export. The primary key must be vector-valued and included in the selected keys; it is not also duplicated in `obsm`.

### Available keys depend on the model

| Family | Supported keys | Default primary key |
| --- | --- | --- |
| Autoencoder | `embedding` | `embedding` |
| VAE | `embedding`, `z_mu`, `z_logvar`, `z_sample` | `embedding` |
| Composite | Non-image outputs declared under `produces` | Must be supplied explicitly |

`mode: all` selects supported non-image outputs. For VAE, it omits `z_mu` because `embedding` already contains the same posterior mean; custom selection can include `z_mu` explicitly. Composite keys are bare declaration names, such as `z_mu`, not `produces.z_mu`.

Both `embedding_vector` and `projection_vector` outputs can serve as a composite primary key. Roles determine available structures and export behavior; they do not restrict the primary representation to a field literally named `embedding`. See [Declared roles](../models/composite.md#declared-roles).

### Observation annotations

Output selection does not determine which contextual annotations are retained:

- Canonical prediction outputs provide optional `sample_id`, `label`, and a `metadata` mapping. Sample IDs become the observation index; labels and metadata become columns.
- Composite outputs provide declared scalar inputs and `batch_metadata`. Scalar inputs and group metadata become columns; an optional metadata field with role `index` supplies the observation index. Selected scalar model outputs also become columns.

Annotations must align with the exported samples. Composite names that would collide across these sources are rejected. A field merely existing in the original dataset does not make it exportable: it must be included in the model's supported prediction output. See [Datasets and sample structure](../data/datasets.md).

## Select reconstruction pairs

Reconstruction export is disabled by default. Enabling it with `mode: all` uses the canonical `input`/`reconstruction` pair, or infers composite pairs from reconstruction-loss wiring.

Use explicit pairs when a composite model has several image paths, no ordinary reconstruction loss, or ambiguous loss wiring:

```yaml
exports:
  reconstructions:
    enabled: true
    mode: custom
    pairs:
      clean_reconstruction:
        input: clean_image
        reconstruction: reconstructed_image
    n_examples: 32
    selection: random
    seed: 137
```

This assumes `clean_image` is an image-valued declaration under `expects`, and `reconstructed_image` is declared with role `reconstruction_image` under `produces`. Use bare names, without namespace prefixes. The pair ID becomes its bundle identifier and directory name. For canonical models, explicit pair fields are always `input` and `reconstruction`.

Composite `all` mode inspects each configured reconstruction loss for exactly one image input and one reconstruction-image output. It deduplicates identical pairs; it does not form every possible input/output combination. If the wiring cannot establish an unambiguous pair, specify it explicitly.

## Choose which examples to retain

`n_examples` applies to reconstruction export, not AnnData or inference. An integer selects at most that many predicted samples; `all` retains every predicted sample in order.

For a subset, `selection: first` takes the first examples and `random` samples without replacement. An optional `stratify_by` annotation makes random selection approximately balanced across its distinct values, rather than proportional to their original frequencies. If the requested subset is smaller than the number of groups, some groups cannot be represented and BenchRep records a warning.

The selection seed uses a local generator and defaults to the resolved inference seed for random selection. It does not alter the global prediction seed. With `n_examples: all`, selection settings do not change the retained examples; a requested stratification field must still exist.

Each pair has a separate bundle containing `input.pt`, `reconstruction.pt`, `obs.pt`, and `reconstruction_export_metadata.pt`. `include_input` and `include_reconstruction` can omit one tensor artifact, but at least one must remain enabled. <strong class="highlight">Reconstruction evaluation requires both input and reconstruction tensors.</strong> See [Artifact formats](../results/artifacts.md) for their contents.

## Implementation: from prediction batches to artifacts

Before execution, the resolver converts model contracts/declarations and export settings into a concrete `PredictionExportSpec`: selected keys, scalar/vector structures, observation sources, and reconstruction pairs. This avoids guessing what output names mean after inference.

After prediction, runtime validation checks the actual batch outputs against their family contract. The [exporter](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/records/prediction_exports.py) then detaches selected tensors, moves them to CPU, and concatenates them along the batch dimension. AnnData packaging uses the resolved primary key and observation sources. Reconstruction packaging selects sample indices, applies them to tensors and annotations together, and records those indices and selection settings in the bundle.

The AnnData branch and each reconstruction pair are exported independently. An error in one branch does not stop the remaining branches; their outcomes are recorded separately. However, a partial or failed prediction run writes its manifest and then raises, rather than returning a successful result object. Invalid prediction outputs prevent export altogether. See [Statuses and failures](../results/statuses.md) for how to interpret these outcomes.

All selection fields are documented in the [prediction schema](../../reference/schemas/prediction.md). The returned [`PredictionExportResult`][benchrep.records.prediction_exports.PredictionExportResult] exposes successful artifact paths and the recorded branch outcomes when the workflow returns normally.
