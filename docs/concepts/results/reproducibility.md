# Reproducibility

Reproducibility involves the data, random choices, numerical execution, and implementation versions—not just a seed. BenchRep exposes controls for these choices and records the environment in which a workflow ran, so differences between runs can be investigated.

## Training randomness and numerical settings

A typical training configuration makes the main choices explicit:

```yaml
reproducibility:
  seed: 137
  seed_workers: true
  float32_matmul_precision: highest
trainer:
  deterministic: true
  benchmark: false
  precision: 32-true
```

BenchRep calls Lightning's `seed_everything()` before constructing config-built data and models, and passes the seed to its datamodule for the training/validation split. `seed_workers` enables Lightning's worker-seeding support. The split uses a separately seeded PyTorch generator, while model initialization and stochastic operations use the relevant runtime random generators.

These numerical controls have different purposes:

| Setting | Purpose |
| --- | --- |
| `trainer.deterministic` | Controls Lightning/PyTorch deterministic execution behavior |
| `trainer.benchmark` | Controls cuDNN's convolution algorithm benchmarking |
| `trainer.precision` | Selects Lightning's numerical precision mode |
| `reproducibility.float32_matmul_precision` | Controls internal precision for float32 matrix multiplication; it does not change tensor dtypes |

Deterministic settings can restrict available operations or reduce performance. They do not guarantee identical results across hardware, dependency versions, or changed execution paths.

!!! warning "Instances may already have made random choices"
    An override instance is constructed before the workflow receives it. Seeding inside `train_vae()` cannot retroactively reproduce its weight initialization or earlier dataset preparation. If those choices need to be repeatable, seed the relevant code before constructing the instance. Class overrides constructed inside the workflow run after its global seeding, though custom code may still manage its own generators.

## Prediction inherits settings, not random-generator state

Prediction can inherit the training seed, worker-seeding choice, deterministic setting, and float32 matrix-multiplication precision. It seeds its own execution from those resolved values; it does not resume the training process's random-generator state.

Random reconstruction-example selection has its own local generator. Its configured seed defaults to the resolved prediction seed, allowing example selection to be controlled separately from inference. Changing the number or order of prediction samples can still change the selected examples.

For internally built VAEs, decoding the posterior mean removes latent sampling from the reconstruction path. It does not guarantee that every other operation is deterministic. External models own their prediction behavior. See [Canonical models](../models/canonical.md) and [Selecting prediction exports](../prediction/exports.md).

Reusing the training seed also does not select its held-out validation subset automatically. Prediction data selection and transform inheritance are separate decisions; see [Splits and dataloaders](../data/datamodules.md).

## Evaluation has component-specific random settings

Evaluation does not apply a single global training-style seed. Stochastic analysis components receive their own settings, such as `random_state` in reduction, clustering, probe, cross-validation, or figure parameters where supported.

Specify those settings for the analyses you intend to compare. A fixed clustering seed does not also fix a UMAP projection or a probe's data split. See [Reductions and clustering](../evaluation/reductions-clustering.md), [Predictability probes](../evaluation/predictability.md), and the [evaluation schema reference](../../reference/schemas/evaluation.md).

## Runtime-environment records

Each workflow writes `records/metadata/<stage>_runtime_environment.yaml`. The record combines:

- Requested and effective execution settings, including applicable seeds and trainer state.
- BenchRep and installed Python package versions, Python and operating-system details.
- Available hardware and accelerator-stack information.
- BenchRep source-repository commit, branch, tag, and dirty state when available.

Collection warnings identify information that could not be obtained. The record captures the environment; it does not install dependencies, bundle the source, or restore random-generator states.

When comparing runs, inspect the resolved configurations and upstream checkpoint/artifact selections first, then use the environment records to identify software or execution differences. Keep the underlying data and custom implementations available alongside those records. [Provenance](provenance.md) explains the separate question of whether configuration can reconstruct a run's components.
