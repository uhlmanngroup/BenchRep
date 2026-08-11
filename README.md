# BenchRep

BenchRep is a configuration-driven, Lightning-based framework for training representation-learning models, exporting their learned representations, and evaluating those representations through reproducible, linked workflows.

> [!NOTE]
> BenchRep is an early, pre-1.0 research project under active development. Its APIs, configuration schemas, contracts, supported components, and output formats may still change. The most complete vertical slices currently cover standard autoencoder and VAE models; broader model support and more flexible integration of custom and external components are planned.

BenchRep separates a benchmark experiment into three workflows:

1. **Training** builds or accepts a model and data module, fits the model, and records checkpoints and provenance.
2. **Prediction** requires a training manifest, restores the trained model from the resolved checkpoint, runs inference, and exports embeddings and optional reconstructions.
3. **Evaluation** consumes a prediction manifest, direct artifact paths, or a combination thereof, and evaluates embeddings and optional reconstructions. Embeddings must be stored as AnnData (`.h5ad`), while reconstructions must be supplied as BenchRep-compatible PyTorch (`.pt`) artifact bundles.

The workflows can be run separately. When used together, their manifests provide the hand-off between stages and retain the provenance of the complete chain.

## Current scope

| Area                           | Built-in support                                                                                                                                                     |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model families                 | Autoencoder and VAE                                                                                                                                                  |
| Data                           | MNIST, CIFAR-10, STL-10; independent train/val transforms                                                                                                            |
| Architecture                   | MLP, configurable Conv2D, and ResNet encoders; MLP and upsampling-Conv2D decoders; configurable activations, normalization, dropout, pooling, and output activations |
| Training                       | Weighted reconstruction and regularization losses, Adam/AdamW/SGD, checkpointing, early stopping, and configurable Lightning callbacks                               |
| Tracking                       | CSV, Weights & Biases, TensorBoard, and MLflow loggers                                                                                                               |
| Reproducibility and provenance | Configurable seeding and deterministic execution; original/resolved configs, runtime-environment records, linked manifests, status reports, and local logs           |
| Prediction                     | Best/last/filename/path checkpoint selection, inherited or independent transforms, deterministic or sampled VAE reconstruction, embedding/reconstruction export      |
| Embedding evaluation           | PCA, UMAP, t-SNE; K-means, Leiden, HDBSCAN; internal and external clustering metrics; embedding statistics; classification/regression predictability probes          |
| Reconstruction evaluation      | MAE, MSE, RMSE, maximum abs error, error maps, TIFF export, and reconstruction grids                                                                                 |
| Extension                      | Registries for many individual components, or complete runtime model/datamodule instance overrides                                                                   |

Use `benchrep.inspect_registry()` to see the exact components available.

## Installation

BenchRep currently targets Python 3.11 or newer and is intended to be installed from source:

```bash
python -m pip install -e .
```

Optional dependency groups are available for scverse-backed evaluation, external loggers, model-graph export, XGBoost probes, and tests:

```bash
python -m pip install -e ".[scverse,logging,model_graph,xgboost,test]"
```


## Usage

The public workflow entry points are `train_ae()`, `train_vae()`, `predict_ae()`, `predict_vae()`, and `evaluate()`. BenchRep supports several ways to supply their configuration and runtime components:

| Mode | How it works                                                                                                                                                                                                                                                                                                                                                                        | Repository reference                                                                                                                                         |
| --- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **YAML** | Pass one YAML path per workflow. This is the canonical declarative route and the simplest way to preserve, review, and rerun a configuration.                                                                                                                                                                                                                                       | [`01_yaml_pipeline.py`](examples/usage/01_yaml_pipeline.py)                                                                                                  |
| **Typed config object** | Construct a complete Pydantic `TrainingConfig`, `PredictionConfig`, or `EvaluationConfig` and pass it as `full_config_object`. This is useful when configuration is generated or validated in Python.                                                                                                                                                                               | [`03_config_object_pipeline.py`](examples/usage/03_config_object_pipeline.py)                                                                                |
| **YAML + typed config components** | Use YAML as the base and pass selected top-level Pydantic sections through `config_components` as a mapping. Each supplied section overrides the complete matching YAML section.                                                                                                                                                           | [`02_yaml_override_pipeline.py`](examples/usage/02_yaml_override_pipeline.py)                                                                                |
| **YAML + registration** |Register a compatible Python class or callable, then refer to its registered name from the YAML. Registration must occur before invoking the workflow and be repeated in each new process.                                                                                                                                                                               | [`04_custom_dataset_pipeline.py`](examples/usage/04_custom_dataset_pipeline.py) and [`05_custom_loss_pipeline.py`](examples/usage/05_custom_loss_pipeline.py) |
| **YAML + model/datamodule override** | Pass instantiated `model` and/or `datamodule` objects directly to training or prediction. A model override replaces the config-built model, encoder, decoder, losses, and optimizer; a datamodule override replaces the configured dataset, transforms, and data module. BenchRep validates the relevant family and batch/prediction contracts according to `compatibility_policy`. | Supported by the workflow entry points; a polished `examples/usage/` script is still pending.                                                                |

A complete `full_config_object` has precedence over YAML and `config_components`. In the mixed YAML/component mode, components take precedence over matching YAML sections. Runtime model and datamodule objects then take precedence over their corresponding effective-config sections.

*Customization is layered: individual components such as datasets, encoders, losses, and optimizer factories can be registered while retaining BenchRep’s config-built workflow. Runtime model or datamodule overrides are the escape hatch for designs that cannot be expressed through those registries, although they must still satisfy BenchRep’s compatibility contracts. Every mode writes its effective configuration to `resolved_config.yaml`; runtime overrides must be supplied again, while custom registrations must be repeated in each new process.*

Prediction requires a training manifest, whereas evaluation accepts a prediction manifest, direct artifact paths, or a combination thereof. Directly passed manifest paths override those stored in the configs, avoiding hard-coded output directories between stages.


## Configuration and component discovery

BenchRep exposes lightweight runtime discovery:

- `benchrep.inspect_registry()` summarizes component registries and their extension policies.
- `benchrep.inspect_registry("dataset")` lists the components available in the dataset registry.
- `benchrep.inspect_registry("dataset", "mnist")` shows the MNIST implementation and constructor signature.
- `benchrep.inspect_config(TrainingConfig)` describes a configuration schema, including field types, defaults, constraints, and nested schema types.
- `benchrep.list_registries()` and `benchrep.list_registered_components()` return programmatic discovery records.

Configuration classes are exported from `benchrep.assembly.schemas`, and registry objects from `benchrep.assembly.registries`.

## Builders

BenchRep is builder-based internally: validated configuration is resolved into datasets, transform pipelines, datamodules, models, optimizers, trainers, and callbacks. Advanced users may call the builders in `benchrep.assembly.builders` directly, but this is not currently an orthodox usage route and does not receive the same integration testing, documentation, or compatibility attention as the workflow entry points.

## Outputs and provenance

Every workflow creates an isolated run directory:

```text
<output_root>/<stage>/<run_name>/
```

`<stage>` is `training`, `prediction`, or `evaluation`. Run names combine available project/model identity with a timestamp; a short suffix is added if the path already exists, so an existing run is not overwritten.

### Records shared by every workflow

```text
<run_name>/
└── records/
    ├── configs/
    │   ├── original_config.yaml       # only when YAML was supplied
    │   └── resolved_config.yaml       # effective validated config
    ├── logs/
    │   ├── benchrep.run.log           # BenchRep lifecycle, resolution, and status log
    │   ├── stderr.log                  # captured stderr from the workflow runtime
    │   └── stdout.log                  # optional; only when stdout capture is enabled
    └── metadata/
        ├── <stage>_manifest.yaml
        └── <stage>_runtime_environment.yaml
```

`original_config.yaml` preserves the supplied YAML exactly, including when another input had higher effective precedence. `resolved_config.yaml` records the harmonized configuration actually used after composition and validation. Runtime-object overrides are described in the manifest because their implementation cannot be serialized; custom registry components likewise need to be registered before that YAML can be reconstructed in another process.

`<stage>_runtime_environment.yaml` records the environment relevant to the workflow, including package, platform, hardware, and enabled-backend context.

Each manifest is the primary machine-readable index for its run. It records the workflow status, granular outcome or step statuses, issues, timestamps, run identity, configuration provenance, source artifacts, export paths, and summary. Unless overridden, downstream workflows resolve paths and inherited settings from the upstream manifest.

### Training outputs

```text
<training_run>/
├── checkpoints/
│   ├── *.ckpt                       # monitored top-k checkpoints
│   └── last.ckpt                    # when save_last is enabled
├── architecture/
│   └── model_graph.png              # optional best-effort torchview export
└── records/
    └── metadata/
        ├── training_manifest.yaml
        └── training_runtime_environment.yaml
```

The training manifest summarizes model and data provenance, checkpoint outputs, workflow status, and reconstructability. Training status is `completed`, `completed_with_warnings`, `completed_after_interruption`, or `failed`. Lightning-handled SIGINT/SIGTERM interruptions are finalized when possible, while non-critical inspection failures are recorded as warnings.

### Prediction outputs

```text
<prediction_run>/
├── embeddings/
│   └── embeddings.h5ad
├── reconstructions/
│   ├── input.pt
│   ├── reconstruction.pt
│   ├── obs.pt
│   └── reconstruction_export_metadata.pt
└── records/
    └── metadata/
        ├── prediction_manifest.yaml
        └── prediction_runtime_environment.yaml
```

`embeddings.h5ad` is the standard embedding artifact. The primary representation is stored in `adata.X`; additional requested representation keys are stored in `adata.obsm`. Available `sample_id`, `label`, and prediction metadata are preserved in `adata.obs`.

Reconstruction exports contain a configurable subset of predictions. `input.pt` and/or `reconstruction.pt` store the tensors, `obs.pt` their annotations, and `reconstruction_export_metadata.pt` selection details. The manifest records counts, strategy, stratification coverage, etc.

Inference and each export are tracked independently, preserving successful artifacts and producing a top-level status of `completed`, `completed_with_warnings`, `partially_completed`, or `failed`. If `partially_completed` or `failed`, BenchRep writes the manifest first and then raises an error, preserving the failure record and any usable artifacts while breaking a workflow chain.

### Evaluation outputs

```text
<evaluation_run>/
├── artifacts/
│   ├── embeddings/
│   │   └── evaluated_embeddings.h5ad
│   └── reconstructions/
│       ├── inputs/
│       │   └── input_example_*.tif
│       ├── predictions/
│       │   └── prediction_example_*.tif
│       └── error_maps/
│           └── <error_kind>/
│               └── error_map_<error_kind>_example_*.tif
├── figures/
│   ├── embeddings/
│   │   ├── reductions/
│   │   │   ├── uncolored/
│   │   │   └── colored_by/<obs_key>/
│   │   └── diagnostics/
│   │       ├── pca/
│   │       └── clustering/<cluster_key>/
│   └── reconstructions/
│       └── reconstruction_grid_channel_*_page_*.<format>
└── records/
    ├── metrics/
    │   └── metrics.json
    └── metadata/
        ├── evaluation_manifest.yaml
        └── evaluation_runtime_environment.yaml
```

`evaluated_embeddings.h5ad` is the complete embedding-side result. It preserves the evaluated embedding in `adata.X` and stores generated reductions in `adata.obsm`, cluster assignments/probabilities in `adata.obs`, neighbor graphs in `adata.obsp`, and BenchRep method metadata and metrics under `adata.uns["benchrep"]`. The evaluation manifest includes these internal AnnData locations.

`metrics.json` consolidates scalar metrics and compact array summaries across embedding, clustering, predictability, and reconstruction evaluation. Reconstruction TIFFs are exported per example with multichannel support, while paginated reconstruction grids can include error maps. Embedding figures include annotation-colored projections, and PCA variance and clustering diagnostics, with configurable DPI/output formats.

Evaluation uses dependency-aware pipelines whose steps run, fail, or skip independently according to upstream outcomes. The manifest records step outcomes and aggregate statuses, allowing recoverable failures without aborting unrelated steps.

### Workflow linkage

Prediction manifests retain their training source and selected checkpoint, while evaluation manifests record whether inputs came from prediction outputs, direct paths, or both. This preserves end-to-end provenance without requiring a complete workflow chain and allows evaluation of artifacts generated entirely outside BenchRep.

## Development status and roadmap

BenchRep is already usable for research experiments within its current autoencoder/VAE scope, but it should not yet be treated as a stable, comprehensive benchmarking platform. Important ongoing work includes:

- expanding the breadth of built-in datasets, transforms, architectures, evaluation methods, metrics, and plots;
- adding contrastive and supervised model families, loss functions, heads, prediction contracts, and evaluation paths;
- supporting an arbitrary `custom` loss that can consume multiple model outputs and targets; currently, registered custom losses remain confined to reconstruction or regularization roles;
- adding loss-weight schedules and warm-up policies and potentially setting the stage for weight tuning;
- adding higher-level experiment and study orchestration above the individual workflows;
- continuing schema documentation, resilience testing, and end-to-end coverage.

The maintained examples and runtime discovery functions are the most reliable guide to the current API while this work continues.