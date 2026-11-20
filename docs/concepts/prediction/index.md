# Prediction

Prediction applies a trained model to a dataset and exports representations, reconstructions, or both. It is a separate run, with its own configuration, directory, status report, and manifest, linked to the training run that supplied the model.

Use [`predict_ae()`][benchrep.predict_ae], [`predict_vae()`][benchrep.predict_vae], or [`predict_composite()`][benchrep.predict_composite] to match the training model family. Supply a training manifest through `source.training_manifest_path` or [`PredictionOverrides`][benchrep.PredictionOverrides]. The manifest provides the recorded configuration and checkpoint locations; the prediction config selects what to read, how to infer, and what to export.

| Decision | Configuration |
| --- | --- |
| Training run and checkpoint | `source` |
| Prediction samples and processing | `dataset`, `transform_pipelines`, `data` |
| Randomness and model-specific inference behavior | `inference` |
| Representations, annotations, and reconstruction examples to save | `exports` |

Prediction does not optimize the model or calculate its training objective. Its exports become inputs to [evaluation](../evaluation/index.md), which measures their properties independently.

- **[Loading models for prediction](loading-models.md):** checkpoint selection, inherited settings, runtime implementations, and the execution mechanism.
- **[Selecting prediction exports](exports.md):** AnnData contents, reconstruction bundles, sample selection, and export handling.

See [Run directories and result objects](../results/runs.md) for the returned predictions and artifact paths, and [Statuses and failures](../results/statuses.md) for failure behavior.
