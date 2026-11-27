# Bring your own model or datamodule

<strong class="highlight">How can I use my own model or data-loading code while keeping BenchRep's workflows and evaluation tools?</strong> Pass a complete model or datamodule as a runtime override. To supply individual components selected by name in a config, use [component registration](register-components.md).

This tutorial walks through three separate examples. We use a custom autoencoder and datamodule, train a YOYO classifier with data configured in YAML, and export pretrained vision transformer embeddings without a BenchRep training run. The snippets show how each script passes its implementation to the workflow and which settings stay in YAML.

<strong class="highlight">Requirements.</strong> The two YOYO examples use the `scverse` extra for t-SNE. The canonical override example needs only the core installation. If you need to add `scverse`, run the following command.

```bash
python -m pip install "benchrep[scverse] @ git+https://github.com/uhlmanngroup/BenchRep.git"
```

## Choose the integration point

| What you supply                   | Batch format | Model requirements |
|-----------------------------------| --- | --- |
| Canonical model only              | Matches the dictionary batches from config-built data, including `x` | Corresponding BenchRep model base and prediction contract |
| Canonical model and datamodule    | Defined by your model and datamodule | Corresponding BenchRep model base and prediction contract |
| Datamodule only                   | Matches the config-built model’s canonical inputs or Composite declarations | Model built from config |
| YOYO model, with either data path | Matches your model’s inputs | `LightningModule` with outputs compatible with the selected exports |

Composite models can only be config-built, but the family accepts an optional datamodule override. [Tune with Optuna](optuna.md#evaluate-on-the-validation-split) uses one to run prediction on the validation subset. See [External models and datamodules](../concepts/extensibility/runtime-implementations.md#compatibility-checks-and-runtime-checks) for compatibility checks.

## Run the examples

From the directory where you want to keep your data and outputs, choose an example to run.

=== "Canonical overrides"

    ```bash
    python -m benchrep.examples.usage.12_canonical_runtime_overrides
    ```

=== "YOYO training"

    ```bash
    python -m benchrep.examples.usage.13_yoyo_classifier_pipeline
    ```

=== "Pretrained prediction"

    ```bash
    python -m benchrep.examples.usage.14_pretrained_yoyo_prediction
    ```

???+ warning "Runtime and resource requirements"
    The first run downloads the corresponding dataset. Pretrained prediction also downloads ViT-B/16 weights. The STL-10 archive and extracted files take about 5.4 GiB together. Removing the archive after extraction frees roughly half that space.

    The autoencoder trains for up to 5 epochs and the classifier for up to 10. Lower `trainer.max_epochs` or set batch limits in their training YAMLs to shorten training. For pretrained prediction, lower `BATCH_SIZE` in the script to reduce GPU memory use. To check prediction on a subset, set `data.max_batches` in its prediction YAML and skip the evaluation call.

## Custom canonical autoencoder and datamodule

`FashionMNISTDataModule` loads [torchvision’s Fashion-MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) directly and handles preprocessing, splits, and dataloaders. Because the script supplies both the datamodule and the model, they can use `(images, labels)` batches.

[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/12_canonical_runtime_overrides.py) · [View configs](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/configs/12_canonical_runtime_overrides)

`FashionMNISTAutoencoder` inherits [`BenchRepAutoencoderModel`][benchrep.interfaces.models.BenchRepAutoencoderModel] and defines its own loss and optimizer. Its `predict_step()` returns an [`AutoencoderPredictionOutput`][benchrep.interfaces.contracts.AutoencoderPredictionOutput] for BenchRep’s standard exports. BenchRep checks the `predict_step()` return annotation before training and prediction. You can use your own dataclass if it declares the required fields with the expected types.

```python
def predict_step(self, batch, batch_idx: int) -> AutoencoderPredictionOutput:
    images, labels = batch
    embedding = self.encoder(images)
    return AutoencoderPredictionOutput(
        embedding=embedding,
        input=images,
        reconstruction=self.decoder(embedding),
        label=labels,
    )
```

Pass both classes through [`TrainingOverrides`][benchrep.TrainingOverrides] and set their constructor arguments under `overrides` in the training YAML.

```python
training_result = train_ae(
    config=example_config_path(EXAMPLE, stage="training"),
    overrides=TrainingOverrides(
        model=FashionMNISTAutoencoder,
        datamodule=FashionMNISTDataModule,
    ),
)
```

```yaml
overrides:
  model:
    params:
      latent_dim: 32
      learning_rate: 0.001
  datamodule:
    params:
      root: data/fashion_mnist
      batch_size: 128
      val_fraction: 0.1
      seed: 137
```

For prediction, the script passes the training manifest path and the returned model and datamodule instances through [`PredictionOverrides`][benchrep.PredictionOverrides] and leaves their constructor parameters empty. The prediction YAML selects `source.checkpoint: best`, so BenchRep loads that checkpoint into the supplied model. The datamodule serves the Fashion-MNIST test split. See [Classes and instances](../concepts/extensibility/runtime-implementations.md#classes-and-instances) and [Loading models for prediction](../concepts/prediction/loading-models.md).

## Train with your own model behavior

[YOYO](../concepts/models/yoyo.md) runs `CIFAR10Classifier` as a complete `LightningModule`. The classifier defines its own training loss, optimizer, and prediction outputs. BenchRep builds the CIFAR-10 data pipeline from the YAML.

[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/13_yoyo_classifier_pipeline.py) · [View configs](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/configs/13_yoyo_classifier_pipeline)

Pass the classifier class to [`train_yoyo()`][benchrep.train_yoyo].

```python
training_result = train_yoyo(
    config=example_config_path(EXAMPLE, stage="training"),
    overrides=TrainingOverrides(model=CIFAR10Classifier),
)
```

The classifier reads `x` and `label` from each batch. The training YAML sets `input: x` because YOYO requires explicit transform inputs. Its `checkpointing.monitor` is `val/cross_entropy`, the validation metric logged by the classifier. With the default YOYO checkpoint settings, omitting the monitor saves only the last checkpoint. See [Callbacks and checkpoints](../concepts/training/callbacks.md).

For [`predict_yoyo()`][benchrep.predict_yoyo], pass the class again with the training manifest path. The prediction YAML omits `overrides.model`, so BenchRep reuses the constructor arguments recorded in the manifest. The YAML selects the best checkpoint and runs prediction on the CIFAR-10 test split.

YOYO AnnData export requires `mode: custom`, explicit `keys` from `predict_step()`, and a `primary_key`. This example exports `features` to `AnnData.X` and `logits` to `AnnData.obsm`. The returned `label` and `sample_id` provide observation annotations.

```yaml
exports:
  anndata:
    enabled: true
    mode: custom
    keys:
      - features
      - logits
    primary_key: features
  reconstructions:
    enabled: false
```

Evaluation clusters and probes the feature vectors in `AnnData.X`. See [YOYO output selection](../concepts/prediction/exports.md#yoyo-output-selection) and [Predictability probes](../concepts/evaluation/predictability.md).

## Predict with a pretrained model

The script removes the classification head from torchvision’s ImageNet-pretrained ViT-B/16 and wraps the backbone in `PretrainedViT` to export image embeddings. Its custom datamodule applies the weights’ preprocessing to the labeled STL-10 training split.

[:material-script-text-outline: View script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/14_pretrained_yoyo_prediction.py) · [View configs](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/configs/14_pretrained_yoyo_prediction)

Pass both initialized instances to `predict_yoyo()` for standalone prediction.

```python
weights = ViT_B_16_Weights.IMAGENET1K_V1
backbone = vit_b_16(weights=weights)
backbone.heads = nn.Identity()
model = PretrainedViT(backbone)
datamodule = STL10DataModule(weights.transforms(), batch_size=BATCH_SIZE)

prediction_result = predict_yoyo(
    config=example_config_path(EXAMPLE, stage="prediction"),
    overrides=PredictionOverrides(model=model, datamodule=datamodule),
)
```

`predict_step()` returns 768-dimensional embeddings and their labels. The prediction YAML sets `source.checkpoint: null` to keep the loaded weights and selects `embedding` as the `primary_key` for `AnnData.X`.

To fine-tune a pretrained encoder inside a config-built model, see [Register a pretrained encoder](register-components.md#register-a-pretrained-encoder).

## Evaluate and inspect the records

Each script passes the prediction manifest path to [`evaluate()`][benchrep.evaluate]. Evaluation reads the saved exports. The Fashion-MNIST example evaluates embeddings and reconstructions. The two YOYO examples evaluate embeddings and produce PCA and t-SNE plots.

The manifests record the runtime classes. Keep their Python implementations available to rerun training or prediction in another process. For a pretrained instance, also record the weights identifier and preprocessing. See [Trace an evaluation result](trace-an-evaluation-result.md) and [Manifests and provenance](../concepts/results/provenance.md).
