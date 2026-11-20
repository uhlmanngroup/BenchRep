# Canonical models

Canonical models provide a predefined autoencoder or VAE implementation. You configure compatible encoders and decoders, choose losses, and let BenchRep connect them. Their standard batch input is an image tensor under `x`; reconstruction losses compare the reconstructed image with that same `x`.

## Autoencoder

[`Autoencoder`][benchrep.architecture.models.autoencoder.Autoencoder] encodes `x` into an embedding, then decodes that embedding into a reconstruction. Its [`forward()`][benchrep.architecture.models.autoencoder.Autoencoder.forward] returns both under `embedding` and `reconstruction`.

The encoder's `output_dim` sets the embedding width. BenchRep supplies that width as the decoder's `input_dim` when omitted and supported by the decoder constructor; an explicit value must agree. The decoder's output shape must also be compatible with the input image shape.

Use this family when the encoder–decoder arrangement already fits your problem, for example to compare MLP and convolutional encoders or different reconstruction losses. It requires at least one reconstruction loss or custom objective.

## Variational autoencoder

[`VAE`][benchrep.architecture.models.vae.VAE] inserts a [`GaussianVariationalHead`][benchrep.architecture.heads.variational.GaussianVariationalHead] between the encoder and decoder. The head converts encoder features into the mean (`z_mu`) and log-variance (`z_logvar`) of a diagonal Gaussian. It forms `z_sample = z_mu + exp(0.5 * z_logvar) * noise`, with standard-normal noise. This reparameterization lets gradients flow through the sampled latent to the encoder and head. Training decodes that sample.

The following training-config fragment defines an MNIST-sized model:

```yaml
model:
  name: vae
  params:
    latent_dim: 32
encoder:
  name: mlp
  params:
    input_shape: [1, 28, 28]
    output_dim: 128
    hidden_dims: [256]
decoder:
  name: mlp
  params:
    output_shape: [1, 28, 28]
    hidden_dims: [256]
    output_activation: sigmoid
```

Here, `128` is the encoder's **feature width**, while `32` is the **latent dimension**. BenchRep constructs the variational head with those dimensions and supplies `32` to the decoder. Unlike the autoencoder, the VAE's encoder output is not itself the final representation.

The usual VAE objective combines reconstruction with Gaussian KL divergence. BenchRep requires both roles unless a custom objective is supplied; a custom objective may complement or replace them. Selecting `vae` determines the architecture, but does not add omitted objective terms for you. See [Loss requirements](losses.md#which-losses-can-a-family-use).

### Forward values and prediction

[`VAE.forward()`][benchrep.architecture.models.vae.VAE.forward] returns:

| Key | Value |
| --- | --- |
| `embedding` | Posterior mean; the same tensor as `z_mu` |
| `z_mu` | Mean of each latent Gaussian dimension |
| `z_logvar` | Log-variance of each latent Gaussian dimension |
| `z_sample` | A sampled latent vector |
| `reconstruction` | Decoder output from the selected latent source |

Training and validation use sampled reconstructions. Prediction defaults to decoding the mean for internally built VAEs. To decode samples instead, set this in the prediction config:

```yaml
inference:
  canonical_vae_reconstruction_latent_source: sample
```

This changes the reconstruction source, not the meaning of `embedding`. Calling `eval()` alone does not make the VAE's default forward computation decode the mean. For direct Python use, `forward(x, reconstruction_latent_source="mean")` selects it explicitly. [`VAE.encode()`][benchrep.architecture.models.vae.VAE.encode] returns the encoder features, whereas [`Autoencoder.encode()`][benchrep.architecture.models.autoencoder.Autoencoder.encode] returns the embedding.

## What can be changed?

Registered encoders and decoders can have different internal architectures, provided they implement the [canonical component interfaces](component-contracts.md#canonical-components-also-expose-dimensions). Loss registries can also be extended. A [custom objective](losses.md#custom-objectives) can use additional batch fields, such as labels, and may contain learnable modules of its own.

The standard canonical configuration does not, however, declare multiple model inputs, add arbitrary heads, rewire the forward graph, or select a separate reconstruction target. Nor does placing labels in a batch create a supervised objective. Use [composite assembly](composite.md) when you want these relationships expressed in configuration.

For behavior implemented directly in Python, canonical entrypoints accept a model class or instance through [runtime overrides](../workflows/overrides.md#supply-an-implementation). You can subclass the concrete AE/VAE or implement the matching [`BenchRepAutoencoderModel`][benchrep.interfaces.models.BenchRepAutoencoderModel] or [`BenchRepVAEModel`][benchrep.interfaces.models.BenchRepVAEModel] interface. Your implementation owns its training behavior and must satisfy the family's prediction contract; replacing the model also bypasses the ordinary model and objective construction settings.

## Implementation

The concrete AE and VAE inherit their corresponding BenchRep model bases, which are Lightning modules. The bases establish the workflow integration contract; the concrete classes implement the actual computation. They do not use composite assembly internally.

### Construction and module ownership

The [model builder](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/model_builder.py) constructs the selected encoder first, then uses its dimension properties to construct a compatible decoder. For VAE, it also creates the Gaussian head from the encoder's feature width and configured latent dimension.

It passes these module instances to the model constructor, together with instantiated [loss terms](losses.md#implementation) and optimizer/scheduler factories. A factory is a callable that creates the optimizer later, when Lightning calls `configure_optimizers()` with the model's parameters available.

The model stores its encoder, decoder, and head as module attributes, and its losses in `ModuleDict` containers grouped by role. PyTorch therefore includes all these submodules in parameter discovery, device movement, and checkpoint state—including trainable parameters inside custom losses.

### Forward computation

The autoencoder's forward path is simply `embedding = encoder(x)`, followed by `reconstruction = decoder(embedding)`. Both tensors are returned in a dictionary; loss calculation happens separately.

For the VAE, the corresponding operations are:

```python
features = encoder(x)
latent = variational_head(features)
reconstruction = decoder(latent["z_sample"])
```

Inside the variational head, 2 independent linear layers produce `z_mu` and `z_logvar`. Reparameterization adds scaled standard-normal noise to the mean. PyTorch records the differentiable operations through the encoder, head, and decoder, allowing the reconstruction and KL terms to update the shared parameters through backpropagation.

The head draws a sample on every call, including in evaluation mode. Selecting `mean` only changes which latent tensor enters the decoder; it does not suppress the sampled output. The VAE exposes the posterior mean as `embedding` regardless of the reconstruction source.

### Training, validation, and prediction hooks

| Hook | What the concrete model does |
| --- | --- |
| `training_step()` | Extracts `x`, calls the forward path, computes and logs each configured loss, and returns their weighted sum |
| `validation_step()` | Uses the same objective calculation with validation metric names |
| `configure_optimizers()` | Calls the optimizer factory with `self.parameters()` and optionally constructs its learning-rate scheduler |
| `predict_step()` | Calls the forward path and packages outputs with the input image and supported annotations |

Lightning handles backward passes and optimizer updates from the returned training loss. The model supplies the objective; it does not manually run a separate optimizer for each loss. See [Loss implementation](losses.md#implementation) for the common aggregation mechanism and [Training and optimization](../training/optimization.md) for execution controls.

Prediction returns [`AutoencoderPredictionOutput`][benchrep.interfaces.contracts.AutoencoderPredictionOutput] or [`VAEPredictionOutput`][benchrep.interfaces.contracts.VAEPredictionOutput], rather than the forward dictionary itself. This is where `sample_id`, `label`, and `metadata` are carried from the batch into the export interface. Prediction does not calculate the training losses.

The [model implementation reference](../../reference/components/models.md) links each constructor and method to its source. A subclass can reuse these computations; an implementation derived directly from the [model base interfaces](../../reference/interfaces.md#whole-model-overrides) must supply its own training behavior and satisfy the prediction contract.
