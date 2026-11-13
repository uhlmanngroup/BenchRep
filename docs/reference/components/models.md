# Model implementations

Concrete models built by BenchRep, with their constructors, forward contracts, and Lightning hooks. Use the [workflow entrypoints](../workflows.md) for normal runs; these entries support direct construction, subclassing, and inspection of the implementation. Whole-model overrides remain limited to the canonical families.

The canonical constructors accept instantiated encoders, decoders, and [`LossTerm`][benchrep.architecture.losses.base.LossTerm] containers. `CompositeModel` instead receives a resolved execution specification and instantiated components from the builder; its constructor is not an alternative config schema.

## Autoencoder

::: benchrep.architecture.models.autoencoder.Autoencoder
    options:
      heading_level: 3
      members:
        - forward
        - encode
        - decode
        - training_step
        - validation_step
        - test_step
        - predict_step
        - configure_optimizers

## Variational autoencoder

::: benchrep.architecture.models.vae.VAE
    options:
      heading_level: 3
      members:
        - forward
        - encode
        - decode
        - training_step
        - validation_step
        - test_step
        - predict_step
        - configure_optimizers

## Composite model

::: benchrep.architecture.models.composite.CompositeModel
    options:
      heading_level: 3
      members:
        - forward
        - training_step
        - validation_step
        - test_step
        - predict_step
        - configure_optimizers
