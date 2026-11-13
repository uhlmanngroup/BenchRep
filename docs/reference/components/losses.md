# Built-in losses

These modules compute raw losses. Weighting and scheduling are handled separately by [`LossTerm`][benchrep.architecture.losses.base.LossTerm]. Registered names and aliases appear in the [catalogue](../components.md); [loss contracts](../interfaces.md#losses-and-weight-schedules) describe how to register your own implementation.

## Reconstruction

::: benchrep.architecture.losses.reconstruction.MSEReconstructionLoss
    options:
      heading_level: 3
      members:
        - forward

::: benchrep.architecture.losses.reconstruction.MAEReconstructionLoss
    options:
      heading_level: 3
      members:
        - forward

## Regularization

::: benchrep.architecture.losses.regularization.GaussianKLDivergenceLoss
    options:
      heading_level: 3
      members:
        - forward

## Contrastive learning

::: benchrep.architecture.losses.contrastive.TripletMarginContrastiveLoss
    options:
      heading_level: 3
      members:
        - forward

## Classification

::: benchrep.architecture.losses.classification.CrossEntropyClassificationLoss
    options:
      heading_level: 3
      members:
        - forward

## Regression

::: benchrep.architecture.losses.regression.MSERegressionLoss
    options:
      heading_level: 3
      members:
        - forward
