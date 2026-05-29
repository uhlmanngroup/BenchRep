from dataclasses import dataclass

from torch import nn


@dataclass
class LossTerm:
    """Container for one configured weighted loss term.

    ``LossTerm`` stores an instantiated loss module and the scalar weight used
    when the model combines losses during training.

    Tensor routing is handled by the model class. For example, an autoencoder
    calls reconstruction losses with its reconstruction and input tensors, while
    a VAE calls reconstruction losses with reconstruction/input tensors and KL
    losses with posterior parameters.

    Args:
        loss:
            Instantiated loss module.
        weight:
            Scalar multiplier applied to this loss term.
    """

    loss: nn.Module
    weight: float