from torch import nn


class LossTerm(nn.Module):
    """Container for one configured weighted loss term."""

    def __init__(
        self,
        *,
        loss: nn.Module,
        weight: float,
    ) -> None:
        super().__init__()

        self.loss = loss
        self.weight = weight
