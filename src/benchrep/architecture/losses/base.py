from dataclasses import dataclass

from torch import nn


@dataclass
class LossTerm:
    loss: nn.Module
    weight: float