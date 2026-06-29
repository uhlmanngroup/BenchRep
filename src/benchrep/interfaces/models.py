from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import lightning as L

from benchrep.interfaces.contracts import (
    AutoencoderPredictionOutput,
    VAEPredictionOutput,
)


class BenchRepAutoencoderModel(L.LightningModule, ABC):
    @abstractmethod
    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
    ) -> AutoencoderPredictionOutput:
        raise NotImplementedError


class BenchRepVAEModel(L.LightningModule, ABC):
    @abstractmethod
    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
    ) -> VAEPredictionOutput:
        raise NotImplementedError