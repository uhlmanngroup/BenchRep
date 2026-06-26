from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from benchrep.architecture.models import (
    AutoencoderPredictionOutput,
    VAEPredictionOutput,
)

PredictionOutputT = TypeVar("PredictionOutputT", covariant=True)

@runtime_checkable
class BenchRepPredictor(Protocol[PredictionOutputT]):
    """Object whose prediction step returns a BenchRep-compatible output."""

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
    ) -> PredictionOutputT:
        ...


AutoencoderPredictor = BenchRepPredictor[AutoencoderPredictionOutput]
VAEPredictor = BenchRepPredictor[VAEPredictionOutput]