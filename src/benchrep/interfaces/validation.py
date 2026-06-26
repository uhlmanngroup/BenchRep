from __future__ import annotations

from typing import Any

import lightning as L

from benchrep.interfaces.models import BenchRepPredictor


def validate_external_model(model: Any) -> None:
    """Validate user-provided model override at workflow entrypoints"""

    if not isinstance(model, L.LightningModule):
        raise TypeError(
            "Model override must be a `lightning.LightningModule` object."
            f"Got {type(model).__name__}."
        )

    if not isinstance(model, BenchRepPredictor):
        raise TypeError(
            "Model override must implement `predict_step(batch, batch_idx)`"
            "and return a BenchRep-compatible prediction output dataclass."
        )