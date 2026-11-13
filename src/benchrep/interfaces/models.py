from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import lightning as L

from benchrep.interfaces.contracts import (
    AutoencoderPredictionOutput,
    VAEPredictionOutput,
)


class BenchRepAutoencoderModel(L.LightningModule, ABC):
    """Base class for a whole-model autoencoder runtime override.

    Subclass this LightningModule and implement ``predict_step()`` with the
    family's prediction output contract. Training also requires the Lightning
    training and optimizer hooks used by your implementation.

    Supply the class or an instance through ``TrainingOverrides.model`` or
    ``PredictionOverrides.model`` for the matching canonical entrypoint.
    This interface is not used to replace a configuration-built composite model.
    """
    @abstractmethod
    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
    ) -> AutoencoderPredictionOutput:
        """Produce one batch of autoencoder predictions.

        Parameters
        ----------
        batch
            Batch supplied by the datamodule.
        batch_idx
            Zero-based batch index supplied by Lightning.

        Returns
        -------
        AutoencoderPredictionOutput
            Embeddings, inputs, reconstructions, and the family's additional fields.
            Observation annotations must align with the batch samples.
        """
        raise NotImplementedError


class BenchRepVAEModel(L.LightningModule, ABC):
    """Base class for a whole-model VAE runtime override.

    Subclass this LightningModule and implement ``predict_step()`` with the
    family's prediction output contract. Training also requires the Lightning
    training and optimizer hooks used by your implementation.

    Supply the class or an instance through ``TrainingOverrides.model`` or
    ``PredictionOverrides.model`` for the matching canonical entrypoint.
    This interface is not used to replace a configuration-built composite model.
    """
    @abstractmethod
    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
    ) -> VAEPredictionOutput:
        """Produce one batch of VAE predictions.

        Parameters
        ----------
        batch
            Batch supplied by the datamodule.
        batch_idx
            Zero-based batch index supplied by Lightning.

        Returns
        -------
        VAEPredictionOutput
            Embeddings, inputs, reconstructions, and the family's additional fields.
            Observation annotations must align with the batch samples.
        """
        raise NotImplementedError