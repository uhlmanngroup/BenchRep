from functools import partial
from pathlib import Path

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from benchrep.architecture.decoders.mlp import MLPDecoder
from benchrep.architecture.encoders.mlp import MLPEncoder
from benchrep.architecture.losses import (
    LinearSchedule,
    LossTerm,
    MSEReconstructionLoss,
)
from benchrep.architecture.models import Autoencoder


class RecordingLossTerm(LossTerm):
    def __init__(self, interval: str) -> None:
        super().__init__(
            loss=MSEReconstructionLoss(),
            weight=2.0,
            weight_schedule=LinearSchedule(
                end=4,
                start_factor=0.2,
                end_factor=1.0,
            ),
            schedule_interval=interval,
        )
        self.observed_weights: list[float] = []

    def effective_weight(self, *, epoch: int, step: int) -> float:
        weight = super().effective_weight(epoch=epoch, step=step)
        self.observed_weights.append(weight)
        return weight


def _make_model(interval: str) -> tuple[Autoencoder, RecordingLossTerm]:
    torch.manual_seed(137)
    term = RecordingLossTerm(interval)
    model = Autoencoder(
        encoder=MLPEncoder(
            input_shape=(2,),
            output_dim=2,
            hidden_dims=(),
        ),
        decoder=MLPDecoder(
            input_dim=2,
            output_shape=(2,),
            hidden_dims=(),
        ),
        reconstruction_losses={"mse": term},
        custom_objective_losses={},
        optimizer_factory=partial(torch.optim.SGD, lr=0.01),
    )
    return model, term


def _make_trainer(max_epochs: int) -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        deterministic=True,
    )


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("step", [0.4, 0.4, 0.8, 0.8, 1.2, 1.2, 1.6, 1.6]),
        ("epoch", [0.4] * 4 + [0.8] * 4),
    ],
)
def test_schedule_progress_survives_accumulation_and_resume(
    tmp_path: Path,
    interval: str,
    expected: list[float],
) -> None:
    loader = DataLoader(
        [{"x": torch.tensor([1.0, 2.0])} for _ in range(8)],
        batch_size=2,
        shuffle=False,
    )

    # Uninterrupted reference: four batches and two optimizer steps per epoch.
    full_model, full_term = _make_model(interval)
    _make_trainer(max_epochs=2).fit(full_model, train_dataloaders=loader)
    assert full_term.observed_weights == pytest.approx(expected)

    # Stop at an epoch boundary and save the training state.
    first_model, first_term = _make_model(interval)
    first_trainer = _make_trainer(max_epochs=1)
    first_trainer.fit(first_model, train_dataloaders=loader)

    checkpoint = tmp_path / "resume.ckpt"
    first_trainer.save_checkpoint(checkpoint)

    # Reconstruct the schedule; progress must come from the restored trainer.
    resumed_model, resumed_term = _make_model(interval)
    _make_trainer(max_epochs=2).fit(
        resumed_model,
        train_dataloaders=loader,
        ckpt_path=str(checkpoint),
    )

    combined_weights = (
        first_term.observed_weights + resumed_term.observed_weights
    )
    assert combined_weights == pytest.approx(expected)

    for name, value in full_model.state_dict().items():
        torch.testing.assert_close(
            resumed_model.state_dict()[name],
            value,
            rtol=0,
            atol=0,
        )