import pytest
import torch
from lightning.pytorch import Callback, Trainer, seed_everything
from torch.utils.data import DataLoader
from lightning.pytorch import LightningModule

from benchrep.assembly.builders.optimizer_builder import (
    build_lr_scheduler_factory,
)
from benchrep.assembly.schemas import TrainingLRSchedulerConfig

from benchrep.architecture.decoders.mlp import MLPDecoder
from benchrep.architecture.encoders.mlp import MLPEncoder
from benchrep.assembly.builders.model_builder import build_autoencoder
from benchrep.assembly.resolvers.loss_resolver import LossSpec
from benchrep.assembly.schemas import TrainingOptimizerConfig


class RecordLearningRates(Callback):
    def __init__(self):
        self.values = []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.values.append(optimizer.param_groups[0]["lr"])


def make_model(interval):
    seed_everything(137, workers=True)

    optimizer_config = {
        "name": "sgd",
        "params": {"lr": 0.1, "momentum": 0.9},
    }
    if interval is not None:
        optimizer_config["lr_scheduler"] = {
            "name": "step",
            "params": {"step_size": 1, "gamma": 0.5},
            "interval": interval,
        }

    return build_autoencoder(
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
        optimizer=TrainingOptimizerConfig.model_validate(optimizer_config),
        loss_specs=(
            LossSpec(
                loss_role="reconstruction",
                loss_id="mse",
                registry_entry_name="mse",
                weight=1.0,
                constructor_params={"reduction": "mean"},
            ),
        ),
    )


def make_loader():
    samples = [
        {"x": torch.tensor([1.0, 2.0])}
        for _ in range(8)
    ]
    return DataLoader(samples, batch_size=2, shuffle=False)


def make_trainer(root, epochs, recorder):
    return Trainer(
        default_root_dir=str(root),
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        accumulate_grad_batches=2,
        callbacks=[recorder],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        deterministic=True,
    )


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (None, [0.1, 0.1, 0.1, 0.1]),
        ("epoch", [0.1, 0.1, 0.05, 0.05]),
        ("step", [0.1, 0.05, 0.025, 0.0125]),
    ],
)
def test_runtime_lr_and_checkpoint_resume(tmp_path, interval, expected):
    # Reference: two uninterrupted epochs.
    full_model = make_model(interval)
    full_recorder = RecordLearningRates()
    full_trainer = make_trainer(tmp_path / "full", 2, full_recorder)
    full_trainer.fit(full_model, train_dataloaders=make_loader())

    assert full_recorder.values == pytest.approx(expected)

    # Interrupted run: save after one complete epoch.
    first_model = make_model(interval)
    first_recorder = RecordLearningRates()
    first_trainer = make_trainer(tmp_path / "first", 1, first_recorder)
    first_trainer.fit(first_model, train_dataloaders=make_loader())

    checkpoint = tmp_path / "resume.ckpt"
    first_trainer.save_checkpoint(str(checkpoint))

    # Resume with newly constructed model, optimizer and scheduler.
    resumed_model = make_model(interval)
    resumed_recorder = RecordLearningRates()
    resumed_trainer = make_trainer(
        tmp_path / "resumed", 2, resumed_recorder
    )
    resumed_trainer.fit(
        resumed_model,
        train_dataloaders=make_loader(),
        ckpt_path=str(checkpoint),
    )

    combined = first_recorder.values + resumed_recorder.values
    assert combined == pytest.approx(expected)
    assert full_trainer.global_step == resumed_trainer.global_step == 4

    # Matching weights also checks that resumed optimization agrees.
    for name, value in full_model.state_dict().items():
        torch.testing.assert_close(
            resumed_model.state_dict()[name],
            value,
            rtol=0,
            atol=0,
        )

    # Check the LR prepared for the next update, too.
    full_optimizer = full_trainer.optimizers[0]
    resumed_optimizer = resumed_trainer.optimizers[0]
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        full_optimizer.param_groups[0]["lr"]
    )

    if interval is not None:
        full_scheduler = full_trainer.lr_scheduler_configs[0].scheduler
        resumed_scheduler = resumed_trainer.lr_scheduler_configs[0].scheduler
        assert resumed_scheduler.state_dict() == full_scheduler.state_dict()


class PlateauTestModel(LightningModule):
    """Use a controlled validation metric to test scheduler timing."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.scheduler_factory = build_lr_scheduler_factory(
            TrainingLRSchedulerConfig(
                name="reduce_on_plateau",
                params={
                    "mode": "min",
                    "patience": 0,
                    "factor": 0.5,
                },
                monitor="validation_signal",
            )
        )

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        # Epoch 0 establishes the best value; later epochs get worse.
        self.log(
            "validation_signal",
            float(self.current_epoch + 1),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.scheduler_factory(optimizer),
        }


def test_lightning_passes_validation_metric_to_plateau(tmp_path):
    model = PlateauTestModel()
    recorder = RecordLearningRates()
    trainer = Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        callbacks=[recorder],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=True,
    )
    loader = DataLoader(torch.ones(2, 1), batch_size=1)

    trainer.fit(
        model,
        train_dataloaders=loader,
        val_dataloaders=loader,
    )

    # Epoch 0: establish baseline.
    # Epoch 1: use initial LR, then reduce after validation.
    # Epoch 2: use reduced LR, then reduce again after validation.
    assert recorder.values == pytest.approx(
        [0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
    )
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(
        0.025
    )
