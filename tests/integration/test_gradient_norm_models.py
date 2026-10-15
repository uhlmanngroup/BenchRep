import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from benchrep.architecture.decoders.mlp import MLPDecoder
from benchrep.architecture.encoders.mlp import MLPEncoder
from benchrep.assembly.builders.model_builder import (
    build_autoencoder,
    build_vae,
)
from benchrep.assembly.resolvers.loss_resolver import resolve_loss_configs
from benchrep.assembly.resolvers.training_config_resolver import GradientNormSpec
from benchrep.assembly.schemas import (
    TrainingLossTermConfig,
    TrainingOptimizerConfig,
)
from benchrep.inspection.gradient_norms import GradientNormInspectionCallback


class RecordEncoderTrainingGradients(L.Callback):
    def __init__(self):
        self.norms = []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Independent reference: concatenate the actual training gradients.
        gradients = [
            parameter.grad.detach().reshape(-1)
            for parameter in pl_module.encoder.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        self.norms.append(torch.cat(gradients).norm().item())


def make_model(model_family):
    loss_config = TrainingLossTermConfig.model_validate({
        "name": "mse",
        "weight": 2.0,
        "weight_schedule": {
            "name": "linear",
            "interval": "step",
            "params": {
                "start": 0,
                "end": 2,
                "start_factor": 0.0,
                "end_factor": 1.0,
            },
        },
    })

    loss_configs = {"reconstruction": [loss_config]}
    if model_family == "vae":
        loss_configs["regularization"] = [
            TrainingLossTermConfig(
                name="gaussian_kld",
                weight=0.0,
                params={"reduction": "mean"},
            ),
        ]

    arguments = {
        "encoder": MLPEncoder(
            input_shape=(2,),
            output_dim=2,
            hidden_dims=(),
        ),
        "decoder": MLPDecoder(
            input_dim=2,
            output_shape=(2,),
            hidden_dims=(),
        ),
        "optimizer": TrainingOptimizerConfig(
            name="sgd", params={"lr": 0.01},
        ),
        "loss_specs": resolve_loss_configs(loss_configs),
    }

    if model_family == "vae":
        return build_vae(**arguments, latent_dim=2)
    return build_autoencoder(**arguments)


@pytest.mark.parametrize("model_family", ["autoencoder", "vae"])
def test_real_model_inspection_matches_training_gradients(
    model_family, tmp_path, monkeypatch,
):
    def train(inspection_enabled):
        # Reset initialization, data-loader RNG, and VAE sampling.
        L.seed_everything(137, workers=True)
        model = make_model(model_family)
        measurements = {}

        original_log = model.log

        def record_log(name, value, *args, **kwargs):
            if name.startswith("grad_norms_wrt_"):
                measurements.setdefault(name, []).append(
                    float(value.detach())
                )
            return original_log(name, value, *args, **kwargs)

        monkeypatch.setattr(model, "log", record_log)

        recorder = RecordEncoderTrainingGradients()
        callbacks = [recorder]

        if inspection_enabled:
            callbacks.append(GradientNormInspectionCallback(
                model_family=model_family,
                gradient_norm_specs=(
                    GradientNormSpec(
                        kind="per_loss",
                        target="encoder",
                        norm_type="l2",
                        every_n=1,
                        loss_keys=(("reconstruction", "mse"),),
                    ),
                ),
            ))

        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            logger=False,
            callbacks=callbacks,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            deterministic=True,
            default_root_dir=tmp_path / str(inspection_enabled),
        )

        loader = DataLoader(
            [
                {"x": torch.tensor([1.0, 2.0])},
                {"x": torch.tensor([2.0, -1.0])},
                {"x": torch.tensor([-1.0, 3.0])},
            ],
            batch_size=1,
            shuffle=False,
        )
        trainer.fit(model, train_dataloaders=loader)
        return model, measurements, recorder.norms

    baseline, baseline_measurements, baseline_norms = train(False)
    inspected, measurements, training_norms = train(True)

    prefix = "grad_norms_wrt_encoder/per_loss/reconstruction/mse/l2"
    raw_norms = measurements[f"{prefix}/raw"]
    weighted_norms = measurements[f"{prefix}/weighted"]

    assert baseline_measurements == {}
    assert len(raw_norms) == len(weighted_norms) == len(training_norms) == 3
    assert all(value > 0 for value in raw_norms)

    # Base weight 2 times scheduled factors [0, 0.5, 1].
    assert weighted_norms == pytest.approx([
        raw * weight
        for raw, weight in zip(raw_norms, [0.0, 1.0, 2.0])
    ])

    # With one loss and no accumulation, its weighted gradient norm must
    # match the gradient obtained by normal backward on the objective.
    assert weighted_norms == pytest.approx(
        training_norms, rel=1e-5, abs=1e-7,
    )
    assert training_norms == pytest.approx(baseline_norms, rel=0, abs=0)

    baseline_state = baseline.state_dict()
    inspected_state = inspected.state_dict()
    assert baseline_state.keys() == inspected_state.keys()
    for name in baseline_state:
        torch.testing.assert_close(
            inspected_state[name], baseline_state[name], rtol=0, atol=0,
        )

    assert inspected.gradient_norm_inspector is None