import math

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchrep.assembly.resolvers.training_config_resolver import GradientNormSpec
from benchrep.inspection.gradient_norms import GradientNormInspectionCallback


class GradientInspectionTrainingModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([3.0, 4.0]))
        self.gradient_norm_inspector = None
        self.measurements = []
        self.gradients_before_clipping = []
        self.parameters_after_updates = []

    def training_step(self, batch, batch_idx):
        x = batch[0].squeeze()

        # Raw gradient is [x, 2*x].
        raw_loss = x * (
            self.coefficients[0] + 2 * self.coefficients[1]
        )

        # Exercise a changing effective weight, including zero.
        effective_weight = [0.0, 0.5, 1.0, 2.0][batch_idx]

        if self.gradient_norm_inspector is not None:
            self.gradient_norm_inspector.inspect_loss(
                self,
                loss_role="reconstruction",
                loss_id="analytic",
                raw_loss=raw_loss,
                effective_weight=effective_weight,
                batch_index=batch_idx,
            )

        return raw_loss * effective_weight

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def on_before_optimizer_step(self, optimizer):
        self.gradients_before_clipping.append(
            self.coefficients.grad.detach().clone()
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.parameters_after_updates.append(
            self.coefficients.detach().clone()
        )

    def log(self, name, value, *args, **kwargs):
        # Capture the actual values submitted by the inspector, while
        # retaining Lightning's normal logging behavior.
        if name.startswith("grad_norms_wrt_"):
            self.measurements.append(
                (name, self.global_step, float(value.detach()))
            )
        return super().log(name, value, *args, **kwargs)


def run_training(*, inspection_enabled, accumulate_grad_batches, tmp_path):
    model = GradientInspectionTrainingModel()

    callback = GradientNormInspectionCallback(
        model_family="autoencoder",
        gradient_norm_specs=(
            GradientNormSpec(
                kind="per_loss",
                target="whole_model",
                norm_type="l2",
                every_n=1,
                loss_keys=(("reconstruction", "analytic"),),
            ),
            GradientNormSpec(
                kind="optimizer_gradients",
                target="whole_model",
                norm_type="l2",
                every_n=1,
            ),
        ),
    )

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=0.1,
        gradient_clip_algorithm="norm",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
        callbacks=[callback] if inspection_enabled else [],
    )

    loader = DataLoader(
        TensorDataset(torch.tensor([1.0, 2.0, 3.0, 4.0])),
        batch_size=1,
        shuffle=False,
    )
    trainer.fit(model, train_dataloaders=loader)
    return model, callback


@pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
def test_inspection_preserves_training_and_measures_before_clipping(
    accumulate_grad_batches, tmp_path,
):
    baseline, _ = run_training(
        inspection_enabled=False,
        accumulate_grad_batches=accumulate_grad_batches,
        tmp_path=tmp_path / "baseline",
    )
    inspected, callback = run_training(
        inspection_enabled=True,
        accumulate_grad_batches=accumulate_grad_batches,
        tmp_path=tmp_path / "inspected",
    )

    # Inspection must leave both the accumulated training gradients and
    # every intermediate parameter update exactly unchanged.
    assert len(inspected.gradients_before_clipping) == (
        4 // accumulate_grad_batches
    )
    assert len(baseline.gradients_before_clipping) == len(
        inspected.gradients_before_clipping
    )
    for actual, expected in zip(
        inspected.gradients_before_clipping,
        baseline.gradients_before_clipping,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert len(inspected.parameters_after_updates) == 4
    for actual, expected in zip(
        inspected.parameters_after_updates,
        baseline.parameters_after_updates,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    prefix = "grad_norms_wrt_whole_model"
    loss_prefix = f"{prefix}/per_loss/reconstruction/analytic/l2"

    def measurements_for(name):
        return [
            value
            for metric_name, _, value in inspected.measurements
            if metric_name == name
        ]

    # Raw per-loss measurements must not be divided by the accumulation
    # factor. Weighted measurements must use the current effective weight.
    assert measurements_for(f"{loss_prefix}/raw") == pytest.approx(
        [x * math.sqrt(5.0) for x in [1.0, 2.0, 3.0, 4.0]]
    )
    assert measurements_for(f"{loss_prefix}/weighted") == pytest.approx(
        [x * math.sqrt(5.0) for x in [0.0, 1.0, 3.0, 8.0]]
    )

    # Lightning averages the weighted batch gradients over each complete
    # accumulation window: [0, 1, 3, 8] or [0.5, 5.5].
    expected_coefficients = (
        [0.0, 1.0, 3.0, 8.0]
        if accumulate_grad_batches == 1
        else [0.5, 5.5]
    )
    optimizer_measurements = [
        (step, value)
        for name, step, value in inspected.measurements
        if name == f"{prefix}/optimizer/l2"
    ]

    assert [step for step, _ in optimizer_measurements] == list(
        range(len(expected_coefficients))
    )
    assert [value for _, value in optimizer_measurements] == pytest.approx(
        [coefficient * math.sqrt(5.0) for coefficient in expected_coefficients]
    )

    for gradient, coefficient in zip(
        inspected.gradients_before_clipping, expected_coefficients,
    ):
        torch.testing.assert_close(
            gradient, torch.tensor([coefficient, 2 * coefficient]),
        )

    # Confirm clipping actually affected the updates. Otherwise a test
    # claiming to establish "before clipping" could pass without clipping.
    previous = torch.tensor([3.0, 4.0])
    for current in inspected.parameters_after_updates:
        update_norm = torch.linalg.vector_norm(previous - current).item()
        assert update_norm <= 0.010001  # learning rate * clipping threshold
        previous = current

    assert not torch.equal(inspected.coefficients.detach(), torch.tensor([3.0, 4.0]))
    assert inspected.gradient_norm_inspector is None
    assert callback.issues == ()


import warnings

from benchrep.inspection.gradient_norms import GradientNormInspector


def test_sampling_skips_autograd_and_disconnected_losses_can_recover():
    model = GradientInspectionTrainingModel()

    # Supply the trainer context required by the inspector's logging call.
    # Capture submitted measurements directly for this focused test.
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    model.trainer = trainer

    measurements = []
    model.log_dict = lambda values, **kwargs: measurements.append(
        {name: float(value.detach()) for name, value in values.items()}
    )

    inspector = GradientNormInspector(
        model,
        model_family="autoencoder",
        gradient_norm_specs=(
            GradientNormSpec(
                kind="per_loss",
                target="whole_model",
                norm_type="l2",
                every_n=2,
                loss_keys=(("reconstruction", "analytic"),),
            ),
        ),
    )

    def inspect(loss, batch_index, loss_id="analytic"):
        inspector.inspect_loss(
            model,
            loss_role="reconstruction",
            loss_id=loss_id,
            raw_loss=loss,
            effective_weight=0.5,
            batch_index=batch_index,
        )

    # Its graph has already been consumed. Attempting another backward
    # through it would fail, so successful skipping proves that an
    # unsampled batch does not invoke autograd.
    consumed_loss = model.coefficients.square().sum()
    consumed_loss.backward()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")

        inspect(consumed_loss, batch_index=1)
        inspect(consumed_loss, batch_index=0, loss_id="not_selected")

    assert measurements == []
    assert recorded == []
    assert inspector.issues == ()

    # This loss has a graph, but no connection to the selected parameters.
    unrelated_parameter = nn.Parameter(torch.tensor([7.0]))
    disconnected_loss = unrelated_parameter.square().sum()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")

        inspect(disconnected_loss, batch_index=2)
        inspect(disconnected_loss, batch_index=4)

    assert measurements == []
    assert len(recorded) == 1
    assert recorded[0].category is RuntimeWarning
    assert "disconnected" in str(recorded[0].message).lower()
    assert inspector.issues == (str(recorded[0].message),)

    # A later batch connects the same loss identity to the target.
    # Previous skips must not disable subsequent measurements.
    connected_loss = (
        3 * model.coefficients[0] + 4 * model.coefficients[1]
    )
    inspect(connected_loss, batch_index=6)

    prefix = (
        "grad_norms_wrt_whole_model/per_loss/"
        "reconstruction/analytic/l2"
    )
    assert measurements == [
        {
            f"{prefix}/raw": 5.0,
            f"{prefix}/weighted": 2.5,
        }
    ]
    assert len(inspector.issues) == 1


def test_optimizer_inspection_respects_target_membership_and_sampling():
    model = L.LightningModule()
    model.encoder = nn.Linear(2, 1, bias=True)
    model.decoder = nn.Linear(1, 1, bias=False)

    model.trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    measurements = []
    model.log = lambda name, value, **kwargs: measurements.append(
        (name, float(value.detach()))
    )

    # Encoder bias belongs to the target but not the active optimizer.
    # Decoder weight belongs to the optimizer but not the target.
    optimizer = torch.optim.SGD(
        [
            {"params": [model.encoder.weight]},
            {"params": [model.decoder.weight]},
        ],
        lr=0.1,
    )

    model.encoder.weight.grad = torch.tensor([[3.0, 4.0]])
    model.encoder.bias.grad = torch.tensor([100.0])
    model.decoder.weight.grad = torch.tensor([[200.0]])

    inspector = GradientNormInspector(
        model,
        model_family="autoencoder",
        gradient_norm_specs=(
            GradientNormSpec(
                kind="optimizer_gradients",
                target="encoder",
                norm_type="l2",
                every_n=2,
            ),
        ),
    )

    def inspect(completed_steps):
        inspector.inspect_optimizer_gradients(
            model,
            optimizer=optimizer,
            completed_optimizer_steps=completed_steps,
        )

    inspect(0)
    assert measurements == [("grad_norms_wrt_encoder/optimizer/l2", 5.0)]

    # New gradient values must be read at each sampled update.
    model.encoder.weight.grad = torch.tensor([[5.0, 12.0]])
    inspect(1)
    assert len(measurements) == 1

    inspect(2)
    assert measurements[-1] == (
        "grad_norms_wrt_encoder/optimizer/l2", 13.0,
    )
    assert len(measurements) == 2

    # Freezing after inspector construction must take effect immediately,
    # even if a stale .grad buffer remains.
    model.encoder.weight.requires_grad_(False)
    with pytest.warns(RuntimeWarning, match="no trainable parameters"):
        inspect(4)

    assert len(measurements) == 2
    torch.testing.assert_close(
        model.encoder.weight.grad, torch.tensor([[5.0, 12.0]]),
    )

    # Unfreezing must restore measurement without rebuilding the inspector.
    model.encoder.weight.requires_grad_(True)
    inspect(6)
    assert measurements[-1][1] == pytest.approx(13.0)
    assert len(measurements) == 3

    # Neither excluded gradient should have been altered.
    torch.testing.assert_close(
        model.encoder.bias.grad, torch.tensor([100.0]),
    )
    torch.testing.assert_close(
        model.decoder.weight.grad, torch.tensor([[200.0]]),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to test float16 gradient scaling.",
)
def test_optimizer_norms_are_unscaled_under_cuda_amp(tmp_path, monkeypatch):
    model = GradientInspectionTrainingModel()
    scaled_backward_norms = []
    backward_scales = []

    def record_after_backward():
        scaler = model.trainer.precision_plugin.scaler
        assert scaler is not None
        backward_scales.append(scaler.get_scale())
        scaled_backward_norms.append(
            model.coefficients.grad.detach().norm().item()
        )

    monkeypatch.setattr(model, "on_after_backward", record_after_backward)

    callback = GradientNormInspectionCallback(
        model_family="autoencoder",
        gradient_norm_specs=(
            GradientNormSpec(
                kind="optimizer_gradients",
                target="whole_model",
                norm_type="l2",
                every_n=1,
            ),
        ),
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=1,
        accumulate_grad_batches=2,
        gradient_clip_val=0.1,
        gradient_clip_algorithm="norm",
        logger=False,
        callbacks=[callback],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    loader = DataLoader(
        TensorDataset(torch.tensor([1.0, 2.0, 3.0, 4.0])),
        batch_size=1,
        shuffle=False,
    )

    trainer.fit(model, train_dataloaders=loader)

    # With accumulation=2, the running gradient coefficients are:
    # window 1: 0, 0.5; window 2: 1.5, 5.5.
    assert len(backward_scales) == len(scaled_backward_norms) == 4
    assert all(scale > 1 for scale in backward_scales)
    assert scaled_backward_norms == pytest.approx([
        coefficient * math.sqrt(5.0) * scale
        for coefficient, scale in zip(
            [0.0, 0.5, 1.5, 5.5], backward_scales,
        )
    ])

    optimizer_measurements = [
        (step, value)
        for name, step, value in model.measurements
        if name == "grad_norms_wrt_whole_model/optimizer/l2"
    ]

    assert [step for step, _ in optimizer_measurements] == [0, 1]
    assert [value for _, value in optimizer_measurements] == pytest.approx([
        0.5 * math.sqrt(5.0),
        5.5 * math.sqrt(5.0),
    ])

    # Both recorded norms exceed the clipping threshold. Verify that
    # clipping subsequently limits the actual parameter updates.
    positions = [
        position.cpu() for position in model.parameters_after_updates
    ]
    assert len(positions) == 4

    torch.testing.assert_close(positions[0], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(positions[2], positions[1])

    for before, after in [
        (positions[0], positions[1]),
        (positions[2], positions[3]),
    ]:
        assert (after - before).norm().item() == pytest.approx(
            0.01, rel=1e-4, abs=1e-6,
        )

    assert callback.issues == ()