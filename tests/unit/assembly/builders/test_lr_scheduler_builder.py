import pytest
import torch

from benchrep.assembly.builders.optimizer_builder import (
    build_lr_scheduler_factory,
)
from benchrep.assembly.schemas import TrainingLRSchedulerConfig
import benchrep.assembly.builders.optimizer_builder as optimizer_builder
from benchrep.assembly.registries.core import Registry

def make_optimizer():
    parameter = torch.nn.Parameter(torch.ones(1))
    return torch.optim.SGD([parameter], lr=0.1)


@pytest.mark.parametrize("name", ["step", "StepLR"])
def test_step_scheduler_updates_the_supplied_optimizer(name):
    config = TrainingLRSchedulerConfig(
        name=name,
        params={"step_size": 2, "gamma": 0.5},
        interval="step",
    )
    factory = build_lr_scheduler_factory(config)
    optimizer = make_optimizer()
    settings = factory(optimizer)
    scheduler = settings["scheduler"]

    assert scheduler.optimizer is optimizer
    assert settings["interval"] == "step"
    assert settings["frequency"] == 1
    assert settings["reduce_on_plateau"] is False
    assert "monitor" not in settings

    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates == pytest.approx(
        [0.1, 0.1, 0.05, 0.05, 0.025]
    )


def test_invalid_constructor_kwargs_fail_before_construction():
    config = TrainingLRSchedulerConfig(
        name="step",
        params={"step_size": 2, "unknown_argument": True},
    )

    with pytest.raises(ValueError, match="invalid constructor arguments"):
        build_lr_scheduler_factory(config)


def test_plateau_requires_monitor():
    config = TrainingLRSchedulerConfig(name="reduce_on_plateau")
    factory = build_lr_scheduler_factory(config)

    with pytest.raises(ValueError, match="monitor is required"):
        factory(make_optimizer())


def test_plateau_uses_metric_and_preserves_monitor_settings():
    config = TrainingLRSchedulerConfig(
        name="reduce_on_plateau",
        params={"mode": "min", "patience": 0, "factor": 0.5},
        monitor="losses/val/total_weighted_loss",
    )
    optimizer = make_optimizer()
    settings = build_lr_scheduler_factory(config)(optimizer)
    scheduler = settings["scheduler"]

    assert scheduler.optimizer is optimizer
    assert settings["monitor"] == "losses/val/total_weighted_loss"
    assert settings["reduce_on_plateau"] is True

    scheduler.step(1.0)  # Establish the best metric.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    scheduler.step(1.1)  # No improvement; patience=0 triggers reduction.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_registered_custom_factory(monkeypatch):
    # Isolate registration so this test doesn't modify the global registry.
    registry = Registry("test LR scheduler")
    monkeypatch.setattr(optimizer_builder, "LR_SCHEDULERS", registry)

    def custom_decay(opt, *, decay):
        return torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda position: decay ** position,
        )

    registry.register("custom_decay", custom_decay)

    config = TrainingLRSchedulerConfig(
        name="custom_decay",
        params={"decay": 0.5},
        interval="step",
    )
    optimizer = make_optimizer()
    settings = build_lr_scheduler_factory(config)(optimizer)
    scheduler = settings["scheduler"]

    assert scheduler.optimizer is optimizer

    rates = []
    for _ in range(3):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert rates == pytest.approx([0.1, 0.05, 0.025])
