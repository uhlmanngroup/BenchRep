from __future__ import annotations

import inspect
from typing import Any
from collections.abc import Callable, Iterable

import torch
from torch import nn

from benchrep.assembly.registries.utils import normalize_name
from benchrep.assembly.schemas import (
    TrainingOptimizerConfig,
    TrainingLRSchedulerConfig,
)
from benchrep.assembly.registries.core import OPTIMIZERS, LR_SCHEDULERS


def build_optimizer_factory(
    optimizer_config: TrainingOptimizerConfig,
) -> Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]:
    """Build a delayed optimizer factory from config.

    Optimizers cannot be instantiated from config alone because PyTorch optimizers
    need the model parameters at construction time. At the point where this builder
    is called, the config is available, but the model parameters may not be.

    This function therefore returns an ``optimizer_factory`` instead of an optimizer
    object directly. The returned factory captures the optimizer class and its config
    parameters now, then receives model parameters later from
    ``LightningModule.configure_optimizers()``.

    Expected config format
    ----------------------
    optimizer_config:
    Validated optimizer config object parsed from a config block such as::

        optimizer:
            name: adam
            params:
                lr: 0.001
                weight_decay: 0.0

    Returns
    -------
    Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
        A callable that takes model parameters and returns an instantiated optimizer.
    """
    optimizer_name = normalize_name(
        optimizer_config.name,
        field_name="config.optimizer.name",
    )

    optimizer_params = dict(optimizer_config.params)

    optimizer_class = OPTIMIZERS.get(optimizer_name)

    # Capture the optimizer class and config params now; model parameters are
    # supplied later by Lightning inside configure_optimizers().
    def optimizer_factory(
        parameters: Iterable[nn.Parameter],
    ) -> torch.optim.Optimizer:
        return optimizer_class(parameters, **optimizer_params)

    return optimizer_factory


def build_lr_scheduler_factory(
    scheduler_config: TrainingLRSchedulerConfig,
) -> Callable[[torch.optim.Optimizer], dict[str, Any]]:
    """Resolve a scheduler now and construct it once its optimizer exists.

    The returned factory produces Lightning's scheduler configuration,
    including the execution interval and optional monitored metric.
    """
    config_path = "config.optimizer.lr_scheduler"
    scheduler_name = normalize_name(
        scheduler_config.name,
        field_name=f"{config_path}.name",
    )
    scheduler_constructor = LR_SCHEDULERS.get(scheduler_name)
    scheduler_params = dict(scheduler_config.params)

    # Validate the call shape without constructing a scheduler or optimizer.
    try:
        inspect.signature(scheduler_constructor).bind(
            None, **scheduler_params
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{config_path}: invalid constructor arguments for "
            f"{scheduler_name!r}: {exc}"
        ) from exc

    interval = scheduler_config.interval
    frequency = scheduler_config.frequency
    monitor = scheduler_config.monitor

    def scheduler_factory(
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        scheduler = scheduler_constructor(optimizer, **scheduler_params)

        is_plateau = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )
        if not isinstance(
            scheduler, torch.optim.lr_scheduler.LRScheduler
        ) and not is_plateau:
            raise TypeError(
                f"{config_path}: {scheduler_name!r} must return a "
                "PyTorch LR scheduler."
            )

        if scheduler.optimizer is not optimizer:
            raise ValueError(
                f"{config_path}: the scheduler must use the optimizer "
                "supplied by BenchRep."
            )

        if is_plateau and (monitor is None or not monitor.strip()):
            raise ValueError(
                f"{config_path}.monitor is required for "
                "ReduceLROnPlateau."
            )

        scheduler_settings: dict[str, Any] = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": frequency,
            "reduce_on_plateau": is_plateau,
        }
        if monitor is not None:
            scheduler_settings["monitor"] = monitor

        return scheduler_settings

    return scheduler_factory
