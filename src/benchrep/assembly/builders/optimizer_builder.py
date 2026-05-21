from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

from benchrep.assembly.registry import OPTIMIZERS


def build_optimizer_factory(
    optimizer_config: dict[str, Any],
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
        Dictionary with optimizer name and keyword arguments, for example::

            {
                "name": "adam",
                "params": {
                    "lr": 0.001,
                    "weight_decay": 0.0,
                },
            }

    Returns
    -------
    Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
        A callable that takes model parameters and returns an instantiated optimizer.
    """
    if not isinstance(optimizer_config, dict):
        raise TypeError(
            "optimizer_config must be a dictionary, "
            f"got {type(optimizer_config).__name__}."
        )

    if "name" not in optimizer_config:
        raise KeyError("optimizer_config must contain key 'name'.")

    optimizer_name = optimizer_config["name"]
    optimizer_params = optimizer_config.get("params", {})

    if not isinstance(optimizer_params, dict):
        raise TypeError(
            "optimizer_config['params'] must be a dictionary if provided, "
            f"got {type(optimizer_params).__name__}."
        )

    optimizer_class = OPTIMIZERS.get(optimizer_name)

    # Capture the optimizer class and config params now; model parameters are
    # supplied later by Lightning inside configure_optimizers().
    def optimizer_factory(
        parameters: Iterable[nn.Parameter],
    ) -> torch.optim.Optimizer:
        return optimizer_class(parameters, **optimizer_params)

    return optimizer_factory