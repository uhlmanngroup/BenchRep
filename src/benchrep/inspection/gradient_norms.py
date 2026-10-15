from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import warnings

import lightning as L
import torch
from torch import nn


if TYPE_CHECKING:
    from benchrep.assembly.resolvers.training_config_resolver import (
        GradientNormSpec,
    )


@dataclass(frozen=True)
class GradientNormResult:
    """A measured norm, or the reason this measurement was skipped."""

    norm: torch.Tensor | None
    skip_reason: str | None = None


class GradientNormInspector:
    """Compute and log gradient norms for configured losses and model components.

    Each resolved request selects a target component (or the whole model),
    a norm type, a measurement kind, and a sampling frequency.

    Per-loss measurements differentiate the current batch's raw loss with
    respect to the target's trainable parameters using torch.autograd.grad.
    This preserves the graph for training backward and does not populate or
    modify parameter .grad buffers. Log both the raw norm and its product
    with the absolute effective loss weight; the latter assumes the weight
    is independent of the selected parameters.

    Optimizer-gradient measurements read existing parameter .grad buffers
    after accumulation and mixed-precision unscaling, before clipping.
    They do not perform an additional backward calculation.

    Norms treat all selected gradient elements as one vector, rather than
    averaging norms across parameter tensors. Frozen parameters are excluded;
    unused parameters contribute no gradient. A connected zero gradient is
    reported as zero, while unavailable measurements are skipped with a
    deduplicated warning exposed through `issues` for training provenance.

    Parameters
    ----------
    model:
        Constructed model whose target parameters are selected during
        initialization. Measurements use these parameter objects throughout
        training; requires_grad is checked again at measurement time.
    model_family:
        Determines component lookup: canonical encoder/decoder/variational_head
        attributes, or composite components_by_id entries.
    gradient_norm_specs:
        Resolved inspection requests containing the measurement kind, target,
        norm type, sampling frequency, and selected loss identities.
    """

    def __init__(
        self,
        model: L.LightningModule,
        *,
        model_family: Literal["autoencoder", "vae", "composite"],
        gradient_norm_specs: tuple[GradientNormSpec, ...],
    ) -> None:
        self.gradient_norm_specs = gradient_norm_specs
        self.parameters_by_target = {
            target: select_gradient_norm_parameters(
                model,
                target,
                model_family=model_family,
            )
            for target in dict.fromkeys(spec.target for spec in gradient_norm_specs)
        }
        self._issues: dict[str, None] = {}

    @property
    def issues(self) -> tuple[str, ...]:
        """Return unique warnings for inclusion in the training report."""
        return tuple(self._issues)

    def _warn_once(self, message: str) -> None:
        if message in self._issues:
            return

        self._issues[message] = None
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def inspect_loss(
        self,
        model: L.LightningModule,
        *,
        loss_role: str,
        loss_id: str,
        raw_loss: torch.Tensor,
        effective_weight: float,
        batch_index: int,
    ) -> None:
        """Measure and log selected gradients for one raw training-batch loss.

        Evaluate only per-loss requests that include this loss and are due on
        the current batch. Differentiate the raw loss once per matching request,
        then derive its weighted norm by multiplying by abs(effective_weight).

        Parameters
        ----------
        model:
            Lightning model used to log measurements.
        loss_role:
            Loss category, such as reconstruction or regularization.
        loss_id:
            Resolved identifier of the loss within its role.
        raw_loss:
            Scalar loss tensor before applying its configured weight. Its
            autograd graph must still be available.
        effective_weight:
            Scalar weight applied to this loss for the current batch, including
            loss-weight scheduling. Must be independent of target parameters.
        batch_index:
            Zero-based training-batch index within the epoch. A request runs
            when batch_index % every_n == 0.

        Notes
        -----
        Call before normal training backward frees the loss graph.
        Unavailable measurements produce a warning once, but are retried on
        later sampled batches. Unexpected calculation errors propagate.

        Sampling controls computation; tracker emission additionally follows
        Lightning's logging cadence. No epoch averaging is performed.
        """
        loss_key = (loss_role, loss_id)

        for spec in self.gradient_norm_specs:
            # Optimizer gradients under separate method.
            if spec.kind != "per_loss":
                continue
            # Skip losses without configured inspection request.
            if loss_key not in spec.loss_keys:
                continue
            # Skip batches where inspection isn't due.
            if batch_index % spec.every_n != 0:
                continue

            result = compute_loss_gradient_norm(
                raw_loss,
                self.parameters_by_target[spec.target],
                spec.norm_type,
            )

            if result.skip_reason is not None:
                self._warn_once(
                    "Gradient inspection skipped "
                    f"loss={loss_role}.{loss_id}, "
                    f"target={spec.target}, norm={spec.norm_type}: "
                    f"{result.skip_reason}"
                )
                continue

            assert result.norm is not None

            prefix = (
                f"grad_norms_wrt_{spec.target}/per_loss/"
                f"{loss_role}/{loss_id}/{spec.norm_type}"
            )

            model.log_dict(
                {
                    f"{prefix}/raw": result.norm,
                    f"{prefix}/weighted": (
                        result.norm * abs(effective_weight)
                    ),
                },
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=bool(model.trainer.loggers),
                sync_dist=False,
            )

    def inspect_optimizer_gradients(
        self,
        model: L.LightningModule,
        *,
        optimizer: torch.optim.Optimizer,
        completed_optimizer_steps: int,
    ) -> None:
        """Measure stored gradients immediately before an optimizer update.

        Call from Lightning's on_before_optimizer_step hook, after gradient
        accumulation and mixed-precision unscaling, but before clipping.

        Only inspect parameters belonging to both the selected target and
        the optimizer performing this update. Read their existing .grad
        buffers without calculating new gradients or modifying them.

        Parameters
        ----------
        model:
            Lightning model used to log measurements.
        optimizer:
            Optimizer about to update its parameters.
        completed_optimizer_steps:
            Number of optimizer updates already completed. A request runs
            when this count % every_n == 0, including before the first update.

        Notes
        -----
        Log one norm per matching request, without epoch averaging.
        Unavailable measurements produce deduplicated warnings and are
        retried at later sampled updates. Unexpected errors propagate.
        """
        optimizer_parameter_ids = {
            id(parameter)
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        }

        for gradient_norm_spec in self.gradient_norm_specs:
            if gradient_norm_spec.kind != "optimizer_gradients":
                continue

            if (
                completed_optimizer_steps % gradient_norm_spec.every_n
                != 0
            ):
                continue

            target_parameters = self.parameters_by_target[
                gradient_norm_spec.target
            ]
            selected_parameters = [
                parameter
                for parameter in target_parameters
                if id(parameter) in optimizer_parameter_ids
            ]

            if not selected_parameters:
                self._warn_once(
                    "Optimizer-gradient inspection skipped "
                    f"target={gradient_norm_spec.target}, "
                    f"norm={gradient_norm_spec.norm_type}: "
                    "Target has no parameters in the active optimizer."
                )
                continue

            result = compute_optimizer_gradient_norm(
                selected_parameters,
                gradient_norm_spec.norm_type,
            )

            if result.skip_reason is not None:
                self._warn_once(
                    "Optimizer-gradient inspection skipped "
                    f"target={gradient_norm_spec.target}, "
                    f"norm={gradient_norm_spec.norm_type}: "
                    f"{result.skip_reason}"
                )
                continue

            assert result.norm is not None

            metric_name = (
                f"grad_norms_wrt_{gradient_norm_spec.target}/"
                f"optimizer/{gradient_norm_spec.norm_type}"
            )

            model.log(
                metric_name,
                result.norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=bool(model.trainer.loggers),
                sync_dist=False,
            )


class GradientNormInspectionCallback(L.Callback):
    """Initialize gradient inspection and run optimizer-gradient requests.

    One inspector is shared with the model's per-loss inspection calls.
    Initialization occurs at fit start, after model setup and device placement.
    Inspection warnings remain available through the callback's issues property.

    This implementation supports single-process, automatic optimization.
    """

    def __init__(
        self,
        *,
        model_family: Literal["autoencoder", "vae", "composite"],
        gradient_norm_specs: tuple[GradientNormSpec, ...],
    ) -> None:
        super().__init__()
        self.model_family = model_family
        self.gradient_norm_specs = gradient_norm_specs
        self.inspector: GradientNormInspector | None = None

    @property
    def issues(self) -> tuple[str, ...]:
        if self.inspector is None:
            return ()
        return self.inspector.issues

    def on_fit_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if not self.gradient_norm_specs:
            return

        if trainer.world_size != 1:
            raise ValueError(
                "Gradient-norm inspection currently supports "
                "single-process training only."
            )

        if not pl_module.automatic_optimization:
            raise ValueError(
                "Gradient-norm inspection currently requires "
                "Lightning automatic optimization."
            )

        if len(trainer.optimizers) != 1:
            raise ValueError(
                "Gradient-norm inspection currently requires one optimizer."
            )

        self.inspector = GradientNormInspector(
            pl_module,
            model_family=self.model_family,
            gradient_norm_specs=self.gradient_norm_specs,
        )
        pl_module.gradient_norm_inspector = self.inspector

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self.inspector is None:
            return

        self.inspector.inspect_optimizer_gradients(
            pl_module,
            optimizer=optimizer,
            completed_optimizer_steps=trainer.global_step,
        )

    def teardown(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        stage: str,
    ) -> None:
        if (
            self.inspector is not None
            and getattr(pl_module, "gradient_norm_inspector", None)
            is self.inspector
        ):
            pl_module.gradient_norm_inspector = None


def select_gradient_norm_parameters(
    model: nn.Module,
    target: str,
    *,
    model_family: Literal["autoencoder", "vae", "composite"],
) -> tuple[nn.Parameter, ...]:
    """Select parameters using the model family's component structure.

    Include frozen parameters; measurement helpers check requires_grad
    each time to respect subsequent freezing or unfreezing.
    """
    if target == "whole_model":
        target_module = model

    elif model_family == "composite":
        target_module = model.components_by_id[target]

    elif model_family == "autoencoder":
        components = {
            "encoder": model.encoder,
            "decoder": model.decoder,
        }
        target_module = components[target]

    elif model_family == "vae":
        components = {
            "encoder": model.encoder,
            "decoder": model.decoder,
            "variational_head": model.variational_head,
        }
        target_module = components[target]

    else:
        raise ValueError(f"Unsupported model family: {model_family!r}.")

    if not isinstance(target_module, nn.Module):
        raise TypeError(
            f"Gradient inspection target {target!r} "
            "does not resolve to a torch.nn.Module."
        )

    return tuple(target_module.parameters())


def compute_optimizer_gradient_norm(
    parameters: Iterable[nn.Parameter],
    norm_type: Literal["l1", "l2", "inf"],
) -> GradientNormResult:
    """Measure existing parameter gradients without modifying them.

    Call before the optimizer update, after gradient accumulation and
    mixed-precision unscaling, but before clipping.
    """
    trainable_parameters = [
        parameter
        for parameter in parameters
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        return GradientNormResult(
            norm=None,
            skip_reason="Target has no trainable parameters.",
        )

    norm = compute_gradient_norm(
        (parameter.grad for parameter in trainable_parameters),
        norm_type,
    )

    if norm is None:
        return GradientNormResult(
            norm=None,
            skip_reason="Target has no stored gradients for this update.",
        )

    return GradientNormResult(norm=norm)


def compute_loss_gradient_norm(
    loss: torch.Tensor,
    parameters: Iterable[nn.Parameter],
    norm_type: Literal["l1", "l2", "inf"],
) -> GradientNormResult:
    """Measure a loss's gradients without modifying parameter .grad buffers.

    Exclude frozen parameters and preserve the graph for normal training
    backward. Expected unavailable measurements return a skip reason;
    unexpected autograd errors propagate.
    """
    trainable_parameters = [
        parameter
        for parameter in parameters
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        return GradientNormResult(
            norm=None,
            skip_reason="Target has no trainable parameters.",
        )

    if not loss.requires_grad:
        return GradientNormResult(
            norm=None,
            skip_reason="Loss does not have a gradient graph.",
        )

    gradients = torch.autograd.grad(
        outputs=loss,
        inputs=trainable_parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    norm = compute_gradient_norm(gradients, norm_type)

    if norm is None:
        return GradientNormResult(
            norm=None,
            skip_reason="Loss is disconnected from the target parameters.",
        )

    return GradientNormResult(norm=norm)


def compute_gradient_norm(
    gradients: Iterable[torch.Tensor | None],
    norm_type: Literal["l1", "l2", "inf"],
) -> torch.Tensor | None:
    """Compute one norm as if all supplied gradients formed a single vector.

    Accept an iterable of gradient tensors; wrap a single tensor in a list.
    Combine per-tensor L1/L2 norms using the same norm, or take the maximum
    of per-tensor absolute maxima for infinity norm. This avoids concatenating
    all gradient elements.

    Ignore None entries. Return None if no tensors are present; zero-valued
    or empty tensors contribute zero.

    Support dense and sparse COO layouts, summing duplicate COO indices
    before measuring stored values. All tensors must be on the same device.

    Detach gradients and promote float16/bfloat16 to float32 without
    modifying the supplied tensors. Preserve NaN/Inf diagnostic values.
    """
    if norm_type not in {"l1", "l2", "inf"}:
        raise ValueError(f"Unsupported gradient norm type: {norm_type!r}.")

    values = []
    for gradient in gradients:
        if gradient is None:
            continue

        gradient = gradient.detach()

        # Coalescing to sum duplicate sparse indices before measuring them.
        if gradient.layout == torch.sparse_coo:
            gradient = gradient.coalesce().values()
        elif gradient.layout != torch.strided:
            raise NotImplementedError(
                f"Unsupported gradient layout: {gradient.layout}."
            )

        if gradient.dtype in {torch.float16, torch.bfloat16}:
            # Ensure high precision without altering stored training gradients.
            gradient = gradient.float()

        values.append(gradient)

    if not values:
        # No gradients > return None, to distinguish from zero gradients.
        return None

    # For infinity norm, find max absolute gradient if tensor is not empty,
    # else substitute zero.
    if norm_type == "inf":
        maxima = [
            gradient.abs().max()
            if gradient.numel()
            else gradient.real.new_zeros(())
            for gradient in values
        ]
        return torch.stack(maxima).max()

    order = 1 if norm_type == "l1" else 2
    parameter_norms = [
        torch.linalg.vector_norm(gradient.reshape(-1), ord=order)
        for gradient in values
    ]
    return torch.linalg.vector_norm(
        torch.stack(parameter_norms),
        ord=order,
    )