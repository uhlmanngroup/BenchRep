from __future__ import annotations

from collections.abc import Mapping

from benchrep.architecture.losses.base import LossTerm


def validate_loss_weights(
    losses_by_role: Mapping[
        str,
        Mapping[str, LossTerm],
    ],
) -> None:
    for role, loss_terms in losses_by_role.items():
        role_label = role.replace("_", " ").title()

        for loss_name, loss_term in loss_terms.items():
            if loss_term.weight < 0:
                raise ValueError(
                    f"{role_label} loss {loss_name!r} has negative "
                    f"weight {loss_term.weight}."
                )