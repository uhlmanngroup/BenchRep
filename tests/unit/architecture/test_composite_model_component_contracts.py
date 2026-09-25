from __future__ import annotations

import pytest
import torch
from torch import nn

from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
    ComponentPort,
    ComponentTensorResult,
)


class MultiInputModule(nn.Module):
    def forward(
        self,
        *,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((left, right), dim=1)


def test_accepts_plain_nn_module_with_matching_forward_contract() -> None:
    component = ArchitectureComponent(
        component=MultiInputModule,
        runtime_inputs=(
            ComponentPort(
                name="left",
                supported_structures=("vector",),
            ),
            ComponentPort(
                name="right",
                supported_structures=("vector",),
            ),
        ),
        runtime_result=ComponentTensorResult(
            supported_structures=("vector",),
        ),
    )

    assert component.component is MultiInputModule


def test_rejects_runtime_inputs_incompatible_with_forward_signature() -> None:
    with pytest.raises(
        TypeError,
        match="ArchitectureComponent runtime inputs",
    ):
        ArchitectureComponent(
            component=MultiInputModule,
            runtime_inputs=(
                ComponentPort(
                    name="left",
                    supported_structures=("vector",),
                ),
            ),
            runtime_result=ComponentTensorResult(
                supported_structures=("vector",),
            ),
        )


def test_rejects_runtime_input_not_accepted_by_forward_signature() -> None:
    with pytest.raises(
        TypeError,
        match="ArchitectureComponent runtime inputs",
    ):
        ArchitectureComponent(
            component=MultiInputModule,
            runtime_inputs=(
                ComponentPort(
                    name="left",
                    supported_structures=("vector",),
                ),
                ComponentPort(
                    name="right",
                    supported_structures=("vector",),
                ),
                ComponentPort(
                    name="extra",
                    supported_structures=("vector",),
                ),
            ),
            runtime_result=ComponentTensorResult(
                supported_structures=("vector",),
            ),
        )