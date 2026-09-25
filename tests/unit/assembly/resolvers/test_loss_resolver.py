from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import nn

from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
)
from benchrep.architecture.losses.custom_objective import (
    BaseCustomObjectiveLoss,
)
from benchrep.assembly.registries.core import (
    LOSS_REGISTRIES_BY_ROLE,
    LossRegistry,
)
from benchrep.assembly.resolvers.loss_resolver import (
    resolve_composite_loss_configs,
    resolve_loss_configs,
)
from benchrep.assembly.schemas.training_config_schema import (
    TrainingLossTermConfig,
)


class CanonicalOnlyReconstructionLoss(nn.Module):
    def forward(
        self,
        *,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mean((reconstruction - target) ** 2)


class ResolverCustomObjectiveLoss(BaseCustomObjectiveLoss):
    def forward(
        self,
        *,
        batch: Mapping[str, Any],
        model_output: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        del batch
        return model_output["embedding"].sum() * 0.0


def test_repeated_loss_can_be_disambiguated_with_ids() -> None:
    loss_specs = resolve_loss_configs(
        {
            "reconstruction": [
                TrainingLossTermConfig(
                    name="mae",
                    id="morphology",
                ),
                TrainingLossTermConfig(
                    name="l1",
                    id="intensity",
                ),
            ],
        }
    )

    assert [
        (
            loss_spec.registry_entry_name,
            loss_spec.loss_id,
        )
        for loss_spec in loss_specs
    ] == [
        ("mae", "morphology_mae"),
        ("mae", "intensity_mae"),
    ]


def test_repeated_aliases_without_ids_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=r"resolve to loss ID 'mae'",
    ):
        resolve_loss_configs(
            {
                "reconstruction": [
                    TrainingLossTermConfig(name="mae"),
                    TrainingLossTermConfig(name="l1"),
                ],
            }
        )


def test_only_one_repeated_loss_needs_an_id() -> None:
    loss_specs = resolve_loss_configs(
        {
            "reconstruction": [
                TrainingLossTermConfig(name="mae"),
                TrainingLossTermConfig(
                    name="l1",
                    id="morphology",
                ),
            ],
        }
    )

    assert [
        loss_spec.loss_id
        for loss_spec in loss_specs
    ] == [
        "mae",
        "morphology_mae",
    ]


def test_generated_loss_metric_name_collision_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collision_registry = LossRegistry(
        "resolver test reconstruction loss",
        canonical_runtime_input_names=(
            "reconstruction",
            "target",
        ),
    )

    loss_component = LossComponent(
        CanonicalOnlyReconstructionLoss
    )

    collision_registry.register(
        "foo",
        loss_component,
    )
    collision_registry.register(
        "foo_weighted",
        loss_component,
    )

    monkeypatch.setitem(
        LOSS_REGISTRIES_BY_ROLE,
        "reconstruction",
        collision_registry,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"produce a metric-name collision at 'foo_weighted'"
        ),
    ):
        resolve_loss_configs(
            {
                "reconstruction": [
                    TrainingLossTermConfig(name="foo"),
                    TrainingLossTermConfig(name="foo_weighted"),
                ],
            }
        )


def test_resolves_composite_loss_alias_and_wiring() -> None:
    loss_specs = resolve_composite_loss_configs(
        {
            "reconstruction": [
                TrainingLossTermConfig(
                    name="mean_squared_error",
                    composite_wiring={
                        "reconstruction": "produces.reconstruction",
                        "target": "expects.image",
                    },
                ),
            ],
        },
        model_input_roles_by_name={
            "image": "sample_image",
        },
        model_output_roles_by_name={
            "reconstruction": "reconstruction_image",
        },
    )

    assert len(loss_specs) == 1

    loss_spec = loss_specs[0]

    assert loss_spec.registry_entry_name == "mse"
    assert loss_spec.loss_id == "mse"
    assert loss_spec.inputs_from_model_inputs == {
        "target": "image",
    }
    assert loss_spec.inputs_from_model_outputs == {
        "reconstruction": "reconstruction",
    }
    assert loss_spec.context_inputs == {}


def test_rejects_loss_input_with_incompatible_semantic_role() -> None:
    with pytest.raises(
        ValueError,
        match="supports only",
    ):
        resolve_composite_loss_configs(
            {
                "reconstruction": [
                    TrainingLossTermConfig(
                        name="mse",
                        composite_wiring={
                            "reconstruction": "produces.embedding",
                            "target": "expects.image",
                        },
                    ),
                ],
            },
            model_input_roles_by_name={
                "image": "sample_image",
            },
            model_output_roles_by_name={
                "embedding": "embedding_vector",
                "reconstruction": "reconstruction_image",
            },
        )


def test_rejects_ordinary_loss_without_composite_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_only_registry = LossRegistry(
        "resolver test reconstruction loss",
        canonical_runtime_input_names=(
            "reconstruction",
            "target",
        ),
    )
    canonical_only_registry.register(
        "canonical_only",
        LossComponent(CanonicalOnlyReconstructionLoss),
    )
    monkeypatch.setitem(
        LOSS_REGISTRIES_BY_ROLE,
        "reconstruction",
        canonical_only_registry,
    )

    with pytest.raises(
        ValueError,
        match="does not declare a Composite runtime contract",
    ):
        resolve_composite_loss_configs(
            {
                "reconstruction": [
                    TrainingLossTermConfig(
                        name="canonical_only",
                        composite_wiring={
                            "reconstruction": "produces.reconstruction",
                            "target": "expects.image",
                        },
                    ),
                ],
            },
            model_input_roles_by_name={
                "image": "sample_image",
            },
            model_output_roles_by_name={
                "reconstruction": "reconstruction_image",
            },
        )


def test_resolves_custom_objective_with_fixed_context_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_objective_registry = LossRegistry(
        "resolver test custom objective loss",
        canonical_runtime_input_names=(
            "batch",
            "model_output",
        ),
        required_component_base=BaseCustomObjectiveLoss,
        explicit_composite_runtime_inputs_supported=False,
    )
    custom_objective_registry.register(
        "resolver_custom_objective",
        LossComponent(ResolverCustomObjectiveLoss),
    )
    monkeypatch.setitem(
        LOSS_REGISTRIES_BY_ROLE,
        "custom_objective",
        custom_objective_registry,
    )

    loss_specs = resolve_composite_loss_configs(
        {
            "custom_objective": [
                TrainingLossTermConfig(
                    name="resolver_custom_objective",
                ),
            ],
        },
        model_input_roles_by_name={
            "image": "sample_image",
        },
        model_output_roles_by_name={
            "embedding": "embedding_vector",
        },
    )

    assert len(loss_specs) == 1
    assert loss_specs[0].context_inputs == {
        "batch": "batch",
        "model_output": "model_output",
    }
    assert loss_specs[0].inputs_from_model_inputs == {}
    assert loss_specs[0].inputs_from_model_outputs == {}


def test_rejects_composite_wiring_for_custom_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_objective_registry = LossRegistry(
        "resolver test custom objective loss",
        canonical_runtime_input_names=(
            "batch",
            "model_output",
        ),
        required_component_base=BaseCustomObjectiveLoss,
        explicit_composite_runtime_inputs_supported=False,
    )
    custom_objective_registry.register(
        "resolver_custom_objective",
        LossComponent(ResolverCustomObjectiveLoss),
    )
    monkeypatch.setitem(
        LOSS_REGISTRIES_BY_ROLE,
        "custom_objective",
        custom_objective_registry,
    )

    with pytest.raises(
        ValueError,
        match="must not be configured for a custom objective",
    ):
        resolve_composite_loss_configs(
            {
                "custom_objective": [
                    TrainingLossTermConfig(
                        name="resolver_custom_objective",
                        composite_wiring={
                            "batch": "expects.image",
                        },
                    ),
                ],
            },
            model_input_roles_by_name={
                "image": "sample_image",
            },
            model_output_roles_by_name={
                "embedding": "embedding_vector",
            },
        )


def test_rejects_missing_and_unexpected_loss_input_keys() -> None:
    with pytest.raises(
        ValueError,
        match=r"missing keys \['target'\]; unexpected keys \['source'\]",
    ):
        resolve_composite_loss_configs(
            {
                "reconstruction": [
                    TrainingLossTermConfig(
                        name="mse",
                        composite_wiring={
                            "reconstruction": "produces.reconstruction",
                            "source": "expects.image",
                        },
                    ),
                ],
            },
            model_input_roles_by_name={
                "image": "sample_image",
            },
            model_output_roles_by_name={
                "reconstruction": "reconstruction_image",
            },
        )


def test_rejects_invalid_loss_constructor_parameters() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Invalid constructor parameters at "
            r"`losses\.reconstruction\[0\]\.params`"
        ),
    ):
        resolve_composite_loss_configs(
            {
                "reconstruction": [
                    TrainingLossTermConfig(
                        name="mse",
                        params={"unknown_parameter": True},
                        composite_wiring={
                            "reconstruction": "produces.reconstruction",
                            "target": "expects.image",
                        },
                    ),
                ],
            },
            model_input_roles_by_name={
                "image": "sample_image",
            },
            model_output_roles_by_name={
                "reconstruction": "reconstruction_image",
            },
        )