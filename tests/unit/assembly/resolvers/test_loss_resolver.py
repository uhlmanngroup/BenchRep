from __future__ import annotations

import pytest

from benchrep.assembly.resolvers.loss_resolver import (
    resolve_loss_configs,
)
from benchrep.assembly.schemas.training_config_schema import (
    TrainingLossTermConfig,
)


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