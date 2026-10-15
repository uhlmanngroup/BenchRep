import pytest

from benchrep.assembly.resolvers.loss_resolver import LossSpec
from benchrep.assembly.resolvers.training_config_resolver import (
    resolve_gradient_norm_configs,
)
from benchrep.assembly.schemas.training_config_schema import (
    TrainingInspectionConfig,
)


@pytest.fixture
def loss_specs():
    # Two distinct instances of the same registered loss.
    return tuple(
        LossSpec(
            loss_role="reconstruction",
            loss_id=loss_id,
            registry_entry_name="mse",
            weight=1.0,
            constructor_params={"reduction": "mean"},
        )
        for loss_id in ("morphology_mse", "skeleton_mse")
    )


def resolve(requests, loss_specs):
    inspection = TrainingInspectionConfig.model_validate({
        "gradient_norms": requests,
    })
    return resolve_gradient_norm_configs(
        inspection.gradient_norms,
        loss_specs=loss_specs,
        available_targets={"encoder", "decoder", "whole_model"},
    )


def per_loss(**overrides):
    return {
        "kind": "per_loss",
        "target": "encoder",
        "norm_type": "l2",
        "losses": ["reconstruction.morphology_mse"],
        **overrides,
    }


def test_selection_uses_resolved_identity_not_registered_loss_name(loss_specs):
    resolved = resolve(
        [
            per_loss(
                losses=["reconstruction.skeleton_mse"],
                every_n_batches=3,
            ),
            per_loss(
                target="decoder",
                losses="all",
                every_n_batches=5,
            ),
        ],
        loss_specs,
    )

    assert resolved[0].loss_keys == (
        ("reconstruction", "skeleton_mse"),
    )
    assert resolved[0].target == "encoder"
    assert resolved[0].every_n == 3

    assert set(resolved[1].loss_keys) == {
        ("reconstruction", "morphology_mse"),
        ("reconstruction", "skeleton_mse"),
    }
    assert resolved[1].target == "decoder"
    assert resolved[1].every_n == 5


@pytest.mark.parametrize(
    ("inspection_request", "message"),
    [
        (
            per_loss(target="missing_component"),
            "unknown target",
        ),
        (
            per_loss(losses=["reconstruction.mse"]),
            "unknown loss",
        ),
        (
            per_loss(losses=["regularization.morphology_mse"]),
            "unknown loss",
        ),
    ],
)
def test_unknown_references_are_rejected(inspection_request, message, loss_specs):
    with pytest.raises(ValueError, match=message):
        resolve([inspection_request], loss_specs)


@pytest.mark.parametrize(
    "requests",
    [
        # "all" overlaps with the explicitly selected loss.
        [
            per_loss(losses="all", every_n_batches=2),
            per_loss(every_n_batches=7),
        ],
        # Partial overlap between two explicit selections.
        [
            per_loss(losses=[
                "reconstruction.morphology_mse",
                "reconstruction.skeleton_mse",
            ]),
            per_loss(losses=["reconstruction.skeleton_mse"]),
        ],
        # Repetition within one request would also double-log.
        [
            per_loss(losses=[
                "reconstruction.morphology_mse",
                "reconstruction.morphology_mse",
            ]),
        ],
        # Different frequencies do not make distinct optimizer plots.
        [
            {
                "kind": "optimizer_gradients",
                "target": "encoder",
                "every_n_optimizer_steps": frequency,
            }
            for frequency in (2, 7)
        ],
    ],
)
def test_overlapping_measurements_are_rejected(requests, loss_specs):
    with pytest.raises(ValueError, match="duplicate gradient measurement"):
        resolve(requests, loss_specs)


def test_distinct_measurements_can_coexist(loss_specs):
    requests = [
        per_loss(),
        per_loss(target="decoder"),
        per_loss(norm_type="l1"),
        per_loss(losses=["reconstruction.skeleton_mse"]),
        {
            "kind": "optimizer_gradients",
            "target": "encoder",
            "norm_type": "l2",
            "every_n_optimizer_steps": 4,
        },
    ]

    resolved = resolve(requests, loss_specs)

    assert len(resolved) == 5
    assert resolved[-1].kind == "optimizer_gradients"
    assert resolved[-1].every_n == 4
    assert resolved[-1].loss_keys == ()


def test_all_requires_losses_but_optimizer_inspection_does_not():
    with pytest.raises(ValueError, match="no configured losses"):
        resolve([per_loss(losses="all")], loss_specs=())

    resolved = resolve(
        [{
            "kind": "optimizer_gradients",
            "target": "whole_model",
        }],
        loss_specs=(),
    )

    assert len(resolved) == 1
    assert resolved[0].loss_keys == ()