"""Exercise Composite execution with small user-owned components and objectives."""

import pytest
import torch
from torch import nn

from benchrep.architecture.losses.base import LossTerm
from benchrep.architecture.losses.composite_model_contracts import LossComponent
from benchrep.architecture.losses.custom_objective import BaseCustomObjectiveLoss
from benchrep.architecture.models import CompositeModel
from benchrep.assembly.registries.core import LOSS_REGISTRIES_BY_ROLE, LossRegistry
from benchrep.assembly.resolvers.composite_model_resolver import (
    CompositeModelAssemblyStepSpec,
    CompositeModelComponentSpec,
    CompositeModelDeclarationsSpec,
    CompositeModelSpec,
)
from benchrep.assembly.resolvers.loss_resolver import resolve_composite_loss_configs
from benchrep.assembly.schemas import TrainingLossTermConfig


class PairedFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        features = x.flatten(1) * self.scale
        return {"features": features, "shifted": features + 1}


class ContextObjective(BaseCustomObjectiveLoss):
    def forward(self, *, batch, model_output):
        self.seen_batch = batch
        self.seen_output = model_output
        return (model_output["embedding"] * batch["importance"]).mean() + model_output[
            "projection"
        ].mean()


@pytest.fixture
def composite(monkeypatch):
    registry = LossRegistry(
        "runtime test custom objective",
        canonical_runtime_input_names=("batch", "model_output"),
        required_component_base=BaseCustomObjectiveLoss,
        explicit_composite_runtime_inputs_supported=False,
    )
    registry.register("context", LossComponent(ContextObjective))
    monkeypatch.setitem(LOSS_REGISTRIES_BY_ROLE, "custom_objective", registry)
    declarations = CompositeModelDeclarationsSpec(
        model_input_roles_by_name={"image": "sample_image"},
        batch_metadata_roles_by_name={"sample_id": "index"},
        model_output_roles_by_name={
            "embedding": "embedding_vector",
            "projection": "projection_vector",
        },
    )
    spec = CompositeModelSpec(
        declarations=declarations,
        components_by_id={
            "encoder": CompositeModelComponentSpec("encoder", "test", {})
        },
        assembly_steps_by_dependency_level={
            0: (
                CompositeModelAssemblyStepSpec(
                    step_id="encode",
                    component_id="encoder",
                    inputs_from_model_inputs={"x": "image"},
                    inputs_from_model_outputs={},
                    results_to_model_outputs={
                        "features": "embedding",
                        "shifted": "projection",
                    },
                ),
            )
        },
    )
    loss_specs = resolve_composite_loss_configs(
        {"custom_objective": [TrainingLossTermConfig(name="context", weight=0.25)]},
        model_input_roles_by_name=declarations.model_input_roles_by_name,
        model_output_roles_by_name=declarations.model_output_roles_by_name,
    )
    model = CompositeModel(
        model_spec=spec,
        loss_specs=loss_specs,
        components_by_id={"encoder": PairedFeatures()},
        losses_by_role={
            "custom_objective": {
                loss_specs[0].loss_id: LossTerm(loss=ContextObjective(), weight=0.25),
            }
        },
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
    )
    # Isolate execution from Lightning's logger; all computation and autograd are real.
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    return model


@pytest.fixture
def batch():
    return {
        "image": torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2),
        "sample_id": ["a", "b"],
        "importance": torch.tensor([[1.0], [3.0]]),
    }


def test_custom_objective_receives_full_context_and_backpropagates(composite, batch):
    loss = composite.training_step(batch, 0)
    objective = composite.losses_by_role["custom_objective"]["context"].loss
    flat = batch["image"].flatten(1)

    assert objective.seen_batch is batch
    assert set(objective.seen_output) == {"embedding", "projection"}
    torch.testing.assert_close(objective.seen_output["embedding"], 2 * flat)
    torch.testing.assert_close(objective.seen_output["projection"], 2 * flat + 1)
    expected = 0.25 * ((2 * flat * batch["importance"]).mean() + (2 * flat + 1).mean())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(
        composite.components_by_id["encoder"].scale.grad,
        0.25 * ((flat * batch["importance"]).mean() + flat.mean()),
    )


@pytest.mark.parametrize(
    ("result", "error_type", "message"),
    [
        pytest.param(
            torch.ones(2, 4),
            TypeError,
            "encode.*return a mapping",
            id="tensor-instead-of-mapping",
        ),
        pytest.param(
            {"features": torch.ones(2, 4)},
            ValueError,
            "missing result keys.*shifted",
            id="missing-output",
        ),
        pytest.param(
            {
                "features": torch.ones(2, 4),
                "shifted": torch.ones(2, 4),
                "extra": torch.ones(2, 4),
            },
            ValueError,
            "unexpected result keys.*extra",
            id="extra-output",
        ),
        pytest.param(
            {"features": "invalid", "shifted": torch.ones(2, 4)},
            TypeError,
            "features.*str.*torch.Tensor",
            id="nontensor-output",
        ),
        pytest.param(
            {"features": torch.ones(2, 1, 4), "shifted": torch.ones(2, 4)},
            ValueError,
            "embedding.*structure",
            id="wrong-rank",
        ),
        pytest.param(
            {"features": torch.ones(1, 4), "shifted": torch.ones(2, 4)},
            ValueError,
            "embedding.*batch",
            id="wrong-batch-size",
        ),
    ],
)
def test_forward_rejects_broken_component_contract(
    composite, batch, monkeypatch, result, error_type, message
):
    monkeypatch.setattr(
        composite.components_by_id["encoder"], "forward", lambda x: result
    )
    with pytest.raises(error_type, match=message):
        composite({"image": batch["image"]})


@pytest.mark.parametrize("field", ["image", "sample_id"])
def test_training_rejects_missing_declared_batch_fields(composite, batch, field):
    del batch[field]
    with pytest.raises(KeyError, match=f"missing declared fields.*{field}"):
        composite.training_step(batch, 0)


@pytest.mark.parametrize(
    ("result", "error_type", "message"),
    [
        (1.0, TypeError, "context.*torch.Tensor"),
        (torch.ones(2), ValueError, "context.*scalar tensor"),
    ],
)
def test_training_rejects_invalid_custom_loss_result(
    composite, batch, monkeypatch, result, error_type, message
):
    objective = composite.losses_by_role["custom_objective"]["context"].loss
    monkeypatch.setattr(objective, "forward", lambda **kwargs: result)
    with pytest.raises(error_type, match=message):
        composite.training_step(batch, 0)
