import math

import pytest

from benchrep.assembly.builders.loss_builder import build_loss_terms
from benchrep.assembly.resolvers.loss_resolver import resolve_loss_configs
from benchrep.assembly.schemas.training_config_schema import TrainingLossTermConfig


@pytest.mark.parametrize("interval", ["epoch", "step"])
@pytest.mark.parametrize(
    ("name", "params", "midpoint"),
    [
        ("linear", {}, 0.55),
        ("cosine", {}, 0.55),
        ("sigmoid", {"steepness": 10.0}, 0.55),
        ("exponential", {}, math.sqrt(0.1)),
        ("piecewise_linear", {"milestones": [(5, 0.1), (15, 1.0)]}, 0.55),
        ("piecewise_constant", {"milestones": [(5, 0.1), (15, 1.0)]}, 0.1),
    ],
)
def test_configured_schedule_reaches_loss_term(
    name: str,
    params: dict,
    midpoint: float,
    interval: str,
) -> None:
    if "milestones" not in params:
        params = {
            "start": 5,
            "end": 15,
            "start_factor": 0.1,
            "end_factor": 1.0,
            **params,
        }

    config = TrainingLossTermConfig.model_validate({
        "name": "mse",
        "weight": 2.0,
        "weight_schedule": {
            "name": name,
            "params": params,
            "interval": interval,
            "cycle": {"length": 20},
        },
    })
    specs = resolve_loss_configs({"reconstruction": [config]})
    term = build_loss_terms(specs)["reconstruction"]["mse"]

    positions = [0, 5, 10, 15, 19, 20]
    expected_factors = [0.1, 0.1, midpoint, 1.0, 1.0, 0.1]

    # Query out of order to also check independence from call history.
    for index in [4, 0, 5, 2, 1, 3]:
        position = positions[index]
        actual = term.effective_weight(
            epoch=position if interval == "epoch" else 99,
            step=position if interval == "step" else 99,
        )
        assert actual == pytest.approx(2.0 * expected_factors[index])

    assert term.weight == 2.0


def test_omitted_schedule_preserves_fixed_weight() -> None:
    config = TrainingLossTermConfig(name="mse", weight=0.3)
    specs = resolve_loss_configs({"reconstruction": [config]})
    term = build_loss_terms(specs)["reconstruction"]["mse"]

    assert term.effective_weight(epoch=0, step=0) == 0.3
    assert term.effective_weight(epoch=10, step=100) == 0.3