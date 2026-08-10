from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchrep import list_registries, list_registered_components
from benchrep.assembly.schemas import TrainingConfig
from tests.fixtures.configs.configs import make_training_config


EXPECTED_ADDITIONAL_CALLBACKS = {
    "device_stats_monitor",
    "gradient_accumulation_scheduler",
    "learning_rate_monitor",
    "model_summary",
    "rich_model_summary",
    "rich_progress_bar",
    "timer",
    "tqdm_progress_bar",
    "weight_averaging",
}


def test_callback_registry_is_discoverable() -> None:
    assert (
        list_registries()["callback"]
        == "benchrep.assembly.registries.CALLBACKS"
    )

    assert list_registered_components("callback") == tuple(
        sorted(EXPECTED_ADDITIONAL_CALLBACKS)
    )


@pytest.mark.parametrize(
    "callback_name",
    [
        "device_stats_monitor",
        "learning_rate_monitor",
    ],
)
def test_logger_dependent_callback_requires_logger(
    callback_name: str,
) -> None:
    config_data = make_training_config().model_dump(mode="python")
    config_data["logger"] = None
    config_data["additional_callbacks"] = [
        {
            "name": callback_name,
            "params": {},
        },
    ]

    with pytest.raises(
        ValidationError,
        match="`logger` must be configured",
    ) as exc_info:
        TrainingConfig.model_validate(config_data)

    assert callback_name in str(exc_info.value)


@pytest.mark.parametrize(
    "callback_name",
    [
        "device_stats_monitor",
        "learning_rate_monitor",
    ],
)
def test_logger_dependent_callback_accepts_configured_logger(
    callback_name: str,
) -> None:
    config_data = make_training_config().model_dump(mode="python")
    config_data["additional_callbacks"] = [
        {
            "name": callback_name,
            "params": {},
        },
    ]

    config = TrainingConfig.model_validate(config_data)

    assert config.logger is not None
    assert config.additional_callbacks[0].name == callback_name