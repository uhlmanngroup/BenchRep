from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import lightning as L
import pytest

from benchrep.assembly.resolvers.training_config_resolver import (
    TrainingComponentSource,
    TrainingRunIdentitySpec,
    TrainingRunSpec,
)
from benchrep.interfaces.model_families import AUTOENCODER_FAMILY
from benchrep.runtime.train_run_validation import (
    validate_train_contract_compatibility,
)
from benchrep.runtime.utils import PreconditionResult
from tests.fixtures.models import (
    CompatibleExternalAutoencoder,
    PrivateBatchExternalAutoencoder,
)


def _make_run_spec(
    *,
    model_source: TrainingComponentSource,
    datamodule_source: TrainingComponentSource,
    compatibility_policy: Literal["error", "warn"] = "error",
) -> TrainingRunSpec:
    return TrainingRunSpec(
        stage="training",
        training_config=Mock(),
        model_family=AUTOENCODER_FAMILY,
        model_source=model_source,
        datamodule_source=datamodule_source,
        compatibility_policy=compatibility_policy,
        run_identity=TrainingRunIdentitySpec(
            output_root=Path("."),
            project_name=None,
            model_name="test_model",
        ),
    )


def test_fully_internal_run_requires_no_compatibility_checks() -> None:
    result = validate_train_contract_compatibility(
        run_spec=_make_run_spec(
            model_source="config",
            datamodule_source="config",
        ),
        model=CompatibleExternalAutoencoder(),
    )

    assert result == PreconditionResult()


def test_compatible_external_model_with_internal_datamodule_passes() -> None:
    result = validate_train_contract_compatibility(
        run_spec=_make_run_spec(
            model_source="external_object",
            datamodule_source="config",
        ),
        model=CompatibleExternalAutoencoder(),
    )

    assert result == PreconditionResult()


def test_private_batch_external_model_with_internal_datamodule_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match=r"training_step.*missing required field.*x",
    ):
        validate_train_contract_compatibility(
            run_spec=_make_run_spec(
                model_source="external_object",
                datamodule_source="config",
            ),
            model=PrivateBatchExternalAutoencoder(),
        )


def test_plain_lightning_model_is_rejected_as_external_model() -> None:
    model = L.LightningModule()

    with pytest.raises(
        TypeError,
        match=(
            "must be an instance of "
            "`BenchRepAutoencoderModel`"
        ),
    ):
        validate_train_contract_compatibility(
            run_spec=_make_run_spec(
                model_source="external_object",
                datamodule_source="config",
            ),
            model=model,
        )


def test_internal_model_with_external_datamodule_requests_runtime_wrapping() -> None:
    result = validate_train_contract_compatibility(
        run_spec=_make_run_spec(
            model_source="config",
            datamodule_source="external_object",
        ),
        model=CompatibleExternalAutoencoder(),
    )

    assert result.should_wrap_batch_contract_errors is True
    assert result.expected_batch_type is AUTOENCODER_FAMILY.expected_batch_type
    assert (
        result.expected_batch_contract_kind
        == AUTOENCODER_FAMILY.expected_batch_contract_kind
    )
    assert result.model_family_name == AUTOENCODER_FAMILY.name


def test_external_model_and_datamodule_may_use_private_batch_contract() -> None:
    result = validate_train_contract_compatibility(
        run_spec=_make_run_spec(
            model_source="external_object",
            datamodule_source="external_object",
        ),
        model=PrivateBatchExternalAutoencoder(),
    )

    assert result == PreconditionResult()


def test_warning_policy_continues_after_batch_annotation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_logger = Mock()

    monkeypatch.setattr(
        "benchrep.runtime.utils.get_run_logger",
        lambda: run_logger,
    )

    result = validate_train_contract_compatibility(
        run_spec=_make_run_spec(
            model_source="external_object",
            datamodule_source="config",
            compatibility_policy="warn",
        ),
        model=PrivateBatchExternalAutoencoder(),
    )

    assert run_logger.warning.call_count == 2

    warning_messages = tuple(
        call.args[0]
        for call in run_logger.warning.call_args_list
    )

    assert result == PreconditionResult(
        warnings=warning_messages,
    )

    combined_warning_messages = " ".join(warning_messages)

    assert "training_step()" in combined_warning_messages
    assert "predict_step()" in combined_warning_messages
    assert "compatibility_policy='warn'" in combined_warning_messages