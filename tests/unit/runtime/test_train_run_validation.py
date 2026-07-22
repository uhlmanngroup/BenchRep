from __future__ import annotations

from unittest.mock import Mock

import lightning as L
import pytest

from benchrep.interfaces.model_families import AUTOENCODER_FAMILY
from benchrep.runtime.train_run_validation import (
    validate_train_contract_compatibility,
)
from benchrep.runtime.utils import PreconditionResult
from tests.fixtures.models import (
    CompatibleExternalAutoencoder,
    PrivateBatchExternalAutoencoder,
)


def test_fully_internal_run_requires_no_compatibility_checks() -> None:
    result = validate_train_contract_compatibility(
        model_family=AUTOENCODER_FAMILY,
        model=CompatibleExternalAutoencoder(),
        model_is_external=False,
        datamodule_is_external=False,
        compatibility_policy="error",
    )

    assert result == PreconditionResult()


def test_compatible_external_model_with_internal_datamodule_passes() -> None:
    result = validate_train_contract_compatibility(
        model_family=AUTOENCODER_FAMILY,
        model=CompatibleExternalAutoencoder(),
        model_is_external=True,
        datamodule_is_external=False,
        compatibility_policy="error",
    )

    assert result == PreconditionResult()


def test_private_batch_external_model_with_internal_datamodule_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match=r"training_step.*missing required field.*x",
    ):
        validate_train_contract_compatibility(
            model_family=AUTOENCODER_FAMILY,
            model=PrivateBatchExternalAutoencoder(),
            model_is_external=True,
            datamodule_is_external=False,
            compatibility_policy="error",
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
            model_family=AUTOENCODER_FAMILY,
            model=model,
            model_is_external=True,
            datamodule_is_external=False,
            compatibility_policy="error",
        )


def test_internal_model_with_external_datamodule_requests_runtime_wrapping() -> None:
    result = validate_train_contract_compatibility(
        model_family=AUTOENCODER_FAMILY,
        model=CompatibleExternalAutoencoder(),
        model_is_external=False,
        datamodule_is_external=True,
        compatibility_policy="error",
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
        model_family=AUTOENCODER_FAMILY,
        model=PrivateBatchExternalAutoencoder(),
        model_is_external=True,
        datamodule_is_external=True,
        compatibility_policy="error",
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
        model_family=AUTOENCODER_FAMILY,
        model=PrivateBatchExternalAutoencoder(),
        model_is_external=True,
        datamodule_is_external=False,
        compatibility_policy="warn",
    )

    assert result == PreconditionResult()
    assert run_logger.warning.call_count == 2

    warning_messages = " ".join(
        str(call)
        for call in run_logger.warning.call_args_list
    )

    assert "training_step()" in warning_messages
    assert "predict_step()" in warning_messages
    assert "compatibility_policy='warn'" in warning_messages