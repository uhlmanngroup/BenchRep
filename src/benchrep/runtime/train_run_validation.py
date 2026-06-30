from __future__ import annotations

import lightning as L

from collections.abc import Callable

from benchrep.records import get_run_logger
from benchrep.runtime.utils import CompatibilityPolicy
from benchrep.interfaces.model_families import ModelFamilySpec
from benchrep.interfaces.compatibility import (
    validate_external_model,
    sanity_check_training_step_batch_annotation,
    sanity_check_predict_step_batch_annotation,
    sanity_check_predict_step_return_annotation,
)


def validate_train_preconditions(
        model_family: ModelFamilySpec,
        model: L.LightningModule,
        datamodule: L.LightningDataModule,
        model_is_external: bool = False,
        datamodule_is_external: bool = False,
        compatibility_policy: CompatibilityPolicy = "error",
) -> None:
    external_model_only = model_is_external and not  datamodule_is_external
    external_datamodule_only = datamodule_is_external and not model_is_external
    fully_internal_run = not model_is_external and not datamodule_is_external

    if fully_internal_run:
        return

    if model_is_external:
        validate_external_model(model, model_family)

        _run_compatibility_check(
            check=lambda: sanity_check_predict_step_return_annotation(
                model=model,
                model_family=model_family,
                check_field_types=True,
            ),
            compatibility_policy=compatibility_policy,
            success_message=(
                "Predict-step return annotation sanity check passed; "
                "this only checks declared annotations and does not guarantee that "
                "prediction/export/evaluation will succeed. Runtime compatibility "
                "will be checked during prediction."
            ),
            error_prefix=(
                "BenchRep compatibility precondition failed. "
                "This run uses an external model, so `predict_step()` must declare a "
                "BenchRep-compatible prediction output annotation to pass this sanity check. "
            ),
            warning_prefix=(
                "BenchRep compatibility precondition failed. "
                "This run uses an external model, so `predict_step()` must declare a "
                "BenchRep-compatible prediction output annotation to pass this sanity check. "
                "Continuing because compatibility_policy='warn'. "
                "BenchRep prediction/export/evaluation may fail later."
            ),
        )

    if external_model_only:
        _run_compatibility_check(
            check=lambda: sanity_check_training_step_batch_annotation(
                model=model,
                model_family=model_family,
                check_field_types=True,
            ),
            compatibility_policy=compatibility_policy,
            success_message=(
                "Training-step batch annotation sanity check passed; "
                "external model declares compatibility with the BenchRep training batch contract."
            ),
            error_prefix=(
                "BenchRep compatibility precondition failed. "
                "This run uses an external model with a BenchRep-managed datamodule, "
                "so `training_step()` must declare a BenchRep-compatible batch annotation "
                "to pass this sanity check. "
            ),
            warning_prefix=(
                "BenchRep compatibility precondition failed. "
                "This run uses an external model with a BenchRep-managed datamodule, "
                "so `training_step()` must declare a BenchRep-compatible batch annotation "
                "to pass this sanity check. "
                "Continuing because compatibility_policy='warn'. "
                "Training may fail later."
            ),
        )

        _run_compatibility_check(
            check=lambda: sanity_check_predict_step_batch_annotation(
                model=model,
                model_family=model_family,
                check_field_types=True,
            ),
            compatibility_policy=compatibility_policy,
            success_message=(
                "Predict-step batch annotation sanity check passed; "
                "external model declares compatibility with the BenchRep prediction batch contract."
            ),
            error_prefix=(
                "BenchRep compatibility precondition failed. "
                "This run uses an external model with a BenchRep-managed datamodule, "
                "so `predict_step()` must declare a BenchRep-compatible batch annotation "
                "to pass this sanity check. "
            ),
            warning_prefix=(
                "BenchRep compatibility precondition failed. "
                "This run uses an external model with a BenchRep-managed datamodule, "
                "so `predict_step()` must declare a BenchRep-compatible batch annotation "
                "to pass this sanity check. "
                "Continuing because compatibility_policy='warn'. "
                "Prediction may fail later."
            ),
        )

    if external_datamodule_only:
        ...


def _run_compatibility_check(
    *,
    check: Callable[[], None],
    compatibility_policy: CompatibilityPolicy,
    success_message: str,
    error_prefix: str,
    warning_prefix: str,
) -> None:
    run_log = get_run_logger()

    try:
        check()
        run_log.info(success_message)

    except TypeError as exc:
        if compatibility_policy == "error":
            raise TypeError(
                f"{error_prefix} "
                f"Original reason: {exc}"
            ) from exc

        run_log.warning(
            "%s Original reason: %s",
            warning_prefix,
            exc,
        )