from __future__ import annotations

import lightning as L

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from benchrep.records import get_run_logger
from benchrep.runtime.utils import CompatibilityPolicy
from benchrep.interfaces.contracts import ContractKind
from benchrep.interfaces.model_families import ModelFamilySpec
from benchrep.interfaces.compatibility import (
    validate_external_model,
    sanity_check_training_step_batch_annotation,
    sanity_check_predict_step_batch_annotation,
    sanity_check_predict_step_return_annotation,
)


@dataclass(frozen=True, slots=True)
class TrainPreconditionResult:
    should_wrap_training_errors_with_batch_hint: bool = False
    expected_batch_type: type[Any] | None = None
    expected_batch_contract_kind: ContractKind | None = None
    model_family_name: str | None = None


def validate_train_preconditions(
        model_family: ModelFamilySpec,
        model: L.LightningModule,
        model_is_external: bool = False,
        datamodule_is_external: bool = False,
        compatibility_policy: CompatibilityPolicy = "error",
) -> TrainPreconditionResult:
    external_model_only = model_is_external and not  datamodule_is_external
    external_datamodule_only = datamodule_is_external and not model_is_external
    fully_internal_run = not model_is_external and not datamodule_is_external

    default_result = TrainPreconditionResult()

    if fully_internal_run:
        return default_result

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
                "BenchRep-compatible prediction output annotation to pass this sanity check."
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
                "to pass this sanity check."
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
                "to pass this sanity check."
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
        return TrainPreconditionResult(
            should_wrap_training_errors_with_batch_hint=True,
            expected_batch_type=model_family.expected_batch_type,
            expected_batch_contract_kind=model_family.expected_batch_contract_kind,
            model_family_name=model_family.name,
        )

    return default_result


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


def format_external_datamodule_training_failure_message(
    *,
    precondition_result: TrainPreconditionResult,
    original_error: BaseException,
) -> str:
    expected_batch_type = precondition_result.expected_batch_type
    expected_batch_contract_kind = precondition_result.expected_batch_contract_kind
    model_family_name = precondition_result.model_family_name

    if expected_batch_type is None or expected_batch_contract_kind is None:
        return (
            "Training failed. "
            f"Original error ({type(original_error).__name__}): {original_error}"
        )

    return (
        "Training failed while using an external datamodule with a BenchRep internal model. "
        "This may indicate that the datamodule does not produce the expected BenchRep "
        "batch contract, although the original error may also be unrelated.\n\n"
        f"Expected batch contract for model family `{model_family_name}`:\n"
        f"{_format_expected_contract(expected_batch_type, expected_batch_contract_kind)}\n\n"
        f"Original error ({type(original_error).__name__}): {original_error}"
    )


def _format_expected_contract(
    expected_type: type[Any],
    expected_contract_kind: ContractKind,
) -> str:
    type_name = getattr(expected_type, "__name__", repr(expected_type))
    lines = [
        f"- contract: `{type_name}`",
        f"- kind: `{expected_contract_kind}`",
    ]

    if expected_contract_kind == "typeddict":
        type_hints = get_type_hints(expected_type)
        required_keys = getattr(expected_type, "__required_keys__", frozenset())
        optional_keys = getattr(expected_type, "__optional_keys__", frozenset())

        if required_keys:
            lines.append("- required fields:")
            for key in sorted(required_keys):
                annotation = type_hints.get(key, Any)
                lines.append(f"  - `{key}`: `{_format_annotation(annotation)}`")

        if optional_keys:
            lines.append("- optional fields:")
            for key in sorted(optional_keys):
                annotation = type_hints.get(key, Any)
                lines.append(f"  - `{key}`: `{_format_annotation(annotation)}`")

    return "\n".join(lines)


def _format_annotation(annotation: Any) -> str:
    return getattr(annotation, "__name__", repr(annotation))