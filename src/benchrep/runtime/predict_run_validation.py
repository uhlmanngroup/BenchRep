from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import lightning as L

import torch

from benchrep.records import get_run_logger
from benchrep.runtime.utils import (
    CompatibilityPolicy,
    PreconditionResult,
    AuditItem,
    run_compatibility_check,
    format_expected_contract,
    audit_existing_file,
    audit_existing_dir,
)
from benchrep.runtime.run_context import RunContext
from benchrep.interfaces.model_families import ModelFamilySpec
from benchrep.interfaces.compatibility import (
    validate_external_model,
    sanity_check_predict_step_batch_annotation,
    sanity_check_predict_step_return_annotation,
)
from benchrep.assembly.config import load_yaml
from benchrep.assembly.resolvers import PredictionRunSpec


@dataclass(frozen=True, slots=True)
class PredictSourceInputsResult:
    """Validated prediction source inputs needed by the prediction runner."""

    checkpoint: Mapping[str, Any]
    state_dict: Mapping[str, Any]


def validate_predict_contract_compatibility(
        model_family: ModelFamilySpec,
        model: L.LightningModule,
        model_is_external: bool = False,
        datamodule_is_external: bool = False,
        compatibility_policy: CompatibilityPolicy = "error",
) -> PreconditionResult:
    external_model_only = model_is_external and not  datamodule_is_external
    external_datamodule_only = datamodule_is_external and not model_is_external
    fully_internal_run = not model_is_external and not datamodule_is_external

    default_result = PreconditionResult()

    if fully_internal_run:
        return default_result

    if model_is_external:
        validate_external_model(model, model_family)

        run_compatibility_check(
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
                "will be checked after producing the predictions."
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
        run_compatibility_check(
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
        return PreconditionResult(
            should_wrap_batch_contract_errors=True,
            expected_batch_type=model_family.expected_batch_type,
            expected_batch_contract_kind=model_family.expected_batch_contract_kind,
            model_family_name=model_family.name,
        )



    return default_result


def validate_predict_source_inputs(
    run_spec: PredictionRunSpec,
) -> PredictSourceInputsResult:
    manifest_status = run_spec.training_manifest.get("status")

    if manifest_status != "completed":
        raise ValueError(
            "Prediction requires a completed training manifest, "
            f"but manifest status is {manifest_status!r}."
        )

    try:
        checkpoint = torch.load(run_spec.checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load checkpoint from '{run_spec.checkpoint_path}'. "
            f"Original error ({type(exc).__name__}): {exc}"
        ) from exc

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Loaded checkpoint must be a mapping, "
            f"got {type(checkpoint).__name__}."
        )

    state_dict = checkpoint.get("state_dict")

    if state_dict is None:
        raise KeyError(
            f"Checkpoint at '{run_spec.checkpoint_path}' does not contain `state_dict`."
        )

    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "Checkpoint `state_dict` must be a mapping, "
            f"got {type(state_dict).__name__}."
        )

    return PredictSourceInputsResult(
        checkpoint=checkpoint,
        state_dict=state_dict,
    )
