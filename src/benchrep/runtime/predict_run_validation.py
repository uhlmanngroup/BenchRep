from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import lightning as L

import torch

from benchrep.runtime.utils import (
    CompatibilityPolicy,
    PreconditionResult,
    run_compatibility_check,
)
from benchrep.interfaces.model_families import ModelFamilySpec
from benchrep.interfaces.compatibility import (
    validate_external_model,
    sanity_check_predict_step_batch_annotation,
    sanity_check_predict_step_return_annotation,
    validate_prediction_output_structure,
)
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
    external_model_only = model_is_external and not datamodule_is_external
    external_datamodule_only = datamodule_is_external and not model_is_external
    fully_internal_run = not model_is_external and not datamodule_is_external

    compatibility_warnings: list[str] = []
    default_result = PreconditionResult()

    if fully_internal_run:
        return default_result

    if model_is_external:
        validate_external_model(model, model_family)

        predict_step_return_annotation_warning = run_compatibility_check(
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
        if predict_step_return_annotation_warning is not None:
            compatibility_warnings.append(predict_step_return_annotation_warning)

    if external_model_only:
        predict_step_batch_annotation_warning = run_compatibility_check(
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
        if predict_step_batch_annotation_warning is not None:
            compatibility_warnings.append(predict_step_batch_annotation_warning)

    if external_datamodule_only:
        return PreconditionResult(
            should_wrap_batch_contract_errors=True,
            expected_batch_type=model_family.expected_batch_type,
            expected_batch_contract_kind=model_family.expected_batch_contract_kind,
            model_family_name=model_family.name,
            warnings=tuple(compatibility_warnings),
        )

    return PreconditionResult(
        warnings=tuple(compatibility_warnings),
    )


def prepare_predict_source_inputs(
    run_spec: PredictionRunSpec,
) -> PredictSourceInputsResult:

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


def validate_prediction_outputs(
    *,
    predictions: Sequence[Any],
    model_family: ModelFamilySpec,
) -> None:
    if not predictions:
        raise ValueError("Prediction returned no batches.")

    for batch_idx, prediction in enumerate(predictions):
        validate_prediction_output_structure(
            prediction=prediction,
            model_family=model_family,
            batch_idx=batch_idx,
            check_value_types=True,
        )


def infer_prediction_observation_count(
    *,
    predictions: Sequence[Any],
) -> int | None:
    if not predictions:
        return None

    n_observations = 0

    for prediction in predictions:
        embedding = getattr(prediction, "embedding", None)

        if (
            not isinstance(embedding, torch.Tensor)
            or embedding.ndim < 1
        ):
            return None

        n_observations += int(embedding.shape[0])

    return n_observations
