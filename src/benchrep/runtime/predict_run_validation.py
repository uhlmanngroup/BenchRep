from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import lightning as L

import torch

from benchrep.runtime.utils import (
    PreconditionResult,
    run_compatibility_check,
)
from benchrep.architecture.composite_model_roles import (
    TENSOR_STRUCTURE_BY_ROLE,
    TensorStructure,
    CompositeModelTensorRole,
)
from benchrep.interfaces.model_families import CanonicalModelFamilySpec
from benchrep.interfaces.compatibility import (
    validate_external_model,
    sanity_check_predict_step_batch_annotation,
    sanity_check_predict_step_return_annotation,
    validate_prediction_output_structure,
)
from benchrep.interfaces.contracts import CompositePredictionOutput
from benchrep.assembly.resolvers import PredictionRunSpec


@dataclass(frozen=True, slots=True)
class PredictSourceInputsResult:
    """Validated prediction source inputs needed by the prediction runner."""

    checkpoint: Mapping[str, Any]
    state_dict: Mapping[str, Any]


def validate_predict_contract_compatibility(
        run_spec: PredictionRunSpec,
        model: L.LightningModule,
) -> PreconditionResult:
    model_family = run_spec.model_family
    model_is_external = run_spec.model_source != "config"
    datamodule_is_external = run_spec.datamodule_source != "config"
    compatibility_policy = run_spec.compatibility_policy
    external_model_only = model_is_external and not datamodule_is_external
    external_datamodule_only = datamodule_is_external and not model_is_external
    fully_internal_run = not model_is_external and not datamodule_is_external

    compatibility_warnings: list[str] = []
    default_result = PreconditionResult()

    if (
            fully_internal_run
            or not isinstance(model_family, CanonicalModelFamilySpec)
    ):
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
    run_spec: PredictionRunSpec,
) -> None:
    if not predictions:
        raise ValueError("Prediction returned no batches.")

    if isinstance(run_spec.model_family, CanonicalModelFamilySpec):
        for batch_idx, prediction in enumerate(predictions):
            validate_prediction_output_structure(
                prediction=prediction,
                model_family=run_spec.model_family,
                batch_idx=batch_idx,
                check_value_types=True,
            )

        return

    if run_spec.composite_model_spec is None:
        raise RuntimeError(
            "Composite prediction output validation requires a resolved "
            "Composite model specification."
        )

    for batch_idx, prediction in enumerate(predictions):
        _validate_composite_prediction_output(
            prediction=prediction,
            run_spec=run_spec,
            batch_idx=batch_idx,
        )


def infer_prediction_observation_count(
    *,
    predictions: Sequence[Any],
    run_spec: PredictionRunSpec,
) -> int | None:
    if not predictions:
        return None

    if isinstance(run_spec.model_family, CanonicalModelFamilySpec):
        batch_size_source = (
            getattr(prediction, "embedding", None)
            for prediction in predictions
        )

    else:
        model_spec = run_spec.composite_model_spec

        if model_spec is None:
            return None

        sample_image_names = [
            name
            for name, role
            in model_spec.declarations.model_input_roles_by_name.items()
            if role == "sample_image"
        ]

        if len(sample_image_names) != 1:
            return None

        sample_image_name = sample_image_names[0]

        batch_size_source = (
            (
                prediction.model_inputs.get(sample_image_name)
                if isinstance(prediction, CompositePredictionOutput)
                else None
            )
            for prediction in predictions
        )

    n_observations = 0

    for value in batch_size_source:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim < 1
        ):
            return None

        n_observations += int(value.shape[0])

    return n_observations


def _validate_composite_prediction_output(
    *,
    prediction: Any,
    run_spec: PredictionRunSpec,
    batch_idx: int,
) -> None:
    if not isinstance(prediction, CompositePredictionOutput):
        raise TypeError(
            f"Prediction batch {batch_idx} must be a "
            "`CompositePredictionOutput`, got "
            f"`{type(prediction).__name__}`."
        )

    model_spec = run_spec.composite_model_spec
    assert model_spec is not None

    declarations = model_spec.declarations

    _validate_composite_namespace_names(
        values=prediction.model_inputs,
        expected_names=set(
            declarations.model_input_roles_by_name
        ),
        namespace="model_inputs",
        batch_idx=batch_idx,
    )
    _validate_composite_namespace_names(
        values=prediction.model_outputs,
        expected_names=set(
            declarations.model_output_roles_by_name
        ),
        namespace="model_outputs",
        batch_idx=batch_idx,
    )
    _validate_composite_namespace_names(
        values=prediction.batch_metadata,
        expected_names=set(
            declarations.batch_metadata_roles_by_name
        ),
        namespace="batch_metadata",
        batch_idx=batch_idx,
    )

    expected_batch_size: int | None = None

    for name, role in declarations.model_input_roles_by_name.items():
        role: CompositeModelTensorRole

        batch_size = _validate_composite_prediction_tensor(
            value=prediction.model_inputs[name],
            structure=TENSOR_STRUCTURE_BY_ROLE[role],
            description=f"model input {name!r}",
            batch_idx=batch_idx,
        )

        expected_batch_size = _merge_prediction_batch_size(
            expected_batch_size=expected_batch_size,
            observed_batch_size=batch_size,
            description=f"model input {name!r}",
            batch_idx=batch_idx,
        )

    for name, role in declarations.model_output_roles_by_name.items():
        batch_size = _validate_composite_prediction_tensor(
            value=prediction.model_outputs[name],
            structure=TENSOR_STRUCTURE_BY_ROLE[role],
            description=f"model output {name!r}",
            batch_idx=batch_idx,
        )

        expected_batch_size = _merge_prediction_batch_size(
            expected_batch_size=expected_batch_size,
            observed_batch_size=batch_size,
            description=f"model output {name!r}",
            batch_idx=batch_idx,
        )

    assert expected_batch_size is not None

    for name, value in prediction.batch_metadata.items():
        if isinstance(value, torch.Tensor):
            if value.ndim < 1:
                raise ValueError(
                    f"Prediction batch {batch_idx} metadata {name!r} must "
                    "have a batch dimension."
                )

            metadata_batch_size = int(value.shape[0])

        elif isinstance(value, list):
            metadata_batch_size = len(value)

        else:
            raise TypeError(
                f"Prediction batch {batch_idx} metadata {name!r} must be a "
                "torch.Tensor or list, got "
                f"`{type(value).__name__}`."
            )

        _merge_prediction_batch_size(
            expected_batch_size=expected_batch_size,
            observed_batch_size=metadata_batch_size,
            description=f"batch metadata {name!r}",
            batch_idx=batch_idx,
        )


def _validate_composite_namespace_names(
    *,
    values: Any,
    expected_names: set[str],
    namespace: str,
    batch_idx: int,
) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(
            f"Prediction batch {batch_idx} `{namespace}` must be a mapping."
        )

    received_names = set(values)
    missing_names = sorted(expected_names - received_names)
    unexpected_names = sorted(received_names - expected_names)

    if missing_names or unexpected_names:
        raise ValueError(
            f"Prediction batch {batch_idx} `{namespace}` does not match its "
            f"Composite declarations: missing={missing_names}, "
            f"unexpected={unexpected_names}."
        )


def _validate_composite_prediction_tensor(
    *,
    value: Any,
    structure: TensorStructure,
    description: str,
    batch_idx: int,
) -> int:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Prediction batch {batch_idx} {description} must be a "
            f"torch.Tensor, got `{type(value).__name__}`."
        )

    if structure == "scalar":
        valid_shape = (
            value.ndim == 1
            or (value.ndim == 2 and value.shape[1] == 1)
        )
        expected_shape = "[batch] or [batch, 1]"

    elif structure == "vector":
        valid_shape = value.ndim == 2
        expected_shape = "[batch, features]"

    elif structure == "image":
        valid_shape = value.ndim == 4
        expected_shape = "[batch, channels, height, width]"

    else:
        raise RuntimeError(
            f"Unsupported Composite tensor structure {structure!r}."
        )

    if not valid_shape:
        raise ValueError(
            f"Prediction batch {batch_idx} {description} has shape "
            f"{tuple(value.shape)}; expected {expected_shape} for declared "
            f"structure {structure!r}."
        )

    return int(value.shape[0])


def _merge_prediction_batch_size(
    *,
    expected_batch_size: int | None,
    observed_batch_size: int,
    description: str,
    batch_idx: int,
) -> int:
    if expected_batch_size is None:
        return observed_batch_size

    if observed_batch_size != expected_batch_size:
        raise ValueError(
            f"Prediction batch {batch_idx} {description} contains "
            f"{observed_batch_size} observations; expected "
            f"{expected_batch_size}."
        )

    return expected_batch_size