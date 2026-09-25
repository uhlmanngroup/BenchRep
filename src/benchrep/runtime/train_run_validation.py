from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L

from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from benchrep.assembly.schemas import TrainingCheckpointConfig
from benchrep.runtime.utils import (
    PreconditionResult,
    run_compatibility_check,
)
from benchrep.interfaces.compatibility import (
    validate_external_model,
    sanity_check_training_step_batch_annotation,
    sanity_check_predict_step_batch_annotation,
    sanity_check_predict_step_return_annotation,
)
from benchrep.interfaces.model_families import (
    CanonicalModelFamilySpec,
)

if TYPE_CHECKING:
    from benchrep.assembly.resolvers.training_config_resolver import (
        TrainingRunSpec,
    )


def validate_train_contract_compatibility(
        run_spec: TrainingRunSpec,
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

    # Composite contracts are declaration-driven and so validated during graph
    # resolution and model execution, not here.
    if fully_internal_run or not isinstance(model_family, CanonicalModelFamilySpec):
        return default_result

    if model_is_external:
        validate_external_model(model, model_family)

        return_annotation_warning = run_compatibility_check(
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
        if return_annotation_warning is not None:
            compatibility_warnings.append(return_annotation_warning)

    if external_model_only:
        train_step_batch_annotation_warning = run_compatibility_check(
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
        if train_step_batch_annotation_warning is not None:
            compatibility_warnings.append(train_step_batch_annotation_warning)

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


def validate_training_checkpoint_outputs(
    *,
    checkpoint_config: TrainingCheckpointConfig,
    checkpoint_callback: ModelCheckpoint,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return checkpoint errors and warnings found after training."""
    last_checkpoint_exists = bool(
        checkpoint_callback.last_model_path
        and Path(checkpoint_callback.last_model_path).is_file()
    )

    best_checkpoint_exists = bool(
        checkpoint_callback.best_model_path
        and Path(checkpoint_callback.best_model_path).is_file()
    )

    ranked_checkpoint_count = sum(
        Path(path).is_file()
        for path in checkpoint_callback.best_k_models
    )

    usable_checkpoint_exists = (
        last_checkpoint_exists
        or best_checkpoint_exists
        or ranked_checkpoint_count > 0
    )

    if not usable_checkpoint_exists:
        return (
            ("Training produced no usable checkpoint.",),
            (),
        )

    warnings: list[str] = []

    if checkpoint_config.save_last and not last_checkpoint_exists:
        warnings.append(
            "Checkpointing requested `save_last=True`, but no last "
            "checkpoint was produced."
        )

    ranked_checkpointing_requested = (
        checkpoint_config.monitor is not None
        and checkpoint_config.save_top_k != 0
    )

    if ranked_checkpointing_requested:
        requested_top_k = checkpoint_config.save_top_k

        if (
            requested_top_k > 0
            and ranked_checkpoint_count < requested_top_k
        ):
            warnings.append(
                f"Checkpointing requested save_top_k={requested_top_k}, "
                f"but only {ranked_checkpoint_count} ranked checkpoint(s) "
                "were produced."
            )

        elif requested_top_k == -1 and ranked_checkpoint_count == 0:
            warnings.append(
                "Checkpointing requested ranked checkpoints, but none "
                "were produced."
            )

        if ranked_checkpoint_count > 0 and not best_checkpoint_exists:
            warnings.append(
                "Ranked checkpoints exist, but the recorded best "
                "checkpoint is unavailable."
            )

    return (), tuple(warnings)
