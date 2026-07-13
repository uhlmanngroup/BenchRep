from __future__ import annotations

import lightning as L

from typing import Any, Literal
from pathlib import Path

from benchrep.runtime.utils import (
    CompatibilityPolicy,
    PreconditionResult,
    AuditItem,
    run_compatibility_check,
    audit_existing_file,
    audit_existing_dir,
    log_audit_summary,
)
from benchrep.runtime.run_context import RunContext
from benchrep.interfaces.model_families import ModelFamilySpec
from benchrep.interfaces.compatibility import (
    validate_external_model,
    sanity_check_training_step_batch_annotation,
    sanity_check_predict_step_batch_annotation,
    sanity_check_predict_step_return_annotation,
)
from benchrep.assembly.config import load_yaml


def validate_train_contract_compatibility(
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
        run_compatibility_check(
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


def audit_train_outputs(
    *,
    run_context: RunContext,
    input_config_path: Path | str | None = None,
    resolved_config_path: Path | str,
    checkpoint_dir: Path | str,
    training_manifest_path: Path | str,
    torchview_requested: bool,
    torchview_graph_path: Path | str | None,
    model_source: Literal["config", "external_object"],
    model_class_name: str,
    datamodule_source: Literal["config", "external_object"],
    datamodule_class_name: str,
) -> list[AuditItem]:
    audit_items: list[AuditItem] = []

    training_manifest_path = Path(training_manifest_path)
    training_manifest: dict[str, Any] | None = None

    # -------------------------
    # Config records
    # -------------------------
    if input_config_path is None:
        audit_items.append(
            AuditItem(
                name="input config",
                status="skipped",
                message=(
                    "no input config path was provided; this is expected for runs started "
                    "from a full config object or config_components rather than a YAML file"
                ),
            )
        )
    else:
        audit_existing_file(
            audit_items=audit_items,
            name="input config",
            path=input_config_path,
            expected_suffixes={".yaml", ".yml"},
        )

    audit_existing_file(
        audit_items=audit_items,
        name="resolved config",
        path=resolved_config_path,
        expected_suffixes={".yaml", ".yml"},
    )

    # -------------------------
    # Training manifest
    # -------------------------
    manifest_path_is_valid = audit_existing_file(
        audit_items=audit_items,
        name="training manifest",
        path=training_manifest_path,
        expected_suffixes={".yaml", ".yml"},
    )

    if manifest_path_is_valid:
        try:
            loaded_manifest = load_yaml(training_manifest_path)

            if not isinstance(loaded_manifest, dict):
                audit_items.append(
                    AuditItem(
                        name="training manifest load",
                        status="error",
                        message=(
                            f"loaded YAML object from '{training_manifest_path}', "
                            f"but expected a mapping and got "
                            f"{type(loaded_manifest).__name__}"
                        ),
                    )
                )
            else:
                training_manifest = loaded_manifest
                audit_items.append(
                    AuditItem(
                        name="training manifest load",
                        status="ok",
                        message=f"loaded YAML mapping from '{training_manifest_path}'",
                    )
                )

        except Exception as exc:
            audit_items.append(
                AuditItem(
                    name="training manifest load",
                    status="error",
                    message=(
                        f"could not be loaded as YAML: '{training_manifest_path}'. "
                        f"Original error ({type(exc).__name__}): {exc}"
                    ),
                )
            )

    # -------------------------
    # Manifest status
    # -------------------------
    if training_manifest is None:
        audit_items.append(
            AuditItem(
                name="manifest status",
                status="skipped",
                message="could not be checked because the training manifest is unavailable",
            )
        )

    else:
        status = training_manifest.get("status")

        if status == "completed":
            audit_items.append(
                AuditItem(
                    name="manifest status",
                    status="ok",
                    message="status is 'completed'",
                )
            )
        else:
            audit_items.append(
                AuditItem(
                    name="manifest status",
                    status="error",
                    message=f"status is {status!r}, expected 'completed'",
                )
            )

    # -------------------------
    # Manifest run output dir
    # -------------------------
    if training_manifest is None:
        audit_items.append(
            AuditItem(
                name="manifest run output dir",
                status="skipped",
                message="could not be checked because the training manifest is unavailable",
            )
        )

    else:
        run_section = training_manifest.get("run")

        if not isinstance(run_section, dict):
            audit_items.append(
                AuditItem(
                    name="manifest run output dir",
                    status="error",
                    message="manifest is missing mapping section `run`",
                )
            )

        elif "output_dir" not in run_section:
            audit_items.append(
                AuditItem(
                    name="manifest run output dir",
                    status="error",
                    message="manifest is missing `run.output_dir`",
                )
            )

        else:
            manifest_output_dir = Path(run_section["output_dir"])

            audit_existing_dir(
                audit_items=audit_items,
                name="manifest run output dir",
                path=manifest_output_dir,
            )

            if manifest_output_dir != run_context.output_dir:
                audit_items.append(
                    AuditItem(
                        name="manifest run output dir consistency",
                        status="warning",
                        message=(
                            f"`run.output_dir` is '{manifest_output_dir}', "
                            f"but current run output directory is '{run_context.output_dir}'"
                        ),
                    )
                )

    # -------------------------
    # Training provenance
    # -------------------------
    audit_items.append(
        AuditItem(
            name="model provenance",
            status="ok" if model_source == "config" else "warning",
            message=(
                f"model_source={model_source!r}, model_class={model_class_name!r}"
                if model_source == "config"
                else (
                    f"model_source={model_source!r}, model_class={model_class_name!r}; "
                    "training used an externally supplied Python model object "
                    "and is not fully reconstructable from config alone"
                )
            ),
        )
    )

    audit_items.append(
        AuditItem(
            name="datamodule provenance",
            status="ok" if datamodule_source == "config" else "warning",
            message=(
                f"datamodule_source={datamodule_source!r}, "
                f"datamodule_class={datamodule_class_name!r}"
                if datamodule_source == "config"
                else (
                    f"datamodule_source={datamodule_source!r}, "
                    f"datamodule_class={datamodule_class_name!r}; "
                    "training used an externally supplied Python datamodule object "
                    "and is not fully reconstructable from config alone"
                )
            ),
        )
    )

    # -------------------------
    # Checkpoints
    # -------------------------
    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir_is_valid = audit_existing_dir(
        audit_items=audit_items,
        name="checkpoint directory",
        path=checkpoint_dir,
    )

    if checkpoint_dir_is_valid:
        ckpt_files = sorted(checkpoint_dir.glob("*.ckpt"))

        if ckpt_files:
            audit_items.append(
                AuditItem(
                    name="checkpoint files",
                    status="ok",
                    message=f"found {len(ckpt_files)} checkpoint file(s) in '{checkpoint_dir}'",
                )
            )
        else:
            audit_items.append(
                AuditItem(
                    name="checkpoint files",
                    status="error",
                    message=f"no `.ckpt` files found in '{checkpoint_dir}'",
                )
            )

    if training_manifest is None:
        audit_items.append(
            AuditItem(
                name="manifest checkpoint directory",
                status="skipped",
                message="could not be checked because the training manifest is unavailable",
            )
        )
        audit_items.append(
            AuditItem(
                name="best checkpoint",
                status="skipped",
                message="could not be checked because the training manifest is unavailable",
            )
        )
        audit_items.append(
            AuditItem(
                name="last checkpoint",
                status="skipped",
                message="could not be checked because the training manifest is unavailable",
            )
        )

    else:
        checkpoints_section = training_manifest.get("checkpoints")

        if not isinstance(checkpoints_section, dict):
            audit_items.append(
                AuditItem(
                    name="manifest checkpoints",
                    status="error",
                    message="manifest is missing mapping section `checkpoints`",
                )
            )

        else:
            manifest_checkpoint_dir_raw = checkpoints_section.get("checkpoint_dir")

            if manifest_checkpoint_dir_raw is None:
                audit_items.append(
                    AuditItem(
                        name="manifest checkpoint directory",
                        status="error",
                        message="manifest is missing `checkpoints.checkpoint_dir`",
                    )
                )

            else:
                manifest_checkpoint_dir = Path(manifest_checkpoint_dir_raw)

                audit_existing_dir(
                    audit_items=audit_items,
                    name="manifest checkpoint directory",
                    path=manifest_checkpoint_dir,
                )

                if manifest_checkpoint_dir != checkpoint_dir:
                    audit_items.append(
                        AuditItem(
                            name="manifest checkpoint directory consistency",
                            status="warning",
                            message=(
                                f"`checkpoints.checkpoint_dir` is '{manifest_checkpoint_dir}', "
                                f"but expected checkpoint directory is '{checkpoint_dir}'"
                            ),
                        )
                    )

            best_checkpoint_raw = checkpoints_section.get("best_checkpoint_path")

            if best_checkpoint_raw is None:
                audit_items.append(
                    AuditItem(
                        name="best checkpoint",
                        status="warning",
                        message="manifest has no `checkpoints.best_checkpoint_path`; prediction with checkpoint='best' will fail",
                    )
                )

            else:
                best_checkpoint_path = Path(best_checkpoint_raw)

                if best_checkpoint_path.exists() and best_checkpoint_path.is_file():
                    audit_items.append(
                        AuditItem(
                            name="best checkpoint",
                            status="ok",
                            message=f"available at '{best_checkpoint_path}'",
                        )
                    )
                else:
                    audit_items.append(
                        AuditItem(
                            name="best checkpoint",
                            status="error",
                            message=f"manifest points to missing/non-file checkpoint: '{best_checkpoint_path}'",
                        )
                    )

            last_checkpoint_raw = checkpoints_section.get("last_checkpoint_path")

            if last_checkpoint_raw is None:
                audit_items.append(
                    AuditItem(
                        name="last checkpoint",
                        status="warning",
                        message="manifest has no `checkpoints.last_checkpoint_path`; prediction with checkpoint='last' will fail",
                    )
                )

            else:
                last_checkpoint_path = Path(last_checkpoint_raw)

                if last_checkpoint_path.exists() and last_checkpoint_path.is_file():
                    audit_items.append(
                        AuditItem(
                            name="last checkpoint",
                            status="ok",
                            message=f"available at '{last_checkpoint_path}'",
                        )
                    )
                else:
                    audit_items.append(
                        AuditItem(
                            name="last checkpoint",
                            status="error",
                            message=f"manifest points to missing/non-file checkpoint: '{last_checkpoint_path}'",
                        )
                    )

    # -------------------------
    # Torchview graph
    # -------------------------
    if not torchview_requested:
        audit_items.append(
            AuditItem(
                name="torchview graph",
                status="skipped",
                message="torchview export was not requested; no graph was expected",
            )
        )

    elif torchview_graph_path is None:
        audit_items.append(
            AuditItem(
                name="torchview graph",
                status="warning",
                message="torchview export was requested but no graph path was produced; export likely failed or was skipped",
            )
        )

    else:
        audit_existing_file(
            audit_items=audit_items,
            name="torchview graph",
            path=torchview_graph_path,
        )

    # -------------------------
    # Final audit summary
    # -------------------------
    log_audit_summary(
        stage="training",
        audit_items=audit_items,
    )

    return audit_items