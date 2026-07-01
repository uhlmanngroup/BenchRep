from __future__ import annotations

import lightning as L

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints, Literal
from pathlib import Path

from benchrep.records import get_run_logger
from benchrep.runtime.utils import CompatibilityPolicy
from benchrep.runtime.run_context import RunContext
from benchrep.interfaces.contracts import ContractKind
from benchrep.interfaces.model_families import ModelFamilySpec
from benchrep.interfaces.compatibility import (
    validate_external_model,
    sanity_check_training_step_batch_annotation,
    sanity_check_predict_step_batch_annotation,
    sanity_check_predict_step_return_annotation,
)
from benchrep.assembly.config import load_yaml


AuditStatus = Literal["ok", "warning", "error", "skipped"]


@dataclass(frozen=True, slots=True)
class TrainPreconditionResult:
    should_wrap_training_errors_with_batch_hint: bool = False
    expected_batch_type: type[Any] | None = None
    expected_batch_contract_kind: ContractKind | None = None
    model_family_name: str | None = None


@dataclass(frozen=True, slots=True)
class AuditItem:
    name: str
    status: AuditStatus
    message: str


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


def audit_train_outputs(
    *,
    run_context: RunContext,
    input_config_path: Path | str | None = None,
    resolved_config_path: Path | str,
    checkpoint_dir: Path | str,
    training_manifest_path: Path | str,
    torchview_requested: bool,
    torchview_graph_path: Path | str | None,
) -> None:
    run_log = get_run_logger()

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
                    "no input config path was provided; this is expected for runs "
                    "started from config objects rather than a YAML file"
                ),
            )
        )
    else:
        _audit_existing_file(
            audit_items=audit_items,
            name="input config",
            path=input_config_path,
        )

    _audit_existing_file(
        audit_items=audit_items,
        name="resolved config",
        path=resolved_config_path,
    )

    # -------------------------
    # Training manifest
    # -------------------------
    manifest_path_is_valid = _audit_existing_file(
        audit_items=audit_items,
        name="training manifest",
        path=training_manifest_path,
        require_yaml_suffix=True,
    )

    if manifest_path_is_valid:
        try:
            training_manifest = load_yaml(training_manifest_path)

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

            _audit_existing_dir(
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
    # Checkpoints
    # -------------------------
    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir_is_valid = _audit_existing_dir(
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

                _audit_existing_dir(
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
        _audit_existing_file(
            audit_items=audit_items,
            name="torchview graph",
            path=torchview_graph_path,
        )

    # -------------------------
    # Final audit summary
    # -------------------------
    run_log.info("")
    run_log.info("=" * 51)
    run_log.info("Training output audit summary")
    run_log.info("=" * 51)


    n_errors = sum(item.status == "error" for item in audit_items)
    n_warnings = sum(item.status == "warning" for item in audit_items)

    run_log.info(
        "Training output audit summary: %s error(s), %s warning(s).",
        n_errors,
        n_warnings,
    )

    run_log.info("")

    for item in audit_items:
        message = "Training output audit: %s: %s" % (item.name, item.message)

        if item.status == "ok":
            run_log.info(message)
        elif item.status == "warning":
            run_log.warning(message)
        elif item.status == "error":
            run_log.error(message)
        elif item.status == "skipped":
            run_log.info(message)


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


def _audit_existing_file(
    *,
    audit_items: list[AuditItem],
    name: str,
    path: Path | str,
    require_yaml_suffix: bool = False,
) -> bool:
    path = Path(path)

    if not path.exists():
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=f"not found at '{path}'",
            )
        )
        return False

    if not path.is_file():
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=f"path exists but is not a file: '{path}'",
            )
        )
        return False

    if require_yaml_suffix and path.suffix.lower() not in {".yaml", ".yml"}:
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=f"path does not have a YAML suffix: '{path}'",
            )
        )
        return False

    audit_items.append(
        AuditItem(
            name=name,
            status="ok",
            message=f"found at '{path}'",
        )
    )
    return True


def _audit_existing_dir(
    *,
    audit_items: list[AuditItem],
    name: str,
    path: Path | str,
) -> bool:
    path = Path(path)

    if not path.exists():
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=f"not found at '{path}'",
            )
        )
        return False

    if not path.is_dir():
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message=f"path exists but is not a directory: '{path}'",
            )
        )
        return False

    audit_items.append(
        AuditItem(
            name=name,
            status="ok",
            message=f"found at '{path}'",
        )
    )
    return True