from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from pathlib import Path

import lightning as L

import torch

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
    sanity_check_predict_step_batch_annotation,
    sanity_check_predict_step_return_annotation,
    validate_prediction_output_structure,
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
    external_model_only = model_is_external and not datamodule_is_external
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


def prepare_predict_source_inputs(
    run_spec: PredictionRunSpec,
) -> PredictSourceInputsResult:
    manifest_status = run_spec.training_manifest.get("status")

    if manifest_status != "completed":
        raise ValueError(
            "Prediction requires a completed training manifest, "
            f"but manifest status is {manifest_status!r}."
        )

    manifest_stage = run_spec.training_manifest.get("stage")
    if manifest_stage != "training":
        raise ValueError(
            "Prediction requires a training manifest, "
            f"but manifest stage is {manifest_stage!r}."
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


def audit_predict_outputs(
    *,
    run_context: RunContext,
    run_spec: PredictionRunSpec,
    model_family: ModelFamilySpec,
    predictions: list[Any],
    export_paths: Any,
    input_config_path: Path | str | None = None,
    resolved_config_path: Path | str,
    prediction_manifest_path: Path | str,
    model_source: Literal["config", "external_object"],
    model_class_name: str,
    datamodule_source: Literal["config", "external_object"],
    datamodule_class_name: str,
) -> list[AuditItem]:
    audit_items: list[AuditItem] = []

    prediction_manifest_path = Path(prediction_manifest_path)
    prediction_manifest: dict[str, Any] | None = None

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
    # Prediction manifest
    # -------------------------
    manifest_path_is_valid = audit_existing_file(
        audit_items=audit_items,
        name="prediction manifest",
        path=prediction_manifest_path,
        expected_suffixes={".yaml", ".yml"},
    )

    if manifest_path_is_valid:
        try:
            loaded_manifest = load_yaml(prediction_manifest_path)

            if not isinstance(loaded_manifest, dict):
                audit_items.append(
                    AuditItem(
                        name="prediction manifest load",
                        status="error",
                        message=(
                            f"loaded YAML object from '{prediction_manifest_path}', "
                            f"but expected a mapping and got "
                            f"{type(loaded_manifest).__name__}"
                        ),
                    )
                )
            else:
                prediction_manifest = loaded_manifest
                audit_items.append(
                    AuditItem(
                        name="prediction manifest load",
                        status="ok",
                        message=f"loaded YAML mapping from '{prediction_manifest_path}'",
                    )
                )

        except Exception as exc:
            audit_items.append(
                AuditItem(
                    name="prediction manifest load",
                    status="error",
                    message=(
                        f"could not be loaded as YAML: '{prediction_manifest_path}'. "
                        f"Original error ({type(exc).__name__}): {exc}"
                    ),
                )
            )

    # -------------------------
    # Manifest status
    # -------------------------
    if prediction_manifest is None:
        audit_items.append(
            AuditItem(
                name="manifest status",
                status="skipped",
                message="could not be checked because the prediction manifest is unavailable",
            )
        )

    else:
        status = prediction_manifest.get("status")

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
    if prediction_manifest is None:
        audit_items.append(
            AuditItem(
                name="manifest run output dir",
                status="skipped",
                message="could not be checked because the prediction manifest is unavailable",
            )
        )

    else:
        run_section = prediction_manifest.get("run")

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
            output_dir_value = run_section.get("output_dir")

            if not isinstance(output_dir_value, str):
                audit_items.append(
                    AuditItem(
                        name="manifest run output dir",
                        status="error",
                        message=(
                            "`run.output_dir` must be a string path, "
                            f"got {type(output_dir_value).__name__}"
                        ),
                    )
                )
            else:
                manifest_output_dir = Path(output_dir_value)

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
    # Prediction source provenance
    # -------------------------
    audit_existing_file(
        audit_items=audit_items,
        name="source training manifest",
        path=run_spec.training_manifest_path,
        expected_suffixes={".yaml", ".yml"},
    )

    audit_existing_file(
        audit_items=audit_items,
        name="source resolved training config",
        path=run_spec.resolved_training_config_path,
        expected_suffixes={".yaml", ".yml"},
    )

    audit_existing_file(
        audit_items=audit_items,
        name="source checkpoint",
        path=run_spec.checkpoint_path,
        expected_suffixes={".ckpt"},
    )

    audit_items.append(
        AuditItem(
            name="model provenance",
            status="ok" if model_source == "config" else "warning",
            message=(
                f"model_source={model_source!r}, model_class={model_class_name!r}"
                if model_source == "config"
                else (
                    f"model_source={model_source!r}, model_class={model_class_name!r}; "
                    "prediction depends on an externally supplied Python model object "
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
                    "prediction depends on an externally supplied Python datamodule "
                    "object and is not fully reconstructable from config alone"
                )
            ),
        )
    )

    # -------------------------
    # Prediction runtime summary
    # -------------------------
    if not predictions:
        audit_items.append(
            AuditItem(
                name="prediction batches",
                status="error",
                message="prediction returned no batches",
            )
        )

    else:
        audit_items.append(
            AuditItem(
                name="prediction batches",
                status="ok",
                message=f"prediction returned {len(predictions)} batch(es)",
            )
        )

        first_prediction_type = type(predictions[0])

        audit_items.append(
            AuditItem(
                name="first prediction batch type",
                status="ok",
                message=f"first prediction batch type is `{first_prediction_type.__name__}`",
            )
        )

        try:
            for batch_idx, prediction in enumerate(predictions):
                validate_prediction_output_structure(
                    prediction=prediction,
                    model_family=model_family,
                    batch_idx=batch_idx,
                    check_value_types=True,
                )

        except Exception as exc:
            audit_items.append(
                AuditItem(
                    name="prediction output contract",
                    status="error",
                    message=str(exc),
                )
            )

        else:
            expected_prediction_type = model_family.expected_prediction_output_type

            audit_items.append(
                AuditItem(
                    name="prediction output contract",
                    status="ok",
                    message=(
                        f"all prediction batches are structurally compatible with "
                        f"`{expected_prediction_type.__name__}` for model family "
                        f"`{model_family.name}`"
                    ),
                )
            )

    # -------------------------
    # Embedding export
    # -------------------------
    embedding_requested = run_spec.export_spec.embeddings.enabled
    embedding_export = getattr(export_paths, "embedding_export", None)

    if not embedding_requested:
        if embedding_export is None:
            audit_items.append(
                AuditItem(
                    name="embedding export",
                    status="skipped",
                    message="embedding export was not requested; no embedding artifact was expected",
                )
            )
        else:
            audit_items.append(
                AuditItem(
                    name="embedding export",
                    status="warning",
                    message=(
                        "embedding export was disabled, but an embedding export record "
                        f"was produced: {embedding_export!r}"
                    ),
                )
            )

    elif embedding_export is None:
        audit_items.append(
            AuditItem(
                name="embedding export",
                status="error",
                message="embedding export was requested, but no embedding export record was produced",
            )
        )

    else:
        _audit_expected_artifact_file(
            audit_items=audit_items,
            name="embedding export",
            path=getattr(embedding_export, "embeddings_h5ad_path", None),
            expected_suffixes={".h5ad"},
        )

        resolved_keys = getattr(embedding_export, "resolved_keys", None)
        resolved_primary_key = getattr(embedding_export, "resolved_primary_key", None)

        if isinstance(resolved_keys, list) and resolved_keys:
            audit_items.append(
                AuditItem(
                    name="embedding export keys",
                    status="ok",
                    message=f"resolved embedding keys: {resolved_keys}",
                )
            )
        else:
            audit_items.append(
                AuditItem(
                    name="embedding export keys",
                    status="warning",
                    message=f"resolved embedding keys are unavailable or empty: {resolved_keys!r}",
                )
            )

        if isinstance(resolved_keys, list) and resolved_primary_key in resolved_keys:
            audit_items.append(
                AuditItem(
                    name="primary embedding key",
                    status="ok",
                    message=f"resolved primary embedding key: {resolved_primary_key!r}",
                )
            )
        else:
            audit_items.append(
                AuditItem(
                    name="primary embedding key",
                    status="warning",
                    message=(
                        f"resolved primary key {resolved_primary_key!r} is unavailable "
                        f"or not present in resolved keys {resolved_keys!r}"
                    ),
                )
            )

    # -------------------------
    # Reconstruction export
    # -------------------------
    reconstruction_requested = run_spec.export_spec.reconstructions.enabled
    reconstruction_paths = getattr(export_paths, "reconstruction_paths", None)

    if not reconstruction_requested:
        if reconstruction_paths is None:
            audit_items.append(
                AuditItem(
                    name="reconstruction export",
                    status="skipped",
                    message=(
                        "reconstruction export was not requested; no reconstruction "
                        "artifacts were expected"
                    ),
                )
            )
        else:
            audit_items.append(
                AuditItem(
                    name="reconstruction export",
                    status="warning",
                    message=(
                        "reconstruction export was disabled, but reconstruction paths "
                        "were produced"
                    ),
                )
            )

    elif reconstruction_paths is None:
        audit_items.append(
            AuditItem(
                name="reconstruction export",
                status="error",
                message=(
                    "reconstruction export was requested, but no reconstruction paths "
                    "were produced"
                ),
            )
        )

    else:
        audit_items.append(
            AuditItem(
                name="reconstruction export",
                status="ok",
                message="reconstruction export produced a reconstruction paths record",
            )
        )

        if run_spec.export_spec.reconstructions.include_input:
            _audit_expected_artifact_file(
                audit_items=audit_items,
                name="reconstruction input artifact",
                path=getattr(reconstruction_paths, "input_path", None),
                expected_suffixes={".pt"},
            )
        else:
            audit_items.append(
                AuditItem(
                    name="reconstruction input artifact",
                    status="skipped",
                    message="input tensor export was not requested",
                )
            )

        if run_spec.export_spec.reconstructions.include_prediction:
            _audit_expected_artifact_file(
                audit_items=audit_items,
                name="reconstruction prediction artifact",
                path=getattr(reconstruction_paths, "reconstruction_path", None),
                expected_suffixes={".pt"},
            )
        else:
            audit_items.append(
                AuditItem(
                    name="reconstruction prediction artifact",
                    status="skipped",
                    message="prediction tensor export was not requested",
                )
            )

        _audit_expected_artifact_file(
            audit_items=audit_items,
            name="reconstruction obs artifact",
            path=getattr(reconstruction_paths, "obs_path", None),
            expected_suffixes={".pt"},
        )

        _audit_expected_artifact_file(
            audit_items=audit_items,
            name="reconstruction metadata artifact",
            path=getattr(reconstruction_paths, "metadata_path", None),
            expected_suffixes={".pt"},
        )

        stratify_by = run_spec.export_spec.reconstructions.stratify_by
        audit_items.append(
            AuditItem(
                name="reconstruction stratification",
                status="ok" if stratify_by is not None else "skipped",
                message=(
                    f"reconstruction examples were stratified by {stratify_by!r}"
                    if stratify_by is not None
                    else "stratified reconstruction sampling was not requested"
                ),
            )
        )

        n_examples_exported = getattr(
            reconstruction_paths,
            "n_examples_exported",
            None,
        )

        if isinstance(n_examples_exported, int) and n_examples_exported > 0:
            audit_items.append(
                AuditItem(
                    name="reconstruction examples",
                    status="ok",
                    message=f"exported {n_examples_exported} reconstruction example(s)",
                )
            )

        elif n_examples_exported == 0:
            audit_items.append(
                AuditItem(
                    name="reconstruction examples",
                    status="warning",
                    message="reconstruction export reported 0 exported examples",
                )
            )

        else:
            audit_items.append(
                AuditItem(
                    name="reconstruction examples",
                    status="warning",
                    message=(
                        "`n_examples_exported` was not available or was not an integer "
                        f"on reconstruction paths object: {n_examples_exported!r}"
                    ),
                )
            )

    # -------------------------
    # Final audit summary
    # -------------------------
    log_audit_summary(
        stage="prediction",
        audit_items=audit_items,
    )

    return audit_items


def _audit_expected_artifact_file(
    *,
    audit_items: list[AuditItem],
    name: str,
    path: Path | str | None,
    expected_suffixes: set[str],
) -> bool:
    if path is None:
        audit_items.append(
            AuditItem(
                name=name,
                status="error",
                message="expected artifact path was not produced",
            )
        )
        return False

    return audit_existing_file(
        audit_items=audit_items,
        name=name,
        path=path,
        expected_suffixes=expected_suffixes,
    )
