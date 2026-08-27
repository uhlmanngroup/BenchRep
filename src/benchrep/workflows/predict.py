from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch

from benchrep.assembly.config import (
    compose_effective_config,
    SupportedPredictionConfigComponent,
)
from benchrep.assembly.builders import (
    build_dataset,
    build_datamodule,
    build_transform_pipeline,
    build_model,
    build_trainer,
    build_runtime_component,
)
from benchrep.interfaces.models import BenchRepAutoencoderModel, BenchRepVAEModel
from benchrep.interfaces.model_families import (
    SupportedModel,
    SupportedModelBaseClass,
    ModelFamilySpec,
    AUTOENCODER_FAMILY,
    VAE_FAMILY,
)
from benchrep.assembly.resolvers import resolve_prediction_config, PredictionRunSpec
from benchrep.assembly.resolvers.utils import (
    get_component_override_name,
    resolve_component_source,
)
from benchrep.assembly.schemas import PredictionConfig
from benchrep.records import (
    save_config_records,
    setup_run_logger,
    capture_console_streams,
    export_prediction_outputs,
    write_prediction_manifest,
    get_runtime_environment_filename,
    collect_prediction_environment_context,
    write_runtime_environment,
)
from benchrep.records.utils import now_isoformat
from benchrep.records.prediction_exports import PredictionExportPaths
from benchrep.runtime import RunContext
from benchrep.runtime.predict_run_validation import (
    validate_predict_contract_compatibility,
    prepare_predict_source_inputs,
    validate_prediction_outputs,
    infer_prediction_observation_count,
)
from benchrep.runtime.utils import (
    CompatibilityPolicy,
    format_external_datamodule_failure_message,
)
from benchrep.runtime.status import (
    PredictionOutcome,
    PredictionStatusReport,
    build_prediction_status_report,
    PredictionOutcomeStatus,
    log_outcome_summary,
)
from benchrep.assembly.registries.builtins import register_builtins


@dataclass
class PredictionWorkflowResult:
    config: PredictionConfig
    run_spec: PredictionRunSpec
    run_context: RunContext
    datamodule: L.LightningDataModule
    model: SupportedModel
    trainer: L.Trainer
    predictions: list[Any]
    export_paths: Any
    status_report: PredictionStatusReport
    manifest_path: Path


# Model-specific wrappers
def predict_ae(
        config_path: Path | str | None = None,
        full_config_object: PredictionConfig | None = None,
        config_components: Mapping[str, SupportedPredictionConfigComponent] | None = None,
        training_manifest_path: Path | str | None = None,
        model: (
                BenchRepAutoencoderModel
                | type[BenchRepAutoencoderModel]
                | None
        ) = None,
        datamodule: (
                L.LightningDataModule
                | type[L.LightningDataModule]
                | None
        ) = None,
        compatibility_policy: CompatibilityPolicy = "error",
) -> PredictionWorkflowResult:
    return _predict(
        model_family=AUTOENCODER_FAMILY,
        config_path=config_path,
        full_config_object=full_config_object,
        config_components=config_components,
        training_manifest_path=training_manifest_path,
        model=model,
        datamodule=datamodule,
        compatibility_policy=compatibility_policy,
    )


def predict_vae(
        config_path: Path | str | None = None,
        full_config_object: PredictionConfig | None = None,
        config_components: Mapping[str, SupportedPredictionConfigComponent] | None = None,
        training_manifest_path: Path | str | None = None,
        model: (
                BenchRepVAEModel
                | type[BenchRepVAEModel]
                | None
        ) = None,
        datamodule: (
                L.LightningDataModule
                | type[L.LightningDataModule]
                | None
        ) = None,
        compatibility_policy: CompatibilityPolicy = "error",
) -> PredictionWorkflowResult:
    return _predict(
        model_family=VAE_FAMILY,
        config_path=config_path,
        full_config_object=full_config_object,
        config_components=config_components,
        training_manifest_path=training_manifest_path,
        model=model,
        datamodule=datamodule,
        compatibility_policy=compatibility_policy,
    )


def _predict(
        model_family: ModelFamilySpec,
        config_path: Path | str | None,
        full_config_object: PredictionConfig | None = None,
        config_components: Mapping[str, SupportedPredictionConfigComponent] | None = None,
        training_manifest_path: Path | str | None = None,
        model: (
                SupportedModel
                | SupportedModelBaseClass
                | None
        ) = None,
        datamodule: (
                L.LightningDataModule
                | type[L.LightningDataModule]
                | None
        ) = None,
        compatibility_policy: CompatibilityPolicy = "error",
) -> PredictionWorkflowResult:
    register_builtins()

    model_source = resolve_component_source(
        model,
        expected_base_class=model_family.model_base_class,
        component_name="model",
    )
    datamodule_source = resolve_component_source(
        datamodule,
        expected_base_class=L.LightningDataModule,
        component_name="datamodule",
    )

    model_is_external = model_source != "config"
    datamodule_is_external = datamodule_source != "config"

    # Training manifest override
    if training_manifest_path is not None:
        training_manifest_path = Path(training_manifest_path).resolve()
        if training_manifest_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                "training_manifest_path override must point to a YAML file."
            )
        training_manifest_path_overridden = True
    else:
        training_manifest_path_overridden = False

    # Compose, parse, and resolve config
    config_composition_result = compose_effective_config(
        schema=PredictionConfig,
        config_path=config_path,
        full_config_object=full_config_object,
        config_components=config_components,
        external_model=model_is_external,
        external_datamodule=datamodule_is_external,
        training_manifest_path_overridden=training_manifest_path_overridden,
    )

    pred_config = config_composition_result.effective_config

    run_spec = resolve_prediction_config(
        prediction_config=pred_config,
        model_family=model_family,
        training_manifest_path_override=training_manifest_path,
        model_source=model_source,
        datamodule_source=datamodule_source,
        model_override_name=(
            get_component_override_name(model)
            if model is not None
            else None
        ),
        compatibility_policy=compatibility_policy,
    )

    resolved_prediction_config = run_spec.prediction_config

    run_context = RunContext.create(
        output_root=run_spec.run_identity.output_root,
        stage=run_spec.stage,
        project_name=run_spec.run_identity.project_name,
        model_name=run_spec.run_identity.model_name,
    )
    created_at = now_isoformat()

    # Initiate local run logger
    run_log = setup_run_logger(log_out_dir=run_context.log_dir)
    inference_warnings: list[str] = []

    # Log composition messages and warnings
    for msg in config_composition_result.composition_messages:
        run_log.info(msg)
    for warning in config_composition_result.composition_warnings:
        run_log.warning(warning)

    run_log.info("Prediction run initialized.")
    run_log.info("Resolved training manifest: '%s'", run_spec.training_manifest_path)
    run_log.info("Prediction effective config source: '%s'", config_composition_result.effective_source)
    run_log.info("Prediction outputs will be saved to: '%s'", run_context.output_dir)
    run_log.info(
        "Resolved checkpoint: source=%s, selection=%s, path='%s'",
        run_spec.checkpoint_source,
        run_spec.checkpoint_selection,
        run_spec.checkpoint_path,
    )
    run_log.info(
        "Resolved prediction exports: mode=%s, embeddings_enabled=%s, "
        "embedding_keys=%s, primary_key=%s, reconstructions_enabled=%s, "
        "n_examples=%s, selection=%s, reconstruction_seed=%s",
        run_spec.export_spec.mode,
        run_spec.export_spec.embeddings.enabled,
        run_spec.export_spec.embeddings.keys,
        run_spec.export_spec.embeddings.primary_key,
        run_spec.export_spec.reconstructions.enabled,
        run_spec.export_spec.reconstructions.n_examples,
        run_spec.export_spec.reconstructions.selection,
        run_spec.export_spec.reconstructions.seed,
    )

    if model_is_external and run_spec.model_family == VAE_FAMILY:
        warning = (
            "External VAE controls its own prediction logic. BenchRep cannot "
            "determine whether reconstructions are generated from the posterior "
            "mean, a sampled latent, or another model-specific path. Verify the "
            "external model's `predict_step()` implementation if deterministic "
            "reconstruction is required."
        )
        inference_warnings.append(warning)
        run_log.warning(warning)

    elif run_spec.model_family == VAE_FAMILY:
        configured_source = (
            resolved_prediction_config.inference.reconstruction_latent_source
        )
        effective_source = run_spec.reconstruction_latent_source
        assert effective_source is not None

        if configured_source is None:
            run_log.info(
                "`inference.reconstruction_latent_source` was not configured; "
                "defaulting to the posterior mean (`z_mu`)."
            )
        elif effective_source == "mean":
            run_log.info(
                "VAE reconstructions will be decoded from the posterior mean "
                "(`z_mu`)."
            )
        else:
            run_log.info(
                "VAE reconstructions will be decoded from the sampled latent "
                "(`z_sample`)."
            )
    if run_spec.transform_source == "default_identity":
        warning = (
            "Training used an external datamodule, so validation-targeted "
            "transforms could not be inherited. No prediction transforms were "
            "configured; using an identity transform pipeline."
        )
        inference_warnings.append(warning)
        run_log.warning(warning)

    elif run_spec.transform_source != "external_datamodule":
        assert run_spec.transform_configs is not None

        run_log.info(
            "Resolved prediction transforms: source=%s, transforms=%s",
            run_spec.transform_source,
            [transform.name for transform in run_spec.transform_configs],
        )

    # Bookkeeping --- config
    save_config_records(
        original_config_path=config_composition_result.original_config_path,
        resolved_config=resolved_prediction_config,
        config_out_dir=run_context.config_dir,
    )

    # Enforce reproducibility
    L.seed_everything(
        run_spec.seed,
        workers=run_spec.seed_workers
    )
    run_log.info("Global seed set to %s", run_spec.seed)

    torch.set_float32_matmul_precision(
        run_spec.float32_matmul_precision
    )
    run_log.info(
        "float32 matmul precision set to '%s'",
        run_spec.float32_matmul_precision,
    )

    # Build dataset and datamodule
    datamodule = build_runtime_component(
        datamodule,
        override_config=(
            resolved_prediction_config.overrides.datamodule
        ),
        component_name="datamodule",
    )

    if not datamodule_is_external:
        dataset_config = run_spec.dataset_config
        datamodule_config = run_spec.datamodule_config

        assert dataset_config is not None
        assert datamodule_config is not None
        assert run_spec.transform_configs is not None

        prediction_pipeline = build_transform_pipeline(
            run_spec.transform_configs,
        )

        dataset = build_dataset(
            dataset_config=dataset_config,
        )

        datamodule = build_datamodule(
            dataset=dataset,
            datamodule_config=datamodule_config,
            seed=run_spec.seed,
            stage=run_spec.stage,
            prediction_pipeline=prediction_pipeline,
        )
    else:
        run_log.info(
            "External datamodule was provided; dataset config, resolved "
            "datamodule settings, and all config-driven transforms will be ignored."
        )

    # Build or use model
    model = build_runtime_component(
        model,
        override_config=resolved_prediction_config.overrides.model,
        component_name="model",
    )

    if not model_is_external:
        assert run_spec.training_config.model is not None
        assert run_spec.training_config.encoder is not None
        assert run_spec.training_config.losses is not None
        assert run_spec.training_config.optimizer is not None

        model = build_model(
            config=run_spec.training_config,
            prediction_reconstruction_latent_source=run_spec.reconstruction_latent_source,
        )
    else:
        run_log.info(
            "External model was provided; resolved model/encoder/decoder/losses/optimizer "
            "config sections will be ignored."
        )

    # Preflight check and source input validation
    assert model is not None
    assert datamodule is not None

    precondition_result = validate_predict_contract_compatibility(
        run_spec=run_spec,
        model=model,
    )

    inference_warnings.extend(precondition_result.warnings)
    inference_warning_issues = tuple(inference_warnings)

    predict_inputs = prepare_predict_source_inputs(run_spec=run_spec)
    model.load_state_dict(predict_inputs.state_dict)
    model.eval()

    run_log.info("Loaded checkpoint weights into prediction model.")

    # Build trainer and predict
    trainer, _, _ = build_trainer(
        trainer_config=run_spec.trainer_config,
        stage=run_spec.stage,
        run_context=run_context,
        max_batches=run_spec.max_batches,
    )

    prediction_environment_context = (
        collect_prediction_environment_context(
            run_spec=run_spec,
            trainer=trainer,
        )
    )

    runtime_environment_path = write_runtime_environment(
        output_path=(
            run_context.metadata_dir
            / get_runtime_environment_filename(run_spec.stage)
        ),
        stage=run_spec.stage,
        run_name=run_context.run_name,
        workflow_context=prediction_environment_context,
    )

    run_log.info(
        "Exported runtime environment to: '%s'",
        runtime_environment_path,
    )

    run_log.info("Starting prediction...")

    try:
        with capture_console_streams(log_out_dir=run_context.log_dir, capture_stdout=False):
            raw_predictions = trainer.predict(
                model,
                datamodule=datamodule,
                return_predictions=True,
            )

    except Exception as exc:
        if precondition_result.should_wrap_batch_contract_errors:
            run_log.error(
                "Prediction failed while using an external datamodule with an internal model.",
                exc_info=True,
            )

            raise RuntimeError(
                format_external_datamodule_failure_message(
                    stage="prediction",
                    precondition_result=precondition_result,
                    original_error=exc,
                )
            ) from exc

        raise

    predictions: list[Any] = (
        [] if raw_predictions is None else raw_predictions
    )
    n_prediction_batches = len(predictions)
    run_log.info("Finished prediction")
    run_log.info("Prediction returned %s batches.", n_prediction_batches)

    try:
        validate_prediction_outputs(
            predictions=predictions,
            model_family=run_spec.model_family,
        )

    except Exception as exc:
        n_predicted_observations = None

        error_issue = f"Error ({type(exc).__name__}): {exc}"

        run_log.error(
            "Prediction output validation failed: %s",
            error_issue,
            exc_info=True,
        )

        inference_outcome = PredictionOutcome(
            name="inference",
            status="failed",
            issues=(
                *inference_warning_issues,
                error_issue,
            ),
        )

        export_paths = PredictionExportPaths()
        skipped_issue = (
            "Skipped because prediction output validation failed."
        )

        if run_spec.export_spec.embeddings.enabled:
            embedding_outcome = PredictionOutcome(
                name="embeddings",
                status="skipped",
                issues=(skipped_issue,),
            )
        else:
            embedding_outcome = PredictionOutcome(
                name="embeddings",
                status="disabled",
            )

        if run_spec.export_spec.reconstructions.enabled:
            reconstruction_outcome = PredictionOutcome(
                name="reconstructions",
                status="skipped",
                issues=(skipped_issue,),
            )
        else:
            reconstruction_outcome = PredictionOutcome(
                name="reconstructions",
                status="disabled",
            )

    else:
        inference_status: PredictionOutcomeStatus = (
            "completed_with_warnings"
            if inference_warning_issues
            else "completed"
        )

        inference_outcome = PredictionOutcome(
            name="inference",
            status=inference_status,
            issues=inference_warning_issues,
        )

        n_predicted_observations = (
            infer_prediction_observation_count(
                predictions=predictions,
            )
        )

        first_prediction = predictions[0]

        run_log.info(
            "First prediction batch type: %s",
            type(first_prediction).__name__,
        )

        run_log.info("Exporting prediction outputs...")

        export_result = export_prediction_outputs(
            predictions=predictions,
            export_spec=run_spec.export_spec,
            embedding_dir=run_context.prediction_embeddings_dir,
            reconstruction_dir=run_context.prediction_reconstructions_dir,
        )

        export_paths = export_result.paths
        embedding_outcome, reconstruction_outcome = (
            export_result.outcomes
        )

        if export_paths.embedding_export is not None:
            run_log.info(
                "Exported embedding artifact to: '%s'",
                export_paths.embedding_export.embeddings_h5ad_path,
            )

        if export_paths.reconstruction_paths is not None:
            run_log.info(
                "Exported reconstruction artifacts: input=%s, "
                "reconstruction=%s, obs=%s, metadata=%s, "
                "n_examples_exported=%s",
                export_paths.reconstruction_paths.input_path,
                export_paths.reconstruction_paths.reconstruction_path,
                export_paths.reconstruction_paths.obs_path,
                export_paths.reconstruction_paths.metadata_path,
                export_paths.reconstruction_paths.n_examples_exported,
            )

        run_log.info("Finished exporting prediction outputs")

    status_report = build_prediction_status_report(
        inference=inference_outcome,
        embeddings_export=embedding_outcome,
        reconstructions_export=reconstruction_outcome,
    )

    completed_at = now_isoformat()

    # Export prediction manifest
    manifest_path = run_context.metadata_dir / "prediction_manifest.yaml"
    prediction_manifest = write_prediction_manifest(
        config_composition_result=config_composition_result,
        output_path=manifest_path,
        run_spec=run_spec,
        run_context=run_context,
        export_paths=export_paths,
        created_at=created_at,
        completed_at=completed_at,
        status_report=status_report,
        model_class_name=type(model).__name__,
        datamodule_class_name=type(datamodule).__name__,
        n_batches=n_prediction_batches,
        n_observations=n_predicted_observations,
    )

    run_log.info("Exported prediction manifest to: '%s'", manifest_path)

    log_outcome_summary(
        run_log=run_log,
        workflow_name="Prediction",
        workflow_status=status_report.status,
        summary=prediction_manifest["outcome_summary"],
    )

    if status_report.status in {"partially_completed", "failed"}:
        failed_outcomes = [
            outcome
            for outcome in (
                status_report.inference,
                status_report.embeddings_export,
                status_report.reconstructions_export,
            )
            if outcome.status == "failed"
        ]

        failure_summary = "; ".join(
            f"{outcome.name}: {'; '.join(outcome.issues)}"
            for outcome in failed_outcomes
        )

        raise RuntimeError(
            "Prediction was finalized with status "
            f"{status_report.status!r}: {failure_summary}. "
            f"The manifest was written to '{manifest_path}'."
        )

    return PredictionWorkflowResult(
        config=resolved_prediction_config,
        run_spec=run_spec,
        run_context=run_context,
        datamodule=datamodule,
        model=model,
        trainer=trainer,
        predictions=predictions,
        export_paths=export_paths,
        status_report=status_report,
        manifest_path=manifest_path,
    )