from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch

from benchrep.assembly.config import (
    compose_effective_config,
    SupportedConfigComponent,
)
from benchrep.assembly.builders import (
    build_dataset,
    build_datamodule,
    build_transform_pipelines,
    build_model,
    build_trainer,
)
from benchrep.interfaces.models import BenchRepAutoencoderModel, BenchRepVAEModel
from benchrep.interfaces.model_families import (
    SupportedModel,
    ModelFamilySpec,
    AUTOENCODER_FAMILY,
    VAE_FAMILY,
)
from benchrep.interfaces.compatibility import validate_prediction_output_structure
from benchrep.assembly.resolvers import resolve_prediction_config, PredictionRunSpec
from benchrep.assembly.schemas import PredictionConfig
from benchrep.records import (
    save_config_records,
    setup_run_logger,
    capture_console_streams,
    export_prediction_outputs,
    write_prediction_manifest,
    write_audit_report,
    get_runtime_environment_filename,
    collect_prediction_environment_context,
    write_runtime_environment,
)
from benchrep.records.utils import now_isoformat
from benchrep.runtime import RunContext
from benchrep.runtime.predict_run_validation import (
    validate_predict_contract_compatibility,
    prepare_predict_source_inputs,
    audit_predict_outputs,
)
from benchrep.runtime.utils import (
    CompatibilityPolicy,
    format_external_datamodule_failure_message,
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
    manifest_path: Path
    audit_report_path: Path


# Model-specific wrappers
def predict_ae(
        config_path: Path | str | None = None,
        full_config_object: PredictionConfig | None = None,
        config_components: Mapping[str, SupportedConfigComponent] | None = None,
        training_manifest_path: Path | str | None = None,
        model: BenchRepAutoencoderModel | None = None,
        datamodule: L.LightningDataModule | None = None,
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
        config_components: Mapping[str, SupportedConfigComponent] | None = None,
        training_manifest_path: Path | str | None = None,
        model: BenchRepVAEModel | None = None,
        datamodule: L.LightningDataModule | None = None,
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
        config_path: Path | str,
        full_config_object: PredictionConfig | None = None,
        config_components: Mapping[str, SupportedConfigComponent] | None = None,
        training_manifest_path: Path | str | None = None,
        model: SupportedModel | None = None,
        datamodule: L.LightningDataModule | None = None,
        compatibility_policy: CompatibilityPolicy = "error",
) -> PredictionWorkflowResult:
    register_builtins()

    if compatibility_policy not in {"error", "warn"}:
        raise ValueError(
            "compatibility_policy must be 'error' or 'warn'."
        )

    model_is_external = model is not None
    model_source: Literal["config", "external_object"] = (
        "external_object" if model_is_external else "config"
    )
    datamodule_is_external = datamodule is not None
    datamodule_source: Literal["config", "external_object"] = (
        "external_object" if datamodule_is_external else "config"
    )

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
        model_overridden=model_is_external,
        datamodule_overridden=datamodule_is_external,
    )

    # Setup paths
    if not model_is_external:
        assert run_spec.training_config.model is not None
        model_name = f"{run_spec.training_config.model.name}"
    else:
        model_name = f"{model_family.name}_external_{type(model).__name__}"

    run_context = RunContext.create(
        output_root=run_spec.training_config.run.output_root,
        stage=run_spec.stage,
        project_name=run_spec.training_config.run.project_name,
        model_name=model_name,
    )
    created_at = now_isoformat()

    # Initiate local run logger
    run_log = setup_run_logger(log_out_dir=run_context.log_dir)

    # Log composition messages and warnings
    for msg in config_composition_result.composition_messages:
        run_log.info(msg)
    for warning in config_composition_result.composition_warnings:
        run_log.warning(warning)

    run_log.info("Prediction run initialized.")
    run_log.info("Resolved training manifest: '%s'", run_spec.training_manifest_path)
    run_log.info("Prediction effective config source: '%s'", config_composition_result.effective_source)
    run_log.info("Prediction outputs will be saved to: '%s'", run_context.output_dir)
    run_log.info("Resolved checkpoint: '%s'", run_spec.checkpoint_path)
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
    if model_is_external and model_family == VAE_FAMILY:
        run_log.warning(
            "External VAE controls its own prediction logic. BenchRep cannot "
            "determine whether reconstructions are generated from the posterior "
            "mean, a sampled latent, or another model-specific path. Verify the "
            "external model's `predict_step()` implementation if deterministic "
            "reconstruction is required."
        )

    elif model_family == VAE_FAMILY:
        configured_source = (
            run_spec.prediction_config.inference.reconstruction_latent_source
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
        run_log.warning(
            "Training used an external datamodule, so preprocessing transforms "
            "could not be inherited. No prediction transforms were configured; "
            "using an identity transform pipeline."
        )

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
        resolved_config=run_spec.prediction_config,
        config_out_dir=run_context.config_dir,
    )

    # Enforce reproducibility
    L.seed_everything(
        run_spec.seed,
        workers=run_spec.seed_workers
    )
    run_log.info("Global seed set to %s", run_spec.seed)

    if run_spec.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(
            run_spec.float32_matmul_precision
        )
        run_log.info(
            "float32 matmul precision set to '%s'",
            run_spec.float32_matmul_precision,
        )

    # Build dataset and datamodule
    if not datamodule_is_external:
        dataset_config = run_spec.dataset_config
        datamodule_config = run_spec.datamodule_config

        assert dataset_config is not None
        assert datamodule_config is not None

        transform_pipelines = build_transform_pipelines(
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
            preprocessing_pipeline=transform_pipelines.preprocessing,
        )
    else:
        run_log.info(
            "External datamodule was provided; dataset config, resolved "
            "datamodule settings, and all config-driven transforms will be ignored."
        )

    # Build or use model
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
        model_family=model_family,
        model=model,
        model_is_external=model_is_external,
        datamodule_is_external=datamodule_is_external,
        compatibility_policy=compatibility_policy,
    )

    predict_inputs = prepare_predict_source_inputs(run_spec=run_spec)
    model.load_state_dict(predict_inputs.state_dict)
    model.eval()

    run_log.info("Loaded checkpoint weights into prediction model.")

    # Build trainer and predict
    trainer, _ = build_trainer(
        trainer_config=run_spec.trainer_config,
        stage=run_spec.stage,
        run_context=run_context,
        max_batches=run_spec.max_batches,
    )

    prediction_environment_context = (
        collect_prediction_environment_context(
            run_spec=run_spec,
            trainer=trainer,
            model_family=model_family,
            model_source=model_source,
            datamodule_source=datamodule_source,
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
            predictions = trainer.predict(
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

    if not predictions:
        raise RuntimeError("Prediction returned no batches.")

    # Sanity test predictions
    sanity_test_batch_idx = 0
    sanity_test_prediction = predictions[sanity_test_batch_idx]

    validate_prediction_output_structure(
        prediction=sanity_test_prediction,
        model_family=model_family,
        batch_idx=sanity_test_batch_idx,
        check_value_types=True,
    )

    run_log.info("Finished prediction")
    run_log.info("Prediction returned %s batches.", len(predictions))
    run_log.info(
        "First prediction batch type: %s",
        type(sanity_test_prediction).__name__,
    )

    run_log.info("Exporting prediction outputs...")

    export_paths = export_prediction_outputs(
        model_family=model_family,
        predictions=predictions,
        export_spec=run_spec.export_spec,
        embedding_dir=run_context.prediction_embeddings_dir,
        reconstruction_dir=run_context.prediction_reconstructions_dir,
    )

    if export_paths.embedding_export is not None:
        run_log.info(
            "Exported embedding artifact to: '%s'",
            export_paths.embedding_export.embeddings_h5ad_path,
        )

    if export_paths.reconstruction_paths is not None:
        run_log.info(
            "Exported reconstruction artifacts: input=%s, reconstruction=%s, obs=%s, "
            "metadata=%s, n_examples_exported=%s",
            export_paths.reconstruction_paths.input_path,
            export_paths.reconstruction_paths.reconstruction_path,
            export_paths.reconstruction_paths.obs_path,
            export_paths.reconstruction_paths.metadata_path,
            export_paths.reconstruction_paths.n_examples_exported,
        )

    run_log.info("Finished exporting prediction outputs")

    completed_at = now_isoformat()

    # Export prediction manifest
    manifest_path = run_context.metadata_dir / "prediction_manifest.yaml"
    write_prediction_manifest(
        config_composition_result=config_composition_result,
        output_path=manifest_path,
        run_spec=run_spec,
        model_family=model_family,
        run_context=run_context,
        export_paths=export_paths,
        created_at=created_at,
        completed_at=completed_at,
        status="completed",
        model_source=model_source,
        model_class_name=type(model).__name__,
        datamodule_source=datamodule_source,
        datamodule_class_name=type(datamodule).__name__,
    )

    run_log.info("Exported prediction manifest to: '%s'", manifest_path)

    audit_items = audit_predict_outputs(
        run_context=run_context,
        run_spec=run_spec,
        model_family=model_family,
        predictions=predictions,
        export_paths=export_paths,
        config_composition_result=config_composition_result,
        resolved_config_path=run_context.config_dir / "resolved_config.yaml",
        prediction_manifest_path=manifest_path,
        model_source=model_source,
        model_class_name=type(model).__name__,
        datamodule_source=datamodule_source,
        datamodule_class_name=type(datamodule).__name__,
        runtime_environment_path=runtime_environment_path,
    )

    audit_report_path = write_audit_report(
        stage="prediction",
        audit_items=audit_items,
        output_path=(
                run_context.metadata_dir / "prediction_audit_report.yaml"
        ),
        audited_at=now_isoformat(),
    )

    run_log.info(
        "Exported prediction audit report to: '%s'",
        audit_report_path,
    )

    return PredictionWorkflowResult(
        config=run_spec.prediction_config,
        run_spec=run_spec,
        run_context=run_context,
        datamodule=datamodule,
        model=model,
        trainer=trainer,
        predictions=predictions,
        export_paths=export_paths,
        manifest_path=manifest_path,
        audit_report_path=audit_report_path,
    )