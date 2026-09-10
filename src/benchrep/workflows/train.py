from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.exceptions import SIGTERMException

import torch

from benchrep.runtime.run_context import RunContext
from benchrep.runtime.train_run_validation import (
    validate_train_contract_compatibility,
    validate_training_checkpoint_outputs,
)
from benchrep.assembly.resolvers import (
    TrainingRunSpec,
    resolve_training_config,
)
from benchrep.runtime.status import (
    build_early_stopping_record,
    TrainingInterruptionSignal,
    TrainingStatusReport,
    build_training_status_report,
    log_outcome_summary,
)

from benchrep.runtime.utils import (
    CompatibilityPolicy,
    format_external_datamodule_failure_message,
)
from benchrep.records import (
    save_config_records,
    capture_console_streams,
    setup_run_logger,
    write_training_manifest,
    export_torchview_graph,
    infer_dummy_input_size,
    get_runtime_environment_filename,
    collect_training_environment_context,
    write_runtime_environment,
)
from benchrep.records.utils import now_isoformat
from benchrep.interfaces.model_families import (
    SupportedModel,
    SupportedModelBaseClass,
    ModelFamilySpec,
    CanonicalModelFamilySpec,
    AUTOENCODER_FAMILY,
    VAE_FAMILY,
    COMPOSITE_FAMILY,
)
from benchrep.interfaces.models import (
    BenchRepAutoencoderModel,
    BenchRepVAEModel,
)
from benchrep.assembly.config import (
    compose_effective_config,
    SupportedTrainingConfigComponent,
)
from benchrep.assembly.schemas import TrainingConfig
from benchrep.assembly.builders import (
    build_datamodule,
    build_dataset,
    build_transform_pipelines,
    build_model,
    build_trainer,
    build_runtime_component,
)
from benchrep.assembly.registries.builtins import register_builtins
from benchrep.assembly.resolvers.utils import (
    get_component_override_name,
    resolve_component_source,
)


@dataclass
class TrainingWorkflowResult:
    config: TrainingConfig
    run_spec: TrainingRunSpec
    run_context: RunContext
    model: SupportedModel
    datamodule: L.LightningDataModule
    trainer: L.Trainer
    checkpoint_callback: ModelCheckpoint
    early_stopping_callback: EarlyStopping | None
    torchview_graph_path: Path | None
    status_report: TrainingStatusReport
    manifest_path: Path


# Model-specific wrappers
def train_ae(
        config_path: Path | str | None = None,
        full_config_object: TrainingConfig | None = None,
        config_components: Mapping[str, SupportedTrainingConfigComponent] | None = None,
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
) -> TrainingWorkflowResult:
    return _train(
        model_family=AUTOENCODER_FAMILY,
        config_path=config_path,
        full_config_object=full_config_object,
        config_components=config_components,
        model=model,
        datamodule=datamodule,
        compatibility_policy=compatibility_policy,
    )


def train_vae(
        config_path: Path | str | None = None,
        full_config_object: TrainingConfig | None = None,
        config_components: Mapping[str, SupportedTrainingConfigComponent] | None = None,
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
) -> TrainingWorkflowResult:
    return _train(
        model_family=VAE_FAMILY,
        config_path=config_path,
        full_config_object=full_config_object,
        config_components=config_components,
        model=model,
        datamodule=datamodule,
        compatibility_policy=compatibility_policy,
    )


def train_composite(
    config_path: Path | str | None = None,
    full_config_object: TrainingConfig | None = None,
    config_components: (
        Mapping[str, SupportedTrainingConfigComponent] | None
    ) = None,
    datamodule: (
        L.LightningDataModule
        | type[L.LightningDataModule]
        | None
    ) = None,
) -> TrainingWorkflowResult:
    return _train(
        model_family=COMPOSITE_FAMILY,
        config_path=config_path,
        full_config_object=full_config_object,
        config_components=config_components,
        datamodule=datamodule,
    )


def _train(
        model_family: ModelFamilySpec,
        config_path: Path | str | None = None,
        full_config_object: TrainingConfig | None = None,
        config_components: Mapping[str, SupportedTrainingConfigComponent] | None = None,
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
        compatibility_policy: CompatibilityPolicy = "error"
) -> TrainingWorkflowResult:
    register_builtins()

    if isinstance(model_family, CanonicalModelFamilySpec):
        model_source = resolve_component_source(
            model,
            expected_base_class=model_family.model_base_class,
            component_name="model",
        )
    else:
        if model is not None:
            raise TypeError(
                "Whole-model overrides are supported only for the canonical "
                "`autoencoder` and `vae` model families; they are not supported "
                "for `composite`."
            )

        model_source = "config"
    datamodule_source = resolve_component_source(
        datamodule,
        expected_base_class=L.LightningDataModule,
        component_name="datamodule",
    )

    model_is_external = model_source != "config"
    datamodule_is_external = datamodule_source != "config"

    # Compose and parse config
    config_composition_result = compose_effective_config(
        schema=TrainingConfig,
        config_path=config_path,
        full_config_object=full_config_object,
        config_components=config_components,
        external_model=model_is_external,
        external_datamodule=datamodule_is_external,
    )

    run_spec = resolve_training_config(
        training_config=config_composition_result.effective_config,
        model_family=model_family,
        model_source=model_source,
        datamodule_source=datamodule_source,
        model_override_name=(
            get_component_override_name(model)
            if model is not None
            else None
        ),
        compatibility_policy=compatibility_policy,
    )

    resolved_training_config = run_spec.training_config

    run_context = RunContext.create(
        output_root=run_spec.run_identity.output_root,
        stage=run_spec.stage,
        project_name=run_spec.run_identity.project_name,
        model_name=run_spec.run_identity.model_name,
    )

    created_at = now_isoformat()

    # Initiate local run logger
    run_log = setup_run_logger(log_out_dir=run_context.log_dir)

    # Log composition messages and warnings
    for msg in config_composition_result.composition_messages:
        run_log.info(msg)
    for warning in config_composition_result.composition_warnings:
        run_log.warning(warning)

    run_log.info("Training run initialized.")
    run_log.info("Training effective config source: '%s'", config_composition_result.effective_source)
    run_log.info("Training outputs will be saved to: '%s'", run_context.output_dir)

    # Bookkeeping --- config
    save_config_records(
        original_config_path=config_composition_result.original_config_path,
        resolved_config=resolved_training_config,
        config_out_dir=run_context.config_dir,
    )

    # Enforce reproducibility
    L.seed_everything(
        resolved_training_config.reproducibility.seed,
        workers=resolved_training_config.reproducibility.seed_workers,
    )
    run_log.info("Global seed set to %s", resolved_training_config.reproducibility.seed)

    if resolved_training_config.reproducibility.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(
            resolved_training_config.reproducibility.float32_matmul_precision
        )
        run_log.info(
            "float32 matmul precision set to '%s'",
            resolved_training_config.reproducibility.float32_matmul_precision,
        )

    datamodule = build_runtime_component(
        datamodule,
        override_config=(
            resolved_training_config.overrides.datamodule
        ),
        component_name="datamodule",
    )

    if not datamodule_is_external:
        dataset_config = resolved_training_config.dataset
        datamodule_config = resolved_training_config.datamodule

        assert dataset_config is not None
        assert datamodule_config is not None

        validation_transform_names = tuple(
            transform.name
            for transform in resolved_training_config.transforms
            if "validation" in transform.apply_to
        )

        if (
            datamodule_config.val_fraction == 0
            and validation_transform_names
        ):
            run_log.warning(
                "Validation-targeted transforms are configured, but "
                "`datamodule.val_fraction` is 0. No validation split will be "
                "created, so these transforms will not run during training. "
                "They remain available for inheritance by linked prediction: %s",
                validation_transform_names,
            )

        transform_pipelines = build_transform_pipelines(
            resolved_training_config.transforms,
        )

        dataset = build_dataset(
            dataset_config=dataset_config,
        )

        datamodule = build_datamodule(
            dataset=dataset,
            datamodule_config=datamodule_config,
            seed=resolved_training_config.reproducibility.seed,
            stage=run_spec.stage,
            training_pipeline=transform_pipelines.training,
            validation_pipeline=transform_pipelines.validation,
        )
    else:
        run_log.info(
            "External datamodule was provided; dataset, datamodule, and transforms "
            "config sections will be ignored regardless of whether they came from "
            "YAML, a full config object, or config_components."
        )

    model = build_runtime_component(
        model,
        override_config=resolved_training_config.overrides.model,
        component_name="model",
    )

    if not model_is_external:
        model = build_model(
            config=resolved_training_config,
            composite_model_spec=run_spec.composite_model_spec,
        )
    else:
        run_log.info(
            "External model was provided; model/encoder/decoder/losses/optimizer "
            "config sections will be ignored regardless of whether they came from "
            "YAML, a full config object, or config_components."
        )

    # Preflight check
    assert model is not None
    assert datamodule is not None

    precondition_result = validate_train_contract_compatibility(
        run_spec=run_spec,
        model=model,
    )

    trainer, checkpoint_callback, early_stopping_callback = build_trainer(
        trainer_config=resolved_training_config.trainer,
        stage=run_spec.stage,
        run_context=run_context,
        logger_config=resolved_training_config.logger,
        checkpoint_config=resolved_training_config.checkpointing,
        early_stopping_config=resolved_training_config.early_stopping,
        additional_callback_configs=resolved_training_config.additional_callbacks,
    )

    if checkpoint_callback is None:
        raise RuntimeError("Training trainer builder did not return a checkpoint callback.")

    training_environment_context = (
        collect_training_environment_context(
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
        workflow_context=training_environment_context,
    )

    run_log.info(
        "Exported runtime environment to: '%s'",
        runtime_environment_path,
    )

    run_log.info("Starting training...")

    interruption_signal: TrainingInterruptionSignal | None = None
    try:
        with capture_console_streams(
            log_out_dir=run_context.log_dir,
            capture_stdout=False,
        ):
            trainer.fit(model, datamodule=datamodule)

    except SIGTERMException:
        if not trainer.received_sigterm:
            raise

        interruption_signal = "sigterm"

        run_log.info(
            "Training was stopped by SIGTERM. Attempting to continue "
            "gracefully with BenchRep post-training finalization."
        )

    except SystemExit as exc:
        lightning_handled_sigint = (
            exc.code == 1
            and trainer.interrupted
            and isinstance(exc.__context__, KeyboardInterrupt)
        )

        if not lightning_handled_sigint:
            raise

        interruption_signal = "sigint"

        run_log.info(
            "Training was stopped by SIGINT. Attempting to continue "
            "gracefully with BenchRep post-training finalization."
        )

    except Exception as exc:
        if precondition_result.should_wrap_batch_contract_errors:
            run_log.error(
                "Training failed while using an external datamodule "
                "with an internal model.",
                exc_info=True,
            )

            raise RuntimeError(
                format_external_datamodule_failure_message(
                    stage="training",
                    precondition_result=precondition_result,
                    original_error=exc,
                )
            ) from exc

        raise

    early_stopping_record = build_early_stopping_record(
        early_stopping_callback
    )

    if early_stopping_record is not None:
        run_log.info(
            "Early stopping outcome: %s; reason=%s; monitor=%s; "
            "stopped_epoch=%s; best_score=%s; wait_count=%s",
            (
                "triggered"
                if early_stopping_record.triggered
                else "not_triggered"
            ),
            early_stopping_record.reason,
            early_stopping_record.monitor,
            early_stopping_record.stopped_epoch,
            early_stopping_record.best_score,
            early_stopping_record.wait_count,
        )

    run_log.info("Finished training")
    completed_at = now_isoformat()

    # Log checkpoint errors and warnings
    checkpoint_errors, checkpoint_warnings = (
        validate_training_checkpoint_outputs(
            checkpoint_config=resolved_training_config.checkpointing,
            checkpoint_callback=checkpoint_callback,
        )
    )

    training_errors = list(checkpoint_errors)
    training_warnings = [
        *precondition_result.warnings,
        *checkpoint_warnings,
    ]

    for error in training_errors:
        run_log.error(error)

    for warning in checkpoint_warnings:
        run_log.warning(warning)

    # Export torchview graph if possible
    torchview_graph_path = None

    if resolved_training_config.inspection.torchview.enabled:
        try:
            dummy_input_size = infer_dummy_input_size(datamodule)
            torchview_graph_path = export_torchview_graph(
                model=model,
                input_size=dummy_input_size,
                output_path=run_context.training_architecture_dir / "model_graph.png",
                expand_nested=resolved_training_config.inspection.torchview.expand_nested,
                depth=resolved_training_config.inspection.torchview.depth,
            )

            if torchview_graph_path is not None:
                run_log.info("Exported torchview graph to: '%s'", torchview_graph_path)
            else:
                warning = (
                    "Torchview graph export was requested, but no graph "
                    "was produced."
                )
                training_warnings.append(warning)
                run_log.warning(warning)

        except Exception as exc:
            torchview_graph_path = None

            warning = (
                "Torchview graph export failed and was skipped: "
                f"{type(exc).__name__}: {exc}"
            )
            training_warnings.append(warning)

            run_log.warning(
                warning,
                exc_info=True,
            )

    # Finalize status
    status_report = build_training_status_report(
        errors=training_errors,
        warnings=training_warnings,
        interruption_signal=interruption_signal,
    )

    # Export training manifest
    assert model is not None
    assert datamodule is not None

    manifest_path = run_context.metadata_dir / "training_manifest.yaml"
    training_manifest = write_training_manifest(
        config_composition_result=config_composition_result,
        output_path=manifest_path,
        run_spec=run_spec,
        run_context=run_context,
        checkpoint_callback=checkpoint_callback,
        early_stopping_record=early_stopping_record,
        torchview_graph_path=torchview_graph_path,
        created_at=created_at,
        completed_at=completed_at,
        status_report=status_report,
        model_class_name=type(model).__name__,
        datamodule_class_name=type(datamodule).__name__,
    )

    run_log.info("Exported training manifest to: '%s'", manifest_path)

    log_outcome_summary(
        run_log=run_log,
        workflow_name="Training",
        workflow_status=status_report.status,
        summary=training_manifest["outcome_summary"],
    )

    if status_report.status == "failed":
        raise RuntimeError(
            "Training was finalized with status 'failed': "
            f"{'; '.join(training_errors)} "
            f"The failure manifest was written to '{manifest_path}'."
        )

    return TrainingWorkflowResult(
        config=resolved_training_config,
        run_spec=run_spec,
        run_context=run_context,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        checkpoint_callback=checkpoint_callback,
        early_stopping_callback=early_stopping_callback,
        status_report=status_report,
        manifest_path=manifest_path,
        torchview_graph_path=torchview_graph_path,
    )