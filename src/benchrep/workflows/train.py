from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.utilities.exceptions import SIGTERMException

import torch

from benchrep.utils.workflow_utils import TrainingConfigInput
from benchrep.utils.workflow_overrides import TrainingOverrides
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
    prepare_composite_torchview_input_data,
    get_runtime_environment_filename,
    collect_training_environment_context,
    write_runtime_environment,
    ModelGraphDependencyError,
    export_composite_model_spec_graph,
    print_divider,
)
from benchrep.records.utils import now_isoformat
from benchrep.interfaces.model_families import (
    SupportedModel,
    ModelFamilySpec,
    AUTOENCODER_FAMILY,
    VAE_FAMILY,
    COMPOSITE_FAMILY,
)
from benchrep.assembly.config import (
    compose_effective_config,
)
from benchrep.assembly.schemas import TrainingConfig
from benchrep.assembly.builders import (
    build_datamodule,
    build_dataset,
    build_transform_pipelines_bundle,
    build_model,
    build_trainer,
    build_runtime_component,
)
from benchrep.assembly.registries.builtins import register_builtins
from benchrep.assembly.resolvers.utils import (
    ComponentSource,
    get_component_override_name,
    resolve_component_source,
)
from benchrep.inspection.gradient_norms import GradientNormInspectionCallback


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
    composite_model_spec_graph_path: Path | None
    torchview_graph_path: Path | None
    status_report: TrainingStatusReport
    manifest_path: Path


# Model-specific wrappers
def train_ae(
        *,
        config: TrainingConfigInput = None,
        overrides: TrainingOverrides | None = None,
        runtime_callbacks: Sequence[Callback] | None = None,
        compatibility_policy: CompatibilityPolicy = "error",
        capture_stdout: bool = False,
) -> TrainingWorkflowResult:
    """Train a canonical autoencoder and export checkpoints and run records.
    
    Parameters
    ----------
    config : TrainingConfigInput, optional
        Training YAML path or `TrainingConfig` object. If omitted, the
        configuration sections in `overrides`, together with schema defaults,
        must supply a valid training configuration.
    overrides : TrainingOverrides, optional
        Whole configuration sections and supported runtime model/datamodule
        classes or instances. Supplied sections replace the corresponding
        sections in `config`.
    runtime_callbacks : Sequence[Callback], optional
        Lightning callback instances appended after configured additional
        callbacks. Use this for callbacks requiring runtime objects, such as
        an Optuna trial. Callback class names are recorded before fitting;
        the instances are not serialized for later reconstruction.
    compatibility_policy : {"error", "warn"}, default="error"
        Handling of supported model/datamodule contract checks. `"warn"`
        logs those compatibility issues and continues; it does not disable
        configuration validation or runtime checks.
    capture_stdout : bool, default=False
        Also save stdout during the captured execution phase to the run's
        log directory. Stderr is captured independently of this option;
        both streams remain visible in the console.
    
    Returns
    -------
    TrainingWorkflowResult
        Resolved config, model, datamodule, trainer, callback references,
        run paths, status report, and manifest path.
    
    Raises
    ------
    RuntimeError
        Post-training validation finalizes the run as failed, for example
        when required checkpoint outputs are unavailable. Setup and execution
        can also raise their own validation or runtime errors.
    
    Notes
    -----
    Inspect `status_report` on the returned result, including
    `interruption_signal` for a handled SIGINT or SIGTERM. Other execution
    errors may propagate before final records are written.
    """
    overrides = overrides if overrides is not None else TrainingOverrides()
    overrides.validate_for(AUTOENCODER_FAMILY)

    return _train(
        model_family=AUTOENCODER_FAMILY,
        overrides=overrides,
        config=config,
        runtime_callbacks=runtime_callbacks,
        compatibility_policy=compatibility_policy,
        capture_stdout=capture_stdout,
    )


def train_vae(
        *,
        config: TrainingConfigInput = None,
        overrides: TrainingOverrides | None = None,
        runtime_callbacks: Sequence[Callback] | None = None,
        compatibility_policy: CompatibilityPolicy = "error",
        capture_stdout: bool = False,
) -> TrainingWorkflowResult:
    """Train a canonical variational autoencoder and export checkpoints and run records.
    
    Parameters
    ----------
    config : TrainingConfigInput, optional
        Training YAML path or `TrainingConfig` object. If omitted, the
        configuration sections in `overrides`, together with schema defaults,
        must supply a valid training configuration.
    overrides : TrainingOverrides, optional
        Whole configuration sections and supported runtime model/datamodule
        classes or instances. Supplied sections replace the corresponding
        sections in `config`.
    runtime_callbacks : Sequence[Callback], optional
        Lightning callback instances appended after configured additional
        callbacks. Use this for callbacks requiring runtime objects, such as
        an Optuna trial. Callback class names are recorded before fitting;
        the instances are not serialized for later reconstruction.
    compatibility_policy : {"error", "warn"}, default="error"
        Handling of supported model/datamodule contract checks. `"warn"`
        logs those compatibility issues and continues; it does not disable
        configuration validation or runtime checks.
    capture_stdout : bool, default=False
        Also save stdout during the captured execution phase to the run's
        log directory. Stderr is captured independently of this option;
        both streams remain visible in the console.
    
    Returns
    -------
    TrainingWorkflowResult
        Resolved config, model, datamodule, trainer, callback references,
        run paths, status report, and manifest path.
    
    Raises
    ------
    RuntimeError
        Post-training validation finalizes the run as failed, for example
        when required checkpoint outputs are unavailable. Setup and execution
        can also raise their own validation or runtime errors.
    
    Notes
    -----
    Inspect `status_report` on the returned result, including
    `interruption_signal` for a handled SIGINT or SIGTERM. Other execution
    errors may propagate before final records are written.
    """
    overrides = overrides if overrides is not None else TrainingOverrides()
    overrides.validate_for(VAE_FAMILY)

    return _train(
        model_family=VAE_FAMILY,
        overrides=overrides,
        config=config,
        runtime_callbacks=runtime_callbacks,
        compatibility_policy=compatibility_policy,
        capture_stdout=capture_stdout,
    )


def train_composite(
        *,
        config: TrainingConfigInput = None,
        overrides: TrainingOverrides | None = None,
        runtime_callbacks: Sequence[Callback] | None = None,
        capture_stdout: bool = False,
) -> TrainingWorkflowResult:
    """Train a composite model and export checkpoints and run records.
    
    Parameters
    ----------
    config : TrainingConfigInput, optional
        Training YAML path or `TrainingConfig` object. If omitted, the
        configuration sections in `overrides`, together with schema defaults,
        must supply a valid training configuration.
    overrides : TrainingOverrides, optional
        Whole configuration sections and a runtime datamodule class or
        instance. Whole-model overrides are not supported for composite models.
    runtime_callbacks : Sequence[Callback], optional
        Lightning callback instances appended after configured additional
        callbacks. Use this for callbacks requiring runtime objects, such as
        an Optuna trial. Callback class names are recorded before fitting;
        the instances are not serialized for later reconstruction.
    capture_stdout : bool, default=False
        Also save stdout during the captured execution phase to the run's
        log directory. Stderr is captured independently of this option;
        both streams remain visible in the console.
    
    Returns
    -------
    TrainingWorkflowResult
        Resolved config, model, datamodule, trainer, callback references,
        run paths, status report, and manifest path.
    
    Raises
    ------
    RuntimeError
        Post-training validation finalizes the run as failed, for example
        when required checkpoint outputs are unavailable. Setup and execution
        can also raise their own validation or runtime errors.
    
    Notes
    -----
    Inspect `status_report` on the returned result, including
    `interruption_signal` for a handled SIGINT or SIGTERM. Other execution
    errors may propagate before final records are written.
    """
    overrides = overrides if overrides is not None else TrainingOverrides()
    overrides.validate_for(COMPOSITE_FAMILY)

    return _train(
        model_family=COMPOSITE_FAMILY,
        overrides=overrides,
        config=config,
        runtime_callbacks=runtime_callbacks,
        capture_stdout=capture_stdout,
    )


def _train(
        model_family: ModelFamilySpec,
        overrides: TrainingOverrides,
        config: TrainingConfigInput = None,
        runtime_callbacks: Sequence[Callback] | None = None,
        compatibility_policy: CompatibilityPolicy = "error",
        capture_stdout: bool = False,
) -> TrainingWorkflowResult:

    print_divider("training")

    config_components = overrides.config_components
    model = overrides.model
    datamodule = overrides.datamodule

    register_builtins()

    model_source: ComponentSource
    if model is None:
        model_source = "config"
    elif isinstance(model, type):
        model_source = "external_class"
    else:
        model_source = "external_instance"

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
        config=config,
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
        run_name_prefix=run_spec.run_identity.run_name_prefix,
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
        original_config=config,
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
            step.name
            for pipeline in resolved_training_config.transform_pipelines
            for step in pipeline.steps
            if "validation" in step.apply_to
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

        transform_pipelines = build_transform_pipelines_bundle(
            resolved_training_config.transform_pipelines,
        )

        dataset = build_dataset(
            dataset_config=dataset_config,
        )

        datamodule = build_datamodule(
            dataset=dataset,
            datamodule_config=datamodule_config,
            seed=resolved_training_config.reproducibility.seed,
            stage=run_spec.stage,
            training_pipelines=transform_pipelines.training,
            validation_pipelines=transform_pipelines.validation,
        )
    else:
        run_log.info(
            "External datamodule was provided; dataset, datamodule, and transform "
            "pipelines config sections will be ignored regardless of whether they "
            "came from YAML, a full config object, or config_components."
        )

    model = build_runtime_component(
        model,
        override_config=resolved_training_config.overrides.model,
        component_name="model",
    )

    if not model_is_external:
        model = build_model(run_spec=run_spec)
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

    gradient_norm_callback = None
    if run_spec.gradient_norm_specs:
        gradient_norm_callback = GradientNormInspectionCallback(
            model_family=run_spec.model_family.name,
            gradient_norm_specs=run_spec.gradient_norm_specs,
        )

    trainer, checkpoint_callback, early_stopping_callback = build_trainer(
        trainer_config=resolved_training_config.trainer,
        stage=run_spec.stage,
        run_context=run_context,
        logger_config=resolved_training_config.logger,
        checkpoint_config=resolved_training_config.checkpointing,
        early_stopping_config=resolved_training_config.early_stopping,
        additional_callback_configs=resolved_training_config.additional_callbacks,
        gradient_norm_callback=gradient_norm_callback,
        runtime_callbacks=runtime_callbacks,
    )

    if checkpoint_callback is None:
        raise RuntimeError("Training trainer builder did not return a checkpoint callback.")

    training_environment_context = (
        collect_training_environment_context(
            run_spec=run_spec,
            trainer=trainer,
        )
    )

    runtime_callback_classes = [
        f"{type(callback).__module__}.{type(callback).__qualname__}"
        for callback in (runtime_callbacks or ())
    ]
    if runtime_callback_classes:
        training_environment_context["runtime_callbacks"] = runtime_callback_classes

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
            capture_stdout=capture_stdout,
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
    if gradient_norm_callback is not None:
        training_warnings.extend(gradient_norm_callback.issues)

        for warning in gradient_norm_callback.issues:
            run_log.warning(warning)

    # Automatically export the resolved Composite model specification graph.
    composite_model_spec_graph_path = None

    if run_spec.composite_model_spec is not None:
        try:
            composite_model_spec_graph_path = (
                export_composite_model_spec_graph(
                    model_spec=run_spec.composite_model_spec,
                    loss_specs=run_spec.loss_specs,
                    output_path=(
                            run_context.training_architecture_dir
                            / "composite_model_spec_graph.svg"
                    ),
                )
            )

            run_log.info(
                "Automatically exported Composite model specification "
                "graph to: '%s'",
                composite_model_spec_graph_path,
            )

        except ModelGraphDependencyError as exc:
            run_log.info(
                "BenchRep automatically exports a Composite model "
                "specification graph for Composite training runs. This "
                "optional export was skipped because a required model-graph "
                "dependency is unavailable: %s",
                exc,
            )

        except Exception as exc:
            warning = (
                "BenchRep automatically exports a Composite model "
                "specification graph for Composite training runs, but the "
                "export failed and was skipped: "
                f"{type(exc).__name__}: {exc}"
            )
            training_warnings.append(warning)
            run_log.warning(warning, exc_info=True)

    for error in training_errors:
        run_log.error(error)

    for warning in checkpoint_warnings:
        run_log.warning(warning)

    # Export Torchview graph if requested.
    torchview_graph_path = None

    if resolved_training_config.inspection.torchview.enabled:
        try:
            torchview_export_kwargs: dict[str, Any]

            if run_spec.composite_model_spec is not None:
                torchview_input_data = prepare_composite_torchview_input_data(
                    datamodule=datamodule,
                    model_spec=run_spec.composite_model_spec,
                )
                torchview_export_kwargs = {
                    "input_data": torchview_input_data,
                }
            else:
                dummy_input_size = infer_dummy_input_size(datamodule)
                torchview_export_kwargs = {
                    "input_size": dummy_input_size,
                }

            torchview_graph_path = export_torchview_graph(
                model=model,
                output_path=(
                        run_context.training_architecture_dir / "torchview_model_graph.svg"
                ),
                expand_nested=(
                    resolved_training_config
                    .inspection
                    .torchview
                    .expand_nested
                ),
                depth=resolved_training_config.inspection.torchview.depth,
                **torchview_export_kwargs,
            )

            run_log.info(
                "Exported Torchview graph to: '%s'",
                torchview_graph_path,
            )

        except Exception as error:
            torchview_graph_path = None

            warning = (
                "Torchview graph export was requested but failed and was "
                f"skipped: {type(error).__name__}: {error}"
            )
            training_warnings.append(warning)
            run_log.warning(warning, exc_info=True)

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
        runtime_callback_classes=runtime_callback_classes,
        early_stopping_record=early_stopping_record,
        composite_model_spec_graph_path=composite_model_spec_graph_path,
        torchview_graph_path=torchview_graph_path,
        created_at=created_at,
        completed_at=completed_at,
        status_report=status_report,
        model_class_name=type(model).__name__,
        datamodule_class_name=type(datamodule).__name__,
        capture_stdout=capture_stdout,
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

    print_divider("training", end=True)

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
        composite_model_spec_graph_path=composite_model_spec_graph_path,
        torchview_graph_path=torchview_graph_path,
    )