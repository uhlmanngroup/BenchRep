from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

import torch

from benchrep.runtime.run_context import RunContext
from benchrep.runtime.train_run_validation import (
    validate_train_contract_compatibility,
    audit_train_outputs,
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
    write_audit_report,
)
from benchrep.interfaces.model_families import (
    SupportedModel,
    ModelFamilySpec,
    AUTOENCODER_FAMILY,
    VAE_FAMILY,
)
from benchrep.interfaces.models import (
    BenchRepAutoencoderModel,
    BenchRepVAEModel,
)
from benchrep.assembly.config import (
    compose_effective_config,
    SupportedConfigComponent,
)
from benchrep.assembly.schemas import TrainingConfig
from benchrep.assembly.builders import build_datamodule, build_model, build_trainer
from benchrep.assembly.registries.builtins import register_builtins


@dataclass
class TrainingWorkflowResult:
    config: TrainingConfig
    run_context: RunContext
    model: SupportedModel
    datamodule: L.LightningDataModule
    trainer: L.Trainer
    checkpoint_callback: ModelCheckpoint
    manifest_path: Path
    audit_report_path: Path
    torchview_graph_path: Path | None


# Model-specific wrappers
def train_ae(
        config_path: Path | str | None = None,
        full_config_object: TrainingConfig | None = None,
        config_components: Mapping[str, SupportedConfigComponent] | None = None,
        model: BenchRepAutoencoderModel | None = None,
        datamodule: L.LightningDataModule | None = None,
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
        config_components: Mapping[str, SupportedConfigComponent] | None = None,
        model: BenchRepVAEModel | None = None,
        datamodule: L.LightningDataModule | None = None,
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


def _train(
        model_family: ModelFamilySpec,
        config_path: Path | str | None = None,
        full_config_object: TrainingConfig | None = None,
        config_components: Mapping[str, SupportedConfigComponent] | None = None,
        model: SupportedModel | None = None,
        datamodule: L.LightningDataModule | None = None,
        compatibility_policy: CompatibilityPolicy = "error"
) -> TrainingWorkflowResult:
    register_builtins()

    if compatibility_policy not in {"error", "warn"}:
        raise ValueError(
            "compatibility_policy must be 'error' or 'warn'."
        )

    # Override flags
    model_is_external = model is not None
    datamodule_is_external = datamodule is not None

    # Compose and parse config
    config_composition_result = compose_effective_config(
        schema=TrainingConfig,
        config_path=config_path,
        full_config_object=full_config_object,
        config_components=config_components,
        external_model=model_is_external,
        external_datamodule=datamodule_is_external,
    )

    train_config = config_composition_result.effective_config

    if not model_is_external:
        assert train_config.model is not None
        assert train_config.encoder is not None

        # Setup paths
        model_name = f"{train_config.model.name}_{train_config.encoder.name}"
        if train_config.decoder is not None:
            model_name = f"{model_name}_{train_config.decoder.name}"
    else:
        model_name = f"{model_family.name}_external_{type(model).__name__}"

    run_context = RunContext.create(
        output_root=train_config.run.output_root,
        stage=train_config.stage,
        project_name=train_config.run.project_name,
        model_name=model_name,
    )

    created_at = datetime.now().isoformat(timespec="seconds")

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
        resolved_config=train_config,
        config_out_dir=run_context.config_dir,
    )

    # Enforce reproducibility
    L.seed_everything(
        train_config.reproducibility.seed,
        workers=train_config.reproducibility.seed_workers,
    )
    run_log.info("Global seed set to %s", train_config.reproducibility.seed)

    if train_config.reproducibility.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(
            train_config.reproducibility.float32_matmul_precision
        )
        run_log.info(
            "float32 matmul precision set to '%s'",
            train_config.reproducibility.float32_matmul_precision,
        )

    if not datamodule_is_external:
        # Avoid confusing type checker...
        dataset_config = train_config.dataset
        datamodule_config = train_config.datamodule

        assert dataset_config is not None
        assert datamodule_config is not None

        datamodule = build_datamodule(
            dataset_config=dataset_config,
            datamodule_config=datamodule_config,
            seed=train_config.reproducibility.seed,
            stage=train_config.stage,
            split=train_config.data.split,
        )
    else:
        run_log.info(
            "External datamodule was provided; dataset/datamodule config sections "
            "will be ignored regardless of whether they came from YAML, a full config "
            "object, or config_components."
        )

    if not model_is_external:
        model = build_model(config=train_config)
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
        model_family=model_family,
        model=model,
        model_is_external=model_is_external,
        datamodule_is_external=datamodule_is_external,
        compatibility_policy=compatibility_policy,
    )

    trainer, checkpoint_callback = build_trainer(
        trainer_config=train_config.trainer,
        stage=train_config.stage,
        run_context=run_context,
        logger_config=train_config.logger,
        checkpoint_config=train_config.checkpointing,
    )

    if checkpoint_callback is None:
        raise RuntimeError("Training trainer builder did not return a checkpoint callback.")

    run_log.info("Starting training...")

    try:
        with capture_console_streams(log_out_dir=run_context.log_dir, capture_stdout=False):
            trainer.fit(model, datamodule=datamodule)

    except Exception as exc:
        if precondition_result.should_wrap_batch_contract_errors:
            run_log.error(
                "Training failed while using an external datamodule with an internal model.",
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

    run_log.info("Finished training")
    completed_at = datetime.now().isoformat(timespec="seconds")

    # Export torchview graph if possible
    torchview_graph_path = None

    if train_config.inspection.torchview.enabled:
        try:
            dummy_input_size = infer_dummy_input_size(datamodule)
            torchview_graph_path = export_torchview_graph(
                model=model,
                input_size=dummy_input_size,
                output_path=run_context.training_architecture_dir / "model_graph.png",
                expand_nested=train_config.inspection.torchview.expand_nested,
                depth=train_config.inspection.torchview.depth,
            )

            if torchview_graph_path is not None:
                run_log.info("Exported torchview graph to: '%s'", torchview_graph_path)
            else:
                run_log.warning("Torchview graph export was skipped or failed.")

        except Exception as exc:
            torchview_graph_path = None
            run_log.warning(
                "Torchview graph export failed and was skipped: %s",
                exc,
                exc_info=True,
            )

    # Export training manifest
    assert model is not None
    assert datamodule is not None

    manifest_path = run_context.metadata_dir / "training_manifest.yaml"
    write_training_manifest(
        config_composition_result=config_composition_result,
        output_path=manifest_path,
        run_context=run_context,
        checkpoint_callback=checkpoint_callback,
        torchview_graph_path=torchview_graph_path,
        created_at=created_at,
        completed_at=completed_at,
        status="completed",
        model_source="external_object" if model_is_external else "config",
        model_class_name=type(model).__name__,
        datamodule_source=(
            "external_object" if datamodule_is_external else "config"
        ),
        datamodule_class_name=type(datamodule).__name__,
    )

    run_log.info("Exported training manifest to: '%s'", manifest_path)

    audit_items = audit_train_outputs(
        run_context=run_context,
        config_composition_result=config_composition_result,
        resolved_config_path=run_context.config_dir / "resolved_config.yaml",
        checkpoint_dir=run_context.training_checkpoint_dir,
        training_manifest_path=manifest_path,
        torchview_requested=train_config.inspection.torchview.enabled,
        torchview_graph_path=torchview_graph_path,
        model_source="external_object" if model_is_external else "config",
        model_class_name=type(model).__name__,
        datamodule_source="external_object" if datamodule_is_external else "config",
        datamodule_class_name=type(datamodule).__name__,
    )

    audit_report_path = write_audit_report(
        stage="training",
        audit_items=audit_items,
        output_path=(
            run_context.metadata_dir / "training_audit_report.yaml"
        ),
        audited_at=datetime.now().isoformat(timespec="seconds"),
    )

    run_log.info(
        "Exported training audit report to: '%s'",
        audit_report_path,
    )


    return TrainingWorkflowResult(
        config=train_config,
        run_context=run_context,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        checkpoint_callback=checkpoint_callback,
        manifest_path=manifest_path,
        audit_report_path=audit_report_path,
        torchview_graph_path=torchview_graph_path,
    )