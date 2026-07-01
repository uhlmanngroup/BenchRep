from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

import torch

from benchrep.runtime.run_context import RunContext
from benchrep.runtime.train_run_validation import (
    validate_train_preconditions,
    format_external_datamodule_training_failure_message,
    audit_train_outputs,
)
from benchrep.runtime.utils import CompatibilityPolicy
from benchrep.records import (
    save_config_records,
    capture_console_streams,
    setup_run_logger,
    write_training_manifest,
    export_torchview_graph,
    infer_dummy_input_size,
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
from benchrep.assembly import load_yaml
from benchrep.assembly.schemas import parse_training_config, TrainingConfig
from benchrep.assembly.builders import build_datamodule, build_model, build_trainer
from benchrep.assembly.register_builtins import register_builtins


@dataclass
class TrainingWorkflowResult:
    config: TrainingConfig
    run_context: RunContext
    model: SupportedModel
    datamodule: L.LightningDataModule
    trainer: L.Trainer
    checkpoint_callback: ModelCheckpoint
    manifest_path: Path
    torchview_graph_path: Path | None


# Model-specific wrappers
def train_ae(
        config_path: Path | str,
        model: BenchRepAutoencoderModel | None = None,
        datamodule: L.LightningDataModule | None = None,
        compatibility_policy: CompatibilityPolicy = "error",
) -> TrainingWorkflowResult:
    return _train(
        model_family=AUTOENCODER_FAMILY,
        config_path=config_path,
        model=model,
        datamodule=datamodule,
        compatibility_policy=compatibility_policy,
    )


def train_vae(
        config_path: Path | str,
        model: BenchRepVAEModel | None = None,
        datamodule: L.LightningDataModule | None = None,
        compatibility_policy: CompatibilityPolicy = "error",
) -> TrainingWorkflowResult:
    return _train(
        model_family=VAE_FAMILY,
        config_path=config_path,
        model=model,
        datamodule=datamodule,
        compatibility_policy=compatibility_policy,
    )


def _train(
        model_family: ModelFamilySpec,
        config_path: Path | str,
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

    # Parse config
    raw_config_path = Path(config_path).resolve()
    raw_config = load_yaml(raw_config_path)
    config = parse_training_config(
        raw_config,
        model_overridden=model_is_external,
        datamodule_overridden=datamodule_is_external,
    )

    if not model_is_external:
        assert config.model is not None
        assert config.encoder is not None

        # Setup paths
        model_name = f"{config.model.name}_{config.encoder.name}"
        if config.decoder is not None:
            model_name = f"{model_name}_{config.decoder.name}"
    else:
        model_name = f"{model_family.name}_external_{type(model).__name__}"

    run_context = RunContext.create(
        output_root=config.run.output_root,
        stage=config.stage,
        project_name=config.run.project_name,
        model_name=model_name,
    )

    created_at = datetime.now().isoformat(timespec="seconds")

    # Initiate local run logger
    run_log = setup_run_logger(log_out_dir=run_context.log_dir)

    run_log.info("Run initialized with config from: '%s'", raw_config_path)
    run_log.info("Run outputs will be saved to: '%s'", run_context.output_dir)

    # Bookkeeping --- config
    save_config_records(
        original_config_path=raw_config_path,
        resolved_config=config,
        config_out_dir=run_context.config_dir,
    )

    # Enforce reproducibility
    L.seed_everything(
        config.reproducibility.seed,
        workers=config.reproducibility.seed_workers,
    )
    run_log.info("Global seed set to %s", config.reproducibility.seed)

    if config.reproducibility.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(
            config.reproducibility.float32_matmul_precision
        )
        run_log.info(
            "float32 matmul precision set to '%s'",
            config.reproducibility.float32_matmul_precision,
        )

    if not datamodule_is_external:
        # Avoid confusing type checker...
        dataset_config = config.dataset
        datamodule_config = config.datamodule

        assert dataset_config is not None
        assert datamodule_config is not None

        datamodule = build_datamodule(
            dataset_config=dataset_config,
            datamodule_config=datamodule_config,
            seed=config.reproducibility.seed,
            stage=config.stage,
            split=config.data.split,
        )
    else:
        run_log.info("External datamodule was provided; config.dataset and config.datamodule were ignored.")

    if not model_is_external:
        model = build_model(config=config)
    else:
        run_log.info(
            "External model was provided; config.model/encoder/decoder/losses/optimizer were ignored."
        )

    # Preflight check
    assert model is not None
    assert datamodule is not None

    precondition_result = validate_train_preconditions(
        model_family=model_family,
        model=model,
        model_is_external=model_is_external,
        datamodule_is_external=datamodule_is_external,
        compatibility_policy=compatibility_policy,
    )

    trainer, checkpoint_callback = build_trainer(
        trainer_config=config.trainer,
        stage=config.stage,
        run_context=run_context,
        logger_config=config.logger,
        checkpoint_config=config.checkpointing,
    )

    if checkpoint_callback is None:
        raise RuntimeError("Training trainer builder did not return a checkpoint callback.")

    run_log.info("Starting training...")

    try:
        with capture_console_streams(log_out_dir=run_context.log_dir, capture_stdout=False):
            trainer.fit(model, datamodule=datamodule)

    except Exception as exc:
        if precondition_result.should_wrap_training_errors_with_batch_hint:
            run_log.error(
                "Training failed while using an external datamodule with an internal model.",
                exc_info=True,
            )

            raise RuntimeError(
                format_external_datamodule_training_failure_message(
                    precondition_result=precondition_result,
                    original_error=exc,
                )
            ) from exc

        raise

    run_log.info("Finished training")
    completed_at = datetime.now().isoformat(timespec="seconds")

    # Export torchview graph if possible
    torchview_graph_path = None

    if config.inspection.torchview.enabled:
        try:
            dummy_input_size = infer_dummy_input_size(datamodule)
            torchview_graph_path = export_torchview_graph(
                model=model,
                input_size=dummy_input_size,
                output_path=run_context.training_architecture_dir / "model_graph.png",
                expand_nested=config.inspection.torchview.expand_nested,
                depth=config.inspection.torchview.depth,
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
        output_path=manifest_path,
        config=config,
        run_context=run_context,
        input_config_path=raw_config_path,
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

    audit_train_outputs(
        run_context=run_context,
        input_config_path=raw_config_path,
        resolved_config_path=run_context.config_dir / "resolved_config.yaml",
        checkpoint_dir=run_context.training_checkpoint_dir,
        training_manifest_path=manifest_path,
        torchview_requested=config.inspection.torchview.enabled,
        torchview_graph_path=torchview_graph_path,
    )

    return TrainingWorkflowResult(
        config=config,
        run_context=run_context,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        checkpoint_callback=checkpoint_callback,
        manifest_path=manifest_path,
        torchview_graph_path=torchview_graph_path,
    )