from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

import torch

from benchrep.runtime import RunContext
from benchrep.records import (
    save_config_records,
    capture_console_streams,
    setup_run_logger,
    write_training_manifest,
    export_torchview_graph,
    infer_dummy_input_size,
)
from benchrep.assembly import load_yaml
from benchrep.assembly.schemas import parse_training_config, TrainingConfig
from benchrep.assembly.builders import build_datamodule, build_model, build_trainer
from benchrep.assembly.register_builtins import register_builtins


@dataclass
class TrainingWorkflowResult:
    config: TrainingConfig
    run_context: RunContext
    model: L.LightningModule
    datamodule: L.LightningDataModule
    trainer: L.Trainer
    checkpoint_callback: ModelCheckpoint
    manifest_path: Path
    torchview_graph_path: Path | None


def train(
        config_path: Path | str,
        model: L.LightningModule | None = None,
        datamodule: L.LightningDataModule | None = None,
) -> TrainingWorkflowResult:
    register_builtins()

    # Override flags
    manual_model_provided = model is not None
    manual_datamodule_provided = datamodule is not None

    # Parse config
    raw_config_path = Path(config_path).resolve()
    raw_config = load_yaml(raw_config_path)
    config = parse_training_config(
        raw_config,
        model_overridden=manual_model_provided,
        datamodule_overridden=manual_datamodule_provided,
    )

    if not manual_model_provided:
        # Setup paths
        model_name = f"{config.model.name}_{config.encoder.name}"
        if config.decoder is not None:
            model_name = f"{model_name}_{config.decoder.name}"
    else:
        model_name = f"manual_{model.__class__.__name__}"

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

    if not manual_datamodule_provided:
        datamodule = build_datamodule(
            dataset_config=config.dataset,
            datamodule_config=config.datamodule,
            seed=config.reproducibility.seed,
            stage=config.stage,
            split=config.data.split,
        )
    else:
        run_log.info("Manual datamodule was provided; config.dataset and config.datamodule were ignored.")

    if not manual_model_provided:
        model = build_model(config=config)
    else:
        run_log.info("Manual model was provided; config.model/encoder/decoder were ignored.")

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

    with capture_console_streams(log_out_dir=run_context.log_dir, capture_stdout=False):
        trainer.fit(model, datamodule=datamodule)

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
        model_source="external_object" if manual_model_provided else "config",
        model_class_name=model.__class__.__name__,
        datamodule_source=(
            "external_object" if manual_datamodule_provided else "config"
        ),
        datamodule_class_name=datamodule.__class__.__name__,
    )

    run_log.info("Exported training manifest to: '%s'", manifest_path)

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