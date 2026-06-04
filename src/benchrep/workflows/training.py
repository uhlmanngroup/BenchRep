from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
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
from benchrep.assembly.schemas import parse_training_config
from benchrep.assembly.builders import build_datamodule, build_model, build_trainer


def main() -> None:
    args = parse_args()

    # Parse config
    raw_config_path = Path(args.config).resolve()
    raw_config = load_yaml(raw_config_path)
    config = parse_training_config(raw_config)

    # Setup paths
    model_name = f"{config.model.name}_{config.encoder.name}"
    if config.decoder is not None:
        model_name = f"{model_name}_{config.decoder.name}"

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

    # Build model and train
    datamodule = build_datamodule(
        dataset_config=config.dataset,
        datamodule_config=config.datamodule,
        seed=config.reproducibility.seed)

    model = build_model(config=config)

    trainer, checkpoint_callback = build_trainer(
        trainer_config=config.trainer,
        logger_config=config.logger,
        checkpoint_config=config.checkpointing,
        run_context=run_context,
    )

    run_log.info("Starting training...")

    with capture_console_streams(log_out_dir=run_context.log_dir, capture_stdout=False):
        trainer.fit(model, datamodule=datamodule)

    run_log.info("Finished training")
    completed_at = datetime.now().isoformat(timespec="seconds")

    # Export torchview graph if possible
    if config.inspection.torchview.enabled:
        dummy_input_size = infer_dummy_input_size(datamodule)
        torchview_graph_path = export_torchview_graph(
            model=model,
            input_size=dummy_input_size,
            output_path=run_context.architecture_dir / "model_graph.png",
            expand_nested=config.inspection.torchview.expand_nested,
            depth=config.inspection.torchview.depth,
        )
        if torchview_graph_path is not None:
            run_log.info("Exported torchview graph to: '%s'", torchview_graph_path)
        else:
            run_log.warning("Torchview graph export was skipped or failed.")
    else:
        torchview_graph_path = None

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
    )

    run_log.info("Exported training manifest to: '%s'", manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BenchRep model from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()