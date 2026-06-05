from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
import torch

from benchrep.assembly import load_yaml
from benchrep.assembly.builders import (
    build_datamodule,
    build_model,
    build_trainer,
)
from benchrep.assembly.resolvers import resolve_prediction_config
from benchrep.assembly.schemas import parse_prediction_config
from benchrep.records import (
    save_config_records,
    setup_run_logger,
    capture_console_streams,
)
from benchrep.runtime import RunContext


def main() -> None:
    args = parse_args()

    # Parse and resolve config
    raw_pred_config_path = Path(args.config).resolve()
    raw_pred_config = load_yaml(raw_pred_config_path)
    pred_config = parse_prediction_config(raw_pred_config)
    run_spec = resolve_prediction_config(pred_config)

    # Setup paths
    model_name = f"{run_spec.training_config.model.name}_prediction"

    run_context = RunContext.create(
        output_root=run_spec.training_config.run.output_root,
        stage=run_spec.stage,
        project_name=run_spec.training_config.run.project_name,
        model_name=model_name,
    )
    created_at = datetime.now().isoformat(timespec="seconds") #TODO use when writing manifest

    run_log = setup_run_logger(log_out_dir=run_context.log_dir)

    run_log.info("Prediction run initialized from config: '%s'", raw_pred_config_path)
    run_log.info("Prediction outputs will be saved to: '%s'", run_context.output_dir)
    run_log.info("Resolved training manifest: '%s'", run_spec.training_manifest_path)
    run_log.info("Resolved training config: '%s'", run_spec.resolved_training_config_path)
    run_log.info("Resolved checkpoint: '%s'", run_spec.checkpoint_path)

    # Bookkeeping --- config
    save_config_records(
        original_config_path=raw_pred_config_path,
        resolved_config=pred_config,
        config_out_dir=run_context.config_dir,
    )

    # Enforce reproducibility
    L.seed_everything(
        run_spec.seed,
        workers=run_spec.seed_workers)
    run_log.info("Global seed set to %s", run_spec.seed)

    if run_spec.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(
            run_spec.float32_matmul_precision
        )
        run_log.info(
            "float32 matmul precision set to '%s'",
            run_spec.float32_matmul_precision,
        )

    # Build model and predict
    datamodule = build_datamodule(
        dataset_config=run_spec.dataset_config,
        datamodule_config=run_spec.datamodule_config,
        seed=run_spec.seed,
        stage=run_spec.stage,
        split=run_spec.split,
    )

    # Build model and load checkpoint
    model = build_model(config=run_spec.training_config)
    checkpoint = torch.load(run_spec.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Build trainer and predict
    trainer, _ = build_trainer(
        trainer_config=run_spec.trainer_config,
        stage=run_spec.stage,
        run_context=run_context,
        max_batches=run_spec.max_batches,
    )

    run_log.info("Starting prediction...")

    with capture_console_streams(log_out_dir=run_context.log_dir, capture_stdout=False):
        predictions = trainer.predict(
            model,
            datamodule=datamodule,
            return_predictions=True,
        )

    run_log.info("Finished prediction")
    completed_at = datetime.now().isoformat(timespec="seconds")






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BenchRep prediction from a YAML config."
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