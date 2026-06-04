from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
import torch

from benchrep.assembly import load_yaml
from benchrep.assembly.resolvers import resolve_prediction_config
from benchrep.assembly.schemas import parse_prediction_config
from benchrep.records import (
    save_config_records,
    setup_run_logger,
)
from benchrep.runtime import RunContext


def main() -> None:
    args = parse_args()

    raw_pred_config_path = Path(args.config).resolve()
    raw_pred_config = load_yaml(raw_pred_config_path)
    pred_config = parse_prediction_config(raw_pred_config)
    pred_plan = resolve_prediction_config(pred_config)

    model_name = f"{pred_plan.training_config.model.name}_prediction"

    run_context = RunContext.create(
        output_root=pred_plan.training_config.run.output_root,
        stage=pred_config.stage,
        project_name=pred_plan.training_config.run.project_name,
        model_name=model_name,
    )
    created_at = datetime.now().isoformat(timespec="seconds") #TODO use when writing manifest

    run_log = setup_run_logger(log_out_dir=run_context.log_dir)

    run_log.info("Prediction run initialized from config: '%s'", raw_pred_config_path)
    run_log.info("Prediction outputs will be saved to: '%s'", run_context.output_dir)
    run_log.info("Resolved training manifest: '%s'", pred_plan.training_manifest_path)
    run_log.info("Resolved training config: '%s'", pred_plan.resolved_training_config_path)
    run_log.info("Resolved checkpoint: '%s'", pred_plan.checkpoint_path)

    save_config_records(
        original_config_path=raw_pred_config_path,
        resolved_config=pred_config,
        config_out_dir=run_context.config_dir,
    )

    L.seed_everything(pred_plan.seed, workers=pred_plan.seed_workers)
    run_log.info("Global seed set to %s", pred_plan.seed)

    if pred_plan.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(pred_plan.float32_matmul_precision)
        run_log.info(
            "float32 matmul precision set to '%s'",
            pred_plan.float32_matmul_precision,
        )

    run_log.info("Prediction workflow shell completed.")


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