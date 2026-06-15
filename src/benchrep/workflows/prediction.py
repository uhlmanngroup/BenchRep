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
    export_prediction_outputs,
    write_prediction_manifest,
)
from benchrep.runtime import RunContext
from benchrep.assembly.register_builtins import register_builtins


def main() -> None:
    register_builtins()

    args = parse_args()

    # Parse and resolve config
    raw_pred_config_path = Path(args.config).resolve()
    raw_pred_config = load_yaml(raw_pred_config_path)
    pred_config = parse_prediction_config(raw_pred_config)
    run_spec = resolve_prediction_config(pred_config)

    # Setup paths
    model_name = f"{run_spec.training_config.model.name}"

    run_context = RunContext.create(
        output_root=run_spec.training_config.run.output_root,
        stage=run_spec.stage,
        project_name=run_spec.training_config.run.project_name,
        model_name=model_name,
    )
    created_at = datetime.now().isoformat(timespec="seconds")

    run_log = setup_run_logger(log_out_dir=run_context.log_dir)

    run_log.info("Prediction run initialized from config: '%s'", raw_pred_config_path)
    run_log.info("Prediction outputs will be saved to: '%s'", run_context.output_dir)
    run_log.info("Resolved training manifest: '%s'", run_spec.training_manifest_path)
    run_log.info("Resolved training config: '%s'", run_spec.resolved_training_config_path)
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
    run_log.info("Prediction returned %s batches.", len(predictions))

    if predictions:
        first_prediction = predictions[0]
        if isinstance(first_prediction, dict):
            run_log.info(
                "First prediction batch keys: %s",
                tuple(first_prediction.keys()),
            )
        else:
            run_log.info(
                "First prediction batch type: %s",
                type(first_prediction).__name__,
            )

    run_log.info("Exporting prediction outputs...")

    export_paths = export_prediction_outputs(
        predictions=predictions,
        export_spec=run_spec.export_spec,
        embedding_dir=run_context.prediction_embeddings_dir,
        reconstruction_dir=run_context.prediction_reconstructions_dir,
    )

    if export_paths.embedding_export is not None:
        run_log.info(
            "Exported embedding artifact to: '%s'",
            export_paths.embedding_export,
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

    completed_at = datetime.now().isoformat(timespec="seconds")

    # Export prediction manifest
    manifest_path = run_context.metadata_dir / "prediction_manifest.yaml"
    write_prediction_manifest(
        output_path=manifest_path,
        run_spec=run_spec,
        run_context=run_context,
        input_config_path=raw_pred_config_path,
        export_paths=export_paths,
        created_at=created_at,
        completed_at=completed_at,
        status="completed",
    )

    run_log.info("Exported prediction manifest to: '%s'", manifest_path)


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