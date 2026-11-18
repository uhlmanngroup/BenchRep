"""Compare From scratch and Pretrained ResNet18 encoders with NT-Xent.

'From scratch' uses random initialization; 'Pretrained' starts with ImageNet
backbone weights. Both are contrastively trained using the same recipe,
then evaluated with shared prediction and evaluation configurations.

This example trains two models for 20 epochs each and evaluates both, so it
can be demanding in runtime and memory. It loads the full STL-10 unlabeled
split despite the batch limits. Lower the epoch count or batch limits in
both training YAMLs to shorten the runs; reduce batch size if GPU memory
is insufficient.

Requires BenchRep's scverse extra for UMAP and Leiden. Run with CSV logging:

    python -m benchrep.examples.usage.07_learn_from_multiple_views

For Weights & Biases, install the optional backend
(pip install -e '.[wandb]'), run `wandb login`, then:

    python -m benchrep.examples.usage.07_learn_from_multiple_views --logger wandb

The comparison CSV contains probe scores and paths to each run's manifests.
"""

import argparse
import csv
import gc
from datetime import datetime
from pathlib import Path

from benchrep import (
    EvaluationOverrides,
    PredictionOverrides,
    TrainingOverrides,
    evaluate,
    predict_composite,
    train_composite,
)
from benchrep.assembly.config import load_yaml
from benchrep.assembly.schemas import TrainingLoggerConfig
from benchrep.examples import example_config_path


EXAMPLE = "07_learn_from_multiple_views"
OUTPUT_DIR = Path("outputs") / EXAMPLE
VARIANTS = (
    ("From scratch", "training.yaml"),
    ("Pretrained", "training_pretrained.yaml"),
)
PROBES = ("dummy", "linear", "knn", "random_forest")


def run_variant(
    initialization: str,
    training_yaml: str,
    *,
    logger: str,
    comparison_id: str,
) -> list[dict]:
    print(f"\nRunning: {initialization}")
    overrides = None
    wandb_backend = None
    if logger == "wandb":
        import wandb as wandb_backend

        logger_path = example_config_path(f"{EXAMPLE}/wandb_logger.yaml")
        logger_config = TrainingLoggerConfig.model_validate(load_yaml(logger_path))
        # Group this comparison together, with a separate run for each variant.
        logger_config.params.update(
            name=initialization,
            group=f"{logger_config.params['group']}-{comparison_id}",
        )
        overrides = TrainingOverrides(config_components={"logger": logger_config})

    training_exit_code = 1
    try:
        training_result = train_composite(
            config=example_config_path(f"{EXAMPLE}/{training_yaml}"),
            overrides=overrides,
        )
        if training_result.status_report.interruption_signal is not None:
            raise KeyboardInterrupt("Training interrupted.")
        training_manifest_path = training_result.manifest_path
        training_exit_code = 0
    finally:
        # W&B otherwise keeps its active run available to later training calls.
        if wandb_backend is not None:
            wandb_backend.finish(exit_code=training_exit_code)

    # Repeated workflows share this process. Discard results and collect
    # unreachable objects before constructing the next model and datamodule.
    del training_result
    gc.collect()

    # Export fixed-view encoder embeddings, before the contrastive MLP head.
    prediction_result = predict_composite(
        config=example_config_path(EXAMPLE, stage="prediction"),
        overrides=PredictionOverrides(
            training_manifest_path=training_manifest_path,
        ),
    )
    prediction_manifest_path = prediction_result.manifest_path
    del prediction_result
    gc.collect()

    # The shared YAML keeps probe splits and evaluation settings identical.
    evaluation_result = evaluate(
        config=example_config_path(EXAMPLE, stage="evaluation"),
        overrides=EvaluationOverrides(
            prediction_manifest_path=prediction_manifest_path,
        ),
    )

    rows = []
    for probe in PROBES:
        metric_path = ("predictability", "label", "probes", probe)
        mean_score = evaluation_result.get_metric(*metric_path, "mean_score")
        std_score = evaluation_result.get_metric(*metric_path, "std_score")
        rows.append({
            "initialization": initialization,
            "probe": probe,
            "mean_balanced_accuracy": mean_score,
            "fold_std": std_score,
            "training_manifest": str(training_manifest_path),
            "prediction_manifest": str(prediction_manifest_path),
            "evaluation_manifest": str(evaluation_result.manifest_path),
        })
        print(f"{initialization} / {probe}: {mean_score:.1%} (fold SD {std_score:.1%})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--logger", choices=("csv", "wandb"), default="csv",
        help="Training logger (default: csv).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    comparison_path = OUTPUT_DIR / f"comparison_{timestamp}.csv"
    rows = []
    for initialization, training_yaml in VARIANTS:
        rows.extend(run_variant(
            initialization, training_yaml,
            logger=args.logger, comparison_id=timestamp,
        ))
        gc.collect()
        # Keep the first result available if the second run is interrupted.
        with comparison_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Comparison saved to {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
