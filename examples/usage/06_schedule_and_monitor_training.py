"""Monitor how reconstruction and KL regularization influence a VAE's encoder.

Train on MNIST with a local CSV logger:

    python -m benchrep.examples.usage.06_schedule_and_monitor_training

To follow the same metrics live in Weights & Biases, install the optional
backend (from a checkout: pip install -e '.[wandb]'), run `wandb login`,
and select the alternate logger:

    python -m benchrep.examples.usage.06_schedule_and_monitor_training --logger wandb

The KL coefficient rises over epochs 0–3. Adam's learning rate drops at
epoch 5. Compare the weighted gradient norms to see how strongly each loss
acts on the encoder. The combined gradient also depends on their directions.
"""

import argparse
from pathlib import Path

import pandas as pd
from lightning.pytorch.loggers import CSVLogger

from benchrep import TrainingOverrides, train_vae
from benchrep.assembly.config import load_yaml
from benchrep.assembly.schemas import TrainingLoggerConfig
from benchrep.examples import example_config_path


EXAMPLE = "06_schedule_and_monitor_training"
ENCODER_GRADIENTS = "grad_norms_wrt_encoder"
MSE_GRADIENT = f"{ENCODER_GRADIENTS}/per_loss/reconstruction/mse/l2/weighted"
KL_GRADIENT = f"{ENCODER_GRADIENTS}/per_loss/regularization/gaussian_kl/l2/weighted"


def show_csv_summary(metrics_path: Path) -> None:
    columns = {
        "lr/group_0": "lr",
        "loss_weights/train/regularization/gaussian_kl": "KL weight",
        MSE_GRADIENT: "MSE grad",
        KL_GRADIENT: "KL grad",
        f"{ENCODER_GRADIENTS}/optimizer/l2": "combined grad",
    }
    history = pd.read_csv(metrics_path)
    # Validation and epoch summaries are logged on separate CSV rows.
    training_steps = history.dropna(subset=list(columns))
    last_steps = training_steps.drop_duplicates("epoch", keep="last")
    summary = last_steps[["epoch", "step", *columns]].rename(columns=columns)
    print("\nEncoder gradient norms at the last logged training step of each epoch")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4g}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logger", choices=("csv", "wandb"), default="csv")
    args = parser.parse_args()

    overrides = None
    if args.logger == "wandb":
        logger_path = example_config_path(f"{EXAMPLE}/wandb_logger.yaml")
        wandb_logger = TrainingLoggerConfig.model_validate(load_yaml(logger_path))
        overrides = TrainingOverrides(config_components={"logger": wandb_logger})

    training_result = train_vae(
        config=example_config_path(EXAMPLE, stage="training"),
        overrides=overrides,
    )
    print(f"Training manifest: {training_result.manifest_path}")

    logger = training_result.trainer.logger
    if isinstance(logger, CSVLogger):
        metrics_path = Path(logger.log_dir) / "metrics.csv"
        if metrics_path.is_file():
            print(f"Full metric history: {metrics_path}")
            if training_result.status_report.interruption_signal is None:
                show_csv_summary(metrics_path)
    else:
        print(f"Compare these two series in your W&B workspace:\n  {MSE_GRADIENT}\n  {KL_GRADIENT}")


if __name__ == "__main__":
    main()
