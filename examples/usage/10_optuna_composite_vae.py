"""Tune a composite VAE with Optuna.

Install: pip install -e '.[optuna]'

This example demonstrates how Optuna can be used as a third-party tool
to tune various hyperparameters in a BenchRep composite VAE model.

Specifically, Optuna optimizes validation clustering AMI by minimizing its negative.
Training-time pruning uses validation reconstruction MSE to discourage
poor reconstruction while keeping clustering quality as the final objective.
"""

import math
from pathlib import Path
from datetime import datetime

try:
    import optuna
    from optuna_integration import PyTorchLightningPruningCallback
except ImportError as exc:
    raise ImportError(
        "This example requires BenchRep's optional Optuna dependencies. "
        "Install them with: pip install -e '.[optuna]'"
    ) from exc

from benchrep import (
    EvaluationOverrides,
    PredictionOverrides,
    evaluate,
    predict_composite,
    train_composite,
    update_config,
)
from benchrep.examples import example_config_path, load_example_config
from benchrep.records import print_divider
from benchrep.architecture.data import BenchRepDataModule


EXAMPLE = "10_optuna_composite_vae"
OUTPUT_DIR = Path("outputs/optuna_composite_vae") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PRUNING_METRIC = "losses/val/reconstruction/mse"
SEED = 137
N_TRIALS = 3

BASE_CONFIG = load_example_config(EXAMPLE, stage="training")


def run_trial(trial: optuna.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    beta = trial.suggest_float("beta", 1e-4, 1.0, log=True)
    latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 64])
    hidden_width = trial.suggest_categorical("hidden_width", [64, 128, 256])

    # Update config with new hyperparameters.
    components = "composite_model_components"
    config = update_config(BASE_CONFIG, {
        "run.output_root": OUTPUT_DIR / f"trial_{trial.number:04d}",
        "optimizer.params.lr": learning_rate,
        "losses.regularization.0.weight": beta,
        f"{components}.image_encoder.params.hidden_dims": [hidden_width],
        f"{components}.image_decoder.params.hidden_dims": [hidden_width],
        f"{components}.variational_head.params.latent_dim": latent_dim,
        f"{components}.image_decoder.params.input_dim": latent_dim,
        # Deliberately fail trial 0 to demonstrate error capture.
        "datamodule.num_workers": (
            0 if trial.number == 0 else BASE_CONFIG.datamodule.num_workers
        ),
        "datamodule.persistent_workers": (
            True if trial.number == 0 else BASE_CONFIG.datamodule.persistent_workers
        ),
    })

    # Optuna can prune during training based on validation reconstruction MSE.
    training_result = train_composite(
        config=config,
        runtime_callbacks=[
            PyTorchLightningPruningCallback(trial, monitor=PRUNING_METRIC),
        ],
    )

    # Allow SIGINT/SIGKILL to terminate the whole study.
    if training_result.status_report.interruption_signal is not None:
        raise KeyboardInterrupt("Training interrupted.")

    # Score the checkpoint on the validation split created during training
    # instead of the test split of the dataset.
    assert isinstance(training_result.datamodule, BenchRepDataModule)
    validation_dataset = training_result.datamodule.val_dataset

    prediction_result = predict_composite(
        config=example_config_path(EXAMPLE, stage="prediction"),
        overrides=PredictionOverrides(
            training_manifest_path=training_result.manifest_path,
            datamodule=BenchRepDataModule(
                predict_dataset=validation_dataset,
                val_fraction=0.0,
                batch_size=128,
                num_workers=0,
                seed=SEED,
            ),
        ),
    )

    evaluation_result = evaluate(
        config=example_config_path(EXAMPLE, stage="evaluation"),
        overrides=EvaluationOverrides(
            prediction_manifest_path=prediction_result.manifest_path,
        ),
    )

    # Use the metric mapping inferred from the exported metrics.json.
    ami = float(evaluation_result.get_metric(
        "clustering", "external", "kmeans", "metrics", "adjusted_mutual_info", "value"
    ))
    if not math.isfinite(ami):
        raise ValueError(f"Evaluation returned non-finite AMI: {ami}")

    return -ami


def save_trials_csv(
    study: optuna.Study,
    _trial: optuna.trial.FrozenTrial,
) -> None:
    # Save study summary after each trial.
    study.trials_dataframe().to_csv(OUTPUT_DIR / "trials.csv", index=False)


def objective(trial: optuna.Trial) -> float:
    label = f"trial {trial.number}"

    # Include the trial's output location in the exported CSV.
    trial.set_user_attr(
        "output_dir", str(OUTPUT_DIR / f"trial_{trial.number:04d}")
    )

    print_divider(label, major=True)
    try:
        return run_trial(trial)

    except optuna.TrialPruned:
        # Pruning is an expected outcome.
        # Re-raise so Optuna marks the trial as PRUNED.
        raise

    except Exception as exc:
        # Optuna logs errors but doesn't automatically include them in the CSV.
        # User attributes become columns in study.trials_dataframe().
        trial.set_user_attr("error_type", type(exc).__name__)
        trial.set_user_attr("error_message", str(exc))

        # Let Optuna mark the trial as FAIL. The study continues because
        # study.optimize() uses catch=(Exception,).
        raise

    finally:
        print_divider(label, end=True, major=True)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            gc_after_trial=True,
            catch=(Exception,),
            callbacks=[save_trials_csv],
        )
    finally:
        csv_path = OUTPUT_DIR / "trials.csv"
        study.trials_dataframe().to_csv(csv_path, index=False)
        print(f"Trial summary saved to {csv_path}")

    if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation AMI: {-study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")