from __future__ import annotations

import os

from pathlib import Path

import warnings

from typing import Literal

import lightning as L

from lightning.pytorch.loggers import (
    Logger,
    CSVLogger,
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
)

from lightning.pytorch.callbacks import ModelCheckpoint

from benchrep.records import get_run_logger
from benchrep.assembly.registries.core import LOGGERS
from benchrep.assembly.schemas import TrainerConfig, LoggerConfig, CheckpointConfig
from benchrep.runtime import RunContext


def build_trainer(
        *,
        trainer_config: TrainerConfig,
        stage: Literal["training", "prediction"],
        run_context: RunContext,
        logger_config: LoggerConfig | None = None,
        checkpoint_config: CheckpointConfig | None = None,
        max_batches: int | None = None,
) -> tuple[L.Trainer, ModelCheckpoint | None]:
    """Build a Lightning Trainer for a BenchRep workflow stage.

    This is the public Trainer builder for BenchRep workflows. It translates a
    validated ``TrainerConfig`` into a Lightning ``Trainer`` while enforcing the
    Trainer arguments that BenchRep owns internally.

    The ``stage`` argument controls stage-specific Trainer behavior. During
    training, this builder attaches the configured Lightning logger and creates a
    BenchRep-managed ``ModelCheckpoint`` callback from the top-level
    ``checkpointing`` config. During prediction, this builder disables Lightning
    logging and checkpointing, ignores any provided training logger/checkpoint
    config with a warning, and optionally maps ``max_batches`` to Lightning's
    ``limit_predict_batches``.

    The following Lightning Trainer arguments are intentionally not allowed in
    ``trainer_config`` because BenchRep manages them directly:
    ``default_root_dir``, ``logger``, ``callbacks``, and
    ``enable_checkpointing``.

    Parameters
    ----------
    trainer_config:
        Validated Trainer config. Its fields and extra allowed values are passed
        through to ``lightning.Trainer`` after BenchRep-owned arguments are
        checked and stage-specific overrides are applied.
    stage:
        Workflow stage for which the Trainer is being built. Supported values are
        ``"training"`` and ``"prediction"``.
    run_context:
        Run context providing BenchRep-managed output paths. Its output directory
        is used as Lightning's ``default_root_dir``.
    logger_config:
        Optional top-level logger config. Used during training. Ignored with a
        warning during prediction.
    checkpoint_config:
        Optional top-level checkpoint config. Required during training. Ignored
        with a warning during prediction.
    max_batches:
        Optional prediction diagnostic limit. During prediction, this is mapped
        to Lightning's ``limit_predict_batches``. Ignored during training.

    Returns
    -------
    tuple[lightning.Trainer, ModelCheckpoint | None]
        The instantiated Lightning Trainer and, for training runs, the
        ModelCheckpoint callback. Prediction runs return ``None`` for the
        checkpoint callback.

    Raises
    ------
    ValueError
        If BenchRep-owned Trainer arguments are provided through
        ``trainer_config``, if training is requested without a checkpoint config,
        or if an unsupported workflow stage is requested.
    """
    run_log = get_run_logger()

    trainer_params = trainer_config.model_dump(exclude_none=True)

    if "default_root_dir" in trainer_params:
        raise ValueError(
            "`trainer.default_root_dir` should not be set in the trainer config. "
            "BenchRep manages the trainer root directory through RunContext."
        )

    if "logger" in trainer_params:
        raise ValueError(
            "`trainer.logger` should not be set in the trainer config. "
            "Use the top-level `logger` config section instead."
        )

    if "callbacks" in trainer_params:
        raise ValueError(
            "`trainer.callbacks` should not be set in the trainer config yet. "
            "BenchRep currently manages required callbacks internally."
        )

    if "enable_checkpointing" in trainer_params:
        raise ValueError(
            "`trainer.enable_checkpointing` should not be set in the trainer config. "
            "BenchRep manages checkpointing through the top-level `checkpointing` config "
            "during training and disables checkpointing during prediction."
        )

    if stage == "training":
        if checkpoint_config is None:
            raise ValueError("Training requires checkpoint_config.")

        logger = _build_logger(logger_config)

        if logger_config is None:
            logger_name = None
            logger_cls_name = "False"
        else:
            logger_name = logger_config.name
            logger_cls = LOGGERS.get(logger_name)
            logger_cls_name = logger_cls.__name__

        checkpoint_callback = _build_checkpoint_callback(
            checkpoint_config=checkpoint_config,
            checkpoint_dir=run_context.training_checkpoint_dir,
        )

        trainer = L.Trainer(
            default_root_dir=str(run_context.output_dir),
            logger=logger,
            callbacks=[checkpoint_callback],
            **trainer_params,
        )

        run_log.info(
            "Built Lightning trainer for training: "
            "(max_epochs=%s, accelerator=%s, devices=%s, precision=%s, "
            "deterministic=%s, logger=%s -> %s, checkpoint_monitor=%s, "
            "save_top_k=%s, save_last=%s)",
            trainer_params.get("max_epochs"),
            trainer_params.get("accelerator"),
            trainer_params.get("devices"),
            trainer_params.get("precision"),
            trainer_params.get("deterministic"),
            logger_name,
            logger_cls_name,
            checkpoint_config.monitor,
            checkpoint_config.save_top_k,
            checkpoint_config.save_last,
        )
    elif stage == "prediction":
        if logger_config is not None:
            run_log.warning(
                "Prediction received logger_config, but prediction logging is currently "
                "disabled by BenchRep. Ignoring logger_config."
            )

        if checkpoint_config is not None:
            run_log.warning(
                "Prediction received checkpoint_config, but prediction does not create "
                "training checkpoints. Ignoring checkpoint_config."
            )

        logger = False
        checkpoint_callback = None
        trainer_params["enable_checkpointing"] = False

        if max_batches is not None:
            trainer_params["limit_predict_batches"] = max_batches

        trainer = L.Trainer(
            default_root_dir=str(run_context.output_dir),
            logger=logger,
            **trainer_params,
        )

        run_log.info(
            "Built Lightning trainer for prediction: "
            "(accelerator=%s, devices=%s, precision=%s, deterministic=%s, "
            "limit_predict_batches=%s, logger=False, enable_checkpointing=False)",
            trainer_params.get("accelerator"),
            trainer_params.get("devices"),
            trainer_params.get("precision"),
            trainer_params.get("deterministic"),
            trainer_params.get("limit_predict_batches"),
        )

    else:
        raise ValueError(
            f"Unsupported stage {stage!r}. "
            "Available options: 'training', 'prediction'."
        )

    return trainer, checkpoint_callback


def _build_logger(logger_config: LoggerConfig | None) -> Logger | bool:
    """Build a Lightning logger from a BenchRep logger config.

    `logger_config.params` is passed directly to the selected Lightning logger.
    `credential_path` is BenchRep-owned and is handled before instantiation.
    """
    if logger_config is None:
        return False

    requested_name = logger_config.name
    logger_cls = LOGGERS.get(requested_name)

    _prepare_logger_credentials(logger_cls, logger_config)

    logger_params = dict(logger_config.params)

    try:
        return logger_cls(**logger_params)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate logger from config name {requested_name!r}. "
            f"Resolved logger class: {logger_cls.__module__}.{logger_cls.__name__}. "
            "This often means `logger.params` contains an invalid keyword argument "
            "for the selected Lightning logger class."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to instantiate logger from config name {requested_name!r}. "
            f"Resolved logger class: {logger_cls.__module__}.{logger_cls.__name__}. "
            "This may be due to missing optional dependencies, authentication/login "
            "state, tracking URI/service configuration, or backend-specific settings. "
            "Check the backend documentation for auth via login, environment "
            "variables, credential files, or logger-specific config."
        ) from exc


def _prepare_logger_credentials(
    logger_cls: type[Logger],
    logger_config: LoggerConfig,
) -> None:
    if logger_config.credential_path is None:
        return

    if logger_cls is WandbLogger:
        credential_path = logger_config.credential_path.expanduser()

        if not credential_path.is_file():
            raise FileNotFoundError(
                f"W&B credential file does not exist: {credential_path}"
            )

        api_key = credential_path.read_text().strip()

        if not api_key:
            raise ValueError(f"W&B credential file is empty: {credential_path}")

        os.environ["WANDB_API_KEY"] = api_key
        return

    if logger_cls in {CSVLogger, TensorBoardLogger}:
        warnings.warn(
            f"`credential_path` was provided for {logger_cls.__name__}, "
            "but this logger does not use credentials. Ignoring it.",
            UserWarning,
            stacklevel=2,
        )
        return

    if logger_cls is MLFlowLogger:
        raise ValueError(
            "`credential_path` is not yet supported for MLFlowLogger. "
            "Pass MLflow auth/tracking settings through environment variables "
            "or `logger.params` instead."
        )


def _build_checkpoint_callback(
    checkpoint_config: CheckpointConfig,
    checkpoint_dir: Path,
) -> ModelCheckpoint:
    if checkpoint_config.monitor is None:
        return ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=None,
            save_top_k=0,
            save_last=checkpoint_config.save_last,
        )
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_config.filename,
        monitor=checkpoint_config.monitor,
        mode=checkpoint_config.mode,
        save_top_k=checkpoint_config.save_top_k,
        save_last=checkpoint_config.save_last,
    )