from __future__ import annotations

import os

from pathlib import Path

import warnings

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
from benchrep.assembly.registry import LOGGERS
from benchrep.assembly.schemas import TrainerConfig, LoggerConfig, CheckpointConfig
from benchrep.runtime import RunContext


def build_trainer(
        trainer_config: TrainerConfig,
        logger_config: LoggerConfig | None,
        checkpoint_config: CheckpointConfig,
        run_context: RunContext,
) -> tuple[L.Trainer, ModelCheckpoint]:
    run_log = get_run_logger()

    trainer_params = trainer_config.model_dump()

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
        checkpoint_dir=run_context.checkpoint_dir,
    )

    trainer = L.Trainer(
        default_root_dir=str(run_context.output_dir),
        logger=logger,
        callbacks=[checkpoint_callback],
        **trainer_params,
    )

    run_log.info(
        "Built Lightning trainer: (max_epochs=%s, logger=%s -> %s, "
        "checkpoint_monitor=%s, save_top_k=%s, save_last=%s)",
        trainer_config.max_epochs,
        logger_name,
        logger_cls_name,
        checkpoint_config.monitor,
        checkpoint_config.save_top_k,
        checkpoint_config.save_last,
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