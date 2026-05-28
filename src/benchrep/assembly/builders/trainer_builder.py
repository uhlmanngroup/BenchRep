from __future__ import annotations

import os

import warnings

import lightning as L

from lightning.pytorch.loggers import (
    Logger,
    CSVLogger,
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
)

from benchrep.records import get_run_logger
from benchrep.assembly.registry import LOGGERS
from benchrep.assembly.schemas import TrainerConfig, LoggerConfig
from benchrep.runtime import RunContext


def build_trainer(
        trainer_config: TrainerConfig,
        logger_config: LoggerConfig,
        run_context: RunContext) -> L.Trainer:
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

    logger_name = logger_config.name
    logger_cls = LOGGERS.get(logger_name)
    logger = _build_logger(logger_config)

    trainer = L.Trainer(
        default_root_dir=str(run_context.output_dir),
        logger=logger,
        **trainer_params,
    )
    run_log.info("Built Lightning trainer: (max_epochs=%s, logger= %s -> %s)",
                 trainer_config.max_epochs,
                 logger_name,
                 logger_cls.__name__)

    return trainer


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