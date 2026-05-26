from __future__ import annotations

import lightning as L

from benchrep.assembly.schemas import TrainerConfig
from benchrep.runtime import RunContext


def build_trainer(trainer_config: TrainerConfig, run_context: RunContext) -> L.Trainer:
    trainer_params = trainer_config.model_dump()

    if "default_root_dir" in trainer_params:
        raise ValueError(
            "`trainer.default_root_dir` should not be set in the config. "
            "BenchRep manages the trainer root directory through RunContext."
        )

    return L.Trainer(
        default_root_dir=str(run_context.output_dir),
        **trainer_params,
    )