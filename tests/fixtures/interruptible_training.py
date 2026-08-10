from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback

from benchrep.assembly.registries.core import DATASETS, CALLBACKS
from benchrep.assembly.schemas import parse_training_config
from benchrep.workflows import train_ae
from tests.fixtures.datasets import TinySyntheticDataset


class PauseAfterFirstTrainingBatch(Callback):
    """Expose a window for sending a signal after one optimizer step."""

    def __init__(
            self,
            *,
            training_started_path: str | Path,
            release_path: str | Path,
    ) -> None:
        self.training_started_path = Path(training_started_path)
        self.release_path = Path(release_path)
        self._pause_completed = False

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._pause_completed:
            return

        self._pause_completed = True
        self.training_started_path.touch()

        while not self.release_path.exists():
            time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-started-path", type=Path, required=True)
    parser.add_argument("--release-path", type=Path, required=True)
    parser.add_argument("--workflow-returned-path", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    raw_config["run"]["output_root"] = str(args.output_root)
    raw_config["dataset"]["params"]["n_samples"] = 128
    raw_config["trainer"]["max_epochs"] = 100
    raw_config["additional_callbacks"] = [
        {
            "name": "pause_after_first_training_batch",
            "params": {
                "training_started_path": str(
                    args.training_started_path
                ),
                "release_path": str(args.release_path),
            },
        }
    ]

    DATASETS.register("tiny_synthetic", TinySyntheticDataset)
    CALLBACKS.register(
        "pause_after_first_training_batch",
        PauseAfterFirstTrainingBatch,
    )

    result = train_ae(
        full_config_object=parse_training_config(raw_config),
    )

    # Written only after the workflow returns through all current finalization.
    args.workflow_returned_path.write_text(
        str(result.manifest_path),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()