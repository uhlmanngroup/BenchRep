from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torchvision.utils import save_image

from benchrep.assembly import load_config
from benchrep.assembly.builders import build_datamodule, build_model


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    seed = config.get("seed", 137)
    L.seed_everything(seed, workers=True)

    output_dir = Path(config["outputs"]["root_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(config["data"])
    model = build_model(config)

    trainer = build_trainer(config=config, output_dir=output_dir)
    trainer.fit(model, datamodule=datamodule)

    export_reconstructions(
        model=model,
        datamodule=datamodule,
        output_path=output_dir / "reconstructions.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BenchRep model from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def build_trainer(config: dict[str, Any], output_dir: Path) -> L.Trainer:
    trainer_config = dict(config.get("trainer", {}))

    return L.Trainer(
        default_root_dir=str(output_dir),
        **trainer_config,
    )


def export_reconstructions(
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    output_path: Path,
    n_images: int = 16,
) -> None:
    datamodule.setup("fit")

    val_loader = datamodule.val_dataloader()
    if val_loader is None:
        raise RuntimeError(
            "Could not export reconstructions because no validation dataloader is available."
        )

    batch = next(iter(val_loader))
    x = batch["x"].to(model.device)

    model.eval()
    with torch.no_grad():
        reconstruction = model(x)

    comparison = torch.cat(
        [
            x[:n_images].cpu(),
            reconstruction[:n_images].cpu(),
        ],
        dim=0,
    )

    save_image(
        comparison,
        output_path,
        nrow=n_images,
    )

    print(f"Saved reconstruction grid to: {output_path}")


if __name__ == "__main__":
    main()