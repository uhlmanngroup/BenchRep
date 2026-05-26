from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from torchvision.utils import save_image

from benchrep.runtime import RunContext
from benchrep.assembly import load_config
from benchrep.assembly.schemas import parse_config
from benchrep.assembly.builders import build_datamodule, build_model, build_trainer


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    raw_config = load_config(config_path)
    config = parse_config(raw_config)

    model_name = f"{config.model.name}_{config.encoder.name}"
    if config.decoder is not None:
        model_name = f"{model_name}_{config.decoder.name}"

    run_context = RunContext.create(
        output_root=config.run.output_root,
        project_name=config.run.project_name,
        model_name=model_name,
    )

    print(f"Run outputs will be saved to: {run_context.output_dir}")

    L.seed_everything(
        config.reproducibility.seed,
        workers=config.reproducibility.seed_workers,
    )

    if config.reproducibility.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(
            config.reproducibility.float32_matmul_precision
        )

    datamodule = build_datamodule(config.data, seed=config.reproducibility.seed)
    model = build_model(config)

    trainer = build_trainer(config=config, run_context=run_context)
    trainer.fit(model, datamodule=datamodule)

    export_reconstructions(
        model=model,
        datamodule=datamodule,
        output_path=run_context.artifact_dir / "reconstructions.png",
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