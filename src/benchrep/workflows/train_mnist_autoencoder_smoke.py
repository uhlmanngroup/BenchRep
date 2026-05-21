from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable, Callable

import lightning as L
import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image

from benchrep.architecture.data import DataModule, MNISTDataset
from benchrep.architecture.decoders import MLPDecoder
from benchrep.architecture.encoders import MLPEncoder
from benchrep.architecture.losses import MSEReconstructionLoss
from benchrep.architecture.models import Autoencoder


def main() -> None:
    seed = 137
    L.seed_everything(seed, workers=True)

    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "examples" / "data" / "mnist"
    output_dir = repo_root / "outputs" / "smoke_mnist"
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    train_dataset = MNISTDataset(
        root=str(data_root),
        train=True,
        transform=transform,
        download=True,
    )

    datamodule = DataModule(
        train_dataset=train_dataset,
        batch_size=128,
        val_fraction=0.1,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        seed=seed,
    )

    encoder = MLPEncoder(
        input_shape=(1, 28, 28),
        latent_dim=32,
        hidden_dims=(512, 256),
        normalization=None,
        dropout=0.0,
    )

    decoder = MLPDecoder(
        latent_dim=encoder.latent_dim,
        output_shape=encoder.input_shape,
        hidden_dims=(256, 512),
        normalization=None,
        dropout=0.0,
    )

    reconstruction_loss = MSEReconstructionLoss()

    optimizer_factory: Callable[
        [Iterable[nn.Parameter]], torch.optim.Optimizer
    ] = lambda params: torch.optim.Adam(params, lr=1e-3)

    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        reconstruction_loss=reconstruction_loss,
        optimizer_factory=optimizer_factory,
    )

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        default_root_dir=str(output_dir),
        log_every_n_steps=20,
    )

    trainer.fit(model, datamodule=datamodule)

    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()
    if val_loader is None:
        raise RuntimeError("Expected a validation dataloader for smoke reconstruction export.")

    batch = next(iter(val_loader))
    x = batch["x"].to(model.device)

    model.eval()
    with torch.no_grad():
        reconstruction = model(x)

    comparison = torch.cat(
        [
            x[:16].cpu(),
            reconstruction[:16].cpu(),
        ],
        dim=0,
    )

    save_image(
        comparison,
        output_dir / "mnist_reconstructions.png",
        nrow=16,
    )

    print(f"Saved reconstruction grid to: {output_dir / 'mnist_reconstructions.png'}")


if __name__ == "__main__":
    main()