"""Generate the static artifacts used by Example 09."""

from pathlib import Path
import shutil
from typing import Final

import torch

from enriched_mnist import CHANNEL_NAMES, EnrichedMNISTDataset

from benchrep import predict_vae, train_vae
from benchrep.assembly.registries import DATASETS
from benchrep.workflows import PredictionWorkflowResult


EXAMPLES_DIR: Final = Path(__file__).resolve().parents[2]
CONFIG_DIR: Final = EXAMPLES_DIR / "configs" / "09_standalone_evaluation_workflow"
ASSET_DIR: Final = (
    EXAMPLES_DIR / "assets" / "09_standalone_evaluation_workflow"
)
RECONSTRUCTION_BUNDLE_DIR: Final = ASSET_DIR / "reconstruction_bundle"


def publish_artifacts(
    prediction_result: PredictionWorkflowResult,
) -> None:
    anndata_export = prediction_result.export_result.anndata
    reconstruction_export = (
        prediction_result.export_result.reconstructions
    )

    if anndata_export.path is None:
        raise RuntimeError(
            "Prediction did not produce the required AnnData artifact."
        )

    if len(reconstruction_export.pairs) != 1:
        raise RuntimeError(
            "Prediction must produce exactly one reconstruction bundle "
            "for this artifact-generation workflow."
        )

    reconstruction_paths = reconstruction_export.pairs[0].paths

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RECONSTRUCTION_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        anndata_export.path,
        ASSET_DIR / "anndata.h5ad",
    )

    reconstruction_sources = {
        "input.pt": reconstruction_paths.input_path,
        "reconstruction.pt": reconstruction_paths.reconstruction_path,
        "obs.pt": reconstruction_paths.obs_path,
    }

    for filename, source_path in reconstruction_sources.items():
        if source_path is None:
            raise RuntimeError(f"Prediction did not produce {filename}.")

        shutil.copy2(
            source_path,
            RECONSTRUCTION_BUNDLE_DIR / filename,
        )

    if reconstruction_paths.metadata_path is None:
        raise RuntimeError(
            "Prediction did not produce reconstruction export metadata."
        )

    reconstruction_metadata = torch.load(
        reconstruction_paths.metadata_path,
        map_location="cpu",
        weights_only=False,
    )
    reconstruction_metadata["channel_names"] = list(CHANNEL_NAMES)

    torch.save(
        reconstruction_metadata,
        RECONSTRUCTION_BUNDLE_DIR
        / "reconstruction_export_metadata.pt",
    )


def main() -> None:
    DATASETS.register(
        "enriched_mnist",
        EnrichedMNISTDataset,
    )

    print("\n=== Training ===")
    training_result = train_vae(
        config_path=CONFIG_DIR / "training.yaml",
    )
    print(f"Training manifest: {training_result.manifest_path}")

    print("\n=== Prediction ===")
    prediction_result = predict_vae(
        config_path=CONFIG_DIR / "prediction.yaml",
        training_manifest_path=training_result.manifest_path,
    )
    print(f"Prediction manifest: {prediction_result.manifest_path}")

    print("\n=== Publishing artifacts ===")
    publish_artifacts(prediction_result)
    print(f"Published artifacts: {ASSET_DIR}")


if __name__ == "__main__":
    main()