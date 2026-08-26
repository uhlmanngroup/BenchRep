"""Run a comprehensive standalone evaluation from static artifacts.

Run from the repository root:

    python examples/usage/09_standalone_evaluation_workflow.py

This example demonstrates that BenchRep evaluation can run independently of
its training and prediction workflows. It evaluates an AnnData embedding
artifact and reconstruction bundle directly, without a prediction manifest.
The same interface can therefore evaluate externally produced artifacts that
satisfy BenchRep's input contracts.

The artifacts contain predictions from a VAE trained on an enriched MNIST
dataset. Each digit has two image channels: its original morphology and a
skeletonized representation. The dataset also records digit identity, estimated
mean stroke width, and a simulated acquisition batch with batch-specific blur,
noise, intensity, and directional background effects.

MNIST was enriched to demonstrate the evaluation workflow's broad analytical
capabilities. The YAML intentionally covers virtually every supported
algorithm, analysis, metric, output format, etc. and explicitly lists
supported configuration fields even where omission defaults would suffice.
It also demonstrates configuration fields that accept additional keyword
arguments passed through to the underlying backends.

The complete embeddings are retained for reduction, clustering, embedding
metrics, and predictability analysis. Only a selected set of inputs and
reconstructions is retained for reconstruction metrics and visualizations,
keeping the committed artifacts reasonably small.

The dataset definition and artifact-generation workflow are available under
``examples/artifact_generation/09_standalone_evaluation_workflow``.
"""

from pathlib import Path

import anndata as ad
import torch

from benchrep import evaluate
from benchrep.assembly.schemas import EvaluationSourceConfig


EXAMPLES_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = (
    EXAMPLES_DIR
    / "configs"
    / "09_standalone_evaluation_workflow"
)
ASSET_DIR = (
    EXAMPLES_DIR
    / "assets"
    / "09_standalone_evaluation_workflow"
)

CONFIG_PATH = CONFIG_DIR / "evaluation.yaml"
EMBEDDINGS_PATH = ASSET_DIR / "embeddings.h5ad"
RECONSTRUCTION_BUNDLE_DIR = ASSET_DIR / "reconstruction_bundle"


def describe_evaluation_inputs() -> None:
    """Print the standalone evaluation artifact contracts."""

    embeddings = ad.read_h5ad(EMBEDDINGS_PATH)
    inputs = torch.load(
        RECONSTRUCTION_BUNDLE_DIR / "input.pt",
        map_location="cpu",
        weights_only=False,
    )
    reconstructions = torch.load(
        RECONSTRUCTION_BUNDLE_DIR / "reconstruction.pt",
        map_location="cpu",
        weights_only=False,
    )
    observations = torch.load(
        RECONSTRUCTION_BUNDLE_DIR / "obs.pt",
        map_location="cpu",
        weights_only=False,
    )
    metadata = torch.load(
        RECONSTRUCTION_BUNDLE_DIR
        / "reconstruction_export_metadata.pt",
        map_location="cpu",
        weights_only=False,
    )

    print("\n=== Standalone evaluation inputs ===")
    print(
        "Embedding AnnData: "
        f"X shape={embeddings.X.shape}; "
        f"obs columns={tuple(embeddings.obs.columns)}; "
        f"obsm keys={tuple(embeddings.obsm.keys())}"
    )
    print(
        "Reconstruction bundle: "
        f"input shape={tuple(inputs.shape)}; "
        f"reconstruction shape={tuple(reconstructions.shape)}; "
        f"obs fields={tuple(observations)}; "
        f"channels={tuple(metadata['channel_names'])}"
    )


def main() -> None:
    describe_evaluation_inputs()

    print("\n=== Evaluation ===")
    # The YAML leaves its source paths unset so this example can demonstrate
    # supplying standalone evaluation artifacts through a config override.
    evaluation_result = evaluate(
        config_path=CONFIG_PATH,
        config_components={
            "source": EvaluationSourceConfig(
                embeddings_path=EMBEDDINGS_PATH,
                reconstructions_path=RECONSTRUCTION_BUNDLE_DIR,
            ),
        },
    )

    print(f"Evaluation status:   {evaluation_result.status_report.status}")
    print(f"Evaluation outputs:  {evaluation_result.run_context.output_dir}")
    print(f"Evaluation manifest: {evaluation_result.manifest_path}")


if __name__ == "__main__":
    main()