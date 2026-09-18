from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pytest
import torch
from torch import nn

from benchrep.architecture.models import Autoencoder, CompositeModel, VAE
from benchrep.assembly.config import load_yaml
from benchrep.assembly.registries.core import DATASETS
from benchrep.assembly.schemas import (
    EvaluationConfig,
    PredictionConfig,
    TrainingConfig,
)
from benchrep.workflows import (
    evaluate,
    predict_ae,
    predict_composite,
    predict_vae,
    train_ae,
    train_composite,
    train_vae,
)
from tests.fixtures.datasets import TinySyntheticDataset


CONFIG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "configs"


def test_canonical_and_composite_autoencoders_are_end_to_end_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalent canonical and Composite AEs produce identical results."""
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)

    monkeypatch.chdir(tmp_path)

    canonical_training = train_ae(
        full_config_object=_make_canonical_ae_training_config(),
    )
    composite_training = train_composite(
        full_config_object=_make_composite_ae_training_config(),
    )

    _assert_completed(canonical_training, composite_training)
    _assert_ae_component_states_equal(
        canonical_training.model,
        composite_training.model,
    )

    canonical_prediction = predict_ae(
        full_config_object=_make_ae_prediction_config(),
        training_manifest_path=canonical_training.manifest_path,
    )
    composite_prediction = predict_composite(
        full_config_object=_make_ae_prediction_config(),
        training_manifest_path=composite_training.manifest_path,
    )

    _assert_completed(canonical_prediction, composite_prediction)
    _assert_ae_prediction_batches_equal(
        canonical_prediction.predictions,
        composite_prediction.predictions,
    )
    _assert_prediction_artifacts_equal(
        canonical_prediction,
        composite_prediction,
    )

    canonical_evaluation = evaluate(
        full_config_object=_make_evaluation_config(),
        prediction_manifest_path=canonical_prediction.manifest_path,
    )
    composite_evaluation = evaluate(
        full_config_object=_make_evaluation_config(),
        prediction_manifest_path=composite_prediction.manifest_path,
    )

    _assert_completed(canonical_evaluation, composite_evaluation)

    canonical_metrics = _load_metrics(canonical_evaluation)
    composite_metrics = _load_metrics(composite_evaluation)

    _assert_expected_evaluation_coverage(canonical_metrics)
    assert composite_metrics == canonical_metrics


def test_canonical_and_composite_vaes_are_end_to_end_equivalent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalent canonical and Composite VAEs produce identical results."""
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)

    monkeypatch.chdir(tmp_path)

    canonical_training = train_vae(
        full_config_object=_make_canonical_vae_training_config(),
    )
    composite_training = train_composite(
        full_config_object=_make_composite_vae_training_config(),
    )

    _assert_completed(canonical_training, composite_training)
    _assert_vae_component_states_equal(
        canonical_training.model,
        composite_training.model,
    )

    canonical_prediction = predict_vae(
        full_config_object=_make_vae_prediction_config(composite=False),
        training_manifest_path=canonical_training.manifest_path,
    )
    composite_prediction = predict_composite(
        full_config_object=_make_vae_prediction_config(composite=True),
        training_manifest_path=composite_training.manifest_path,
    )

    _assert_completed(canonical_prediction, composite_prediction)
    _assert_composite_vae_prediction_reroute(composite_prediction)
    _assert_vae_prediction_batches_equal(
        canonical_prediction.predictions,
        composite_prediction.predictions,
    )
    _assert_prediction_artifacts_equal(
        canonical_prediction,
        composite_prediction,
    )

    canonical_evaluation = evaluate(
        full_config_object=_make_evaluation_config(),
        prediction_manifest_path=canonical_prediction.manifest_path,
    )
    composite_evaluation = evaluate(
        full_config_object=_make_evaluation_config(),
        prediction_manifest_path=composite_prediction.manifest_path,
    )

    _assert_completed(canonical_evaluation, composite_evaluation)

    canonical_metrics = _load_metrics(canonical_evaluation)
    composite_metrics = _load_metrics(composite_evaluation)

    _assert_expected_evaluation_coverage(canonical_metrics)
    assert composite_metrics == canonical_metrics


def _make_canonical_ae_training_config() -> TrainingConfig:
    raw = load_yaml(CONFIG_DIR / "training_tiny_synthetic_ae.yaml")
    raw["run"]["project_name"] = "canonical_ae_equivalence"
    return TrainingConfig.model_validate(raw)


def _make_composite_ae_training_config() -> TrainingConfig:
    raw = load_yaml(CONFIG_DIR / "training_tiny_synthetic_ae.yaml")
    raw["run"]["project_name"] = "composite_ae_equivalence"
    raw["model"] = {"name": "composite"}

    encoder = raw.pop("encoder")
    decoder = raw.pop("decoder")
    decoder["params"]["input_dim"] = encoder["params"]["output_dim"]

    raw["composite_model_declarations"] = {
        "expects": {
            "x": "sample_image",
            "label": "categorical_prediction_target_scalar",
        },
        "batch_metadata": {
            "sample_id": "index",
        },
        "produces": {
            "embedding": "embedding_vector",
            "reconstruction": "reconstruction_image",
        },
    }
    raw["composite_model_components"] = {
        "encoder": {
            "kind": "encoder",
            **encoder,
        },
        "decoder": {
            "kind": "decoder",
            **decoder,
        },
    }
    raw["composite_model_assembly"] = {
        "encode": {
            "component": "encoder",
            "inputs": {"x": "expects.x"},
            "outputs": "produces.embedding",
        },
        "reconstruct": {
            "component": "decoder",
            "inputs": {"z": "produces.embedding"},
            "outputs": "produces.reconstruction",
        },
    }
    raw["losses"]["reconstruction"]["mse"]["composite_wiring"] = {
        "reconstruction": "produces.reconstruction",
        "target": "expects.x",
    }

    return TrainingConfig.model_validate(raw)


def _make_canonical_vae_training_config() -> TrainingConfig:
    raw = load_yaml(CONFIG_DIR / "training_tiny_synthetic_vae.yaml")
    raw["run"]["project_name"] = "canonical_vae_equivalence"
    return TrainingConfig.model_validate(raw)


def _make_composite_vae_training_config() -> TrainingConfig:
    raw = load_yaml(CONFIG_DIR / "training_tiny_synthetic_vae.yaml")
    raw["run"]["project_name"] = "composite_vae_equivalence"

    latent_dim = raw["model"]["params"]["latent_dim"]
    raw["model"] = {"name": "composite"}

    encoder = raw.pop("encoder")
    decoder = raw.pop("decoder")
    decoder["params"]["input_dim"] = latent_dim

    raw["composite_model_declarations"] = {
        "expects": {
            "x": "sample_image",
            "label": "categorical_prediction_target_scalar",
        },
        "batch_metadata": {
            "sample_id": "index",
        },
        "produces": {
            "encoder_features": "projection_vector",
            "z_sample": "embedding_vector",
            "z_mu": "embedding_vector",
            "z_logvar": "continuous_auxiliary_vector",
            "reconstruction": "reconstruction_image",
        },
    }
    raw["composite_model_components"] = {
        # Match the canonical builder's initialization order exactly.
        "encoder": {
            "kind": "encoder",
            **encoder,
        },
        "decoder": {
            "kind": "decoder",
            **decoder,
        },
        "variational_head": {
            "kind": "head",
            "name": "gaussian_variational",
            "params": {
                "in_features": encoder["params"]["output_dim"],
                "latent_dim": latent_dim,
            },
        },
    }
    raw["composite_model_assembly"] = {
        "encode": {
            "component": "encoder",
            "inputs": {"x": "expects.x"},
            "outputs": "produces.encoder_features",
        },
        "parameterize": {
            "component": "variational_head",
            "inputs": {"x": "produces.encoder_features"},
            "outputs": {
                "z_sample": "produces.z_sample",
                "z_mu": "produces.z_mu",
                "z_logvar": "produces.z_logvar",
            },
        },
        "reconstruct": {
            "component": "decoder",
            "inputs": {"z": "produces.z_sample"},
            "outputs": "produces.reconstruction",
        },
    }
    raw["losses"]["reconstruction"]["mse"]["composite_wiring"] = {
        "reconstruction": "produces.reconstruction",
        "target": "expects.x",
    }
    raw["losses"]["regularization"]["gaussian_kld"][
        "composite_wiring"
    ] = {
        "z_mu": "produces.z_mu",
        "z_logvar": "produces.z_logvar",
    }

    return TrainingConfig.model_validate(raw)


def _make_ae_prediction_config() -> PredictionConfig:
    raw = load_yaml(CONFIG_DIR / "prediction_tiny_synthetic.yaml")
    raw["exports"]["anndata"]["primary_key"] = "embedding"

    return PredictionConfig.model_validate(
        raw,
        context={"training_manifest_path_overridden": True},
    )


def _make_vae_prediction_config(*, composite: bool) -> PredictionConfig:
    raw = load_yaml(CONFIG_DIR / "prediction_tiny_synthetic.yaml")
    raw["exports"]["anndata"].update(
        {
            "mode": "custom",
            "keys": ["z_mu", "z_logvar", "z_sample"],
            "primary_key": "z_mu",
        }
    )

    if composite:
        raw["inference"]["composite_model_assembly_input_overrides"] = {
            "reconstruct": {
                "inputs": {"z": "produces.z_mu"},
            },
        }
    else:
        raw["inference"][
            "canonical_vae_reconstruction_latent_source"
        ] = "mean"

    return PredictionConfig.model_validate(
        raw,
        context={"training_manifest_path_overridden": True},
    )


def _make_evaluation_config() -> EvaluationConfig:
    raw = load_yaml(CONFIG_DIR / "evaluation_tiny_synthetic.yaml")

    # Keep the requested analytical coverage while avoiding plot/TIFF I/O.
    raw["plots"]["enabled"] = False
    raw["reconstruction"]["export_tiffs"] = False
    raw["reconstruction"]["n_examples"] = None

    return EvaluationConfig.model_validate(
        raw,
        context={"prediction_manifest_path_overridden": True},
    )


def _assert_ae_component_states_equal(
    canonical_model: nn.Module,
    composite_model: nn.Module,
) -> None:
    assert isinstance(canonical_model, Autoencoder)
    assert isinstance(composite_model, CompositeModel)

    _assert_module_states_equal(
        canonical_model.encoder,
        composite_model.components_by_id["encoder"],
    )
    _assert_module_states_equal(
        canonical_model.decoder,
        composite_model.components_by_id["decoder"],
    )


def _assert_vae_component_states_equal(
    canonical_model: nn.Module,
    composite_model: nn.Module,
) -> None:
    assert isinstance(canonical_model, VAE)
    assert isinstance(composite_model, CompositeModel)

    _assert_module_states_equal(
        canonical_model.encoder,
        composite_model.components_by_id["encoder"],
    )
    _assert_module_states_equal(
        canonical_model.decoder,
        composite_model.components_by_id["decoder"],
    )
    _assert_module_states_equal(
        canonical_model.variational_head,
        composite_model.components_by_id["variational_head"],
    )


def _assert_module_states_equal(
    canonical_module: nn.Module,
    composite_module: nn.Module,
) -> None:
    canonical_state = canonical_module.state_dict()
    composite_state = composite_module.state_dict()

    assert canonical_state.keys() == composite_state.keys()

    for parameter_name in canonical_state:
        torch.testing.assert_close(
            composite_state[parameter_name],
            canonical_state[parameter_name],
            rtol=0.0,
            atol=0.0,
            msg=f"State mismatch for {parameter_name!r}",
        )


def _assert_ae_prediction_batches_equal(
    canonical_predictions: list[Any],
    composite_predictions: list[Any],
) -> None:
    assert len(composite_predictions) == len(canonical_predictions)

    for canonical_batch, composite_batch in zip(
        canonical_predictions,
        composite_predictions,
        strict=True,
    ):
        assert (
            composite_batch.batch_metadata["sample_id"]
            == canonical_batch.sample_id
        )

        torch.testing.assert_close(
            composite_batch.model_inputs["label"],
            canonical_batch.label,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            composite_batch.model_inputs["x"],
            canonical_batch.input,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            composite_batch.model_outputs["embedding"],
            canonical_batch.embedding,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            composite_batch.model_outputs["reconstruction"],
            canonical_batch.reconstruction,
            rtol=0.0,
            atol=0.0,
        )


def _assert_vae_prediction_batches_equal(
    canonical_predictions: list[Any],
    composite_predictions: list[Any],
) -> None:
    assert len(composite_predictions) == len(canonical_predictions)

    for canonical_batch, composite_batch in zip(
        canonical_predictions,
        composite_predictions,
        strict=True,
    ):
        assert (
            composite_batch.batch_metadata["sample_id"]
            == canonical_batch.sample_id
        )

        torch.testing.assert_close(
            composite_batch.model_inputs["label"],
            canonical_batch.label,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            composite_batch.model_inputs["x"],
            canonical_batch.input,
            rtol=0.0,
            atol=0.0,
        )

        output_pairs = (
            ("z_sample", canonical_batch.z_sample),
            ("z_mu", canonical_batch.z_mu),
            ("z_logvar", canonical_batch.z_logvar),
            ("reconstruction", canonical_batch.reconstruction),
        )

        for output_name, canonical_output in output_pairs:
            torch.testing.assert_close(
                composite_batch.model_outputs[output_name],
                canonical_output,
                rtol=0.0,
                atol=0.0,
            )

        torch.testing.assert_close(
            canonical_batch.embedding,
            canonical_batch.z_mu,
            rtol=0.0,
            atol=0.0,
        )


def _assert_composite_vae_prediction_reroute(
    composite_prediction: Any,
) -> None:
    training_config = composite_prediction.run_spec.training_config
    training_assembly = training_config.composite_model_assembly

    assert training_assembly is not None
    assert training_assembly["reconstruct"].inputs["z"] == "produces.z_sample"

    prediction_spec = composite_prediction.run_spec.composite_model_spec
    assert prediction_spec is not None

    reconstruct_step = next(
        step
        for steps in prediction_spec.assembly_steps_by_dependency_level.values()
        for step in steps
        if step.step_id == "reconstruct"
    )

    assert reconstruct_step.inputs_from_model_outputs == {"z": "z_mu"}


def _assert_prediction_artifacts_equal(
    canonical_prediction: Any,
    composite_prediction: Any,
) -> None:
    canonical_anndata_path = canonical_prediction.export_result.anndata.path
    composite_anndata_path = composite_prediction.export_result.anndata.path

    assert canonical_anndata_path is not None
    assert composite_anndata_path is not None

    canonical_adata = ad.read_h5ad(canonical_anndata_path)
    composite_adata = ad.read_h5ad(composite_anndata_path)

    np.testing.assert_array_equal(composite_adata.X, canonical_adata.X)
    np.testing.assert_array_equal(
        composite_adata.obs["label"].to_numpy(),
        canonical_adata.obs["label"].to_numpy(),
    )
    assert composite_adata.obs_names.tolist() == canonical_adata.obs_names.tolist()
    assert set(composite_adata.obsm) == set(canonical_adata.obsm)

    for key in canonical_adata.obsm:
        np.testing.assert_array_equal(
            composite_adata.obsm[key],
            canonical_adata.obsm[key],
        )

    canonical_pairs = canonical_prediction.export_result.reconstructions.pairs
    composite_pairs = composite_prediction.export_result.reconstructions.pairs

    assert len(canonical_pairs) == len(composite_pairs) == 1

    canonical_paths = canonical_pairs[0].paths
    composite_paths = composite_pairs[0].paths

    assert canonical_paths.input_path is not None
    assert composite_paths.input_path is not None
    assert canonical_paths.reconstruction_path is not None
    assert composite_paths.reconstruction_path is not None
    assert canonical_paths.obs_path is not None
    assert composite_paths.obs_path is not None

    canonical_inputs = torch.load(
        canonical_paths.input_path,
        map_location="cpu",
        weights_only=False,
    )
    composite_inputs = torch.load(
        composite_paths.input_path,
        map_location="cpu",
        weights_only=False,
    )
    canonical_reconstructions = torch.load(
        canonical_paths.reconstruction_path,
        map_location="cpu",
        weights_only=False,
    )
    composite_reconstructions = torch.load(
        composite_paths.reconstruction_path,
        map_location="cpu",
        weights_only=False,
    )

    torch.testing.assert_close(
        composite_inputs,
        canonical_inputs,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        composite_reconstructions,
        canonical_reconstructions,
        rtol=0.0,
        atol=0.0,
    )

    canonical_obs = torch.load(
        canonical_paths.obs_path,
        map_location="cpu",
        weights_only=False,
    )
    composite_obs = torch.load(
        composite_paths.obs_path,
        map_location="cpu",
        weights_only=False,
    )

    for observation_name in ("source_index", "sample_id", "label"):
        assert (
            composite_obs[observation_name]
            == canonical_obs[observation_name]
        )


def _load_metrics(evaluation_result: Any) -> dict[str, Any]:
    metrics_path = evaluation_result.export_paths.metrics_json_path
    assert metrics_path is not None

    with metrics_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _assert_expected_evaluation_coverage(
    metrics: dict[str, Any],
) -> None:
    assert "kmeans" in metrics["clustering"]["internal"]
    assert "kmeans" in metrics["clustering"]["external"]
    assert "label" in metrics["predictability"]
    assert "global" in metrics["reconstruction"]["metrics"]


def _assert_completed(*workflow_results: Any) -> None:
    for result in workflow_results:
        assert result.status_report.status == "completed"
