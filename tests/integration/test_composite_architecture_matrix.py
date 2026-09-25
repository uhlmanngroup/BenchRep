from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
import torch

import anndata as ad
import numpy as np

from benchrep.architecture.models import CompositeModel
from benchrep.assembly.config import load_yaml
from benchrep.assembly.registries.core import DATASETS
from benchrep.assembly.schemas import PredictionConfig, TrainingConfig
from benchrep.workflows import predict_composite, train_composite
from tests.fixtures.datasets import TinySyntheticDataset


CONFIG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "configs"
BATCH_SIZE = 4


@dataclass(frozen=True)
class CompositeArchitectureCase:
    name: str
    declarations: dict[str, Any]
    components: dict[str, Any]
    assembly: dict[str, Any]
    losses: dict[str, Any]
    transform_pipelines: list[dict[str, Any]]
    primary_key: str
    expected_output_shapes: dict[str, tuple[int, ...]]
    expected_losses: frozenset[tuple[str, str]]
    shared_component_step_ids: tuple[str, ...] = ()
    export_reconstructions: bool = False


def _encoder_component(*, output_dim: int = 6) -> dict[str, Any]:
    return {
        "kind": "encoder",
        "name": "mlp",
        "params": {
            "input_shape": [1, 8, 8],
            "output_dim": output_dim,
            "hidden_dims": [12],
            "dropout": 0.0,
        },
    }


def _decoder_component(*, input_dim: int = 6) -> dict[str, Any]:
    return {
        "kind": "decoder",
        "name": "mlp",
        "params": {
            "input_dim": input_dim,
            "output_shape": [1, 8, 8],
            "hidden_dims": [12],
            "dropout": 0.0,
        },
    }


def _head_component(
    *,
    input_dim: int,
    output_dim: int,
) -> dict[str, Any]:
    return {
        "kind": "head",
        "name": "mlp",
        "params": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": [],
            "dropout": 0.0,
        },
    }


def _copy_image_branch(output: str) -> dict[str, Any]:
    return {
        "input": "x",
        "output": output,
        "steps": [
            {
                "name": "to_dtype",
                "apply_to": ["training", "validation"],
                "params": {
                    "dtype": "float32",
                    "scale": False,
                },
            },
        ],
    }


def _supervised_head_case() -> CompositeArchitectureCase:
    return CompositeArchitectureCase(
        name="supervised_head_only",
        declarations={
            "expects": {
                "x": "sample_image",
                "label": "categorical_prediction_target_scalar",
            },
            "batch_metadata": {"sample_id": "index"},
            "produces": {
                "embedding": "embedding_vector",
                "logits": "categorical_prediction_vector",
            },
        },
        components={
            "encoder": _encoder_component(),
            "classifier": _head_component(input_dim=6, output_dim=4),
        },
        assembly={
            "encode": {
                "component": "encoder",
                "inputs": {"x": "expects.x"},
                "outputs": "produces.embedding",
            },
            "classify": {
                "component": "classifier",
                "inputs": {"x": "produces.embedding"},
                "outputs": "produces.logits",
            },
        },
        losses={
            "classification": [
                {
                    "name": "cross_entropy",
                    "weight": 1.0,
                    "params": {"reduction": "mean"},
                    "composite_wiring": {
                        "prediction": "produces.logits",
                        "target": "expects.label",
                    },
                },
            ],
        },
        transform_pipelines=[],
        primary_key="embedding",
        expected_output_shapes={
            "embedding": (6,),
            "logits": (4,),
        },
        expected_losses=frozenset({
            ("classification", "cross_entropy"),
        }),
    )


def _shared_encoder_triplet_case() -> CompositeArchitectureCase:
    return CompositeArchitectureCase(
        name="shared_encoder_triplet",
        declarations={
            "expects": {
                "x": "sample_image",
                "positive_x": "positive_image",
                "negative_x": "negative_image",
                "label": "categorical_prediction_target_scalar",
            },
            "batch_metadata": {"sample_id": "index"},
            "produces": {
                "anchor_embedding": "embedding_vector",
                "positive_embedding": "projection_vector",
                "negative_embedding": "projection_vector",
            },
        },
        components={
            "shared_encoder": _encoder_component(),
        },
        assembly={
            "encode_anchor": {
                "component": "shared_encoder",
                "inputs": {"x": "expects.x"},
                "outputs": "produces.anchor_embedding",
            },
            "encode_positive": {
                "component": "shared_encoder",
                "inputs": {"x": "expects.positive_x"},
                "outputs": "produces.positive_embedding",
            },
            "encode_negative": {
                "component": "shared_encoder",
                "inputs": {"x": "expects.negative_x"},
                "outputs": "produces.negative_embedding",
            },
        },
        losses={
            "contrastive": [
                {
                    "name": "triplet_margin",
                    "weight": 1.0,
                    "params": {
                        "margin": 0.2,
                        "reduction": "mean",
                    },
                    "composite_wiring": {
                        "anchor": "produces.anchor_embedding",
                        "positive": "produces.positive_embedding",
                        "negative": "produces.negative_embedding",
                    },
                },
            ],
        },
        transform_pipelines=[
            _copy_image_branch("positive_x"),
            _copy_image_branch("negative_x"),
        ],
        primary_key="anchor_embedding",
        expected_output_shapes={
            "anchor_embedding": (6,),
            "positive_embedding": (6,),
            "negative_embedding": (6,),
        },
        expected_losses=frozenset({
            ("contrastive", "triplet_margin"),
        }),
        shared_component_step_ids=(
            "encode_anchor",
            "encode_positive",
            "encode_negative",
        ),
    )


def _dual_encoder_decoder_case() -> CompositeArchitectureCase:
    return CompositeArchitectureCase(
        name="dual_encoder_dual_decoder",
        declarations={
            "expects": {
                "x": "sample_image",
                "second_image": "sample_image",
                "label": "categorical_prediction_target_scalar",
            },
            "batch_metadata": {"sample_id": "index"},
            "produces": {
                "embedding_a": "embedding_vector",
                "embedding_b": "projection_vector",
                "reconstruction_a": "reconstruction_image",
                "reconstruction_b": "reconstruction_image",
            },
        },
        components={
            "encoder_a": _encoder_component(),
            "encoder_b": _encoder_component(),
            "decoder_a": _decoder_component(),
            "decoder_b": _decoder_component(),
        },
        assembly={
            "encode_a": {
                "component": "encoder_a",
                "inputs": {"x": "expects.x"},
                "outputs": "produces.embedding_a",
            },
            "encode_b": {
                "component": "encoder_b",
                "inputs": {"x": "expects.second_image"},
                "outputs": "produces.embedding_b",
            },
            "decode_a": {
                "component": "decoder_a",
                "inputs": {"z": "produces.embedding_a"},
                "outputs": "produces.reconstruction_a",
            },
            "decode_b": {
                "component": "decoder_b",
                "inputs": {"z": "produces.embedding_b"},
                "outputs": "produces.reconstruction_b",
            },
        },
        losses={
            "reconstruction": [
                {
                    "name": "mse",
                    "weight": 1.0,
                    "params": {"reduction": "mean"},
                    "composite_wiring": {
                        "reconstruction": "produces.reconstruction_a",
                        "target": "expects.x",
                    },
                },
                {
                    "name": "mae",
                    "weight": 0.5,
                    "params": {"reduction": "mean"},
                    "composite_wiring": {
                        "reconstruction": "produces.reconstruction_b",
                        "target": "expects.second_image",
                    },
                },
            ],
        },
        transform_pipelines=[
            _copy_image_branch("second_image"),
        ],
        primary_key="embedding_a",
        expected_output_shapes={
            "embedding_a": (6,),
            "embedding_b": (6,),
            "reconstruction_a": (1, 8, 8),
            "reconstruction_b": (1, 8, 8),
        },
        expected_losses=frozenset({
            ("reconstruction", "mse"),
            ("reconstruction", "mae"),
        }),
        export_reconstructions=True,
    )


def _variational_multitask_case() -> CompositeArchitectureCase:
    return CompositeArchitectureCase(
        name="variational_multitask_frankenstein",
        declarations={
            "expects": {
                "x": "sample_image",
                "label": "categorical_prediction_target_scalar",
            },
            "batch_metadata": {"sample_id": "index"},
            "produces": {
                "encoder_features": "projection_vector",
                "z_sample": "embedding_vector",
                "z_mu": "embedding_vector",
                "z_logvar": "continuous_auxiliary_vector",
                "reconstruction": "reconstruction_image",
                "logits": "categorical_prediction_vector",
            },
        },
        components={
            "encoder": _encoder_component(),
            "variational_head": {
                "kind": "head",
                "name": "gaussian_variational",
                "params": {
                    "in_features": 6,
                    "latent_dim": 4,
                },
            },
            "decoder": _decoder_component(input_dim=4),
            "classifier": _head_component(input_dim=4, output_dim=4),
        },
        assembly={
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
            "classify": {
                "component": "classifier",
                "inputs": {"x": "produces.z_sample"},
                "outputs": "produces.logits",
            },
        },
        losses={
            "reconstruction": [
                {
                    "name": "mse",
                    "weight": 1.0,
                    "params": {"reduction": "mean"},
                    "composite_wiring": {
                        "reconstruction": "produces.reconstruction",
                        "target": "expects.x",
                    },
                },
            ],
            "regularization": [
                {
                    "name": "gaussian_kld",
                    "weight": 0.001,
                    "params": {"reduction": "mean"},
                    "composite_wiring": {
                        "z_mu": "produces.z_mu",
                        "z_logvar": "produces.z_logvar",
                    },
                },
            ],
            "classification": [
                {
                    "name": "cross_entropy",
                    "weight": 0.25,
                    "params": {"reduction": "mean"},
                    "composite_wiring": {
                        "prediction": "produces.logits",
                        "target": "expects.label",
                    },
                },
            ],
        },
        transform_pipelines=[],
        primary_key="z_mu",
        expected_output_shapes={
            "encoder_features": (6,),
            "z_sample": (4,),
            "z_mu": (4,),
            "z_logvar": (4,),
            "reconstruction": (1, 8, 8),
            "logits": (4,),
        },
        expected_losses=frozenset({
            ("reconstruction", "mse"),
            ("regularization", "gaussian_kl"),
            ("classification", "cross_entropy"),
        }),
    )


ARCHITECTURE_CASES = (
    _supervised_head_case(),
    _shared_encoder_triplet_case(),
    _dual_encoder_decoder_case(),
    _variational_multitask_case(),
)


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.name) for case in ARCHITECTURE_CASES],
)
def test_composite_architecture_trains_checkpoints_and_predicts(
    case: CompositeArchitectureCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)

    monkeypatch.chdir(tmp_path)

    training_result = train_composite(
        full_config_object=_make_training_config(case),
    )

    assert training_result.status_report.status == "completed"
    assert isinstance(training_result.model, CompositeModel)

    best_checkpoint = Path(
        training_result.checkpoint_callback.best_model_path
    )
    last_checkpoint = Path(
        training_result.checkpoint_callback.last_model_path
    )
    assert best_checkpoint.is_file()
    assert last_checkpoint.is_file()

    assert training_result.run_spec.composite_model_spec is not None
    assert {
               (loss.loss_role, loss.loss_id)
               for loss in training_result.run_spec.loss_specs
           } == case.expected_losses
    assert set(training_result.model.components_by_id) == set(
        case.components
    )
    _assert_shared_component_identity(
        training_result.model,
        case.shared_component_step_ids,
    )

    prediction_result = predict_composite(
        full_config_object=_make_prediction_config(case),
        training_manifest_path=training_result.manifest_path,
    )

    assert prediction_result.status_report.status == "completed"
    assert isinstance(prediction_result.model, CompositeModel)
    _assert_model_states_equal(
        training_result.model,
        prediction_result.model,
    )
    _assert_shared_component_identity(
        prediction_result.model,
        case.shared_component_step_ids,
    )

    assert len(prediction_result.predictions) == 1
    prediction_batch = prediction_result.predictions[0]

    prediction_manifest = load_yaml(prediction_result.manifest_path)
    assert (
        prediction_manifest["status_report"]["inference"]["n_observations"]
        == BATCH_SIZE
    )

    assert set(prediction_batch.model_inputs) == set(
        case.declarations["expects"]
    )
    assert set(prediction_batch.batch_metadata) == {"sample_id"}
    assert set(prediction_batch.model_outputs) == set(
        case.expected_output_shapes
    )

    for output_name, expected_sample_shape in (
        case.expected_output_shapes.items()
    ):
        output = prediction_batch.model_outputs[output_name]
        assert output.shape[0] == BATCH_SIZE
        assert tuple(output.shape[1:]) == expected_sample_shape

    _assert_anndata_export_complete(
        prediction_result=prediction_result,
        prediction_batch=prediction_batch,
        case=case,
    )
    _assert_reconstruction_exports(
        prediction_result=prediction_result,
        prediction_batch=prediction_batch,
        case=case,
    )

    for input_name, input_tensor in prediction_batch.model_inputs.items():
        mismatched_inputs = dict(prediction_batch.model_inputs)
        mismatched_inputs[input_name] = input_tensor[:-1]

        with pytest.raises(ValueError, match="has batch size"):
            prediction_result.model(mismatched_inputs)


def test_composite_requires_at_least_one_sample_image() -> None:
    raw = _make_training_config(
        _dual_encoder_decoder_case()
    ).model_dump(mode="python")

    input_roles = raw["composite_model_declarations"]["expects"]
    for name, role in input_roles.items():
        if role == "sample_image":
            input_roles[name] = "condition_image"

    with pytest.raises(
        ValidationError,
        match="at least one input with role `sample_image`",
    ):
        TrainingConfig.model_validate(raw)


def _make_training_config(
    case: CompositeArchitectureCase,
) -> TrainingConfig:
    raw = load_yaml(CONFIG_DIR / "training_tiny_synthetic_ae.yaml")
    raw["run"]["project_name"] = f"architecture_matrix_{case.name}"
    raw["dataset"]["params"] = _tiny_dataset_params()
    raw["transform_pipelines"] = case.transform_pipelines
    raw["datamodule"].update(
        {
            "batch_size": BATCH_SIZE,
            "val_fraction": 0.5,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
        }
    )

    raw["model"] = {"name": "composite"}
    raw.pop("encoder")
    raw.pop("decoder")
    raw["composite_model_declarations"] = case.declarations
    raw["composite_model_components"] = case.components
    raw["composite_model_assembly"] = case.assembly
    raw["losses"] = case.losses

    raw["trainer"].update(
        {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "deterministic": True,
            "benchmark": False,
            "precision": "32-true",
        }
    )
    raw["logger"] = None
    raw["checkpointing"].update(
        {
            "monitor": "val/loss",
            "mode": "min",
            "save_top_k": 1,
            "save_last": True,
        }
    )
    raw["inspection"]["torchview"]["enabled"] = False

    return TrainingConfig.model_validate(raw)


def _make_prediction_config(
    case: CompositeArchitectureCase,
) -> PredictionConfig:
    raw = load_yaml(CONFIG_DIR / "prediction_tiny_synthetic.yaml")
    raw["dataset"]["params"] = _tiny_dataset_params()
    raw["data"].update(
        {
            "batch_size": BATCH_SIZE,
            "num_workers": 0,
            "max_batches": 1,
        }
    )
    raw["exports"]["anndata"].update(
        {
            "enabled": True,
            "mode": "all",
            "keys": None,
            "primary_key": case.primary_key,
        }
    )
    raw["exports"]["reconstructions"].update(
        {
            "enabled": case.export_reconstructions,
            "mode": "all",
            "pairs": None,
            "n_examples": "all",
            "selection": "first",
            "stratify_by": None,
            "include_input": True,
            "include_reconstruction": True,
        }
    )

    return PredictionConfig.model_validate(
        raw,
        context={"training_manifest_path_overridden": True},
    )


def _tiny_dataset_params() -> dict[str, Any]:
    return {
        "n_samples": 8,
        "image_shape": [1, 8, 8],
        "n_classes": 4,
        "n_groups": 2,
        "signal_strength": 0.8,
        "noise_std": 0.05,
        "seed": 137,
    }


def _assert_model_states_equal(
    trained_model: CompositeModel,
    loaded_model: CompositeModel,
) -> None:
    trained_state = trained_model.state_dict()
    loaded_state = loaded_model.state_dict()

    assert trained_state.keys() == loaded_state.keys()

    for parameter_name in trained_state:
        torch.testing.assert_close(
            loaded_state[parameter_name],
            trained_state[parameter_name],
            rtol=0.0,
            atol=0.0,
            msg=f"Checkpoint state mismatch for {parameter_name!r}",
        )


def _assert_shared_component_identity(
    model: CompositeModel,
    step_ids: tuple[str, ...],
) -> None:
    if not step_ids:
        return

    steps_by_id = {
        step.step_id: step
        for steps in (
            model.model_spec.assembly_steps_by_dependency_level.values()
        )
        for step in steps
    }
    component_instances = [
        model.components_by_id[steps_by_id[step_id].component_id]
        for step_id in step_ids
    ]

    assert all(
        component is component_instances[0]
        for component in component_instances[1:]
    )


def _assert_anndata_export_complete(
    *,
    prediction_result: Any,
    prediction_batch: Any,
    case: CompositeArchitectureCase,
) -> None:
    anndata_result = prediction_result.export_result.anndata

    assert anndata_result.outcome.status == "completed"
    assert anndata_result.path is not None
    assert anndata_result.path.is_file()

    adata = ad.read_h5ad(anndata_result.path)
    expected_vector_keys = [
        output_name
        for output_name, sample_shape
        in case.expected_output_shapes.items()
        if len(sample_shape) == 1
    ]

    assert case.primary_key in expected_vector_keys
    np.testing.assert_array_equal(
        np.asarray(adata.X),
        prediction_batch.model_outputs[
            case.primary_key
        ].detach().cpu().numpy(),
    )

    expected_obsm_keys = set(expected_vector_keys) - {
        case.primary_key,
    }
    assert set(adata.obsm) == expected_obsm_keys

    for output_name in expected_obsm_keys:
        np.testing.assert_array_equal(
            adata.obsm[output_name],
            prediction_batch.model_outputs[
                output_name
            ].detach().cpu().numpy(),
        )

    assert adata.obs.index.name == "sample_id"
    assert adata.obs_names.tolist() == list(
        prediction_batch.batch_metadata["sample_id"]
    )
    assert list(adata.obs.columns) == ["label"]
    np.testing.assert_array_equal(
        adata.obs["label"].to_numpy(),
        prediction_batch.model_inputs["label"].detach().cpu().numpy(),
    )

    export_metadata = adata.uns["benchrep"]["prediction_export"]
    assert list(export_metadata["keys"]) == expected_vector_keys
    assert export_metadata["primary_key"] == case.primary_key


def _assert_reconstruction_exports(
    *,
    prediction_result: Any,
    prediction_batch: Any,
    case: CompositeArchitectureCase,
) -> None:
    reconstruction_result = (
        prediction_result.export_result.reconstructions
    )

    if not case.export_reconstructions:
        assert reconstruction_result.outcome.status == "disabled"
        assert reconstruction_result.pairs == ()
        return

    expected_pairs = [
        (
            "reconstruction_bundle_01",
            "x",
            "reconstruction_a",
        ),
        (
            "reconstruction_bundle_02",
            "second_image",
            "reconstruction_b",
        ),
    ]

    assert reconstruction_result.outcome.status == "completed"
    assert [
        (
            pair_result.pair.id,
            pair_result.pair.input,
            pair_result.pair.reconstruction,
        )
        for pair_result in reconstruction_result.pairs
    ] == expected_pairs

    for pair_result, (
            pair_id,
            input_name,
            reconstruction_name,
    ) in zip(
        reconstruction_result.pairs,
        expected_pairs,
        strict=True,
    ):
        assert pair_result.outcome.status == "completed"

        paths = pair_result.paths
        assert paths.bundle_dir.name == pair_id
        assert paths.n_examples_exported == BATCH_SIZE
        assert paths.input_path is not None
        assert paths.reconstruction_path is not None
        assert paths.obs_path is not None
        assert paths.metadata_path is not None

        for path in (
            paths.input_path,
            paths.reconstruction_path,
            paths.obs_path,
            paths.metadata_path,
        ):
            assert path.is_file()

        exported_inputs = torch.load(
            paths.input_path,
            map_location="cpu",
            weights_only=False,
        )
        exported_reconstructions = torch.load(
            paths.reconstruction_path,
            map_location="cpu",
            weights_only=False,
        )

        torch.testing.assert_close(
            exported_inputs,
            prediction_batch.model_inputs[input_name],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            exported_reconstructions,
            prediction_batch.model_outputs[reconstruction_name],
            rtol=0.0,
            atol=0.0,
        )

    manifest = load_yaml(prediction_result.manifest_path)
    expected_pair_ids = [pair[0] for pair in expected_pairs]
    manifest_exports = manifest["exports"]["reconstructions"]
    manifest_statuses = manifest["status_report"]["exports"][
        "reconstructions"
    ]

    assert list(manifest_exports["pairs"]) == expected_pair_ids
    assert list(manifest_statuses["pairs"]) == expected_pair_ids
    assert manifest["summary"]["exports"]["reconstructions"][
        "pairs"
    ] == expected_pair_ids

    for pair_result, (
        pair_id,
        input_name,
        reconstruction_name,
    ) in zip(
        reconstruction_result.pairs,
        expected_pairs,
        strict=True,
    ):
        manifest_pair = manifest_exports["pairs"][pair_id]

        assert manifest_pair["input"] == input_name
        assert manifest_pair["reconstruction"] == reconstruction_name
        assert manifest_pair["n_examples_exported"] == BATCH_SIZE
        assert manifest_pair["paths"] == {
            "bundle_dir": str(pair_result.paths.bundle_dir),
            "input": str(pair_result.paths.input_path),
            "reconstruction": str(
                pair_result.paths.reconstruction_path
            ),
            "observations": str(pair_result.paths.obs_path),
            "metadata": str(pair_result.paths.metadata_path),
        }
        assert manifest_statuses["pairs"][pair_id] == {
            "status": "completed",
            "issues": [],
        }