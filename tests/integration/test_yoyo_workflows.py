"""Exercise YOYO through real Lightning runs and the written artifact lineage."""

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
import torch
import yaml

from benchrep import (
    train_yoyo,
    predict_yoyo,
    predict_ae,
    predict_vae,
    predict_composite,
    evaluate,
    TrainingOverrides,
    PredictionOverrides,
    EvaluationOverrides,
    update_config,
)
from benchrep.assembly.config import load_yaml
from benchrep.assembly.registries.builtins import register_builtins
from benchrep.assembly.registries.core import DATASETS
from benchrep.assembly.schemas.parsing import (
    TrainingConfigContext,
    PredictionConfigContext,
    EvaluationConfigContext,
    parse_training_config,
    parse_prediction_config,
    parse_evaluation_config,
)
from benchrep.interfaces.model_families import YOYO_FAMILY
from tests.fixtures.models import TinyYOYOModel, CompatibleExternalAutoencoder
from tests.fixtures.datasets import TinySyntheticDataset, PrivateImageBatchDataset
from tests.fixtures.datamodules import (
    ExternalDataModule,
    ParameterizedExternalDataModule,
)


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    """Isolate artifacts and keep auto-selected trainers on CPU on GPU hosts too."""
    monkeypatch.chdir(tmp_path)
    from lightning.fabric.accelerators import (
        CUDAAccelerator as FabricCUDAAccelerator,
        MPSAccelerator as FabricMPSAccelerator,
    )
    from lightning.pytorch.accelerators import CUDAAccelerator, MPSAccelerator

    # Auto-selection uses Fabric; Trainer validation uses PyTorch Lightning.
    for accelerator in (
        FabricCUDAAccelerator,
        FabricMPSAccelerator,
        CUDAAccelerator,
        MPSAccelerator,
    ):
        monkeypatch.setattr(accelerator, "is_available", staticmethod(lambda: False))
    register_builtins()
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)


def external_data(n=11):
    data = TinySyntheticDataset(n_samples=n, image_shape=(1, 4, 4), n_classes=2)
    return ExternalDataModule(
        train_dataset=data, val_dataset=data, predict_dataset=data, batch_size=4
    )


def training_config(*, external_dm=True, params=None, monitor=False, epochs=1):
    raw = {
        "run": {"output_root": "outputs", "run_name_prefix": "yoyo_test"},
        "trainer": {
            "max_epochs": epochs,
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
        "logger": None,
        "inspection": {"torchview": {"enabled": False}},
    }
    if params is not None:
        raw["overrides"] = {"model": {"params": params}}
    if monitor:
        # Retain each epoch: max ranking identifies an earlier checkpoint,
        # while the last checkpoint contains the final weights.
        raw["checkpointing"] = {
            "monitor": "val/custom",
            "mode": "max",
            "save_top_k": -1,
        }
    if not external_dm:
        raw.update(
            {
                "dataset": {
                    "name": "tiny_synthetic",
                    "params": {
                        "n_samples": 12,
                        "image_shape": [1, 4, 4],
                        "n_classes": 2,
                    },
                },
                "datamodule": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "val_fraction": 0.25,
                    "drop_last": False,
                },
                "transform_pipelines": [
                    {
                        "input": "x",
                        "steps": [
                            {
                                "name": "to_dtype",
                                "params": {"dtype": "float32"},
                                "apply_to": ["training", "validation"],
                            }
                        ],
                    }
                ],
            }
        )
    return parse_training_config(
        raw,
        context=TrainingConfigContext(
            external_model=True, external_datamodule=external_dm
        ),
    )


def prediction_raw():
    return {
        "exports": {
            "anndata": {
                "mode": "custom",
                "keys": ["features", "score", "column"],
                "primary_key": "features",
            },
            "reconstructions": {
                "enabled": True,
                "mode": "custom",
                "pairs": {
                    "image_pair": {"input": "image", "reconstruction": "decoded"}
                },
                "n_examples": "all",
            },
        }
    }


def prediction_config(raw=None, *, linked=False):
    return parse_prediction_config(
        prediction_raw() if raw is None else raw,
        context=PredictionConfigContext(
            model_family=YOYO_FAMILY, training_manifest_path_overridden=linked
        ),
    )


def read_manifest(path):
    return yaml.safe_load(path.read_text())


def last_prediction_manifest():
    paths = list(Path("outputs/prediction").glob("*/records/metadata/*manifest*.yaml"))
    assert len(paths) == 1
    return read_manifest(paths[0])


@pytest.mark.parametrize("model_class", [False, True])
@pytest.mark.parametrize("configured_dm", [False, True])
def test_train_predict_and_reload_resolved_configuration(model_class, configured_dm):
    config = training_config(
        external_dm=not configured_dm,
        params={"initial_gain": 0.4} if model_class else None,
    )
    # Regression for the authored/default bug: editing another section must
    # not turn an omitted YOYO monitor into the canonical metric.
    config = update_config(
        config,
        {"trainer.max_epochs": 1},
        context=TrainingConfigContext(
            external_model=True, external_datamodule=not configured_dm
        ),
    )
    supplied_model = TinyYOYOModel if model_class else TinyYOYOModel(initial_gain=0.4)
    trained = train_yoyo(
        config=config,
        overrides=TrainingOverrides(
            model=supplied_model, datamodule=None if configured_dm else external_data()
        ),
    )
    assert trained.model.initial_weight_seen == pytest.approx(0.4)
    assert float(trained.model.gain.detach()) != pytest.approx(0.4)
    training = read_manifest(trained.manifest_path)
    assert training["construction"]["model"]["family"] == "yoyo"
    assert training["checkpoints"]["monitor"] is None
    assert training["checkpoints"]["best_checkpoint_path"] is None
    assert Path(training["checkpoints"]["last_checkpoint_path"]).is_file()
    assert trained.config.checkpointing.save_top_k == 0
    original = read_manifest(trained.run_context.config_dir / "original_config.yaml")
    assert "monitor" not in original.get("checkpointing", {})

    predicted = predict_yoyo(
        config=prediction_config(linked=True),
        overrides=PredictionOverrides(
            model=TinyYOYOModel if model_class else TinyYOYOModel(initial_gain=99.0),
            datamodule=None if configured_dm else external_data(),
            training_manifest_path=trained.manifest_path,
        ),
    )
    torch.testing.assert_close(predicted.model.gain, trained.model.gain)
    assert predicted.run_spec.checkpoint_source == "training_manifest_last"
    data = ad.read_h5ad(predicted.export_result.anndata.path)
    assert data.shape == (12 if configured_dm else 11, 2)
    assert data.obsm["column"].shape == (data.n_obs, 1)
    np.testing.assert_allclose(data.X[:, 0], data.obs["score"])
    assert list(data.obs_names) == [f"sample_{i:04d}" for i in range(data.n_obs)]
    assert (
        data.uns["benchrep"]["prediction_export"]["output_structures_by_key"]["score"]
        == "scalar"
    )
    manifest = read_manifest(predicted.manifest_path)
    assert manifest["status_report"]["inference"]["n_observations"] == data.n_obs
    assert manifest["source"]["training_manifest_path"] == str(trained.manifest_path)

    replay = predict_yoyo(
        config=predicted.run_context.config_dir / "resolved_config.yaml",
        overrides=PredictionOverrides(
            model=TinyYOYOModel if model_class else TinyYOYOModel(),
            datamodule=None if configured_dm else external_data(),
        ),
    )
    np.testing.assert_allclose(
        ad.read_h5ad(replay.export_result.anndata.path).X, data.X
    )


@pytest.mark.parametrize(
    "output_kind,metadata",
    [
        ("mapping", True),
        ("attributes", True),
        ("mapping", False),
        ("attributes", False),
    ],
)
def test_standalone_current_weights_and_evaluation(output_kind, metadata):
    model = TinyYOYOModel(
        initial_gain=0.7, output_kind=output_kind, emit_metadata=metadata
    )
    result = predict_yoyo(
        config=prediction_config(),
        overrides=PredictionOverrides(model=model, datamodule=external_data()),
    )
    assert result.model is model
    assert result.trainer.accelerator.name() == "cpu"
    assert model.gain.item() == pytest.approx(0.7)
    manifest = read_manifest(result.manifest_path)
    assert manifest["source"]["checkpoint_path"] is None
    assert manifest["source"]["training_manifest_path"] is None
    assert manifest["source"]["checkpoint_source"] == "runtime_model"
    assert manifest["appendix"]["training_manifest"] is None
    assert manifest["config_inheritance"]["inherited_from_training"] == []
    assert (
        manifest["construction"]["config"]["run_reconstructable_from_resolved_config"]
        is False
    )
    raw = {
        "reductions": {
            "pca": {"enabled": False},
            "umap": {"enabled": False},
            "tsne": {"enabled": False},
        },
        "clustering": {
            "kmeans": {"enabled": False},
            "leiden": {"enabled": False},
            "hdbscan": {"enabled": False},
        },
        "metrics": {
            "embedding": {"enabled": True, "selected": ["mean"]},
            "clustering": {
                "internal": {"enabled": False},
                "external": {"enabled": False},
            },
            "predictability": {"enabled": False},
            "reconstruction": {"enabled": True, "selected": ["mse"]},
        },
        "reconstruction": {"export_tiffs": False},
        "plots": {"enabled": False},
    }
    evaluated = evaluate(
        config=parse_evaluation_config(
            raw,
            context=EvaluationConfigContext(prediction_manifest_path_overridden=True),
        ),
        overrides=EvaluationOverrides(prediction_manifest_path=result.manifest_path),
    )
    assert evaluated.adata.n_obs == 11
    assert evaluated.manifest_path.is_file()
    assert evaluated.status_report.status in {"completed", "completed_with_warnings"}


def test_private_batch_and_class_overrides_without_canonical_annotations():
    data = PrivateImageBatchDataset(n_samples=7, image_shape=(1, 4, 4))
    dm = ExternalDataModule(
        train_dataset=data, val_dataset=data, predict_dataset=data, batch_size=3
    )
    config = training_config(params={"input_key": "image"})
    trained = train_yoyo(
        config=config, overrides=TrainingOverrides(model=TinyYOYOModel, datamodule=dm)
    )
    predicted = predict_yoyo(
        config=prediction_config(linked=True),
        overrides=PredictionOverrides(
            model=TinyYOYOModel,
            datamodule=dm,
            training_manifest_path=trained.manifest_path,
        ),
    )
    assert predicted.model.input_key == "image"
    assert ad.read_h5ad(predicted.export_result.anndata.path).shape == (7, 2)


@pytest.mark.parametrize(
    "selection", ["omitted", "last", "null", "absolute", "filename"]
)
def test_checkpoint_selection_changes_actual_weights(selection):
    trained = train_yoyo(
        config=training_config(monitor=True, epochs=2),
        overrides=TrainingOverrides(model=TinyYOYOModel(), datamodule=external_data()),
    )
    training = read_manifest(trained.manifest_path)
    best = Path(training["checkpoints"]["best_checkpoint_path"])
    last = Path(training["checkpoints"]["last_checkpoint_path"])
    best_gain = torch.load(best, map_location="cpu")["state_dict"]["gain"]
    last_gain = torch.load(last, map_location="cpu")["state_dict"]["gain"]
    assert not torch.equal(best_gain, last_gain)
    raw = prediction_raw()
    if selection != "omitted":
        raw["source"] = {
            "checkpoint": {
                "last": "last",
                "null": None,
                "absolute": str(best),
                "filename": best.name,
            }[selection]
        }
    model = TinyYOYOModel(initial_gain=9.0)
    result = predict_yoyo(
        config=prediction_config(raw, linked=True),
        overrides=PredictionOverrides(
            model=model,
            datamodule=external_data(),
            training_manifest_path=trained.manifest_path,
        ),
    )
    expected = (
        torch.tensor(9.0)
        if selection == "null"
        else last_gain if selection == "last" else best_gain
    )
    torch.testing.assert_close(model.gain, expected)
    if selection == "null":
        assert read_manifest(result.manifest_path)["source"]["checkpoint_path"] is None


def test_standalone_class_constructor_checkpoint_and_reproducibility(tmp_path):
    raw = prediction_raw()
    raw["overrides"] = {
        "model": {"params": {"initial_gain": None}},
        "datamodule": {"params": {"predict_samples": 7, "batch_size": 3}},
    }
    raw["inference"] = {"seed": 0, "seed_workers": False}
    result = predict_yoyo(
        config=prediction_config(raw),
        overrides=PredictionOverrides(
            model=TinyYOYOModel, datamodule=ParameterizedExternalDataModule
        ),
    )
    torch.testing.assert_close(
        result.model.gain, torch.rand((), generator=torch.Generator().manual_seed(0))
    )
    assert ad.read_h5ad(result.export_result.anndata.path).n_obs == 7
    checkpoint = tmp_path / "external.ckpt"
    torch.save({"state_dict": {"gain": torch.tensor(3.0)}}, checkpoint)
    raw["source"] = {"checkpoint": str(checkpoint)}
    restored = predict_yoyo(
        config=prediction_config(raw),
        overrides=PredictionOverrides(
            model=TinyYOYOModel, datamodule=ParameterizedExternalDataModule
        ),
    )
    assert restored.model.gain.item() == 3.0
    assert (
        read_manifest(restored.manifest_path)["source"]["training_manifest_path"]
        is None
    )


@pytest.mark.parametrize(
    "fault,branch,message",
    [
        ("missing_primary", "anndata", "features"),
        ("scalar_primary", "anndata", "primary key"),
        ("image_vector", "anndata", "column"),
        ("batch_alignment", "anndata", "per-batch sizes"),
        ("annotation_collision", "anndata", "conflicts"),
        ("reconstruction_alignment", "reconstructions", "per-batch sizes"),
        ("metadata_alignment", "both", "annotation"),
        ("metadata_keys", "both", "metadata columns"),
    ],
)
def test_export_failures_write_diagnostic_manifest_and_preserve_other_branch(
    fault, branch, message
):
    with pytest.raises(RuntimeError):
        predict_yoyo(
            config=prediction_config(),
            overrides=PredictionOverrides(
                model=TinyYOYOModel(fault=fault), datamodule=external_data()
            ),
        )
    manifest = last_prediction_manifest()
    assert manifest["status_report"]["inference"]["status"] == "completed"
    assert manifest["status"] == (
        "failed" if branch == "both" else "partially_completed"
    )
    for name in ("anndata", "reconstructions"):
        outcome = manifest["status_report"]["exports"][name]
        if branch in {name, "both"}:
            assert outcome["status"] == "failed"
            assert message in " ".join(outcome["issues"])
        else:
            assert outcome["status"] == "completed"


def test_failed_reconstruction_pair_does_not_block_valid_pair():
    raw = prediction_raw()
    raw["exports"]["reconstructions"]["pairs"]["bad"] = {
        "input": "image",
        "reconstruction": "absent",
    }
    with pytest.raises(RuntimeError):
        predict_yoyo(
            config=prediction_config(raw),
            overrides=PredictionOverrides(
                model=TinyYOYOModel(), datamodule=external_data()
            ),
        )
    manifest = last_prediction_manifest()
    pairs = manifest["status_report"]["exports"]["reconstructions"]["pairs"]
    assert pairs["image_pair"]["status"] == "completed"
    assert pairs["bad"]["status"] == "failed"
    assert Path(
        manifest["exports"]["reconstructions"]["pairs"]["image_pair"]["paths"][
            "reconstruction"
        ]
    ).is_file()


def test_disabled_exports_allow_unnamed_predictions_and_unknown_count():
    raw = {
        "exports": {
            "anndata": {"enabled": False},
            "reconstructions": {"enabled": False},
        }
    }
    result = predict_yoyo(
        config=prediction_config(raw),
        overrides=PredictionOverrides(
            model=TinyYOYOModel(output_kind="tensor"), datamodule=external_data()
        ),
    )
    assert torch.is_tensor(result.predictions[0])
    assert (
        read_manifest(result.manifest_path)["status_report"]["inference"][
            "n_observations"
        ]
        is None
    )


def test_selected_annotation_names_are_exported_as_tensors():
    raw = {
        "exports": {
            "anndata": {
                "mode": "custom",
                "keys": ["label", "sample_id", "metadata"],
                "primary_key": "label",
            }
        }
    }
    result = predict_yoyo(
        config=prediction_config(raw),
        overrides=PredictionOverrides(
            model=TinyYOYOModel(fault="reserved_names"), datamodule=external_data()
        ),
    )
    data = ad.read_h5ad(result.export_result.anndata.path)
    assert data.shape == (11, 2)
    assert "sample_id" in data.obs
    assert data.obsm["metadata"].shape == (11, 1)


class AnnotationTensorYOYOModel(TinyYOYOModel):
    """Return a tensor under an optional annotation name alongside other outputs."""

    def __init__(self, annotation_key, tensor_source, output_kind="mapping"):
        super().__init__()
        self.annotation_key = annotation_key
        self.tensor_source = tensor_source
        self.attribute_output = output_kind == "attributes"

    def predict_step(self, batch, batch_idx):
        outputs = super().predict_step(batch, batch_idx)
        outputs[self.annotation_key] = outputs[self.tensor_source]
        return SimpleNamespace(**outputs) if self.attribute_output else outputs


@pytest.mark.parametrize("annotation_key", ["metadata", "label", "sample_id"])
@pytest.mark.parametrize("output_kind", ["mapping", "attributes"])
@pytest.mark.parametrize("selected_by", ["anndata", "first_pair_input", "last_pair_output"])
def test_annotation_tensor_selection_applies_across_all_exports(
    annotation_key, output_kind, selected_by
):
    """Selections in one branch or pair must not become annotations elsewhere."""
    raw = prediction_raw()
    pairs = raw["exports"]["reconstructions"]["pairs"]
    if selected_by == "anndata":
        raw["exports"]["anndata"]["keys"].append(annotation_key)
        tensor_source = "features"
    elif selected_by == "first_pair_input":
        pairs["image_pair"]["input"] = annotation_key
        pairs["other_pair"] = {"input": "image", "reconstruction": "decoded"}
        tensor_source = "image"
    else:
        pairs["selected_pair"] = {"input": "image", "reconstruction": annotation_key}
        tensor_source = "decoded"

    result = predict_yoyo(
        config=prediction_config(raw),
        overrides=PredictionOverrides(
            model=AnnotationTensorYOYOModel(annotation_key, tensor_source, output_kind),
            datamodule=external_data(),
        ),
    )
    assert read_manifest(result.manifest_path)["status"] == "completed"
    data = ad.read_h5ad(result.export_result.anndata.path)
    assert data.shape == (11, 2)
    assert annotation_key not in data.obs.columns
    if selected_by == "anndata":
        np.testing.assert_array_equal(data.obsm[annotation_key], data.X)
    if annotation_key == "sample_id":
        assert list(data.obs_names) == [f"sample_{i}" for i in range(11)]

    for pair in result.export_result.reconstructions.pairs:
        assert pair.outcome.status == "completed"
        observations = torch.load(pair.paths.obs_path)
        assert annotation_key not in observations
        torch.testing.assert_close(
            torch.load(pair.paths.reconstruction_path),
            torch.load(pair.paths.input_path) * 0.25,
        )
        if annotation_key != "sample_id":
            assert observations["sample_id"] == [f"sample_{i:04d}" for i in range(11)]
        if annotation_key != "label":
            assert observations["label"] == [i % 2 for i in range(11)]
        if annotation_key != "metadata":
            assert observations["group"] == [f"group_{(i // 2) % 2}" for i in range(11)]


@pytest.mark.parametrize("disabled", ["anndata", "reconstructions", "input", "reconstruction"])
def test_disabled_tensor_selections_preserve_annotations(disabled):
    """Inactive selections must not suppress real annotation dictionaries."""
    raw = prediction_raw()
    reconstruction = raw["exports"]["reconstructions"]
    if disabled == "anndata":
        raw["exports"]["anndata"] = {
            "enabled": False, "mode": "custom", "keys": ["metadata"],
            "primary_key": "metadata",
        }
    elif disabled == "reconstructions":
        reconstruction["enabled"] = False
        reconstruction["pairs"]["image_pair"]["input"] = "metadata"
    else:
        reconstruction[f"include_{disabled}"] = False
        reconstruction["pairs"]["image_pair"][disabled] = "metadata"
    result = predict_yoyo(
        config=prediction_config(raw),
        overrides=PredictionOverrides(model=TinyYOYOModel(), datamodule=external_data()),
    )
    groups = [f"group_{(i // 2) % 2}" for i in range(11)]
    assert read_manifest(result.manifest_path)["status"] == "completed"
    if result.export_result.anndata.path is not None:
        assert list(ad.read_h5ad(result.export_result.anndata.path).obs["group"]) == groups
    for pair in result.export_result.reconstructions.pairs:
        assert torch.load(pair.paths.obs_path)["group"] == groups


def test_invalid_selected_annotation_tensor_only_fails_its_export():
    """A bad AnnData tensor must not become annotations in a valid image export."""
    raw = prediction_raw()
    raw["exports"]["anndata"]["keys"].append("metadata")
    with pytest.raises(RuntimeError, match="partially_completed"):
        predict_yoyo(
            config=prediction_config(raw),
            overrides=PredictionOverrides(
                model=AnnotationTensorYOYOModel("metadata", "image"),
                datamodule=external_data(),
            ),
        )
    manifest = last_prediction_manifest()
    outcomes = manifest["status_report"]["exports"]
    assert outcomes["anndata"]["status"] == "failed"
    assert "must provide [samples] or [samples, features]" in " ".join(
        outcomes["anndata"]["issues"]
    )
    assert outcomes["reconstructions"]["status"] == "completed"
    paths = manifest["exports"]["reconstructions"]["pairs"]["image_pair"]["paths"]
    assert Path(paths["reconstruction"]).is_file()


def test_standalone_configured_data_transform_and_batch_limit():
    raw = prediction_raw()
    raw.update(
        {
            "dataset": {
                "name": "tiny_synthetic",
                "params": {"n_samples": 7, "image_shape": [1, 4, 4]},
            },
            "data": {"batch_size": 2, "num_workers": 0, "max_batches": 2},
            "transform_pipelines": [
                {
                    "input": "x",
                    "output": "private_image",
                    "steps": [{"name": "to_dtype", "params": {"dtype": "float32"}}],
                }
            ],
            "inference": {"seed": 0, "seed_workers": False},
        }
    )
    result = predict_yoyo(
        config=prediction_config(raw),
        overrides=PredictionOverrides(model=TinyYOYOModel(input_key="private_image")),
    )
    assert ad.read_h5ad(result.export_result.anndata.path).n_obs == 4
    assert result.run_spec.seed == 0 and result.run_spec.seed_workers is False
    assert result.run_spec.transform_pipeline_source == "prediction_config"
    assert (
        read_manifest(result.manifest_path)["status_report"]["inference"][
            "n_observations"
        ]
        == 4
    )


def test_linked_external_datamodule_requires_explicit_prediction_data():
    trained = train_yoyo(
        config=training_config(),
        overrides=TrainingOverrides(model=TinyYOYOModel(), datamodule=external_data()),
    )
    overrides = PredictionOverrides(
        model=TinyYOYOModel(), training_manifest_path=trained.manifest_path
    )
    with pytest.raises(ValueError, match="dataset"):
        predict_yoyo(config=prediction_config(linked=True), overrides=overrides)
    raw = prediction_raw()
    raw["dataset"] = {
        "name": "tiny_synthetic",
        "params": {"n_samples": 5, "image_shape": [1, 4, 4]},
    }
    raw["data"] = {"num_workers": 0}
    result = predict_yoyo(
        config=prediction_config(raw, linked=True), overrides=overrides
    )
    assert ad.read_h5ad(result.export_result.anndata.path).n_obs == 5
    assert result.run_spec.transform_pipeline_source == "default_identity"


@pytest.mark.parametrize("family", ["autoencoder", "vae", "composite"])
def test_linked_manifest_family_cannot_be_bypassed(family, tmp_path):
    trained = train_yoyo(
        config=training_config(),
        overrides=TrainingOverrides(model=TinyYOYOModel(), datamodule=external_data()),
    )
    manifest = read_manifest(trained.manifest_path)
    manifest["construction"]["model"]["family"] = family
    path = tmp_path / "other_family.yaml"
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="family|expects|match"):
        predict_yoyo(
            config=prediction_config(linked=True),
            overrides=PredictionOverrides(
                model=TinyYOYOModel(),
                datamodule=external_data(),
                training_manifest_path=path,
            ),
        )


def test_existing_families_still_reject_yoyo_only_escape_hatches():
    for entrypoint in (predict_ae, predict_vae, predict_composite):
        with pytest.raises(TypeError):
            entrypoint(
                config=prediction_config(),
                overrides=PredictionOverrides(
                    model=TinyYOYOModel(), datamodule=external_data()
                ),
            )
    for raw in (prediction_raw(), {**prediction_raw(), "source": {"checkpoint": None}}):
        with pytest.raises(ValueError, match="manifest|checkpoint"):
            predict_ae(
                config=prediction_config(raw),
                overrides=PredictionOverrides(
                    model=CompatibleExternalAutoencoder(), datamodule=external_data()
                ),
            )


@pytest.mark.parametrize(
    "raw,match",
    [
        ({}, "custom"),
        (
            {"exports": {"anndata": {"mode": "custom", "keys": ["features"]}}},
            "primary_key",
        ),
        (
            {
                "exports": {
                    "anndata": {"enabled": False},
                    "reconstructions": {"enabled": True},
                }
            },
            "custom",
        ),
        (
            {
                **prediction_raw(),
                "transform_pipelines": [
                    {"steps": [{"name": "to_dtype", "params": {"dtype": "float32"}}]}
                ],
                "dataset": {"name": "tiny_synthetic"},
            },
            "input",
        ),
    ],
)
def test_model_dependent_requirements_fail_before_inference(raw, match):
    # No DM override for the transform case: the configured route must be resolved.
    dm = None if "transform_pipelines" in raw else external_data()
    with pytest.raises(ValueError, match=match):
        predict_yoyo(
            config=prediction_config(raw),
            overrides=PredictionOverrides(model=TinyYOYOModel(), datamodule=dm),
        )


@pytest.mark.parametrize("checkpoint", ["best", "last", "external.ckpt"])
def test_standalone_rejects_manifest_relative_checkpoint_selectors(checkpoint):
    raw = prediction_raw()
    raw["source"] = {"checkpoint": checkpoint}
    with pytest.raises(ValueError, match="manifest"):
        predict_yoyo(
            config=prediction_config(raw),
            overrides=PredictionOverrides(
                model=TinyYOYOModel(), datamodule=external_data()
            ),
        )


@pytest.mark.parametrize(
    "contents,match",
    [
        ({}, "state_dict"),
        ({"state_dict": []}, "mapping"),
        ({"state_dict": {"wrong": torch.tensor(1.0)}}, "state_dict"),
    ],
)
def test_bad_checkpoint_is_not_ignored(contents, match, tmp_path):
    path = tmp_path / "invalid.ckpt"
    torch.save(contents, path)
    raw = prediction_raw()
    raw["source"] = {"checkpoint": str(path)}
    model = TinyYOYOModel(initial_gain=7.0)
    with pytest.raises((KeyError, TypeError, RuntimeError), match=match):
        predict_yoyo(
            config=prediction_config(raw),
            overrides=PredictionOverrides(model=model, datamodule=external_data()),
        )
    assert model.gain.item() == 7.0


def test_actual_model_failure_is_not_wrapped_as_canonical_contract_error():
    with pytest.raises(RuntimeError, match="deliberate YOYO predict_step failure"):
        predict_yoyo(
            config=prediction_config(),
            overrides=PredictionOverrides(
                model=TinyYOYOModel(fault="runtime"), datamodule=external_data()
            ),
        )


def test_disabled_input_export_does_not_require_input_in_predictions():
    raw = prediction_raw()
    raw["exports"]["anndata"]["enabled"] = False
    raw["exports"]["reconstructions"]["include_input"] = False
    result = predict_yoyo(
        config=prediction_config(raw),
        overrides=PredictionOverrides(
            model=TinyYOYOModel(fault="missing_input"), datamodule=external_data()
        ),
    )
    paths = result.export_result.reconstructions.pairs[0].paths
    assert paths.input_path is None
    assert torch.load(paths.reconstruction_path).shape[0] == 11
    assert (
        read_manifest(result.manifest_path)["status_report"]["inference"][
            "n_observations"
        ]
        == 11
    )


def test_optimizer_gradient_inspection_supports_yoyo_whole_model():
    config = training_config()
    config = update_config(
        config,
        {
            "inspection.gradient_norms": [
                {"kind": "optimizer_gradients", "target": "whole_model"}
            ]
        },
        context=TrainingConfigContext(external_model=True, external_datamodule=True),
    )
    result = train_yoyo(
        config=config,
        overrides=TrainingOverrides(model=TinyYOYOModel(), datamodule=external_data()),
    )
    assert any(
        key.startswith("grad_norms_wrt_whole_model/")
        for key in result.trainer.callback_metrics
    )


def test_reconstruction_sampling_keeps_tensors_and_annotations_aligned():
    raw = prediction_raw()
    raw["exports"]["anndata"]["enabled"] = False
    raw["exports"]["reconstructions"].update(
        {"selection": "random", "stratify_by": "label", "n_examples": 6}
    )
    raw["inference"] = {"seed": 137}
    results = [
        predict_yoyo(
            config=prediction_config(raw),
            overrides=PredictionOverrides(
                model=TinyYOYOModel(initial_gain=0.7), datamodule=external_data()
            ),
        )
        for _ in range(2)
    ]
    bundles = [
        result.export_result.reconstructions.pairs[0].paths for result in results
    ]
    first_obs, second_obs = [torch.load(bundle.obs_path) for bundle in bundles]
    assert first_obs == second_obs
    assert len(first_obs["source_index"]) == 6 and len(set(first_obs["label"])) == 2
    images = torch.load(bundles[0].input_path)
    reconstructions = torch.load(bundles[0].reconstruction_path)
    torch.testing.assert_close(reconstructions, images * 0.7)
    dataset = TinySyntheticDataset(n_samples=11, image_shape=(1, 4, 4), n_classes=2)
    for row, source_index in enumerate(first_obs["source_index"]):
        sample = dataset[source_index]
        torch.testing.assert_close(images[row], sample["x"])
        assert first_obs["sample_id"][row] == sample["sample_id"]
        assert first_obs["label"][row] == sample["label"]


def test_recorded_missing_best_checkpoint_is_not_silently_replaced_by_last(tmp_path):
    trained = train_yoyo(
        config=training_config(),
        overrides=TrainingOverrides(model=TinyYOYOModel(), datamodule=external_data()),
    )
    manifest = read_manifest(trained.manifest_path)
    manifest["checkpoints"]["best_checkpoint_path"] = str(tmp_path / "missing.ckpt")
    trained.manifest_path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(FileNotFoundError, match="missing.ckpt"):
        predict_yoyo(
            config=prediction_config(linked=True),
            overrides=PredictionOverrides(
                model=TinyYOYOModel(),
                datamodule=external_data(),
                training_manifest_path=trained.manifest_path,
            ),
        )


@pytest.mark.parametrize("stage", ["training", "prediction"])
@pytest.mark.parametrize("component", ["model", "datamodule"])
def test_instance_constructor_parameters_are_rejected(stage, component):
    if stage == "training":
        config = update_config(
            training_config(),
            {f"overrides.{component}": {"params": {"unused": 137}}},
            context=TrainingConfigContext(
                external_model=True, external_datamodule=True
            ),
        )
        entrypoint, overrides = train_yoyo, TrainingOverrides(
            model=TinyYOYOModel(), datamodule=external_data()
        )
    else:
        raw = prediction_raw()
        raw["overrides"] = {component: {"params": {"unused": 137}}}
        config = prediction_config(raw)
        entrypoint, overrides = predict_yoyo, PredictionOverrides(
            model=TinyYOYOModel(), datamodule=external_data()
        )
    with pytest.raises(ValueError, match="instance|params"):
        entrypoint(config=config, overrides=overrides)


@pytest.mark.parametrize(
    "inspection_request,match",
    [
        ({"kind": "per_loss", "target": "whole_model"}, "Per-loss"),
        ({"kind": "optimizer_gradients", "target": "encoder"}, "target"),
    ],
)
def test_yoyo_rejects_inspection_that_requires_a_declared_model_contract(
    inspection_request, match
):
    config = update_config(
        training_config(),
        {"inspection.gradient_norms": [inspection_request]},
        context=TrainingConfigContext(external_model=True, external_datamodule=True),
    )
    with pytest.raises(ValueError, match=match):
        train_yoyo(
            config=config,
            overrides=TrainingOverrides(
                model=TinyYOYOModel(), datamodule=external_data()
            ),
        )
