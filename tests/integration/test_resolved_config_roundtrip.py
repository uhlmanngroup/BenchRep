from __future__ import annotations

from pathlib import Path

import torch
import yaml

from benchrep.assembly.config import load_yaml
from benchrep.assembly.resolvers import (
    resolve_evaluation_config,
    resolve_prediction_config,
    resolve_training_config,
)
from benchrep.assembly.schemas import (
    parse_evaluation_config,
    parse_prediction_config,
    parse_training_config,
)
from benchrep.interfaces.model_families import AUTOENCODER_FAMILY, VAE_FAMILY
from benchrep.records.configs import save_resolved_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "configs"


def test_training_resolved_config_roundtrip_materializes_benchrep_decisions(
    tmp_path: Path,
) -> None:
    """A resolved training config should be idempotent under re-resolution."""
    raw_config = load_yaml(CONFIG_DIR / "training_tiny_synthetic_ae.yaml")
    raw_config["datamodule"]["pin_memory"] = "auto"
    raw_config["checkpointing"]["monitor"] = None
    raw_config["checkpointing"]["save_top_k"] = 3

    config = parse_training_config(raw_config)

    first = resolve_training_config(
        config,
        model_family=AUTOENCODER_FAMILY,
    )

    assert first.training_config.datamodule is not None
    assert (
        first.training_config.datamodule.pin_memory
        == torch.cuda.is_available()
    )
    assert first.training_config.checkpointing.save_top_k == 0

    resolved_path = _save_resolved_config(
        tmp_path / "training",
        first.training_config,
    )
    replay_config = parse_training_config(load_yaml(resolved_path))

    second = resolve_training_config(
        replay_config,
        model_family=AUTOENCODER_FAMILY,
    )

    assert _config_dump(second.training_config) == _config_dump(
        first.training_config
    )
    assert second.run_identity == first.run_identity
    assert second.model_family == first.model_family
    assert second.model_source == first.model_source
    assert second.datamodule_source == first.datamodule_source


def test_prediction_resolved_config_roundtrip_materializes_inheritance(
    tmp_path: Path,
) -> None:
    """Inherited prediction values become explicit without changing behavior."""
    training_config = _resolved_vae_training_config()
    training_manifest_path = _write_training_manifest(
        tmp_path,
        training_config=training_config,
    )

    raw_config = load_yaml(CONFIG_DIR / "prediction_tiny_synthetic.yaml")
    raw_config["source"]["training_manifest_path"] = str(
        training_manifest_path
    )
    raw_config["dataset"] = None
    raw_config["transforms"] = None
    raw_config["data"]["batch_size"] = None
    raw_config["data"]["num_workers"] = None
    raw_config["inference"]["seed"] = None
    raw_config["inference"]["seed_workers"] = None
    raw_config["inference"]["deterministic"] = None
    raw_config["inference"]["float32_matmul_precision"] = None
    raw_config["inference"]["reconstruction_latent_source"] = None
    raw_config["exports"]["reconstructions"]["seed"] = None

    config = parse_prediction_config(raw_config)

    first = resolve_prediction_config(
        config,
        model_family=VAE_FAMILY,
    )

    assert first.inherited_config_fields == frozenset(
        {
            "dataset",
            "transforms",
            "data.batch_size",
            "data.num_workers",
            "inference.seed",
            "inference.seed_workers",
            "inference.deterministic",
            "inference.float32_matmul_precision",
            "exports.reconstructions.seed",
        }
    )
    assert first.reconstruction_latent_source == "mean"

    resolved_path = _save_resolved_config(
        tmp_path / "prediction",
        first.prediction_config,
    )
    replay_config = parse_prediction_config(load_yaml(resolved_path))

    second = resolve_prediction_config(
        replay_config,
        model_family=VAE_FAMILY,
    )

    # The replay no longer inherits these values: they are explicit in the
    # resolved config. The effective configuration and runtime choices must
    # nevertheless be identical.
    assert second.inherited_config_fields == frozenset()
    assert _config_dump(second.prediction_config) == _config_dump(
        first.prediction_config
    )
    assert second.run_identity == first.run_identity
    assert second.checkpoint_path == first.checkpoint_path
    assert second.dataset_config == first.dataset_config
    assert second.transform_configs == first.transform_configs
    assert second.datamodule_config == first.datamodule_config
    assert second.batch_size == first.batch_size
    assert second.num_workers == first.num_workers
    assert second.trainer_config == first.trainer_config
    assert second.max_batches == first.max_batches
    assert second.seed == first.seed
    assert second.seed_workers == first.seed_workers
    assert (
        second.float32_matmul_precision
        == first.float32_matmul_precision
    )
    assert (
        second.reconstruction_latent_source
        == first.reconstruction_latent_source
    )
    assert second.export_spec == first.export_spec


def test_evaluation_resolved_config_roundtrip_materializes_inputs_and_inheritance(
    tmp_path: Path,
) -> None:
    """Resolved evaluation inputs replay the same effective evaluation plan."""
    prediction_manifest_path = _write_prediction_manifest(tmp_path)

    raw_config = load_yaml(CONFIG_DIR / "evaluation_tiny_synthetic.yaml")
    raw_config["source"]["prediction_manifest_path"] = str(
        prediction_manifest_path
    )
    raw_config["source"]["embeddings_path"] = None
    raw_config["source"]["reconstructions_path"] = None
    raw_config["reconstruction"]["n_examples"] = None

    # Exercise stable resolver-owned defaults that intentionally remain
    # sentinels in the public config rather than being materialized.
    raw_config["reductions"]["pca"]["enabled"] = None
    raw_config["metrics"]["clustering"]["internal"]["enabled"] = None
    raw_config["metrics"]["predictability"]["targets"]["label"]["cv"][
        "method"
    ] = None
    raw_config["metrics"]["predictability"]["targets"]["label"]["cv"][
        "scoring"
    ] = None
    raw_config["plots"]["params"]["color_by"] = None

    config = parse_evaluation_config(raw_config)

    first = resolve_evaluation_config(config)

    assert first.inherited_config_fields == frozenset(
        {
            "run.output_root",
            "reconstruction.n_examples",
        }
    )
    assert first.evaluation_config.source.embeddings_path is not None
    assert first.evaluation_config.source.reconstructions_path is not None
    assert first.evaluation_config.run.output_root == (
        tmp_path / "outputs"
    ).resolve()
    assert first.evaluation_config.reconstruction.n_examples == 8

    resolved_path = _save_resolved_config(
        tmp_path / "evaluation",
        first.evaluation_config,
    )
    replay_config = parse_evaluation_config(load_yaml(resolved_path))

    second = resolve_evaluation_config(replay_config)

    # The replay consumes explicit paths/values from the resolved config, so
    # provenance-of-resolution differs while the effective plan stays equal.
    assert second.inherited_config_fields == frozenset()
    assert _config_dump(second.evaluation_config) == _config_dump(
        first.evaluation_config
    )
    assert second.run_identity == first.run_identity
    assert second.step_spec == first.step_spec
    assert (
        second.input_spec.prediction_manifest_path
        == first.input_spec.prediction_manifest_path
    )
    assert (
        second.input_spec.embeddings_path
        == first.input_spec.embeddings_path
    )
    assert second.input_spec.reconstructions == first.input_spec.reconstructions


def _resolved_vae_training_config():
    raw_config = load_yaml(CONFIG_DIR / "training_tiny_synthetic_vae.yaml")
    raw_config["datamodule"]["pin_memory"] = "auto"
    raw_config["checkpointing"]["monitor"] = None
    raw_config["checkpointing"]["save_top_k"] = 3

    config = parse_training_config(raw_config)
    run_spec = resolve_training_config(
        config,
        model_family=VAE_FAMILY,
    )
    return run_spec.training_config


def _write_training_manifest(
    tmp_path: Path,
    *,
    training_config,
) -> Path:
    training_dir = tmp_path / "training_source"
    checkpoint_dir = training_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    checkpoint_path = checkpoint_dir / "last.ckpt"
    checkpoint_path.touch()

    config_dir = training_dir / "records" / "configs"
    resolved_config_path = _save_resolved_config(
        config_dir,
        training_config,
    )

    output_dir = tmp_path / "outputs" / "training" / "source_run"
    manifest_path = training_dir / "training_manifest.yaml"
    _write_yaml(
        manifest_path,
        {
            "stage": "training",
            "status": "completed",
            "run": {
                "run_name": "source_run",
                "output_dir": str(output_dir),
            },
            "records": {
                "resolved_config_path": str(resolved_config_path),
            },
            "provenance": {
                "model": {
                    "source": "config",
                    "family": "vae",
                },
                "datamodule": {
                    "source": "config",
                },
            },
            "checkpoints": {
                "checkpoint_dir": str(checkpoint_dir),
                "best_checkpoint_path": str(checkpoint_path),
                "last_checkpoint_path": str(checkpoint_path),
            },
        },
    )
    return manifest_path


def _write_prediction_manifest(tmp_path: Path) -> Path:
    prediction_output_dir = (
        tmp_path / "outputs" / "prediction" / "source_run"
    )
    prediction_output_dir.mkdir(parents=True)

    embeddings_path = prediction_output_dir / "embeddings" / "embeddings.h5ad"
    embeddings_path.parent.mkdir(parents=True)
    embeddings_path.touch()

    reconstruction_dir = prediction_output_dir / "reconstructions"
    reconstruction_dir.mkdir()
    input_path = reconstruction_dir / "input.pt"
    reconstruction_path = reconstruction_dir / "reconstruction.pt"
    obs_path = reconstruction_dir / "obs.pt"
    metadata_path = reconstruction_dir / "reconstruction_export_metadata.pt"

    for path in (
        input_path,
        reconstruction_path,
        obs_path,
        metadata_path,
    ):
        path.touch()

    manifest_path = prediction_output_dir / "prediction_manifest.yaml"
    _write_yaml(
        manifest_path,
        {
            "stage": "prediction",
            "status": "completed",
            "run": {
                "run_name": "source_run",
                "output_dir": str(prediction_output_dir),
            },
            "exports": {
                "embeddings": {
                    "path": str(embeddings_path),
                },
                "reconstructions": {
                    "n_examples_exported": 8,
                    "paths": {
                        "input": str(input_path),
                        "reconstruction": str(reconstruction_path),
                        "obs": str(obs_path),
                        "metadata": str(metadata_path),
                    },
                },
            },
            "summary": {
                "project_name": "integration_test",
                "model": "vae",
                "encoder": "mlp",
                "decoder": "mlp",
            },
        },
    )
    return manifest_path


def _save_resolved_config(out_dir: Path, config) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return save_resolved_config(
        resolved_config=config,
        out_dir=out_dir,
    )


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _config_dump(config) -> dict:
    return config.model_dump(mode="json")