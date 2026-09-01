from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchrep.assembly.config import load_yaml
from benchrep.assembly.registries.core import DATASETS
from benchrep.assembly.schemas import (
    EvaluationSourceConfig,
    PredictionSourceConfig,
)
from benchrep.workflows import evaluate, predict_vae, train_vae
from tests.fixtures.datasets import TinySyntheticDataset


CONFIG_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "configs"


def test_resolved_configs_reproduce_end_to_end_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh chain rebuilt from resolved configs reproduces final metrics.

    Resolved prediction/evaluation configs deliberately pin the concrete upstream
    checkpoint/artifact paths used by the original run. For a *fresh* second
    train -> prediction -> evaluation chain, this test therefore rebinds only
    those upstream-linkage source sections to the newly produced manifests.
    Every other setting comes from the first chain's resolved configs.
    """
    if "tiny_synthetic" not in DATASETS.keys():
        DATASETS.register("tiny_synthetic", TinySyntheticDataset)

    monkeypatch.chdir(tmp_path)

    original_training_config = _write_resolution_heavy_training_config(
        tmp_path
    )
    original_prediction_config = _write_inheritance_heavy_prediction_config(
        tmp_path
    )
    original_evaluation_config = _write_inheritance_heavy_evaluation_config(
        tmp_path
    )

    first_training = train_vae(config_path=original_training_config)
    first_prediction = predict_vae(
        config_path=original_prediction_config,
        training_manifest_path=first_training.manifest_path,
    )
    first_evaluation = evaluate(
        config_path=original_evaluation_config,
        prediction_manifest_path=first_prediction.manifest_path,
    )

    _assert_chain_completed(
        first_training,
        first_prediction,
        first_evaluation,
    )

    first_training_resolved = _resolved_config_path(first_training)
    first_prediction_resolved = _resolved_config_path(first_prediction)
    first_evaluation_resolved = _resolved_config_path(first_evaluation)

    # Confirm that this chain actually exercised the resolution behavior the
    # reproducibility test is intended to cover.
    training_resolved = load_yaml(first_training_resolved)
    prediction_resolved = load_yaml(first_prediction_resolved)
    evaluation_resolved = load_yaml(first_evaluation_resolved)

    assert training_resolved["datamodule"]["pin_memory"] in {True, False}
    assert training_resolved["checkpointing"]["save_top_k"] == 0

    assert prediction_resolved["dataset"] is not None
    assert prediction_resolved["transforms"] is not None
    assert prediction_resolved["data"]["batch_size"] == 8
    assert prediction_resolved["data"]["num_workers"] == 0
    assert prediction_resolved["inference"]["seed"] == 137
    assert prediction_resolved["inference"]["seed_workers"] is True
    assert prediction_resolved["inference"]["deterministic"] is True
    assert (
        prediction_resolved["inference"]["float32_matmul_precision"]
        == "highest"
    )
    assert prediction_resolved["exports"]["reconstructions"]["seed"] == 137

    assert evaluation_resolved["source"]["embeddings_path"] is not None
    assert evaluation_resolved["source"]["reconstructions_path"] is not None
    assert evaluation_resolved["run"]["output_root"] is not None
    assert evaluation_resolved["reconstruction"]["n_examples"] == 8

    # Replay training from its exact resolved config.
    second_training = train_vae(config_path=first_training_resolved)

    # The resolved prediction config pins run 1's exact checkpoint path. Rebind
    # the source section so this replay consumes the freshly reproduced training
    # run while retaining all other resolved prediction settings.
    second_prediction = predict_vae(
        config_path=first_prediction_resolved,
        config_components={
            "source": PredictionSourceConfig(
                training_manifest_path=second_training.manifest_path,
                checkpoint="last",
            ),
        },
    )

    # Likewise, the resolved evaluation config pins run 1's concrete artifact
    # paths. Rebind only its source section so evaluation discovers the freshly
    # reproduced prediction artifacts from the second prediction manifest.
    second_evaluation = evaluate(
        config_path=first_evaluation_resolved,
        config_components={
            "source": EvaluationSourceConfig(
                prediction_manifest_path=second_prediction.manifest_path,
            ),
        },
    )

    _assert_chain_completed(
        second_training,
        second_prediction,
        second_evaluation,
    )

    first_metrics = _load_json(
        first_evaluation.export_paths.metrics_json_path
    )
    second_metrics = _load_json(
        second_evaluation.export_paths.metrics_json_path
    )

    _assert_json_equivalent(first_metrics, second_metrics)


def _write_resolution_heavy_training_config(tmp_path: Path) -> Path:
    raw = load_yaml(CONFIG_DIR / "training_tiny_synthetic_vae.yaml")

    # BenchRep-owned dynamic decisions that must be materialized.
    raw["datamodule"]["pin_memory"] = "auto"
    raw["checkpointing"]["monitor"] = None
    raw["checkpointing"]["save_top_k"] = 1
    raw["checkpointing"]["save_last"] = True

    path = tmp_path / "roundtrip_training.yaml"
    _write_yaml(path, raw)
    return path


def _write_inheritance_heavy_prediction_config(tmp_path: Path) -> Path:
    raw = load_yaml(CONFIG_DIR / "prediction_tiny_synthetic.yaml")

    # Use last.ckpt because the training config above deliberately disables
    # metric-ranked checkpoints by resolving save_top_k -> 0.
    raw["source"]["checkpoint"] = "last"
    raw["dataset"] = None
    raw["transforms"] = None
    raw["data"]["batch_size"] = None
    raw["data"]["num_workers"] = None
    raw["inference"]["seed"] = None
    raw["inference"]["seed_workers"] = None
    raw["inference"]["deterministic"] = None
    raw["inference"]["float32_matmul_precision"] = None
    raw["inference"]["reconstruction_latent_source"] = None
    raw["exports"]["reconstructions"]["seed"] = None

    path = tmp_path / "roundtrip_prediction.yaml"
    _write_yaml(path, raw)
    return path


def _write_inheritance_heavy_evaluation_config(tmp_path: Path) -> Path:
    raw = load_yaml(CONFIG_DIR / "evaluation_tiny_synthetic.yaml")

    # Prediction manifest discovery supplies both artifact paths. Evaluation also
    # inherits output_root and reconstruction.n_examples from that prediction.
    raw["source"]["prediction_manifest_path"] = None
    raw["source"]["embeddings_path"] = None
    raw["source"]["reconstructions_path"] = None
    raw["reconstruction"]["n_examples"] = None

    # Leave a few resolver defaults implicit to prove the replay still resolves
    # them the same way under the same BenchRep version.
    raw["reductions"]["pca"]["enabled"] = None
    raw["metrics"]["clustering"]["internal"]["enabled"] = None
    raw["metrics"]["predictability"]["targets"]["label"]["cv"][
        "method"
    ] = None
    raw["metrics"]["predictability"]["targets"]["label"]["cv"][
        "scoring"
    ] = None

    path = tmp_path / "roundtrip_evaluation.yaml"
    _write_yaml(path, raw)
    return path


def _resolved_config_path(result: Any) -> Path:
    path = result.run_context.config_dir / "resolved_config.yaml"
    assert path.is_file()
    return path


def _assert_chain_completed(training, prediction, evaluation) -> None:
    assert training.status_report.status == "completed"
    assert prediction.status_report.status == "completed"
    assert evaluation.status_report.status == "completed"

    assert evaluation.export_paths.metrics_json_path is not None
    assert evaluation.export_paths.metrics_json_path.is_file()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _assert_json_equivalent(
    first: Any,
    second: Any,
    *,
    path: str = "metrics",
) -> None:
    """Recursively compare JSON structure with tight numeric tolerance."""
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        assert set(second) == set(first), f"Different keys at {path}"
        for key in first:
            _assert_json_equivalent(
                first[key],
                second[key],
                path=f"{path}.{key}",
            )
        return

    if (
        isinstance(first, Sequence)
        and isinstance(second, Sequence)
        and not isinstance(first, str | bytes)
        and not isinstance(second, str | bytes)
    ):
        assert len(second) == len(first), f"Different length at {path}"
        for index, (first_item, second_item) in enumerate(zip(first, second)):
            _assert_json_equivalent(
                first_item,
                second_item,
                path=f"{path}[{index}]",
            )
        return

    if (
        isinstance(first, Real)
        and isinstance(second, Real)
        and not isinstance(first, bool)
        and not isinstance(second, bool)
    ):
        first_value = float(first)
        second_value = float(second)

        if math.isnan(first_value) or math.isnan(second_value):
            assert math.isnan(first_value) and math.isnan(second_value), (
                f"NaN mismatch at {path}: {first!r} != {second!r}"
            )
            return

        assert second_value == pytest.approx(
            first_value,
            rel=1e-7,
            abs=1e-9,
        ), f"Numeric mismatch at {path}"
        return

    assert second == first, f"Mismatch at {path}: {first!r} != {second!r}"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)