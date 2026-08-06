from pathlib import Path

import pytest
from pydantic import ValidationError

from benchrep.assembly.resolvers.prediction_config_resolver import (
    _resolve_checkpoint_path,
)
from benchrep.assembly.schemas import PredictionSourceConfig


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("best", "best"),
        ("last", "last"),
        ("epoch=001-step=10.ckpt", Path("epoch=001-step=10.ckpt")),
    ],
)
def test_prediction_source_accepts_checkpoint_selectors(
    configured: str,
    expected: str | Path,
) -> None:
    source = PredictionSourceConfig(checkpoint=configured)

    assert source.checkpoint == expected


def test_prediction_source_accepts_absolute_checkpoint_path(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "external.ckpt"

    source = PredictionSourceConfig(checkpoint=checkpoint_path)

    assert source.checkpoint == checkpoint_path.resolve()


@pytest.mark.parametrize(
    "checkpoint",
    [
        Path("other/checkpoint.ckpt"),
        Path("../checkpoint.ckpt"),
        Path("checkpoint.pt"),
    ],
)
def test_prediction_source_rejects_invalid_checkpoint_selection(
    checkpoint: Path,
) -> None:
    with pytest.raises(ValidationError):
        PredictionSourceConfig(checkpoint=checkpoint)


def test_resolve_checkpoint_path_supports_all_sources(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    best_path = checkpoint_dir / "best.ckpt"
    last_path = checkpoint_dir / "last.ckpt"
    named_path = checkpoint_dir / "epoch=001-step=10.ckpt"

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_path = external_dir / "replacement.ckpt"

    for path in (best_path, last_path, named_path, external_path):
        path.touch()

    training_manifest = {
        "checkpoints": {
            "checkpoint_dir": str(checkpoint_dir),
            "best_checkpoint_path": str(best_path),
            "last_checkpoint_path": str(last_path),
        },
    }
    manifest_path = tmp_path / "training_manifest.yaml"

    cases = [
        (
            "best",
            best_path,
            "training_manifest_best",
        ),
        (
            "last",
            last_path,
            "training_manifest_last",
        ),
        (
            Path(named_path.name),
            named_path,
            "training_checkpoint_filename",
        ),
        (
            external_path,
            external_path,
            "explicit_path",
        ),
    ]

    for selection, expected_path, expected_source in cases:
        resolved_path, source = _resolve_checkpoint_path(
            checkpoint=selection,
            training_manifest=training_manifest,
            manifest_path=manifest_path,
        )

        assert resolved_path == expected_path.resolve()
        assert source == expected_source