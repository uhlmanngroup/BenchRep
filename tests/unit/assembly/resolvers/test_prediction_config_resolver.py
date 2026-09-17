from __future__ import annotations

import pytest
from pydantic import ValidationError
from pathlib import Path

from benchrep.assembly.resolvers.prediction_config_resolver import (
    _resolve_checkpoint_path,
    _apply_composite_model_assembly_input_overrides,
    _resolve_prediction_composite_model_spec,
)
from benchrep.assembly.schemas import (
    CompositeModelAssemblyInputOverrideConfig,
    CompositeModelAssemblyStepConfig,
    PredictionConfig,
    PredictionInferenceConfig,
    TrainingConfig,
    PredictionSourceConfig,
)
from benchrep.interfaces.model_families import AUTOENCODER_FAMILY


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


def _build_assembly_config() -> dict[
    str,
    CompositeModelAssemblyStepConfig,
]:
    return {
        "encode": CompositeModelAssemblyStepConfig(
            component="encoder",
            inputs={
                "x": "expects.x",
            },
            outputs="produces.embedding",
        ),
        "parameterize": CompositeModelAssemblyStepConfig(
            component="variational_head",
            inputs={
                "x": "produces.embedding",
            },
            outputs={
                "z_sample": "produces.z_sample",
                "z_mu": "produces.z_mu",
                "z_logvar": "produces.z_logvar",
            },
        ),
        "reconstruct": CompositeModelAssemblyStepConfig(
            component="decoder",
            inputs={
                "z": "produces.z_sample",
            },
            outputs="produces.reconstruction",
        ),
    }


def _build_input_override(
    *,
    input_name: str = "z",
    reference: str = "produces.z_mu",
) -> CompositeModelAssemblyInputOverrideConfig:
    return CompositeModelAssemblyInputOverrideConfig(
        inputs={
            input_name: reference,
        },
    )


def test_composite_assembly_input_override_selects_sibling_output() -> None:
    assembly_config = _build_assembly_config()

    effective_assembly_config = (
        _apply_composite_model_assembly_input_overrides(
            assembly_config=assembly_config,
            input_overrides={
                "reconstruct": _build_input_override(),
            },
        )
    )

    assert effective_assembly_config["reconstruct"].inputs == {
        "z": "produces.z_mu",
    }

    # Prediction resolution must not mutate the embedded training config.
    assert assembly_config["reconstruct"].inputs == {
        "z": "produces.z_sample",
    }


def test_composite_assembly_input_override_rejects_unknown_step() -> None:
    with pytest.raises(
        ValueError,
        match="references unknown assembly step 'missing'",
    ):
        _apply_composite_model_assembly_input_overrides(
            assembly_config=_build_assembly_config(),
            input_overrides={
                "missing": _build_input_override(),
            },
        )


def test_composite_assembly_input_override_rejects_unknown_input() -> None:
    with pytest.raises(
        ValueError,
        match="references unknown component input 'missing'",
    ):
        _apply_composite_model_assembly_input_overrides(
            assembly_config=_build_assembly_config(),
            input_overrides={
                "reconstruct": _build_input_override(
                    input_name="missing",
                ),
            },
        )


def test_composite_assembly_input_override_rejects_model_input_route() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Only inputs currently routed from a `produces.\\*` output "
            "can be overridden"
        ),
    ):
        _apply_composite_model_assembly_input_overrides(
            assembly_config=_build_assembly_config(),
            input_overrides={
                "encode": _build_input_override(
                    input_name="x",
                ),
            },
        )


def test_composite_assembly_input_override_rejects_unknown_output() -> None:
    with pytest.raises(
        ValueError,
        match="references unknown model output 'produces.missing'",
    ):
        _apply_composite_model_assembly_input_overrides(
            assembly_config=_build_assembly_config(),
            input_overrides={
                "reconstruct": _build_input_override(
                    reference="produces.missing",
                ),
            },
        )


def test_composite_assembly_input_override_rejects_different_producer() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "must select another output from the same producer step "
            "'parameterize'"
        ),
    ):
        _apply_composite_model_assembly_input_overrides(
            assembly_config=_build_assembly_config(),
            input_overrides={
                "reconstruct": _build_input_override(
                    reference="produces.embedding",
                ),
            },
        )


def test_composite_assembly_input_overrides_rejected_for_canonical_model() -> None:
    prediction_config = PredictionConfig.model_construct(
        inference=PredictionInferenceConfig(
            composite_model_assembly_input_overrides={
                "reconstruct": _build_input_override(),
            },
        ),
    )

    with pytest.raises(
        ValueError,
        match="supported only for composite models",
    ):
        _resolve_prediction_composite_model_spec(
            prediction_config=prediction_config,
            training_config=TrainingConfig.model_construct(),
            model_family=AUTOENCODER_FAMILY,
            model_is_external=False,
        )


def test_canonical_and_composite_inference_settings_are_mutually_exclusive(
) -> None:
    with pytest.raises(
        ValidationError,
        match="cannot be configured together",
    ):
        PredictionInferenceConfig(
            canonical_vae_reconstruction_latent_source="mean",
            composite_model_assembly_input_overrides={
                "reconstruct": _build_input_override(),
            },
        )