"""YOYO wizard checks and exported YAML are exercised through actual workflows."""

from pathlib import Path

import pytest
import yaml

pytest.importorskip("dash_mantine_components")
import anndata as ad
from lightning.fabric.accelerators import (
    CUDAAccelerator as FabricCUDAAccelerator,
    MPSAccelerator as FabricMPSAccelerator,
)
from lightning.pytorch.accelerators import CUDAAccelerator, MPSAccelerator

from benchrep import train_yoyo, predict_yoyo, TrainingOverrides, PredictionOverrides
from benchrep.gui.train import create_training_app
from benchrep.gui.predict import create_prediction_app
from benchrep.gui.rules.train import (
    prepare_training_draft,
    training_sections,
    training_review_session,
)
from benchrep.gui.rules.predict import (
    prepare_prediction_draft,
    PredictionSourceContext,
    prediction_sections,
    prediction_review_session,
)
from benchrep.gui.shared.forms import evaluate_section
from benchrep.gui.shared.state import MISSING, editor_spec, new_node
from benchrep.interfaces.model_families import YOYO_FAMILY, AUTOENCODER_FAMILY
from tests.fixtures.models import TinyYOYOModel
from tests.fixtures.datamodules import ParameterizedExternalDataModule


@pytest.fixture(autouse=True)
def cpu_runs(tmp_path, monkeypatch):
    """Keep wizard-exported workflows on CPU regardless of the host's GPUs."""
    monkeypatch.chdir(tmp_path)
    # Auto-selection uses Fabric; Trainer validation uses PyTorch Lightning.
    for accelerator in (
        FabricCUDAAccelerator,
        FabricMPSAccelerator,
        CUDAAccelerator,
        MPSAccelerator,
    ):
        monkeypatch.setattr(accelerator, "is_available", staticmethod(lambda: False))


def datamodule():
    return ParameterizedExternalDataModule(
        train_samples=8, val_samples=4, predict_samples=7, batch_size=3
    )


def training_draft():
    return {
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
        "logger": None,
        "inspection": {"torchview": {"enabled": False}},
    }


def prediction_draft():
    return {
        "exports": {
            "anndata": {
                "mode": "custom",
                "keys": ["features", "score"],
                "primary_key": "features",
            }
        }
    }


def test_training_wizard_preserves_omitted_monitor_and_exports_executable_yaml(
    tmp_path,
):
    overrides = TrainingOverrides(model=TinyYOYOModel(), datamodule=datamodule())
    review = training_review_session(model_family=YOYO_FAMILY, overrides=overrides)
    raw = training_draft()
    directory = str(tmp_path / "training_yaml")
    report = review.check(raw, directory, "training.yaml", {})
    assert report["ready"], report
    assert review.resolved.model_source == "external_instance"
    assert review.resolved.training_config.checkpointing.monitor is None
    review.save(raw, directory, "training.yaml", {})
    exported = review.result
    authored = yaml.safe_load(exported.config_path.read_text())
    assert "monitor" not in authored.get("checkpointing", {})
    trained = train_yoyo(config=exported.config_path, overrides=exported.overrides)
    assert trained.config.checkpointing.monitor is None
    assert Path(trained.checkpoint_callback.last_model_path).is_file()


def test_standalone_prediction_wizard_roundtrip_does_not_invent_checkpoint_or_manifest(
    tmp_path,
):
    overrides = PredictionOverrides(model=TinyYOYOModel(), datamodule=datamodule())
    context = PredictionSourceContext(model_family=YOYO_FAMILY, overrides=overrides)
    raw = prediction_draft()
    context.load(raw)
    assert context.error is None and context.training is None
    assert context.validate_source({}) == {}
    assert (
        context.presentation(("exports", "anndata", "keys", 0), None, raw).suggestions
        is None
    )
    assert context.presentation(
        ("exports", "anndata", "primary_key"), None, raw
    ).suggestions == ("features", "score")
    review = prediction_review_session(context=context)
    directory = str(tmp_path / "prediction_yaml")
    report = review.check(raw, directory, "prediction.yaml", {})
    assert report["ready"], report
    assert report["summary"]["checkpoint"] is None
    assert report["summary"]["checkpoint_source"] == "runtime_model"
    review.save(raw, directory, "prediction.yaml", {})
    exported = review.result
    authored = yaml.safe_load(exported.config_path.read_text())
    assert "source" not in authored
    resolved = yaml.safe_load(exported.resolved_config_path.read_text())
    assert resolved["source"]["checkpoint"] is None
    for path in (exported.config_path, exported.resolved_config_path):
        result = predict_yoyo(config=path, overrides=exported.overrides)
        assert result.trainer.accelerator.name() == "cpu"
        assert ad.read_h5ad(result.export_result.anndata.path).n_obs == 7


def test_linked_source_clear_drops_inheritance_and_review_detects_manifest_changes(
    tmp_path,
):
    overrides = TrainingOverrides(model=TinyYOYOModel(), datamodule=datamodule())
    training_review = training_review_session(
        model_family=YOYO_FAMILY, overrides=overrides
    )
    directory = str(tmp_path / "train")
    raw_training = training_draft()
    assert training_review.check(raw_training, directory, "training.yaml", {})["ready"]
    training_review.save(raw_training, directory, "training.yaml", {})
    configured = training_review.result
    trained = train_yoyo(config=configured.config, overrides=configured.overrides)
    context = PredictionSourceContext(
        model_family=YOYO_FAMILY,
        overrides=PredictionOverrides(model=TinyYOYOModel(), datamodule=datamodule()),
    )
    raw = prediction_draft()
    raw["source"] = {"training_manifest_path": str(trained.manifest_path)}
    context.load(raw)
    assert context.current(raw)
    assert context.inherited[("inference", "seed")] == 137
    review = prediction_review_session(context=context)
    directory = str(tmp_path / "prediction")
    report = review.check(raw, directory, "prediction.yaml", {})
    assert report["ready"], report
    assert report["summary"]["checkpoint_source"] == "training_manifest_last"
    trained.manifest_path.write_text(
        trained.manifest_path.read_text() + "\n# changed\n"
    )
    with pytest.raises(ValueError, match="source changed"):
        review.save(raw, directory, "prediction.yaml", {})
    raw.pop("source")
    assert not context.current(raw)
    assert context.presentation(("dataset",), None, raw).inherited_value is MISSING
    context.load(raw)
    assert (
        context.inherited == {} and context.training is None and context.error is None
    )
    assert (
        review.check(raw, directory, "prediction.yaml", {})["summary"][
            "checkpoint_source"
        ]
        == "runtime_model"
    )


@pytest.mark.parametrize("stage", ["training", "prediction"])
def test_app_renders_without_constructing_runtime_classes(stage):
    class NeverConstruct(TinyYOYOModel):
        def __init__(self):
            raise AssertionError("Wizard constructed the model")

    if stage == "training":
        app = create_training_app(
            model_family=YOYO_FAMILY,
            overrides=TrainingOverrides(
                model=NeverConstruct, datamodule=ParameterizedExternalDataModule
            ),
        )
        raw = training_draft()
    else:
        app = create_prediction_app(
            model_family=YOYO_FAMILY,
            overrides=PredictionOverrides(
                model=NeverConstruct, datamodule=ParameterizedExternalDataModule
            ),
        )
        raw = prediction_draft()
    client = app.server.test_client()
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/_dash-dependencies").status_code == 200
    report = app.wizard_session.check(
        raw, str(Path("export") / stage), f"{stage}.yaml", {}
    )
    assert report["ready"], report


def test_yoyo_sections_hide_model_configuration_but_allow_configured_data():
    overrides = TrainingOverrides(model=TinyYOYOModel())
    initial = prepare_training_draft(
        model_family=YOYO_FAMILY, config=None, overrides=overrides
    )
    sections = {
        s.key: s
        for s in training_sections(
            model_family=YOYO_FAMILY, overrides=overrides, initial=initial
        )
    }
    assert (
        not {
            "model",
            "encoder",
            "decoder",
            "losses",
            "optimizer",
            "composite_model_assembly",
        }
        & sections.keys()
    )
    assert {
        "dataset",
        "datamodule",
        "transform_pipelines",
        "checkpointing",
    } <= sections.keys()
    guide = sections["transform_pipelines"].presentation(
        ("transform_pipelines", 0, "input"), None, {}
    )
    assert guide.suggestions is None and "explicit input" in guide.note
    guide = sections["checkpointing"].presentation(
        ("checkpointing", "monitor"), None, {}
    )
    assert "last-only" in guide.note
    context = PredictionSourceContext(
        model_family=YOYO_FAMILY, overrides=PredictionOverrides(model=TinyYOYOModel())
    )
    assert (
        context.presentation(("transform_pipelines", 0, "input"), None, {}).suggestions
        is None
    )
    assert (
        context.presentation(
            ("exports", "reconstructions", "pairs", "p", "input"), None, {}
        ).suggestions
        is None
    )


def test_family_specific_source_and_export_errors_are_shown_before_execution():
    context = PredictionSourceContext(
        model_family=YOYO_FAMILY,
        overrides=PredictionOverrides(model=TinyYOYOModel(), datamodule=datamodule()),
    )
    sections = {s.key: s for s in prediction_sections(context=context)}
    exports = sections["exports"]
    spec = editor_spec(exports.annotation, exports.field_info)
    initial = prepare_prediction_draft(
        model_family=YOYO_FAMILY, config=None, overrides=context.overrides
    )
    _, errors = evaluate_section(exports, spec, new_node(spec, {}), initial=initial)
    assert "custom" in " ".join(errors.values())
    raw = prediction_draft()
    raw["source"] = {"checkpoint": "best"}
    report = prediction_review_session(context=context).check(
        raw, "export", "prediction.yaml", {}
    )
    assert not report["ready"]
    assert "manifest" in " ".join(row["message"] for row in report["rows"])
    canonical = PredictionSourceContext(
        model_family=AUTOENCODER_FAMILY, overrides=PredictionOverrides()
    )
    assert canonical.validate_source({})
    assert ("checkpoint",) in canonical.validate_source({"checkpoint": None})


def test_missing_runtime_model_is_rejected_when_launching_either_wizard():
    with pytest.raises(ValueError, match="model"):
        prepare_training_draft(
            model_family=YOYO_FAMILY, config=None, overrides=TrainingOverrides()
        )
    with pytest.raises(ValueError, match="model"):
        prepare_prediction_draft(
            model_family=YOYO_FAMILY, config=None, overrides=PredictionOverrides()
        )


@pytest.mark.parametrize("stage", ["training", "prediction"])
def test_browser_yoyo_review_export_and_execute(stage, tmp_path, traced_page):
    """Exercise the real Dash form through saving and executing its authored YAML."""
    import os
    from threading import Thread
    from werkzeug.serving import make_server

    playwright = pytest.importorskip("playwright.sync_api")
    raw = training_draft() if stage == "training" else prediction_draft()
    config_path = tmp_path / f"initial_{stage}.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    if stage == "training":
        overrides = TrainingOverrides(model=TinyYOYOModel(), datamodule=datamodule())
        app = create_training_app(
            model_family=YOYO_FAMILY,
            config=config_path,
            overrides=overrides,
            export_directory=tmp_path / "export",
        )
    else:
        overrides = PredictionOverrides(model=TinyYOYOModel(), datamodule=datamodule())
        app = create_prediction_app(
            model_family=YOYO_FAMILY,
            config=config_path,
            overrides=overrides,
            export_directory=tmp_path / "export",
        )
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as manager:
            browser = manager.chromium.launch(
                executable_path=os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            with traced_page(browser, viewport={"width": 1600, "height": 1000}) as page:
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}")
                page.wait_for_function("window.brFieldInteractionsInstalled === true")
                if stage == "prediction":
                    playwright.expect(page.locator(".br-source-context")).to_contain_text(
                        "Standalone prediction"
                    )
                    playwright.expect(
                        page.get_by_role("button", name="Load source", exact=True)
                    ).to_be_disabled()
                playwright.expect(
                    page.get_by_role("tab", name="Dataset", exact=False)
                ).to_have_count(0)
                page.get_by_role("tab", name="Review & finish", exact=False).click()
                page.get_by_role("button", name="Check configuration", exact=True).click()
                playwright.expect(
                    page.get_by_role("button", name="Save & finish", exact=True)
                ).to_be_enabled(timeout=15000)
                page.get_by_role("button", name="Save & finish", exact=True).click()
                playwright.expect(page.locator("#wizard-save-message")).to_contain_text(
                    "Authored configuration:"
                )
                assert not errors
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
    exported = app.wizard_session.result
    authored = yaml.safe_load(exported.config_path.read_text())
    if stage == "training":
        assert "monitor" not in authored.get("checkpointing", {})
        result = train_yoyo(config=exported.config_path, overrides=exported.overrides)
        assert Path(result.checkpoint_callback.last_model_path).is_file()
    else:
        assert "source" not in authored
        result = predict_yoyo(config=exported.config_path, overrides=exported.overrides)
        assert ad.read_h5ad(result.export_result.anndata.path).n_obs == 7
