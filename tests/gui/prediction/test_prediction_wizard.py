"""Prediction authoring, inheritance, suggestions, and resolver-backed export."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("dash_mantine_components")
from benchrep.assembly.config import load_yaml
from benchrep.assembly.registries.builtins import register_builtins
from benchrep.assembly.resolvers.training_config_resolver import resolve_training_config
from benchrep.assembly.schemas.parsing import parse_training_config
from benchrep.assembly.schemas.prediction_config_schema import (
    PredictionDataConfig,
)
from benchrep.gui.predict import create_prediction_app
from benchrep.gui.rules.predict import (
    PredictionSourceContext,
    prepare_prediction_draft,
    prediction_sections,
    prediction_review_session,
)
from benchrep.gui.shared.forms import evaluate_section
from benchrep.gui.shared.state import MISSING, editor_spec, new_node
from benchrep.interfaces.model_families import (
    COMPOSITE_FAMILY,
    AUTOENCODER_FAMILY,
    VAE_FAMILY,
)
from benchrep.utils.workflow_overrides import PredictionOverrides
import lightning as L


@pytest.fixture
def source_factory(tmp_path):
    register_builtins()

    def make(family=COMPOSITE_FAMILY, name="source", external_dm=False):
        raw = load_yaml(
            Path(__file__).parents[3]
            / "examples/configs/training_mnist_composite_multitask.yaml"
        )
        if family != COMPOSITE_FAMILY:
            for key in list(raw):
                if key.startswith("composite_"):
                    raw.pop(key)
            raw["model"] = {"name": family.name}
            if family == VAE_FAMILY:
                raw["model"]["params"] = {"latent_dim": 8}
            raw["encoder"] = {
                "name": "mlp",
                "params": {"input_shape": [1, 28, 28], "output_dim": 8},
            }
            raw["decoder"] = {"name": "mlp", "params": {"output_shape": [1, 28, 28]}}
            raw["losses"] = {"reconstruction": [{"name": "mse"}]}
            if family == VAE_FAMILY:
                raw["losses"]["regularization"] = [{"name": "gaussian_kl"}]
            raw.pop("inspection", None)
        if external_dm:
            for key in ("dataset", "transform_pipelines", "datamodule"):
                raw.pop(key, None)
        config = parse_training_config(raw, datamodule_overridden=external_dm)
        spec = resolve_training_config(
            config,
            model_family=family,
            datamodule_source="external_instance" if external_dm else "config",
        )
        directory = tmp_path / name
        directory.mkdir()
        checkpoint = directory / "best.ckpt"
        checkpoint.touch()  # Configuration checks inspect existence, never load weights.
        manifest = {
            "stage": "training",
            "status": "completed",
            "run": {"run_name": name, "output_dir": str(directory)},
            "construction": {
                "model": {"source": "config", "family": family.name},
                "datamodule": {
                    "source": "external_instance" if external_dm else "config"
                },
            },
            "records": {
                "resolved_config_path": str(directory / "resolved_training_config.yaml")
            },
            "appendix": {
                "resolved_config": spec.training_config.model_dump(mode="json")
            },
            "checkpoints": {
                "checkpoint_dir": str(directory),
                "best_checkpoint_path": str(checkpoint),
                "last_checkpoint_path": str(checkpoint),
            },
        }
        path = directory / "training_manifest.yaml"
        path.write_text(yaml.safe_dump(manifest))
        return path, manifest

    return make


def context_for(path, family=COMPOSITE_FAMILY, overrides=None):
    context = PredictionSourceContext(
        model_family=family, overrides=overrides or PredictionOverrides()
    )
    draft = {"source": {"training_manifest_path": str(path)}}
    context.load(draft)
    assert context.error is None, context.error
    return context, draft


def test_source_load_is_independent_of_unfinished_exports_and_checkpoint(
    source_factory,
):
    path, manifest = source_factory()
    Path(manifest["checkpoints"]["best_checkpoint_path"]).unlink()
    context, draft = context_for(path)
    assert context.current(draft)
    assert context.inherited[("data", "batch_size")] == 128
    assert (
        context.inherited[("transform_pipelines",)][0]["steps"][0]["name"] == "to_dtype"
    )
    assert "apply_to" not in context.inherited[("transform_pipelines",)][0]["steps"][0]
    assert "dataset" not in draft


def test_inheritance_and_explicit_copy_do_not_change_omission(source_factory):
    path, _ = source_factory()
    context, draft = context_for(path)
    before = deepcopy(draft)
    guide = context.presentation(("dataset",), None, draft)
    assert "Inherited" in guide.note
    assert guide.inherited_value["params"]["split"] == "train"
    copied = context.action("copy_inherited", ("dataset",), draft)
    copied["params"]["split"] = "test"
    assert context.inherited[("dataset",)]["params"]["split"] == "train"
    assert draft == before
    for value in ([], [{"steps": []}]):
        assert (
            context.presentation(
                ("transform_pipelines",), value, {**draft, "transform_pipelines": value}
            ).inherited_value
            is MISSING
        )


def test_source_path_change_and_failed_reload_drop_suggestions(
    source_factory, tmp_path
):
    path, _ = source_factory()
    context, draft = context_for(path)
    key = ("inference", "composite_model_assembly_input_overrides")
    assert "reconstruct" in context.presentation(key, None, draft).key_suggestions
    changed = {"source": {"training_manifest_path": str(tmp_path / "absent.yaml")}}
    assert not context.current(changed)
    assert context.presentation(key, None, changed).key_suggestions is None
    context.load(changed)
    assert context.error and context.training is None and context.inherited == {}
    assert not context.current(draft)


def test_override_path_is_authoritative_and_only_manifest_is_locked(source_factory):
    path, _ = source_factory()
    overrides = PredictionOverrides(training_manifest_path=path)
    context, draft = context_for(path, overrides=overrides)
    assert context.path({"source": {"training_manifest_path": "other.yaml"}}) == path
    source = prediction_sections(context=context)[0]
    assert source.locked_paths == frozenset({("source", "training_manifest_path")})
    assert ("source", "checkpoint") not in source.locked_paths


def test_composite_suggestions_follow_producer_and_export_contracts(source_factory):
    path, _ = source_factory()
    context, draft = context_for(path)
    base = ("inference", "composite_model_assembly_input_overrides")
    assert "encode" not in context.presentation(base, None, draft).key_suggestions
    assert context.presentation(
        (*base, "reconstruct", "inputs"), None, draft
    ).key_suggestions == ("z",)
    guide = context.presentation((*base, "reconstruct", "inputs", "z"), None, draft)
    assert set(guide.suggestions) == {
        "produces.z_sample",
        "produces.z_mu",
        "produces.z_logvar",
    }
    assert "produces.z_sample" in guide.note
    keys = context.presentation(
        ("exports", "anndata", "keys", 0), None, draft
    ).suggestions
    assert "z_mu" in keys and "reconstruction" not in keys
    custom = {**draft, "exports": {"anndata": {"mode": "custom", "keys": ["z_mu"]}}}
    assert context.presentation(
        ("exports", "anndata", "primary_key"), None, custom
    ).suggestions == ("z_mu",)
    assert context.presentation(
        ("exports", "reconstructions", "pairs", "p", "input"), None, draft
    ).suggestions == ("x",)
    assert context.presentation(
        ("exports", "reconstructions", "pairs", "p", "reconstruction"), None, draft
    ).suggestions == ("reconstruction",)
    assert set(
        context.presentation(
            ("exports", "reconstructions", "stratify_by"), None, draft
        ).suggestions
    ) == {"label", "sample_id"}


@pytest.mark.parametrize("family", [AUTOENCODER_FAMILY, VAE_FAMILY, COMPOSITE_FAMILY])
def test_complete_review_preserves_authored_and_saves_resolved(
    source_factory, tmp_path, family
):
    path, _ = source_factory(family)
    context, draft = context_for(path, family)
    if family == COMPOSITE_FAMILY:
        draft["exports"] = {"anndata": {"primary_key": "z_mu"}}
        draft["inference"] = {
            "composite_model_assembly_input_overrides": {
                "reconstruct": {"inputs": {"z": "produces.z_mu"}}
            }
        }
    session = prediction_review_session(context=context)
    report = session.check(draft, str(tmp_path / "export"), "prediction.yaml", {})
    assert report["ready"], report
    assert "dataset" not in yaml.safe_load(report["authored_yaml"])
    assert "dataset" in yaml.safe_load(report["resolved_yaml"])
    assert "dataset" in report["summary"]["inherited_fields"]
    if family == COMPOSITE_FAMILY:
        assert session.resolved.composite_model_spec is not None
    authored, resolved = session.save(
        draft, str(tmp_path / "export"), "prediction.yaml", {}
    )
    assert authored.name == "authored_prediction_config.yaml"
    assert resolved.name == "resolved_prediction_config.yaml"
    assert yaml.safe_load(authored.read_text()) == draft


def test_source_modified_after_check_cannot_save(source_factory, tmp_path):
    path, _ = source_factory(AUTOENCODER_FAMILY)
    context, draft = context_for(path, AUTOENCODER_FAMILY)
    session = prediction_review_session(context=context)
    assert session.check(draft, str(tmp_path / "export"), "prediction.yaml", {})[
        "ready"
    ]
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        session.save(draft, str(tmp_path / "export"), "prediction.yaml", {})


def test_external_datamodule_keeps_max_batches_and_has_no_inherited_dataset(
    source_factory, tmp_path
):
    path, _ = source_factory(AUTOENCODER_FAMILY, external_dm=True)
    context, draft = context_for(path, AUTOENCODER_FAMILY)
    assert ("dataset",) not in context.inherited
    assert (
        "schema default"
        in context.presentation(("data", "batch_size"), None, draft).note
    )
    overrides = PredictionOverrides(
        datamodule=L.LightningDataModule, training_manifest_path=path
    )
    initial = prepare_prediction_draft(
        model_family=AUTOENCODER_FAMILY, config=None, overrides=overrides
    )
    context = PredictionSourceContext(
        model_family=AUTOENCODER_FAMILY, overrides=overrides
    )
    context.load(initial.initial_data)
    sections = prediction_sections(context=context)
    assert "dataset" not in {s.key for s in sections}
    assert "transform_pipelines" not in {s.key for s in sections}
    assert "overrides" in {s.key for s in sections}
    assert context.presentation(("data",), None, {}).excluded_fields == (
        "batch_size",
        "num_workers",
    )
    session = prediction_review_session(context=context)
    report = session.check(
        {"data": {"max_batches": 2}}, str(tmp_path / "export"), "prediction.yaml", {}
    )
    assert report["ready"], report
    assert session.resolved.max_batches == 2


def test_external_model_metadata_is_free_text_but_export_keys_are_known():
    context = PredictionSourceContext(
        model_family=VAE_FAMILY, overrides=PredictionOverrides(model=object())
    )
    assert (
        context.presentation(
            ("exports", "reconstructions", "stratify_by"), None, {}
        ).suggestions
        is None
    )
    assert (
        "z_mu"
        in context.presentation(("exports", "anndata", "keys", 0), None, {}).suggestions
    )
    assert context.presentation(
        ("inference", "canonical_vae_reconstruction_latent_source"), None, {}
    ).hidden


def test_partial_yaml_and_components_preserve_presence(tmp_path):
    path = tmp_path / "prediction.yaml"
    raw = {
        "stage": "prediction",
        "dataset": None,
        "transform_pipelines": [],
        "data": {"num_workers": 0},
    }
    path.write_text(yaml.safe_dump(raw))
    initial = prepare_prediction_draft(
        model_family=AUTOENCODER_FAMILY, config=path, overrides=PredictionOverrides()
    )
    assert initial.initial_data == raw
    overridden = prepare_prediction_draft(
        model_family=AUTOENCODER_FAMILY,
        config=path,
        overrides=PredictionOverrides(
            config_components={"data": PredictionDataConfig(batch_size=7)}
        ),
    )
    assert "data" in overridden.supplied_sections
    assert overridden.initial_data["data"]["batch_size"] == 7
    context = PredictionSourceContext(
        model_family=AUTOENCODER_FAMILY, overrides=PredictionOverrides()
    )
    for section in prediction_sections(context=context):
        spec = editor_spec(section.annotation, section.field_info)
        value = raw.get(section.key, MISSING)
        collected, _ = evaluate_section(
            section, spec, new_node(spec, value), initial=initial
        )
        assert collected == value


def test_app_builds_without_source():
    app = create_prediction_app(model_family=AUTOENCODER_FAMILY)
    assert app.server.test_client().get("/_dash-layout").status_code == 200
    assert app.server.test_client().get("/_dash-dependencies").status_code == 200


@pytest.fixture
def prediction_browser(source_factory, tmp_path, request):
    import os
    from threading import Thread
    from werkzeug.serving import make_server

    playwright = pytest.importorskip("playwright.sync_api")
    source, _ = source_factory()
    config_path = None
    if getattr(request, "param", False):
        config_path = tmp_path / "prediction.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "source": {"training_manifest_path": str(source)},
                    "exports": {"anndata": {"primary_key": "z_mu"}},
                }
            )
        )
    app = create_prediction_app(
        config=config_path, export_directory=tmp_path / "exports"
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
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.wait_for_function("window.brFieldInteractionsInstalled === true")
            yield page, source, app
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_browser_load_copy_inheritance_and_preserve_edits(prediction_browser):
    from playwright.sync_api import expect

    page, source, app = prediction_browser
    page.get_by_role(
        "button", name="Enter a value for Training Manifest Path", exact=True
    ).click()
    page.get_by_role("textbox", name="Training Manifest Path", exact=True).fill(
        str(source)
    )
    page.get_by_role("button", name="Load source", exact=True).click()
    expect(page.locator(".br-source-summary")).to_contain_text("Training run: source")
    expect(page.locator(".br-source-summary")).to_contain_text("Model family: composite")
    page.get_by_role("tab", name="Dataset", exact=False).click()
    copy = page.get_by_role("button", name="Copy inherited value to editor", exact=True)
    expect(copy).to_be_visible()
    copy.click()
    # The inherited dataset becomes an editable named schema, not a JSON blob.
    expect(page.get_by_role("textbox", name="Root", exact=True)).to_be_visible()
    page.get_by_role("textbox", name="Root", exact=True).fill("edited/dataset")
    page.wait_for_function(
        "document.getElementById('wizard-yaml-preview').textContent.includes('edited/dataset')"
    )
    page.get_by_role("tab", name="Transforms", exact=False).click()
    page.get_by_role(
        "button", name="Copy inherited value to editor", exact=True
    ).click()
    expect(
        page.get_by_role("button", name="Collapse Steps", exact=True)
    ).to_be_visible()
    assert "apply_to" not in page.locator("#wizard-yaml-preview").text_content()
    page.get_by_role("tab", name="Source", exact=False).click()
    page.get_by_role("textbox", name="Training Manifest Path", exact=True).fill(
        str(source.parent / "missing.yaml")
    )
    page.get_by_role("button", name="Load source", exact=True).click()
    expect(page.locator(".br-source-context")).to_contain_text("does not exist")
    page.get_by_role("tab", name="Dataset", exact=False).click()
    expect(page.get_by_role("textbox", name="Root", exact=True)).to_have_value(
        "edited/dataset"
    )
    page.get_by_role("tab", name="Source", exact=False).click()
    page.get_by_role("textbox", name="Training Manifest Path", exact=True).fill(
        str(source)
    )
    page.get_by_role("button", name="Load source", exact=True).click()
    expect(page.get_by_role("button", name="Reload source", exact=True)).to_be_visible()


@pytest.mark.parametrize("prediction_browser", [True], indirect=True)
def test_browser_review_reload_invalidates_receipt_and_can_recheck(prediction_browser):
    from playwright.sync_api import expect

    page, source, app = prediction_browser
    page.get_by_role("tab", name="Review & finish", exact=False).click()
    page.get_by_role("button", name="Check configuration", exact=True).click()
    expect(page.get_by_role("button", name="Save & finish", exact=True)).to_be_enabled(
        timeout=15000
    )
    expect(page.locator("#wizard-draft-preview")).to_be_hidden()
    page.get_by_role("tab", name="Source", exact=False).click()
    page.get_by_role("button", name="Reload source", exact=True).click()
    page.get_by_role("tab", name="Review & finish", exact=False).click()
    expect(
        page.get_by_role("button", name="Save & finish", exact=True)
    ).to_be_disabled()
    expect(page.locator("#wizard-draft-preview")).to_be_visible()
    assert app.wizard_session.fingerprint is None
    page.get_by_role("button", name="Check configuration", exact=True).click()
    expect(page.get_by_role("button", name="Save & finish", exact=True)).to_be_enabled(
        timeout=15000
    )
    page.get_by_role("button", name="Save & finish", exact=True).click()
    expect(page.locator("#wizard-save-message")).to_contain_text(
        "Authored configuration:"
    )
    assert app.wizard_session.result.config_path.is_file()
    assert "dataset" not in app.wizard_session.result.authored_config


def test_external_model_class_inherits_constructor_parameters(source_factory, tmp_path):
    from benchrep.architecture.models.autoencoder import Autoencoder

    path, manifest = source_factory(AUTOENCODER_FAMILY)
    raw = manifest["appendix"]["resolved_config"]
    for key in ("model", "encoder", "decoder", "losses", "optimizer"):
        raw.pop(key, None)
    raw["overrides"] = {"model": {"params": {"example_argument": 7}}}
    manifest["construction"]["model"]["source"] = "external_class"
    path.write_text(yaml.safe_dump(manifest))
    context = PredictionSourceContext(
        model_family=AUTOENCODER_FAMILY, overrides=PredictionOverrides()
    )
    draft = {"source": {"training_manifest_path": str(path)}}
    context.load(draft)
    assert "no model override" in context.error
    overrides = PredictionOverrides(model=Autoencoder)
    context, draft = context_for(path, AUTOENCODER_FAMILY, overrides)
    assert context.action("copy_inherited", ("overrides", "model"), draft) == {
        "params": {"example_argument": 7}
    }
    session = prediction_review_session(context=context)
    report = session.check(draft, str(tmp_path / "exports"), "prediction.yaml", {})
    assert report["ready"], report
    assert session.resolved.prediction_config.overrides.model.params == {
        "example_argument": 7
    }
    assert "overrides.model" in session.resolved.inherited_config_fields
    # Explicit empty parameters replace the inherited mapping.
    draft["overrides"] = {"model": {"params": {}}}
    report = session.check(draft, str(tmp_path / "exports"), "prediction.yaml", {})
    assert report["ready"], report
    assert session.resolved.prediction_config.overrides.model.params == {}


def test_family_mismatch_and_unacceptable_status_are_source_errors(source_factory):
    path, manifest = source_factory()
    context = PredictionSourceContext(
        model_family=VAE_FAMILY, overrides=PredictionOverrides()
    )
    draft = {"source": {"training_manifest_path": str(path)}}
    context.load(draft)
    assert "family" in context.error and not context.current(draft)
    manifest["status"] = "failed"
    path.write_text(yaml.safe_dump(manifest))
    context.load(draft)
    assert "status" in context.error


def test_prediction_launcher_selects_factory_and_retains_policy(monkeypatch):
    import webbrowser
    from threading import Event
    from flask import Flask
    from benchrep.gui import configure_workflow
    import benchrep.gui.predict as prediction

    done = Event()
    done.set()
    session = SimpleNamespace(done=done, result=object())
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(server=Flask(__name__), wizard_session=session)

    monkeypatch.setattr(prediction, "create_prediction_app", factory)
    monkeypatch.setattr(webbrowser, "open", lambda url: pytest.fail("Browser disabled"))
    override = PredictionOverrides()
    assert (
        configure_workflow(
            stage="prediction",
            overrides=override,
            compatibility_policy="warn",
            port=0,
            open_browser=False,
        )
        is session.result
    )
    assert (
        captured["overrides"] is override and captured["compatibility_policy"] == "warn"
    )
