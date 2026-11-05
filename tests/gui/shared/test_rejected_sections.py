"""Unknown/incompatible inputs remain inspectable and explicitly removable."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("dash_mantine_components")
from benchrep.assembly.schemas import TrainingConfig
from benchrep.gui.train import create_training_app
from benchrep.gui.predict import create_prediction_app
from benchrep.gui.evaluate import create_evaluation_app
from benchrep.gui.shared import forms
from benchrep.gui.shared.forms import include_rejected_sections
from benchrep.gui.shared.state import WizardDraft
from benchrep.gui.rules.train import training_incompatible_sections
from benchrep.interfaces.model_families import VAE_FAMILY, COMPOSITE_FAMILY


def stores(app):
    return {c.id: deepcopy(c.data) for c in app.layout.children
            if getattr(c, "id", "").startswith("wizard-") and hasattr(c, "data")}


def update_callback(app):
    return next(v["callback"].__wrapped__ for k, v in app.callback_map.items()
                if "wizard-editor-state.data" in k)


def button(app, monkeypatch, data, key, action, node=None):
    identity = {"type": "br-button", "section": key,
                "node": node or data["wizard-editor-state"][key]["id"], "action": action}
    monkeypatch.setattr(forms, "ctx", SimpleNamespace(
        triggered_id=identity, triggered_prop_ids={"button.n_clicks": identity}))
    changes = {}
    monkeypatch.setattr(forms, "set_props", lambda key, value: changes.update({key: value}))
    result = update_callback(app)(
        key, [], [1], [], [identity], data["wizard-editor-state"],
        data["wizard-draft"], data["wizard-section-errors"], data["wizard-inline-errors"],
    )
    return result, changes


@pytest.mark.parametrize("factory", [create_training_app, create_prediction_app, create_evaluation_app])
@pytest.mark.parametrize("key", ["optimzer", "__review__", ""])
def test_unknown_section_removal_preserves_draft_and_clears_review(tmp_path, monkeypatch, factory, key):
    raw = {key: {"nested": [1, None, {"foo": 2}]}}
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(raw))
    app = factory(config=path)
    data = stores(app)
    rejected = next(k for k in data["wizard-editor-state"] if k.startswith("__rejected_"))
    assert data["wizard-draft"][key] == raw[key]
    assert data["wizard-section-errors"][rejected]
    app.wizard_session.fingerprint = "previous-check"
    result, changes = button(app, monkeypatch, data, rejected, "remove_section")
    state, _, draft, errors, *_ = result
    expected = deepcopy(data["wizard-draft"])
    expected.pop(key)
    assert draft == expected
    assert state[rejected]["removed"]
    assert not errors[rejected]
    assert changes["wizard-sections"]["value"] != rejected
    assert changes["wizard-review-receipt"] == {"data": None}
    assert app.wizard_session.fingerprint is None
    nav = next(v["callback"].__wrapped__ for k, v in app.callback_map.items()
               if "wizard-section-tab-" in k)
    assert nav(state) == [{"display": "none"}]


@pytest.mark.parametrize("family,key", [
    (VAE_FAMILY, "composite_model_components"),
    (VAE_FAMILY, "composite_model_declarations"),
    (VAE_FAMILY, "composite_model_assembly"),
    (COMPOSITE_FAMILY, "encoder"), (COMPOSITE_FAMILY, "decoder"),
])
def test_family_incompatible_sections_are_explicitly_removable(tmp_path, monkeypatch, family, key):
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump({key: {"name": "whatever"}}))
    app = create_training_app(config=path, model_family=family)
    data = stores(app)
    rejected = next(k for k in data["wizard-editor-state"] if k.startswith("__rejected_"))
    assert "incompatible" in data["wizard-section-errors"][rejected][0]["message"]
    result, _ = button(app, monkeypatch, data, rejected, "remove_section")
    assert key not in result[2]
    assert not result[3][rejected]


def test_absent_or_null_incompatible_sections_are_not_errors():
    initial = WizardDraft(initial_data={"encoder": None, "not_a_section": None}, supplied_sections=frozenset())
    sections = include_rejected_sections([], initial, TrainingConfig, training_incompatible_sections(COMPOSITE_FAMILY))
    assert [s.source_key for s in sections] == ["not_a_section"]


def test_config_component_remains_locked(tmp_path, monkeypatch):
    from benchrep.assembly.schemas import TrainingEncoderConfig
    from benchrep.utils.workflow_overrides import TrainingOverrides
    from dash.exceptions import PreventUpdate
    app = create_training_app(model_family=COMPOSITE_FAMILY, overrides=TrainingOverrides(
        config_components={"encoder": TrainingEncoderConfig(name="mlp")},
    ))
    data = stores(app)
    rejected = next(k for k in data["wizard-editor-state"] if k.startswith("__rejected_"))
    with pytest.raises(PreventUpdate):
        button(app, monkeypatch, data, rejected, "remove_section")


@pytest.mark.parametrize("family,key,value", [
    (VAE_FAMILY, "composite_model_assembly_input_overrides", {}),
    (COMPOSITE_FAMILY, "canonical_vae_reconstruction_latent_source", "mean"),
])
def test_prediction_incompatible_fields_are_removable(tmp_path, monkeypatch, family, key, value):
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump({"inference": {key: value, "seed": 137}}))
    app = create_prediction_app(config=path, model_family=family)
    data = stores(app)
    assert any(e["path"] == [key] for e in data["wizard-section-errors"]["inference"])
    node = data["wizard-editor-state"]["inference"]["fields"][key]
    result, _ = button(app, monkeypatch, data, "inference", "remove_incompatible", node["id"])
    assert result[2]["inference"] == {"seed": 137}
    assert not result[3]["inference"]


def test_missing_sections_stay_omitted_except_fixed_model():
    data = stores(create_training_app(model_family=VAE_FAMILY))
    assert data["wizard-editor-state"]["model"]["mode"] == "value"
    assert all(n["mode"] == "omit" for k, n in data["wizard-editor-state"].items() if k != "model")


def test_wrong_stage_is_exposed(tmp_path):
    path = tmp_path / "draft.yaml"
    path.write_text("stage: prediction\n")
    data = stores(create_training_app(config=path))
    rejected = next(k for k in data["wizard-editor-state"] if k.startswith("__rejected_"))
    assert data["wizard-draft"]["stage"] == "prediction"
    assert "stage 'training'" in data["wizard-section-errors"][rejected][0]["message"]


def test_navigation_callback_serializes_single_style_object(tmp_path, monkeypatch):
    """Exercise Dash's real response shape, including the single-rejected-tab case."""
    path = tmp_path / "draft.yaml"
    path.write_text("typo: 137\n")
    app = create_training_app(config=path)
    data = stores(app)
    rejected = next(k for k in data["wizard-editor-state"] if k.startswith("__rejected_"))
    result, _ = button(app, monkeypatch, data, rejected, "remove_section")
    callback_key = next(k for k in app.callback_map if "wizard-section-tab-" in k)
    client = app.server.test_client()
    response = client.post("/_dash-update-component", json={
        "output": callback_key,
        "outputs": [{"id": f"wizard-section-tab-{rejected}", "property": "style"}],
        "inputs": [{"id": "wizard-editor-state", "property": "data", "value": result[0]}],
        "state": [], "changedPropIds": ["wizard-editor-state.data"],
    })
    assert response.status_code == 200, response.data
    assert response.json["response"][f"wizard-section-tab-{rejected}"]["style"] == {"display": "none"}


def test_loss_wiring_removal_preserves_loss_parameters(tmp_path, monkeypatch):
    raw = {"losses": {"reconstruction": [{"name": "mse", "weight": 2.0,
        "composite_wiring": {"input": "produces.x", "target": "expects.x"}}]}}
    path = tmp_path / "draft.yaml"
    path.write_text(yaml.safe_dump(raw))
    app = create_training_app(config=path, model_family=VAE_FAMILY)
    data = stores(app)
    term = data["wizard-editor-state"]["losses"]["entries"][0]["node"]["entries"][0]["node"]
    target = term["fields"]["composite_wiring"]
    assert any(e["path"] == ["reconstruction", 0, "composite_wiring"]
               for e in data["wizard-section-errors"]["losses"])
    result, _ = button(app, monkeypatch, data, "losses", "remove_incompatible", target["id"])
    assert result[2]["losses"] == {"reconstruction": [{"name": "mse", "weight": 2.0}]}
    assert not result[3]["losses"]


def test_multiple_removals_do_not_restore_earlier_section(tmp_path, monkeypatch):
    path = tmp_path / "draft.yaml"
    path.write_text("first_typo: 1\nsecond_typo: 2\n")
    app = create_training_app(config=path)
    data = stores(app)
    rejected = [k for k in data["wizard-editor-state"] if k.startswith("__rejected_")]
    for key in rejected:
        result, _ = button(app, monkeypatch, data, key, "remove_section")
        for name, index in [("wizard-editor-state", 0), ("wizard-draft", 2),
                            ("wizard-section-errors", 3), ("wizard-inline-errors", 4)]:
            from dash import no_update
            if result[index] is not no_update:
                data[name] = result[index]
    assert "first_typo" not in data["wizard-draft"]
    assert "second_typo" not in data["wizard-draft"]
    assert all(data["wizard-editor-state"][key]["removed"] for key in rejected)


def test_inapplicable_vae_parameter_uses_existing_removal_panel():
    from benchrep.gui.rules.train import training_model_section
    from benchrep.gui.shared.state import editor_spec, new_node, edit_node
    from benchrep.gui.shared.forms import evaluate_section
    initial = WizardDraft(initial_data={"model": {"name": "composite", "params": {"latent_dim": 32}}},
                          supplied_sections=frozenset())
    section = training_model_section(model_family=COMPOSITE_FAMILY, initial=initial)
    spec = editor_spec(section.annotation, section.field_info)
    node = new_node(spec, section.to_editor(initial.initial_data["model"]))
    assert node["extra"] == {"model['params']['latent_dim']": 32}
    assert evaluate_section(section, spec, node, initial=initial)[1]
    edit_node(spec, node, node["id"], "remove_unexpected", None)
    assert evaluate_section(section, spec, node, initial=initial) == (
        {"name": "composite", "params": {}}, {},
    )


def test_remove_section_through_dash_http_callback(tmp_path):
    import json
    path = tmp_path / "draft.yaml"
    path.write_text("typo: {keep_until_removed: true}\n")
    app = create_training_app(config=path)
    data = stores(app)
    rejected = next(k for k in data["wizard-editor-state"] if k.startswith("__rejected_"))
    key = next(k for k in app.callback_map if "wizard-editor-state.data" in k)
    metadata = app.callback_map[key]
    identity = {"type": "br-button", "section": rejected,
                "node": data["wizard-editor-state"][rejected]["id"], "action": "remove_section"}
    state_values = [[], [identity], data["wizard-editor-state"], data["wizard-draft"],
                    data["wizard-section-errors"], data["wizard-inline-errors"]]
    response = app.server.test_client().post("/_dash-update-component", json={
        "output": key,
        "outputs": [{"id": o.component_id, "property": o.component_property} for o in metadata["output"]],
        "inputs": [{**item, "value": value} for item, value in zip(metadata["inputs"], [rejected, [], [1]], strict=True)],
        "state": [{**item, "value": value} for item, value in zip(metadata["state"], state_values, strict=True)],
        "changedPropIds": [json.dumps(identity, sort_keys=True, separators=(",", ":")) + ".n_clicks"],
    })
    assert response.status_code == 200, response.data
    result = response.json
    assert "typo" not in result["response"]["wizard-draft"]["data"]
    assert result["response"]["wizard-editor-state"]["data"][rejected]["removed"]
    assert result["sideUpdate"]["wizard-review-receipt"]["data"] is None
    assert result["sideUpdate"]["wizard-sections"]["value"] == "run"


def test_external_datamodule_visibility_is_not_family_incompatibility(tmp_path):
    import lightning as L
    from benchrep.utils.workflow_overrides import TrainingOverrides
    path = tmp_path / "draft.yaml"
    path.write_text("dataset: {}\ndatamodule: {}\ntransform_pipelines: []\n")
    app = create_training_app(config=path, overrides=TrainingOverrides(datamodule=L.LightningDataModule()))
    data = stores(app)
    assert not any(k.startswith("__rejected_") for k in data["wizard-editor-state"])
    assert "datamodule" not in data["wizard-editor-state"]


def test_rejected_section_errors_block_review(tmp_path):
    path = tmp_path / "draft.yaml"
    path.write_text("typo: 137\n")
    app = create_training_app(config=path)
    data = stores(app)
    report = app.wizard_session.check(data["wizard-draft"], str(tmp_path / "exports"),
                                      "training.yaml", data["wizard-section-errors"])
    assert not report["ready"]
    assert "section errors" in report["rows"][0]["message"]


@pytest.mark.parametrize("factory", [create_training_app, create_prediction_app, create_evaluation_app])
def test_browser_rejected_tab_can_be_removed(tmp_path, factory):
    import os
    from threading import Thread
    from werkzeug.serving import make_server
    playwright = pytest.importorskip("playwright.sync_api")
    path = tmp_path / "draft.yaml"
    path.write_text("typo: {preserve: 137}\n")
    app = factory(config=path)
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with playwright.sync_playwright() as manager:
            browser = manager.chromium.launch(
                executable_path=os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            tab = page.get_by_role("tab", name="typo", exact=False)
            playwright.expect(tab.locator(".br-section-status-invalid")).to_be_visible()
            tab.click()
            playwright.expect(page.locator("#wizard-active-content")).to_contain_text("Unknown section")
            playwright.expect(page.locator("#wizard-active-content")).to_contain_text("preserve: 137")
            page.get_by_role("button", name="Remove section", exact=True).click()
            playwright.expect(tab).to_be_hidden()
            playwright.expect(page.locator("#wizard-yaml-preview")).not_to_contain_text("typo:")
            playwright.expect(page.locator("#wizard-active-heading")).not_to_contain_text("typo")
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
