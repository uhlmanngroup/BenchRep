"""Training drafts keep fixed family controls and explicitly removable bad input."""
from copy import deepcopy

import pytest
import yaml

pytest.importorskip("dash_mantine_components")

from benchrep.gui.train import create_training_app
from benchrep.gui.rules.train import prepare_training_draft, training_sections
from benchrep.gui.shared.fields import render_editor
from benchrep.gui.shared.forms import evaluate_section
from benchrep.gui.shared.state import (
    MISSING, WizardDraft, edit_node, editor_spec, new_node,
)
from benchrep.interfaces.model_families import (
    AUTOENCODER_FAMILY, COMPOSITE_FAMILY, VAE_FAMILY,
)
from benchrep.utils.workflow_overrides import TrainingOverrides


def model_editor(raw, family=VAE_FAMILY):
    initial = WizardDraft(initial_data={"model": deepcopy(raw)}, supplied_sections=frozenset())
    section = next(s for s in training_sections(
        model_family=family, overrides=TrainingOverrides(), initial=initial,
    ) if s.key == "model")
    spec = editor_spec(section.annotation, section.field_info)
    node = new_node(spec, section.to_editor(raw))
    return initial, section, spec, node


def collect(initial, section, spec, node):
    return evaluate_section(section, spec, node, initial=initial)


def components(component):
    yield component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if hasattr(child, "to_plotly_json"):
            yield from components(child)


@pytest.mark.parametrize("raw", [None, {}, {"params": None}, {"params": {}}, {"name": "vae"}])
def test_empty_model_has_fixed_name_and_editable_dimension(raw):
    initial, section, spec, node = model_editor(raw)
    assert node["mode"] == "value"
    assert node["fields"]["name"]["value"] == "vae"
    assert node["fields"]["name"]["mode"] == "value"
    rendered = render_editor(spec, node, section="model", path=("model",),
                             locked_paths=section.locked_paths)
    name_inputs = [c for c in components(rendered)
                   if isinstance(getattr(c, "id", None), dict)
                   and c.id.get("node") == node["fields"]["name"]["id"]
                   and c.id.get("type") == "br-input"]
    assert name_inputs and all(c.disabled for c in name_inputs)
    assert collect(initial, section, spec, node)[1]
    latent = node["fields"]["latent_dim"]
    edit_node(spec, node, latent["id"], "mode", "value")
    edit_node(spec, node, latent["id"], "value", 16)
    assert collect(initial, section, spec, node) == (
        {"name": "vae", "params": {"latent_dim": 16}}, {},
    )


@pytest.mark.parametrize("family", [VAE_FAMILY, AUTOENCODER_FAMILY, COMPOSITE_FAMILY])
def test_reset_and_mode_switch_preserve_fixed_name(family):
    initial, section, spec, node = model_editor(None, family)
    for mode in ("null", "omit", "value"):
        edit_node(spec, node, node["id"], "mode", mode)
        assert node["fields"]["name"]["value"] == family.name
    edit_node(spec, node, node["id"], "reset", None)
    assert node["mode"] == "value"
    assert node["fields"]["name"]["value"] == family.name
    assert node["fields"]["name"]["mode"] == "value"


@pytest.mark.parametrize("raw", [
    {"name": "vae", "params": {"latent_dim": 8}},
    {"name": "vae", "params": {"latent_dim": -1}},
    {"name": "vae", "params": {"latent_dim": 8, "blah": 10}},
    {"name": "vae", "params": {"latent_dim": 8}, "blah": 137},
    {"name": "autoencoder", "params": {"latent_dim": 8}},
    {"name": "vae", "params": ["invalid"]},
    ["invalid"],
])
def test_values_and_conflicts_are_preserved_until_removed(raw):
    before = deepcopy(raw)
    initial, section, spec, node = model_editor(raw)
    assert collect(initial, section, spec, node)[0] == before
    assert raw == before
    assert node["fields"]["name"]["value"] == "vae"
    if node["extra"]:
        assert collect(initial, section, spec, node)[1]
        edit_node(spec, node, node["id"], "remove_unexpected", None)
        latent = node["fields"]["latent_dim"]
        edit_node(spec, node, latent["id"], "mode", "value")
        edit_node(spec, node, latent["id"], "value", 8)
        assert collect(initial, section, spec, node) == (
            {"name": "vae", "params": {"latent_dim": 8}}, {},
        )


@pytest.mark.parametrize("raw", [{}, {"model": None}, {"model": {}}, {"model": {"params": None}}])
def test_empty_yaml_model_opens_with_same_controls(tmp_path, raw):
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw))
    app = create_training_app(config=path, model_family=VAE_FAMILY)
    state = next(c.data for c in app.layout.children if getattr(c, "id", None) == "wizard-editor-state")
    assert state["model"]["mode"] == "value"
    assert state["model"]["fields"]["name"]["value"] == "vae"
    assert "latent_dim" in state["model"]["fields"]


def test_fresh_draft_has_fixed_name():
    draft = prepare_training_draft(model_family=VAE_FAMILY, config=None, overrides=TrainingOverrides())
    assert draft.initial_data["model"] == {"name": "vae"}


@pytest.mark.parametrize("invalid_section", [
    {"composite_model_declarations": {"blah": 137}},
    {"datamodule": {"num_workers": 0, "persistent_workers": True}},
])
def test_invalid_training_yaml_opens_without_losing_other_sections(tmp_path, invalid_section):
    raw = {"stage": "training", "model": {"name": "composite"}, **invalid_section}
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw))
    app = create_training_app(config=path, model_family=COMPOSITE_FAMILY)
    draft = next(c.data for c in app.layout.children if getattr(c, "id", None) == "wizard-draft")
    assert draft == raw


def test_invalid_training_draft_cannot_pass_review_or_save(tmp_path):
    raw = {"stage": "training", "model": {"name": "composite"},
           "composite_model_declarations": {"blah": 137}}
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw))
    app = create_training_app(config=path, model_family=COMPOSITE_FAMILY)
    session = app.wizard_session
    directory = str(tmp_path / "exports")
    report = session.check(raw, directory, "training.yaml", {})
    assert not report["ready"]
    assert report["rows"][0]["status"] == "failed"
    with pytest.raises(ValueError, match="Check configuration again"):
        session.save(raw, directory, "training.yaml", {})
    assert not list((tmp_path / "exports").glob("*.yaml"))