"""Collection presentation and persistent collapse behavior."""

from types import SimpleNamespace

import pytest

pytest.importorskip("dash_mantine_components")
from benchrep.gui.rules.train import loss_presentation, transform_presentation
from benchrep.gui.shared import forms
from benchrep.gui.shared.fields import FieldPresentation, render_editor
from benchrep.gui.shared.state import (
    WizardDraft,
    collect_node,
    edit_entry,
    editor_spec,
    new_node,
)
from dash import Dash
from pydantic import BaseModel


def walk(component):
    yield component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, list) else [children]:
        if hasattr(child, "to_plotly_json"):
            yield from walk(child)


class Entry(BaseModel):
    name: str


def test_collapsed_item_retains_value_and_identity_after_move():
    spec = editor_spec(list[Entry])
    node = new_node(spec, [{"name": "a"}, {"name": "b"}])
    item = node["entries"][0]["node"]
    item["collapsed"] = True
    edit_entry(spec, node, item["id"], "down", None)
    assert node["entries"][1]["node"] is item
    assert item["collapsed"]
    assert collect_node(spec, node)[0] == [{"name": "b"}, {"name": "a"}]
    rendered = render_editor(
        spec,
        node,
        section="entries",
        path=("entries",),
        presentation=lambda p, v: FieldPresentation(item_label="Step"),
    )
    components = list(walk(rendered))
    assert any(getattr(c, "children", None) == "Step 2" for c in components)
    assert any(getattr(c, "hidden", False) for c in components)
    assert any(
        isinstance(getattr(c, "id", None), dict)
        and c.id.get("type") == "br-group-error"
        for c in components
    )


def test_scalar_items_have_no_collapse_controls():
    spec = editor_spec(list[str])
    rendered = render_editor(
        spec, new_node(spec, ["a"]), section="items", path=("items",)
    )
    assert not any(
        isinstance(getattr(c, "id", None), dict) and c.id.get("action") == "collapse"
        for c in walk(rendered)
    )


def test_semantic_labels():
    assert (
        transform_presentation(
            ("transform_pipelines",), None, {}, composite=False
        ).item_label
        == "Pipeline"
    )
    assert (
        transform_presentation(
            ("transform_pipelines", 0, "steps"), None, {}, composite=False
        ).item_label
        == "Step"
    )
    assert (
        loss_presentation(
            ("losses", "reconstruction"), None, {}, composite=False
        ).item_label
        == "Loss term"
    )


def test_locked_section_can_collapse_without_changing_config(monkeypatch):
    raw = [{"name": "a"}]
    initial = WizardDraft(
        initial_data={"entries": raw}, supplied_sections=frozenset({"entries"})
    )
    section = forms.SectionSpec(key="entries", title="Entries", annotation=list[Entry])
    app = Dash(__name__, suppress_callback_exceptions=True)
    forms.build_workflow_app(
        app=app,
        stage="training",
        sections=[section],
        initial=initial,
        title="Test",
        subtitle="",
    )
    node = new_node(editor_spec(list[Entry]), raw)
    item = node["entries"][0]["node"]
    identifier = {
        "type": "br-button",
        "section": "entries",
        "node": item["id"],
        "action": "collapse",
    }
    monkeypatch.setattr(
        forms,
        "ctx",
        SimpleNamespace(
            triggered_prop_ids={"button.n_clicks": identifier}, triggered_id=identifier
        ),
    )
    update = next(
        v["callback"].__wrapped__
        for k, v in app.callback_map.items()
        if "wizard-editor-state.data" in k
    )
    result = update(
        "entries",
        [],
        [1],
        [],
        [identifier],
        {"entries": node},
        initial.initial_data,
        {"entries": []},
        {},
    )
    assert result[0]["entries"]["entries"][0]["node"]["collapsed"]
    assert collect_node(editor_spec(list[Entry]), result[0]["entries"])[0] == raw


def test_explicit_registry_follows_section():
    app = Dash(__name__, suppress_callback_exceptions=True)
    sections = [
        forms.SectionSpec(key=k, title=k, annotation=dict, default_registry=r)
        for k, r in [
            ("components", "encoder"),
            ("transforms", "transform"),
            ("losses", "reconstruction_loss"),
        ]
    ]
    forms.build_workflow_app(
        app=app,
        stage="training",
        sections=sections,
        initial=WizardDraft(initial_data={}, supplied_sections=frozenset()),
        title="Test",
        subtitle="",
    )
    follow = app.callback_map["wizard-docs-registry.value"]["callback"].__wrapped__
    assert follow(None, True, "components") == "encoder"
    assert follow(None, True, "transforms") == "transform"
    assert follow(None, True, "losses") == "reconstruction_loss"
