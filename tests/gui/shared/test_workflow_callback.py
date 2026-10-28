"""Regression coverage for section validation and batched editor updates."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("dash_mantine_components")

from benchrep.assembly.schemas.training_config_schema import TrainingConfig
from benchrep.gui.shared import forms
from benchrep.gui.shared.state import (
    WizardDraft,
    collect_node,
    editor_spec,
    new_node,
    union_variant_details,
)
from dash import Dash


def loss_fixture(raw):
    field = TrainingConfig.model_fields["losses"]
    section = forms.SectionSpec(
        key="losses",
        title="Losses",
        annotation=field.rebuild_annotation(),
        field_info=field,
    )
    initial = WizardDraft(initial_data={"losses": raw}, supplied_sections=frozenset())
    spec = editor_spec(section.annotation, field)
    return section, initial, spec, new_node(spec, raw)


def test_schedule_error_points_to_parameter_without_union_tag():
    section, initial, spec, node = loss_fixture(
        {
            "reconstruction": [
                {
                    "name": "mse",
                    "weight_schedule": {"name": "linear", "params": {"end": -1}},
                }
            ]
        }
    )
    _, errors = forms.evaluate_section(section, spec, node, initial=initial)
    assert ("reconstruction", 0, "weight_schedule", "params", "end") in errors


def test_switching_scheduler_captures_latest_mounted_parameter(monkeypatch):
    section, initial, spec, node = loss_fixture(
        {
            "reconstruction": [
                {
                    "name": "mse",
                    "weight_schedule": {"name": "custom_a", "params": {"n": 1}},
                }
            ]
        }
    )
    app = Dash(__name__, suppress_callback_exceptions=True)
    forms.build_workflow_app(
        app=app,
        stage="training",
        sections=[section],
        initial=initial,
        title="Test",
        subtitle="",
    )
    update = next(
        value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "wizard-editor-state.data" in key
    )
    schedule = node["entries"][0]["node"]["entries"][0]["node"]["fields"][
        "weight_schedule"
    ]
    schedule_spec = spec.item.item.fields["weight_schedule"]
    branch = next(
        i
        for i in range(len(schedule_spec.variants))
        if union_variant_details(schedule_spec, i)[0] == "custom"
    )
    parameter = schedule["branches"][str(branch)]["fields"]["params"]
    selector_id = {
        "type": "br-input",
        "section": "losses",
        "node": schedule["id"],
        "action": "union_name",
    }
    params_id = {**selector_id, "node": parameter["id"], "action": "value"}
    monkeypatch.setattr(
        forms,
        "ctx",
        SimpleNamespace(
            triggered_prop_ids={"selector.value": selector_id}, triggered_id=selector_id
        ),
    )
    result = update(
        "losses",
        [json.dumps(["custom_b", branch]), '{"n": 2}'],
        [],
        [selector_id, params_id],
        [],
        {"losses": node},
        initial.initial_data,
        {"losses": []},
        {},
    )
    state = result[0]["losses"]
    from benchrep.gui.shared.state import edit_node

    edit_node(
        spec, state, schedule["id"], "union_name", json.dumps(["custom_a", branch])
    )
    assert collect_node(spec, state)[0]["reconstruction"][0]["weight_schedule"][
        "params"
    ] == {"n": 2}
