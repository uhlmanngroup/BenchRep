"""Frontend assistance must preserve raw values and defer wiring compatibility."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("dash_mantine_components")

from benchrep.gui.rules import train as rules
from benchrep.gui.shared.forms import evaluate_section
from benchrep.gui.shared.state import WizardDraft, editor_spec, new_node
from benchrep.interfaces.model_families import COMPOSITE_FAMILY, VAE_FAMILY


def loss_section(family=COMPOSITE_FAMILY, *, model=None):
    initial = WizardDraft(
        initial_data={"model": {"name": family.name}}, supplied_sections=frozenset()
    )
    sections = rules.training_sections(
        model_family=family,
        overrides=SimpleNamespace(model=model, datamodule=None),
        initial=initial,
    )
    return next((s for s in sections if s.key == "losses"), None), initial


@pytest.mark.parametrize("family", [COMPOSITE_FAMILY, VAE_FAMILY])
def test_losses_available_for_config_built_models(family):
    assert loss_section(family)[0] is not None
    assert loss_section(family, model=object())[0] is None


@pytest.mark.parametrize(
    "data,valid",
    [
        ({"reconstruction": [{"name": "mse", "weight": 1}]}, True),
        ({"reconstruction": []}, False),
        ({"reconstruction": [{"name": "mse", "weight": -1}]}, False),
        ({"unknown_role": [{"name": "mse"}]}, False),
        (
            {
                "reconstruction": [
                    {
                        "name": "mse",
                        "weight_schedule": {"name": "linear", "params": {"end": 10}},
                    }
                ]
            },
            True,
        ),
        (
            {
                "reconstruction": [
                    {
                        "name": "mse",
                        "weight_schedule": {
                            "name": "linear",
                            "params": {"start": 10, "end": 5},
                        },
                    }
                ]
            },
            False,
        ),
        (
            {
                "reconstruction": [
                    {"name": "mse", "composite_wiring": {"target": "expects.future"}}
                ]
            },
            True,
        ),
    ],
)
def test_loss_schema_validation_does_not_rewrite_draft(data, valid):
    section, initial = loss_section()
    spec = editor_spec(section.annotation, section.field_info)
    raw, errors = evaluate_section(section, spec, new_node(spec, data), initial=initial)
    assert raw == data
    assert bool(errors) is not valid


def test_kind_membership_resolves_aliases_without_clearing_values(monkeypatch):
    class Registry:
        def resolve_key(self, name):
            if name in ("known", "alias"):
                return "known"
            raise KeyError(name)

    monkeypatch.setattr(rules, "_COMPOSITE_COMPONENT_REGISTRIES", {"head": Registry()})
    data = {"c": {"kind": "head", "name": "encoder_name", "params": {"size": 10}}}
    before = deepcopy(data)
    assert ("c", "name") in rules.validate_component_names(data)
    assert data == before
    data["c"]["name"] = "alias"
    assert rules.validate_component_names(data) == {}


def test_components_require_kind_before_name_and_order_kind_first():
    path = ("composite_model_components", "c")
    assert rules.composite_presentation(path, None, {}).order == (
        "kind",
        "name",
        "params",
    )
    assert rules.composite_presentation((*path, "name"), None, {}).disabled_reason


def test_references_follow_current_draft_without_mutation():
    draft = {
        "composite_model_components": {"enc": {}},
        "composite_model_declarations": {
            "expects": {"x": "sample_image"},
            "produces": {"z": "embedding_vector"},
        },
    }
    before = deepcopy(draft)
    guide = rules.composite_presentation(
        ("composite_model_assembly", "step", "component"), "manual", draft
    )
    assert guide.suggestions == ("enc",)
    assert rules.composite_presentation(
        ("composite_model_assembly", "step", "inputs", "x"), None, draft
    ).suggestions == ("expects.x", "produces.z")
    assert rules.composite_presentation(
        ("composite_model_assembly", "step", "outputs"), None, draft
    ).suggestions == ("produces.z",)
    assert draft == before
    draft["composite_model_components"] = {"renamed": {}}
    assert rules.composite_presentation(
        ("composite_model_assembly", "step", "component"), "enc", draft
    ).suggestions == ("renamed",)


@pytest.mark.parametrize(
    "composite,role,hidden",
    [
        (False, "reconstruction", True),
        (True, "custom_objective", True),
        (True, "reconstruction", False),
    ],
)
def test_wiring_visibility(composite, role, hidden):
    guide = rules.loss_presentation(
        ("losses", role, 0, "composite_wiring"), None, {}, composite=composite
    )
    assert guide.hidden is hidden
