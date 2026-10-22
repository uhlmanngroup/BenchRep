"""Composite section selection and registry guidance; requires GUI dependencies."""

from types import SimpleNamespace

import pytest

pytest.importorskip("dash_mantine_components")

from benchrep.gui.rules import train as rules
from benchrep.gui.shared.forms import evaluate_section
from benchrep.gui.shared.state import WizardDraft, editor_spec, new_node
from benchrep.interfaces.model_families import COMPOSITE_FAMILY, VAE_FAMILY


def sections(family, *, external_datamodule=False):
    initial = WizardDraft(
        initial_data={"model": {"name": family.name}},
        supplied_sections=frozenset(),
    )
    return rules.training_sections(
        model_family=family,
        overrides=SimpleNamespace(
            model=None, datamodule=object() if external_datamodule else None
        ),
        initial=initial,
    ), initial


def test_composite_section_order():
    result, _ = sections(COMPOSITE_FAMILY)
    assert [s.key for s in result] == [
        "run",
        "reproducibility",
        "model",
        "composite_model_declarations",
        "composite_model_components",
        "composite_model_assembly",
        "losses",
        "datamodule",
    ]


def test_canonical_sections_unchanged():
    result, _ = sections(VAE_FAMILY)
    assert [s.key for s in result] == [
        "run",
        "reproducibility",
        "model",
        "encoder",
        "decoder",
        "losses",
        "datamodule",
    ]


def test_external_datamodule_does_not_hide_composite_architecture():
    result, _ = sections(COMPOSITE_FAMILY, external_datamodule=True)
    assert "composite_model_assembly" in [s.key for s in result]
    assert "datamodule" not in [s.key for s in result]


def test_registry_choices_follow_kind_without_erasing_existing_name(monkeypatch):
    registries = {
        kind: SimpleNamespace(canonical_keys=lambda names=names: names)
        for kind, names in {
            "encoder": ("enc_a",),
            "decoder": ("dec_a",),
            "head": ("head_a",),
        }.items()
    }
    monkeypatch.setattr(rules, "_COMPOSITE_COMPONENT_REGISTRIES", registries)
    path = ("composite_model_components", "my_component", "name")
    assert rules.composite_component_name_options(path, None, {}) == []
    assert rules.composite_component_name_options(
        path, "loaded_alias", {"my_component": {"kind": "head"}}
    ) == ["head_a", "loaded_alias"]
    assert (
        rules.composite_component_name_options((*path[:-1], "params"), None, {}) is None
    )


@pytest.mark.parametrize("outputs", ["produces.z", {"features": "produces.z"}])
def test_assembly_supports_tensor_and_mapping_outputs(outputs):
    result, initial = sections(COMPOSITE_FAMILY)
    section = next(s for s in result if s.key == "composite_model_assembly")
    spec = editor_spec(section.annotation, section.field_info)
    data = {
        "encode": {"component": "enc", "inputs": {"x": "expects.x"}, "outputs": outputs}
    }
    raw, errors = evaluate_section(section, spec, new_node(spec, data), initial=initial)
    assert raw == data
    assert errors == {}


def test_supplied_component_section_remains_authoritative():
    result, _ = sections(COMPOSITE_FAMILY)
    section = next(s for s in result if s.key == "composite_model_components")
    supplied = {"enc": {"kind": "encoder", "name": "custom", "params": {"width": 32}}}
    initial = WizardDraft(
        initial_data={section.key: supplied}, supplied_sections=frozenset({section.key})
    )
    spec = editor_spec(section.annotation, section.field_info)
    raw, errors = evaluate_section(section, spec, new_node(spec, None), initial=initial)
    assert raw == supplied
    assert errors == {}
