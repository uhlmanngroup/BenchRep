"""Dataset selection and transform editing retain schema validation and raw input."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("dash_mantine_components")

from benchrep.gui.rules import train as rules
from benchrep.gui.shared.forms import evaluate_section
from benchrep.gui.shared.state import (
    WizardDraft,
    collect_node,
    edit_entry,
    edit_node,
    editor_spec,
    new_node,
)
from benchrep.interfaces.model_families import COMPOSITE_FAMILY, VAE_FAMILY


def sections(family=VAE_FAMILY, *, external_datamodule=False):
    initial = WizardDraft(
        initial_data={"model": {"name": family.name}}, supplied_sections=frozenset()
    )
    values = rules.training_sections(
        model_family=family,
        overrides=SimpleNamespace(
            model=None, datamodule=object() if external_datamodule else None
        ),
        initial=initial,
    )
    return {section.key: section for section in values}, initial


@pytest.mark.parametrize("family", [VAE_FAMILY, COMPOSITE_FAMILY])
def test_external_datamodule_hides_all_data_sections(family):
    values, _ = sections(family, external_datamodule=True)
    assert not {"dataset", "transform_pipelines", "datamodule"} & values.keys()


def test_dataset_selector_uses_registry_names_and_preserves_loaded_spelling(
    monkeypatch,
):
    monkeypatch.setattr(
        rules,
        "DATASETS",
        SimpleNamespace(canonical_keys=lambda: ("mnist", "cifar10", "custom_data")),
    )
    guide = rules.dataset_presentation(
        ("dataset",), None, {"dataset": {"name": "cifar_10"}}
    )
    variants = dict(guide.named_variants)
    assert set(variants) == {"mnist", "cifar10", "cifar_10", "custom_data"}
    assert variants["cifar10"] == variants["cifar_10"]
    assert variants["mnist"] != variants["custom_data"]


def test_switching_dataset_restores_each_parameter_set(monkeypatch):
    monkeypatch.setattr(
        rules,
        "DATASETS",
        SimpleNamespace(canonical_keys=lambda: ("mnist", "custom_data")),
    )
    values, _ = sections()
    section = values["dataset"]
    spec = editor_spec(section.annotation, section.field_info)
    raw = {"name": "mnist", "params": {"root": "data", "split": "train"}}
    node = new_node(spec, raw)
    branches = dict(rules.dataset_presentation(("dataset",), None, {}).named_variants)
    edit_node(
        spec,
        node,
        node["id"],
        "union_name",
        json.dumps(["custom_data", branches["custom_data"]]),
    )
    custom = node["branches"][str(node["branch"])]["fields"]["params"]
    custom.update(mode="value", value='{"count": 5}')
    edit_node(
        spec, node, node["id"], "union_name", json.dumps(["mnist", branches["mnist"]])
    )
    assert collect_node(spec, node)[0] == raw
    edit_node(
        spec,
        node,
        node["id"],
        "union_name",
        json.dumps(["custom_data", branches["custom_data"]]),
    )
    assert collect_node(spec, node)[0] == {
        "name": "custom_data",
        "params": {"count": 5},
    }


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"name": "mnist", "params": {"root": "data", "split": "train"}},
        {"name": "custom_data", "params": {}},
    ],
)
def test_dataset_roundtrip(raw):
    values, _ = sections()
    section = values["dataset"]
    spec = editor_spec(section.annotation, section.field_info)
    assert collect_node(spec, new_node(spec, raw))[0] == raw


def test_transform_routes_are_suggestions_without_rewriting_input():
    draft = {
        "composite_model_declarations": {
            "expects": {
                "image": "sample_image",
                "other": "positive_image",
                "label": "categorical_target",
                "bad": [],
            }
        }
    }
    guide = rules.transform_presentation(
        ("transform_pipelines", 0, "input"), "future", draft, composite=True
    )
    assert guide.suggestions == ("image", "other")
    assert rules.transform_presentation(
        ("transform_pipelines", 0, "input"), None, {}, composite=False
    ).suggestions == ("x",)
    assert (
        rules.transform_presentation(
            ("transform_pipelines", 0, "input"),
            None,
            {"composite_model_declarations": None},
            composite=True,
        ).suggestions
        == ()
    )


def test_only_ordered_transform_lists_offer_reordering():
    for path in [("transform_pipelines",), ("transform_pipelines", 0, "steps")]:
        assert rules.transform_presentation(path, None, {}, composite=True).reorderable
    assert not rules.transform_presentation(
        ("transform_pipelines", 0, "steps", 0, "apply_to"), None, {}, composite=True
    ).reorderable


def test_transform_local_validation_uses_schema_but_defers_route_compatibility(
    monkeypatch,
):
    monkeypatch.setattr(
        rules, "TRANSFORMS", SimpleNamespace(resolve_key=lambda name: name)
    )
    values, initial = sections()
    section = values["transform_pipelines"]
    spec = editor_spec(section.annotation, section.field_info)
    raw = [
        {
            "input": "not_declared_yet",
            "steps": [
                {"name": "resize", "apply_to": ["training"], "random_apply": 0.5}
            ],
        }
    ]
    assert evaluate_section(section, spec, new_node(spec, raw), initial=initial) == (
        raw,
        {},
    )
    raw[0]["steps"][0]["apply_to"] = ["training", "training"]
    assert (0, "steps", 0, "apply_to") in evaluate_section(
        section, spec, new_node(spec, raw), initial=initial
    )[1]


def test_transform_reordering_changes_collected_order():
    values, _ = sections()
    section = values["transform_pipelines"]
    spec = editor_spec(section.annotation, section.field_info)
    raw = [
        {"steps": [{"name": "first"}, {"name": "second"}]},
        {"steps": [{"name": "third"}]},
    ]
    node = new_node(spec, raw)
    steps = node["entries"][0]["node"]["fields"]["steps"]
    edit_entry(spec, node, steps["entries"][0]["node"]["id"], "down", None)
    edit_entry(spec, node, node["entries"][0]["node"]["id"], "down", None)
    assert collect_node(spec, node)[0] == [
        {"steps": [{"name": "third"}]},
        {"steps": [{"name": "second"}, {"name": "first"}]},
    ]


def test_transform_dropdown_preserves_alias_and_registry_errors_are_local(monkeypatch):
    def resolve(name):
        if name in {"resize", "resize_alias"}:
            return "resize"
        raise KeyError(name)

    monkeypatch.setattr(
        rules,
        "TRANSFORMS",
        SimpleNamespace(canonical_keys=lambda: ("resize",), resolve_key=resolve),
    )
    assert rules.transform_name_options(
        ("transform_pipelines", 0, "steps", 0, "name"), "resize_alias", []
    ) == ["resize", "resize_alias"]
    assert rules.validate_transform_names([{"steps": [{"name": "resize_alias"}]}]) == {}
    assert (0, "steps", 0, "name") in rules.validate_transform_names(
        [{"steps": [{"name": "missing"}]}]
    )
