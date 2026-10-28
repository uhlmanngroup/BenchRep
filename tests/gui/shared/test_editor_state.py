"""Regression coverage for shared GUI state. Requires the gui optional dependencies."""

from copy import deepcopy
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

pytest.importorskip("dash_mantine_components")
from benchrep.assembly.schemas import EvaluationConfig, PredictionConfig, TrainingConfig
from benchrep.gui.shared.state import (
    MISSING,
    collect_node,
    edit_entry,
    edit_node,
    editor_spec,
    json_value,
    new_node,
)
from benchrep.gui.shared.validation import validate_value


class Child(BaseModel):
    size: int = 4


class Example(BaseModel):
    model_config = ConfigDict(extra="allow")
    nested: Child | None = None
    items: list[Child] = Field(default_factory=list)
    mapping: dict[str, Child] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    setting: bool | Literal["auto"] = "auto"


@pytest.mark.parametrize("schema", [TrainingConfig, PredictionConfig, EvaluationConfig])
def test_every_workflow_field_can_be_described(schema):
    for info in schema.model_fields.values():
        spec = editor_spec(info.rebuild_annotation(), info)
        assert collect_node(spec, new_node(spec))[0] is MISSING
        if not info.is_required():
            default = json_value(info.get_default(call_default_factory=True))
            assert collect_node(spec, new_node(spec, default))[0] == default


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"nested": None},
        {"params": {}},
        {"params": {"x": 1}},
        {"setting": False},
        {"setting": "auto"},
        {"extension": 123},
    ],
)
def test_roundtrip_preserves_presence_and_types(raw):
    spec = editor_spec(Example)
    assert collect_node(spec, new_node(spec, raw)) == (raw, {})


def test_omission_retains_hidden_editor_value():
    spec = editor_spec(Example)
    node = new_node(spec, {"nested": {"size": 9}})
    identity = node["fields"]["nested"]["id"]
    edit_node(spec, node, identity, "mode", "omit")
    assert collect_node(spec, node)[0] == {}
    edit_node(spec, node, identity, "mode", "value")
    assert collect_node(spec, node)[0] == {"nested": {"size": 9}}


def test_reset_materializes_schema_default():
    spec = editor_spec(Example)
    node = new_node(spec, {"nested": {"size": 9}})
    identity = node["fields"]["nested"]["fields"]["size"]["id"]
    edit_node(spec, node, identity, "reset", None)
    assert collect_node(spec, node)[0]["nested"]["size"] == 4
    assert node["fields"]["nested"]["fields"]["size"]["mode"] == "value"


def test_list_reordering_keeps_entry_identity_and_invalid_values():
    spec = editor_spec(Example)
    node = new_node(spec, {"items": [{"size": "invalid"}, {"size": 2}]})
    identity = node["fields"]["items"]["entries"][0]["node"]["id"]
    edit_entry(spec, node, identity, "down", None)
    assert node["fields"]["items"]["entries"][1]["node"]["id"] == identity
    assert collect_node(spec, node)[0]["items"][1]["size"] == "invalid"
    edit_entry(spec, node, identity, "remove", None)
    assert collect_node(spec, node)[0]["items"] == [{"size": 2}]


def test_duplicate_mapping_keys_are_errors_not_silent_overwrites():
    spec = editor_spec(Example)
    node = new_node(spec, {"mapping": {"a": {"size": 1}, "b": {"size": 2}}})
    identity = node["fields"]["mapping"]["entries"][1]["node"]["id"]
    edit_entry(spec, node, identity, "key", "a")
    assert ("mapping", "a") in collect_node(spec, node)[1]
    edit_entry(spec, node, identity, "key", "renamed")
    assert collect_node(spec, node)[0]["mapping"]["renamed"] == {"size": 2}


def test_invalid_json_is_retained_and_reported():
    spec = editor_spec(Example)
    node = new_node(spec, {"params": {}})
    target = node["fields"]["params"]
    edit_node(spec, node, target["id"], "mode", "value")
    edit_node(spec, node, target["id"], "value", "{unfinished")
    raw, errors = collect_node(spec, node)
    assert raw["params"] == "{unfinished"
    assert ("params",) in errors


def test_switching_union_branches_retains_each_value():
    spec = editor_spec(bool | str)
    node = new_node(spec, False)
    original = node["branch"]
    edit_node(spec, node, node["id"], "branch", str(1 - original))
    child = node["branches"][str(node["branch"])]
    edit_node(spec, node, child["id"], "value", "auto")
    assert collect_node(spec, node)[0] == "auto"
    edit_node(spec, node, node["id"], "branch", str(original))
    assert collect_node(spec, node)[0] is False


def test_enclosing_collection_constraints_are_validated():
    annotation = Annotated[list[Child], Field(min_length=1)]
    assert validate_value(annotation, [])
    assert not validate_value(annotation, [{"size": 2}])


def test_extra_fields_cannot_replace_declared_fields():
    spec = editor_spec(Example)
    node = new_node(spec, {"nested": {"size": 2}, "extension": 9})
    node["extra_text"] = '{"nested": null}'
    assert () in collect_node(spec, node)[1]
    assert collect_node(spec, node)[0]["nested"] == {"size": 2}


def test_input_source_is_not_mutated():
    raw = {"nested": {"size": 9}}
    before = deepcopy(raw)
    spec = editor_spec(Example)
    node = new_node(spec, raw)
    edit_node(spec, node, node["fields"]["nested"]["id"], "reset", None)
    assert raw == before


def test_real_dataset_and_schedule_alternatives_roundtrip():
    from benchrep.assembly.schemas.training_config_schema import (
        SupportedDatasetConfig,
        SupportedLossWeightScheduleConfig,
    )

    cases = [
        (
            SupportedDatasetConfig,
            {"name": "mnist", "params": {"root": "/tmp/data", "split": "train"}},
        ),
        (
            SupportedDatasetConfig,
            {"name": "custom_images", "params": {"channels": [0, 1]}},
        ),
        (
            SupportedLossWeightScheduleConfig,
            {
                "name": "sigmoid",
                "interval": "epoch",
                "params": {
                    "start": 0,
                    "end": 10,
                    "start_factor": 0.0,
                    "end_factor": 1.0,
                },
            },
        ),
    ]
    for annotation, raw in cases:
        spec = editor_spec(annotation)
        assert collect_node(spec, new_node(spec, raw)) == (raw, {})
        assert not validate_value(annotation, raw)


def test_real_composite_nested_mapping_roundtrip():
    info = TrainingConfig.model_fields["composite_model_assembly"]
    raw = {
        "encode": {
            "component": "encoder",
            "inputs": {"x": "expects.image"},
            "outputs": {"features": "produces.features"},
        }
    }
    spec = editor_spec(info.rebuild_annotation(), info)
    assert collect_node(spec, new_node(spec, raw)) == (raw, {})
    assert not validate_value(info.rebuild_annotation(), raw)


def test_real_evaluation_scalar_or_list_roundtrip():
    from benchrep.assembly.schemas.evaluation_config_schema import (
        EvaluationRidgeProbeConfig,
    )

    for value in (1.0, [0.1, 1.0, 10.0]):
        raw = {"alpha": value}
        spec = editor_spec(EvaluationRidgeProbeConfig)
        assert collect_node(spec, new_node(spec, raw)) == (raw, {})
        assert not validate_value(EvaluationRidgeProbeConfig, raw)


def test_union_dispatch_preserves_loaded_aliases():
    from pydantic import BeforeValidator, Discriminator, Tag

    class Builtin(BaseModel):
        name: Literal["linear"]
        params: Child

    class Custom(BaseModel):
        name: str
        params: dict[str, Any]

    def normalize(value):
        return (
            {**value, "name": "linear"}
            if value.get("name") == "linear_alias"
            else value
        )

    annotation = Annotated[
        Annotated[Builtin, Tag("linear")] | Annotated[Custom, Tag("custom")],
        Discriminator(
            lambda value: "linear" if value.get("name") == "linear" else "custom"
        ),
        BeforeValidator(normalize),
    ]
    spec = editor_spec(annotation)
    raw = {"name": "linear_alias", "params": {"size": 10}}
    node = new_node(spec, raw)
    assert node["branch"] == 0
    assert collect_node(spec, node) == (raw, {})


def test_registered_union_keeps_separate_custom_parameter_edits():
    import json

    from benchrep.assembly.schemas.training_config_schema import TrainingLossTermConfig
    from benchrep.gui.shared.state import union_variant_details

    field = TrainingLossTermConfig.model_fields["weight_schedule"]
    spec = editor_spec(field.rebuild_annotation(), field)
    branches = {union_variant_details(spec, i)[0]: i for i in range(len(spec.variants))}
    raw = {"name": "linear", "params": {"end": 10}}
    node = new_node(spec, raw)
    for name, params in (("custom_a", {"a": 1}), ("custom_b", {"b": 2})):
        edit_node(
            spec, node, node["id"], "union_name", json.dumps([name, branches["custom"]])
        )
        child = node["branches"][str(node["branch"])]["fields"]["params"]
        child.update(mode="value", value=json.dumps(params))
    edit_node(
        spec,
        node,
        node["id"],
        "union_name",
        json.dumps(["custom_a", branches["custom"]]),
    )
    assert collect_node(spec, node)[0] == {"name": "custom_a", "params": {"a": 1}}
    edit_node(
        spec, node, node["id"], "union_name", json.dumps(["linear", branches["linear"]])
    )
    assert collect_node(spec, node)[0] == raw


@pytest.mark.parametrize("annotation,value", [(dict[str, Any] | None, {}), (dict | None, {}), (tuple[int, ...] | None, [])])
def test_optional_json_containers_preserve_empty(annotation, value):
    spec = editor_spec(annotation)
    raw, errors = collect_node(spec, new_node(spec, value))
    assert raw == value
    assert not errors
