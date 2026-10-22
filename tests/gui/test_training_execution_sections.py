"""Remaining training sections reuse schema validation and preserve raw inputs."""

from types import SimpleNamespace

import pytest

pytest.importorskip("dash_mantine_components")

from benchrep.assembly.schemas.training_config_schema import TrainingConfig
from benchrep.gui.rules import train as rules
from benchrep.gui.shared.fields import render_editor
from benchrep.gui.shared.forms import evaluate_section
from benchrep.gui.shared.state import (
    MISSING,
    WizardDraft,
    collect_node,
    editor_spec,
    new_node,
)
from benchrep.interfaces.model_families import COMPOSITE_FAMILY, VAE_FAMILY


class Registry:
    def canonical_keys(self):
        return ("custom", "builtin")

    def resolve_key(self, name):
        if name == "alias":
            return "builtin"
        if name not in self.canonical_keys():
            raise KeyError(name)
        return name


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    monkeypatch.setattr(
        rules,
        "_REGISTERED_TRAINING_FIELDS",
        {
            route: (Registry(), label)
            for route, (_, label) in rules._REGISTERED_TRAINING_FIELDS.items()
        },
    )


def sections(family=VAE_FAMILY, model=None, datamodule=None, supplied=None):
    initial = WizardDraft(
        initial_data=supplied or {}, supplied_sections=frozenset(supplied or {})
    )
    result = rules.training_sections(
        model_family=family,
        overrides=SimpleNamespace(model=model, datamodule=datamodule),
        initial=initial,
    )
    return {section.key: section for section in result}, initial


def evaluate(key, data, **kwargs):
    if key == "overrides":
        kwargs.setdefault("model", type("ExternalModel", (), {}))

    all_sections, initial = sections(**kwargs)
    section = all_sections[key]
    spec = editor_spec(section.annotation, section.field_info)
    node = new_node(spec, data)
    return evaluate_section(section, spec, node, initial=initial)


def test_all_training_fields_are_covered():
    canonical, _ = sections()
    composite, _ = sections(COMPOSITE_FAMILY)
    external, _ = sections(model=type("ExternalModel", (), {}))

    assert (
        set(canonical) | set(composite) | set(external)
        == set(TrainingConfig.model_fields) - {"stage"}
    )


def test_external_model_hides_optimizer_but_not_training_controls():
    result, _ = sections(model=object(), datamodule=object())
    assert "optimizer" not in result
    assert {
        "trainer",
        "logger",
        "checkpointing",
        "early_stopping",
        "additional_callbacks",
        "inspection",
    } <= result.keys()
    assert "whole_model only" in result["inspection"].description
    assert "overrides" not in result


@pytest.mark.parametrize(
    "key,data",
    [
        (
            "optimizer",
            {
                "name": "alias",
                "params": {"lr": 0.001},
                "lr_scheduler": {
                    "name": "custom",
                    "params": {"step_size": 5},
                    "interval": "step",
                    "frequency": 2,
                    "monitor": None,
                },
            },
        ),
        (
            "trainer",
            {
                "max_epochs": 3,
                "devices": [0, 1],
                "precision": "16-mixed",
                "deterministic": "warn",
                "accumulate_grad_batches": 4,
            },
        ),
        (
            "trainer",
            {"devices": 1, "precision": 32, "deterministic": True, "max_epochs": None},
        ),
        ("trainer", {"devices": "auto", "deterministic": False}),
        (
            "logger",
            {
                "name": "custom",
                "params": {"save_dir": "logs"},
                "wandb_api_key_path": "private/key.txt",
            },
        ),
        ("checkpointing", {"monitor": None, "save_top_k": 3, "save_last": True}),
        (
            "early_stopping",
            {"monitor": "custom/metric", "patience": 0, "verbose": True},
        ),
        (
            "additional_callbacks",
            [
                {"name": "custom", "params": {"nested": {"a": [1, 2]}}},
                {"name": "builtin"},
            ],
        ),
        (
            "inspection",
            {
                "torchview": {"enabled": True},
                "gradient_norms": [
                    {
                        "kind": "per_loss",
                        "target": "arbitrary_component",
                        "losses": ["reconstruction.example"],
                        "every_n_batches": 2,
                    },
                    {
                        "kind": "optimizer_gradients",
                        "target": "whole_model",
                        "every_n_optimizer_steps": 3,
                    },
                ],
            },
        ),
        ("overrides", {"model": {"params": {"width": 64}}, "datamodule": None}),
    ],
)
def test_raw_values_roundtrip_without_resolution(key, data):
    raw, errors = evaluate(key, data)
    assert errors == {}
    assert raw == data
    all_sections, _ = sections(
        model=type("ExternalModel", (), {}) if key == "overrides" else None,
    )
    section = all_sections[key]
    spec = editor_spec(section.annotation, section.field_info)
    node = new_node(spec, data)
    rendered = render_editor(spec, node, section=key, path=(key,))
    assert rendered is not None
    assert collect_node(spec, node)[0] == data


@pytest.mark.parametrize(
    "key",
    [
        "optimizer",
        "trainer",
        "logger",
        "checkpointing",
        "early_stopping",
        "additional_callbacks",
        "inspection",
        "overrides",
    ],
)
def test_omission_does_not_materialize_defaults(key):
    raw, errors = evaluate(key, MISSING)
    assert raw is MISSING
    assert errors == {}


@pytest.mark.parametrize(
    "key,data",
    [
        (
            "optimizer",
            {
                "name": "builtin",
                "lr_scheduler": {"name": "custom", "params": {"optimizer": "bad"}},
            },
        ),
        ("trainer", {"logger": False}),
        ("trainer", {"max_epochs": -1}),
        ("checkpointing", {"monitor": None, "save_last": False}),
        ("early_stopping", {"patience": -1}),
        (
            "inspection",
            {
                "gradient_norms": [
                    {"kind": "per_loss", "target": "whole_model", "every_n_batches": 0}
                ]
            },
        ),
        ("overrides", {"model": {"params": None}}),
    ],
)
def test_schema_errors_are_reused(key, data):
    assert evaluate(key, data)[1]


def test_registered_choices_and_nested_error_paths():
    assert rules.training_registered_options(
        ("optimizer", "lr_scheduler", "name"), "alias", {}
    ) == ["alias", "builtin", "custom"]
    assert rules.training_registered_options(
        ("additional_callbacks", 2, "name"), None, []
    ) == ["builtin", "custom"]
    assert rules.training_registered_options(("optimizer", "params"), None, {}) is None
    assert ("lr_scheduler", "name") in rules.validate_registered_section(
        "optimizer", {"name": "builtin", "lr_scheduler": {"name": "bad"}}
    )
    assert (1, "name") in rules.validate_registered_section(
        "additional_callbacks", [{"name": "custom"}, {"name": "bad"}]
    )


def test_contextual_suggestions_do_not_validate_other_sections(monkeypatch):
    draft = {
        "composite_model_components": {"encoder_a": {}},
        "losses": {"reconstruction": [{"name": "alias", "id": "image"}]},
    }
    monkeypatch.setattr(rules, "_LOSS_REGISTRIES", {"reconstruction": Registry()})
    path = ("inspection", "gradient_norms", 0, "target")
    assert rules.inspection_presentation(
        path, None, draft, model_family=COMPOSITE_FAMILY, external_model=False
    ).suggestions == ("encoder_a", "whole_model")
    assert rules.inspection_presentation(
        path, None, draft, model_family=VAE_FAMILY, external_model=True
    ).suggestions == ("whole_model",)
    assert (
        "variational_head"
        in rules.inspection_presentation(
            path, None, draft, model_family=VAE_FAMILY, external_model=False
        ).suggestions
    )
    assert rules.inspection_loss_references(draft) == ("reconstruction.image_builtin",)
    assert not evaluate(
        "inspection",
        {"gradient_norms": [{"kind": "per_loss", "target": "not_yet_defined"}]},
    )[1]


def test_supplied_sections_remain_authoritative():
    data = {"name": "custom", "params": {"lr": 0.1}}
    raw, errors = evaluate("optimizer", None, supplied={"optimizer": data})
    assert raw == data
    assert errors == {}


@pytest.mark.parametrize(
    "model,datamodule,excluded",
    [
        (None, None, None),
        (object(), object(), None),
        (type("ExternalModel", (), {}), None, ("datamodule",)),
        (None, type("ExternalDataModule", (), {}), ("model",)),
        (type("ExternalModel", (), {}), object(), ("datamodule",)),
        (object(), type("ExternalDataModule", (), {}), ("model",)),
        (
            type("ExternalModel", (), {}),
            type("ExternalDataModule", (), {}),
            (),
        ),
    ],
)
def test_override_parameters_follow_launcher_classes(model, datamodule, excluded):
    result, _ = sections(model=model, datamodule=datamodule)

    if excluded is None:
        assert "overrides" not in result
        return

    section = result["overrides"]
    assert section.title == "Override parameters"
    assert (
        section.presentation(("overrides",), None, {}).excluded_fields
        == excluded
    )


def test_override_parameters_preserve_values():
    data = {"model": {"params": {"width": 12}}}
    assert evaluate("overrides", data)[0] == data


def test_new_section_registry_defaults():
    result, _ = sections()
    assert result["optimizer"].default_registry == "optimizer"
    assert result["logger"].default_registry == "logger"
    assert result["additional_callbacks"].default_registry == "callback"
