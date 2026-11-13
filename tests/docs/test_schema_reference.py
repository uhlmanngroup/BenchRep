"""Keep generated schema docs complete and faithful without importing BenchRep."""
import ast
import importlib.util
from pathlib import Path
import re

import pytest

pytest.importorskip("griffe", reason="Schema reference tests require the docs extra.")

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "schema_reference", ROOT / "docs_extensions/schema_reference.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.fixture(scope="module")
def catalogue():
    return reference.Catalogue(ROOT)


def fields(catalogue, name):
    return {field["name"]: field for field in catalogue.schema(name)["fields"]}


def test_every_public_schema_and_reachable_alias_is_documented_once(catalogue):
    entries = []
    for path in (ROOT / "docs/reference/schemas").glob("*.md"):
        entries += re.findall(r"^::: (\S+)", path.read_text(), re.MULTILINE)
    assert len(entries) == len(set(entries))
    assert set(entries) == set(catalogue.targets.values())


def test_inherited_fields_and_specializations(catalogue):
    model = fields(catalogue, "TrainingModelConfig")
    assert set(model) == {"name", "params"}
    assert model["name"]["required"]
    assert model["params"]["default"] == "dict()"
    linear = fields(catalogue, "LinearScheduleConfig")
    assert "RampScheduleParams" in linear["params"]["type_html"]
    assert "ScheduleParamsT" not in linear["params"]["type_html"]
    sigmoid = fields(catalogue, "SigmoidScheduleParams")
    assert "end_factor" in sigmoid and "steepness" in sigmoid
    custom = fields(catalogue, "CustomScheduleConfig")
    assert not custom["params"]["required"]
    assert custom["params"]["default"] == "dict()"


def test_defaults_do_not_conflate_required_null_and_factories(catalogue):
    assert fields(catalogue, "MNISTDatasetParams")["root"]["required"]
    nullable = fields(catalogue, "TrainingConfig")["model"]
    assert not nullable["required"] and nullable["default"] == "None"
    factory = fields(catalogue, "TrainingConfig")["run"]
    assert factory["factory"] and factory["default"] == "TrainingRunConfig()"
    assert fields(catalogue, "MNISTDatasetParams")["download"]["default"] == "False"


def test_field_metadata_is_preserved(catalogue):
    for name in catalogue.classes:
        for field in catalogue.schema(name)["fields"]:
            assert field["description"], (name, field["name"])
            assert field["extra"]["omit_behavior"]
            assert field["extra"]["null_behavior"]
    model = fields(catalogue, "TrainingConfig")["model"]
    assert "external model" in model["extra"]["omit_behavior"]
    assert model["notes"]


def test_extra_key_policy_follows_inheritance(catalogue):
    assert catalogue.schema("TrainingModelConfig")["extra_mode"] == "forbid"
    assert catalogue.schema("EvaluationPCAParams")["extra_mode"] == "allow"
    assert catalogue.schema("CompositeModelComponentConfig")["extra_mode"] == "forbid"


def test_aliases_link_types_but_not_literal_strings(catalogue):
    html = catalogue.linked(ast.parse('Literal["TrainingConfig"] | TrainingConfig', mode="eval").body)
    assert html.count("<autoref") == 1
    assert "&quot;" in html or "&#x27;" in html
    alias = catalogue.linked(catalogue.aliases["SupportedDatasetConfig"])
    assert "MNISTDatasetConfig</autoref>" in alias


def test_lambda_factory_is_displayed_without_execution(catalogue):
    # Use an isolated in-memory schema; a factory must never be executed by docs.
    catalogue.classes["DocsOnlyFactory"] = ast.parse('''
class DocsOnlyFactory:
    value: int = Field(default_factory=lambda: 1 / 0)
''').body[0]
    catalogue.paths["DocsOnlyFactory"] = "test.DocsOnlyFactory"
    field = fields(catalogue, "DocsOnlyFactory")["value"]
    assert field["factory"] and field["default"] == "1 / 0"
    del catalogue.classes["DocsOnlyFactory"]
    del catalogue.paths["DocsOnlyFactory"]
