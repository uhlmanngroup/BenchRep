"""Review reports, export safety and authored/resolved separation."""

from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("dash_mantine_components")
from benchrep.gui.shared.review import (
    ReviewSession,
    check_export_destination,
    export_destination,
    save_yaml_exclusively,
)
from pydantic import BaseModel


class Config(BaseModel):
    value: int = 3


@pytest.fixture
def session():
    def resolve(config):
        config.value = 99
        return SimpleNamespace(training_config=config)

    return ReviewSession(
        parse=Config.model_validate,
        resolve=resolve,
        dependencies=lambda config: ("passed", "available"),
        graph=lambda spec: None,
        resolved_config=lambda spec: spec.training_config,
        execution_overrides=object(),
    )


def test_check_and_save_preserve_authored_omission(session, tmp_path):
    report = session.check({}, str(tmp_path), "training.yaml", {})
    assert report["ready"]
    assert report["rows"][-1]["status"] == "skipped"
    assert "99" in report["resolved_yaml"]
    assert session.config.value == 3
    authored_path, resolved_path = session.save(
        {}, str(tmp_path), "training.yaml", {}
    )
    assert authored_path.name == "authored_training_config.yaml"
    assert resolved_path.name == "resolved_training_config.yaml"
    assert authored_path.read_text() == "{}\n"
    assert yaml.safe_load(resolved_path.read_text()) == {"value": 99}
    assert session.result.config_path == authored_path
    assert session.result.resolved_config_path == resolved_path
    assert session.result.config.value == 3
    assert session.result.resolved_config.value == 99
    assert session.result.authored_config == {}
    assert session.result.overrides is session.execution_overrides
    assert session.done.is_set()


@pytest.mark.parametrize("change", ["draft", "directory", "filename", "errors"])
def test_changed_inputs_cannot_save(session, tmp_path, change):
    session.check({}, str(tmp_path), "training.yaml", {})
    draft, directory, filename, errors = {}, str(tmp_path), "training.yaml", {}
    if change == "draft":
        draft = {"value": 7}
    if change == "directory":
        directory = str(tmp_path / "other")
    if change == "filename":
        filename = "other.yaml"
    if change == "errors":
        errors = {"run": [{"message": "bad"}]}
    with pytest.raises(ValueError):
        session.save(draft, directory, filename, errors)
    assert not list(tmp_path.iterdir())


def test_validation_failures_skip_resolution_and_cannot_save(session, tmp_path):
    session.resolve = lambda config: pytest.fail("must not resolve invalid config")
    report = session.check({"value": "bad"}, str(tmp_path), "training.yaml", {})
    assert report["rows"][0]["status"] == "failed"
    assert report["rows"][1]["status"] == "skipped"
    assert report["rows"][3]["status"] == "passed"
    with pytest.raises(ValueError):
        session.save({"value": "bad"}, str(tmp_path), "training.yaml", {})


def test_resolution_failure_is_blocking(session, tmp_path):
    def fail(config):
        raise ValueError("incompatible components")

    session.resolve = fail
    report = session.check({}, str(tmp_path), "training.yaml", {})
    assert not report["ready"]
    assert report["rows"][1]["message"] == "incompatible components"


def test_optional_graph_and_dependency_warnings_allow_save(session, tmp_path):
    session.dependencies = lambda config: ("warning", "torchview unavailable")

    def fail(spec):
        raise RuntimeError("dot unavailable")

    session.graph = fail
    report = session.check({}, str(tmp_path), "training.yaml", {})
    assert report["ready"]
    assert report["rows"][4]["status"] == "warning"
    session.save({}, str(tmp_path), "training.yaml", {})


def test_required_dependency_failure_blocks_save(session, tmp_path):
    def fail(config):
        raise ModuleNotFoundError("logger backend unavailable")

    session.dependencies = fail
    report = session.check({}, str(tmp_path), "training.yaml", {})
    assert report["rows"][2]["status"] == "failed"
    assert not report["ready"]


@pytest.mark.parametrize(
    "filename",
    ["authored_training_config.yaml", "resolved_training_config.yaml"],
)
def test_conflict_after_check_does_not_overwrite(session, tmp_path, filename):
    session.check({}, str(tmp_path), "training.yaml", {})
    target = tmp_path / filename
    target.write_text("existing")

    with pytest.raises(FileExistsError):
        session.save({}, str(tmp_path), "training.yaml", {})

    assert target.read_text() == "existing"
    assert list(tmp_path.iterdir()) == [target]
    assert not session.done.is_set()


def test_exclusive_publication_and_missing_directories(tmp_path):
    target = tmp_path / "nested" / "training.yaml"
    check_export_destination(target)
    assert not target.parent.exists()
    save_yaml_exclusively(target, {"value": 1})
    with pytest.raises(FileExistsError):
        save_yaml_exclusively(target, {"value": 2})
    assert target.read_text() == "value: 1\n"
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize(
    "name", ["../outside.yaml", "nested/config.yaml", "config.txt", "", "."]
)
def test_invalid_export_filename(name, tmp_path):
    with pytest.raises(ValueError):
        export_destination(str(tmp_path), name)


def test_file_as_parent_is_rejected(tmp_path):
    parent = tmp_path / "file"
    parent.write_text("data")
    with pytest.raises(NotADirectoryError):
        check_export_destination(parent / "training.yaml")


def test_failed_serialization_leaves_no_target(tmp_path):
    with pytest.raises(yaml.representer.RepresenterError):
        save_yaml_exclusively(tmp_path / "training.yaml", {"bad": object()})
    assert not list(tmp_path.iterdir())


def test_graph_preview_is_an_image_data_url(session, tmp_path):
    session.graph = lambda spec: '<svg xmlns="http://www.w3.org/2000/svg"/>'
    report = session.check({}, str(tmp_path), "training.yaml", {})
    assert report["graph"].startswith("data:image/svg+xml;base64,")
    assert report["rows"][-1]["status"] == "passed"


@pytest.mark.parametrize("cancel", [False, True])
def test_launcher_returns_and_stops_local_server(monkeypatch, cancel):
    import sys
    import urllib.request
    import webbrowser
    from threading import Event

    from benchrep.gui import configure_workflow
    from flask import Flask

    result = None if cancel else object()
    session = SimpleNamespace(done=Event(), result=result)
    server = Flask(__name__)

    @server.route("/finish")
    def finish():
        session.done.set()
        return "finished"

    fake_app = SimpleNamespace(server=server, wizard_session=session)
    monkeypatch.setitem(
        sys.modules,
        "benchrep.gui.train",
        SimpleNamespace(create_training_app=lambda **kwargs: fake_app),
    )
    address = []

    def open_browser(url):
        address.append(url)
        with urllib.request.urlopen(url + "/finish", timeout=5) as response:
            assert response.read() == b"finished"
        return True

    monkeypatch.setattr(webbrowser, "open", open_browser)
    assert configure_workflow(port=0) is result
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(address[0], timeout=1)


def test_training_review_uses_override_context_and_strips_components(monkeypatch):
    from dataclasses import dataclass

    from benchrep.gui.rules import train as rules

    @dataclass
    class Overrides:
        model: object
        datamodule: object
        config_components: object

        def validate_for(self, family):
            pass

    class Family:
        name = "vae"
        model_base_class = object

    calls = {}
    monkeypatch.setattr(rules, "CanonicalModelFamilySpec", Family)
    monkeypatch.setattr(
        rules, "L", SimpleNamespace(LightningDataModule=object), raising=False
    )
    monkeypatch.setattr(
        rules,
        "parse_training_config",
        lambda raw, **kwargs: calls.update(parse=kwargs) or Config.model_validate(raw),
        raising=False,
    )
    monkeypatch.setattr(
        rules,
        "resolve_component_source",
        lambda value, **kwargs: (
            "external_class" if isinstance(value, type) else "external_instance"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        rules, "get_component_override_name", lambda value: "external", raising=False
    )
    monkeypatch.setattr(
        rules,
        "resolve_training_config",
        lambda cfg, **kwargs: (
            calls.update(resolve=kwargs) or SimpleNamespace(training_config=cfg)
        ),
        raising=False,
    )
    source = Overrides(model=object, datamodule=object(), config_components={"run": {}})
    session = rules.training_review_session(model_family=Family(), overrides=source)
    session.resolve(session.parse({}))
    assert calls["parse"] == {"model_overridden": True, "datamodule_overridden": True}
    assert calls["resolve"]["model_source"] == "external_class"
    assert calls["resolve"]["datamodule_source"] == "external_instance"
    assert session.execution_overrides.config_components is None
    assert session.execution_overrides.model is source.model
    assert session.execution_overrides.datamodule is source.datamodule
    assert source.config_components == {"run": {}}

def test_failed_resolved_save_removes_authored_file(
    session, tmp_path, monkeypatch
):
    from benchrep.gui.shared import review

    session.check({}, str(tmp_path), "training.yaml", {})
    original_save = review.save_yaml_exclusively

    def fail_resolved(path, data):
        if path.name == "resolved_training_config.yaml":
            raise OSError("Simulated write failure")
        original_save(path, data)

    monkeypatch.setattr(review, "save_yaml_exclusively", fail_resolved)

    with pytest.raises(OSError, match="Simulated write failure"):
        session.save({}, str(tmp_path), "training.yaml", {})

    assert not list(tmp_path.iterdir())
    assert session.result is None
    assert not session.done.is_set()
