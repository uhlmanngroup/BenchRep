"""Evaluation metadata, authoring, resolver checks, and export integration."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("dash_mantine_components")
import anndata as ad
import numpy as np
import pandas as pd
import torch
from plotly.io.json import to_json_plotly

from benchrep.assembly.config import load_yaml
from benchrep.assembly.schemas.evaluation_config_schema import (
    EvaluationConfig,
    EvaluationRunConfig,
)
from benchrep.gui.evaluate import create_evaluation_app
from benchrep.gui.rules import evaluate as rules
from benchrep.gui.shared.fields import render_editor
from benchrep.gui.shared.forms import evaluate_section
from benchrep.gui.shared.state import MISSING, collect_node, editor_spec, new_node
from benchrep.utils.workflow_overrides import EvaluationOverrides


@pytest.fixture
def artifacts(tmp_path):
    path = tmp_path / "embeddings.h5ad"
    adata = ad.AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {"label": [0, 1, 0, 1], "subject": ["a", "b", "a", "b"]},
            index=[str(i) for i in range(4)],
        ),
    )
    adata.obsm["X_custom"] = np.ones((4, 2))
    adata.write_h5ad(path)
    bundles = []
    for name in ("first", "second"):
        bundle = tmp_path / name
        bundle.mkdir()
        # Invalid tensor bytes prove that metadata inspection never loads images.
        (bundle / "input.pt").write_bytes(b"not image tensors")
        (bundle / "reconstruction.pt").write_bytes(b"not image tensors")
        torch.save({"label": [0, 1], "sample_id": [3, 4]}, bundle / "obs.pt")
        torch.save(
            {"pair": {"id": name}, "n_examples_exported": 2},
            bundle / "reconstruction_export_metadata.pt",
        )
        bundles.append(bundle)
    manifest = {
        "stage": "prediction",
        "status": "completed",
        "run": {"output_dir": str(tmp_path / "outputs/prediction/test")},
        "exports": {
            "anndata": {"path": "embeddings.h5ad"},
            "reconstructions": {
                "pairs": {
                    name: {"paths": {"bundle_dir": name}, "n_examples_exported": 2}
                    for name in ("first", "second")
                }
            },
        },
    }
    manifest_path = tmp_path / "prediction_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))
    return SimpleNamespace(
        anndata=path, bundles=bundles, manifest=manifest_path, raw=manifest
    )


@pytest.fixture
def context():
    rules.register_builtins()
    return rules.EvaluationSourceContext(overrides=EvaluationOverrides())


def source_draft(artifacts):
    return {
        "stage": "evaluation",
        "source": {
            "prediction_manifest_path": str(artifacts.manifest),
            "reconstruction_pair_id": "first",
        },
    }


def test_metadata_loading_does_not_load_arrays_or_rewrite_authoring(
    context, artifacts, monkeypatch
):
    monkeypatch.setattr(
        ad, "read_h5ad", lambda *a, **kw: pytest.fail("full AnnData read")
    )
    draft = source_draft(artifacts)
    before = deepcopy(draft)
    context.load(draft)
    assert draft == before
    assert all(status == "loaded" for status, _ in context.statuses.values()), (
        context.statuses
    )
    assert context.anndata["columns"] == ("label", "subject")
    assert context.anndata["representations"] == ("X_custom",)
    assert context.reconstructions["columns"] == ("label", "sample_id")
    assert context.inherited[("source", "anndata_path")] == str(artifacts.anndata)
    assert context.action(
        "copy_inherited", ("source", "reconstructions_path"), draft
    ) == str(artifacts.bundles[0])
    assert context.inherited[("run", "output_root")] == str(
        artifacts.manifest.parent / "outputs"
    )
    assert context.inherited[("reconstruction", "n_examples")] == 2


@pytest.mark.parametrize(
    "path,expected",
    [
        (("source", "reconstruction_pair_id"), ("first", "second")),
        (("metrics", "clustering", "external", "label_key"), ("label", "subject")),
        (
            ("metrics", "predictability", "targets", "label", "cv", "group_key"),
            ("label", "subject"),
        ),
        (("reductions", "umap", "params", "use_rep"), ("X", "X_custom")),
        (("clustering", "leiden", "params", "use_rep"), ("X", "X_custom")),
        (("plots", "params", "color_by", 0), ("label", "subject")),
        (("plots", "params", "color_by", 0, "key"), ("label", "subject")),
        (
            ("plots", "params", "reconstruction_grid", "stratify_by"),
            ("label", "sample_id"),
        ),
    ],
)
def test_metadata_suggestions(context, artifacts, path, expected):
    draft = source_draft(artifacts)
    context.load(draft)
    assert context.presentation(path, None, draft).suggestions == expected
    assert context.presentation(
        ("metrics", "predictability", "targets"), None, draft
    ).key_suggestions == ("label", "subject")


def test_changed_sources_clear_suggestions_but_keep_manifest_pair_choices(
    context, artifacts
):
    draft = source_draft(artifacts)
    context.load(draft)
    draft["source"]["reconstruction_pair_id"] = "second"
    assert not context.current(draft)
    assert (
        context.presentation(
            ("metrics", "clustering", "external", "label_key"), None, draft
        ).suggestions
        == ()
    )
    assert context.presentation(
        ("source", "reconstruction_pair_id"), None, draft
    ).suggestions == ("first", "second")
    assert (
        context.action("copy_inherited", ("source", "anndata_path"), draft) is MISSING
    )
    context.load(draft)
    assert context.reconstruction_spec.pair_id == "second"
    draft["source"]["prediction_manifest_path"] = "missing.yaml"
    assert (
        context.presentation(
            ("source", "reconstruction_pair_id"), None, draft
        ).suggestions
        == ()
    )
    context.load(draft)
    assert context.statuses["Manifest"][0] == "error"
    assert not context.anndata and not context.reconstructions


def test_direct_paths_win_and_bad_manifest_does_not_block_metadata(context, artifacts):
    draft = {
        "source": {
            "prediction_manifest_path": "missing.yaml",
            "anndata_path": str(artifacts.anndata),
            "reconstructions_path": str(artifacts.bundles[1]),
        }
    }
    context.load(draft)
    assert context.statuses["Manifest"][0] == "error"
    assert context.statuses["AnnData metadata"][0] == "loaded"
    assert context.statuses["Reconstruction metadata"][0] == "loaded"
    assert not context.inherited
    draft["source"]["prediction_manifest_path"] = str(artifacts.manifest)
    draft["source"]["reconstruction_pair_id"] = "first"
    context.load(draft)
    assert context.reconstruction_spec.bundle_dir == artifacts.bundles[1]
    assert context.anndata_spec.source == "direct_path"


def test_ambiguous_pair_is_explained_without_blocking_anndata(context, artifacts):
    draft = source_draft(artifacts)
    del draft["source"]["reconstruction_pair_id"]
    context.load(draft)
    assert context.statuses["Reconstruction metadata"][0] == "warning"
    assert context.statuses["AnnData metadata"][0] == "loaded"
    assert context.pair_ids == ("first", "second")


def test_override_is_authoritative_and_only_manifest_field_locked(artifacts):
    context = rules.EvaluationSourceContext(
        overrides=EvaluationOverrides(prediction_manifest_path=artifacts.manifest)
    )
    draft = {
        "source": {
            "prediction_manifest_path": "missing.yaml",
            "reconstruction_pair_id": "first",
        }
    }
    context.load(draft)
    assert context.statuses["Manifest"][0] == "loaded"
    sections = rules.evaluation_sections(context=context)
    assert sections[0].locked_paths == frozenset(
        {("source", "prediction_manifest_path")}
    )
    assert not context.validate_source({})
    assert draft["source"]["prediction_manifest_path"] == "missing.yaml"


@pytest.mark.parametrize(
    "raw,valid",
    [
        (MISSING, False),
        ({}, False),
        (None, False),
        ({"anndata_path": "data.h5ad"}, True),
        ({"reconstructions_path": "bundle"}, True),
    ],
)
def test_local_source_requirement_reuses_backend(context, raw, valid):
    assert bool(context.validate_source(raw)) is not valid


def test_all_sections_and_examples_roundtrip_without_resolution(context):
    sections = rules.evaluation_sections(context=context)
    assert {s.key for s in sections} == set(EvaluationConfig.model_fields) - {"stage"}
    for path in (Path(__file__).parents[3] / "examples/configs").rglob(
        "evaluation.yaml"
    ):
        raw = load_yaml(path)
        for section in sections:
            spec = editor_spec(section.annotation, section.field_info)
            value = raw.get(section.key, MISSING)
            node = new_node(spec, value)
            collected, errors = collect_node(spec, node)
            assert not errors, (path, section.key, errors)
            assert collected == value, (path, section.key)
            # Exercise every example's nested editor, including probe unions.
            component = render_editor(
                spec,
                node,
                section=section.key,
                path=(section.key,),
                title=section.title,
                presentation=lambda p, v, draft=raw: context.presentation(p, v, draft),
            )
            to_json_plotly(component)


def test_partial_config_components_preserve_presence_and_lock(tmp_path, context):
    path = tmp_path / "draft.yaml"
    path.write_text(
        "stage: evaluation\nsource:\n  anndata_path: null\nplots:\n  enabled: false\n"
    )
    overrides = EvaluationOverrides(
        config_components={"run": EvaluationRunConfig(run_name="supplied")}
    )
    initial = rules.prepare_evaluation_draft(config=path, overrides=overrides)
    assert initial.initial_data["source"] == {"anndata_path": None}
    assert "reductions" not in initial.initial_data
    assert initial.supplied_sections == frozenset({"run"})
    section = next(
        s for s in rules.evaluation_sections(context=context) if s.key == "run"
    )
    spec = editor_spec(section.annotation, section.field_info)
    raw, errors = evaluate_section(
        section, spec, new_node(spec, {"run_name": "changed"}), initial=initial
    )
    assert raw["run_name"] == "supplied" and not errors


def test_advisory_metadata_does_not_reject_manual_column_names(context, artifacts):
    draft = source_draft(artifacts)
    context.load(draft)
    draft["metrics"] = {
        "clustering": {"external": {"enabled": True, "label_key": "runtime_column"}}
    }
    draft["clustering"] = {"kmeans": {"enabled": True, "params": {"n_clusters": 2}}}
    session = rules.evaluation_review_session(context=context)
    parsed = session.parse(draft)
    assert (
        session.resolve(parsed).step_spec.external_clustering_label_key
        == "runtime_column"
    )


@pytest.mark.parametrize("source_kind", ["anndata", "reconstruction", "manifest"])
def test_final_checks_and_export_without_loading_large_artifacts(
    context, artifacts, tmp_path, source_kind, monkeypatch
):
    monkeypatch.setattr(
        ad, "read_h5ad", lambda *a, **kw: pytest.fail("full AnnData read")
    )
    if source_kind == "manifest":
        draft = source_draft(artifacts)
    else:
        field = "anndata_path" if source_kind == "anndata" else "reconstructions_path"
        path = artifacts.anndata if source_kind == "anndata" else artifacts.bundles[0]
        draft = {"stage": "evaluation", "source": {field: str(path)}}
    session = rules.evaluation_review_session(context=context)
    destination = str(tmp_path / "exports")
    report = session.check(draft, destination, "evaluation.yaml", {})
    assert report["ready"], report
    assert len(report["rows"]) == 4 and report["graph"] is None
    assert yaml.safe_load(report["authored_yaml"]) == draft
    assert "steps" in report["summary"]
    authored, resolved = session.save(draft, destination, "evaluation.yaml", {})
    assert authored.name == "authored_evaluation_config.yaml"
    assert resolved.name == "resolved_evaluation_config.yaml"
    assert yaml.safe_load(authored.read_text()) == draft
    assert session.result.overrides.config_components is None


def test_source_change_after_check_blocks_save(context, artifacts, tmp_path):
    draft = source_draft(artifacts)
    session = rules.evaluation_review_session(context=context)
    destination = str(tmp_path / "exports")
    assert session.check(draft, destination, "evaluation.yaml", {})["ready"]
    with artifacts.anndata.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="source|Source"):
        session.save(draft, destination, "evaluation.yaml", {})


def test_dependency_checks_use_backend_preconditions(
    context, artifacts, tmp_path, monkeypatch
):
    from benchrep.runtime import evaluate_run_validation as backend

    seen = []

    def fail(*, run_spec):
        seen.append(run_spec)
        raise ImportError("Missing optional backend")

    monkeypatch.setattr(backend, "_validate_enabled_step_preconditions", fail)
    report = rules.evaluation_review_session(context=context).check(
        source_draft(artifacts), str(tmp_path / "exports"), "evaluation.yaml", {}
    )
    assert len(seen) == 1
    assert report["rows"][2]["status"] == "failed"
    assert not report["ready"]


def test_app_builds_without_source_and_omits_graph_stage():
    app = create_evaluation_app()
    response = app.server.test_client().get("/_dash-layout")
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "Evaluation configuration" in text
    assert "Model graph" not in text
    assert app.wizard_session.graph is None


@pytest.fixture
def evaluation_browser(artifacts, tmp_path):
    import os
    from threading import Thread

    from werkzeug.serving import make_server

    playwright = pytest.importorskip("playwright.sync_api")
    # Prefill unrelated sections so tests exercise source authoring independently.
    config = tmp_path / "draft.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "metrics": {"clustering": {"external": {"label_key": "manual_label"}}},
                "plots": {"params": {"color_by": ["manual_label"]}},
            }
        )
    )
    app = create_evaluation_app(config=config, export_directory=tmp_path / "exports")
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as manager:
            browser = manager.chromium.launch(
                executable_path=os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.wait_for_function("window.brFieldInteractionsInstalled === true")
            yield page, app, artifacts
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_browser_source_suggestions_and_review(evaluation_browser):
    from playwright.sync_api import expect

    page, app, artifacts = evaluation_browser
    page.get_by_role(
        "button", name="Enter a value for Prediction Manifest Path", exact=True
    ).click()
    page.get_by_role("textbox", name="Prediction Manifest Path", exact=True).fill(
        str(artifacts.manifest)
    )
    page.get_by_role("button", name="Load source", exact=True).click()
    expect(page.locator(".br-source-context")).to_contain_text("2 observation columns")
    page.get_by_role(
        "button", name="Enter a value for Reconstruction pair ID", exact=True
    ).click()
    pair = page.get_by_role("textbox", name="Reconstruction pair ID", exact=True)
    pair.click()
    page.get_by_role("option", name="first", exact=True).click()
    page.get_by_role("button", name="Load source", exact=True).click()
    expect(page.locator(".br-source-context")).to_contain_text("Pair: first")
    page.get_by_role("tab", name="Metrics", exact=False).click()
    label = page.get_by_role("textbox", name="Label Key", exact=True)
    label.fill("")
    label.click()
    page.get_by_role("option", name="subject", exact=True).click()
    page.get_by_role("tab", name="Source", exact=False).click()
    page.get_by_role("textbox", name="Reconstruction pair ID", exact=True).fill(
        "second"
    )
    page.get_by_role("button", name="Load source", exact=True).click()
    expect(page.locator(".br-source-context")).to_contain_text("Pair: second")
    page.get_by_role("tab", name="Metrics", exact=False).click()
    expect(page.get_by_role("textbox", name="Label Key", exact=True)).to_have_value(
        "subject"
    )
    page.get_by_role("tab", name="Review & finish", exact=False).click()
    page.get_by_role("button", name="Check configuration", exact=True).click()
    expect(page.get_by_role("button", name="Save & finish", exact=True)).to_be_enabled(
        timeout=15000
    )
    expect(page.locator("#wizard-check-report .br-check-row")).to_have_count(4)
    assert app.wizard_session.report["ready"]
    page.get_by_role("tab", name="Source", exact=False).click()
    page.get_by_role("button", name="Reload source", exact=True).click()
    page.get_by_role("tab", name="Review & finish", exact=False).click()
    expect(
        page.get_by_role("button", name="Save & finish", exact=True)
    ).to_be_disabled()
    page.get_by_role("button", name="Check configuration", exact=True).click()
    expect(page.get_by_role("button", name="Save & finish", exact=True)).to_be_enabled(
        timeout=15000
    )
    page.get_by_role("button", name="Save & finish", exact=True).click()
    expect(page.locator("#wizard-save-message")).to_contain_text(
        "Authored configuration:"
    )
    assert (
        app.wizard_session.result.authored_config["source"]["reconstruction_pair_id"]
        == "second"
    )
    assert "anndata_path" not in app.wizard_session.result.authored_config["source"]


def test_h5ad_inspection_never_reads_dataset_values(artifacts, monkeypatch):
    import h5py

    def forbidden(*args, **kwargs):
        pytest.fail("HDF5 dataset values must not be read for suggestions")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbidden)
    assert rules.inspect_anndata_metadata(artifacts.anndata)["columns"] == (
        "label",
        "subject",
    )


def test_optional_reconstruction_metadata_and_dataframe_observations(
    context, artifacts
):
    bundle = artifacts.bundles[0]
    (bundle / "reconstruction_export_metadata.pt").unlink()
    torch.save(pd.DataFrame({"patient": ["a", "b"]}), bundle / "obs.pt")
    draft = {"source": {"reconstructions_path": str(bundle)}}
    context.load(draft)
    assert context.statuses["Reconstruction metadata"][0] == "loaded"
    assert context.reconstructions["columns"] == ("patient",)
    assert context.reconstructions["pair"] == {}


def test_evaluation_launcher_uses_artifacts_without_model_family(monkeypatch):
    import sys
    from threading import Event

    from flask import Flask

    from benchrep.gui import configure_workflow

    result = object()
    session = SimpleNamespace(done=Event(), result=result)
    session.done.set()
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(server=Flask(__name__), wizard_session=session)

    monkeypatch.setitem(
        sys.modules,
        "benchrep.gui.evaluate",
        SimpleNamespace(create_evaluation_app=factory),
    )
    overrides = EvaluationOverrides(prediction_manifest_path="manifest.yaml")
    assert (
        configure_workflow(
            stage="evaluation", overrides=overrides, port=0, open_browser=False
        )
        is result
    )
    assert captured["overrides"] is overrides
    assert "model_family" not in captured and "compatibility_policy" not in captured


def test_evaluation_launcher_rejects_inapplicable_arguments():
    from benchrep.gui import configure_workflow
    from benchrep.interfaces.model_families import COMPOSITE_FAMILY
    from benchrep.utils.workflow_overrides import PredictionOverrides

    with pytest.raises(TypeError, match="EvaluationOverrides"):
        configure_workflow(stage="evaluation", overrides=PredictionOverrides())
    with pytest.raises(ValueError, match="model family"):
        configure_workflow(stage="evaluation", model_family=COMPOSITE_FAMILY)
    with pytest.raises(ValueError, match="only configurable for prediction"):
        configure_workflow(stage="evaluation", compatibility_policy="warn")
