"""External component docs link out instead of rendering Sphinx as Markdown."""

import pytest

pytest.importorskip("dash_mantine_components")

from dash import dcc, html

from benchrep.gui.shared import layout


def walk(component):
    yield component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, list) else [children]:
        if hasattr(child, "to_plotly_json"):
            yield from walk(child)


@pytest.mark.parametrize(
    "registry, name, expected_url",
    [
        ("optimizer", "adam", "https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html"),
        ("optimizer", "adamw", "https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html"),
        ("optimizer", "sgd", "https://docs.pytorch.org/docs/2.15/generated/torch.optim.SGD.html"),
        ("lr_scheduler", "step", "https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.StepLR.html"),
        ("transform", "color_jitter", "https://docs.pytorch.org/vision/0.29/generated/torchvision.transforms.v2.ColorJitter.html"),
        ("callback", "rich_progress_bar", "https://lightning.ai/docs/pytorch/2.6.6/api/lightning.pytorch.callbacks.RichProgressBar.html"),
        ("logger", "csv", "https://lightning.ai/docs/pytorch/2.6.6/api/lightning.pytorch.loggers.csv_logs.html#lightning.pytorch.loggers.csv_logs.CSVLogger"),
        ("internal_clustering_metric", "silhouette", "https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.silhouette_score.html"),
    ],
)
def test_external_component_links_preserve_signature(monkeypatch, registry, name, expected_url):
    versions = {
        "torch": "2.15.0+cpu", "torchvision": "0.29.2+cu130",
        "lightning": "2.6.6", "scikit-learn": "1.9.1",
    }
    monkeypatch.setattr(layout, "version", versions.__getitem__)
    details = layout.registry_documentation(registry, name)["selected_component"]
    components = list(walk(layout.render_registry_documentation(registry, name)))
    link = next(c for c in components if isinstance(c, html.A))
    assert link.href == expected_url
    assert link.target == "_blank"
    assert "noopener" in link.rel
    assert any(isinstance(c, html.Code) and c.children == details["signature"] for c in components)
    assert any(isinstance(c, html.Code) and c.children == details["target"] for c in components)
    assert not any(isinstance(c, dcc.Markdown) and c.children == details["description"] for c in components)


@pytest.mark.parametrize("installed", ["2.16.0.dev20261124+cpu", None])
def test_unpublished_or_missing_version_uses_stable_docs(monkeypatch, installed):
    def version(distribution):
        if installed is None:
            raise layout.PackageNotFoundError(distribution)
        return installed

    monkeypatch.setattr(layout, "version", version)
    components = list(walk(layout.render_registry_documentation("optimizer", "adam")))
    link = next(c for c in components if isinstance(c, html.A))
    assert link.href == "https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html"


@pytest.mark.parametrize(
    "target, expected_type",
    [("benchrep.custom.Component", dcc.Markdown), ("my_lab.components.Custom", html.Pre)],
)
def test_owned_and_custom_descriptions_remain_available(monkeypatch, target, expected_type):
    data = layout.registry_documentation("optimizer", "adam")
    description = "Custom **description**\n\n.. math::\n    x < y\n<script>alert(1)</script>"
    data["selected_component"].update(target=target, description=description)
    monkeypatch.setattr(layout, "registry_documentation", lambda *args: data)
    components = list(walk(layout.render_registry_documentation("optimizer", "adam")))
    assert any(isinstance(c, expected_type) and c.children == description for c in components)
    assert not any(isinstance(c, html.A) for c in components)


def test_external_component_without_docstring_has_fallback(monkeypatch):
    data = layout.registry_documentation("optimizer", "adam")
    data["selected_component"].update(target="my_lab.Custom", description=None)
    monkeypatch.setattr(layout, "registry_documentation", lambda *args: data)
    components = list(walk(layout.render_registry_documentation("optimizer", "adam")))
    assert any(isinstance(c, html.Pre) and c.children == "No docstring available." for c in components)
