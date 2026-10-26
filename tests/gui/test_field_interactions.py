"""Optional browser regressions for field help and persistent local collapse.

Requires Playwright and its Chromium browser. PLAYWRIGHT_CHROMIUM_EXECUTABLE
can point to an existing Chromium executable instead.
"""

import os
from pathlib import Path
from threading import Thread

import pytest

pytest.importorskip("dash_mantine_components")
playwright = pytest.importorskip("playwright.sync_api")

from dash import Dash
from pydantic import BaseModel, Field
from werkzeug.serving import make_server

from benchrep.gui.shared import fields
from benchrep.gui.shared.forms import SectionSpec, build_workflow_app
from benchrep.gui.shared.state import WizardDraft


class Request(BaseModel):
    label: str = Field(default="original", description="A **label** with `inline code`.")
    frequency: int = Field(default=20, gt=0)


class Settings(BaseModel):
    requests: list[Request]


@pytest.fixture
def page():
    app = Dash(
        __name__,
        assets_folder=str(Path(fields.__file__).parents[1] / "assets"),
        suppress_callback_exceptions=True,
    )
    build_workflow_app(
        app=app,
        stage="training",
        sections=[
            SectionSpec(key="settings", title="Settings", annotation=Settings),
            SectionSpec(key="other", title="Other", annotation=Request),
        ],
        initial=WizardDraft(
            initial_data={
                "settings": {"requests": [{"label": "original", "frequency": 20}]},
                "other": {"label": "other", "frequency": 10},
            },
            supplied_sections=frozenset(),
        ),
        title="Field interactions",
        subtitle="",
    )
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as manager:
            browser = manager.chromium.launch(
                executable_path=os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.wait_for_function("window.brFieldInteractionsInstalled === true")
            yield page
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_help_opens_on_first_click_and_dismisses_natively(page):
    button = page.get_by_role("button", name="Documentation for Label", exact=True)
    button.click()
    popup = page.locator(".br-field-help-popup:popover-open")
    playwright.expect(popup).to_be_visible()
    playwright.expect(popup.locator("code").last).to_have_text("inline code")
    playwright.expect(button).to_have_attribute("aria-expanded", "true")
    page.keyboard.press("Escape")
    playwright.expect(popup).to_have_count(0)
    playwright.expect(button).to_be_focused()
    playwright.expect(button).to_have_attribute("aria-expanded", "false")
    page.keyboard.press("Enter")
    playwright.expect(popup).to_be_visible()
    page.get_by_role("heading", name="Field interactions", exact=True).click()
    playwright.expect(popup).to_have_count(0)
    button.click()
    page.set_viewport_size({"width": 550, "height": 600})
    page.wait_for_timeout(100)
    box = popup.bounding_box()
    assert 0 <= box["x"] and box["x"] + box["width"] <= 550
    assert 0 <= box["y"] and box["y"] + box["height"] <= 600


def test_collapse_preserves_fast_edits_and_survives_navigation(page):
    page.get_by_role("button", name="Documentation for Label", exact=True).click()
    page.keyboard.press("Escape")
    label = page.get_by_role("textbox", name="Label", exact=True)
    frequency = page.get_by_role("textbox", name="Frequency", exact=True)
    frequency.fill("0")
    error = page.locator(".br-field-mode-error").filter(has_text="greater than 0")
    playwright.expect(error).to_be_visible()
    frequency.fill("21")
    playwright.expect(error).to_have_count(0)
    label.fill("")
    label.press_sequentially("rapid changes", delay=5)
    page.get_by_role("button", name="Collapse Requests", exact=True).click()
    # Leave immediately, without waiting for the collapse persistence response.
    page.get_by_role("tab", name="Other").click()
    page.get_by_role("textbox", name="Label", exact=True).wait_for()
    page.get_by_role("tab", name="Settings").click()
    expand = page.get_by_role("button", name="Expand Requests", exact=True)
    playwright.expect(expand).to_be_visible()
    playwright.expect(page.get_by_role("textbox", name="Label", exact=True)).to_be_hidden()
    expand.click()
    playwright.expect(page.get_by_role("textbox", name="Label", exact=True)).to_have_value("rapid changes")
    page.wait_for_function("document.getElementById('wizard-yaml-preview').textContent.includes('rapid changes')")
    # Native help must still work after the editor subtree has been replaced.
    page.get_by_role("button", name="Documentation for Label", exact=True).click()
    playwright.expect(page.locator(".br-field-help-popup:popover-open")).to_be_visible()
