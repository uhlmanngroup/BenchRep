"""Review, export and completion lifecycle shared by workflow wizards."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, RLock
from typing import Any

import dash_mantine_components as dmc
import yaml
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from benchrep.gui.shared.results import ConfiguredWorkflow

REVIEW_KEY = "__review__"
CHECK_LABELS = (
    "Complete config",
    "Workflow compatibility",
    "Required packages",
    "Export destination",
    "Model graph",
)


def export_destination(directory: str, filename: str) -> Path:
    if not directory or not directory.strip():
        raise ValueError("Enter an export directory.")
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("Enter a filename without directory components.")
    if Path(filename).suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Use a .yaml or .yml filename.")
    return Path(directory).expanduser().absolute() / filename


def export_paths(directory: str, filename: str) -> tuple[Path, Path]:
    """Derive authored and resolved filenames from the workflow filename."""
    base = export_destination(directory, filename)
    stem = f"{base.stem}_config"
    return (
        base.with_name(f"authored_{stem}.yaml"),
        base.with_name(f"resolved_{stem}.yaml"),
    )


def check_export_destination(path: Path) -> str:
    if os.path.lexists(path):
        raise FileExistsError(
            f"Already exists: {path}. Choose another export directory; files are never overwritten."
        )
    ancestor = path.parent
    while not ancestor.exists():
        if ancestor.is_symlink():
            raise ValueError(f"Broken directory symlink: {ancestor}")
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise NotADirectoryError(f"Not a directory: {ancestor}")
    with tempfile.TemporaryFile(dir=ancestor):
        pass
    return f"Ready to save {path}." + (
        " Missing directories will be created when saving."
        if not path.parent.exists()
        else ""
    )


def save_yaml_exclusively(path: Path, data: dict) -> None:
    """Write a complete temporary file, then publish it without replacing a target."""
    payload = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class ReviewSession:
    """Server-owned validation results; browser state cannot authorize a save."""

    parse: Callable[[dict], Any]
    resolve: Callable[[Any], Any]
    dependencies: Callable[[Any], tuple[str, str]]
    graph: Callable[[Any], str | None]
    resolved_config: Callable[[Any], Any]
    execution_overrides: Any
    result: ConfiguredWorkflow | None = None
    done: Event = field(default_factory=Event)
    lock: RLock = field(default_factory=RLock)
    fingerprint: str | None = None
    config: Any = None
    resolved: Any = None
    authored: dict | None = None
    report: dict | None = None

    @staticmethod
    def identity(draft, directory, filename):
        return json.dumps(
            [draft, directory, filename], sort_keys=True, ensure_ascii=False
        )

    def check(self, draft, directory, filename, local_errors):
        with self.lock:
            if self.done.is_set():
                raise ValueError("This wizard session has finished.")
            self.fingerprint = None
            self.config = self.resolved = self.authored = None
            rows = [
                {
                    "label": label,
                    "status": "skipped",
                    "message": "Not run because an earlier check failed.",
                }
                for label in CHECK_LABELS
            ]
            report = {"rows": rows, "ready": False, "graph": None}
            self.report = report
            parsed = resolved = None
            try:
                if any(local_errors.values()):
                    raise ValueError(
                        "Fix the section errors before checking the complete config."
                    )
                parsed = self.parse(deepcopy(draft))
                rows[0].update(
                    status="passed", message="All sections validate together."
                )
            except Exception as error:  # noqa: BLE001 -- surface check failures without crashing the wizard
                rows[0].update(status="failed", message=str(error))
            if parsed is not None:
                try:
                    resolved = self.resolve(parsed.model_copy(deep=True))
                    rows[1].update(
                        status="passed",
                        message="Workflow configuration resolved successfully.",
                    )
                except Exception as error:  # noqa: BLE001 -- surface check failures without crashing the wizard
                    rows[1].update(status="failed", message=str(error))
                try:
                    status, message = self.dependencies(parsed)
                    rows[2].update(status=status, message=message)
                except Exception as error:  # noqa: BLE001 -- surface check failures without crashing the wizard
                    rows[2].update(status="failed", message=str(error))
            try:
                paths = export_paths(directory, filename)
                messages = [check_export_destination(path) for path in paths]
                rows[3].update(status="passed", message="\n".join(messages))
            except Exception as error:  # noqa: BLE001 -- surface check failures without crashing the wizard
                rows[3].update(status="failed", message=str(error))
            if resolved is not None:
                try:
                    svg = self.graph(resolved)
                    report["graph"] = (
                        "data:image/svg+xml;base64,"
                        + base64.b64encode(svg.encode()).decode()
                        if svg
                        else None
                    )
                    rows[4].update(
                        status="passed" if svg else "skipped",
                        message="Composite model specification."
                        if svg
                        else "Only applicable to Composite models.",
                    )
                except Exception as error:  # noqa: BLE001 -- surface check failures without crashing the wizard
                    rows[4].update(
                        status="warning", message=f"Graph preview unavailable: {error}"
                    )
            report["ready"] = (
                parsed is not None
                and resolved is not None
                and all(row["status"] != "failed" for row in rows)
            )
            report["authored_yaml"] = yaml.safe_dump(
                draft, sort_keys=False, allow_unicode=True
            )
            if resolved is not None:
                report["resolved_yaml"] = yaml.safe_dump(
                    self.resolved_config(resolved).model_dump(mode="json"),
                    sort_keys=False,
                    allow_unicode=True,
                )
            self.fingerprint = self.identity(draft, directory, filename)
            self.config, self.resolved, self.authored = (
                parsed,
                resolved,
                deepcopy(draft),
            )
            return deepcopy(report)

    def save(self, draft, directory, filename, local_errors):
        with self.lock:
            if self.done.is_set():
                raise ValueError("This wizard session has finished.")
            if (
                any(local_errors.values())
                or not self.report
                or not self.report["ready"]
                or self.fingerprint != self.identity(draft, directory, filename)
            ):
                raise ValueError(
                    "Configuration or export settings changed. "
                    "Check configuration again before saving."
                )

            authored_path, resolved_path = export_paths(directory, filename)
            for path in (authored_path, resolved_path):
                check_export_destination(path)

            resolved_config = self.resolved_config(self.resolved)
            resolved_data = resolved_config.model_dump(mode="json")

            result = ConfiguredWorkflow(
                config=self.config,
                overrides=self.execution_overrides,
                config_path=authored_path,
                resolved_config_path=resolved_path,
                authored_config=deepcopy(self.authored),
                resolved_config=resolved_config,
            )

            save_yaml_exclusively(authored_path, self.authored)
            try:
                save_yaml_exclusively(resolved_path, resolved_data)
            except Exception:
                authored_path.unlink()
                raise

            self.result = result
            self.done.set()
            return authored_path, resolved_path


def build_review_panel(export_directory: str, filename: str):
    return html.Div(
        [
            html.P(
                "Check the complete configuration before saving. These checks do not run training or guarantee that runtime resources and custom components will work."
            ),
            dmc.TextInput(
                id="wizard-export-directory",
                label="Export directory",
                value=export_directory,
            ),
            dcc.Store(id="wizard-export-filename", data=filename),
            html.P(
                "Saves authored and resolved configurations as separate YAML files."
            ),
            html.Div(id="wizard-review-stale", role="status"),
            dmc.Button("Check configuration", id="wizard-check", n_clicks=0),
            dcc.Loading(
                html.Div(
                    id="wizard-check-report",
                    children=[
                        html.Div(
                            [html.Span("○ "), html.Strong(label)],
                            className="br-check-row",
                        )
                        for label in CHECK_LABELS
                    ],
                ),
                type="circle",
            ),
            html.Div(
                [
                    dmc.Button(
                        "Save & finish", id="wizard-finish", n_clicks=0, disabled=True
                    ),
                    dmc.Button(
                        "Cancel", id="wizard-cancel", n_clicks=0, variant="subtle"
                    ),
                ],
                className="br-actions",
            ),
            html.Div(id="wizard-save-message", role="status"),
        ],
        id="wizard-review-panel",
        style={"display": "none"},
    )


def render_report(report):
    symbols = {"passed": "✓", "failed": "✕", "warning": "!", "skipped": "—"}
    rows = [
        html.Div(
            [
                html.Span(
                    symbols[row["status"]],
                    className="br-check-symbol",
                    **{"aria-label": row["status"]},
                ),
                html.Div(
                    [
                        html.Strong(row["label"]),
                        html.Pre(row["message"], className="br-check-message"),
                    ]
                ),
            ],
            className=f"br-check-row br-check-{row['status']}",
        )
        for row in report["rows"]
    ]
    if "authored_yaml" in report:
        rows.append(
            dmc.Tabs(
                value="authored",
                children=[
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Authored YAML", value="authored"),
                            dmc.TabsTab(
                                "Resolved YAML",
                                value="resolved",
                                disabled="resolved_yaml" not in report,
                            ),
                        ]
                    ),
                    dmc.TabsPanel(
                        [
                            html.P("Saved as the authored configuration, preserving your choices."),
                            html.Pre(
                                report["authored_yaml"], className="br-yaml-preview"
                            ),
                        ],
                        value="authored",
                    ),
                    dmc.TabsPanel(
                        [
                            html.P(
                                "Saved as the resolved configuration for this environment. "
                                "The workflow resolves again when run."
                            ),
                            html.Pre(
                                report.get("resolved_yaml", ""),
                                className="br-yaml-preview",
                            ),
                        ],
                        value="resolved",
                    ),
                ],
            )
        )
    if report.get("graph"):
        rows.append(
            html.Details(
                [
                    html.Summary("Model graph"),
                    html.Img(
                        src=report["graph"],
                        className="br-spec-graph",
                        alt="Composite model specification graph",
                    ),
                ],
                open=True,
            )
        )
    return rows


def register_review_callbacks(app: Dash, session: ReviewSession):
    @app.callback(
        Output("wizard-check-report", "children"),
        Output("wizard-review-receipt", "data"),
        Input("wizard-check", "n_clicks"),
        State("wizard-draft", "data"),
        State("wizard-export-directory", "value"),
        State("wizard-export-filename", "data"),
        State("wizard-section-errors", "data"),
        prevent_initial_call=True,
        running=[(Output("wizard-check", "loading"), True, False)],
    )
    def check(clicks, draft, directory, filename, errors):
        if not clicks:
            raise PreventUpdate
        report = session.check(draft, directory, filename, errors)
        return render_report(report), {
            "draft": draft,
            "directory": directory,
            "filename": filename,
            "ready": report["ready"],
        }

    app.clientside_callback(
        """function(receipt, draft, directory, filename, errors, finished) {
        const invalid = Object.values(errors || {}).some(items => items.length);
        const stale = receipt && (JSON.stringify(receipt.draft) !== JSON.stringify(draft) || receipt.directory !== directory || receipt.filename !== filename);
        return [!!finished || invalid, !!finished || invalid || !receipt || !receipt.ready || !!stale,
            finished ? 'Session finished.' : stale ? 'Previous checks are outdated. Check configuration again.' : invalid ? 'Fix section errors before checking the complete config.' : '',
            !!finished, stale || !receipt ? '○' : receipt.ready ? '✓' : '!'];
    }""",
        Output("wizard-check", "disabled"),
        Output("wizard-finish", "disabled"),
        Output("wizard-review-stale", "children"),
        Output("wizard-cancel", "disabled"),
        Output("wizard-section-status-__review__", "children"),
        Input("wizard-review-receipt", "data"),
        Input("wizard-draft", "data"),
        Input("wizard-export-directory", "value"),
        Input("wizard-export-filename", "data"),
        Input("wizard-section-errors", "data"),
        Input("wizard-finished", "data"),
    )

    @app.callback(
        Output("wizard-save-message", "children"),
        Output("wizard-finished", "data"),
        Input("wizard-finish", "n_clicks"),
        Input("wizard-cancel", "n_clicks"),
        State("wizard-draft", "data"),
        State("wizard-export-directory", "value"),
        State("wizard-export-filename", "data"),
        State("wizard-section-errors", "data"),
        prevent_initial_call=True,
    )
    def finish(save_clicks, cancel_clicks, draft, directory, filename, errors):
        if ctx.triggered_id == "wizard-cancel":
            with session.lock:
                session.done.set()
            return dmc.Alert("Cancelled. You can close this tab.", color="gray"), True
        if not save_clicks:
            raise PreventUpdate
        try:
            authored_path, resolved_path = session.save(
                draft, directory, filename, errors
            )
        except Exception as error:  # noqa: BLE001
            return dmc.Alert(str(error), color="red"), False

        return dmc.Alert(
            [
                html.Div(f"Authored configuration: {authored_path}"),
                html.Div(f"Resolved configuration: {resolved_path}"),
                html.Div("You can close this tab and return to Python."),
            ],
            color="green",
        ), True
