"""Section declarations and one shared lifecycle for all workflow editors."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from inspect import cleandoc
from typing import Any
from uuid import uuid4

import dash_mantine_components as dmc
import yaml
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update, set_props
from dash.exceptions import PreventUpdate
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from benchrep.gui.shared.fields import (
    FieldPresentation,
    FieldPresentation,
    register_field_interactions,
    render_editor,
)
from benchrep.gui.shared.layout import (
    WizardSection,
    build_wizard_layout,
    register_documentation_callbacks,
    section_status,
)
from benchrep.gui.shared.review import build_review_panel, register_review_callbacks
from benchrep.gui.shared.state import (
    MISSING,
    EditorSpec,
    WizardDraft,
    collect_node,
    edit_entry,
    edit_node,
    editor_spec,
    find_node,
    new_node,
    union_variant_details,
)
from benchrep.gui.shared.validation import validate_value


@dataclass(frozen=True, kw_only=True)
class SectionSpec:
    key: str
    title: str
    annotation: Any
    description: str = ""
    field_info: FieldInfo | None = None
    default_registry: str | None = None
    documentation_schema: type[BaseModel] | None = None
    options: Callable[[tuple, Any, Any], list[str] | None] | None = None
    presentation: Callable[[tuple, Any, dict], FieldPresentation] | None = None
    validate_local: Callable[[Any], dict[tuple, str]] | None = None
    validate_context: Callable[[Any], dict[tuple, str]] | None = None
    refresh_on_change: Callable[[tuple], bool] | None = None
    locked_paths: frozenset[tuple] = frozenset()
    # Adapters are exceptions: ordinary sections need neither.
    to_editor: Callable[[Any], Any] | None = None
    from_editor: Callable[[Any], tuple[Any, dict[tuple, str]]] | None = None
    error_path: Callable[[tuple], tuple] | None = None
    incompatible_fields: Callable[[Any], dict[tuple, str]] | None = None
    rejection_reason: str | None = None
    source_key: str | None = None
    context_panel: Callable[[dict, dict], Any] | None = None
    context_action: Callable[[str, tuple, dict], Any] | None = None


def include_rejected_sections(
    sections: list[SectionSpec],
    initial: WizardDraft,
    schema: type[BaseModel],
    incompatible: dict[str, str] | None = None,
) -> list[SectionSpec]:
    """Expose unknown and explicitly incompatible inputs without rewriting them."""
    result = list(sections)
    used = {section.key for section in sections} | {"__review__"}
    incompatible = incompatible or {}
    for source_key, value in initial.initial_data.items():
        reason = None
        if source_key not in schema.model_fields:
            reason = f"Unknown section: {source_key!r} is not defined by {schema.__name__}."
        elif source_key in incompatible and value is not None:
            reason = incompatible[source_key]
        elif source_key == "stage" and value != schema.model_fields["stage"].default:
            expected = schema.model_fields["stage"].default
            reason = f"This wizard configures stage {expected!r}, not {value!r}. Remove this field to use the workflow default."
        if reason is None:
            continue
        key = f"__rejected_{len(result)}"
        while key in used or key in initial.initial_data:
            key += "_"
        used.add(key)
        result.append(SectionSpec(
            key=key,
            source_key=source_key,
            title=(source_key.replace("_", " ").title()
                   if source_key in schema.model_fields else str(source_key)),
            description=reason,
            annotation=Any,
            rejection_reason=reason,
        ))
    return result


def _node_paths(
    spec: EditorSpec, node: dict, path: tuple = (), *, include_hidden: bool = False
):
    yield node["id"], path
    if node["mode"] != "value" and not include_hidden:
        return
    if spec.kind == "object":
        for key, child in spec.fields.items():
            yield from _node_paths(
                child, node["fields"][key], (*path, key), include_hidden=include_hidden
            )
    elif spec.kind in ("list", "mapping"):
        position = 0
        for entry in node["entries"]:
            key = position if spec.kind == "list" else entry["key"]
            yield from _node_paths(
                spec.item, entry["node"], (*path, key), include_hidden=include_hidden
            )
            if entry["node"]["mode"] != "omit":
                position += 1
    elif spec.kind == "union":
        yield from _node_paths(
            spec.variants[node["branch"]],
            node["branches"][str(node["branch"])],
            path,
            include_hidden=include_hidden,
        )


def _editor_error_path(spec: EditorSpec, node: dict, path: tuple) -> tuple:
    """Remove union variant labels from schema error locations."""
    if not path:
        return path
    if spec.kind == "union":
        label, _ = union_variant_details(spec, node["branch"])
        if path[0] == label:
            path = path[1:]
        return _editor_error_path(
            spec.variants[node["branch"]],
            node["branches"][str(node["branch"])],
            path,
        )
    key, *rest = path
    if spec.kind == "object" and key in spec.fields:
        return (
            key,
            *_editor_error_path(spec.fields[key], node["fields"][key], tuple(rest)),
        )
    if spec.kind in {"list", "mapping"}:
        position = 0
        for entry in node["entries"]:
            if entry["node"]["mode"] == "omit":
                continue
            entry_key = position if spec.kind == "list" else entry["key"]
            if entry_key == key:
                return (key, *_editor_error_path(spec.item, entry["node"], tuple(rest)))
            position += 1
    return path


def evaluate_section(
    section: SectionSpec, spec: EditorSpec, node: dict, *, initial: WizardDraft
):
    raw, errors = collect_node(spec, node)
    if section.rejection_reason:
        if node.get("removed"):
            return MISSING, {}
        return raw, {(): section.rejection_reason}
    supplied = section.key in initial.supplied_sections

    if supplied:
        raw = deepcopy(initial.initial_data[section.key])

    context_errors = (
        section.validate_context(raw) if section.validate_context else {}
    )

    if supplied:
        return raw, {
            **context_errors,
            **(section.incompatible_fields(raw) if section.incompatible_fields else {}),
        }

    if raw is MISSING:
        return raw, {
            **({(): "Field required"} if spec.info.is_required() else {}),
            **context_errors,
        }

    errors = {
        **{
            _editor_error_path(spec, node, path): message
            for path, message in validate_value(section.annotation, raw).items()
        },
        **errors,
        **context_errors,
    }

    if section.validate_local:
        errors.update(section.validate_local(raw))

    if section.from_editor:
        raw, adapter_errors = section.from_editor(raw)
        errors.update(adapter_errors)

    if section.incompatible_fields:
        errors.update(section.incompatible_fields(raw))

    if node.get("auto_root") and raw == {}:
        raw = MISSING

    return raw, errors


def build_workflow_app(
    *,
    app: Dash,
    stage: str,
    sections: list[SectionSpec],
    initial: WizardDraft,
    title: str,
    subtitle: str,
    review=None,
    export_directory: str = "wizard_configs",
    export_filename: str = "training.yaml",
    config_schema: type[BaseModel] | None = None,
    incompatible_sections: dict[str, str] | None = None,
) -> Dash:
    """Mount one section; retain all draft state outside the editor subtree."""
    if config_schema is not None:
        sections = include_rejected_sections(
            sections, initial, config_schema, incompatible_sections,
        )
    specs = {
        section.key: editor_spec(section.annotation, section.field_info)
        for section in sections
    }
    by_key = {section.key: section for section in sections}
    if not sections or len(by_key) != len(sections):
        raise ValueError("Provide at least one section with unique keys.")
    state = {}
    for section in sections:
        raw = initial.initial_data.get(section.source_key if section.source_key is not None else section.key, MISSING)
        if section.to_editor and raw is not MISSING:
            raw = section.to_editor(raw)
        node = new_node(specs[section.key], raw)
        state[section.key] = node

    def section_view(key, current, draft):
        if key == "__review__" or current[key].get("removed"):
            return html.Div()
        section = by_key[key]
        if section.rejection_reason:
            supplied = section.source_key in initial.supplied_sections
            return dmc.Alert(
                [
                    html.P(section.rejection_reason),
                    html.Pre(
                        yaml.safe_dump({section.source_key: draft.get(section.source_key)},
                                       sort_keys=False, allow_unicode=True),
                        className="br-yaml-preview",
                    ),
                    html.P("Supplied in Python. Edit that component and restart the wizard.")
                    if supplied else None,
                    dmc.Button(
                        "Remove section",
                        id={"type": "br-button", "section": key,
                            "node": current[key]["id"], "action": "remove_section"},
                        n_clicks=0, color="red", variant="light", disabled=supplied,
                    ),
                ],
                title="Section unavailable", color="red",
            )
        content = []
        if key in initial.supplied_sections:
            content.append(
                dmc.Alert(
                    "This section was supplied as a config-component override in Python. "
                    "Its values are included in the generated configuration. "
                    "Edit the Python component and restart the wizard to change them.",
                    title="Provided in Python",
                    color="purple",
                    className="br-component-notice",
                )
            )
        if section.context_panel:
            content.append(section.context_panel(draft, current[key]))
        section_data, _ = collect_node(specs[key], current[key])
        incompatible = (
            section.incompatible_fields(section_data)
            if section.incompatible_fields else {}
        )

        def presentation(path, value):
            guide = (section.presentation(path, value, draft)
                     if section.presentation else FieldPresentation())
            reason = incompatible.get(path[1:])
            return replace(guide, incompatibility_reason=reason) if reason else guide

        content.append(
            render_editor(
                specs[key],
                current[key],
                section=key,
                path=(key,),
                title=section.title,
                presentation=presentation,
                locked=key in initial.supplied_sections,
                locked_paths=section.locked_paths,
                options=(
                    (lambda path, value: section.options(path, value, section_data))
                    if section.options
                    else None
                ),
            )
        )
        return html.Div(content)

    def result(current):
        draft = deepcopy(initial.initial_data)
        errors = {}
        inline = {}
        for section in sections:
            key = section.key
            raw, issues = evaluate_section(
                section, specs[key], current[key], initial=initial
            )
            source_key = by_key[key].source_key if by_key[key].source_key is not None else key
            if raw is MISSING:
                draft.pop(source_key, None)
            else:
                draft[source_key] = raw
            errors[key] = [
                {"path": list(path), "message": message}
                for path, message in issues.items()
            ]
            for identity, path in _node_paths(specs[key], current[key]):
                inline[identity] = issues.get(path, "")
                inline[identity + ":group"] = any(
                    p[: len(path)] == path for p in issues
                )
        return draft, errors, inline

    initial_draft, initial_errors, initial_inline = result(state)
    layout_sections = []
    for section in sections:
        schema = section.documentation_schema
        if (
            schema is None
            and isinstance(section.annotation, type)
            and issubclass(section.annotation, BaseModel)
        ):
            schema = section.annotation
        doc = schema.__dict__.get("__doc__") if schema else None
        layout_sections.append(
            WizardSection(
                key=section.key,
                title=section.title,
                description=section.description,
                content=html.Div(),
                schema=schema,
                default_registry=section.default_registry,
                invalid=bool(initial_errors.get(section.key)),
                documentation=(
                    f"**Python schema:** `{schema.__name__}`\n\n{cleandoc(doc)}"
                    if doc
                    else None
                ),
            )
        )
    if review is not None:
        layout_sections.append(
            WizardSection(
                key="__review__",
                title="Review & finish",
                description="Check the configuration, review the results and save your YAML.",
                content=html.Div(),
            )
        )
    # The shell contains one live panel, not copies of all forms hidden in tabs.
    app.layout = html.Div(
        [
            dcc.Store(id="wizard-review-receipt"),
            dcc.Store(id="wizard-finished", data=False),
            dcc.Store(id="wizard-editor-state", data=state),
            dcc.Store(id="wizard-focus-request"),
            dcc.Store(id="wizard-focus-result"),
            dcc.Store(id="wizard-field-interactions-ready"),
            dcc.Store(id="wizard-draft", data=initial_draft),
            dcc.Store(id="wizard-section-errors", data=initial_errors),
            dcc.Store(id="wizard-inline-errors", data=initial_inline),
            build_wizard_layout(
                title=title,
                subtitle=subtitle,
                sections=layout_sections,
                active_content=section_view(sections[0].key, state, initial_draft),
                review_panel=(
                    build_review_panel(
                        export_directory, export_filename, show_graph=review.graph is not None
                    )
                    if review is not None
                    else None
                ),
            ),
        ]
    )

    input_pattern = {"type": "br-input", "section": ALL, "node": ALL, "action": ALL}
    button_pattern = {"type": "br-button", "section": ALL, "node": ALL, "action": ALL}

    @app.callback(
        Output("wizard-editor-state", "data"),
        Output("wizard-active-content", "children"),
        Output("wizard-draft", "data"),
        Output("wizard-section-errors", "data"),
        Output("wizard-inline-errors", "data"),
        Output("wizard-focus-request", "data"),
        Input("wizard-sections", "value"),
        Input(input_pattern, "value"),
        Input(button_pattern, "n_clicks"),
        State(input_pattern, "id"),
        State(button_pattern, "id"),
        State("wizard-editor-state", "data"),
        State("wizard-draft", "data"),
        State("wizard-section-errors", "data"),
        State("wizard-inline-errors", "data"),
        prevent_initial_call=True,
    )
    def update(
        active,
        values,
        clicks,
        identifiers,
        button_ids,
        current,
        previous_draft,
        previous_errors,
        previous_inline,
    ):
        triggered = list(ctx.triggered_prop_ids.values())
        # Blurring an edited mapping key and clicking a control can arrive in
        # the same Dash event batch. Capture input values before structural edits.
        trigger = next(
            (
                item
                for item in triggered
                if isinstance(item, dict) and item.get("type") == "br-button"
            ),
            "wizard-sections" if "wizard-sections" in triggered else ctx.triggered_id,
        )
        if not trigger:
            raise PreventUpdate
        focus_request = no_update
        current = deepcopy(current)
        structural = "wizard-sections" in triggered
        edited = set()
        # Capture the complete mounted input snapshot, including changes made just before a tab switch.
        input_updates = sorted(
            zip(identifiers, values, strict=True),
            key=lambda item: item[0]["action"] in {"branch", "union_name"},
        )
        for identifier, value in input_updates:
            key, identity, action = (
                identifier["section"],
                identifier["node"],
                identifier["action"],
            )
            if (key not in specs or key in initial.supplied_sections
                    or by_key[key].rejection_reason):
                continue
            found = find_node(specs[key], current[key], identity)
            if not found:
                continue
            node_spec, node = found
            path = next(
                (p for i, p in _node_paths(specs[key], current[key]) if i == identity),
                (),
            )
            if any((key, *path)[: len(p)] == p for p in by_key[key].locked_paths):
                continue
            if (
                action in ("mode", "branch", "union_name")
                and identifier not in triggered
            ):
                continue
            if action == "value" and node_spec.kind == "choice" and value is not None:
                value = json.loads(value)
            value_changed = (action == "value" and value != node.get("value")) or (
                action == "key" and path and path[-1] != value
            )
            if action == "key":
                structural |= edit_entry(
                    specs[key], current[key], identity, action, value
                )
            else:
                structural |= edit_node(
                    specs[key], current[key], identity, action, value
                )
                if action == "mode" and identity == current[key]["id"]:
                    current[key].pop("auto_root", None)
            if (
                value_changed
                and identifier in triggered
                and by_key[key].refresh_on_change
            ):
                structural |= by_key[key].refresh_on_change((key, *path))
            edited.add(key)
        # Capture collapse counts too, including a click immediately before leaving a section.
        # Absolute counts avoid losing rapid toggles when requests share an older state.
        for identifier, count in zip(button_ids, clicks, strict=True):
            key = identifier["section"]
            if identifier["action"] != "collapse" or not count or key not in specs:
                continue
            found = find_node(specs[key], current[key], identifier["node"])
            if found:
                node = found[1]
                previous_count = node.get("collapse_clicks", 0)
                if count > previous_count:
                    if (count - previous_count) % 2:
                        node["collapsed"] = not node.get("collapsed", False)
                    node["collapse_clicks"] = count
        action_error = None
        context_action = None
        if isinstance(trigger, dict) and trigger["type"] == "br-button":
            count = next(
                (v for i, v in zip(button_ids, clicks, strict=True) if i == trigger),
                None,
            )
            if not count:
                raise PreventUpdate
            key = trigger["section"]
            path = next(
                (
                    p
                    for i, p in _node_paths(specs[key], current[key])
                    if i == trigger["node"]
                ),
                (),
            )
            locked = any((key, *path)[: len(p)] == p for p in by_key[key].locked_paths)
            if (
                trigger["action"] != "collapse"
                and (
                    key not in initial.supplied_sections
                    or trigger["action"] == "load_source"
                )
                and (not locked or trigger["action"] == "load_source")
            ):
                try:
                    action = trigger["action"]
                    if action == "remove_section":
                        section = by_key[key]
                        if (not section.rejection_reason
                                or section.source_key in initial.supplied_sections):
                            raise PreventUpdate
                        current[key]["removed"] = True
                        structural = True
                        if active == key:
                            active = next(s.key for s in sections if not s.rejection_reason)
                            set_props("wizard-sections", {"value": active})
                    elif action == "remove_incompatible":
                        section = by_key[key]
                        raw, _ = collect_node(specs[key], current[key])
                        invalid = section.incompatible_fields(raw) if section.incompatible_fields else {}
                        if path not in invalid:
                            raise PreventUpdate
                        structural |= edit_node(
                            specs[key], current[key], trigger["node"], "mode", "omit"
                        )
                    elif (
                        action in {"load_source", "copy_inherited"}
                        and by_key[key].context_action
                    ):
                        context_action = (key, action, path, trigger["node"])
                    elif action == "activate":
                        found = find_node(specs[key], current[key], trigger["node"])
                        if found and found[1]["mode"] == "omit":
                            structural |= edit_node(
                                specs[key],
                                current[key],
                                trigger["node"],
                                "mode",
                                "value",
                            )
                            found[1].pop("auto_root", None)
                            focus_request = {
                                "node": trigger["node"],
                                "nonce": uuid4().hex,
                            }
                    elif action in ("remove", "up", "down"):
                        structural |= edit_entry(
                            specs[key], current[key], trigger["node"], action, None
                        )
                    else:
                        # Reset editable values without discarding fixed launcher fields.
                        fixed = {}
                        if action == "reset":
                            for identity, field_path in _node_paths(
                                specs[key], current[key], include_hidden=True
                            ):
                                if (key, *field_path) in by_key[key].locked_paths:
                                    fixed[field_path] = deepcopy(
                                        find_node(specs[key], current[key], identity)[1]
                                    )
                        structural |= edit_node(
                            specs[key], current[key], trigger["node"], action, None
                        )
                        if fixed:
                            for identity, field_path in _node_paths(
                                specs[key], current[key], include_hidden=True
                            ):
                                if field_path in fixed:
                                    target = find_node(
                                        specs[key], current[key], identity
                                    )[1]
                                    target.clear()
                                    target.update(fixed[field_path])
                    edited.add(key)
                    if action in {"remove_section", "remove_incompatible"}:
                        if review is not None:
                            with review.lock:
                                review.fingerprint = None
                        set_props("wizard-review-receipt", {"data": None})
                        set_props("wizard-check-report", {"children": []})
                except ValueError as error:
                    action_error = (key, str(error))
        # Only edited sections are validated. Unvisited sections retain their initial status.
        draft, errors, inline = (
            deepcopy(previous_draft),
            deepcopy(previous_errors),
            deepcopy(previous_inline),
        )
        for key in edited:
            raw, issues = evaluate_section(
                by_key[key], specs[key], current[key], initial=initial
            )
            source_key = by_key[key].source_key if by_key[key].source_key is not None else key
            if raw is MISSING:
                draft.pop(source_key, None)
            else:
                draft[source_key] = raw
            errors[key] = [{"path": list(p), "message": m} for p, m in issues.items()]
            for identity, path in _node_paths(specs[key], current[key]):
                inline[identity] = issues.get(path, "")
                inline[identity + ":group"] = any(
                    p[: len(path)] == path for p in issues
                )
        if context_action:
            key, action, path, identity = context_action
            replacement = by_key[key].context_action(action, (key, *path), draft)
            if replacement is not MISSING:
                found = find_node(specs[key], current[key], identity)
                if found:
                    node_spec, node = found
                    node.clear()
                    node.update(new_node(node_spec, replacement))
                    raw, issues = evaluate_section(
                        by_key[key], specs[key], current[key], initial=initial
                    )
                    if raw is MISSING:
                        draft.pop(key, None)
                    else:
                        draft[key] = raw
                    errors[key] = [
                        {"path": list(p), "message": m} for p, m in issues.items()
                    ]
                    for node_id, node_path in _node_paths(specs[key], current[key]):
                        inline[node_id] = issues.get(node_path, "")
                        inline[node_id + ":group"] = any(
                            p[: len(node_path)] == node_path for p in issues
                        )
            structural = True
            if action == "load_source" and review is not None:
                with review.lock:
                    review.fingerprint = None
                set_props("wizard-review-receipt", {"data": None})
                set_props("wizard-check-report", {"children": []})
        if action_error:
            key, message = action_error
            errors[key].append({"path": [], "message": message})
        return (
            current,
            section_view(active, current, draft) if structural else no_update,
            draft if draft != previous_draft else no_update,
            errors if errors != previous_errors else no_update,
            inline if inline != previous_inline else no_update,
            focus_request,
        )

    @app.callback(
        Output("wizard-feedback", "children"),
        *[Output(f"wizard-section-status-{s.key}", "children") for s in sections],
        Input("wizard-section-errors", "data"),
        Input("wizard-sections", "value"),
        Input("wizard-draft", "data"),
    )
    def feedback(errors, active, draft):
        if active == "__review__":
            return (
                None,
                *[
                    section_status(
                        {"issues": "present"} if errors.get(s.key) else {},
                        mode=(
                            "omit"
                            if s.key not in draft
                            else "null" if draft[s.key] is None else "value"
                        ),
                    )
                    for s in sections
                ],
            )
        problems = errors.get(active, [])

        def section_mode(key):
            if key not in draft:
                return "omit"
            return "null" if draft[key] is None else "value"

        mode = section_mode(active)
        neutral = not problems and mode in {"omit", "null"}
        selected = "Omitted" if mode == "omit" else "Null"

        def describe(error):
            path = tuple(error["path"])
            if by_key[active].error_path:
                path = by_key[active].error_path(path)
            source_key = by_key[active].source_key if by_key[active].source_key is not None else active
            return f"{'.'.join(map(str, (source_key, *path)))}: {error['message']}"

        status = dmc.Alert(
            (
                html.Ul([html.Li(describe(e)) for e in problems])
                if problems
                else (
                    "Whether this section may be omitted or null is checked during final validation."
                    if neutral
                    else "This section passes local validation. Complete workflow validation is still required."
                )
            ),
            title=(
                "Check this section"
                if problems
                else (selected if neutral else "Local validation passed")
            ),
            color="red" if problems else ("gray" if neutral else "green"),
        )
        return (
            status,
            *[
                section_status(
                    {"issues": "present"} if errors.get(s.key) else {},
                    mode=section_mode(s.key),
                )
                for s in sections
            ],
        )

    rejected_keys = [section.key for section in sections if section.rejection_reason]
    if rejected_keys:
        @app.callback(
            [Output(f"wizard-section-tab-{key}", "style") for key in rejected_keys],
            Input("wizard-editor-state", "data"),
        )
        def rejected_navigation(current):
            return [
                {"display": "none"} if current[key].get("removed") else {}
                for key in rejected_keys
            ]

    @app.callback(
        Output("wizard-yaml-preview", "children"), Input("wizard-draft", "data")
    )
    def preview(draft):
        return yaml.safe_dump(draft, sort_keys=False, allow_unicode=True)

    app.clientside_callback(
        """function(errors, ids, previous) {
            return ids.map((id, index) => {
                const value = (errors || {})[id.node + ':group'] ? '!' : '';
                return value === (previous[index] || '') ? window.dash_clientside.no_update : value;
            });
        }""",
        Output({"type": "br-group-error", "section": ALL, "node": ALL}, "children"),
        Input("wizard-inline-errors", "data"),
        Input({"type": "br-group-error", "section": ALL, "node": ALL}, "id"),
        State({"type": "br-group-error", "section": ALL, "node": ALL}, "children"),
    )
    app.clientside_callback(
        """function(errors, ids, previous) {
            return ids.map((id, index) => {
                const value = (errors || {})[id.node] || '';
                return value === (previous[index] || '') ? window.dash_clientside.no_update : value;
            });
        }""",
        Output({"type": "br-error", "section": ALL, "node": ALL}, "children"),
        Input("wizard-inline-errors", "data"),
        Input({"type": "br-error", "section": ALL, "node": ALL}, "id"),
        State({"type": "br-error", "section": ALL, "node": ALL}, "children"),
    )
    app.clientside_callback(
        """function(request) {
            if (!request) return window.dash_clientside.no_update;
            let tries = 0;
            function focusEditor() {
                const element = document.querySelector(
                    '[data-editor-node="' + request.node + '"][data-editor-mode="value"]'
                );
                if (element) {
                    const body = element.querySelector(':scope > .br-field-body');
                    const target = body && body.querySelector(
                        'input:not([type="hidden"]):not(:disabled), textarea:not(:disabled), button:not(:disabled)'
                    );
                    if (target) { target.focus(); return; }
                }
                if (++tries < 30) window.requestAnimationFrame(focusEditor);
            }
            window.requestAnimationFrame(focusEditor);
            return request.nonce;
        }""",
        Output("wizard-focus-result", "data"),
        Input("wizard-focus-request", "data"),
        prevent_initial_call=True,
    )
    if review is not None:
        register_review_callbacks(app, review)
        app.clientside_callback(
            "function(active) { return active === '__review__' ? {} : {display: 'none'}; }",
            Output("wizard-review-panel", "style"),
            Input("wizard-sections", "value"),
        )
    register_field_interactions(app)
    register_documentation_callbacks(app=app, stage=stage, sections=layout_sections)
    return app
