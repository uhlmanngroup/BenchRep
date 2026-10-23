"""Recursive schema editors. Rendering has no workflow-specific validation rules."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, get_args, get_origin

import dash_mantine_components as dmc
from dash import dcc, html
from dash.development.base_component import Component

from benchrep.gui.shared.state import (
    MISSING,
    EditorSpec,
    active_spec,
    base_type,
    empty_value,
    union_variant_details,
)


@dataclass(frozen=True, kw_only=True)
class FieldPresentation:
    """Optional editing guidance. Schema validation remains independent."""

    suggestions: tuple[str, ...] | None = None
    key_suggestions: tuple[str, ...] | None = None
    disabled_reason: str | None = None
    order: tuple[str, ...] = ()
    hidden: bool = False
    excluded_fields: tuple[str, ...] = ()
    reorderable: bool = True
    item_label: str = "Item"
    named_variants: tuple[tuple[str, int], ...] | None = None


def field_id(section: str, node: dict, action: str, *, button: bool = False) -> dict:
    return {
        "type": "br-button" if button else "br-input",
        "section": section,
        "node": node["id"],
        "action": action,
    }


def field_help(spec: EditorSpec, title: str, path: tuple) -> Component:
    extra = spec.info.json_schema_extra
    metadata = extra if isinstance(extra, dict) else {}
    config_path = metadata.get("config_path", ".".join(map(str, path)))
    paragraphs = [f"**Config path:** `{config_path}`"]
    schema_path = metadata.get("schema_path", spec.schema_path)
    if schema_path:
        paragraphs.append(f"**Python schema:** `{schema_path}`")
    if spec.info.description:
        paragraphs.append(spec.info.description)
    for key, label in (
        ("omit_behavior", "When omitted"),
        ("null_behavior", "When null"),
    ):
        if metadata.get(key):
            paragraphs.append(f"**{label}:** {metadata[key]}")
    notes = metadata.get("notes", [])
    paragraphs.extend([notes] if isinstance(notes, str) else map(str, notes))
    return dmc.Popover(
        [
            dmc.PopoverTarget(
                html.Button(
                    "i",
                    type="button",
                    className="br-help-button",
                    **{"aria-label": f"Documentation for {title}"},
                )
            ),
            dmc.PopoverDropdown(
                dcc.Markdown("\n\n".join(paragraphs), className="br-field-docs")
            ),
        ],
        position="bottom-end",
        withArrow=True,
        withinPortal=True,
        width=420,
    )


def nesting_attributes(depth: int) -> dict[str, str]:
    return {
        "data-container-depth": str(depth),
        "data-container-tone": (
            "plain" if depth < 2 else "white" if depth % 2 else "lavender"
        ),
    }


def render_editor(
    spec: EditorSpec,
    node: dict,
    *,
    section: str,
    path: tuple,
    title: str | None = None,
    locked: bool = False,
    locked_paths: frozenset[tuple] = frozenset(),
    options: Callable[[tuple, Any], list[str] | None] | None = None,
    show_mode: bool = True,
    container_depth: int = 0,
    collection_item: bool = False,
    presentation: Callable[[tuple, Any], FieldPresentation] | None = None,
) -> Component:
    is_section = len(path) == 1 and show_mode
    title = (
            title
            or spec.info.title
            or str(path[-1]).replace("_", " ").title()
    )
    locked = locked or path in locked_paths
    identifier = lambda action: field_id(section, node, action)
    button_id = lambda action: field_id(section, node, action, button=True)
    effective = active_spec(spec, node)
    guide = (
        presentation(path, node.get("value")) if presentation else FieldPresentation()
    )
    if guide.hidden and node["mode"] in {"omit", "null"}:
        return html.Div()
    empty = empty_value(effective)
    controls = []
    if show_mode:
        controls.append(
            dmc.Select(
                id=identifier("mode"),
                value=node["mode"],
                data=[
                    {
                        "value": mode,
                        "label": mode.title(),
                        "disabled": mode == "empty" and empty is MISSING,
                    }
                    for mode in ("value", "omit", "null", "empty")
                ],
                allowDeselect=False,
                disabled=locked,
                size="xs",
                className="br-field-mode",
                **{"aria-label": f"{title} mode"},
            )
        )
    controls.append(field_help(spec, title, path))
    header = html.Div(
        [
            html.Label(title, className="br-field-label"),
            html.Div(controls, className="br-field-header-actions"),
        ],
        className="br-field-header",
    )
    body: list[Component] = []
    mode = node["mode"] if show_mode else "value"
    is_config_group = (
        show_mode and not is_section and mode == "value" and effective.kind == "object"
    )
    owns_container = is_section or (is_config_group and not collection_item)
    visible_depth = container_depth + int(owns_container)
    collapsible = (
        show_mode
        and not is_section
        and mode == "value"
        and effective.kind in {"object", "list", "mapping"}
    )
    collapsed = collapsible and node.get("collapsed", False)
    if collapsible:
        header.children.insert(
            0,
            html.Button(
                "▸" if collapsed else "▾",
                id=button_id("collapse"),
                type="button",
                className="br-collapse-toggle",
                **{
                    "aria-label": f"{'Expand' if collapsed else 'Collapse'} {title}",
                    "aria-expanded": not collapsed,
                },
            ),
        )
        header.children.insert(
            2,
            html.Span(
                id={"type": "br-group-error", "section": section, "node": node["id"]},
                className="br-group-error",
                title="This group contains validation errors.",
            ),
        )
    if mode != "value":
        note = {
            "omit": "Omitted; key not included",
            "null": "Explicit null",
            "empty": f"Explicit empty value: {json.dumps(empty) if empty is not MISSING else 'unavailable'}",
        }[mode]
        if mode == "omit" and not locked:
            body.append(
                html.Button(
                    [
                        html.Span(note),
                        html.Span("— Click to enter a value", className="br-omit-hint"),
                    ],
                    id=button_id("activate"),
                    type="button",
                    className="br-field-mode-note br-omit-activate",
                    **{"aria-label": f"Enter a value for {title}"},
                )
            )
        else:
            body.append(html.Div(note, className="br-field-mode-note"))
    elif spec.kind == "union" and guide.named_variants is not None:
        branch_node = node["branches"][str(node["branch"])]
        name_node = branch_node["fields"]["name"]
        current_name = name_node.get("value") if name_node["mode"] == "value" else None
        choices = [
            {"label": name, "value": json.dumps([name, branch])}
            for name, branch in guide.named_variants
        ]
        selected = next(
            (choice["value"] for choice in choices if choice["label"] == current_name),
            None,
        )

        body.append(
            dmc.Select(
                id=identifier("union_name"),
                label="Name",
                value=selected,
                data=choices,
                searchable=True,
                allowDeselect=False,
                disabled=locked,
                placeholder="Select a registered name",
            )
        )

        if selected is not None:
            body.append(
                render_editor(
                    replace(
                        effective,
                        fields={
                            key: child
                            for key, child in effective.fields.items()
                            if key != "name"
                        },
                    ),
                    branch_node,
                    section=section,
                    path=path,
                    title=title,
                    locked=locked,
                    locked_paths=locked_paths,
                    options=options,
                    presentation=presentation,
                    show_mode=False,
                    container_depth=visible_depth,
                    collection_item=collection_item,
                )
            )
    elif spec.kind == "union":
        branch_node = node["branches"][str(node["branch"])]
        _, fixed = union_variant_details(spec, node["branch"])
        selection = str(node["branch"])
        visible_spec = effective

        if fixed is not None:
            name, tag = fixed
            tag_node = branch_node["fields"][name]

            # Preserve omitted/null/invalid input until the user selects a tag.
            if tag_node["mode"] != "value" or tag_node.get("value") != tag:
                selection = None

            # The selector owns this field; don't render a duplicate input.
            # Only the rendering spec changes, not the schema or stored state.
            visible_spec = replace(
                effective,
                fields={
                    key: child for key, child in effective.fields.items() if key != name
                },
            )

        body.append(
            dmc.Select(
                id=identifier("branch"),
                **{"aria-label": f"{title} type"},
                placeholder="Select a type",
                value=selection,
                data=[
                    {
                        "value": str(i),
                        "label": union_variant_details(spec, i)[0],
                    }
                    for i in range(len(spec.variants))
                ],
                allowDeselect=False,
                disabled=locked,
            )
        )
        singleton_choice = (
                effective.kind == "choice"
                and len(effective.choices) == 1
                and type(branch_node.get("value")) is type(effective.choices[0])
                and branch_node.get("value") == effective.choices[0]
        )

        if not singleton_choice:
            body.append(
                render_editor(
                    visible_spec,
                    branch_node,
                    section=section,
                    path=path,
                    title=title,
                    locked=locked,
                    locked_paths=locked_paths,
                    options=options,
                    presentation=presentation,
                    show_mode=False,
                    container_depth=visible_depth,
                    collection_item=collection_item,
                )
            )
            body = [
                html.Div(
                    body,
                    className="br-union-row",
                )
            ]

    elif spec.kind == "object":
        body.append(
            html.Div(
                [
                    render_editor(
                        child,
                        node["fields"][key],
                        section=section,
                        path=(*path, key),
                        container_depth=visible_depth,
                        locked=locked,
                        locked_paths=locked_paths,
                        options=options,
                        presentation=presentation,
                    )
                    for key in dict.fromkeys((*guide.order, *spec.fields))
                    if key in spec.fields and key not in guide.excluded_fields
                    for child in (spec.fields[key],)
                ],
                className="br-object-fields",
            )
        )
        if spec.extra:
            body.append(
                dmc.JsonInput(
                    id=identifier("extra"),
                    label="Additional fields (JSON object)",
                    formatOnBlur=True,
                    validationError="Enter valid JSON.",
                    value=node["extra_text"],
                    autosize=True,
                    minRows=3,
                    maxRows=15,
                    disabled=locked,
                    styles={"input": {"fontFamily": "monospace"}},
                )
            )
    elif spec.kind in ("list", "mapping"):
        value_index = 0
        for index, entry in enumerate(node["entries"]):
            child = entry["node"]
            actions = []
            if spec.kind == "mapping":
                actions.append(
                    dmc.Autocomplete(
                        id=field_id(section, child, "key"),
                        value=entry["key"],
                        label="Key",
                        data=list(
                            guide.key_suggestions
                            if guide.key_suggestions is not None
                            else (
                                get_args(base_type(spec.key_type))
                                if get_origin(base_type(spec.key_type)) is Literal
                                else ()
                            )
                        ),
                        debounce=True,
                        openOnFocus=True,
                        disabled=locked,
                    )
                )
            if spec.kind == "list" and guide.reorderable:
                for action, label, disabled in [
                    ("up", "Move up", index == 0),
                    ("down", "Move down", index == len(node["entries"]) - 1),
                ]:
                    actions.append(
                        html.Button(
                            label,
                            id=field_id(section, child, action, button=True),
                            disabled=locked or disabled,
                            type="button",
                        )
                    )
            actions.append(
                html.Button(
                    "Remove",
                    id=field_id(section, child, "remove", button=True),
                    disabled=locked,
                    type="button",
                )
            )
            child_path = (*path, value_index if spec.kind == "list" else entry["key"])
            if child["mode"] != "omit":
                value_index += 1
            body.append(
                html.Div(
                    [
                        html.Div(actions, className="br-entry-actions"),
                        render_editor(
                            spec.item,
                            child,
                            section=section,
                            path=child_path,
                            container_depth=visible_depth + 1,
                            collection_item=True,
                            title=f"{guide.item_label} {index + 1}"
                            if spec.kind == "list"
                            else "Value",
                            locked=locked,
                            locked_paths=locked_paths,
                            options=options,
                            presentation=presentation,
                        ),
                    ],
                    className="br-collection-entry",
                    **nesting_attributes(visible_depth + 1),
                )
            )
        body.append(
            html.Button(
                f"Add {guide.item_label.lower()}"
                if spec.kind == "list"
                else "Add entry",
                id=button_id("add"),
                disabled=locked,
                type="button",
            )
        )
    else:
        value = node["value"]
        choices = options(path, value) if options else None
        common = {
            "id": identifier("value"),
            "disabled": locked or bool(guide.disabled_reason),
            "aria-label": title,
        }
        if guide.disabled_reason:
            body.append(html.Div(guide.disabled_reason, className="br-field-mode-note"))
        if choices is not None:
            body.append(
                dmc.Select(
                    **common,
                    value=value,
                    data=choices,
                    searchable=True,
                    allowDeselect=False,
                )
            )
        elif spec.kind == "choice":
            body.append(
                dmc.Select(
                    **common,
                    value=json.dumps(value) if value is not None else None,
                    data=[
                        {
                            "value": json.dumps(v),
                            "label": str(v).lower() if isinstance(v, bool) else str(v),
                        }
                        for v in spec.choices
                    ],
                    allowDeselect=False,
                )
            )
        elif spec.kind == "number":
            body.append(
                dmc.NumberInput(
                    **common,
                    # Mantine hides its step controls for None; an empty editor
                    # uses an empty string, while Null mode owns semantic None.
                    value="" if value is None else value,
                    allowDecimal=not (
                        base_type(spec.annotation) is int
                        or any(
                            base_type(arg) is int
                            for arg in get_args(base_type(spec.annotation))
                        )
                    ),
                    hideControls=False,
                )
            )
        elif spec.kind == "text":
            body.append(
                dmc.Autocomplete(
                    **common,
                    value=value,
                    data=list(guide.suggestions),
                    openOnFocus=True,
                )
                if guide.suggestions is not None
                else dmc.TextInput(**common, value=value)
            )
        else:
            body.append(
                dmc.JsonInput(
                    **common,
                    value=value,
                    formatOnBlur=True,
                    validationError="Enter valid JSON.",
                    autosize=True,
                    minRows=4,
                    maxRows=20,
                    styles={"input": {"fontFamily": "monospace", "fontSize": "13px"}},
                )
            )
    reset = None
    if show_mode:
        reset = html.Button(
            "Reset section to default" if is_section else "Reset to default",
            id=button_id("reset"),
            type="button",
            disabled=locked or spec.info.is_required(),
            className="br-field-reset",
            title="This field has no schema default."
            if spec.info.is_required()
            else "Explicitly insert the schema default.",
        )
        if not is_section:
            body.append(reset)
    return html.Div(
        [
            header if show_mode else None,
            reset if is_section else None,
            html.Div(body, className="br-field-body", hidden=collapsed),
            html.Div(
                id={"type": "br-error", "section": section, "node": node["id"]},
                className="br-field-mode-error",
            ),
        ],
        className="br-field br-recursive-field"
        + (" br-section-editor" if is_section else "")
        + (
            " br-config-group"
            if show_mode
            and not is_section
            and mode == "value"
            and effective.kind == "object"
            else ""
        ),
        **{
            **(nesting_attributes(visible_depth) if owns_container else {}),
            "data-config-path": json.dumps(list(path)),
            "data-editor-node": node["id"],
            "data-editor-mode": mode,
        },
    )
