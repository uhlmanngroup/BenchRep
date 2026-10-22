from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.development.base_component import Component
from pydantic import BaseModel

from benchrep.assembly import schemas
from benchrep.discovery.config import (
    config_customization,
    config_registries,
    format_annotation,
    format_constraints,
    format_default,
)
from benchrep.discovery.registry import (
    list_registered_components,
    list_registries,
    registry_documentation,
)


@dataclass(frozen=True, kw_only=True)
class WizardSection:
    """A workflow section and its rendered contents."""

    key: str
    title: str
    description: str
    content: Component
    documentation: str | None = None
    schema: type[BaseModel] | None = None
    default_registry: str | None = None


def section_status(
    errors: dict[str, str] | None, *, mode: str | None = None
) -> Component:
    """Render a section's local validation status."""
    if errors is None:
        symbol, state, label = "○", "pending", "Not yet checked"
    elif errors:
        symbol, state, label = "!", "invalid", "Needs attention"
    elif mode in {"omit", "null"}:
        selected = "Omitted" if mode == "omit" else "Null"
        symbol, state, label = (
            "—",
            "neutral",
            f"{selected} — checked during final validation",
        )
    else:
        symbol, state, label = "✓", "valid", "Local validation passed"

    return html.Span(
        symbol,
        className=f"br-section-status br-section-status-{state}",
        title=label,
        **{"aria-label": label},
    )


def section_heading(section: WizardSection) -> Component:
    """Show the section title and optional schema documentation."""
    children: list[Component] = [html.H2(section.title)]

    if section.documentation:
        children.append(
            dmc.Popover(
                [
                    dmc.PopoverTarget(
                        html.Button(
                            "i",
                            type="button",
                            className="br-help-button",
                            **{"aria-label": f"Documentation for {section.title}"},
                        )
                    ),
                    dmc.PopoverDropdown(
                        dcc.Markdown(
                            section.documentation,
                            className="br-field-docs",
                        )
                    ),
                ],
                position="bottom-start",
                withArrow=True,
                withinPortal=True,
                width=480,
            )
        )

    return html.Div(children, className="br-section-heading")


def build_wizard_layout(
    *,
    title: str,
    subtitle: str,
    sections: Sequence[WizardSection],
    actions: Sequence[Component] = (),
    active_content: Component | None = None,
) -> Component:
    """Build the shared shell; the workflow supplies sections and actions."""
    if not sections:
        raise ValueError("The wizard requires at least one section.")

    keys = [section.key for section in sections]
    if len(keys) != len(set(keys)):
        raise ValueError("Wizard section keys must be unique.")

    tabs = dmc.Tabs(
        id="wizard-sections",
        value=sections[0].key,
        keepMounted=False,
        orientation="vertical",
        className="br-section-navigation",
        children=[
            dmc.TabsList(
                [
                    dmc.TabsTab(
                        html.Span(
                            [
                                html.Span(
                                    section.title,
                                    className="br-section-tab-title",
                                ),
                                html.Span(
                                    section_status(None),
                                    id=f"wizard-section-status-{section.key}",
                                ),
                            ],
                            className="br-section-tab-label",
                        ),
                        value=section.key,
                    )
                    for section in sections
                ],
                className="br-section-tabs",
            ),
            html.Section(
                [
                    html.Div(section_heading(sections[0]), id="wizard-active-heading"),
                    html.P(
                        sections[0].description,
                        id="wizard-active-description",
                        className="br-section-description",
                    ),
                    html.Div(active_content, id="wizard-active-content"),
                ],
                className="br-section br-section-panel",
            ),
        ],
    )

    return dmc.MantineProvider(
        forceColorScheme="light",
        theme=cast(
            dmc.MantineProvider.Theme,
            cast(
                object,
                {
                    "primaryColor": "purple",
                    "primaryShade": 7,
                    "colors": {
                        "purple": [
                            "#faf5ff",
                            "#f3e8ff",
                            "#e9d5ff",
                            "#d8b4fe",
                            "#c084fc",
                            "#a855f7",
                            "#9333ea",
                            "#7e22ce",
                            "#6b21a8",
                            "#581c87",
                        ],
                    },
                    "fontFamily": "system-ui, -apple-system, sans-serif",
                    "defaultRadius": "md",
                },
            ),
        ),
        children=html.Div(
            [
                html.Header(
                    [
                        html.Div("BENCHREP", className="br-brand"),
                        html.H1(title),
                        html.P(subtitle),
                    ],
                    className="br-header",
                ),
                html.Main(
                    html.Div(
                        [
                            html.Div(
                                [
                                    tabs,
                                    html.Div(
                                        id="wizard-feedback",
                                        role="status",
                                        **{"aria-live": "polite"},
                                    ),
                                    html.Details(
                                        [
                                            html.Summary("Preview YAML draft"),
                                            html.P(
                                                "Includes base config and supplied components. Schema defaults may change during resolution. This draft does not guarantee training can run.",
                                                className="br-draft-note",
                                            ),
                                            html.Pre(
                                                id="wizard-yaml-preview",
                                                className="br-yaml-preview",
                                            ),
                                        ],
                                        className="br-draft-preview",
                                    ),
                                    html.Footer(
                                        list(actions),
                                        className="br-actions",
                                    ),
                                ],
                                className="br-editor",
                            ),
                            build_documentation_panel(),
                        ],
                        className="br-workspace",
                    ),
                    className="br-main",
                ),
            ],
            className="br-wizard",
        ),
    )


def config_schema_catalog() -> dict[str, type[BaseModel]]:
    """Discover the configuration classes exported by BenchRep."""
    return {
        name: cast(type[BaseModel], value)
        for name, value in vars(schemas).items()
        if not name.startswith("_")
        and isinstance(value, type)
        and issubclass(value, BaseModel)
        and value is not BaseModel
    }


def build_documentation_panel() -> Component:
    """Build persistent config and registry inspection tools."""
    registry_options = [
        {
            "value": name,
            "label": f"{name} ({path.rsplit('.', 1)[-1]})",
        }
        for name, path in list_registries().items()
    ]

    return html.Details(
        [
            html.Summary("Documentation"),
            dcc.Store(id="wizard-docs-component-source", storage_type="memory"),
            html.Div(
                [
                    dmc.Switch(
                        id="wizard-docs-follow",
                        label="Follow active section",
                        checked=True,
                        size="sm",
                    ),
                    html.H3("Config schema"),
                    dmc.Select(
                        id="wizard-docs-config",
                        data=sorted(config_schema_catalog()),
                        placeholder="Search configuration classes",
                        searchable=True,
                        clearable=True,
                        size="sm",
                    ),
                    html.Div(
                        id="wizard-docs-config-output",
                        className="br-docs-output",
                    ),
                    html.H3("Registered components"),
                    dmc.Select(
                        id="wizard-docs-registry",
                        data=registry_options,
                        placeholder="Choose a registry",
                        searchable=True,
                        clearable=True,
                        size="sm",
                    ),
                    dmc.Select(
                        id="wizard-docs-component",
                        data=[],
                        placeholder="Choose a component",
                        searchable=True,
                        clearable=True,
                        size="sm",
                        mt=8,
                    ),
                    html.Div(
                        id="wizard-docs-registry-output",
                        className="br-docs-output",
                    ),
                ],
                className="br-docs-content",
            ),
        ],
        open=True,
        className="br-documentation",
    )


def register_documentation_callbacks(
    *,
    app: Dash,
    stage: str,
    sections: Sequence[WizardSection],
) -> None:
    """Connect inspection tools to the current section and form values."""
    catalog = config_schema_catalog()
    sections_by_key = {section.key: section for section in sections}

    @app.callback(
        Output("wizard-active-heading", "children"),
        Output("wizard-active-description", "children"),
        Input("wizard-sections", "value"),
    )
    def active_heading(key):
        section = sections_by_key[key]
        return section_heading(section), section.description

    @app.callback(
        Output("wizard-docs-config", "value"),
        Input("wizard-sections", "value"),
        Input("wizard-docs-follow", "checked"),
    )
    def follow_section(section_key, follow):
        if not follow:
            return no_update

        section = sections_by_key.get(section_key)
        schema = section.schema if section else None
        return next(
            (name for name, candidate in catalog.items() if candidate is schema),
            None,
        )

    @app.callback(
        Output("wizard-docs-registry", "value"),
        Input("wizard-docs-config", "value"),
        Input("wizard-docs-follow", "checked"),
        Input("wizard-sections", "value"),
    )
    def follow_config_registry(schema_name, follow, active_key):
        if not follow:
            return no_update

        section = sections_by_key.get(active_key)
        if section and section.default_registry:
            return section.default_registry
        schema = catalog.get(schema_name)
        routes = config_registries(schema) if schema else ()
        return routes[0] if routes else None

    @app.callback(
        Output("wizard-docs-component", "data"),
        Output("wizard-docs-component", "value"),
        Output("wizard-docs-component-source", "data"),
        Input("wizard-docs-registry", "value"),
        Input("wizard-sections", "value"),
        Input("wizard-docs-follow", "checked"),
        Input("wizard-draft", "data"),
        State("wizard-docs-component", "data"),
        State("wizard-docs-component", "value"),
        State("wizard-docs-component-source", "data"),
    )
    def component_options(
        registry_name,
        active_key,
        follow,
        form_values,
        current_options,
        current_component,
        previous_source,
    ):
        names: list[str] = []
        source = None
        selected = None

        if registry_name:
            aliases = list_registered_components(
                registry_name,
                include_aliases=True,
            )
            names = sorted(aliases)
            selected = current_component if current_component in names else None

            section = sections_by_key.get(active_key)
            routes = (
                config_registries(section.schema) if section and section.schema else ()
            )
            if follow and registry_name in routes:
                canonical_by_name = {
                    name: canonical
                    for canonical, alternatives in aliases.items()
                    for name in (canonical, *alternatives)
                }
                values = form_values.get(active_key) or {}
                name = values.get("name") if isinstance(values, dict) else None
                canonical = (
                    canonical_by_name.get(name.lower().strip().replace("-", "_"))
                    if isinstance(name, str)
                    else None
                )
                source = [registry_name, active_key, canonical]
                if source != previous_source:
                    selected = canonical

        return (
            names if names != current_options else no_update,
            selected if selected != current_component else no_update,
            source if source != previous_source else no_update,
        )

    @app.callback(
        Output("wizard-docs-config-output", "children"),
        Input("wizard-docs-config", "value"),
    )
    def show_config(schema_name):
        return render_config_documentation(catalog.get(schema_name))

    @app.callback(
        Output("wizard-docs-registry-output", "children"),
        Input("wizard-docs-registry", "value"),
        Input("wizard-docs-component", "value"),
    )
    def show_registry(registry_name, component_name):
        try:
            return render_registry_documentation(
                registry_name,
                component_name if registry_name else None,
            )
        except (KeyError, ValueError, TypeError) as error:
            return dmc.Alert(
                str(error),
                title="Could not load documentation",
                color="red",
            )


def _documentation_metadata(
    rows: Sequence[tuple[str, Component | str]],
) -> Component:
    """Render consistently aligned documentation labels and values."""
    return html.Dl(
        [
            element
            for label, value in rows
            for element in (html.Dt(label), html.Dd(value))
        ],
        className="br-docs-field-metadata",
    )


def _support_text(value: bool | None) -> str:
    return {
        True: "Supported",
        False: "Not supported",
        None: "Not documented",
    }[value]


def render_registry_documentation(
    registry_name: str | None,
    component_name: str | None,
) -> Component:
    """Render structured registry documentation."""
    if not registry_name:
        return dcc.Markdown(
            "Select a registry to discover its components and requirements.\n\n"
            "Then select a component to inspect its constructor, "
            "documentation, and runtime contract.\n\n"
            "**Suggested starting points:**\n\n"
            "- `encoder`\n"
            "- `decoder`\n"
            "- `optimizer`",
            className="br-inspection-markdown",
        )

    data = registry_documentation(registry_name, component_name)

    def code_list(values) -> Component:
        return html.Ul([html.Li(html.Code(value)) for value in values])

    children: list[Component] = [
        html.H3(data["name"]),
        _documentation_metadata(
            [
                ("Registry object", html.Code(data["object_path"])),
                ("Custom registration", _support_text(data["custom_registration"])),
                ("Registration entry", html.Code(data["registration_entry"])),
                (
                    "Runtime instance override",
                    _support_text(data["runtime_instance_override"]),
                ),
            ]
        ),
    ]

    if data["config_locations"]:
        children.extend(
            [
                html.H4("Config locations"),
                code_list(data["config_locations"]),
            ]
        )

    for key, title in (
        ("contract", "Requirements"),
        ("runtime_override", "Runtime override"),
    ):
        if data[key]:
            children.extend(
                [
                    html.H4(title),
                    dcc.Markdown(data[key]),
                ]
            )

    component = data["selected_component"]

    if component is None:
        children.extend(
            [
                html.H4("Registered components"),
                html.P(
                    "Select a component above to inspect its parameters "
                    "and documentation."
                ),
            ]
        )

        for entry in data["components"]:
            contents: list[Component] = [
                html.H5(html.Code(entry["name"])),
            ]
            if entry["aliases"]:
                contents.append(
                    _documentation_metadata(
                        [
                            ("Aliases", code_list(entry["aliases"])),
                        ]
                    )
                )

            children.append(html.Section(contents, className="br-docs-field"))

        if not data["components"]:
            children.append(html.P("No components are registered."))

    else:
        children.extend(
            [
                html.H4("Selected component"),
                html.H5(html.Code(component["name"])),
                _documentation_metadata(
                    [
                        ("Target", html.Code(component["target"])),
                        (
                            "Aliases",
                            code_list(component["aliases"])
                            if component["aliases"]
                            else "None",
                        ),
                    ]
                ),
                html.H4(component["signature_label"]),
                html.Pre(html.Code(component["signature"] or "Signature unavailable.")),
                html.P(
                    "BenchRep may supply some arguments internally; "
                    "this signature does not necessarily map directly "
                    "to configurable params."
                ),
                html.H4("Description"),
                dcc.Markdown(component["description"] or "No docstring available."),
            ]
        )

        contract = component.get("runtime_contract")
        if contract is not None:
            children.append(html.H4("Composite runtime contract"))

            if contract.get("input_policy"):
                children.append(
                    _documentation_metadata(
                        [
                            ("Input policy", contract["input_policy"]),
                        ]
                    )
                )

            inputs = contract.get("inputs")
            if inputs is not None:
                children.append(
                    html.Div(
                        "Inputs",
                        className="br-docs-field-subheading",
                    )
                )
                children.append(
                    _documentation_metadata(
                        [
                            (
                                port["name"],
                                html.Code(
                                    " | ".join(
                                        port.get("structures", port.get("roles", ()))
                                    )
                                ),
                            )
                            for port in inputs
                        ]
                    )
                    if inputs
                    else html.P("No declared inputs.")
                )

            children.append(
                _documentation_metadata(
                    [
                        ("Result", contract["result_kind"]),
                    ]
                )
            )

            if contract.get("outputs"):
                children.extend(
                    [
                        html.Div(
                            "Outputs",
                            className="br-docs-field-subheading",
                        ),
                        _documentation_metadata(
                            [
                                (
                                    output["name"],
                                    html.Code(" | ".join(output["structures"])),
                                )
                                for output in contract["outputs"]
                            ]
                        ),
                    ]
                )

        if component.get("result_kind") is not None:
            rows = [("Result kind", component["result_kind"])]
            if component.get("vector_axis") is not None:
                rows.append(("Vector axis", component["vector_axis"]))
            children.extend(
                [
                    html.H4("Evaluation output"),
                    _documentation_metadata(rows),
                ]
            )

    return html.Div(
        children,
        className="br-registry-documentation br-inspection-markdown",
    )


def render_config_documentation(
    schema: type[BaseModel] | None,
) -> Component:
    """Render schema documentation without parsing console output."""
    if schema is None:
        return dcc.Markdown(
            "Select a configuration class to inspect its fields.\n\n"
            "**Suggested starting points:**\n\n"
            "- `TrainingConfig`\n"
            "- `PredictionConfig`\n"
            "- `EvaluationConfig`\n\n"
            "Nested configuration types can be selected here too.",
            className="br-inspection-markdown",
        )

    children: list[Component] = [
        html.H3(schema.__name__),
    ]

    description = inspect.getdoc(schema)
    if description:
        children.append(dcc.Markdown(description))

    customization = config_customization(schema)
    if customization:
        children.append(html.H4("Customization"))
        for entry in customization:
            rows = [
                (
                    "Registry",
                    html.Code(entry["registry"]) if entry["registry"] else "None",
                ),
                ("Custom registration", _support_text(entry["custom_registration"])),
                (
                    "Runtime instance override",
                    _support_text(entry["runtime_instance_override"]),
                ),
            ]
            content: list[Component] = [_documentation_metadata(rows)]
            if entry["runtime_override"]:
                content.append(dcc.Markdown(entry["runtime_override"]))
            children.append(html.Section(content, className="br-docs-field"))

    schema_extra = schema.model_config.get("json_schema_extra")
    if isinstance(schema_extra, dict):
        behavior = schema_extra.get("extra_field_behavior")
        if behavior:
            children.extend(
                [
                    html.H4("Extra fields"),
                    dcc.Markdown(str(behavior)),
                ]
            )

    children.append(html.H4("Fields"))

    for name, field in schema.model_fields.items():
        metadata_rows: list[tuple[str, Component | str]] = [
            ("Type", html.Code(format_annotation(field.annotation))),
            ("Required", "Yes" if field.is_required() else "No"),
        ]
        if not field.is_required():
            metadata_rows.append(("Default", html.Code(format_default(field))))
        constraints = format_constraints(field.metadata)
        if constraints:
            metadata_rows.append(("Constraints", ", ".join(constraints)))

        field_children: list[Component] = [
            html.H5(html.Code(name)),
            _documentation_metadata(metadata_rows),
        ]

        if field.description:
            field_children.append(dcc.Markdown(field.description))

        extra = field.json_schema_extra
        if isinstance(extra, dict):
            for key, label in (
                ("omit_behavior", "When omitted"),
                ("null_behavior", "When null"),
            ):
                value = extra.get(key)
                if value is not None:
                    field_children.extend(
                        [
                            html.Div(
                                label,
                                className="br-docs-field-subheading",
                            ),
                            dcc.Markdown(str(value)),
                        ]
                    )

            notes = extra.get("notes")
            if notes:
                if isinstance(notes, str):
                    notes = [notes]

                field_children.extend(
                    [
                        html.Div(
                            "Notes",
                            className="br-docs-field-subheading",
                        ),
                        html.Ul([html.Li(dcc.Markdown(str(note))) for note in notes]),
                    ]
                )

        children.append(
            html.Section(
                field_children,
                className="br-docs-field",
            )
        )

    return html.Div(
        children,
        className="br-config-documentation br-inspection-markdown",
    )
