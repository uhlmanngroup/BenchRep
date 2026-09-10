from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
from collections import Counter
from html import escape
from typing import Any, Final, TYPE_CHECKING, TypeAlias
from shutil import which
import textwrap

import lightning as L
import torch

if TYPE_CHECKING:
    from graphviz import Digraph

    from benchrep.assembly.resolvers.composite_model_resolver import (
        CompositeModelAssemblyStepSpec,
        CompositeModelSpec,
    )


# ---------------------------------------------------------------------------
# Constants for BenchRep owned architecture graph generation of composite models
# ---------------------------------------------------------------------------
GraphNodeDetail: TypeAlias = str | tuple[str, str]

_FONT_NAME: Final[str] = "DejaVu Sans"
_GRAPH_TITLE_FONT_SIZE: Final[str] = "24"
_NODE_TITLE_FONT_SIZE: Final[str] = "16"
_NODE_DETAIL_FONT_SIZE: Final[str] = "14"
_BADGE_FONT_SIZE: Final[str] = "13"
_LEGEND_FONT_SIZE: Final[str] = "11"
_EDGE_FONT_SIZE: Final[str] = "13"

_EXPECTS_FILL_COLOR: Final[str] = "#EAF2FF"
_EXPECTS_BORDER_COLOR: Final[str] = "#3B82F6"

_PRODUCES_FILL_COLOR: Final[str] = "#ECFDF5"
_PRODUCES_BORDER_COLOR: Final[str] = "#10B981"

_METADATA_FILL_COLOR: Final[str] = "#FFF7E6"
_METADATA_BORDER_COLOR: Final[str] = "#F59E0B"

_ASSEMBLY_FILL_COLOR: Final[str] = "#F0E7FF"
_ASSEMBLY_BORDER_COLOR: Final[str] = "#5B21B6"

_LOSS_FILL_COLOR: Final[str] = "#FFF1F2"
_LOSS_BORDER_COLOR: Final[str] = "#F43F5E"

_CONTEXT_FILL_COLOR: Final[str] = "#F5F3FF"
_CONTEXT_BORDER_COLOR: Final[str] = "#8B5CF6"

_MUTED_TEXT_COLOR: Final[str] = "#475569"
_DEFAULT_EDGE_COLOR: Final[str] = "#94A3B8"

_SHARED_BADGE_FILL_COLOR: Final[str] = "#F6C85F"
_SHARED_BADGE_BORDER_COLOR: Final[str] = "#D97706"
_SHARED_BADGE_TEXT_COLOR: Final[str] = "#713F12"

_NODE_LABEL_WIDTH_POINTS: Final[str] = "248"
_NODE_LABEL_HEIGHT_POINTS: Final[str] = "116"
_NODE_TITLE_CELL_WIDTH_POINTS: Final[str] = "198"
_NODE_BADGE_CELL_WIDTH_POINTS: Final[str] = "46"
_NODE_TEXT_WRAP_WIDTH: Final[int] = 28
_NODE_TITLE_WRAP_WIDTH: Final[int] = 28


class ModelGraphDependencyError(RuntimeError):
    """Raised when optional model-graph dependencies are unavailable."""


# ---------------------------------------------------------------------------
# Torchview graph export
# ---------------------------------------------------------------------------
def export_torchview_graph(
    *,
    model: L.LightningModule,
    output_path: Path,
    expand_nested: bool,
    depth: int,
    input_size: Sequence[int] | None = None,
    input_data: Any | None = None,
) -> Path:
    """Export a Torchview graph from either a dummy size or runtime data."""

    if (input_size is None) == (input_data is None):
        raise ValueError(
            "Exactly one of `input_size` or `input_data` must be provided "
            "for Torchview graph export."
        )

    try:
        import torchview
    except ImportError as error:
        raise ModelGraphDependencyError(
            "The `torchview` Python package is unavailable. Install "
            "BenchRep's `model_graph` optional dependencies."
        ) from error

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torchview_input_kwargs: dict[str, Any]

    if input_size is not None:
        torchview_input_kwargs = {
            "input_size": tuple(input_size),
        }
    else:
        torchview_input_kwargs = {
            "input_data": input_data,
        }

    try:
        model_graph = torchview.draw_graph(
            model,
            expand_nested=expand_nested,
            depth=depth,
            **torchview_input_kwargs,
        )

        graph_base_path = output_path.with_suffix("")
        rendered_path = model_graph.visual_graph.render(
            str(graph_base_path),
            format=output_path.suffix.lstrip(".") or "png",
            cleanup=True,
        )

    except Exception as error:
        root_cause = error

        while root_cause.__cause__ is not None:
            root_cause = root_cause.__cause__

        raise RuntimeError(
            "Could not export Torchview graph: "
            f"{type(root_cause).__name__}: {root_cause}"
        ) from error

    return Path(rendered_path).resolve()


def infer_dummy_input_size(
    datamodule: L.LightningDataModule,
) -> tuple[int, ...]:
    """Infer the canonical model input size from one training batch."""

    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    try:
        x = batch["x"]
    except KeyError as error:
        raise KeyError(
            "Could not infer dummy input size because the training batch "
            "does not contain key 'x'. BenchRep datamodules must return "
            "batches with batch['x'] as the model input tensor."
        ) from error

    return (1, *x.shape[1:])


def prepare_composite_torchview_input_data(
    *,
    datamodule: L.LightningDataModule,
    model_spec: CompositeModelSpec,
) -> list[dict[str, torch.Tensor]]:
    """Prepare one real Composite model input from a training batch."""

    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    if not isinstance(batch, Mapping):
        raise TypeError(
            "Could not prepare Composite Torchview input because the "
            "training batch is not a mapping; got "
            f"{type(batch).__name__}."
        )

    expected_input_names = tuple(
        model_spec.declarations.model_input_roles_by_name
    )
    missing_input_names = [
        input_name
        for input_name in expected_input_names
        if input_name not in batch
    ]

    if missing_input_names:
        raise KeyError(
            "Could not prepare Composite Torchview input because the "
            f"training batch is missing declared inputs {missing_input_names}. "
            f"Available batch fields: {tuple(batch)}."
        )

    model_inputs: dict[str, torch.Tensor] = {}

    for input_name in expected_input_names:
        model_input = batch[input_name]

        if not isinstance(model_input, torch.Tensor):
            raise TypeError(
                f"Composite model input {input_name!r} must be a "
                f"torch.Tensor, got {type(model_input).__name__}."
            )

        # Retain the batch dimension while limiting graph construction
        # to one observation.
        model_inputs[input_name] = model_input[:1]

    # Torchview interprets a mapping directly as keyword arguments.
    # Wrapping it makes the mapping the sole positional forward argument:
    # CompositeModel.forward(model_inputs).
    return [model_inputs]


# ---------------------------------------------------------------------------
# BenchRep composite model specification graph rendering and export
# ---------------------------------------------------------------------------
def build_composite_model_spec_graph(
    model_spec: CompositeModelSpec,
    *,
    graph_name: str = "Composite model specification",
) -> Digraph:
    """Build a conceptual architecture graph from a resolved model spec.

    The graph represents BenchRep declarations, assembly steps, component
    sharing, produced outputs, and configured losses. It intentionally omits
    internal PyTorch operations such as activations, reshaping, and individual
    neural-network layers.
    """

    try:
        from graphviz import Digraph
    except ImportError as error:
        raise ModelGraphDependencyError(
            "The `graphviz` Python package is unavailable. Install BenchRep's "
            "`model_graph` optional dependencies."
        ) from error


    graph = Digraph(
        name=graph_name,
        comment="Resolved BenchRep Composite model architecture",
    )

    graph.attr(
        "graph",
        rankdir="LR",
        bgcolor="transparent",
        pad="0.35",
        nodesep="0.38",
        ranksep="0.90",
        splines="spline",
        outputorder="edgesfirst",
        fontname=_FONT_NAME,
        fontcolor="#0F172A",
        label=_build_graph_header_label(graph_name),
        labelloc="t",
        labeljust="l",
    )

    graph.attr(
        "node",
        shape="plain",
        fontname=_FONT_NAME,
        margin="0",
    )

    graph.attr(
        "edge",
        color=_DEFAULT_EDGE_COLOR,
        fontcolor=_MUTED_TEXT_COLOR,
        fontname=_FONT_NAME,
        fontsize=_EDGE_FONT_SIZE,
        penwidth="2.5",
        arrowsize="1.5",
        arrowhead="vee",
    )

    model_input_node_ids: dict[str, str] = {}
    batch_metadata_node_ids: dict[str, str] = {}
    model_output_node_ids: dict[str, str] = {}

    # Create one node for every declaration under `expects`.
    for declaration_index, (
        model_input_name,
        model_input_role,
    ) in enumerate(
        model_spec.declarations.model_input_roles_by_name.items()
    ):
        node_id = f"expects_{declaration_index}"
        model_input_node_ids[model_input_name] = node_id

        _add_graph_node(
            graph,
            node_id=node_id,
            title=model_input_name,
            details=(
                "model input",
                ("role", model_input_role),
            ),
            fill_color=_EXPECTS_FILL_COLOR,
            border_color=_EXPECTS_BORDER_COLOR,
        )

    # Batch metadata is part of the declared Composite interface even though
    # it is not passed into ordinary architecture components.
    for metadata_index, (
        metadata_name,
        metadata_role,
    ) in enumerate(
        model_spec.declarations.batch_metadata_roles_by_name.items()
    ):
        node_id = f"batch_metadata_{metadata_index}"
        batch_metadata_node_ids[metadata_name] = node_id

        _add_graph_node(
            graph,
            node_id=node_id,
            title=metadata_name,
            details=(
                "batch metadata",
                metadata_role,
            ),
            fill_color=_METADATA_FILL_COLOR,
            border_color=_METADATA_BORDER_COLOR,
        )

    # Create one node for every declaration under `produces`.
    for output_index, (
        model_output_name,
        model_output_role,
    ) in enumerate(
        model_spec.declarations.model_output_roles_by_name.items()
    ):
        node_id = f"produces_{output_index}"
        model_output_node_ids[model_output_name] = node_id

        _add_graph_node(
            graph,
            node_id=node_id,
            title=model_output_name,
            details=(
                "model output",
                ("role", model_output_role),
            ),
            fill_color=_PRODUCES_FILL_COLOR,
            border_color=_PRODUCES_BORDER_COLOR,
        )

    assembly_steps = [
        assembly_step
        for dependency_level in sorted(
            model_spec.assembly_steps_by_dependency_level
        )
        for assembly_step in (
            model_spec.assembly_steps_by_dependency_level[
                dependency_level
            ]
        )
    ]

    component_call_counts = Counter(
        assembly_step.component_id
        for assembly_step in assembly_steps
    )

    shared_component_ids = sorted(
        component_id
        for component_id, call_count
        in component_call_counts.items()
        if call_count > 1
    )

    shared_component_badge_by_id = {
        component_id: _get_shared_component_badge(
            shared_component_index
        )
        for shared_component_index, component_id
        in enumerate(shared_component_ids)
    }

    assembly_step_node_ids: dict[str, str] = {}
    assembly_step_node_ids_by_dependency_level: dict[
        int,
        list[str],
    ] = {}

    # Create one node per assembly step. Several step nodes may identify the
    # same component ID because one shared module can be called repeatedly.
    for dependency_level in sorted(
        model_spec.assembly_steps_by_dependency_level
    ):
        assembly_step_node_ids_by_dependency_level[
            dependency_level
        ] = []

        assembly_steps_at_level = (
            model_spec.assembly_steps_by_dependency_level[
                dependency_level
            ]
        )

        for step_index, assembly_step in enumerate(
            assembly_steps_at_level
        ):
            node_id = (
                f"assembly_step_{dependency_level}_{step_index}"
            )
            assembly_step_node_ids[
                assembly_step.step_id
            ] = node_id
            assembly_step_node_ids_by_dependency_level[
                dependency_level
            ].append(node_id)

            component_spec = model_spec.components_by_id[
                assembly_step.component_id
            ]

            component_call_count = component_call_counts[
                assembly_step.component_id
            ]

            if component_call_count > 1:
                component_detail: GraphNodeDetail = (
                    "shared component",
                    (
                        f"{assembly_step.component_id} "
                        f"({component_call_count} calls)"
                    ),
                )
                shared_component_badge = (
                    shared_component_badge_by_id[
                        assembly_step.component_id
                    ]
                )
            else:
                component_detail = (
                    "component",
                    assembly_step.component_id,
                )
                shared_component_badge = None

            _add_graph_node(
                graph,
                node_id=node_id,
                title=assembly_step.step_id,
                details=(
                    component_detail,
                    (
                        component_spec.component_kind,
                        component_spec.registry_entry_name,
                    ),
                    (
                        "dependency level",
                        str(dependency_level),
                    ),
                ),
                fill_color=_ASSEMBLY_FILL_COLOR,
                border_color=_ASSEMBLY_BORDER_COLOR,
                badge=shared_component_badge,
            )

    # Keep assembly steps from the same dependency level in one column.
    for dependency_level, step_node_ids in (
        assembly_step_node_ids_by_dependency_level.items()
    ):
        with graph.subgraph(
            name=f"dependency_level_{dependency_level}"
        ) as level_graph:
            level_graph.attr(rank="same")

            for step_node_id in step_node_ids:
                level_graph.node(step_node_id)

    # Keep declared inputs and batch metadata at the graph's source side.
    with graph.subgraph(name="declaration_sources") as source_graph:
        source_graph.attr(rank="source")

        for node_id in model_input_node_ids.values():
            source_graph.node(node_id)

        for node_id in batch_metadata_node_ids.values():
            source_graph.node(node_id)

    # Connect model inputs and previously produced outputs to component calls.
    for assembly_step in assembly_steps:
        assembly_step_node_id = assembly_step_node_ids[
            assembly_step.step_id
        ]

        for (
            component_forward_parameter_name,
            model_input_name,
        ) in assembly_step.inputs_from_model_inputs.items():
            graph.edge(
                model_input_node_ids[model_input_name],
                assembly_step_node_id,
                label=component_forward_parameter_name,
                color=_EXPECTS_BORDER_COLOR,
            )

        for (
            component_forward_parameter_name,
            model_output_name,
        ) in assembly_step.inputs_from_model_outputs.items():
            graph.edge(
                model_output_node_ids[model_output_name],
                assembly_step_node_id,
                label=component_forward_parameter_name,
                color=_PRODUCES_BORDER_COLOR,
            )

        _connect_assembly_step_results(
            graph,
            assembly_step=assembly_step,
            assembly_step_node_id=assembly_step_node_id,
            model_output_node_ids=model_output_node_ids,
        )

    loss_node_ids: list[str] = []
    context_node_ids: dict[str, str] = {}

    # Losses are separate graph sinks. Their incoming edge labels identify
    # the loss module's forward parameter receiving each tensor or context.
    for loss_index, loss_spec in enumerate(
        model_spec.loss_specs
    ):
        loss_node_id = f"loss_{loss_index}"
        loss_node_ids.append(loss_node_id)

        _add_graph_node(
            graph,
            node_id=loss_node_id,
            title=loss_spec.configured_loss_name,
            details=(
                (
                    "role",
                    loss_spec.loss_role,
                ),
                (
                    "registry",
                    loss_spec.registry_entry_name,
                ),
                (
                    "weight",
                    f"{loss_spec.weight:g}",
                ),
            ),
            fill_color=_LOSS_FILL_COLOR,
            border_color=_LOSS_BORDER_COLOR,
        )

        for (
            loss_forward_parameter_name,
            model_input_name,
        ) in loss_spec.inputs_from_model_inputs.items():
            graph.edge(
                model_input_node_ids[model_input_name],
                loss_node_id,
                label=loss_forward_parameter_name,
                color=_LOSS_BORDER_COLOR,
            )

        for (
            loss_forward_parameter_name,
            model_output_name,
        ) in loss_spec.inputs_from_model_outputs.items():
            graph.edge(
                model_output_node_ids[model_output_name],
                loss_node_id,
                label=loss_forward_parameter_name,
                color=_LOSS_BORDER_COLOR,
            )

        for (
            loss_forward_parameter_name,
            context_source,
        ) in loss_spec.context_inputs.items():
            context_node_id = context_node_ids.get(
                context_source
            )

            if context_node_id is None:
                context_node_id = (
                    f"context_{len(context_node_ids)}"
                )
                context_node_ids[
                    context_source
                ] = context_node_id

                context_description = {
                    "batch": "complete batch mapping",
                    "model_output": "complete model output mapping",
                }.get(
                    context_source,
                    context_source,
                )

                _add_graph_node(
                    graph,
                    node_id=context_node_id,
                    title=context_source,
                    details=(
                        "automatic context",
                        context_description,
                    ),
                    fill_color=_CONTEXT_FILL_COLOR,
                    border_color=_CONTEXT_BORDER_COLOR,
                )

            graph.edge(
                context_node_id,
                loss_node_id,
                label=loss_forward_parameter_name,
                color=_CONTEXT_BORDER_COLOR,
                style="dashed",
            )

    if loss_node_ids:
        with graph.subgraph(name="loss_sinks") as loss_graph:
            loss_graph.attr(rank="sink")

            for loss_node_id in loss_node_ids:
                loss_graph.node(loss_node_id)

    return graph


def export_composite_model_spec_graph(
    model_spec: CompositeModelSpec,
    *,
    output_path: Path | str,
    graph_name: str = "Composite model specification",
) -> Path:
    """Render and export a resolved Composite model specification as SVG."""

    output_path = Path(output_path).expanduser()

    if output_path.suffix.lower() not in {"", ".svg"}:
        raise ValueError(
            "Composite model specification graphs are exported as SVG; "
            f"received output path suffix {output_path.suffix!r}."
        )

    output_path = output_path.with_suffix(".svg").resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph = build_composite_model_spec_graph(
        model_spec,
        graph_name=graph_name,
    )

    graph_base_path = output_path.with_suffix("")

    if which("dot") is None:
        raise ModelGraphDependencyError(
            "The Graphviz `dot` executable is unavailable. Install Graphviz "
            "and ensure `dot` is available on PATH."
        )

    try:
        rendered_path = graph.render(
            str(graph_base_path),
            format="svg",
            cleanup=True,
            quiet=True,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not render the Composite model specification graph: "
            f"{error}"
        ) from error

    return Path(rendered_path).resolve()


def _connect_assembly_step_results(
    graph: Digraph,
    *,
    assembly_step: CompositeModelAssemblyStepSpec,
    assembly_step_node_id: str,
    model_output_node_ids: dict[str, str],
) -> None:
    """Connect one assembly step to its declared model outputs."""

    if isinstance(
        assembly_step.results_to_model_outputs,
        str,
    ):
        graph.edge(
            assembly_step_node_id,
            model_output_node_ids[
                assembly_step.results_to_model_outputs
            ],
            label="result",
            color=_PRODUCES_BORDER_COLOR,
        )
        return

    for (
        runtime_result_name,
        model_output_name,
    ) in assembly_step.results_to_model_outputs.items():
        graph.edge(
            assembly_step_node_id,
            model_output_node_ids[model_output_name],
            label=runtime_result_name,
            color=_PRODUCES_BORDER_COLOR,
        )


def _add_graph_node(
    graph: Digraph,
    *,
    node_id: str,
    title: str,
    details: tuple[GraphNodeDetail, ...],
    fill_color: str,
    border_color: str,
    badge: str | None = None,
) -> None:
    """Add one consistently styled graph node."""

    graph.node(
        node_id,
        label=_build_html_node_label(
            title=title,
            details=details,
            fill_color=fill_color,
            border_color=border_color,
            badge=badge,
        ),
        tooltip=title,
    )


def _build_html_node_label(
    *,
    title: str,
    details: tuple[GraphNodeDetail, ...],
    fill_color: str,
    border_color: str,
    badge: str | None,
) -> str:
    """Build one fixed-size node as a Graphviz HTML table."""

    title_html = _format_wrapped_html_text(
        title,
        width=_NODE_TITLE_WRAP_WIDTH,
        bold=True,
    )

    if badge is None:
        badge_cell = (
            '<TD '
            'ALIGN="RIGHT" '
            'VALIGN="TOP" '
            f'WIDTH="{_NODE_BADGE_CELL_WIDTH_POINTS}" '
            'HEIGHT="30" '
            'CELLPADDING="0">'
            "</TD>"
        )
    else:
        escaped_badge = escape(
            badge,
            quote=True,
        )

        badge_cell = (
            '<TD '
            'ALIGN="RIGHT" '
            'VALIGN="TOP" '
            f'WIDTH="{_NODE_BADGE_CELL_WIDTH_POINTS}" '
            'CELLPADDING="0">'
            "<TABLE "
            'BORDER="2" '
            'CELLBORDER="0" '
            'CELLSPACING="0" '
            'CELLPADDING="0" '
            'STYLE="ROUNDED" '
            f'BGCOLOR="{_SHARED_BADGE_FILL_COLOR}" '
            f'COLOR="{_SHARED_BADGE_BORDER_COLOR}">'
            "<TR>"
            '<TD '
            'ALIGN="CENTER" '
            'VALIGN="MIDDLE" '
            'CELLPADDING="3">'
            f'<FONT POINT-SIZE="{_BADGE_FONT_SIZE}" '
            f'COLOR="{_SHARED_BADGE_TEXT_COLOR}">'
            f"<B>{escaped_badge}</B>"
            "</FONT>"
            "</TD>"
            "</TR>"
            "</TABLE>"
            "</TD>"
        )

    rows = [
        (
            "<TR>"
            '<TD '
            'ALIGN="LEFT" '
            'VALIGN="TOP" '
            f'WIDTH="{_NODE_TITLE_CELL_WIDTH_POINTS}" '
            'HEIGHT="30" '
            'CELLPADDING="5">'
            f'<FONT POINT-SIZE="{_NODE_TITLE_FONT_SIZE}" '
            'COLOR="#0F172A">'
            f"{title_html}"
            "</FONT>"
            "</TD>"
            f"{badge_cell}"
            "</TR>"
        )
    ]

    for detail in details:
        detail_html = _format_detail_html(detail)

        rows.append(
            "<TR>"
            '<TD '
            'ALIGN="LEFT" '
            'VALIGN="TOP" '
            'COLSPAN="2" '
            'CELLPADDING="3">'
            f'<FONT POINT-SIZE="{_NODE_DETAIL_FONT_SIZE}" '
            f'COLOR="{_MUTED_TEXT_COLOR}">'
            f"{detail_html}"
            "</FONT>"
            "</TD>"
            "</TR>"
        )

    return (
        "<<TABLE "
        'BORDER="3" '
        'CELLBORDER="0" '
        'CELLSPACING="0" '
        'CELLPADDING="0" '
        'FIXEDSIZE="TRUE" '
        f'WIDTH="{_NODE_LABEL_WIDTH_POINTS}" '
        f'HEIGHT="{_NODE_LABEL_HEIGHT_POINTS}" '
        f'BGCOLOR="{fill_color}" '
        f'COLOR="{border_color}" '
        'STYLE="ROUNDED">'
        f"{''.join(rows)}"
        "</TABLE>>"
    )


def _wrap_graph_text(
    text: str,
    *,
    width: int,
) -> list[str]:
    """Wrap text at spaces, underscores, or the hard width limit."""

    preserved_hyphen = "\uE000"

    # textwrap supports hyphen breakpoints but not underscore breakpoints.
    # Temporarily treat underscores as hyphens without changing text length.
    breakable_text = (
        str(text)
        .replace("-", preserved_hyphen)
        .replace("_", "-")
    )

    wrapped_lines = textwrap.wrap(
        breakable_text,
        width=width,
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=True,
    )

    return [
        line
        .replace("-", "_")
        .replace(preserved_hyphen, "-")
        for line in wrapped_lines
    ]


def _format_detail_html(
    detail: GraphNodeDetail,
) -> str:
    """Format one body entry with wrapped continuation lines."""

    if detail in {"model input", "model output"}:
        return f"<I>{escape(detail, quote=True)}</I>"

    if isinstance(detail, tuple):
        detail_label, detail_value = detail
        prefix = f"{detail_label}: "

        wrapped_lines = _wrap_graph_text(
            f"{prefix}{detail_value}",
            width=_NODE_TEXT_WRAP_WIDTH,
        )

        if not wrapped_lines:
            wrapped_lines = [prefix]

        first_line = wrapped_lines[0]
        first_line_value = first_line[len(prefix):]

        rendered_lines = [
            (
                f"<B>{escape(detail_label, quote=True)}:</B> "
                f"{escape(first_line_value, quote=True)}"
            )
        ]

        rendered_lines.extend(
            _escape_html_line_with_indentation(line)
            for line in wrapped_lines[1:]
        )

        return '<BR ALIGN="LEFT"/>'.join(rendered_lines)

    return _format_wrapped_html_text(
        detail,
        width=_NODE_TEXT_WRAP_WIDTH,
        bold=False,
    )


def _format_wrapped_html_text(
    text: str,
    *,
    width: int,
    bold: bool,
) -> str:
    """Wrap plain text and preserve continuation indentation."""

    wrapped_lines = _wrap_graph_text(
        str(text),
        width=width,
    )

    if not wrapped_lines:
        wrapped_lines = [""]

    rendered_text = '<BR ALIGN="LEFT"/>'.join(
        _escape_html_line_with_indentation(line)
        for line in wrapped_lines
    )

    if bold:
        return f"<B>{rendered_text}</B>"

    return rendered_text


def _escape_html_line_with_indentation(
    line: str,
) -> str:
    """Escape one line while retaining its leading indentation."""

    stripped_line = line.lstrip(" ")
    indentation = len(line) - len(stripped_line)

    return (
        "&#160;" * indentation
        + escape(stripped_line, quote=True)
    )


def _get_shared_component_badge(
    shared_component_index: int,
) -> str:
    """Return badges as @A through @Z, then @AA, @AB, etc."""

    badge_number = shared_component_index + 1
    badge_characters: list[str] = []

    while badge_number > 0:
        badge_number, remainder = divmod(
            badge_number - 1,
            26,
        )
        badge_characters.append(
            chr(ord("A") + remainder)
        )

    badge_name = "".join(
        reversed(badge_characters)
    )

    return f"@{badge_name}"


def _build_graph_header_label(
    graph_name: str,
) -> str:
    """Build the graph title and shared-instance legend."""

    escaped_graph_name = escape(
        graph_name,
        quote=True,
    )

    return (
        "<<TABLE "
        'BORDER="0" '
        'CELLBORDER="0" '
        'CELLSPACING="6" '
        'CELLPADDING="2">'
        "<TR>"
        '<TD ALIGN="LEFT">'
        f'<FONT POINT-SIZE="{_GRAPH_TITLE_FONT_SIZE}" '
        'COLOR="#0F172A">'
        f"<B>{escaped_graph_name}</B>"
        "</FONT>"
        "</TD>"
        '<TD WIDTH="30"></TD>'
        '<TD ALIGN="CENTER">'
        "<TABLE "
        'BORDER="2" '
        'CELLBORDER="0" '
        'CELLSPACING="0" '
        'CELLPADDING="3" '
        'STYLE="ROUNDED" '
        f'BGCOLOR="{_SHARED_BADGE_FILL_COLOR}" '
        f'COLOR="{_SHARED_BADGE_BORDER_COLOR}">'
        "<TR>"
        '<TD ALIGN="CENTER">'
        f'<FONT POINT-SIZE="{_BADGE_FONT_SIZE}" '
        f'COLOR="{_SHARED_BADGE_TEXT_COLOR}">'
        "<B>@ID</B>"
        "</FONT>"
        "</TD>"
        "</TR>"
        "</TABLE>"
        "</TD>"
        '<TD ALIGN="LEFT">'
        f'<FONT POINT-SIZE="{_LEGEND_FONT_SIZE}" '
        f'COLOR="{_MUTED_TEXT_COLOR}">'
        "Matching badge IDs use the same module instance"
        "</FONT>"
        "</TD>"
        "</TR>"
        "</TABLE>>"
    )