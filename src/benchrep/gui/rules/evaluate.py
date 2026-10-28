"""Evaluation authoring, advisory source metadata, and backend-backed final checks."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import dash_mantine_components as dmc
from dash import html
from dash.development.base_component import Component
from pydantic import TypeAdapter, ValidationError

from benchrep.assembly.config import load_yaml, normalize_config_components
from benchrep.assembly.registries.builtins import register_builtins
from benchrep.assembly.registries.core import (
    EVAL_EMBEDDING_METRICS,
    EVAL_EXTERNAL_CLUSTERING_METRICS,
    EVAL_INTERNAL_CLUSTERING_METRICS,
    EVAL_PREDICTABILITY_PROBES,
    EVAL_RECONSTRUCTION_METRICS,
)
from benchrep.assembly.resolvers.evaluation_config_resolver import (
    _load_prediction_manifest,
    _resolve_prediction_manifest_path,
    resolve_anndata_input,
    resolve_evaluation_config,
    resolve_reconstructions,
    resolve_run_identity,
)
from benchrep.assembly.schemas.evaluation_config_schema import (
    EvaluationConfig,
    EvaluationRunConfig,
    EvaluationSourceConfig,
)
from benchrep.assembly.schemas.parsing import parse_evaluation_config
from benchrep.evaluation.reconstructions.data import _load_pt
from benchrep.gui.shared.fields import FieldPresentation, field_id
from benchrep.gui.shared.forms import SectionSpec
from benchrep.gui.shared.review import ReviewSession
from benchrep.gui.shared.state import MISSING, WizardDraft
from benchrep.utils.workflow_overrides import EvaluationOverrides
from benchrep.utils.workflow_utils import EvaluationConfigInput, route_config_input


def prepare_evaluation_draft(
    *, config: EvaluationConfigInput, overrides: EvaluationOverrides
) -> WizardDraft:
    """Preserve supplied values and key presence without resolving an unfinished run."""
    register_builtins()
    path, typed = route_config_input(config, EvaluationConfig)
    data = (
        load_yaml(path)
        if path is not None
        else typed.model_dump(mode="json", exclude_unset=True)
        if typed is not None
        else {}
    )
    if not isinstance(data, dict):
        raise TypeError("Evaluation configuration must be a mapping.")
    components = normalize_config_components(
        schema=EvaluationConfig, config_components=overrides.config_components
    )
    data = deepcopy(data)
    data.update(TypeAdapter(dict[str, Any]).dump_python(components, mode="json"))
    if config is None:
        data.setdefault("stage", "evaluation")
    return WizardDraft(initial_data=data, supplied_sections=frozenset(components))


def _value(data, *path):
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def inspect_anndata_metadata(path: Path) -> dict:
    """Read HDF5 names only; never materialize X, layers, or representation arrays."""
    import h5py

    with h5py.File(path, "r") as handle:
        obs = handle.get("obs")
        if obs is None:
            columns = ()
        elif isinstance(obs, h5py.Group):
            index = _text(obs.attrs.get("_index", "_index"))
            columns = tuple(
                _text(k)
                for k in obs.attrs.get("column-order", obs.keys())
                if _text(k) != index
            )
        else:
            # Older AnnData files may store observations as a compound dataset.
            columns = tuple(k for k in (obs.dtype.names or ()) if k != "_index")
        obsm = handle.get("obsm")
        representations = tuple(obsm.keys()) if isinstance(obsm, h5py.Group) else ()
    return {"columns": columns, "representations": representations}


def inspect_reconstruction_metadata(spec) -> dict:
    """Read observations and export metadata using the backend's artifact reader."""
    obs = _load_pt(spec.observations_path)
    if isinstance(obs, Mapping):
        columns = tuple(k for k in obs if isinstance(k, str))
    elif hasattr(obs, "columns"):
        columns = tuple(k for k in obs.columns if isinstance(k, str))
    else:
        columns = ()
    metadata = _load_pt(spec.metadata_path) if spec.metadata_path is not None else None
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("Reconstruction export metadata must be a mapping.")
    # Retain only small descriptive values, not the observations themselves.
    return {
        "columns": columns,
        "pair": dict(metadata.get("pair", {})) if metadata else {},
        "n_examples_exported": metadata.get("n_examples_exported")
        if metadata
        else None,
    }


@dataclass
class EvaluationSourceContext:
    """Advisory snapshot; source loading never rewrites the authored configuration."""

    overrides: EvaluationOverrides
    loaded_identity: str | None = None
    manifest: dict = field(default_factory=dict)
    manifest_path: Path | None = None
    pair_ids: tuple[str, ...] = ()
    anndata: dict = field(default_factory=dict)
    reconstructions: dict = field(default_factory=dict)
    anndata_spec: Any = None
    reconstruction_spec: Any = None
    inherited: dict[tuple, Any] = field(default_factory=dict)
    statuses: dict[str, tuple[str, str]] = field(default_factory=dict)

    def source_values(self, draft: dict) -> dict:
        source = draft.get("source")
        values = deepcopy(source) if isinstance(source, dict) else {}
        if self.overrides.prediction_manifest_path is not None:
            values["prediction_manifest_path"] = str(
                Path(self.overrides.prediction_manifest_path).expanduser().resolve()
            )
        return values

    def identity(self, draft: dict) -> str:
        return json.dumps(self.source_values(draft), sort_keys=True, default=str)

    def current(self, draft: dict) -> bool:
        return self.loaded_identity == self.identity(draft)

    def has_source(self, draft: dict) -> bool:
        values = self.source_values(draft)
        return any(
            values.get(k)
            for k in (
                "prediction_manifest_path",
                "anndata_path",
                "reconstructions_path",
            )
        )

    def validate_source(self, data: Any) -> dict[tuple, str]:
        # Reuse the top-level source requirement with unrelated sections at defaults.
        source = {} if data is MISSING else data
        try:
            parse_evaluation_config(
                {"source": source},
                prediction_manifest_path_overridden=self.overrides.prediction_manifest_path
                is not None,
            )
        except ValidationError as error:
            return {
                tuple(item["loc"][1:])
                if item["loc"][:1] == ("source",)
                else tuple(item["loc"]): item["msg"]
                for item in error.errors()
            }
        return {}

    def load(self, draft: dict) -> None:
        self.loaded_identity = self.identity(draft)
        self.manifest, self.anndata, self.reconstructions = {}, {}, {}
        self.inherited, self.statuses = {}, {}
        self.manifest_path = self.anndata_spec = self.reconstruction_spec = None
        self.pair_ids = ()
        values = self.source_values(draft)

        def source_value(name):
            info = EvaluationSourceConfig.model_fields[name]
            return TypeAdapter(info.rebuild_annotation()).validate_python(
                values.get(name)
            )

        try:
            source = EvaluationSourceConfig(
                prediction_manifest_path=source_value("prediction_manifest_path")
            )
            config = EvaluationConfig.model_construct(source=source)
            _, path = _resolve_prediction_manifest_path(
                evaluation_config=config, prediction_manifest_path_override=None
            )
            if path is not None:
                with warnings.catch_warnings(record=True) as notices:
                    warnings.simplefilter("always")
                    self.manifest = _load_prediction_manifest(path)
                self.manifest_path = path
                pairs = _value(self.manifest, "exports", "reconstructions", "pairs")
                if isinstance(pairs, Mapping):
                    self.pair_ids = tuple(k for k in pairs if isinstance(k, str))
                detail = "Prediction manifest loaded."
                if notices:
                    detail += " " + " ".join(str(w.message) for w in notices)
                self.statuses["Manifest"] = ("warning" if notices else "loaded", detail)
                identity = resolve_run_identity(
                    EvaluationRunConfig(), self.manifest, path.parent
                )
                if _value(self.manifest, "run", "output_dir") is not None:
                    self.inherited[("run", "output_root")] = str(identity.output_root)
            else:
                self.statuses["Manifest"] = (
                    "unused",
                    "No prediction manifest supplied; direct artifacts can be used.",
                )
        except Exception as error:  # noqa: BLE001 -- report each optional metadata read independently
            self.statuses["Manifest"] = ("error", str(error))

        base = self.manifest_path.parent if self.manifest_path else None
        try:
            self.anndata_spec = resolve_anndata_input(
                anndata_path=source_value("anndata_path"),
                prediction_manifest=self.manifest or None,
                manifest_base_dir=base,
            )
            if self.anndata_spec is None:
                self.statuses["AnnData metadata"] = (
                    "unused",
                    "No AnnData artifact resolved.",
                )
            else:
                if self.anndata_spec.source == "prediction_manifest":
                    self.inherited[("source", "anndata_path")] = str(
                        self.anndata_spec.path
                    )
                self.anndata = inspect_anndata_metadata(self.anndata_spec.path)
                self.statuses["AnnData metadata"] = (
                    "loaded",
                    (
                        f"{len(self.anndata['columns'])} observation columns and "
                        f"{len(self.anndata['representations'])} stored representations found."
                    ),
                )
        except Exception as error:  # noqa: BLE001 -- report each optional metadata read independently
            self.statuses["AnnData metadata"] = ("error", str(error))

        try:
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                self.reconstruction_spec = resolve_reconstructions(
                    reconstructions_path=source_value("reconstructions_path"),
                    reconstruction_pair_id=source_value("reconstruction_pair_id"),
                    prediction_manifest=self.manifest or None,
                    manifest_base_dir=base,
                    n_examples=None,
                )
            if self.reconstruction_spec is None:
                message = "No reconstruction bundle resolved."
                if notices:
                    message = " ".join(str(w.message) for w in notices)
                self.statuses["Reconstruction metadata"] = (
                    "warning" if notices else "unused",
                    message,
                )
            else:
                spec = self.reconstruction_spec
                if spec.source == "prediction_manifest":
                    self.inherited[("source", "reconstructions_path")] = str(
                        spec.bundle_dir
                    )
                    if spec.n_examples is not None:
                        self.inherited[("reconstruction", "n_examples")] = (
                            spec.n_examples
                        )
                self.reconstructions = inspect_reconstruction_metadata(spec)
                message = f"{len(self.reconstructions['columns'])} observation fields found. Image tensors were not loaded."
                if spec.pair_id:
                    message = f"Pair: {spec.pair_id}. " + message
                self.statuses["Reconstruction metadata"] = ("loaded", message)
        except Exception as error:  # noqa: BLE001 -- report each optional metadata read independently
            self.statuses["Reconstruction metadata"] = ("error", str(error))

    def action(self, action: str, path: tuple, draft: dict):
        if action == "load_source":
            self.load(draft)
        elif action == "copy_inherited" and self.current(draft):
            return deepcopy(self.inherited.get(path, MISSING))
        return MISSING

    def panel(self, draft: dict, node: dict):
        current = self.current(draft)
        title = (
            "Source metadata inspected"
            if current
            else "Source settings changed"
            if self.loaded_identity is not None
            else "Source not loaded"
        )
        message = (
            "Suggestions are advisory. Full artifact loading and data checks happen when evaluation runs."
            if current
            else "Load the source to inspect available artifacts and refresh suggestions."
        )
        rows = []
        if current:
            symbols = {"loaded": "✓", "error": "!", "warning": "!", "unused": "—"}
            for label, (status, detail) in self.statuses.items():
                rows.append(
                    html.Div(
                        [
                            html.Strong(f"{symbols[status]} {label}"),
                            html.Div(detail, className="br-source-status-message"),
                        ],
                        className="br-evaluation-source-row",
                        **{"data-status": status},
                    )
                )
        status_content: list[Component] = [
            html.Strong(title, className="br-source-status-title"),
            html.P(message, className="br-source-status-message"),
        ]
        override_path = self.overrides.prediction_manifest_path
        if override_path is not None:
            path = str(Path(override_path).expanduser().resolve())
            status_content.append(
                html.Div(
                    [
                        html.Span("Supplied in Python · "),
                        dmc.Popover(
                            [
                                dmc.PopoverTarget(
                                    html.Button(
                                        "Manifest path ⓘ",
                                        type="button",
                                        title=path,
                                        className="br-source-path-button",
                                    )
                                ),
                                dmc.PopoverDropdown(
                                    html.Code(path, className="br-source-path-value")
                                ),
                            ],
                            position="bottom-start",
                            withArrow=True,
                            withinPortal=True,
                            styles={
                                "dropdown": {
                                    "maxWidth": "min(480px, calc(100vw - 32px))"
                                }
                            },
                        ),
                    ],
                    className="br-source-origin",
                )
            )
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            status_content,
                            className="br-source-status-copy",
                            role="status",
                        ),
                        html.Div(
                            dmc.Tooltip(
                                dmc.Button(
                                    "Reload source" if current else "Load source",
                                    id=field_id(
                                        "source", node, "load_source", button=True
                                    ),
                                    color="purple",
                                ),
                                label="Read the optional prediction manifest and effective artifact metadata again. Explicit edits are preserved; previous checks are invalidated.",
                                multiline=True,
                                w=320,
                                withArrow=True,
                            ),
                            className="br-source-status-action",
                        ),
                    ],
                    className="br-source-status",
                ),
                *rows,
            ],
            className="br-source-context",
        )

    def presentation(self, path: tuple, value: Any, draft: dict) -> FieldPresentation:
        guide = FieldPresentation()
        current = self.current(draft)
        obs = self.anndata.get("columns", ()) if current else ()
        reps = self.anndata.get("representations", ()) if current else ()
        reconstruction_obs = self.reconstructions.get("columns", ()) if current else ()
        if (
            path == ("source", "prediction_manifest_path")
            and self.overrides.prediction_manifest_path is not None
        ):
            guide = replace(guide, note="Supplied in Python.")
        elif path == ("source", "reconstruction_pair_id"):
            # Pair names belong to the manifest, not to the currently selected pair.
            same_manifest = self.source_values(draft).get(
                "prediction_manifest_path"
            ) == (
                json.loads(self.loaded_identity).get("prediction_manifest_path")
                if self.loaded_identity
                else MISSING
            )
            guide = replace(
                guide,
                suggestions=self.pair_ids if same_manifest else (),
                note="Select a manifest pair when multiple complete bundles are available. A direct reconstruction path takes precedence.",
            )
        elif path == ("metrics", "clustering", "external", "label_key"):
            guide = replace(guide, suggestions=obs)
        elif path == ("metrics", "predictability", "targets"):
            guide = replace(
                guide, key_suggestions=obs, item_label="Target", reorderable=False
            )
        elif path[:3] == ("metrics", "predictability", "targets") and path[-2:] == (
            "cv",
            "group_key",
        ):
            guide = replace(guide, suggestions=obs)
        elif path[-1:] == ("use_rep",) and path[0] in {"reductions", "clustering"}:
            guide = replace(
                guide,
                suggestions=tuple(dict.fromkeys(("X", *reps))),
                note="Suggestions include stored representations. You may also enter a representation created by an earlier evaluation step.",
            )
        elif path[:3] == ("plots", "params", "color_by"):
            if len(path) == 3:
                guide = replace(guide, item_label="Coloring field")
            elif len(path) == 4 or (len(path) == 5 and path[-1] == "key"):
                guide = replace(guide, suggestions=obs)
        elif path == ("plots", "params", "reconstruction_grid", "stratify_by"):
            guide = replace(guide, suggestions=reconstruction_obs)
        if path == ("plots", "params", "formats"):
            guide = replace(guide, item_label="Format", reorderable=False)
        if path == ("reconstruction", "error_maps", "kinds"):
            guide = replace(guide, item_label="Error map", reorderable=False)
        for prefix, registry in _METRIC_REGISTRIES.items():
            if path == (*prefix, "selected") or (
                path[: len(prefix) + 1] == (*prefix, "selected")
                and len(path) == len(prefix) + 2
            ):
                guide = replace(
                    guide,
                    suggestions=tuple(registry.canonical_keys()),
                    item_label="Metric",
                    reorderable=False,
                )
            elif path == (*prefix, "params"):
                guide = replace(
                    guide,
                    key_suggestions=tuple(registry.canonical_keys()),
                    item_label="Metric parameters",
                )
        if path[:3] == ("metrics", "predictability", "targets") and len(path) >= 5:
            if path[4] == "selected":
                guide = replace(
                    guide,
                    suggestions=tuple(EVAL_PREDICTABILITY_PROBES.canonical_keys()),
                    item_label="Probe",
                    reorderable=False,
                )
            elif len(path) == 5 and path[4] == "params":
                guide = replace(
                    guide,
                    note="Only selected probes use these parameters. List-valued candidates require tuning to be enabled.",
                )
        if current and path in self.inherited and _value(draft, *path) is None:
            inherited = self.inherited[path]
            guide = replace(
                guide,
                note="Inherited from the prediction manifest; preview or copy the value below.",
                inherited_value=inherited,
            )
        return guide


_METRIC_REGISTRIES = {
    ("metrics", "clustering", "internal"): EVAL_INTERNAL_CLUSTERING_METRICS,
    ("metrics", "clustering", "external"): EVAL_EXTERNAL_CLUSTERING_METRICS,
    ("metrics", "embedding"): EVAL_EMBEDDING_METRICS,
    ("metrics", "reconstruction"): EVAL_RECONSTRUCTION_METRICS,
}


def evaluation_sections(*, context: EvaluationSourceContext) -> list[SectionSpec]:
    descriptions = {
        "source": (
            "Source",
            "Use a prediction manifest, direct artifacts, or both. Direct artifact paths take precedence. Load source for optional metadata suggestions.",
        ),
        "run": (
            "Run",
            "Choose the evaluation output location and an optional run name.",
        ),
        "reductions": (
            "Reductions",
            "Configure PCA, UMAP, and t-SNE. Automatic enablement depends on the step and available inputs.",
        ),
        "clustering": (
            "Clustering",
            "Configure clustering methods. Explicitly enable the methods you want to run.",
        ),
        "metrics": (
            "Metrics",
            "Choose clustering, embedding, reconstruction, and predictability measurements. Suggestions do not validate artifact contents.",
        ),
        "reconstruction": (
            "Reconstructions",
            "Configure reconstruction TIFF exports and error maps.",
        ),
        "plots": (
            "Plots",
            "Configure figures, coloring fields, output formats, and reconstruction grids.",
        ),
    }
    sections = []
    for key, (title, description) in descriptions.items():
        info = EvaluationConfig.model_fields[key]
        sections.append(
            SectionSpec(
                key=key,
                title=title,
                description=description,
                annotation=info.rebuild_annotation(),
                field_info=info,
                presentation=context.presentation,
                default_registry={
                    "reductions": "reduction",
                    "clustering": "clustering_method",
                    "metrics": "embedding_metric",
                    "reconstruction": "reconstruction_metric",
                }.get(key),
                validate_context=context.validate_source if key == "source" else None,
                locked_paths=frozenset({("source", "prediction_manifest_path")})
                if key == "source"
                and context.overrides.prediction_manifest_path is not None
                else frozenset(),
                refresh_on_change=lambda path: path[0] == "source",
                context_panel=context.panel if key == "source" else None,
                context_action=context.action,
            )
        )
    return sections


def evaluation_review_session(*, context: EvaluationSourceContext) -> ReviewSession:
    def parse(raw):
        return parse_evaluation_config(
            raw,
            prediction_manifest_path_overridden=context.overrides.prediction_manifest_path
            is not None,
        )

    def resolve(config):
        override = context.overrides.prediction_manifest_path
        return resolve_evaluation_config(
            config,
            prediction_manifest_path_override=(
                Path(override).expanduser().resolve() if override is not None else None
            ),
        )

    def dependencies(spec):
        # Backend preconditions include its optional-package checks; no artifacts are loaded.
        from benchrep.runtime.evaluate_run_validation import (
            _validate_enabled_step_preconditions,
        )

        _validate_enabled_step_preconditions(run_spec=spec)
        return (
            "passed",
            "Backend dependency and data-independent preconditions passed. Full artifact and custom-component checks happen during execution.",
        )

    def external_identity(draft):
        source = EvaluationSourceConfig.model_validate(context.source_values(draft))
        paths = []
        if source.prediction_manifest_path is not None:
            path = source.prediction_manifest_path
            paths.append((str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
            manifest = _load_prediction_manifest(path)
            base = path.parent
        else:
            manifest, base = None, None
        adata = resolve_anndata_input(
            anndata_path=source.anndata_path,
            prediction_manifest=manifest,
            manifest_base_dir=base,
        )
        recon = resolve_reconstructions(
            reconstructions_path=source.reconstructions_path,
            reconstruction_pair_id=source.reconstruction_pair_id,
            prediction_manifest=manifest,
            manifest_base_dir=base,
            n_examples=None,
        )
        artifacts = ([adata.path] if adata else []) + (
            [recon.input_path, recon.reconstruction_path, recon.observations_path]
            if recon
            else []
        )
        if recon and recon.metadata_path:
            artifacts.append(recon.metadata_path)
        for path in artifacts:
            stat = path.stat()
            paths.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return json.dumps(paths, sort_keys=True)

    def summary(spec):
        return TypeAdapter(dict[str, Any]).dump_python(
            {
                "inputs": asdict(spec.input_spec),
                "run": asdict(spec.run_identity),
                "inherited_fields": sorted(spec.inherited_config_fields),
                "steps": asdict(spec.step_spec),
                "note": "Automatic settings that depend on loaded data are finalized during execution; artifact values have not been validated by the wizard.",
            },
            mode="json",
        )

    return ReviewSession(
        parse=parse,
        resolve=resolve,
        dependencies=lambda config: (
            "skipped",
            "Requires successful workflow resolution.",
        ),
        resolved_dependencies=dependencies,
        graph=None,
        resolved_config=lambda spec: spec.evaluation_config,
        execution_overrides=replace(context.overrides, config_components=None),
        summary=summary,
        external_identity=external_identity,
    )
