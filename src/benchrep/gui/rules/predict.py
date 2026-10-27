"""Prediction sections, source context, inheritance, and final checks."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import dash_mantine_components as dmc
import lightning as L
import yaml
from dash import html
from dash.development.base_component import Component
from pydantic import TypeAdapter

from benchrep.architecture.composite_model_roles import TENSOR_STRUCTURE_BY_ROLE
from benchrep.assembly.config import load_yaml, normalize_config_components
from benchrep.assembly.registries.builtins import register_builtins
from benchrep.assembly.resolvers.prediction_config_resolver import (
    _load_training_manifest,
    _validate_prediction_model_family,
    _validate_prediction_model_source,
    _resolve_prediction_transform_pipelines,
    resolve_prediction_config,
)
from benchrep.assembly.resolvers.utils import (
    get_component_override_name,
    resolve_component_source,
)
from benchrep.assembly.schemas.parsing import (
    parse_prediction_config,
    parse_training_config,
)
from benchrep.assembly.schemas.prediction_config_schema import (
    PredictionConfig,
    PredictionSourceConfig,
    PredictionDataConfig,
    PredictionInferenceConfig,
    PredictionTransformPipelineConfig,
    PredictionExportConfig,
)
from benchrep.assembly.schemas.training_config_schema import (
    TrainingConfig,
    TrainingDataModuleConfig,
    DatasetConfig,
)
from benchrep.assembly.schemas.runtime_override_config_schema import (
    RuntimeOverridesConfig,
)
from benchrep.gui.rules.train import (
    dataset_presentation,
    validate_dataset_name,
    transform_name_options,
    validate_transform_names,
    training_spec_graph,
)
from benchrep.gui.shared.fields import FieldPresentation, field_id
from benchrep.gui.shared.forms import SectionSpec
from benchrep.gui.shared.review import ReviewSession
from benchrep.gui.shared.state import MISSING, WizardDraft
from benchrep.interfaces.model_families import (
    COMPOSITE_FAMILY,
    VAE_FAMILY,
    ModelFamilySpec,
)
from benchrep.utils.workflow_overrides import PredictionOverrides
from benchrep.utils.workflow_utils import PredictionConfigInput, route_config_input


def _at(data: Any, path: tuple, default=None):
    for part in path:
        if isinstance(data, dict):
            data = data.get(part, MISSING)
        elif isinstance(data, list) and isinstance(part, int) and 0 <= part < len(data):
            data = data[part]
        else:
            return default
        if data is MISSING:
            return default
    return data


def prepare_prediction_draft(
    *,
    model_family: ModelFamilySpec,
    config: PredictionConfigInput,
    overrides: PredictionOverrides,
) -> WizardDraft:
    """Preserve authored presence; complete checks happen in the wizard."""
    register_builtins()
    overrides.validate_for(model_family)
    path, typed = route_config_input(config, PredictionConfig)
    data = (
        load_yaml(path)
        if path is not None
        else (
            typed.model_dump(mode="json", exclude_unset=True)
            if typed is not None
            else {}
        )
    )
    if not isinstance(data, dict):
        raise TypeError("Prediction configuration must be a mapping.")
    data = deepcopy(data)
    components = normalize_config_components(
        schema=PredictionConfig, config_components=overrides.config_components
    )
    data.update(TypeAdapter(dict[str, Any]).dump_python(components, mode="json"))
    if config is None:
        data.setdefault("stage", "prediction")
    omitted = set()
    if overrides.datamodule is not None:
        omitted.update({"dataset", "transform_pipelines"})
        if isinstance(data.get("data"), dict):
            for name in ("batch_size", "num_workers"):
                data["data"].pop(name, None)
    for name in omitted:
        data.pop(name, None)
    return WizardDraft(
        initial_data=data, supplied_sections=frozenset(components) - omitted
    )


@dataclass
class PredictionSourceContext:
    """Read-only source snapshot; inherited values never alter the draft implicitly."""

    model_family: ModelFamilySpec
    overrides: PredictionOverrides
    loaded_path: Path | None = None
    attempted_path: str | None = None
    manifest: dict = field(default_factory=dict)
    training: TrainingConfig | None = None
    inherited: dict[tuple, Any] = field(default_factory=dict)
    error: str | None = None

    def path(self, draft: dict) -> Path | None:
        value = self.overrides.training_manifest_path
        if value is None:
            value = _at(draft, ("source", "training_manifest_path"))
        if not isinstance(value, (str, Path)) or not str(value).strip():
            return None
        return Path(value).expanduser().resolve()

    def current(self, draft: dict) -> bool:
        return self.training is not None and self.loaded_path == self.path(draft)

    def load(self, draft: dict) -> None:
        self.training, self.loaded_path, self.error = None, None, None
        self.inherited, self.manifest = {}, {}
        path = self.path(draft)
        self.attempted_path = str(path) if path else None
        try:
            if path is None:
                raise ValueError("Enter a training manifest path first.")
            if path.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError("The training manifest must be a YAML file.")
            manifest = _load_training_manifest(path)
            construction = manifest.get("construction", {})
            model, dm = construction.get("model"), construction.get("datamodule")
            if not isinstance(model, dict) or not isinstance(dm, dict):
                raise ValueError(
                    "The manifest must record model and datamodule construction."
                )
            external_model = model.get("source") != "config"
            external_dm = dm.get("source") != "config"
            training = parse_training_config(
                manifest.get("appendix", {}).get("resolved_config"),
                model_overridden=external_model,
                datamodule_overridden=external_dm,
            )
            _validate_prediction_model_source(
                training_model_external=external_model,
                model_is_external=self.overrides.model is not None,
            )
            _validate_prediction_model_family(
                training_model_construction=model,
                training_config=training,
                model_family=self.model_family,
                model_is_external=self.overrides.model is not None,
            )
            inherited = {
                ("inference", "seed"): training.reproducibility.seed,
                ("inference", "seed_workers"): training.reproducibility.seed_workers,
                ("inference", "deterministic"): training.trainer.deterministic,
                (
                    "inference",
                    "float32_matmul_precision",
                ): training.reproducibility.float32_matmul_precision,
            }
            if self.overrides.datamodule is None and not external_dm:
                if training.dataset is not None:
                    inherited[("dataset",)] = training.dataset.model_dump(mode="json")
                if training.datamodule is not None:
                    inherited[("data", "batch_size")] = training.datamodule.batch_size
                    inherited[("data", "num_workers")] = training.datamodule.num_workers
                pipelines, _ = _resolve_prediction_transform_pipelines(
                    prediction_config=PredictionConfig.model_construct(),
                    training_config=training,
                    training_datamodule_external=False,
                    model_family=self.model_family,
                )
                inherited[("transform_pipelines",)] = [
                    p.model_dump(mode="json") for p in pipelines
                ]
            if (
                isinstance(self.overrides.model, type)
                and training.overrides.model is not None
            ):
                inherited[("overrides", "model")] = training.overrides.model.model_dump(
                    mode="json"
                )
            self.manifest, self.training, self.inherited = manifest, training, inherited
            self.loaded_path = path
        except Exception as error:  # Show source failures without preventing editing.
            self.error = str(error)

    def validate_source(self, data: Any) -> dict[tuple, str]:
        """Require a manifest locally unless the launcher supplies its path."""
        if self.overrides.training_manifest_path is not None:
            return {}

        value = (
            data.get("training_manifest_path")
            if isinstance(data, dict)
            else None
        )
        if value is None or (isinstance(value, str) and not value.strip()):
            return {
                ("training_manifest_path",): (
                    "Provide a training manifest path here or through "
                    "PredictionOverrides.training_manifest_path."
                ),
            }

        return {}

    def panel(self, draft: dict, node: dict):
        current, path = self.current(draft), self.path(draft)
        title = "Manifest not loaded"
        message = (
            "Load the source manifest to view inherited settings "
            "and model-specific suggestions."
        )
        failed = False

        if current:
            title = "Manifest loaded"
            run_name = self.manifest.get("run", {}).get("run_name", "unnamed")
            message = html.Ul(
                [
                    html.Li([html.Strong("Training run: "), run_name]),
                    html.Li(
                        [
                            html.Strong("Model family: "),
                            self.model_family.name,
                        ]
                    ),
                ],
                className="br-source-summary",
            )
        elif self.error and self.attempted_path == (str(path) if path else None):
            title = "Could not load manifest"
            message = self.error
            failed = True
        elif self.loaded_path:
            title = "Manifest path changed"
            message = (
                "Load the source manifest to refresh inherited settings "
                "and suggestions."
            )

        status_content: list[Component] = [
            html.Strong(title, className="br-source-status-title"),
            html.Div(message, className="br-source-status-message"),
        ]
        if self.overrides.training_manifest_path is not None:
            status_content.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Supplied in Python · "),
                                dmc.Popover(
                                    [
                                        dmc.PopoverTarget(
                                            html.Button(
                                                "Manifest path ⓘ",
                                                type="button",
                                                title=str(path),
                                                className="br-source-path-button",
                                            )
                                        ),
                                        dmc.PopoverDropdown(
                                            html.Code(
                                                str(path),
                                                className="br-source-path-value",
                                            )
                                        ),
                                    ],
                                    position="bottom-start",
                                    withArrow=True,
                                    withinPortal=True,
                                    styles={
                                        "dropdown": {
                                            "maxWidth": "min(480px, calc(100vw - 32px))",
                                        },
                                    },
                                ),
                            ],
                            className="br-source-origin",
                        ),
                        html.Div("Checkpoint selection remains editable."),
                    ],
                    className="br-source-note",
                )
            )

        details = [
            html.Div(
                [
                    html.Div(
                        status_content,
                        className="br-source-status-copy",
                        role="status",
                        **{"aria-live": "polite"},
                    ),
                    html.Div(
                        dmc.Tooltip(
                            dmc.Button(
                                "Reload manifest" if current else "Load manifest",
                                id=field_id(
                                    "source", node, "load_source", button=True
                                ),
                                color="purple",
                            ),
                            label=(
                                "Read the training manifest again to refresh "
                                "inherited values and suggestions. Your explicit "
                                "edits are preserved; previous configuration "
                                "checks are invalidated."
                            ),
                            multiline=True,
                            w=320,
                            withArrow=True,
                        ),
                        className="br-source-status-action",
                    ),
                ],
                className=(
                    "br-source-status br-source-status-error"
                    if failed
                    else "br-source-status"
                ),
            ),
        ]

        if current:
            summary = {
                "manifest": str(path),
                "construction": self.manifest["construction"],
                "checkpoints": self.manifest.get("checkpoints", {}),
                "run": self.training.run.model_dump(mode="json"),
                "trainer": self.training.trainer.model_dump(mode="json"),
            }
            details.append(
                html.Details(
                    [
                        html.Summary("Training source details"),
                        html.Pre(
                            yaml.safe_dump(summary, sort_keys=False),
                            className="br-yaml-preview",
                        ),
                    ]
                )
            )

        return html.Div(details, className="br-source-context")

    def action(self, action: str, path: tuple, draft: dict):
        if action == "load_source":
            self.load(draft)
        elif action == "copy_inherited" and self.current(draft):
            return deepcopy(self.inherited.get(path, MISSING))
        return MISSING

    def inheritance(self, path: tuple, draft: dict, guide: FieldPresentation):
        if not self.current(draft) or _at(draft, path) is not None:
            return guide
        if path in self.inherited:
            value = self.inherited[path]
            if isinstance(value, (dict, list)):
                note = "Inherited from the linked training run. Copying makes this an explicit replacement."
                if path == ("dataset",):
                    note += " This includes its dataset split, not the training-validation partition."
                if path == ("transform_pipelines",):
                    note = (
                        note
                        + " Only validation-targeted steps are inherited; an empty list disables transforms."
                        if value
                        else "No validation-targeted training steps to inherit. Prediction uses no configured transforms."
                    )
                return replace(guide, note=note, inherited_value=value)
            return replace(guide, note=f"Inherited from training: {value!r}.")
        if (
            path in {("data", "batch_size"), ("data", "num_workers")}
            and self.overrides.datamodule is None
        ):
            return replace(
                guide,
                note=f"No reconstructable training datamodule; uses the schema default: {getattr(TrainingDataModuleConfig(), path[-1])}.",
            )
        if path == ("transform_pipelines",) and self.overrides.datamodule is None:
            return replace(
                guide,
                note="No transforms inherited from the external training datamodule.",
            )
        if path == ("dataset",) and self.overrides.datamodule is None:
            return replace(
                guide,
                note="No reconstructable training dataset. Supply a prediction dataset.",
            )
        if path == ("exports", "reconstructions", "seed"):
            seed = _at(draft, ("inference", "seed"))
            if seed is None:
                seed = self.training.reproducibility.seed
            return replace(
                guide,
                note=f"For random selection, falls back to the prediction seed: {seed!r}.",
            )
        return guide

    def presentation(self, path: tuple, value: Any, draft: dict) -> FieldPresentation:
        guide = FieldPresentation()
        composite = self.model_family == COMPOSITE_FAMILY
        training = self.training if self.current(draft) else None
        declarations = training.composite_model_declarations if training else None
        expects = declarations.expects if declarations else {}
        produces = declarations.produces if declarations else {}
        if path[0] == "dataset":
            guide = dataset_presentation(path, value, draft)
        elif (
            path == ("source", "training_manifest_path")
            and self.overrides.training_manifest_path is not None
        ):
            guide = replace(guide, note="Supplied in Python.")
        elif path == ("source", "checkpoint"):
            guide = replace(
                guide,
                branch_labels=("From manifest", "Filename or path"),
            )
        elif path == ("data",) and self.overrides.datamodule is not None:
            guide = replace(guide, excluded_fields=("batch_size", "num_workers"))
        elif path == ("overrides",):
            guide = replace(
                guide,
                excluded_fields=tuple(
                    name
                    for name in ("model", "datamodule")
                    if not isinstance(getattr(self.overrides, name), type)
                ),
            )
        elif path == ("inference", "canonical_vae_reconstruction_latent_source"):
            guide = replace(
                guide,
                hidden=self.model_family != VAE_FAMILY
                or self.overrides.model is not None,
            )
        elif path == ("inference", "composite_model_assembly_input_overrides"):
            guide = replace(guide, hidden=not composite, item_label="Assembly step")
        if path == ("transform_pipelines",):
            guide = replace(guide, item_label="Pipeline")
        elif path[0] == "transform_pipelines" and len(path) == 3:
            if path[-1] == "steps":
                guide = replace(guide, item_label="Step")
            elif path[-1] in {"input", "output"}:
                guide = replace(
                    guide,
                    suggestions=(
                        tuple(
                            n
                            for n, r in expects.items()
                            if TENSOR_STRUCTURE_BY_ROLE[r] == "image"
                        )
                        if composite
                        else ("x",)
                    ),
                )
        prefix = ("inference", "composite_model_assembly_input_overrides")
        if path[:2] == prefix and training and composite:
            assembly = training.composite_model_assembly or {}
            outputs = {
                name: (
                    tuple(step.outputs.values())
                    if isinstance(step.outputs, dict)
                    else (step.outputs,)
                )
                for name, step in assembly.items()
            }
            producers = {ref: name for name, refs in outputs.items() for ref in refs}
            if path == prefix:
                guide = replace(
                    guide,
                    key_suggestions=tuple(
                        name
                        for name, step in assembly.items()
                        if any(ref in producers for ref in step.inputs.values())
                    ),
                )
            elif len(path) >= 4 and path[3] == "inputs":
                step = assembly.get(path[2])
                if step and len(path) == 4:
                    guide = replace(
                        guide,
                        key_suggestions=tuple(
                            n for n, ref in step.inputs.items() if ref in producers
                        ),
                    )
                elif step and len(path) == 5:
                    original = step.inputs.get(path[4])
                    guide = replace(
                        guide,
                        suggestions=outputs.get(producers.get(original), ()),
                        note=f"Original route: {original}. Select an output from the same producer step.",
                    )
        if path[:2] == ("exports", "anndata"):
            available = (
                tuple(
                    n
                    for n, r in produces.items()
                    if TENSOR_STRUCTURE_BY_ROLE[r] != "image"
                )
                if composite
                else (
                    ("embedding", "z_mu", "z_logvar", "z_sample")
                    if self.model_family == VAE_FAMILY
                    else ("embedding",)
                )
            )
            if path == ("exports", "anndata", "keys"):
                guide = replace(guide, item_label="Output key")
            elif len(path) == 4 and path[2] == "keys":
                guide = replace(guide, suggestions=available)
            elif path == ("exports", "anndata", "primary_key"):
                vectors = tuple(
                    n
                    for n in available
                    if not composite
                    or TENSOR_STRUCTURE_BY_ROLE[produces[n]] == "vector"
                )
                if _at(draft, ("exports", "anndata", "mode")) == "custom":
                    selected = _at(draft, ("exports", "anndata", "keys"), []) or []
                    vectors = tuple(n for n in vectors if n in selected)
                guide = replace(
                    guide,
                    suggestions=vectors,
                    note=(
                        "Composite AnnData export requires an explicit vector-valued primary key."
                        if composite
                        else None
                    ),
                )
        if path[:2] == ("exports", "reconstructions"):
            if path == ("exports", "reconstructions", "pairs"):
                guide = replace(
                    guide, item_label="Reconstruction pair", reorderable=False
                )
            elif len(path) == 5 and path[2] == "pairs":
                if path[-1] == "input":
                    guide = replace(
                        guide,
                        suggestions=(
                            tuple(
                                n
                                for n, r in expects.items()
                                if TENSOR_STRUCTURE_BY_ROLE[r] == "image"
                            )
                            if composite
                            else ("input",)
                        ),
                    )
                elif path[-1] == "reconstruction":
                    guide = replace(
                        guide,
                        suggestions=(
                            tuple(
                                n
                                for n, r in produces.items()
                                if r == "reconstruction_image"
                            )
                            if composite
                            else ("reconstruction",)
                        ),
                    )
            elif path[-1] == "stratify_by":
                suggestions = None
                if composite and declarations:
                    suggestions = tuple(
                        n
                        for n, r in expects.items()
                        if TENSOR_STRUCTURE_BY_ROLE[r] == "scalar"
                    ) + tuple(declarations.batch_metadata)
                elif not composite and self.overrides.model is None:
                    suggestions = ("sample_id", "label")
                guide = replace(guide, suggestions=suggestions)
        return self.inheritance(path, draft, guide)


def prediction_sections(*, context: PredictionSourceContext) -> list[SectionSpec]:
    docs = {
        "source": PredictionSourceConfig,
        "overrides": RuntimeOverridesConfig,
        "dataset": DatasetConfig,
        "transform_pipelines": PredictionTransformPipelineConfig,
        "data": PredictionDataConfig,
        "inference": PredictionInferenceConfig,
        "exports": PredictionExportConfig,
    }
    descriptions = {
        "source": (
            "Source",
            "Select the training manifest and checkpoint. Load the source for inheritance previews and suggestions.",
        ),
        "overrides": (
            "Override parameters",
            "Constructor arguments for classes supplied in Python. Model arguments may inherit from training; datamodule arguments do not.",
        ),
        "dataset": (
            "Dataset",
            "Omit or use null to inherit the training dataset, including its split. An explicit dataset replaces it entirely.",
        ),
        "transform_pipelines": (
            "Transforms",
            "Omit or use null to inherit validation-targeted training steps. An empty list disables transforms; an explicit list replaces them.",
        ),
        "data": (
            "Data loading",
            "Configure prediction batching and the optional batch limit. External datamodules control their loaders; the batch limit still applies.",
        ),
        "inference": (
            "Inference",
            "Configure prediction randomness and model-specific inference behavior.",
        ),
        "exports": (
            "Exports",
            "Select AnnData outputs and optional reconstruction bundles. Model-dependent compatibility is checked at final review.",
        ),
    }
    keys = ["source"]
    if any(
        isinstance(getattr(context.overrides, name), type)
        for name in ("model", "datamodule")
    ):
        keys.append("overrides")
    if context.overrides.datamodule is None:
        keys.extend(["dataset", "transform_pipelines"])
    keys.extend(["data", "inference", "exports"])
    sections = []
    for key in keys:
        info = PredictionConfig.model_fields[key]
        title, description = descriptions[key]
        sections.append(
            SectionSpec(
                key=key,
                title=title,
                description=description,
                annotation=info.rebuild_annotation(),
                field_info=info,
                documentation_schema=docs[key],
                default_registry={
                    "dataset": "dataset",
                    "transform_pipelines": "transform",
                }.get(key),
                presentation=context.presentation,
                options=(
                    transform_name_options if key == "transform_pipelines" else None
                ),
                validate_context=context.validate_source if key == "source" else None,
                validate_local={
                    "dataset": validate_dataset_name,
                    "transform_pipelines": validate_transform_names,
                }.get(key),
                refresh_on_change=lambda path: path
                in {
                    ("source", "training_manifest_path"),
                    ("inference", "seed"),
                    ("exports", "anndata", "mode"),
                }
                or path[:3] == ("exports", "anndata", "keys"),
                locked_paths=(
                    frozenset({("source", "training_manifest_path")})
                    if key == "source"
                    and context.overrides.training_manifest_path is not None
                    else frozenset()
                ),
                context_panel=context.panel if key == "source" else None,
                context_action=context.action,
            )
        )
    return sections


def prediction_review_session(
    *, context: PredictionSourceContext, compatibility_policy: str = "error"
) -> ReviewSession:
    overrides, family = context.overrides, context.model_family
    if compatibility_policy not in {"error", "warn"}:
        raise ValueError("compatibility_policy must be 'error' or 'warn'.")

    def parse(raw):
        return parse_prediction_config(
            raw,
            model_overridden=overrides.model is not None,
            datamodule_overridden=overrides.datamodule is not None,
            training_manifest_path_overridden=overrides.training_manifest_path
            is not None,
        )

    def resolve(config):
        return resolve_prediction_config(
            config,
            model_family=family,
            training_manifest_path_override=overrides.training_manifest_path,
            model_source=(
                "config"
                if overrides.model is None
                else (
                    "external_class"
                    if isinstance(overrides.model, type)
                    else "external_instance"
                )
            ),
            datamodule_source=resolve_component_source(
                overrides.datamodule,
                expected_base_class=L.LightningDataModule,
                component_name="datamodule",
            ),
            model_override_name=(
                get_component_override_name(overrides.model)
                if overrides.model is not None
                else None
            ),
            compatibility_policy=compatibility_policy,
        )

    def external_identity(draft):
        path = context.path(draft)
        return (
            str(path) + ":" + hashlib.sha256(path.read_bytes()).hexdigest()
            if path
            else "missing"
        )

    def dependencies(config):
        from importlib.util import find_spec

        if config.exports.anndata.enabled and find_spec("anndata") is None:
            raise ImportError("AnnData export requires the anndata package.")
        return (
            "passed",
            "Known export dependencies are available. Custom component dependencies and checkpoint loading are checked during execution.",
        )

    def summary(spec):
        return {
            "checkpoint": str(spec.checkpoint_path),
            "checkpoint_source": spec.checkpoint_source,
            "inherited_fields": sorted(spec.inherited_config_fields),
            "data": {
                "batch_size": spec.batch_size,
                "num_workers": spec.num_workers,
                "max_batches": spec.max_batches,
            },
            "transform_source": spec.transform_pipeline_source,
            "canonical_vae_reconstruction_latent_source": spec.canonical_vae_reconstruction_latent_source,
            "exports": TypeAdapter(Any).dump_python(
                asdict(spec.export_spec), mode="json"
            ),
        }

    return ReviewSession(
        parse=parse,
        resolve=resolve,
        dependencies=dependencies,
        graph=training_spec_graph,
        resolved_config=lambda spec: spec.prediction_config,
        execution_overrides=replace(overrides, config_components=None),
        summary=summary,
        external_identity=external_identity,
    )
