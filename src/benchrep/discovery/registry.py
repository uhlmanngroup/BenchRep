from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import partial
from textwrap import fill, indent
from typing import Any, Final, TextIO

from benchrep.assembly.registries.core import (
    DATASETS,
    TRANSFORMS,
    ENCODERS,
    DECODERS,
    HEADS,
    MODELS,
    RECONSTRUCTION_LOSSES,
    REGULARIZATION_LOSSES,
    CONTRASTIVE_LOSSES,
    CLASSIFICATION_LOSSES,
    REGRESSION_LOSSES,
    CUSTOM_OBJECTIVE_LOSSES,
    LOSS_WEIGHT_SCHEDULES,
    OPTIMIZERS,
    LR_SCHEDULERS,
    LOGGERS,
    CALLBACKS,
    EVAL_REDUCTIONS,
    EVAL_CLUSTERING_METHODS,
    EVAL_INTERNAL_CLUSTERING_METRICS,
    EVAL_EXTERNAL_CLUSTERING_METRICS,
    EVAL_EMBEDDING_METRICS,
    EVAL_PREDICTABILITY_PROBES,
    EVAL_RECONSTRUCTION_METRICS,
    Registry,
    LossRegistry,
)
from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
    ComponentMappingResult,
    ComponentTensorResult,
)
from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
)
from benchrep.evaluation.metrics import EvaluationMetric


@dataclass(frozen=True)
class ComponentRegistryInfo:
    symbol: str
    registry: Registry
    runtime_instance_override_supported: bool | None = None
    config_locations: tuple[str, ...] = ()
    contract: str | None = None
    runtime_override: str | None = None

    @property
    def custom_registration_supported(self) -> bool:
        """Return the registration policy enforced by the registry."""
        return self.registry.custom_registration_supported


_COMPOSITE_ONLY_LOSS_COMPONENT_CONTRACT: Final[str] = (
    "Entries must be LossComponent instances wrapping an nn.Module loss "
    "that returns a scalar tensor. This loss role is supported only by "
    "Composite models, so runtime_inputs must explicitly declare the "
    "forward parameter names and semantic roles accepted by the component."
)

_COMPONENT_REGISTRIES: Final[dict[str, ComponentRegistryInfo]] = {
    "dataset": ComponentRegistryInfo(
        symbol="DATASETS",
        registry=DATASETS,
        runtime_instance_override_supported=False,
        config_locations=(
            "TrainingConfig.dataset",
            "PredictionConfig.dataset",
        ),
        contract=(
            "The registered callable must accept the configured `params` and return a "
            "`BaseDataset` instance."
        ),
    ),
    "transform": ComponentRegistryInfo(
        symbol="TRANSFORMS",
        registry=TRANSFORMS,
        runtime_instance_override_supported=False,
        config_locations=(
            "TrainingConfig.transform_pipelines[*].steps[*]",
            "PredictionConfig.transform_pipelines[*].steps[*]",
        ),
        contract=(
            "The registered factory must accept the configured `params` and return a "
            "callable that transforms a routed sample tensor before batching. "
            "Pipeline routes determine the input and output sample fields."
        ),
    ),
    "encoder": ComponentRegistryInfo(
        symbol="ENCODERS",
        registry=ENCODERS,
        runtime_instance_override_supported=False,
        config_locations=(
            "TrainingConfig.encoder",
            "TrainingConfig.composite_model_components[*] (kind='encoder')",
        ),
        contract=(
            "ArchitectureComponent.component must be an `nn.Module` class. "
            "Its declared runtime contract defines the supported Composite "
            "wiring. Canonical encoder use additionally requires the component "
            "to satisfy BenchRep's `BaseEncoder` interface."
        ),
    ),
    "decoder": ComponentRegistryInfo(
        symbol="DECODERS",
        registry=DECODERS,
        runtime_instance_override_supported=False,
        config_locations=(
            "TrainingConfig.decoder",
            "TrainingConfig.composite_model_components[*] (kind='decoder')",
        ),
        contract=(
            "ArchitectureComponent.component must be an `nn.Module` class. "
            "Its declared runtime contract defines the supported Composite "
            "wiring. Canonical decoder use additionally requires the component "
            "to satisfy BenchRep's `BaseDecoder` interface."
        ),
    ),
    "head": ComponentRegistryInfo(
        symbol="HEADS",
        registry=HEADS,
        runtime_instance_override_supported=False,
        config_locations=(
            "TrainingConfig.composite_model_components[*] (kind='head')",
        ),
        contract=(
            "ArchitectureComponent.component must be an `nn.Module` class. "
            "Its declared runtime contract defines the supported Composite "
            "wiring, including accepted forward inputs and result structure."
        ),
    ),
    "model": ComponentRegistryInfo(
        symbol="MODELS",
        registry=MODELS,
        runtime_instance_override_supported=True,
        config_locations=("TrainingConfig.model",),
        runtime_override=(
            "Pass a compatible model instance or class through "
            "`TrainingOverrides(model=...)` or `PredictionOverrides(model=...)`, "
            "supplied as the workflow entrypoint's `overrides` argument. "
            "Whole-model overrides are supported only for the canonical "
            "autoencoder and VAE workflows."
        ),
    ),
    "reconstruction_loss": ComponentRegistryInfo(
        symbol="RECONSTRUCTION_LOSSES",
        registry=RECONSTRUCTION_LOSSES,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.losses.reconstruction",),
        contract=(
            "Entries must be LossComponent instances wrapping an nn.Module "
            "loss that returns a scalar tensor. Canonical models call "
            "forward() with `reconstruction` and `target` as keyword "
            "arguments; use LossComponent(MyLoss) for canonical-only use. "
            "Composite use additionally requires explicit runtime_inputs "
            "declaring the accepted parameter names and semantic roles."
        ),
    ),
    "regularization_loss": ComponentRegistryInfo(
        symbol="REGULARIZATION_LOSSES",
        registry=REGULARIZATION_LOSSES,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.losses.regularization",),
        contract=(
            "Entries must be LossComponent instances wrapping an nn.Module "
            "loss that returns a scalar tensor. Canonical VAEs call forward() "
            "with `z_mu` and `z_logvar` as keyword arguments; use "
            "LossComponent(MyLoss) for canonical-only use. Composite use "
            "additionally requires explicit runtime_inputs declaring the "
            "accepted parameter names and semantic roles."
        ),
    ),
    "contrastive_loss": ComponentRegistryInfo(
        symbol="CONTRASTIVE_LOSSES",
        registry=CONTRASTIVE_LOSSES,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.losses.contrastive",),
        contract=_COMPOSITE_ONLY_LOSS_COMPONENT_CONTRACT,
    ),
    "classification_loss": ComponentRegistryInfo(
        symbol="CLASSIFICATION_LOSSES",
        registry=CLASSIFICATION_LOSSES,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.losses.classification",),
        contract=_COMPOSITE_ONLY_LOSS_COMPONENT_CONTRACT,
    ),
    "regression_loss": ComponentRegistryInfo(
        symbol="REGRESSION_LOSSES",
        registry=REGRESSION_LOSSES,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.losses.regression",),
        contract=_COMPOSITE_ONLY_LOSS_COMPONENT_CONTRACT,
    ),
    "custom_objective_loss": ComponentRegistryInfo(
        symbol="CUSTOM_OBJECTIVE_LOSSES",
        registry=CUSTOM_OBJECTIVE_LOSSES,
        runtime_instance_override_supported=False,
        config_locations=(
            "TrainingConfig.losses.custom_objective",
        ),
        contract=(
            "Entries must be LossComponent instances wrapping a "
            "BaseCustomObjectiveLoss subclass. runtime_inputs must be omitted "
            "because BenchRep always supplies the complete `batch` and "
            "`model_output` mappings automatically under canonical and "
            "Composite models. The component's forward() method must accept "
            "both as keyword arguments and return a scalar tensor."
        ),
    ),
    "loss_weight_schedule": ComponentRegistryInfo(
        symbol="LOSS_WEIGHT_SCHEDULES",
        registry=LOSS_WEIGHT_SCHEDULES,
        runtime_instance_override_supported=False,
        config_locations=(
            "TrainingConfig.losses.<role>[*].weight_schedule",
        ),
        contract=(
            "The registered constructor accepts the configured `params` "
            "as keyword arguments and returns an object satisfying the "
            "`LossWeightSchedule` Protocol; explicit inheritance is not "
            "required. Its `factor` method must accept a position as one "
            "positional argument, regardless of parameter name, and return "
            "a finite, nonnegative scalar multiplier for the loss's base "
            "weight. Position is a nonnegative epoch index or completed "
            "optimizer-step count. Results must be deterministic and "
            "independent of call history. Optional cycling is applied "
            "by BenchRep."
        ),
    ),
    "optimizer": ComponentRegistryInfo(
        symbol="OPTIMIZERS",
        registry=OPTIMIZERS,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.optimizer",),
        contract=(
            "The registered callable must accept model parameters as its first argument "
            "and the configured `params` as keyword arguments, and return a "
            "`torch.optim.Optimizer`."
        ),
    ),
    "lr_scheduler": ComponentRegistryInfo(
        symbol="LR_SCHEDULERS",
        registry=LR_SCHEDULERS,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.optimizer.lr_scheduler",),
        contract=(
            "The registered class or factory is called as "
            "`factory(optimizer, **params)`, where `optimizer` is the "
            "optimizer BenchRep created for training. It must return a "
            "`torch.optim.lr_scheduler.LRScheduler` or "
            "`ReduceLROnPlateau` instance constructed with that same "
            "optimizer. The scheduler stores a reference to it; it does "
            "not create or replace the optimizer. "
            "Lightning uses the optimizer to update model parameters "
            "and calls the scheduler to update that optimizer's learning "
            "rate. Lightning also saves and restores scheduler state. "
            "`interval`, `frequency`, and `monitor` control scheduler "
            "execution and are not passed to the constructor. "
            "If the returned scheduler is a `ReduceLROnPlateau` instance, "
            "`monitor` must name the logged metric it should track."
        ),
    ),
    "logger": ComponentRegistryInfo(
        symbol="LOGGERS",
        registry=LOGGERS,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.logger",),
        contract=(
            "The registered class must satisfy Lightning's logger interface."
        ),
    ),
    "callback": ComponentRegistryInfo(
        symbol="CALLBACKS",
        registry=CALLBACKS,
        runtime_instance_override_supported=False,
        config_locations=("TrainingConfig.additional_callbacks",),
        contract=(
            "The registered class or factory must accept the configured "
            "`params` and return a Lightning `Callback` instance."
        ),
    ),
    "reduction": ComponentRegistryInfo(
        symbol="EVAL_REDUCTIONS",
        registry=EVAL_REDUCTIONS,
        runtime_instance_override_supported=False,
        config_locations=("EvaluationConfig.reductions",),
    ),
    "clustering_method": ComponentRegistryInfo(
        symbol="EVAL_CLUSTERING_METHODS",
        registry=EVAL_CLUSTERING_METHODS,
        runtime_instance_override_supported=False,
        config_locations=("EvaluationConfig.clustering",),
    ),
    "internal_clustering_metric": ComponentRegistryInfo(
        symbol="EVAL_INTERNAL_CLUSTERING_METRICS",
        registry=EVAL_INTERNAL_CLUSTERING_METRICS,
        runtime_instance_override_supported=False,
        config_locations=(
            "EvaluationConfig.metrics.clustering.internal",
        ),
        contract=(
            "The registered value must be an EvaluationMetric whose callable "
            "accepts an embedding matrix and cluster labels, followed by the "
            "configured parameters. Its return value must satisfy the declared "
            "result_kind and vector_axis contract."
        ),
    ),
    "external_clustering_metric": ComponentRegistryInfo(
        symbol="EVAL_EXTERNAL_CLUSTERING_METRICS",
        registry=EVAL_EXTERNAL_CLUSTERING_METRICS,
        runtime_instance_override_supported=False,
        config_locations=(
            "EvaluationConfig.metrics.clustering.external",
        ),
        contract=(
            "The registered value must be an EvaluationMetric whose callable "
            "accepts reference labels and predicted cluster labels, followed by "
            "the configured parameters. Its return value must satisfy the "
            "declared result_kind and vector_axis contract."
        ),
    ),
    "embedding_metric": ComponentRegistryInfo(
        symbol="EVAL_EMBEDDING_METRICS",
        registry=EVAL_EMBEDDING_METRICS,
        runtime_instance_override_supported=False,
        config_locations=("EvaluationConfig.metrics.embedding",),
        contract=(
            "The registered value must be an EvaluationMetric whose callable "
            "accepts an embedding matrix followed by the configured parameters. "
            "Its return value must satisfy the declared result_kind and "
            "vector_axis contract."
        ),
    ),
    "predictability_probe": ComponentRegistryInfo(
        symbol="EVAL_PREDICTABILITY_PROBES",
        registry=EVAL_PREDICTABILITY_PROBES,
        runtime_instance_override_supported=False,
        config_locations=(
            "EvaluationConfig.metrics.predictability.targets[*]",
        ),
    ),
    "reconstruction_metric": ComponentRegistryInfo(
        symbol="EVAL_RECONSTRUCTION_METRICS",
        registry=EVAL_RECONSTRUCTION_METRICS,
        runtime_instance_override_supported=False,
        config_locations=("EvaluationConfig.metrics.reconstruction",),
        contract=(
            "The registered value must be an EvaluationMetric whose callable "
            "accepts input and reconstruction arrays, followed by the configured "
            "parameters. Its return value must satisfy the declared result_kind "
            "and vector_axis contract."
        ),
    ),
}


def inspect_registry(
    registry: str | None = None,
    component: str | None = None,
    *,
    file: TextIO | None = None,
) -> None:
    """Print human-readable information about BenchRep registries.

    With no arguments, summarizes all available registries. With a registry
    name, lists its registered components and customization policy. With both
    a registry and component name, describes the selected implementation.

    Component names may be canonical names or registered aliases.
    """
    emit = partial(print, file=file)

    if registry is None:
        if component is not None:
            raise ValueError(
                "component cannot be provided without a registry."
            )

        emit("BenchRep registries")

        for registry_name, registry_info in _COMPONENT_REGISTRIES.items():
            selected_registry = registry_info.registry
            n_components = len(selected_registry.canonical_keys())
            component_label = (
                "component" if n_components == 1 else "components"
            )

            emit(
                f"- {registry_name}: {n_components} {component_label} "
                f"({_registry_object_path(registry_info)})"
            )
            emit(
                "  custom registration: "
                f"{_support_label(registry_info.custom_registration_supported)}; "
                "registration entry: "
                f"{_registration_entry_label(registry_info)}; "
                "runtime instance override: "
                f"{_support_label(registry_info.runtime_instance_override_supported)}"
            )

        return

    data = registry_documentation(
        registry,
        component,
        include_component_details=component is None,
    )
    _, registry_info = _resolve_registry(registry)

    emit(f"Registry: {data['name']}")
    emit(f"Registry object: {data['object_path']}")
    _print_registry_customization(registry_info, file=file)

    if component is None:
        emit("\nComponents:")
        for details in data["components"]:
            _print_component_documentation(
                details,
                detailed=False,
                file=file,
            )
    else:
        _print_component_documentation(
            data["selected_component"],
            detailed=True,
            file=file,
        )


def registry_documentation(
    registry: str,
    component: str | None = None,
    *,
    include_component_details: bool = False,
) -> dict[str, Any]:
    """Return registry documentation without console formatting.

    Component listings contain names and aliases by default. Detailed listings
    also include targets and runtime contracts, without inspecting docstrings
    or signatures. A selected component includes its full documentation.
    """
    registry_name, info = _resolve_registry(registry)
    selected_registry = info.registry
    aliases = selected_registry.aliases_by_canonical()

    def describe_component(
            name: str,
            *,
            include_description: bool = True,
    ) -> dict[str, Any]:
        entry = selected_registry.get(name)
        target = _registry_entry_target(entry)

        details: dict[str, Any] = {
            "name": name,
            "aliases": aliases[name],
            "target": _qualified_name(target),
            "signature": (
                _callable_signature(target) if include_description else None
            ),
            "signature_label": (
                "Constructor signature"
                if inspect.isclass(target)
                else "Callable signature"
            ),
            "description": inspect.getdoc(target) if include_description else None,
        }

        if isinstance(entry, ArchitectureComponent):
            result = entry.runtime_result

            if isinstance(result, ComponentTensorResult):
                outputs = [{
                    "name": "Result",
                    "structures": result.supported_structures,
                }]
                result_kind = "tensor"
            elif isinstance(result, ComponentMappingResult):
                outputs = [
                    {
                        "name": output.name,
                        "structures": output.supported_structures,
                    }
                    for output in result.outputs
                ]
                result_kind = "mapping"
            else:
                raise RuntimeError(
                    "Unsupported architecture result type: "
                    f"{type(result).__name__}."
                )

            details["runtime_contract"] = {
                "inputs": [
                    {
                        "name": port.name,
                        "structures": port.supported_structures,
                    }
                    for port in entry.runtime_inputs
                ],
                "result_kind": result_kind,
                "outputs": outputs,
            }

        elif isinstance(entry, LossComponent):
            if entry.runtime_inputs is None:
                input_policy = (
                    "fixed by registry"
                    if isinstance(selected_registry, LossRegistry)
                    and not selected_registry.explicit_composite_runtime_inputs_supported
                    else "not declared"
                )
                inputs = None
            else:
                input_policy = "declared"
                inputs = [
                    {
                        "name": port.name,
                        "roles": port.supported_roles,
                    }
                    for port in entry.runtime_inputs
                ]

            details["runtime_contract"] = {
                "input_policy": input_policy,
                "inputs": inputs,
                "result_kind": "scalar tensor",
            }

        elif isinstance(entry, EvaluationMetric):
            details["result_kind"] = entry.result_kind
            details["vector_axis"] = entry.vector_axis

        return details

    return {
        "name": registry_name,
        "object_path": _registry_object_path(info),
        "custom_registration": info.custom_registration_supported,
        "registration_entry": _registration_entry_label(info),
        "runtime_instance_override": info.runtime_instance_override_supported,
        "config_locations": info.config_locations,
        "contract": info.contract,
        "runtime_override": info.runtime_override,
        "components": [
            describe_component(name, include_description=False)
            if include_component_details
            else {"name": name, "aliases": component_aliases}
            for name, component_aliases in aliases.items()
        ],
        "selected_component": (
            describe_component(selected_registry.resolve_key(component))
            if component is not None
            else None
        ),
    }


def _resolve_registry(
    registry: str,
) -> tuple[str, ComponentRegistryInfo]:
    if not isinstance(registry, str):
        raise TypeError(
            f"registry must be a string, got {type(registry).__name__}."
        )

    registry_name = registry.lower().strip().replace("-", "_")

    if registry_name not in _COMPONENT_REGISTRIES:
        available = tuple(_COMPONENT_REGISTRIES)
        raise ValueError(
            f"Unknown registry {registry!r}. "
            f"Available registries: {available}. "
            "Run `benchrep.inspect_registry()` with no arguments for a "
            "registry overview."
        )

    return registry_name, _COMPONENT_REGISTRIES[registry_name]


def _print_registry_customization(
    registry_info: ComponentRegistryInfo,
    *,
    file: TextIO | None = None,
) -> None:
    emit = partial(print, file=file)

    emit(
        "Custom registration: "
        f"{_support_label(registry_info.custom_registration_supported)}"
    )
    emit(
        "Registration entry: "
        f"{_registration_entry_label(registry_info)}"
    )
    emit(
        "Runtime instance override: "
        f"{_support_label(registry_info.runtime_instance_override_supported)}"
    )

    if registry_info.config_locations:
        emit("Config locations:")
        for location in registry_info.config_locations:
            emit(f"  - {location}")

    if registry_info.contract is not None:
        emit("Contract:")
        emit(
            indent(
                fill(
                    registry_info.contract,
                    width=88,
                ),
                "  ",
            )
        )

    if registry_info.runtime_override is not None:
        emit("Runtime override:")
        emit(
            indent(
                fill(
                    registry_info.runtime_override,
                    width=88,
                ),
                "  ",
            )
        )


def _registration_entry_label(
    registry_info: ComponentRegistryInfo,
) -> str:
    entry_type = registry_info.registry.entry_type

    if entry_type is None:
        return "callable"

    return entry_type.__name__


def _support_label(value: bool | None) -> str:
    if value is True:
        return "supported"

    if value is False:
        return "not supported"

    return "not documented"


def _registry_object_path(
    registry_info: ComponentRegistryInfo,
) -> str:
    return (
        "benchrep.assembly.registries."
        f"{registry_info.symbol}"
    )


def _print_component_documentation(
    details: dict[str, Any],
    *,
    detailed: bool,
    file: TextIO | None = None,
) -> None:
    """Print component data shared with graphical discovery."""
    emit = partial(print, file=file)
    prefix = "" if detailed else "    "

    if detailed:
        emit(f"Component: {details['name']}")
    else:
        emit(f"\n  {details['name']}")

    aliases = ", ".join(details["aliases"]) or "none"
    emit(f"{prefix}Aliases: {aliases}")
    emit(f"{prefix}Target: {details['target']}")

    contract = details.get("runtime_contract")
    if contract is not None:
        emit(f"{prefix}Composite runtime contract:")

        if contract.get("input_policy"):
            emit(f"{prefix}  Input policy: {contract['input_policy']}")

        if contract.get("inputs") is not None:
            emit(f"{prefix}  Inputs:")
            for port in contract["inputs"]:
                supported = " | ".join(
                    port.get("structures", port.get("roles", ()))
                )
                emit(f"{prefix}    {port['name']}: {supported}")

        emit(f"{prefix}  Result: {contract['result_kind']}")
        for output in contract.get("outputs", ()):
            supported = " | ".join(output["structures"])
            emit(f"{prefix}    {output['name']}: {supported}")

    if details.get("result_kind") is not None:
        emit(f"{prefix}Result kind: {details['result_kind']}")
        if details.get("vector_axis") is not None:
            emit(f"{prefix}Vector axis: {details['vector_axis']}")

    if detailed:
        emit(
            f"{details['signature_label']}: "
            f"{details['signature'] or 'unavailable'}"
        )
        emit("\nDescription:")
        emit(
            indent(
                details["description"] or "No docstring available.",
                "  ",
            )
        )
        emit(
            "\nNote: the callable signature describes the registered "
            "implementation. BenchRep may supply some arguments internally, "
            "so it does not always map directly to the component's config `params`."
        )


def _registry_entry_target(entry: Any) -> Any:
    """Return the callable represented by a registry entry."""

    if isinstance(entry, (ArchitectureComponent, LossComponent)):
        return entry.component

    elif isinstance(entry, EvaluationMetric):
        return entry.fn

    return entry


def _qualified_name(item: Any) -> str:
    module = getattr(item, "__module__", None)
    qualname = getattr(item, "__qualname__", None)

    if module is not None and qualname is not None:
        return f"{module}.{qualname}"

    return type(item).__name__


def _callable_signature(item: Any) -> str | None:
    try:
        signature = inspect.signature(item)
    except (TypeError, ValueError):
        return None

    if inspect.isclass(item):
        signature = signature.replace(
            return_annotation=inspect.Signature.empty,
        )

    return str(signature)


def list_registries() -> dict[str, str]:
    """Return registry names and their public registry-object paths."""
    return {
        name: _registry_object_path(registry_info)
        for name, registry_info in _COMPONENT_REGISTRIES.items()
    }


def list_registered_components(
    registry: str,
    *,
    include_aliases: bool = False,
) -> tuple[str, ...] | dict[str, tuple[str, ...]]:
    """Return the components registered in a named registry."""
    _, registry_info = _resolve_registry(registry)
    selected_registry = registry_info.registry

    if include_aliases:
        return selected_registry.aliases_by_canonical()

    return selected_registry.canonical_keys()
