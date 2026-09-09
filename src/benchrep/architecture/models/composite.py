"""Configuration-built Composite model runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TYPE_CHECKING

import lightning as L
import torch
from torch import nn

from benchrep.architecture.composite_model_roles import (
    TENSOR_STRUCTURE_BY_ROLE,
    TensorStructure,
)

if TYPE_CHECKING:
    from benchrep.assembly.resolvers.composite_model_resolver import (
        CompositeModelAssemblyStepSpec,
        CompositeModelSpec,
    )


class CompositeModel(L.LightningModule):
    """Execute a resolved Composite model assembly graph.

    The builder instantiates each configured architecture component exactly
    once and supplies it under its component ID. Assembly steps may invoke the
    same component ID repeatedly, preserving parameter sharing.

    The resolved model specification remains the source of truth for assembly
    order, tensor bindings, loss wiring, and configured loss weights.
    """

    def __init__(
        self,
        *,
        model_spec: CompositeModelSpec,
        components_by_id: dict[str, nn.Module],
        loss_modules_by_role: dict[str, dict[str, nn.Module]],
        optimizer_factory: Callable[
            [Iterable[nn.Parameter]],
            torch.optim.Optimizer,
        ],
    ) -> None:
        super().__init__()

        _validate_components_match_spec(
            model_spec=model_spec,
            components_by_id=components_by_id,
        )
        _validate_loss_modules_match_spec(
            model_spec=model_spec,
            loss_modules_by_role=loss_modules_by_role,
        )

        self.model_spec = model_spec
        # Retain the primary sample input name only for determining the
        # runtime batch size used by Lightning logging.
        self._sample_image_input_name = (
            _get_sample_image_input_name(model_spec)
        )
        self.components_by_id = nn.ModuleDict(components_by_id)
        self.optimizer_factory = optimizer_factory

        # Register loss modules by role so any learnable loss parameters
        # participate in optimization, device movement, and checkpointing.
        loss_module_dicts_by_role: dict[str, nn.ModuleDict] = {}

        for loss_role, loss_modules in loss_modules_by_role.items():
            if loss_modules:
                loss_module_dicts_by_role[loss_role] = nn.ModuleDict(
                    loss_modules
                )

        self.loss_modules_by_role = nn.ModuleDict(
            loss_module_dicts_by_role
        )

    def forward(
        self,
        model_inputs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Execute the resolved assembly graph."""

        _validate_model_inputs(
            model_spec=self.model_spec,
            model_inputs=model_inputs,
        )

        model_batch_size = model_inputs[
            self._sample_image_input_name
        ].shape[0]

        model_output: dict[str, torch.Tensor] = {}

        # Structure: {dependency_level: (assembly_step_spec, ...)}.
        # The resolver grouped assembly steps by dependency level. Execute
        # levels in order so every model output exists before it is consumed.
        for dependency_level in sorted(
            self.model_spec.assembly_steps_by_dependency_level
        ):
            assembly_steps = (
                self.model_spec.assembly_steps_by_dependency_level[
                    dependency_level
                ]
            )

            # Each assembly_step is a CompositeModelAssemblyStepSpec
            # dataclass with the fields: step_id, component_id,
            # inputs_from_model_inputs, inputs_from_model_outputs, and
            # results_to_model_outputs.
            for assembly_step in assembly_steps:
                component_inputs: dict[str, torch.Tensor] = {}

                # Get this component's inputs from model inputs declared
                # under `expects`.
                for (
                    component_forward_parameter_name,
                    model_input_name,
                ) in assembly_step.inputs_from_model_inputs.items():
                    component_inputs[component_forward_parameter_name] = (
                        model_inputs[model_input_name]
                    )

                # Get this component's inputs from model outputs declared
                # under `produces`.
                for (
                    component_forward_parameter_name,
                    model_output_name,
                ) in assembly_step.inputs_from_model_outputs.items():
                    component_inputs[component_forward_parameter_name] = (
                        model_output[model_output_name]
                    )

                component = self.components_by_id[
                    assembly_step.component_id
                ]

                # Inputs from `expects` are available in model_inputs. Inputs
                # from `produces` are available because their components ran
                # in earlier dependency levels.
                runtime_result = component(**component_inputs)

                # Check the returned tensor or mapping, including its declared
                # structure and batch size, then bind it under `produces`.
                assembly_step_model_outputs = (
                    _validate_and_bind_assembly_step_result(
                        assembly_step=assembly_step,
                        runtime_result=runtime_result,
                        model_spec=self.model_spec,
                        expected_batch_size=model_batch_size,
                    )
                )
                model_output.update(assembly_step_model_outputs)

        return model_output

    def training_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._compute_loss_step(
            batch,
            stage="train",
        )

    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._compute_loss_step(
            batch,
            stage="val",
        )

    def test_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._compute_loss_step(
            batch,
            stage="test",
        )

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Prediction is not yet implemented for CompositeModel."
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_factory(self.parameters())

    def _compute_loss_step(
        self,
        batch: Mapping[str, Any],
        *,
        stage: str,
    ) -> torch.Tensor:
        """Execute the model and calculate every configured loss term."""

        # Check that the batch contains all declared fields, then extract
        # the model inputs declared under `expects`.
        model_inputs = _extract_model_inputs_from_batch(
            batch=batch,
            model_spec=self.model_spec,
        )
        model_output = self(model_inputs)

        sample_image = model_inputs[self._sample_image_input_name]

        batch_size = sample_image.shape[0]
        total_loss: torch.Tensor | None = None

        # model_spec.loss_specs is a flat tuple containing one
        # CompositeModelLossSpec for every configured loss across all roles.
        # Each spec contains the loss role, name, weight, and resolved wiring.
        #
        # Instantiated loss modules are stored under:
        # loss_modules_by_role[loss_role][loss_name].
        for loss_spec in self.model_spec.loss_specs:
            loss_role = loss_spec.loss_role
            loss_name = loss_spec.configured_loss_name

            loss_module = _get_loss_module(
                loss_modules_by_role=self.loss_modules_by_role,
                loss_role=loss_role,
                loss_name=loss_name,
            )

            loss_runtime_inputs: dict[str, Any] = {}

            # Get this loss module's inputs from model inputs declared
            # under `expects`.
            for (
                loss_forward_parameter_name,
                model_input_name,
            ) in loss_spec.inputs_from_model_inputs.items():
                loss_runtime_inputs[loss_forward_parameter_name] = (
                    model_inputs[model_input_name]
                )

            # Get this loss module's inputs from model outputs declared
            # under `produces`.
            for (
                loss_forward_parameter_name,
                model_output_name,
            ) in loss_spec.inputs_from_model_outputs.items():
                loss_runtime_inputs[loss_forward_parameter_name] = (
                    model_output[model_output_name]
                )

            # Custom objectives receive the complete batch and model output
            # mappings instead of individually wired tensors.
            for (
                loss_forward_parameter_name,
                context_source,
            ) in loss_spec.context_inputs.items():
                if context_source == "batch":
                    loss_runtime_inputs[loss_forward_parameter_name] = batch
                elif context_source == "model_output":
                    loss_runtime_inputs[loss_forward_parameter_name] = model_output
                else:
                    raise RuntimeError(
                        f"Composite loss term "
                        f"{loss_spec.configured_loss_name!r} declares "
                        f"unsupported context source "
                        f"{context_source!r}."
                    )

            raw_loss = loss_module(**loss_runtime_inputs)
            raw_loss = _validate_scalar_loss_result(
                raw_loss,
                loss_role=loss_spec.loss_role,
                configured_loss_name=(
                    loss_spec.configured_loss_name
                ),
            )

            weighted_loss = loss_spec.weight * raw_loss

            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss

            self.log(
                f"{stage}/{loss_spec.loss_role}/"
                f"{loss_spec.configured_loss_name}",
                raw_loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

            self.log(
                f"{stage}/{loss_spec.loss_role}/"
                f"{loss_spec.configured_loss_name}_weighted",
                weighted_loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        if total_loss is None:
            raise RuntimeError(
                "CompositeModel cannot calculate a loss because its "
                "resolved specification contains no configured losses."
            )

        self.log(
            f"{stage}/loss",
            total_loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return total_loss


def _validate_components_match_spec(
    *,
    model_spec: CompositeModelSpec,
    components_by_id: dict[str, nn.Module],
) -> None:
    """Validate that every resolved component was built exactly once."""

    expected_component_ids = set(model_spec.components_by_id)
    received_component_ids = set(components_by_id)

    missing_component_ids = sorted(
        expected_component_ids - received_component_ids
    )
    unexpected_component_ids = sorted(
        received_component_ids - expected_component_ids
    )

    if not missing_component_ids and not unexpected_component_ids:
        return

    problems: list[str] = []

    if missing_component_ids:
        problems.append(
            f"missing component IDs {missing_component_ids}"
        )

    if unexpected_component_ids:
        problems.append(
            f"unexpected component IDs {unexpected_component_ids}"
        )

    raise ValueError(
        "CompositeModel components do not match the resolved model "
        f"specification: {'; '.join(problems)}."
    )


def _validate_loss_modules_match_spec(
    *,
    model_spec: CompositeModelSpec,
    loss_modules_by_role: dict[str, dict[str, nn.Module]],
) -> None:
    """Validate that every resolved loss was instantiated exactly once."""

    # Loss names are scoped by role, so each loss is identified by
    # (loss_role, configured_loss_name).
    expected_loss_ids = {
        (
            loss_spec.loss_role,
            loss_spec.configured_loss_name,
        )
        for loss_spec in model_spec.loss_specs
    }

    received_loss_ids: set[tuple[str, str]] = set()

    for loss_role, loss_modules in loss_modules_by_role.items():
        for configured_loss_name in loss_modules:
            received_loss_ids.add(
                (
                    loss_role,
                    configured_loss_name,
                )
            )

    missing_loss_ids = sorted(
        expected_loss_ids - received_loss_ids
    )
    unexpected_loss_ids = sorted(
        received_loss_ids - expected_loss_ids
    )

    if not missing_loss_ids and not unexpected_loss_ids:
        return

    problems: list[str] = []

    if missing_loss_ids:
        problems.append(
            f"missing losses {missing_loss_ids}"
        )

    if unexpected_loss_ids:
        problems.append(
            f"unexpected losses {unexpected_loss_ids}"
        )

    raise ValueError(
        "CompositeModel loss modules do not match the resolved model "
        f"specification: {'; '.join(problems)}."
    )


def _validate_model_inputs(
    *,
    model_spec: CompositeModelSpec,
    model_inputs: Mapping[str, torch.Tensor],
) -> None:
    """Validate inputs supplied to one Composite forward pass."""

    if not isinstance(model_inputs, Mapping):
        raise TypeError(
            "CompositeModel.forward() expects a mapping from declared "
            "model input names to tensors, got "
            f"{type(model_inputs).__name__}."
        )

    expected_model_input_names = set(
        model_spec.declarations.model_input_roles_by_name
    )
    received_model_input_names = set(model_inputs)

    missing_model_input_names = sorted(
        expected_model_input_names - received_model_input_names,
        key=repr,
    )
    unexpected_model_input_names = sorted(
        received_model_input_names - expected_model_input_names,
        key=repr,
    )

    if missing_model_input_names or unexpected_model_input_names:
        problems: list[str] = []

        if missing_model_input_names:
            problems.append(
                f"missing model inputs {missing_model_input_names}"
            )

        if unexpected_model_input_names:
            problems.append(
                f"unexpected model inputs {unexpected_model_input_names}"
            )

        raise ValueError(
            "CompositeModel.forward() inputs do not match the resolved "
            f"model declarations: {'; '.join(problems)}."
        )

    for model_input_name, model_input in model_inputs.items():
        if not isinstance(model_input, torch.Tensor):
            raise TypeError(
                f"Composite model input {model_input_name!r} must be a "
                f"torch.Tensor, got {type(model_input).__name__}."
            )

    sample_image_input_name = _get_sample_image_input_name(
        model_spec
    )
    sample_image = model_inputs[sample_image_input_name]
    sample_image_role = (
        model_spec.declarations.model_input_roles_by_name[
            sample_image_input_name
        ]
    )
    sample_image_structure = TENSOR_STRUCTURE_BY_ROLE[
        sample_image_role
    ]

    expected_batch_size = _validate_batched_tensor_structure(
        tensor=sample_image,
        tensor_structure=sample_image_structure,
        tensor_description=(
            f"Composite model input {sample_image_input_name!r}"
        ),
    )

    for model_input_name, model_input in model_inputs.items():
        if model_input_name == sample_image_input_name:
            continue

        model_input_role = (
            model_spec.declarations.model_input_roles_by_name[
                model_input_name
            ]
        )
        model_input_structure = TENSOR_STRUCTURE_BY_ROLE[
            model_input_role
        ]

        _validate_batched_tensor_structure(
            tensor=model_input,
            tensor_structure=model_input_structure,
            tensor_description=(
                f"Composite model input {model_input_name!r}"
            ),
            expected_batch_size=expected_batch_size,
        )


def _validate_and_bind_assembly_step_result(
    *,
    assembly_step: CompositeModelAssemblyStepSpec,
    runtime_result: Any,
    model_spec: CompositeModelSpec,
    expected_batch_size: int,
) -> dict[str, torch.Tensor]:
    """Validate one component result and bind it to model-output names."""

    result_bindings = assembly_step.results_to_model_outputs

    if isinstance(result_bindings, str):
        if not isinstance(runtime_result, torch.Tensor):
            raise TypeError(
                f"Composite assembly step {assembly_step.step_id!r} "
                f"invokes component ID "
                f"{assembly_step.component_id!r}, which must return one "
                "torch.Tensor according to its resolved contract; got "
                f"{type(runtime_result).__name__}."
            )

        model_output_name = result_bindings
        model_output_role = (
            model_spec.declarations.model_output_roles_by_name[
                model_output_name
            ]
        )
        model_output_structure = TENSOR_STRUCTURE_BY_ROLE[
            model_output_role
        ]

        _validate_batched_tensor_structure(
            tensor=runtime_result,
            tensor_structure=model_output_structure,
            tensor_description=(
                f"Composite model output {model_output_name!r}"
            ),
            expected_batch_size=expected_batch_size,
        )

        return {
            model_output_name: runtime_result,
        }

    if not isinstance(runtime_result, Mapping):
        raise TypeError(
            f"Composite assembly step {assembly_step.step_id!r} invokes "
            f"component ID {assembly_step.component_id!r}, which must "
            "return a mapping according to its resolved contract; got "
            f"{type(runtime_result).__name__}."
        )

    expected_result_keys = set(result_bindings)
    received_result_keys = set(runtime_result)

    missing_result_keys = sorted(
        expected_result_keys - received_result_keys,
        key=repr,
    )
    unexpected_result_keys = sorted(
        received_result_keys - expected_result_keys,
        key=repr,
    )

    if missing_result_keys or unexpected_result_keys:
        problems: list[str] = []

        if missing_result_keys:
            problems.append(
                f"missing result keys {missing_result_keys}"
            )

        if unexpected_result_keys:
            problems.append(
                f"unexpected result keys {unexpected_result_keys}"
            )

        raise ValueError(
            f"Composite assembly step {assembly_step.step_id!r} returned "
            "a mapping that does not match its resolved runtime-result "
            f"contract: {'; '.join(problems)}."
        )

    model_outputs_from_assembly_step: dict[
        str,
        torch.Tensor,
    ] = {}

    for result_key, model_output_name in result_bindings.items():
        result_tensor = runtime_result[result_key]

        if not isinstance(result_tensor, torch.Tensor):
            raise TypeError(
                f"Composite assembly step {assembly_step.step_id!r} "
                f"returned {result_key!r} as "
                f"{type(result_tensor).__name__}; every mapped runtime "
                "result must be a torch.Tensor."
            )

        model_output_role = (
            model_spec.declarations.model_output_roles_by_name[
                model_output_name
            ]
        )
        model_output_structure = TENSOR_STRUCTURE_BY_ROLE[
            model_output_role
        ]

        _validate_batched_tensor_structure(
            tensor=result_tensor,
            tensor_structure=model_output_structure,
            tensor_description=(
                f"Composite model output {model_output_name!r}"
            ),
            expected_batch_size=expected_batch_size,
        )

        model_outputs_from_assembly_step[
            model_output_name
        ] = result_tensor

    return model_outputs_from_assembly_step


def _get_sample_image_input_name(
    model_spec: CompositeModelSpec,
) -> str:
    """Return the declaration name carrying the primary sample image."""

    sample_image_input_names = [
        model_input_name
        for model_input_name, model_input_role
        in model_spec.declarations.model_input_roles_by_name.items()
        if model_input_role == "sample_image"
    ]

    if len(sample_image_input_names) != 1:
        raise ValueError(
            "CompositeModel requires exactly one declared model input "
            "with role 'sample_image'; found "
            f"{len(sample_image_input_names)}."
        )

    return sample_image_input_names[0]


def _extract_model_inputs_from_batch(
    *,
    batch: Mapping[str, Any],
    model_spec: CompositeModelSpec,
) -> dict[str, torch.Tensor]:
    """Extract declared model inputs from one complete runtime batch."""

    if not isinstance(batch, Mapping):
        raise TypeError(
            "CompositeModel expects each batch to be a mapping, got "
            f"{type(batch).__name__}."
        )

    required_batch_field_names = set(
        model_spec.declarations.model_input_roles_by_name
    )
    required_batch_field_names.update(
        model_spec.declarations.batch_metadata_roles_by_name
    )

    missing_batch_field_names = sorted(
        required_batch_field_names - set(batch),
        key=repr,
    )

    if missing_batch_field_names:
        raise KeyError(
            "CompositeModel batch is missing declared fields "
            f"{missing_batch_field_names}. Available batch fields: "
            f"{tuple(batch)}."
        )

    model_inputs: dict[str, torch.Tensor] = {}

    for model_input_name in (
        model_spec.declarations.model_input_roles_by_name
    ):
        model_input = batch[model_input_name]

        if not isinstance(model_input, torch.Tensor):
            raise TypeError(
                f"Composite model input {model_input_name!r} must be a "
                f"torch.Tensor, got {type(model_input).__name__}."
            )

        model_inputs[model_input_name] = model_input

    return model_inputs


def _validate_scalar_loss_result(
    raw_loss: Any,
    *,
    loss_role: str,
    configured_loss_name: str,
) -> torch.Tensor:
    """Validate the runtime result returned by one loss component."""

    loss_description = (
        f"Composite {loss_role.replace('_', ' ')} loss "
        f"{configured_loss_name!r}"
    )

    if not isinstance(raw_loss, torch.Tensor):
        raise TypeError(
            f"{loss_description} must return a torch.Tensor, got "
            f"{type(raw_loss).__name__}."
        )

    if raw_loss.ndim != 0:
        raise ValueError(
            f"{loss_description} must return a scalar tensor, got shape "
            f"{tuple(raw_loss.shape)}."
        )

    return raw_loss


def _validate_batched_tensor_structure(
    *,
    tensor: torch.Tensor,
    tensor_structure: TensorStructure,
    tensor_description: str,
    expected_batch_size: int | None = None,
) -> int:
    """Validate one tensor's batched runtime structure."""

    tensor_shape = tuple(tensor.shape)

    if tensor_structure == "scalar":
        structure_is_valid = (
            tensor.ndim == 1
            or (
                tensor.ndim == 2
                and tensor.shape[1] == 1
            )
        )
        expected_shape_description = "[batch] or [batch, 1]"

    elif tensor_structure == "vector":
        structure_is_valid = tensor.ndim == 2
        expected_shape_description = "[batch, features]"

    elif tensor_structure == "image":
        structure_is_valid = tensor.ndim == 4
        expected_shape_description = "[batch, channels, height, width]"

    else:
        raise RuntimeError(
            f"Unsupported Composite tensor structure "
            f"{tensor_structure!r}."
        )

    if not structure_is_valid:
        raise ValueError(
            f"{tensor_description} has structure "
            f"{tensor_structure!r}, so its runtime shape must follow "
            f"{expected_shape_description}; got shape {tensor_shape}."
        )

    batch_size = tensor.shape[0]

    if (
        expected_batch_size is not None
        and batch_size != expected_batch_size
    ):
        raise ValueError(
            f"{tensor_description} has batch size {batch_size}, but the "
            f"Composite model sample input has batch size "
            f"{expected_batch_size}."
        )

    return batch_size


def _get_loss_module(
    *,
    loss_modules_by_role: nn.ModuleDict,
    loss_role: str,
    loss_name: str,
) -> nn.Module:
    """Retrieve one instantiated loss module by its role and name."""

    loss_modules_for_role = loss_modules_by_role[loss_role]

    if not isinstance(loss_modules_for_role, nn.ModuleDict):
        raise RuntimeError(
            f"CompositeModel loss role {loss_role!r} does not contain "
            "a registered ModuleDict."
        )

    return loss_modules_for_role[loss_name]