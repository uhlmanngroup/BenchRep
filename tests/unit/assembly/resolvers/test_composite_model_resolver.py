from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
)
from benchrep.architecture.losses.custom_objective import (
    BaseCustomObjectiveLoss,
)
from benchrep.assembly.registries.core import (
    LOSS_REGISTRIES_BY_ROLE,
    LossRegistry,
)
from benchrep.assembly.resolvers.composite_model_resolver import (
    CompositeModelSpec,
    resolve_composite_model_config,
)
from benchrep.assembly.schemas.composite_model_config_schema import (
    CompositeModelAssemblyStepConfig,
    CompositeModelComponentConfig,
    CompositeModelDeclarationsConfig,
)
from benchrep.assembly.schemas.training_config_schema import (
    SupportedLossRole,
    TrainingLossTermConfig,
)


@dataclass
class CompositeResolverArguments:
    """Store one complete set of inputs to the Composite model resolver."""

    declarations_config: CompositeModelDeclarationsConfig
    components_config: dict[str, CompositeModelComponentConfig]
    assembly_config: dict[str, CompositeModelAssemblyStepConfig]
    losses_config: dict[
        SupportedLossRole,
        dict[str, TrainingLossTermConfig],
    ]

    def resolve(self) -> CompositeModelSpec:
        return resolve_composite_model_config(
            declarations_config=self.declarations_config,
            components_config=self.components_config,
            assembly_config=self.assembly_config,
            losses_config=self.losses_config,
        )


class CanonicalOnlyReconstructionLoss(nn.Module):
    """Provide a valid canonical signature without a Composite contract."""

    def forward(
        self,
        *,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mean((reconstruction - target) ** 2)


class ResolverCustomObjectiveLoss(BaseCustomObjectiveLoss):
    """Provide the fixed full-context custom-objective interface."""

    def forward(
        self,
        *,
        batch: Mapping[str, Any],
        model_output: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        del batch
        return model_output["embedding"].sum() * 0.0


def _minimal_autoencoder_arguments() -> CompositeResolverArguments:
    return CompositeResolverArguments(
        declarations_config=CompositeModelDeclarationsConfig(
            expects={
                "image": "sample_image",
            },
            produces={
                "embedding": "embedding_vector",
                "reconstruction": "reconstruction_image",
            },
        ),
        components_config={
            "encoder": CompositeModelComponentConfig(
                kind="encoder",
                name="mlp",
                params={
                    "input_shape": (1, 8, 8),
                    "output_dim": 4,
                },
            ),
            "decoder": CompositeModelComponentConfig(
                kind="decoder",
                name="mlp",
                params={
                    "input_dim": 4,
                    "output_shape": (1, 8, 8),
                },
            ),
        },
        assembly_config={
            "encode": CompositeModelAssemblyStepConfig(
                component="encoder",
                inputs={
                    "x": "expects.image",
                },
                outputs="produces.embedding",
            ),
            "decode": CompositeModelAssemblyStepConfig(
                component="decoder",
                inputs={
                    "z": "produces.embedding",
                },
                outputs="produces.reconstruction",
            ),
        },
        losses_config={
            "reconstruction": {
                "mse": TrainingLossTermConfig(
                    composite_wiring={
                        "reconstruction": "produces.reconstruction",
                        "target": "expects.image",
                    },
                ),
            },
        },
    )


def test_resolves_variational_graph_mapping_results_and_loss_aliases() -> None:
    resolver_arguments = CompositeResolverArguments(
        declarations_config=CompositeModelDeclarationsConfig(
            expects={
                "image": "sample_image",
            },
            batch_metadata={
                "sample_id": "index",
            },
            produces={
                "encoder_features": "embedding_vector",
                "latent_sample": "embedding_vector",
                "latent_mean": "embedding_vector",
                "latent_log_variance": "continuous_auxiliary_vector",
                "reconstruction": "reconstruction_image",
            },
        ),
        components_config={
            "encoder": CompositeModelComponentConfig(
                kind="encoder",
                name="fc",
                params={
                    "input_shape": (1, 8, 8),
                    "output_dim": 6,
                },
            ),
            "latent_head": CompositeModelComponentConfig(
                kind="head",
                name="variational",
                params={
                    "in_features": 6,
                    "latent_dim": 3,
                },
            ),
            "decoder": CompositeModelComponentConfig(
                kind="decoder",
                name="fc",
                params={
                    "input_dim": 3,
                    "output_shape": (1, 8, 8),
                },
            ),
        },
        assembly_config={
            "encode": CompositeModelAssemblyStepConfig(
                component="encoder",
                inputs={
                    "x": "expects.image",
                },
                outputs="produces.encoder_features",
            ),
            "parameterize_latent": CompositeModelAssemblyStepConfig(
                component="latent_head",
                inputs={
                    "x": "produces.encoder_features",
                },
                outputs={
                    "z_sample": "produces.latent_sample",
                    "z_mu": "produces.latent_mean",
                    "z_logvar": "produces.latent_log_variance",
                },
            ),
            "decode": CompositeModelAssemblyStepConfig(
                component="decoder",
                inputs={
                    "z": "produces.latent_sample",
                },
                outputs="produces.reconstruction",
            ),
        },
        losses_config={
            "reconstruction": {
                "mean_squared_error": TrainingLossTermConfig(
                    composite_wiring={
                        "reconstruction": "produces.reconstruction",
                        "target": "expects.image",
                    },
                ),
            },
            "regularization": {
                "kl": TrainingLossTermConfig(
                    weight=0.001,
                    composite_wiring={
                        "z_mu": "produces.latent_mean",
                        "z_logvar": "produces.latent_log_variance",
                    },
                ),
            },
        },
    )

    model_spec = resolver_arguments.resolve()

    assert model_spec.declarations.model_input_roles_by_name == {
        "image": "sample_image",
    }
    assert model_spec.declarations.batch_metadata_roles_by_name == {
        "sample_id": "index",
    }
    assert model_spec.components_by_id["encoder"].registry_entry_name == "mlp"
    assert model_spec.components_by_id["latent_head"].registry_entry_name == (
        "gaussian_variational"
    )
    assert model_spec.components_by_id["decoder"].registry_entry_name == "mlp"

    step_ids_by_dependency_level = {
        dependency_level: tuple(
            assembly_step.step_id
            for assembly_step in assembly_steps
        )
        for dependency_level, assembly_steps
        in model_spec.assembly_steps_by_dependency_level.items()
    }
    assert step_ids_by_dependency_level == {
        0: ("encode",),
        1: ("parameterize_latent",),
        2: ("decode",),
    }

    latent_step = model_spec.assembly_steps_by_dependency_level[1][0]
    assert latent_step.results_to_model_outputs == {
        "z_sample": "latent_sample",
        "z_mu": "latent_mean",
        "z_logvar": "latent_log_variance",
    }

    loss_specs_by_role = {
        loss_term.loss_role: loss_term
        for loss_term in model_spec.loss_terms
    }
    assert loss_specs_by_role["reconstruction"].registry_entry_name == "mse"
    assert loss_specs_by_role["regularization"].registry_entry_name == (
        "gaussian_kl"
    )
    assert loss_specs_by_role["regularization"].inputs_from_model_outputs == {
        "z_mu": "latent_mean",
        "z_logvar": "latent_log_variance",
    }


def test_orders_branches_and_preserves_shared_component_ids() -> None:
    resolver_arguments = CompositeResolverArguments(
        declarations_config=CompositeModelDeclarationsConfig(
            expects={
                "anchor_image": "sample_image",
                "positive_image": "positive_image",
                "negative_image": "negative_image",
            },
            produces={
                "anchor_embedding": "embedding_vector",
                "positive_embedding": "embedding_vector",
                "negative_embedding": "embedding_vector",
                "anchor_projection": "projection_vector",
            },
        ),
        components_config={
            "shared_encoder": CompositeModelComponentConfig(
                kind="encoder",
                name="mlp",
                params={
                    "input_shape": (1, 8, 8),
                    "output_dim": 4,
                },
            ),
            "projector": CompositeModelComponentConfig(
                kind="head",
                name="mlp",
                params={
                    "input_dim": 4,
                    "output_dim": 2,
                },
            ),
        },
        assembly_config={
            "encode_anchor": CompositeModelAssemblyStepConfig(
                component="shared_encoder",
                inputs={"x": "expects.anchor_image"},
                outputs="produces.anchor_embedding",
            ),
            "encode_positive": CompositeModelAssemblyStepConfig(
                component="shared_encoder",
                inputs={"x": "expects.positive_image"},
                outputs="produces.positive_embedding",
            ),
            "encode_negative": CompositeModelAssemblyStepConfig(
                component="shared_encoder",
                inputs={"x": "expects.negative_image"},
                outputs="produces.negative_embedding",
            ),
            "project_anchor": CompositeModelAssemblyStepConfig(
                component="projector",
                inputs={"x": "produces.anchor_embedding"},
                outputs="produces.anchor_projection",
            ),
        },
        losses_config={
            "contrastive": {
                "triplet_margin": TrainingLossTermConfig(
                    composite_wiring={
                        "anchor": "produces.anchor_embedding",
                        "positive": "produces.positive_embedding",
                        "negative": "produces.negative_embedding",
                    },
                ),
            },
        },
    )

    model_spec = resolver_arguments.resolve()

    assert tuple(model_spec.assembly_steps_by_dependency_level) == (0, 1)
    assert tuple(
        step.step_id
        for step in model_spec.assembly_steps_by_dependency_level[0]
    ) == (
        "encode_anchor",
        "encode_positive",
        "encode_negative",
    )
    assert tuple(
        step.step_id
        for step in model_spec.assembly_steps_by_dependency_level[1]
    ) == ("project_anchor",)

    shared_encoder_invocations = []

    for assembly_steps in (
        model_spec.assembly_steps_by_dependency_level.values()
    ):
        for assembly_step in assembly_steps:
            if assembly_step.component_id == "shared_encoder":
                shared_encoder_invocations.append(assembly_step.step_id)

    assert shared_encoder_invocations == [
        "encode_anchor",
        "encode_positive",
        "encode_negative",
    ]
    assert set(model_spec.components_by_id) == {
        "shared_encoder",
        "projector",
    }


def test_rejects_model_output_produced_by_multiple_steps() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.assembly_config["encode_again"] = (
        CompositeModelAssemblyStepConfig(
            component="encoder",
            inputs={"x": "expects.image"},
            outputs="produces.embedding",
        )
    )

    with pytest.raises(
        ValueError,
        match="produced by both assembly steps",
    ):
        resolver_arguments.resolve()


def test_rejects_declared_model_output_without_producing_step() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.declarations_config = (
        CompositeModelDeclarationsConfig(
            expects={"image": "sample_image"},
            produces={
                "embedding": "embedding_vector",
                "reconstruction": "reconstruction_image",
                "unused_projection": "projection_vector",
            },
        )
    )

    with pytest.raises(
        ValueError,
        match="not produced by any assembly step",
    ):
        resolver_arguments.resolve()


def test_rejects_configured_component_without_assembly_invocation() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.components_config["unused_head"] = (
        CompositeModelComponentConfig(
            kind="head",
            name="mlp",
            params={
                "input_dim": 4,
                "output_dim": 2,
            },
        )
    )

    with pytest.raises(
        ValueError,
        match="not used by any assembly step",
    ):
        resolver_arguments.resolve()


def test_rejects_cyclic_assembly_dependencies() -> None:
    resolver_arguments = CompositeResolverArguments(
        declarations_config=CompositeModelDeclarationsConfig(
            expects={
                "image": "sample_image",
                "label": "categorical_prediction_target_scalar",
            },
            produces={
                "prediction": "categorical_prediction_vector",
                "projection": "projection_vector",
            },
        ),
        components_config={
            "head": CompositeModelComponentConfig(
                kind="head",
                name="mlp",
                params={
                    "input_dim": 4,
                    "output_dim": 4,
                },
            ),
        },
        assembly_config={
            "predict": CompositeModelAssemblyStepConfig(
                component="head",
                inputs={"x": "produces.projection"},
                outputs="produces.prediction",
            ),
            "project": CompositeModelAssemblyStepConfig(
                component="head",
                inputs={"x": "produces.prediction"},
                outputs="produces.projection",
            ),
        },
        losses_config={
            "classification": {
                "cross_entropy": TrainingLossTermConfig(
                    composite_wiring={
                        "prediction": "produces.prediction",
                        "target": "expects.label",
                    },
                ),
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="dependency cycle",
    ):
        resolver_arguments.resolve()


def test_rejects_component_input_with_incompatible_tensor_structure() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.declarations_config = (
        CompositeModelDeclarationsConfig(
            expects={
                "image": "sample_image",
                "label": "categorical_prediction_target_scalar",
            },
            produces={
                "embedding": "embedding_vector",
                "reconstruction": "reconstruction_image",
            },
        )
    )
    resolver_arguments.assembly_config["encode"] = (
        CompositeModelAssemblyStepConfig(
            component="encoder",
            inputs={"x": "expects.label"},
            outputs="produces.embedding",
        )
    )

    with pytest.raises(
        ValueError,
        match="tensor structure 'scalar'",
    ):
        resolver_arguments.resolve()


def test_rejects_loss_input_with_incompatible_semantic_role() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.losses_config["reconstruction"]["mse"] = (
        TrainingLossTermConfig(
            composite_wiring={
                "reconstruction": "produces.embedding",
                "target": "expects.image",
            },
        )
    )

    with pytest.raises(
        ValueError,
        match="supports only",
    ):
        resolver_arguments.resolve()


def test_rejects_ordinary_loss_without_composite_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_only_registry = LossRegistry(
        "resolver test reconstruction loss",
        canonical_runtime_input_names=(
            "reconstruction",
            "target",
        ),
    )
    canonical_only_registry.register(
        "canonical_only",
        LossComponent(CanonicalOnlyReconstructionLoss),
    )
    monkeypatch.setitem(
        LOSS_REGISTRIES_BY_ROLE,
        "reconstruction",
        canonical_only_registry,
    )

    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.losses_config = {
        "reconstruction": {
            "canonical_only": TrainingLossTermConfig(
                composite_wiring={
                    "reconstruction": "produces.reconstruction",
                    "target": "expects.image",
                },
            ),
        },
    }

    with pytest.raises(
        ValueError,
        match="does not declare a Composite runtime contract",
    ):
        resolver_arguments.resolve()


def test_resolves_custom_objective_with_fixed_context_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_objective_registry = LossRegistry(
        "resolver test custom objective loss",
        canonical_runtime_input_names=(
            "batch",
            "model_output",
        ),
        required_component_base=BaseCustomObjectiveLoss,
        explicit_composite_runtime_inputs_supported=False,
    )
    custom_objective_registry.register(
        "resolver_custom_objective",
        LossComponent(ResolverCustomObjectiveLoss),
    )
    monkeypatch.setitem(
        LOSS_REGISTRIES_BY_ROLE,
        "custom_objective",
        custom_objective_registry,
    )

    resolver_arguments = CompositeResolverArguments(
        declarations_config=CompositeModelDeclarationsConfig(
            expects={"image": "sample_image"},
            produces={"embedding": "embedding_vector"},
        ),
        components_config={
            "encoder": CompositeModelComponentConfig(
                kind="encoder",
                name="mlp",
                params={
                    "input_shape": (1, 8, 8),
                    "output_dim": 4,
                },
            ),
        },
        assembly_config={
            "encode": CompositeModelAssemblyStepConfig(
                component="encoder",
                inputs={"x": "expects.image"},
                outputs="produces.embedding",
            ),
        },
        losses_config={
            "custom_objective": {
                "resolver_custom_objective": TrainingLossTermConfig(),
            },
        },
    )

    model_spec = resolver_arguments.resolve()

    assert len(model_spec.loss_terms) == 1
    assert model_spec.loss_terms[0].context_inputs == {
        "batch": "batch",
        "model_output": "model_output",
    }
    assert model_spec.loss_terms[0].inputs_from_model_inputs == {}
    assert model_spec.loss_terms[0].inputs_from_model_outputs == {}


def test_rejects_composite_wiring_for_custom_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_objective_registry = LossRegistry(
        "resolver test custom objective loss",
        canonical_runtime_input_names=(
            "batch",
            "model_output",
        ),
        required_component_base=BaseCustomObjectiveLoss,
        explicit_composite_runtime_inputs_supported=False,
    )
    custom_objective_registry.register(
        "resolver_custom_objective",
        LossComponent(ResolverCustomObjectiveLoss),
    )
    monkeypatch.setitem(
        LOSS_REGISTRIES_BY_ROLE,
        "custom_objective",
        custom_objective_registry,
    )

    resolver_arguments = CompositeResolverArguments(
        declarations_config=CompositeModelDeclarationsConfig(
            expects={"image": "sample_image"},
            produces={"embedding": "embedding_vector"},
        ),
        components_config={
            "encoder": CompositeModelComponentConfig(
                kind="encoder",
                name="mlp",
                params={
                    "input_shape": (1, 8, 8),
                    "output_dim": 4,
                },
            ),
        },
        assembly_config={
            "encode": CompositeModelAssemblyStepConfig(
                component="encoder",
                inputs={"x": "expects.image"},
                outputs="produces.embedding",
            ),
        },
        losses_config={
            "custom_objective": {
                "resolver_custom_objective": TrainingLossTermConfig(
                    composite_wiring={
                        "batch": "expects.image",
                    },
                ),
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="must not be configured for a custom objective",
    ):
        resolver_arguments.resolve()

def test_rejects_mapping_outputs_for_single_tensor_result() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.assembly_config["encode"] = (
        CompositeModelAssemblyStepConfig(
            component="encoder",
            inputs={"x": "expects.image"},
            outputs={
                "embedding": "produces.embedding",
            },
        )
    )

    with pytest.raises(
        ValueError,
        match="returns one unnamed tensor",
    ):
        resolver_arguments.resolve()


def test_rejects_single_output_for_mapping_result() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.declarations_config = (
        CompositeModelDeclarationsConfig(
            expects={"image": "sample_image"},
            produces={
                "embedding": "embedding_vector",
                "latent_sample": "embedding_vector",
                "latent_mean": "embedding_vector",
                "latent_log_variance": "continuous_auxiliary_vector",
                "reconstruction": "reconstruction_image",
            },
        )
    )
    resolver_arguments.components_config["latent_head"] = (
        CompositeModelComponentConfig(
            kind="head",
            name="gaussian_variational",
            params={
                "in_features": 4,
                "latent_dim": 2,
            },
        )
    )
    resolver_arguments.assembly_config["parameterize_latent"] = (
        CompositeModelAssemblyStepConfig(
            component="latent_head",
            inputs={"x": "produces.embedding"},
            outputs="produces.latent_sample",
        )
    )

    with pytest.raises(
        ValueError,
        match="returns a named mapping",
    ):
        resolver_arguments.resolve()


def test_rejects_missing_and_unexpected_mapping_result_keys() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.declarations_config = (
        CompositeModelDeclarationsConfig(
            expects={"image": "sample_image"},
            produces={
                "embedding": "embedding_vector",
                "latent_sample": "embedding_vector",
                "latent_mean": "embedding_vector",
                "reconstruction": "reconstruction_image",
            },
        )
    )
    resolver_arguments.components_config["latent_head"] = (
        CompositeModelComponentConfig(
            kind="head",
            name="gaussian_variational",
            params={
                "in_features": 4,
                "latent_dim": 2,
            },
        )
    )
    resolver_arguments.assembly_config["parameterize_latent"] = (
        CompositeModelAssemblyStepConfig(
            component="latent_head",
            inputs={"x": "produces.embedding"},
            outputs={
                "z_sample": "produces.latent_sample",
                "z_mu": "produces.latent_mean",
                "variance": "produces.latent_mean",
            },
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            r"missing keys \['z_logvar'\]; "
            r"unexpected keys \['variance'\]"
        ),
    ):
        resolver_arguments.resolve()


def test_rejects_missing_and_unexpected_component_input_keys() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.assembly_config["encode"] = (
        CompositeModelAssemblyStepConfig(
            component="encoder",
            inputs={
                "image": "expects.image",
            },
            outputs="produces.embedding",
        )
    )

    with pytest.raises(
        ValueError,
        match=r"missing keys \['x'\]; unexpected keys \['image'\]",
    ):
        resolver_arguments.resolve()


def test_rejects_missing_and_unexpected_loss_input_keys() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.losses_config["reconstruction"]["mse"] = (
        TrainingLossTermConfig(
            composite_wiring={
                "reconstruction": "produces.reconstruction",
                "source": "expects.image",
            },
        )
    )

    with pytest.raises(
        ValueError,
        match=r"missing keys \['target'\]; unexpected keys \['source'\]",
    ):
        resolver_arguments.resolve()


def test_rejects_invalid_component_constructor_parameters() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.components_config["encoder"] = (
        CompositeModelComponentConfig(
            kind="encoder",
            name="mlp",
            params={
                "input_shape": (1, 8, 8),
                "unknown_parameter": True,
            },
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid constructor parameters at "
            "`composite_model_components.encoder.params`"
        ),
    ):
        resolver_arguments.resolve()


def test_rejects_invalid_loss_constructor_parameters() -> None:
    resolver_arguments = _minimal_autoencoder_arguments()
    resolver_arguments.losses_config["reconstruction"]["mse"] = (
        TrainingLossTermConfig(
            params={"unknown_parameter": True},
            composite_wiring={
                "reconstruction": "produces.reconstruction",
                "target": "expects.image",
            },
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid constructor parameters at "
            "`losses.reconstruction.mse.params`"
        ),
    ):
        resolver_arguments.resolve()