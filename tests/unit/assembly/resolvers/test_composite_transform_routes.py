from __future__ import annotations

import pytest

from benchrep.assembly.resolvers.training_config_resolver import (
    _resolve_transform_pipelines,
)
from benchrep.assembly.resolvers.prediction_config_resolver import (
    _resolve_explicit_prediction_transform_routes,
)
from benchrep.assembly.schemas import (
    CompositeModelDeclarationsConfig,
    TrainingTransformPipelineConfig,
    TrainingTransformStepConfig,
    PredictionTransformPipelineConfig,
    PredictionTransformStepConfig,
)
from benchrep.interfaces.model_families import COMPOSITE_FAMILY


def _resolve_route(
    *,
    stage: str,
    multiple_samples: bool,
    input_name: str | None,
    output_name: str | None,
) -> tuple[str | None, str | None]:
    declarations = CompositeModelDeclarationsConfig(
        expects={
            "label": "categorical_prediction_target_scalar",
            "morphology": "sample_image",
            "skeleton": (
                "sample_image" if multiple_samples else "condition_image"
            ),
            "mask": "condition_image",
        },
        produces={"embedding": "embedding_vector"},
    )

    if stage == "training":
        resolved = _resolve_transform_pipelines(
            [
                TrainingTransformPipelineConfig(
                    input=input_name,
                    output=output_name,
                    steps=[
                        TrainingTransformStepConfig(
                            name="to_dtype",
                            apply_to=["training", "validation"],
                        ),
                    ],
                ),
            ],
            model_family=COMPOSITE_FAMILY,
            declarations_config=declarations,
            datamodule_overridden=False,
        )
    else:
        resolved = _resolve_explicit_prediction_transform_routes(
            [
                PredictionTransformPipelineConfig(
                    input=input_name,
                    output=output_name,
                    steps=[
                        PredictionTransformStepConfig(name="to_dtype"),
                    ],
                ),
            ],
            model_family=COMPOSITE_FAMILY,
            declarations_config=declarations,
        )

    assert len(resolved) == 1
    return resolved[0].input, resolved[0].output


@pytest.mark.parametrize("stage", ["training", "prediction"])
@pytest.mark.parametrize(
    ("multiple_samples", "input_name", "output_name", "expected"),
    [
        pytest.param(
            False, None, None,
            ("morphology", "morphology"),
            id="single-sample-defaults",
        ),
        pytest.param(
            False, None, "skeleton",
            ("morphology", "skeleton"),
            id="single-sample-explicit-output",
        ),
        pytest.param(
            True, "skeleton", None,
            ("skeleton", "skeleton"),
            id="multiple-samples-default-output",
        ),
        pytest.param(
            True, "morphology", "skeleton",
            ("morphology", "skeleton"),
            id="multiple-samples-explicit-route",
        ),
        pytest.param(
            True, "mask", None,
            ("mask", "mask"),
            id="explicit-condition-image-input",
        ),
    ],
)
def test_composite_transform_route_resolution(
    stage: str,
    multiple_samples: bool,
    input_name: str | None,
    output_name: str | None,
    expected: tuple[str, str],
) -> None:
    assert _resolve_route(
        stage=stage,
        multiple_samples=multiple_samples,
        input_name=input_name,
        output_name=output_name,
    ) == expected


@pytest.mark.parametrize("stage", ["training", "prediction"])
@pytest.mark.parametrize("output_name", [None, "skeleton"])
def test_multiple_sample_images_require_explicit_transform_input(
    stage: str,
    output_name: str | None,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"transform_pipelines\[0\]\.input.*must be provided",
    ):
        _resolve_route(
            stage=stage,
            multiple_samples=True,
            input_name=None,
            output_name=output_name,
        )


@pytest.mark.parametrize("stage", ["training", "prediction"])
@pytest.mark.parametrize(
    ("input_name", "output_name", "invalid_field"),
    [
        ("missing", None, "input"),
        ("label", None, "input"),
        ("morphology", "missing", "output"),
        ("morphology", "label", "output"),
    ],
)
def test_explicit_transform_routes_require_declared_images(
    stage: str,
    input_name: str,
    output_name: str | None,
    invalid_field: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"transform_pipelines\[0\]\.{invalid_field}",
    ):
        _resolve_route(
            stage=stage,
            multiple_samples=True,
            input_name=input_name,
            output_name=output_name,
        )