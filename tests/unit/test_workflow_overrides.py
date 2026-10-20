import pytest

from benchrep import (
    PredictionOverrides,
    TrainingOverrides,
    predict_ae,
    predict_composite,
    predict_vae,
    train_ae,
    train_composite,
    train_vae,
)
from benchrep.interfaces.models import (
    BenchRepAutoencoderModel,
    BenchRepVAEModel,
)


@pytest.mark.parametrize(
    ("entrypoint", "overrides", "message"),
    [
        pytest.param(
            train_ae,
            TrainingOverrides(model=BenchRepVAEModel),
            "BenchRepAutoencoderModel",
            id="train-ae-rejects-vae",
        ),
        pytest.param(
            train_vae,
            TrainingOverrides(model=BenchRepAutoencoderModel),
            "BenchRepVAEModel",
            id="train-vae-rejects-ae",
        ),
        pytest.param(
            predict_ae,
            PredictionOverrides(model=BenchRepVAEModel),
            "BenchRepAutoencoderModel",
            id="predict-ae-rejects-vae",
        ),
        pytest.param(
            predict_vae,
            PredictionOverrides(model=BenchRepAutoencoderModel),
            "BenchRepVAEModel",
            id="predict-vae-rejects-ae",
        ),
        pytest.param(
            train_composite,
            TrainingOverrides(model=BenchRepAutoencoderModel),
            "Whole-model overrides",
            id="train-composite-rejects-model",
        ),
        pytest.param(
            predict_composite,
            PredictionOverrides(model=BenchRepAutoencoderModel),
            "Whole-model overrides",
            id="predict-composite-rejects-model",
        ),
    ],
)
def test_entrypoint_rejects_incompatible_model_override(
    entrypoint,
    overrides,
    message,
) -> None:
    # Family validation must reject the override before config loading.
    with pytest.raises(TypeError, match=message):
        entrypoint(config="nonexistent-config.yaml", overrides=overrides)