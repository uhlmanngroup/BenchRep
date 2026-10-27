"""Prediction wizard entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from dash import Dash

from benchrep.gui.rules.predict import (
    PredictionSourceContext,
    prepare_prediction_draft,
    prediction_review_session,
    prediction_sections,
)
from benchrep.gui.shared.forms import build_workflow_app
from benchrep.interfaces.model_families import COMPOSITE_FAMILY, ModelFamilySpec
from benchrep.utils.workflow_overrides import PredictionOverrides
from benchrep.utils.workflow_utils import PredictionConfigInput


def create_prediction_app(
    *,
    model_family: ModelFamilySpec = COMPOSITE_FAMILY,
    config: PredictionConfigInput = None,
    overrides: PredictionOverrides | None = None,
    compatibility_policy: Literal["error", "warn"] = "error",
    export_directory: Path | str = "wizard_configs",
) -> Dash:
    """Build a prediction wizard without constructing models or loading weights."""
    overrides = overrides if overrides is not None else PredictionOverrides()
    initial = prepare_prediction_draft(
        model_family=model_family, config=config, overrides=overrides
    )
    context = PredictionSourceContext(model_family=model_family, overrides=overrides)
    if context.path(initial.initial_data) is not None:
        context.load(initial.initial_data)
    app = Dash(
        __name__,
        assets_folder=str(Path(__file__).parent / "assets"),
        title="BenchRep — Prediction configuration",
        update_title=None,
        suppress_callback_exceptions=True,
    )
    review = prediction_review_session(
        context=context, compatibility_policy=compatibility_policy
    )
    app.wizard_session = review
    return build_workflow_app(
        app=app,
        stage="prediction",
        initial=initial,
        sections=prediction_sections(context=context),
        title="Prediction configuration",
        subtitle="Build, validate, and export a configuration for your prediction run",
        review=review,
        export_directory=str(export_directory),
        export_filename="prediction.yaml",
    )


if __name__ == "__main__":
    create_prediction_app().run(host="127.0.0.1", port=8050, debug=False)
