"""Evaluation wizard entrypoint."""

from __future__ import annotations

from pathlib import Path

from dash import Dash

from benchrep.gui.rules.evaluate import (
    EvaluationSourceContext,
    evaluation_review_session,
    evaluation_sections,
    prepare_evaluation_draft,
)
from benchrep.gui.shared.forms import build_workflow_app
from benchrep.utils.workflow_overrides import EvaluationOverrides
from benchrep.utils.workflow_utils import EvaluationConfigInput


def create_evaluation_app(
    *,
    config: EvaluationConfigInput = None,
    overrides: EvaluationOverrides | None = None,
    export_directory: Path | str = "wizard_configs",
) -> Dash:
    """Build an evaluation wizard without loading image or representation arrays."""
    overrides = overrides if overrides is not None else EvaluationOverrides()
    initial = prepare_evaluation_draft(config=config, overrides=overrides)
    context = EvaluationSourceContext(overrides=overrides)
    if context.has_source(initial.initial_data):
        context.load(initial.initial_data)
    app = Dash(
        __name__,
        assets_folder=str(Path(__file__).parent / "assets"),
        title="BenchRep — Evaluation configuration",
        update_title=None,
        suppress_callback_exceptions=True,
    )
    review = evaluation_review_session(context=context)
    app.wizard_session = review
    return build_workflow_app(
        app=app,
        stage="evaluation",
        initial=initial,
        sections=evaluation_sections(context=context),
        review=review,
        title="Evaluation configuration",
        subtitle="Build, validate, and export a configuration for your evaluation run",
        export_directory=str(export_directory),
        export_filename="evaluation.yaml",
    )


if __name__ == "__main__":
    create_evaluation_app().run(host="127.0.0.1", port=8050, debug=False)
