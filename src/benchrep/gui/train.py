"""Training wizard entrypoint; reusable form behavior lives in shared/forms.py."""

from __future__ import annotations

from pathlib import Path

from dash import Dash

from benchrep.gui.rules.train import prepare_training_draft, training_sections
from benchrep.gui.shared.forms import build_workflow_app
from benchrep.interfaces.model_families import (
    COMPOSITE_FAMILY,
    ModelFamilySpec,
)
from benchrep.utils.workflow_overrides import TrainingOverrides
from benchrep.utils.workflow_utils import TrainingConfigInput


def create_training_app(
    *,
    model_family: ModelFamilySpec = COMPOSITE_FAMILY,
    config: TrainingConfigInput = None,
    overrides: TrainingOverrides | None = None,
) -> Dash:
    """Build the training wizard from a base config and optional overrides."""
    overrides = overrides if overrides is not None else TrainingOverrides()
    initial = prepare_training_draft(
        model_family=model_family, config=config, overrides=overrides
    )
    app = Dash(
        __name__,
        assets_folder=str(Path(__file__).parent / "assets"),
        title="BenchRep — Training configuration",
        update_title=None,
        suppress_callback_exceptions=True,
    )
    return build_workflow_app(
        app=app,
        stage="training",
        initial=initial,
        sections=training_sections(
            model_family=model_family, overrides=overrides, initial=initial
        ),
        title="Training workflow configuration",
        subtitle="Configure a training run using BenchRep's schemas.",
    )


if __name__ == "__main__":
    create_training_app().run(host="127.0.0.1", port=8050, debug=False)
