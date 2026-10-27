"""Optional graphical configuration wizard."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from benchrep.assembly.schemas.training_config_schema import TrainingConfig
    from benchrep.assembly.schemas.prediction_config_schema import PredictionConfig
    from benchrep.gui.shared.results import ConfiguredWorkflow
    from benchrep.interfaces.model_families import ModelFamilySpec
    from benchrep.utils.workflow_overrides import TrainingOverrides, PredictionOverrides
    from benchrep.utils.workflow_utils import TrainingConfigInput, PredictionConfigInput


def configure_workflow(
    *,
    stage: Literal["training", "prediction"] = "training",
    model_family: ModelFamilySpec | None = None,
    config: TrainingConfigInput | PredictionConfigInput = None,
    overrides: TrainingOverrides | PredictionOverrides | None = None,
    compatibility_policy: Literal["error", "warn"] = "error",
    export_directory: Path | str = "wizard_configs",
    port: int = 8050,
    open_browser: bool = True,
) -> (
    ConfiguredWorkflow[
        TrainingConfig | PredictionConfig, TrainingOverrides | PredictionOverrides
    ]
    | None
):
    """Run a local wizard until Save & finish or Cancel; return None on cancellation.

    Saved YAML preserves authored omissions and includes config components.
    The returned overrides retain runtime objects, but omit baked-in components.
    Closing the browser alone does not cancel; use Cancel or Ctrl+C in Python.
    Training and prediction are supported. GUI dependencies are imported lazily.
    """
    if stage not in {"training", "prediction"}:
        raise ValueError("Choose the training or prediction wizard.")
    import threading
    import webbrowser

    from werkzeug.serving import make_server

    from benchrep.gui.train import create_training_app
    from benchrep.interfaces.model_families import COMPOSITE_FAMILY

    from benchrep.utils.workflow_overrides import TrainingOverrides, PredictionOverrides

    expected = TrainingOverrides if stage == "training" else PredictionOverrides
    if overrides is not None and not isinstance(overrides, expected):
        raise TypeError(f"{stage} requires {expected.__name__}.")
    kwargs = {}
    factory = create_training_app
    if stage == "prediction":
        from benchrep.gui.predict import create_prediction_app

        factory = create_prediction_app
        kwargs["compatibility_policy"] = compatibility_policy
    elif compatibility_policy != "error":
        raise ValueError("compatibility_policy is only configurable for prediction.")

    app = factory(
        model_family=model_family or COMPOSITE_FAMILY,
        config=config,
        overrides=overrides,
        export_directory=export_directory,
        **kwargs,
    )
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    server.daemon_threads = False
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"BenchRep config wizard: {url}", flush=True)
    try:
        if open_browser:
            webbrowser.open(url)
        app.wizard_session.done.wait()
    except KeyboardInterrupt:
        with app.wizard_session.lock:
            app.wizard_session.done.set()
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
    return app.wizard_session.result
