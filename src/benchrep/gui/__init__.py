"""Optional graphical configuration wizard."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from benchrep.assembly.schemas.training_config_schema import TrainingConfig
    from benchrep.assembly.schemas.prediction_config_schema import PredictionConfig
    from benchrep.assembly.schemas.evaluation_config_schema import EvaluationConfig
    from benchrep.gui.shared.results import ConfiguredWorkflow
    from benchrep.interfaces.model_families import ModelFamilySpec
    from benchrep.utils.workflow_overrides import (
        TrainingOverrides,
        PredictionOverrides,
        EvaluationOverrides,
    )
    from benchrep.utils.workflow_utils import (
        TrainingConfigInput,
        PredictionConfigInput,
        EvaluationConfigInput,
    )


@overload
def configure_workflow(
    *,
    stage: Literal["training"] = "training",
    model_family: ModelFamilySpec | None = None,
    config: TrainingConfigInput = None,
    overrides: TrainingOverrides | None = None,
    compatibility_policy: Literal["error"] = "error",
    export_directory: Path | str = "wizard_configs",
    port: int = 8050,
    open_browser: bool = True,
) -> ConfiguredWorkflow[TrainingConfig, TrainingOverrides] | None:
    ...


@overload
def configure_workflow(
    *,
    stage: Literal["prediction"],
    model_family: ModelFamilySpec | None = None,
    config: PredictionConfigInput = None,
    overrides: PredictionOverrides | None = None,
    compatibility_policy: Literal["error", "warn"] = "error",
    export_directory: Path | str = "wizard_configs",
    port: int = 8050,
    open_browser: bool = True,
) -> ConfiguredWorkflow[PredictionConfig, PredictionOverrides] | None:
    ...


@overload
def configure_workflow(
    *,
    stage: Literal["evaluation"],
    model_family: None = None,
    config: EvaluationConfigInput = None,
    overrides: EvaluationOverrides | None = None,
    compatibility_policy: Literal["error"] = "error",
    export_directory: Path | str = "wizard_configs",
    port: int = 8050,
    open_browser: bool = True,
) -> ConfiguredWorkflow[EvaluationConfig, EvaluationOverrides] | None:
    ...


def configure_workflow(
    *,
    stage: Literal["training", "prediction", "evaluation"] = "training",
    model_family: ModelFamilySpec | None = None,
    config: TrainingConfigInput | PredictionConfigInput | EvaluationConfigInput = None,
    overrides: TrainingOverrides
    | PredictionOverrides
    | EvaluationOverrides
    | None = None,
    compatibility_policy: Literal["error", "warn"] = "error",
    export_directory: Path | str = "wizard_configs",
    port: int = 8050,
    open_browser: bool = True,
) -> (
    ConfiguredWorkflow[TrainingConfig, TrainingOverrides]
    | ConfiguredWorkflow[PredictionConfig, PredictionOverrides]
    | ConfiguredWorkflow[EvaluationConfig, EvaluationOverrides]
    | None
):
    """Run a local wizard until Save & finish or Cancel; return None on cancellation.

    Saved YAML preserves authored omissions and includes config components.
    The returned overrides retain runtime objects, but omit baked-in components.
    Closing the browser alone does not cancel; use Cancel or Ctrl+C in Python.
    Training, prediction, and evaluation are supported. GUI dependencies are imported lazily.
    """
    if stage not in {"training", "prediction", "evaluation"}:
        raise ValueError("Choose the training, prediction, or evaluation wizard.")
    import threading
    import webbrowser

    from werkzeug.serving import (
        make_server,
        ThreadedWSGIServer,
        WSGIRequestHandler,
    )

    from benchrep.interfaces.model_families import COMPOSITE_FAMILY
    from benchrep.utils.workflow_overrides import (
        TrainingOverrides,
        PredictionOverrides,
        EvaluationOverrides,
    )

    if stage != "prediction" and compatibility_policy != "error":
        raise ValueError(
            "compatibility_policy is only configurable for prediction."
        )

    if stage == "training":
        from benchrep.assembly.schemas.training_config_schema import TrainingConfig
        from benchrep.gui.train import create_training_app

        if overrides is not None and not isinstance(overrides, TrainingOverrides):
            raise TypeError("training requires TrainingOverrides.")
        if config is not None and not isinstance(config, (Path, str, TrainingConfig)):
            raise TypeError("training requires a path or TrainingConfig.")

        app = create_training_app(
            config=config,
            overrides=overrides,
            model_family=model_family or COMPOSITE_FAMILY,
            export_directory=export_directory,
        )

    elif stage == "prediction":
        from benchrep.assembly.schemas.prediction_config_schema import PredictionConfig
        from benchrep.gui.predict import create_prediction_app

        if overrides is not None and not isinstance(overrides, PredictionOverrides):
            raise TypeError("prediction requires PredictionOverrides.")
        if config is not None and not isinstance(config, (Path, str, PredictionConfig)):
            raise TypeError("prediction requires a path or PredictionConfig.")

        app = create_prediction_app(
            config=config,
            overrides=overrides,
            model_family=model_family or COMPOSITE_FAMILY,
            compatibility_policy=compatibility_policy,
            export_directory=export_directory,
        )

    else:
        from benchrep.assembly.schemas.evaluation_config_schema import EvaluationConfig
        from benchrep.gui.evaluate import create_evaluation_app

        if model_family is not None:
            raise ValueError(
                "Evaluation uses artifacts and does not select a model family."
            )
        if overrides is not None and not isinstance(overrides, EvaluationOverrides):
            raise TypeError("evaluation requires EvaluationOverrides.")
        if config is not None and not isinstance(config, (Path, str, EvaluationConfig)):
            raise TypeError("evaluation requires a path or EvaluationConfig.")

        app = create_evaluation_app(
            config=config,
            overrides=overrides,
            export_directory=export_directory,
        )

    class WizardRequestHandler(WSGIRequestHandler):
        """Log failed HTTP requests while keeping routine traffic quiet."""

        def log_request(
                self,
                code: int | str = "-",
                size: int | str = "-",
        ) -> None:
            try:
                status = int(code)
            except (TypeError, ValueError):
                super().log_request(code, size)
                return

            if status >= 400:
                super().log_request(code, size)

    server = make_server(
        "127.0.0.1",
        port,
        app.server,
        threaded=True,
        request_handler=WizardRequestHandler,
    )

    assert isinstance(server, ThreadedWSGIServer)
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
