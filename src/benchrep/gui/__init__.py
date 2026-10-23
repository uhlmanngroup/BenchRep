"""Optional graphical configuration wizard."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from benchrep.assembly.schemas.training_config_schema import TrainingConfig
    from benchrep.gui.shared.results import ConfiguredWorkflow
    from benchrep.interfaces.model_families import ModelFamilySpec
    from benchrep.utils.workflow_overrides import TrainingOverrides
    from benchrep.utils.workflow_utils import TrainingConfigInput


def configure_workflow(
    *,
    stage: Literal["training"] = "training",
    model_family: ModelFamilySpec | None = None,
    config: TrainingConfigInput = None,
    overrides: TrainingOverrides | None = None,
    export_directory: Path | str = "wizard_configs",
    port: int = 8050,
    open_browser: bool = True,
) -> ConfiguredWorkflow[TrainingConfig, TrainingOverrides] | None:
    """Run a local wizard until Save & finish or Cancel; return None on cancellation.

    Saved YAML preserves authored omissions and includes config components.
    The returned overrides retain runtime objects, but omit baked-in components.
    Closing the browser alone does not cancel; use Cancel or Ctrl+C in Python.
    Only training is currently implemented. GUI dependencies are imported lazily.
    """
    if stage != "training":
        raise ValueError("Only the training wizard is currently implemented.")
    import threading
    import webbrowser

    from werkzeug.serving import make_server

    from benchrep.gui.train import create_training_app
    from benchrep.interfaces.model_families import COMPOSITE_FAMILY

    app = create_training_app(
        model_family=model_family or COMPOSITE_FAMILY,
        config=config,
        overrides=overrides,
        export_directory=export_directory,
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
