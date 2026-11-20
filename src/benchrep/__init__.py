from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("benchrep")
except PackageNotFoundError:
    __version__ = "0+unknown"

from benchrep.workflows.train import train_ae, train_vae, train_composite, train_yoyo
from benchrep.workflows.predict import (
    predict_ae,
    predict_vae,
    predict_composite,
    predict_yoyo,
)
from benchrep.workflows.evaluate import evaluate
from benchrep.discovery.registry import (
    list_registries,
    list_registered_components,
    inspect_registry,
)
from benchrep.discovery.config import inspect_config
from benchrep.utils.workflow_overrides import (
    TrainingOverrides,
    PredictionOverrides,
    EvaluationOverrides,
)
from benchrep.assembly.config import update_config
from benchrep.assembly.schemas.parsing import (
    ConfigContext,
    TrainingConfigContext,
    PredictionConfigContext,
    EvaluationConfigContext,
)
from benchrep.gui import configure_workflow


__all__ = [
    "__version__",
    "train_ae",
    "train_vae",
    "train_composite",
    "train_yoyo",
    "predict_ae",
    "predict_vae",
    "predict_composite",
    "predict_yoyo",
    "evaluate",
    "list_registries",
    "list_registered_components",
    "inspect_registry",
    "inspect_config",
    "TrainingOverrides",
    "PredictionOverrides",
    "EvaluationOverrides",
    "update_config",
    "ConfigContext",
    "TrainingConfigContext",
    "PredictionConfigContext",
    "EvaluationConfigContext",
    "configure_workflow",
]