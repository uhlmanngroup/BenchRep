from benchrep.workflows.train import TrainingWorkflowResult, train_ae, train_vae
from benchrep.workflows.predict import PredictionWorkflowResult, predict_ae, predict_vae
from benchrep.workflows.evaluate import EvaluationWorkflowResult, evaluate


__all__ = [
    "TrainingWorkflowResult",
    "train_ae",
    "train_vae",
    "PredictionWorkflowResult",
    "predict_ae",
    "predict_vae",
    "EvaluationWorkflowResult",
    "evaluate",
]