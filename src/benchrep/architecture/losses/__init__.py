from benchrep.architecture.losses.base import LossTerm
from benchrep.architecture.losses.reconstruction import (
    MSEReconstructionLoss,
    MAEReconstructionLoss,
)
from benchrep.architecture.losses.regularization import GaussianKLDivergenceLoss
from benchrep.architecture.losses.contrastive import (
    NTXentContrastiveLoss,
    TripletMarginContrastiveLoss,
)
from benchrep.architecture.losses.classification import CrossEntropyClassificationLoss
from benchrep.architecture.losses.regression import MSERegressionLoss
from benchrep.architecture.losses.custom_objective import BaseCustomObjectiveLoss
from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
    LossTensorPort,
)
from benchrep.architecture.losses.weight_schedules import (
    LossWeightSchedule,
    LinearSchedule,
    CosineSchedule,
    SigmoidSchedule,
    ExponentialSchedule,
    PiecewiseLinearSchedule,
    PiecewiseConstantSchedule,
    CyclicSchedule,
)

__all__ = [
    "LossTerm",
    "MSEReconstructionLoss",
    "MAEReconstructionLoss",
    "GaussianKLDivergenceLoss",
    "NTXentContrastiveLoss",
    "TripletMarginContrastiveLoss",
    "CrossEntropyClassificationLoss",
    "MSERegressionLoss",
    "BaseCustomObjectiveLoss",
    "LossComponent",
    "LossTensorPort",
    "LossWeightSchedule",
    "LinearSchedule",
    "CosineSchedule",
    "SigmoidSchedule",
    "ExponentialSchedule",
    "PiecewiseLinearSchedule",
    "PiecewiseConstantSchedule",
    "CyclicSchedule",
]