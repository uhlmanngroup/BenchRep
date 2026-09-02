from benchrep.architecture.heads.base import BaseHead
from benchrep.architecture.heads.variational import (
    GaussianVariationalHead,
)
from benchrep.architecture.heads.mlp import MLPHead

__all__ = [
    "BaseHead",
    "GaussianVariationalHead",
    "MLPHead",
]