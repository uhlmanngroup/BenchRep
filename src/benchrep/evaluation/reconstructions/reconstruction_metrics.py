from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from benchrep.evaluation.reconstructions.data import validate_reconstruction_arrays


def mean_absolute_error(inputs: ArrayLike, reconstructions: ArrayLike) -> float:
    input_array, reconstruction_array = validate_reconstruction_arrays(
        inputs=inputs,
        reconstructions=reconstructions,
    )
    return float(np.mean(np.abs(input_array - reconstruction_array)))


def mean_squared_error(inputs: ArrayLike, reconstructions: ArrayLike) -> float:
    input_array, reconstruction_array = validate_reconstruction_arrays(
        inputs=inputs,
        reconstructions=reconstructions,
    )
    return float(np.mean((input_array - reconstruction_array) ** 2))


def root_mean_squared_error(inputs: ArrayLike, reconstructions: ArrayLike) -> float:
    return float(np.sqrt(mean_squared_error(inputs, reconstructions)))


def max_absolute_error(inputs: ArrayLike, reconstructions: ArrayLike) -> float:
    input_array, reconstruction_array = validate_reconstruction_arrays(
        inputs=inputs,
        reconstructions=reconstructions,
    )
    return float(np.max(np.abs(input_array - reconstruction_array)))
