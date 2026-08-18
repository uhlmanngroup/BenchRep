from __future__ import annotations

import warnings

import numpy as np
import pytest

from benchrep.evaluation.reconstructions.data import (
    ReconstructionEvaluationInput,
)
from benchrep.evaluation.reconstructions.error_maps import (
    compute_error_maps,
)


def _make_reconstruction_input() -> ReconstructionEvaluationInput:
    inputs = np.array(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[10.0, 12.0], [14.0, 16.0]],
            ]
        ],
        dtype=np.float32,
    )

    return ReconstructionEvaluationInput(
        inputs=inputs,
        reconstructions=inputs + 2.0,
        obs=None,
    )


def test_explicit_data_range_applies_only_to_global_normalization() -> None:
    reconstruction_input = _make_reconstruction_input()

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        results = compute_error_maps(
            reconstruction_input,
            kinds=[
                "normalized_absolute_global",
                "normalized_absolute_per_channel",
            ],
            data_range=10.0,
        )

    assert set(results) == {
        "normalized_absolute_global",
        "normalized_absolute_per_channel",
    }

    np.testing.assert_allclose(
        results["normalized_absolute_global"]["error_maps"],
        np.full_like(reconstruction_input.inputs, 0.2),
    )

    expected_per_channel = np.empty_like(reconstruction_input.inputs)
    expected_per_channel[:, 0] = 2.0 / 3.0
    expected_per_channel[:, 1] = 2.0 / 6.0

    np.testing.assert_allclose(
        results["normalized_absolute_per_channel"]["error_maps"],
        expected_per_channel,
    )

    assert (
        results["normalized_absolute_global"]["params"]["data_range"]
        == 10.0
    )
    assert (
        results["normalized_absolute_per_channel"]["params"]["data_range"]
        is None
    )


@pytest.mark.parametrize(
    "kind",
    [
        "absolute",
        "normalized_absolute_per_channel",
    ],
)
def test_unused_data_range_warns_and_is_ignored(kind: str) -> None:
    with pytest.warns(
        UserWarning,
        match="data_range.*ignored",
    ):
        results = compute_error_maps(
            _make_reconstruction_input(),
            kinds=[kind],
            data_range=10.0,
        )

    assert set(results) == {kind}
    assert results[kind]["params"]["data_range"] is None