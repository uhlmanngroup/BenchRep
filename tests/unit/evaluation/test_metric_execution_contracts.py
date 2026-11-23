"""Check registered metric outputs and the recoverable/fatal failure boundary."""

import re

import numpy as np
import pytest

from benchrep.assembly.registries.core import Registry
from benchrep.evaluation.metrics import EvaluationMetric
from benchrep.evaluation.metrics.execution import execute_metric_group
from benchrep.evaluation.utils import RecoverableEvaluationStepError


def _registry(**metrics):
    registry = Registry("test metric", entry_type=EvaluationMetric)
    for name, metric in metrics.items():
        registry.register(name, metric)
    return registry


def test_scalar_mapping_preserves_names_order_values_and_parameters():
    def summary(values, *, offset):
        return {"maximum": np.max(values) + offset, 7: np.min(values)}

    results, failures = execute_metric_group(
        registry=_registry(summary=EvaluationMetric(summary, "scalar_mapping")),
        canonical_metric_names=["summary"],
        metric_positional_args=(np.array([1.0, 4.0]),),
        metric_kwargs_by_name={"summary": {"offset": 2.0}},
    )

    assert failures == {}
    assert results == {
        "summary": {
            "kind": "scalar_mapping",
            "keys": ["maximum", "7"],
            "values": [6.0, 1.0],
        }
    }
    assert all(type(value) is float for value in results["summary"]["values"])


@pytest.mark.parametrize(
    ("kind", "value", "message"),
    [
        pytest.param("scalar", True, "real numeric scalar", id="boolean-scalar"),
        pytest.param("scalar", np.nan, "non-finite", id="nonfinite-scalar"),
        pytest.param("scalar", np.array([1, 2]), "scalar", id="array-as-scalar"),
        pytest.param(
            "scalar_mapping", [1, 2], "must return a mapping", id="not-mapping"
        ),
        pytest.param("scalar_mapping", {}, "empty mapping", id="empty-mapping"),
        pytest.param(
            "scalar_mapping",
            {1: 1.0, "1": 2.0},
            "duplicate result key",
            id="colliding-keys",
        ),
        pytest.param(
            "scalar_mapping",
            {"ok": 1.0, "bad": np.inf},
            "bad.bad.*non-finite",
            id="invalid-mapping-value",
        ),
        pytest.param("vector", [[1, 2]], "one-dimensional", id="matrix-as-vector"),
        pytest.param("vector", [1], "Expected 2, got 1", id="axis-length-mismatch"),
        pytest.param("vector", [1j, 2j], "real numeric", id="complex-vector"),
        pytest.param("vector", [1, np.inf], "non-finite", id="nonfinite-vector"),
        pytest.param(
            "vector_mapping",
            {"ok": [1, 2], "bad": [3]},
            "Expected 2, got 1",
            id="invalid-nested-vector",
        ),
    ],
)
def test_invalid_result_fails_only_its_metric(kind, value, message):
    calls = []

    def good():
        calls.append("good")
        return 0.5

    registry = _registry(
        bad=EvaluationMetric(
            lambda: value,
            kind,
            vector_axis="feature" if kind in {"vector", "vector_mapping"} else None,
        ),
        good=EvaluationMetric(good, "scalar"),
    )
    results, failures = execute_metric_group(
        registry=registry,
        canonical_metric_names=["bad", "good"],
        metric_positional_args=(),
        axis_labels_by_name={"feature": ["left", "right"]},
    )

    assert calls == ["good"]
    assert results == {"good": {"kind": "scalar", "value": 0.5}}
    assert set(failures) == {"bad"}
    assert re.search(message, failures["bad"])


def test_vector_mapping_preserves_axis_and_result_alignment():
    registry = _registry(
        summary=EvaluationMetric(
            lambda: {"mean": np.array([1.0, 2.0]), "spread": [3.0, 4.0]},
            "vector_mapping",
            vector_axis="feature",
        ),
    )
    results, failures = execute_metric_group(
        registry=registry,
        canonical_metric_names=["summary"],
        metric_positional_args=(),
        axis_labels_by_name={"feature": ["second", "first"]},
    )
    assert failures == {}
    assert results["summary"] == {
        "kind": "vector_mapping",
        "keys": ["mean", "spread"],
        "axis": {"name": "feature", "labels": ["second", "first"]},
        "values": [[1.0, 2.0], [3.0, 4.0]],
    }


@pytest.mark.parametrize(
    "recoverable", [True, False], ids=["recoverable", "unexpected"]
)
def test_callable_failure_continues_only_when_explicitly_recoverable(recoverable):
    error = (RecoverableEvaluationStepError if recoverable else ValueError)(
        "cannot compute"
    )
    calls = []

    def broken():
        raise error

    def later():
        calls.append("later")
        return 2.0

    kwargs = dict(
        registry=_registry(
            broken=EvaluationMetric(broken, "scalar"),
            later=EvaluationMetric(later, "scalar"),
        ),
        canonical_metric_names=["broken", "later"],
        metric_positional_args=(),
    )
    if recoverable:
        results, failures = execute_metric_group(**kwargs)
        assert results == {"later": {"kind": "scalar", "value": 2.0}}
        assert failures == {"broken": "cannot compute"}
        assert calls == ["later"]
    else:
        with pytest.raises(
            RuntimeError, match="broken.*ValueError.*cannot compute"
        ) as caught:
            execute_metric_group(**kwargs)
        assert caught.value.__cause__ is error
        assert calls == []


@pytest.mark.parametrize(
    ("labels", "error_type", "message"),
    [(None, RuntimeError, "No labels"), ({"feature": "ab"}, TypeError, "not a string")],
)
def test_invalid_axis_metadata_is_fatal_before_metric_execution(
    labels, error_type, message
):
    calls = []

    def vector():
        calls.append("vector")
        return [1, 2]

    with pytest.raises(error_type, match=message):
        execute_metric_group(
            registry=_registry(vector=EvaluationMetric(vector, "vector", "feature")),
            canonical_metric_names=["vector"],
            metric_positional_args=(),
            axis_labels_by_name=labels,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("kind", "axis", "message"),
    [
        ("vector", None, "requires vector_axis"),
        ("scalar", "feature", "does not accept vector_axis"),
    ],
)
def test_metric_declaration_rejects_inconsistent_axis_contract(kind, axis, message):
    with pytest.raises(ValueError, match=message):
        EvaluationMetric(lambda: 1.0, kind, axis)
