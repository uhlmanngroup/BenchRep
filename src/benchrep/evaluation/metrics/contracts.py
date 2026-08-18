from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal, get_args


MetricResultKind = Literal[
    "scalar",
    "vector",
    "scalar_mapping",
    "vector_mapping",
]


_VALID_RESULT_KINDS: Final[frozenset[str]] = frozenset(
    get_args(MetricResultKind)
)

_VECTOR_RESULT_KINDS: Final[frozenset[str]] = frozenset(
    {
        "vector",
        "vector_mapping",
    }
)


@dataclass(frozen=True)
class EvaluationMetric:
    """Registered evaluation metric and its declared result contract."""

    fn: Callable[..., Any]
    result_kind: MetricResultKind
    vector_axis: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError(
                "EvaluationMetric.fn must be callable, got "
                f"{type(self.fn).__name__}."
            )

        if not isinstance(self.result_kind, str):
            raise TypeError(
                "EvaluationMetric.result_kind must be a string, got "
                f"{type(self.result_kind).__name__}."
            )

        if self.result_kind not in _VALID_RESULT_KINDS:
            raise ValueError(
                f"Unknown metric result kind {self.result_kind!r}. "
                f"Available kinds: {tuple(sorted(_VALID_RESULT_KINDS))}."
            )

        if self.vector_axis is not None:
            if not isinstance(self.vector_axis, str):
                raise TypeError(
                    "EvaluationMetric.vector_axis must be a string or None, "
                    f"got {type(self.vector_axis).__name__}."
                )

            vector_axis = self.vector_axis.strip()

            if not vector_axis:
                raise ValueError(
                    "EvaluationMetric.vector_axis must be a non-empty string."
                )

            object.__setattr__(
                self,
                "vector_axis",
                vector_axis,
            )

        if (
            self.result_kind in _VECTOR_RESULT_KINDS
            and self.vector_axis is None
        ):
            raise ValueError(
                f"Metric result kind {self.result_kind!r} requires "
                "vector_axis."
            )

        if (
            self.result_kind not in _VECTOR_RESULT_KINDS
            and self.vector_axis is not None
        ):
            raise ValueError(
                f"Metric result kind {self.result_kind!r} does not accept "
                "vector_axis."
            )