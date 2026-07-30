from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


TransformCallable = Callable[[Any], Any]


@dataclass(frozen=True)
class TransformStep:
    """One named callable within a transform pipeline."""

    name: str
    transform: TransformCallable

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Transform step name must be non-empty.")

        if not callable(self.transform):
            raise TypeError(
                "Transform step must contain a callable, "
                f"got {type(self.transform).__name__}."
            )

    def run(self, value: Any) -> Any:
        """Apply the transform to the provided value."""
        return self.transform(value)


class TransformPipeline:
    """Apply an ordered sequence of transforms to an arbitrary value.

    Each step receives the preceding step's output, and the pipeline returns
    only the final result. The pipeline does not interpret, validate, or copy
    its input; callers determine what is transformed and whether cloning is
    required.

    Examples
    --------
    Transform one tensor::

        transformed_x = pipeline(x)

    Replace ``"x"`` while preserving the other sample fields::

        transformed_sample = {
            **sample,
            "x": pipeline(sample["x"]),
        }

    Add one augmented positive while retaining the existing ``"x"``::

        contrastive_sample = {
            **sample,
            "positive": pipeline(sample["x"].clone()),
        }

    Produce two independently augmented views::

        source = sample["x"]
        contrastive_sample = {
            **sample,
            "x": pipeline(source.clone()),
            "positive": pipeline(source.clone()),
        }
    """

    def __init__(self, steps: Sequence[TransformStep] = ()) -> None:
        self.steps = tuple(steps)

    def __call__(self, value: Any) -> Any:
        return self.run(value)

    def run(self, value: Any) -> Any:
        """Apply every transform step in its configured order."""
        for step in self.steps:
            value = step.run(value)

        return value

    def __len__(self) -> int:
        return len(self.steps)