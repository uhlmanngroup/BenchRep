from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torchvision.transforms import v2


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
    """An ordered transform sequence routed between two sample fields.

    ``input_key`` and ``output_key`` describe sample-dictionary routing.
    The pipeline itself only applies its steps; ``TransformedDataset`` handles
    input lookup, cloning, validation, and output assignment.
    """

    def __init__(
        self,
        *,
        input_key: str,
        output_key: str,
        steps: Sequence[TransformStep] = (),
    ) -> None:
        if not input_key.strip():
            raise ValueError("Transform pipeline input key must be nonempty.")

        if not output_key.strip():
            raise ValueError("Transform pipeline output key must be nonempty.")

        self.input_key = input_key
        self.output_key = output_key
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


def create_to_dtype_transform(
    dtype: str | torch.dtype,
    scale: bool = False,
) -> v2.ToDtype:
    """Create a torchvision ToDtype transform from a dtype object or name."""

    if isinstance(dtype, str):
        dtype_name = dtype.removeprefix("torch.")
        resolved_dtype = getattr(torch, dtype_name, None)

        if not isinstance(resolved_dtype, torch.dtype):
            raise ValueError(
                f"Unknown PyTorch dtype {dtype!r}."
            )

        dtype = resolved_dtype

    elif not isinstance(dtype, torch.dtype):
        raise TypeError(
            "`dtype` must be a PyTorch dtype or dtype name, "
            f"got {type(dtype).__name__}."
        )

    return v2.ToDtype(
        dtype=dtype,
        scale=scale,
    )