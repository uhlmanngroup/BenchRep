from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import anndata as ad


# -------------------------
# Step specs
# -------------------------
@dataclass(frozen=True)
class AnnDataEvaluationStep:
    """A single AnnData-based evaluation step.

    The wrapped function must follow the BenchRep AnnData evaluation contract:

        AnnData -> AnnData

    The function may mutate the input AnnData in place, but it must still return
    the updated AnnData so steps can be chained by the pipeline.
    """

    name: str
    fn: Callable[..., ad.AnnData]
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def run(self, adata: ad.AnnData) -> ad.AnnData:
        """Run the step on an AnnData object."""

        if not self.enabled:
            return adata

        result = self.fn(adata, **dict(self.params))

        if not isinstance(result, ad.AnnData):
            raise TypeError(
                f"AnnData evaluation step {self.name!r} must return an AnnData "
                f"object, got {type(result).__name__}."
            )

        return result


@dataclass(frozen=True)
class ReconstructionEvaluationStep:
    """A single reconstruction-based evaluation step.

    Reconstruction steps use a looser contract than AnnData steps because
    reconstruction artifacts are not naturally stored in AnnData. The wrapped
    function receives the reconstruction input object plus step params and
    returns a mapping of outputs, metrics, or artifact references.
    """

    name: str
    fn: Callable[..., Mapping[str, Any]]
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def run(self, reconstruction_input: Any) -> dict[str, Any]:
        """Run the step on reconstruction input data."""

        if not self.enabled:
            return {}

        result = self.fn(reconstruction_input, **dict(self.params))

        if not isinstance(result, Mapping):
            raise TypeError(
                f"Reconstruction evaluation step {self.name!r} must return a "
                f"mapping, got {type(result).__name__}."
            )

        return dict(result)


# -------------------------
# Pipelines
# -------------------------
class AnnDataEvaluationPipeline:
    """Sequential pipeline for AnnData-based evaluation steps.

    The pipeline owns only orchestration: it runs enabled steps in order and
    passes the updated AnnData object from one step to the next. Actual
    evaluation logic should live in the step functions.
    """

    def __init__(
        self,
        steps: Sequence[AnnDataEvaluationStep],
    ) -> None:
        self.steps = list(steps)

    def run(self, adata: ad.AnnData) -> ad.AnnData:
        """Run all enabled AnnData evaluation steps."""

        for step in self.steps:
            adata = step.run(adata)

        return adata


class ReconstructionEvaluationPipeline:
    """Sequential pipeline for reconstruction-based evaluation steps.

    Each enabled step receives the same reconstruction input object. Step outputs
    are collected into a nested dictionary keyed by step name.
    """

    def __init__(
        self,
        steps: Sequence[ReconstructionEvaluationStep],
    ) -> None:
        self.steps = list(steps)

    def run(self, reconstruction_input: Any | None) -> dict[str, Any]:
        """Run all enabled reconstruction evaluation steps."""

        if reconstruction_input is None:
            return {}

        outputs: dict[str, Any] = {}

        for step in self.steps:
            step_output = step.run(reconstruction_input)

            if not step_output:
                continue

            if step.name in outputs:
                raise KeyError(
                    f"Reconstruction pipeline output already contains step "
                    f"{step.name!r}."
                )

            outputs[step.name] = step_output

        return outputs
