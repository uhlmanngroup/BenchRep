from __future__ import annotations

from typing import Any

from benchrep.interfaces.model_families import (
    ModelFamilySpec,
)


def validate_external_model(model: Any, model_family: ModelFamilySpec) -> None:
    """Validate user-provided model override at workflow entrypoints"""
    expected_model_class = model_family.model_base_class

    if not isinstance(model, expected_model_class):
        raise TypeError(
            f"Model override for family {model_family.name!r} must be an "
            f"instance of `{expected_model_class.__name__}`. "
            "Custom models should subclass the appropriate BenchRep family base "
            "class, e.g. `BenchRepAutoencoderModel` or `BenchRepVAEModel`. "
            f"Got {type(model).__name__}."
        )