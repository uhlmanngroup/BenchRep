from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


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


def sanity_check_predict_step_return_annotation(
    *,
    model: Any,
    model_family: ModelFamilySpec,
    check_field_types: bool = True,
) -> None:
    """Inspect declared predict_step return contract without running the model.

    Required fields are inferred from the expected BenchRep prediction-output
    dataclass as fields whose annotation does not allow None. Nullable fields are
    treated as optional and do not need to exist on the declared output type.
    """

    expected_output_type = model_family.prediction_output_type
    assert _is_dataclass_type(expected_output_type)

    predict_step = type(model).predict_step
    hints = get_type_hints(predict_step)
    declared_output_type = hints.get("return")

    if declared_output_type is None:
        raise TypeError(
            "Prediction-output annotation sanity check failed: "
            "No return annotation found on external model `predict_step()`. "
            f"BenchRep expected a declared return type compatible with "
            f"`{expected_output_type.__name__}`."
        )

    if not _is_dataclass_type(declared_output_type):
        raise TypeError(
            "Prediction-output annotation sanity check failed: "
            "Declared `predict_step()` return type is not a dataclass type. "
            f"BenchRep expected a dataclass compatible with "
            f"`{expected_output_type.__name__}`; got {declared_output_type!r}."
        )

    expected_required_fields = _non_nullable_dataclass_fields(expected_output_type)
    declared_fields = _dataclass_field_types(declared_output_type)

    missing_fields = set(expected_required_fields) - set(declared_fields)

    if missing_fields:
        raise TypeError(
            f"Prediction-output annotation sanity check failed: "
            f"Declared `predict_step()` return type "
            f"`{declared_output_type.__name__}` is missing required non-nullable "
            f"field(s) for family {model_family.name!r}: {sorted(missing_fields)}. "
            f"Expected compatibility with `{expected_output_type.__name__}`."
        )

    if not check_field_types:
        return

    mismatched_fields = {
        name: (expected_required_fields[name], declared_fields[name])
        for name in expected_required_fields
        if declared_fields[name] != expected_required_fields[name]
    }

    if mismatched_fields:
        details = "; ".join(
            f"{name}: expected {expected!r}, got {actual!r}"
            for name, (expected, actual) in mismatched_fields.items()
        )

        raise TypeError(
            f"Prediction-output annotation sanity check failed: "
            f"Declared `predict_step()` return type "
            f"`{declared_output_type.__name__}` has incompatible annotations "
            f"for required non-nullable field(s): {details}. "
            f"Expected compatibility with `{expected_output_type.__name__}`."
        )


def _non_nullable_dataclass_fields(cls: Any) -> dict[str, Any]:
    type_hints = get_type_hints(cls)

    return {
        field.name: type_hints.get(field.name, field.type)
        for field in fields(cls)
        if not _allows_none(type_hints.get(field.name, field.type))
    }


def _dataclass_field_types(cls: Any) -> dict[str, Any]:
    type_hints = get_type_hints(cls)

    return {
        field.name: type_hints.get(field.name, field.type)
        for field in fields(cls)
    }


def _is_dataclass_type(obj: Any) -> bool:
    return isinstance(obj, type) and is_dataclass(obj)


def _allows_none(annotation: Any) -> bool:
    if annotation is None or annotation is type(None):
        return True

    origin = get_origin(annotation)

    if origin in {Union, UnionType}:
        return type(None) in get_args(annotation)

    return False