from __future__ import annotations

from dataclasses import fields, is_dataclass
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints
from typing_extensions import is_typeddict

from benchrep.interfaces.contracts import ContractKind
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


# Wrappers
def sanity_check_training_step_batch_annotation(
    *,
    model: Any,
    model_family: ModelFamilySpec,
    check_field_types: bool = True,
) -> None:
    sanity_check_method_parameter_annotation(
        model=model,
        method_name="training_step",
        parameter_name="batch",
        expected_parameter_type=model_family.expected_batch_type,
        expected_parameter_contract_kind=model_family.expected_batch_contract_kind,
        check_field_types=check_field_types,
    )


def sanity_check_predict_step_batch_annotation(
    *,
    model: Any,
    model_family: ModelFamilySpec,
    check_field_types: bool = True,
) -> None:
    sanity_check_method_parameter_annotation(
        model=model,
        method_name="predict_step",
        parameter_name="batch",
        expected_parameter_type=model_family.expected_batch_type,
        expected_parameter_contract_kind=model_family.expected_batch_contract_kind,
        check_field_types=check_field_types,
    )


def sanity_check_predict_step_return_annotation(
    *,
    model: Any,
    model_family: ModelFamilySpec,
    check_field_types: bool = True,
) -> None:
    sanity_check_method_return_annotation(
        model=model,
        method_name="predict_step",
        expected_return_type=model_family.expected_prediction_output_type,
        expected_return_contract_kind=(
            model_family.expected_prediction_output_contract_kind
        ),
        check_field_types=check_field_types,
    )


# Generics
def sanity_check_method_parameter_annotation(
    *,
    model: Any,
    method_name: str,
    parameter_name: str,
    expected_parameter_type: type[Any],
    expected_parameter_contract_kind: ContractKind,
    check_field_types: bool = True,
) -> None:
    method = getattr(type(model), method_name, None)
    if method is None:
        raise TypeError(
            f"Method parameter annotation sanity check failed: "
            f"Model `{type(model).__name__}` has no `{method_name}()` method."
        )

    hints = get_type_hints(method)
    declared_type = hints.get(parameter_name)

    _check_declared_contract_annotation(
        declared_type=declared_type,
        expected_type=expected_parameter_type,
        expected_contract_kind=expected_parameter_contract_kind,
        check_field_types=check_field_types,
        context=f"`{method_name}()` parameter `{parameter_name}`",
    )


def sanity_check_method_return_annotation(
    *,
    model: Any,
    method_name: str,
    expected_return_type: type[Any],
    expected_return_contract_kind: ContractKind,
    check_field_types: bool = True,
) -> None:
    """Inspect a declared method return contract without running the model."""

    method = getattr(type(model), method_name, None)
    if method is None:
        raise TypeError(
            f"Method return annotation sanity check failed: "
            f"Model `{type(model).__name__}` has no `{method_name}()` method."
        )

    hints = get_type_hints(method)
    declared_type = hints.get("return")

    _check_declared_contract_annotation(
        declared_type=declared_type,
        expected_type=expected_return_type,
        expected_contract_kind=expected_return_contract_kind,
        check_field_types=check_field_types,
        context=f"`{method_name}()` return type",
    )


def _check_declared_contract_annotation(
    *,
    declared_type: Any,
    expected_type: type[Any],
    expected_contract_kind: ContractKind,
    check_field_types: bool,
    context: str,
) -> None:
    if not _is_supported_contract_type(expected_type, expected_contract_kind):
        raise TypeError(
            f"Internal BenchRep error: expected contract `{expected_type!r}` "
            f"is not a valid {expected_contract_kind!r} contract type."
        )

    if declared_type is None:
        raise TypeError(
            f"Method annotation sanity check failed: "
            f"No annotation found for {context}. "
            f"BenchRep expected a declared type compatible with "
            f"`{expected_type.__name__}`."
        )

    if not _is_supported_contract_type(declared_type, expected_contract_kind):
        raise TypeError(
            f"Method annotation sanity check failed: "
            f"Declared {context} is not a {expected_contract_kind!r} contract type. "
            f"BenchRep expected compatibility with `{expected_type.__name__}`; "
            f"got {declared_type!r}."
        )

    expected_required_fields = _required_contract_fields(
        expected_type,
        expected_contract_kind,
    )
    declared_fields = _contract_field_types(
        declared_type,
        expected_contract_kind,
    )

    missing_fields = set(expected_required_fields) - set(declared_fields)

    if missing_fields:
        raise TypeError(
            f"Method annotation sanity check failed: "
            f"Declared {context} `{declared_type.__name__}` is missing required "
            f"field(s): {sorted(missing_fields)}. Expected compatibility with "
            f"`{expected_type.__name__}`."
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
            f"Method annotation sanity check failed: "
            f"Declared {context} `{declared_type.__name__}` has incompatible "
            f"annotations for required field(s): {details}. "
            f"Expected compatibility with `{expected_type.__name__}`."
        )


def _is_supported_contract_type(obj: Any, kind: ContractKind) -> bool:
    if kind == "dataclass":
        return _is_dataclass_type(obj)

    if kind == "typeddict":
        return _is_typeddict_type(obj)

    raise ValueError(f"Unsupported contract kind: {kind!r}")


def _required_contract_fields(cls: Any, kind: ContractKind) -> dict[str, Any]:
    if kind == "dataclass":
        return _non_nullable_dataclass_fields(cls)

    if kind == "typeddict":
        return _required_typeddict_fields(cls)

    raise ValueError(f"Unsupported contract kind: {kind!r}")


def _contract_field_types(cls: Any, kind: ContractKind) -> dict[str, Any]:
    if kind == "dataclass":
        return _dataclass_field_types(cls)

    if kind == "typeddict":
        return _typeddict_field_types(cls)

    raise ValueError(f"Unsupported contract kind: {kind!r}")


def _required_typeddict_fields(cls: Any) -> dict[str, Any]:
    type_hints = get_type_hints(cls)
    required_keys = getattr(cls, "__required_keys__", frozenset())

    return {
        key: type_hints[key]
        for key in required_keys
        if key in type_hints
    }


def _typeddict_field_types(cls: Any) -> dict[str, Any]:
    return get_type_hints(cls)


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


def _is_typeddict_type(obj: Any) -> bool:
    return isinstance(obj, type) and is_typeddict(obj)


def _allows_none(annotation: Any) -> bool:
    if annotation is None or annotation is type(None):
        return True

    origin = get_origin(annotation)

    if origin in {Union, UnionType}:
        return type(None) in get_args(annotation)

    return False