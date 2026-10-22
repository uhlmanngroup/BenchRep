"""Validate original schema types, keeping structured error locations intact."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter, ValidationError


def validate_value(
    annotation: Any, data: Any, context: dict | None = None
) -> dict[tuple, str]:
    try:
        TypeAdapter(annotation).validate_python(data, context=context)
    except ValidationError as exc:
        errors: dict[tuple, str] = {}
        for error in exc.errors(include_url=False):
            path = tuple(error["loc"])
            message = error["msg"]
            errors[path] = f"{errors[path]}; {message}" if path in errors else message
        return errors
    return {}
