"""Serializable editor state, independent of mounted controls and validated models."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin
from uuid import uuid4

from pydantic import BaseModel, Tag, TypeAdapter
from pydantic.fields import FieldInfo


class _Missing(Enum):
    TOKEN = "missing"


MISSING = _Missing.TOKEN
FieldMode = Literal["value", "omit", "null", "empty"]


def json_value(value: Any) -> Any:
    return TypeAdapter(Any).dump_python(value, mode="json")


def initial_field_mode(value: Any = MISSING) -> FieldMode:
    if value is MISSING:
        return "omit"
    if value is None:
        return "null"
    if isinstance(value, (str, list, dict)) and not value:
        return "empty"
    return "value"


@dataclass(frozen=True, kw_only=True)
class WizardDraft:
    initial_data: dict[str, Any]
    supplied_sections: frozenset[str]


@dataclass(frozen=True)
class EditorSpec:
    """A structural description, derived from the original annotated type."""

    annotation: Any
    kind: str
    info: FieldInfo
    fields: dict[str, EditorSpec] = field(default_factory=dict)
    item: EditorSpec | None = None
    key_type: Any = str
    choices: tuple[Any, ...] = ()
    variants: tuple[EditorSpec, ...] = ()
    extra: bool = False
    schema_path: str | None = None


def base_type(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def editor_spec(annotation: Any, info: FieldInfo | None = None) -> EditorSpec:
    info = info if info is not None else FieldInfo.from_annotation(annotation)
    base = base_type(annotation)
    origin, args = get_origin(base), get_args(base)
    common = {"annotation": annotation, "info": info}
    if origin in (Union, UnionType):
        variants = tuple(editor_spec(a) for a in args if a is not type(None))
        if variants and all(variant.kind == "choice" for variant in variants):
            choices = []
            seen = set()
            for variant in variants:
                for value in variant.choices:
                    identity = (type(value), value)
                    if identity not in seen:
                        seen.add(identity)
                        choices.append(value)
            return EditorSpec(**common, kind="choice", choices=tuple(choices))
        if len(variants) == 1:
            child = variants[0]
            return EditorSpec(
                **common,
                kind=child.kind,
                fields=child.fields,
                item=child.item,
                key_type=child.key_type,
                choices=child.choices,
                variants=child.variants,
                extra=child.extra,
            )
        return EditorSpec(**common, kind="union", variants=variants)
    if isinstance(base, type) and issubclass(base, BaseModel):
        return EditorSpec(
            **common,
            kind="object",
            fields={
                name: replace(
                    editor_spec(f.rebuild_annotation(), f),
                    schema_path=f"{base.__name__}.{name}",
                )
                for name, f in base.model_fields.items()
            },
            extra=base.model_config.get("extra") == "allow",
        )
    if origin is list:
        return EditorSpec(**common, kind="list", item=editor_spec(args[0]))
    if origin is dict and len(args) == 2 and args[1] is not Any:
        return EditorSpec(
            **common, kind="mapping", item=editor_spec(args[1]), key_type=args[0]
        )
    if origin is Literal:
        return EditorSpec(**common, kind="choice", choices=args)
    if base is bool:
        return EditorSpec(**common, kind="choice", choices=(True, False))
    if base in (int, float):
        return EditorSpec(**common, kind="number")
    if base in (str, Path):
        return EditorSpec(**common, kind="text")
    # Untyped mappings, fixed tuples, and unsupported scalar types retain JSON editing.
    return EditorSpec(**common, kind="json")


def union_variant_details(
    spec: EditorSpec, index: int
) -> tuple[str, tuple[str, Any] | None]:
    """Return a schema-derived label and its fixed tag field, when unambiguous."""
    variant = spec.variants[index]
    annotation = variant.annotation
    tag = None
    while get_origin(annotation) is Annotated:
        annotation, *metadata = get_args(annotation)
        for item in metadata:
            if isinstance(item, Tag):
                tag = item.tag

    discriminator = spec.info.discriminator
    discriminator = getattr(discriminator, "discriminator", discriminator)
    fixed = None

    if isinstance(discriminator, str):
        child = variant.fields.get(discriminator)
        if child is not None and len(child.choices) == 1:
            fixed = (discriminator, child.choices[0])
    elif tag is not None:
        # Callable discriminators need not correspond to a field.
        # Infer one only when exactly one singleton Literal matches the tag.
        candidates = [
            (name, child.choices[0])
            for name, child in variant.fields.items()
            if len(child.choices) == 1 and child.choices[0] == tag
        ]
        if len(candidates) == 1:
            fixed = candidates[0]

    if tag is not None:
        label = str(tag)
    elif fixed is not None:
        label = str(fixed[1])
    elif variant.kind == "choice":
        label = " / ".join(str(value).lower() for value in variant.choices)
    else:
        base = base_type(variant.annotation)
        label = getattr(base, "__name__", str(base).replace("typing.", ""))

    return label, fixed


def empty_value(spec: EditorSpec) -> Any:
    if spec.kind == "union":
        return MISSING
    base = base_type(spec.annotation)
    if get_origin(base) in (Union, UnionType):
        non_null = tuple(a for a in get_args(base) if a is not type(None))
        if len(non_null) == 1:
            base = base_type(non_null[0])
    if spec.kind in ("object", "mapping") or base is dict or get_origin(base) is dict:
        return {}
    if spec.kind == "list" or base in (list, tuple) or get_origin(base) in (list, tuple):
        return []
    if spec.kind == "text" or (
        spec.kind == "choice" and any(isinstance(x, str) for x in spec.choices)
    ):
        return ""
    return MISSING


def choose_variant(spec: EditorSpec, value: Any) -> int:
    """Prefer tagged models, then successful validation, then matching raw shape."""
    if isinstance(value, dict):
        try:
            parsed = TypeAdapter(spec.annotation).validate_python(value)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, BaseModel):
            for index, variant in enumerate(spec.variants):
                if type(parsed) is base_type(variant.annotation):
                    return index
        for index, variant in enumerate(spec.variants):
            literals = {
                key: child.choices
                for key, child in variant.fields.items()
                if child.kind == "choice"
            }
            if (
                literals
                and any(key in value for key in literals)
                and all(
                    key not in value or value[key] in choices
                    for key, choices in literals.items()
                )
            ):
                return index
    if value is not MISSING and value is not None:
        for index, variant in enumerate(spec.variants):
            try:
                TypeAdapter(variant.annotation).validate_python(value, strict=True)
                return index
            except (ValueError, TypeError):
                pass
        for index, variant in enumerate(spec.variants):
            if (
                (
                    isinstance(value, dict)
                    and variant.kind in ("object", "mapping", "json")
                )
                or (isinstance(value, list) and variant.kind == "list")
                or (type(value) in (int, float) and variant.kind == "number")
                or (isinstance(value, str) and variant.kind == "text")
            ):
                return index
    return 0


def new_node(spec: EditorSpec, value: Any = MISSING) -> dict:
    node = {"id": uuid4().hex, "mode": initial_field_mode(value)}
    if spec.kind == "union":
        branch = choose_variant(spec, value)
        node.update(
            branch=branch,
            branches={str(branch): new_node(spec.variants[branch], value)},
        )
        # The outer mode owns omission/null/empty, the child owns its value editor.
        child = node["branches"][str(branch)]
        child["mode"] = "value"
        variant = spec.variants[branch]
        if (
                variant.kind == "choice"
                and len(variant.choices) == 1
                and (value is MISSING or value is None)
        ):
            child["value"] = deepcopy(variant.choices[0])
    elif spec.kind == "object":
        values = value if isinstance(value, dict) else {}
        node["fields"] = {
            key: new_node(child, values.get(key, MISSING))
            for key, child in spec.fields.items()
        }
        node["extra"] = {
            key: deepcopy(v) for key, v in values.items() if key not in spec.fields
        }
        node["extra_text"] = json.dumps(node["extra"], indent=2)
    elif spec.kind in ("list", "mapping"):
        entries = (
            enumerate(value)
            if spec.kind == "list" and isinstance(value, list)
            else (
                value.items()
                if spec.kind == "mapping" and isinstance(value, dict)
                else []
            )
        )
        node["entries"] = [
            {"key": str(key), "node": new_node(spec.item, v)} for key, v in entries
        ]
    elif spec.kind == "json":
        node["value"] = (
            "" if value is MISSING else json.dumps(value, indent=2, ensure_ascii=False)
        )
    elif spec.kind == "number":
        node["value"] = (
            "" if value is MISSING or value is None else deepcopy(value)
        )
    else:
        node["value"] = None if value is MISSING else deepcopy(value)
    return node


def active_spec(spec: EditorSpec, node: dict) -> EditorSpec:
    return spec.variants[node["branch"]] if spec.kind == "union" else spec


def collect_node(
    spec: EditorSpec, node: dict, path: tuple = ()
) -> tuple[Any, dict[tuple, str]]:
    """Collect without validation/coercion; preserve invalid editor text and key presence."""
    mode = node["mode"]
    if mode == "omit":
        return MISSING, {}
    if mode == "null":
        return None, {}
    if mode == "empty":
        value = empty_value(active_spec(spec, node))
        return (
            (value, {})
            if value is not MISSING
            else (None, {path: "This type has no empty representation."})
        )
    errors: dict[tuple, str] = {}
    if spec.kind == "union":
        key = str(node["branch"])
        return collect_node(spec.variants[node["branch"]], node["branches"][key], path)
    if spec.kind == "object":
        result = deepcopy(node["extra"])
        if spec.extra:
            try:
                extra = json.loads(node["extra_text"])
                if not isinstance(extra, dict):
                    raise TypeError("Expected a JSON object.")
                if set(extra) & set(spec.fields):
                    raise ValueError("Additional keys cannot replace declared fields.")
                result = extra
            except (ValueError, TypeError) as error:
                errors[path] = f"Additional fields: {error}"
        for key, child in spec.fields.items():
            value, issues = collect_node(child, node["fields"][key], (*path, key))
            errors.update(issues)
            if value is not MISSING:
                result[key] = value
        return result, errors
    if spec.kind in ("list", "mapping"):
        result = [] if spec.kind == "list" else {}
        for entry in node["entries"]:
            key = len(result) if spec.kind == "list" else entry["key"]
            value, issues = collect_node(spec.item, entry["node"], (*path, key))
            errors.update(issues)
            if value is MISSING:
                continue
            if spec.kind == "list":
                result.append(value)
            elif key in result:
                errors[(*path, key)] = (
                    "Duplicate mapping key; rename or remove one entry."
                )
            else:
                result[key] = value
        return result, errors
    if spec.kind == "json":
        try:
            return json.loads(node["value"]), {}
        except (ValueError, TypeError) as error:
            return node["value"], {path: f"Enter valid JSON: {error}"}
    return node["value"], {}


def find_node(spec: EditorSpec, node: dict, identity: str):
    """Find a stable UI identity independently of list positions or mapping names."""
    if node["id"] == identity:
        return spec, node
    children = []
    if spec.kind == "object":
        children = [(child, node["fields"][key]) for key, child in spec.fields.items()]
    elif spec.kind in ("list", "mapping"):
        children = [(spec.item, entry["node"]) for entry in node["entries"]]
    elif spec.kind == "union":
        children = [
            (spec.variants[int(key)], child) for key, child in node["branches"].items()
        ]
    for child_spec, child_node in children:
        found = find_node(child_spec, child_node, identity)
        if found:
            return found
    return None


def edit_node(
    spec: EditorSpec, root: dict, identity: str, action: str, value: Any
) -> bool:
    """Apply an interaction. Return whether the visible editor structure changed."""
    found = find_node(spec, root, identity)
    if not found:
        return False
    current_spec, node = found
    if action == "mode":
        changed = node["mode"] != value
        node["mode"] = value
        return changed
    if action == "value":
        node["value"] = (
            "" if current_spec.kind == "number" and value is None else value
        )
        return False
    if action == "extra":
        node["extra_text"] = value
        return False
    if action == "union_name":
        if value is None:
            return False

        name, branch = json.loads(value)
        if not isinstance(name, str) or type(branch) is not int:
            return False
        if not 0 <= branch < len(current_spec.variants):
            return False

        # Keep edits independently for each registered implementation,
        # including custom implementations sharing the same schema.
        cache = node.setdefault("named_branches", {})
        previous = node["branches"][str(node["branch"])]
        previous_name = previous["fields"]["name"]
        if previous_name["mode"] == "value":
            cache[previous_name["value"]] = deepcopy(previous)

        child = (
            deepcopy(cache[name])
            if name in cache
            else new_node(current_spec.variants[branch])
        )
        child["mode"] = "value"
        child["fields"]["name"].update(mode="value", value=name)

        node["branch"] = branch
        node["branches"][str(branch)] = child
        return True
    if action == "branch":
        if value is None:
            return False

        branch = int(value)
        if not 0 <= branch < len(current_spec.variants):
            return False

        changed = branch != node["branch"]
        node["branch"] = branch

        if str(branch) not in node["branches"]:
            child = new_node(current_spec.variants[branch])
            child["mode"] = "value"
            node["branches"][str(branch)] = child

        variant = current_spec.variants[branch]
        if variant.kind == "choice" and len(variant.choices) == 1:
            child = node["branches"][str(branch)]
            fixed_value = variant.choices[0]
            changed |= child.get("value") != fixed_value
            child.update(mode="value", value=deepcopy(fixed_value))
        _, fixed = union_variant_details(current_spec, branch)
        if fixed is not None:
            name, tag = fixed
            child = node["branches"][str(branch)]["fields"][name]
            changed |= child["mode"] != "value" or child.get("value") != tag
            child.update(mode="value", value=tag)

        return changed
    if action == "reset":
        info = current_spec.info
        if info.is_required():
            return False
        if info.default_factory_takes_validated_data:
            raise ValueError(
                "This default depends on validated sibling values; use Omit to let the schema compute it."
            )
        default = info.get_default(call_default_factory=True)
        replacement = new_node(current_spec, json_value(default))
        replacement["id"] = node["id"]
        node.clear()
        node.update(replacement)
        return True
    if action == "add":
        child = new_node(current_spec.item)
        child["mode"] = "value"
        node["entries"].append({"key": "", "node": child})
        return True
    return False


def edit_entry(
    spec: EditorSpec, root: dict, identity: str, action: str, value: Any
) -> bool:
    if spec.kind in ("list", "mapping"):
        for index, entry in enumerate(root["entries"]):
            if entry["node"]["id"] == identity:
                if action == "key":
                    entry["key"] = value
                    return False
                if action == "remove":
                    root["entries"].pop(index)
                elif action in ("up", "down"):
                    target = index + (-1 if action == "up" else 1)
                    if 0 <= target < len(root["entries"]):
                        root["entries"][index], root["entries"][target] = (
                            root["entries"][target],
                            root["entries"][index],
                        )
                return True
    children = []
    if spec.kind == "object":
        children = [(child, root["fields"][key]) for key, child in spec.fields.items()]
    elif spec.kind in ("list", "mapping"):
        children = [(spec.item, entry["node"]) for entry in root["entries"]]
    elif spec.kind == "union":
        children = [(spec.variants[int(k)], v) for k, v in root["branches"].items()]
    for child_spec, child in children:
        if find_node(child_spec, child, identity):
            return edit_entry(child_spec, child, identity, action, value)
    return False
