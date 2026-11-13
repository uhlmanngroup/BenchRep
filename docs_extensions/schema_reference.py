"""Expose schema Field metadata to mkdocstrings without importing BenchRep.

This is a docs-only Griffe extension. It reads declarations, never executes
constructors or validators, and leaves the Python API and its docstrings alone.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from functools import lru_cache
from html import escape
from io import StringIO
from pathlib import Path
import tokenize

from griffe import Extension

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "src/benchrep/assembly/schemas"


def literal(node):
    """Metadata must be literal; fail visibly rather than silently omit it."""
    return ast.literal_eval(node)


def name_of(node):
    return ast.unparse(node).split("[", 1)[0]


class Substitute(ast.NodeTransformer):
    def __init__(self, bindings):
        self.bindings = bindings

    def visit_Name(self, node):
        return deepcopy(self.bindings.get(node.id, node))


class Catalogue:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.classes = {}
        self.aliases = {}
        self.paths = {}
        modules = sorted((self.root / "src/benchrep/assembly/schemas").glob("*schema.py"))
        modules += [
            self.root / "src/benchrep/assembly/registries/utils.py",
            self.root / "src/benchrep/architecture/composite_model_roles.py",
        ]
        for path in modules:
            module = ".".join(path.relative_to(self.root / "src").with_suffix("").parts)
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.ClassDef):
                    self.classes[node.name] = node
                    self.paths[node.name] = f"{module}.{node.name}"
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    if ast.unparse(node.annotation) == "TypeAlias":
                        self.aliases[node.target.id] = node.value
                        self.paths[node.target.id] = f"{module}.{node.target.id}"
                elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                    if isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Subscript):
                        if name_of(node.value) == "Annotated":
                            self.aliases[node.targets[0].id] = node.value
                            self.paths[node.targets[0].id] = f"{module}.{node.targets[0].id}"
        # Only types reachable from schema fields/aliases belong in this reference.
        reachable = set()
        pending = [n.annotation for c in self.classes.values() for n in c.body if isinstance(n, ast.AnnAssign)]
        while pending:
            node = pending.pop()
            for token in ast.walk(node):
                if isinstance(token, ast.Name) and token.id in self.aliases and token.id not in reachable:
                    reachable.add(token.id)
                    pending.append(self.aliases[token.id])
        self.aliases = {k: v for k, v in self.aliases.items() if k in reachable}
        self.targets = {k: v for k, v in self.paths.items() if k in self.aliases or (k in self.classes and not k.startswith("_"))}

    def linked(self, expression):
        """Link type names while leaving string literals and punctuation alone."""
        text = ast.unparse(expression) if isinstance(expression, ast.AST) else expression
        lines = text.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        result, end = [], 0
        for token in tokenize.generate_tokens(StringIO(text).readline):
            if token.type == tokenize.NAME and token.string in self.targets:
                start = offsets[token.start[0] - 1] + token.start[1]
                stop = offsets[token.end[0] - 1] + token.end[1]
                result.append(escape(text[end:start]))
                result.append(f'<autoref identifier="{self.targets[token.string]}">{token.string}</autoref>')
                end = stop
        result.append(escape(text[end:]))
        return "".join(result)

    def generic_parameters(self, name):
        for base in self.classes[name].bases:
            if isinstance(base, ast.Subscript) and name_of(base) == "Generic":
                args = base.slice.elts if isinstance(base.slice, ast.Tuple) else [base.slice]
                return [ast.unparse(x) for x in args]
        return []

    @lru_cache(maxsize=None)
    def raw(self, name):
        cls = self.classes[name]
        fields, config = {}, {}
        for base in reversed(cls.bases):
            base_name = name_of(base)
            if base_name not in self.classes:
                continue
            inherited, base_config = self.raw(base_name)
            config.update(deepcopy(base_config))
            bindings = {}
            if isinstance(base, ast.Subscript):
                args = base.slice.elts if isinstance(base.slice, ast.Tuple) else [base.slice]
                bindings = dict(zip(self.generic_parameters(base_name), args))
            for key, value in inherited.items():
                field = deepcopy(value)
                field["annotation"] = Substitute(bindings).visit(field["annotation"])
                fields[key] = field
        for node in cls.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model_config" for t in node.targets):
                if not isinstance(node.value, ast.Call) or name_of(node.value.func) != "ConfigDict":
                    raise ValueError(f"Unsupported model_config declaration in {name}")
                config.update({kw.arg: literal(kw.value) for kw in node.value.keywords})
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            field_name = node.target.id
            if field_name.startswith("_") or field_name == "model_config":
                continue
            if name_of(node.annotation) in {"ClassVar", "Final"}:
                continue
            kwargs = {}
            value = node.value
            if isinstance(value, ast.Call) and name_of(value.func) == "Field":
                kwargs = {k.arg: k.value for k in value.keywords}
                if value.args:
                    if len(value.args) != 1:
                        raise ValueError(f"Unexpected Field arguments: {name}.{field_name}")
                    kwargs["default"] = value.args[0]
            elif value is not None:
                kwargs["default"] = value
            default = kwargs.get("default")
            factory = kwargs.get("default_factory")
            required = default is None and factory is None
            if isinstance(default, ast.Constant) and default.value is Ellipsis:
                required = True
            if factory is not None:
                default_text = ast.unparse(factory.body) if isinstance(factory, ast.Lambda) else f"{ast.unparse(factory)}()"
            else:
                default_text = "Required" if required else ast.unparse(default)
            extra = literal(kwargs["json_schema_extra"]) if "json_schema_extra" in kwargs else {}
            fields[field_name] = {
                "name": field_name,
                "annotation": deepcopy(node.annotation),
                "default": default_text,
                "factory": factory is not None,
                "required": required,
                "description": literal(kwargs["description"]) if "description" in kwargs else "",
                "extra": extra,
                "settings": {k: ast.unparse(v) for k, v in kwargs.items() if k not in {"default", "default_factory", "description", "json_schema_extra", "title"}},
                "defined_in": self.paths[name],
            }
        return fields, config

    def schema(self, name):
        fields, config = deepcopy(self.raw(name))
        for field in fields.values():
            field["type_html"] = self.linked(field.pop("annotation"))
            field["default_html"] = self.linked(field["default"]) if not field["required"] else "Required"
            field["notes"] = field["extra"].get("notes", [])
            if isinstance(field["notes"], str):
                field["notes"] = [field["notes"]]
        extra_mode = config.get("extra", "ignore")
        return {
            "fields": list(fields.values()),
            "extra_mode": extra_mode,
            "extra_description": config.get("json_schema_extra", {}).get("extra_field_behavior", {
                "forbid": "Unknown fields raise a validation error.",
                "allow": "Additional fields are accepted and retained.",
                "ignore": "Additional fields are ignored by schema validation.",
            }[extra_mode]),
            "bases_html": [self.linked(base) for base in self.classes[name].bases if name_of(base) in self.targets],
        }


class SchemaReference(Extension):
    """Attach presentation metadata only; no runtime imports or validation."""
    def __init__(self):
        self.catalogue = Catalogue()

    def on_class(self, *, cls, **kwargs):
        if self.catalogue.paths.get(cls.name) == cls.path:
            cls.extra["benchrep_schema"] = self.catalogue.schema(cls.name)

    def on_attribute(self, *, attr, **kwargs):
        if attr.name in self.catalogue.aliases and self.catalogue.paths[attr.name] == attr.path:
            attr.extra["benchrep_type_alias"] = self.catalogue.linked(self.catalogue.aliases[attr.name])
