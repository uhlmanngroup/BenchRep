"""Build public-reference metadata without importing BenchRep dependencies.

Read registry declarations and registrations as syntax, and expose the packaged
examples under the same name used by setuptools. No docstrings are synthesized.
"""
from __future__ import annotations

import ast
from pathlib import Path

from griffe import Extension, visit

ROOT = Path(__file__).resolve().parents[1]


def normalize(name):
    return name.strip().lower().replace("-", "_")


def assigned(tree):
    """Read top-level named assignments, including annotated assignments."""
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name):
            result[target.id] = value
    return result


def read_catalogue(root=ROOT):
    """Extract every public registry and built-in entry; reject unknown syntax."""
    root = Path(root)
    core = ast.parse((root / "src/benchrep/assembly/registries/core.py").read_text())
    discovery = ast.parse((root / "src/benchrep/discovery/registry.py").read_text())
    builtins = ast.parse((root / "src/benchrep/assembly/registries/builtins.py").read_text())
    declarations = assigned(core)
    discovery_values = assigned(discovery)
    inventory = discovery_values["_COMPONENT_REGISTRIES"]
    result = []
    for key, value in zip(inventory.keys, inventory.values):
        options = {k.arg: k.value for k in value.keywords}
        symbol = ast.literal_eval(options["symbol"])
        declaration = declarations[symbol]
        policy = {k.arg: k.value for k in declaration.keywords}
        entry_type = "LossComponent" if ast.unparse(declaration.func) == "LossRegistry" else ast.unparse(policy["entry_type"]) if "entry_type" in policy else "Callable"
        contract = options.get("contract", ast.Constant(""))
        if isinstance(contract, ast.Name):
            contract = discovery_values[contract.id]
        result.append({
            "name": ast.literal_eval(key),
            "symbol": symbol,
            "entry_type": entry_type,
            "custom_registration": ast.literal_eval(policy.get("custom_registration_supported", ast.Constant(True))),
            "locations": ast.literal_eval(options["config_locations"]),
            "contract": ast.literal_eval(contract),
            "entries": [],
        })
    by_symbol = {item["symbol"]: item for item in result}
    imports = {}
    for node in ast.walk(builtins):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name] = alias.name

    def target_name(node):
        text = ast.unparse(node)
        head, _, tail = text.partition(".")
        if head not in imports:
            raise ValueError(f"Unknown registration target: {text}")
        return imports[head] + ("." + tail if tail else "")

    calls = sorted((n for n in ast.walk(builtins) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "_register_builtin"), key=lambda n: n.lineno)
    scheduler_calls = []
    for call in calls:
        symbol = ast.unparse(call.func.value)
        if symbol not in by_symbol:
            raise ValueError(f"Uncatalogued registry: {symbol}")
        if not isinstance(call.args[0], ast.Constant):
            # The sole loop-based registration is checked explicitly below.
            if symbol != "LR_SCHEDULERS" or [ast.unparse(a) for a in call.args] != ["name", "scheduler_class", "scheduler_class.__name__"]:
                raise ValueError(f"Unsupported dynamic registration at line {call.lineno}")
            scheduler_calls.append(call)
            continue
        component = call.args[1]
        contract = {}
        if isinstance(component, ast.Call):
            wrapper = ast.unparse(component.func)
            if wrapper not in {"ArchitectureComponent", "LossComponent", "EvaluationMetric"}:
                raise ValueError(f"Unknown entry wrapper: {wrapper}")
            kwargs = {k.arg: k.value for k in component.keywords}
            component = kwargs.pop("fn" if wrapper == "EvaluationMetric" else "component")
            contract = {name: ast.unparse(value) for name, value in kwargs.items()}
        by_symbol[symbol]["entries"].append({
            "name": normalize(ast.literal_eval(call.args[0])),
            "aliases": [normalize(ast.literal_eval(a)) for a in call.args[2:]],
            "target": target_name(component),
            "contract": contract,
        })
    if scheduler_calls:
        if len(scheduler_calls) != 1:
            raise ValueError("Expected one scheduler registration loop")
        values = [n.value for n in ast.walk(builtins) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "lr_scheduler_builtins" for t in n.targets)]
        if len(values) != 1 or not isinstance(values[0], ast.Tuple):
            raise ValueError("Expected literal scheduler declarations")
        for item in values[0].elts:
            name, component = item.elts
            target = target_name(component)
            by_symbol["LR_SCHEDULERS"]["entries"].append({
                "name": normalize(ast.literal_eval(name)),
                "aliases": [normalize(target.rsplit(".", 1)[-1])],
                "target": target,
                "contract": {},
            })
    for item in result:
        item["entries"].sort(key=lambda entry: entry["name"])
        names = [name for entry in item["entries"] for name in [entry["name"], *entry["aliases"]]]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate registered names in {item['symbol']}")
    return result


class PublicReference(Extension):
    """Read source metadata for the catalogue and packaged-example helpers."""

    def on_module_members(self, *, mod, agent, **kwargs):
        if mod.path != "benchrep":
            return
        # Mirror pyproject.toml's package-dir mapping in Griffe's static tree.
        path = ROOT / "examples/__init__.py"
        examples = visit(
            "examples", path, path.read_text(encoding="utf-8"), parent=mod,
            docstring_parser=agent.docstring_parser,
            docstring_options=agent.docstring_options,
            lines_collection=agent.lines_collection,
            modules_collection=agent.modules_collection,
        )
        mod.set_member("examples", examples)

    def on_module(self, *, mod, loader, **kwargs):
        if mod.path != "benchrep.assembly.registries.core":
            return
        catalogue = read_catalogue()
        for registry in catalogue:
            for entry in registry["entries"]:
                entry["source"] = None
                if entry["target"].startswith("benchrep."):
                    obj = loader.modules_collection.get_member(entry["target"])
                    if obj.is_alias:
                        obj = obj.final_target
                    entry["source"] = f"{obj.relative_filepath.as_posix()}#L{obj.lineno}"
        mod.extra["benchrep_catalogue"] = catalogue
