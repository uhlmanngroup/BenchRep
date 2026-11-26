"""Record tracked resources, then check a non-editable install outside the checkout.

Run ``record manifest.json`` from a checkout before building. Run
``python -I /path/to/this_script.py check manifest.json`` with the installed
environment's Python. Neither mode runs training or downloads datasets.
"""

import argparse
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys


RESOURCE_PATHS = (
    "src/benchrep/examples/__init__.py",
    "src/benchrep/examples/usage",
    "src/benchrep/examples/configs",
    "src/benchrep/examples/assets",
    "src/benchrep/gui/assets",
)

# Imported only in check mode, after installing the distribution with [full].
OPTIONAL_MODULES = (
    "dash", "dash_mantine_components", "scanpy", "leidenalg", "igraph",
    "optuna", "optuna_integration", "optuna_dashboard", "wandb", "tensorboard",
    "mlflow", "torchview", "graphviz", "xgboost", "skimage",
)


def record_resources(manifest: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    tracked = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "-z", "--", *RESOURCE_PATHS],
    ).decode("utf-8").split("\0")
    expected = {
        Path(name).relative_to("src/benchrep").as_posix():
            sha256((repository / name).read_bytes()).hexdigest()
        for name in tracked if name
    }
    if not expected:
        raise RuntimeError("No tracked package resources found.")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Recorded {len(expected)} tracked resources.")


def check_installation(manifest: Path) -> None:
    if not sys.flags.isolated:
        raise RuntimeError("Run the installation check with python -I.")
    import benchrep
    from benchrep.examples import example_config_path, load_example_config

    package = Path(benchrep.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"BenchRep was imported outside the environment: {package}")

    expected = json.loads(manifest.read_text(encoding="utf-8"))
    if not expected:
        raise RuntimeError("The resource manifest is empty.")
    for relative, digest in expected.items():
        path = package / relative
        if not path.is_file():
            raise RuntimeError(f"Missing installed resource: {relative}")
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Installed resource differs from checkout: {relative}")

    config = example_config_path("01_yaml_pipeline", stage="training")
    if not config.resolve().is_relative_to(package) or not config.is_file():
        raise RuntimeError(f"Example helper returned an unexpected path: {config}")
    load_example_config("01_yaml_pipeline", stage="training")
    import_module("benchrep.gui")
    for name in OPTIONAL_MODULES:
        import_module(name)
    print(f"PASS: installed BenchRep at {package}")
    print(f"Verified {len(expected)} resources, config loading and optional imports.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "check"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.mode == "record":
        record_resources(args.manifest)
    else:
        check_installation(args.manifest)
