"""Check the public reference against source without importing runtime packages."""
import ast
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

gr = pytest.importorskip("griffe", reason="Public reference tests require the docs extra.")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("public_reference", ROOT / "docs_extensions/public_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.fixture(scope="module")
def package():
    loader = gr.GriffeLoader(
        search_paths=[ROOT / "src"], allow_inspection=False,
        docstring_parser="numpy",
        extensions=gr.load_extensions(str(ROOT / "docs_extensions/public_reference.py")),
    )
    return loader.load("benchrep")


def test_catalogue_covers_every_exported_registry():
    tree = ast.parse((ROOT / "src/benchrep/assembly/registries/__init__.py").read_text())
    exports = ast.literal_eval(reference.assigned(tree)["__all__"])
    catalogue = reference.read_catalogue(ROOT)
    assert {r["symbol"] for r in catalogue} == set(exports)
    # Empty registries must remain visible too.
    custom = next(r for r in catalogue if r["symbol"] == "CUSTOM_OBJECTIVE_LOSSES")
    assert custom["entries"] == []
    assert custom["custom_registration"]
    assert custom["entry_type"] == "LossComponent"


def test_policies_targets_aliases_and_loop_registrations():
    catalogue = {r["name"]: r for r in reference.read_catalogue(ROOT)}
    assert {n for n, r in catalogue.items() if not r["custom_registration"]} == {
        "model", "reduction", "clustering_method", "predictability_probe",
    }
    encoder = {e["name"]: e for e in catalogue["encoder"]["entries"]}["conv2d"]
    assert "cnn" in encoder["aliases"]
    assert encoder["target"] == "benchrep.architecture.encoders.Conv2DEncoder"
    assert "runtime_inputs" in encoder["contract"]
    schedulers = {e["name"]: e for e in catalogue["lr_scheduler"]["entries"]}
    assert schedulers["reduce_on_plateau"]["aliases"] == ["reducelronplateau"]
    metric = next(e for e in catalogue["embedding_metric"]["entries"] if e["name"] == "quantiles")
    assert ast.literal_eval(metric["contract"]["result_kind"]) == "vector_mapping"
    assert ast.literal_eval(metric["contract"]["vector_axis"]) == "embedding_dimension"


def test_unrecognized_dynamic_registration_fails_instead_of_disappearing(tmp_path):
    for name in ["assembly/registries/core.py", "assembly/registries/builtins.py", "discovery/registry.py"]:
        target = tmp_path / "src/benchrep" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "src/benchrep" / name, target)
    path = tmp_path / "src/benchrep/assembly/registries/builtins.py"
    path.write_text(path.read_text() + '\nDATASETS._register_builtin(dynamic_name, MNISTDataset)\n')
    with pytest.raises(ValueError, match="Unsupported dynamic registration"):
        reference.read_catalogue(tmp_path)


def test_explicit_reference_targets_resolve_and_have_real_docstrings(package):
    for page in (ROOT / "docs/reference").rglob("*.md"):
        if "schemas" in page.parts:
            continue
        text = page.read_text()
        for target in re.findall(r"^::: (\S+)", text, re.MULTILINE):
            obj = package[target.removeprefix("benchrep.")]
            if obj.is_alias:
                obj = obj.final_target
            if obj.is_module:
                continue
            if obj.is_attribute:
                assert obj.value is not None, target
            else:
                assert obj.docstring, target
            assert obj.filepath.is_relative_to(ROOT), target


def test_result_field_descriptions_cover_declared_fields(package):
    paths = [
        "workflows.train.TrainingWorkflowResult", "workflows.predict.PredictionWorkflowResult",
        "workflows.evaluate.EvaluationWorkflowResult", "gui.shared.results.ConfiguredWorkflow",
        "runtime.status.training.TrainingStatusReport", "runtime.status.prediction.PredictionStatusReport",
        "runtime.status.prediction.PredictionOutcome", "runtime.status.evaluation.EvaluationStatusReport",
        "runtime.status.evaluation.EvaluationSectionStatus", "runtime.status.evaluation.EvaluationOutcome",
    ]
    for path in paths:
        cls = package[path]
        described = {item.name for section in cls.docstring.parsed if section.kind.value == "attributes" for item in section.value}
        declared = {name for name, attr in cls.attributes.items() if not name.startswith("_")}
        assert described == declared, (path, described ^ declared)


def test_example_helpers_keep_public_paths_and_repository_source(package):
    for name in ["example_config_path", "example_asset_path", "load_example_config"]:
        obj = package["examples." + name]
        assert obj.path == "benchrep.examples." + name
        assert obj.filepath == ROOT / "examples/__init__.py"
    assert len(package["examples.load_example_config"].overloads) == 3


def test_documentation_loading_does_not_import_runtime_dependencies():
    script = '''
import sys
from griffe import GriffeLoader, load_extensions
class BlockRuntimeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'benchrep', 'torch', 'torchvision', 'lightning', 'dash', 'scanpy'}:
            raise AssertionError('Runtime import attempted: ' + fullname)
sys.meta_path.insert(0, BlockRuntimeImports())
loader = GriffeLoader(search_paths=['src'], allow_inspection=False,
    extensions=load_extensions('docs_extensions/public_reference.py'))
package = loader.load('benchrep')
assert package['examples.example_config_path'].docstring
assert package['assembly.registries.core'].extra['benchrep_catalogue']
'''
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)


def test_catalogue_links_documented_implementations(package):
    catalogue = package["assembly.registries.core"].extra["benchrep_catalogue"]
    entries = {entry["name"]: entry for registry in catalogue for entry in registry["entries"]}
    assert entries["conv2d"]["reference"] == "benchrep.architecture.encoders.conv2d.Conv2DEncoder"
    assert entries["vae"]["reference"] == "benchrep.architecture.models.vae.VAE"
    assert entries["linear"]["source"]  # Schedule or probe implementation remains traceable.
    for registry in catalogue:
        for entry in registry["entries"]:
            if entry["reference"]:
                assert package[entry["reference"].removeprefix("benchrep.")].is_class
