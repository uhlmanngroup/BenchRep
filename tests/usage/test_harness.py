"""Fast checks that the harness preserves the examples it is meant to test."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from .profiles import MODULES, changes, fast_config
from .runner import check_status

ROOT = Path(__file__).resolve().parents[2]


def test_every_usage_script_has_a_profile():
    shipped = {p.stem for p in (ROOT / "src/benchrep/examples/usage").glob("[0-9]*.py")}
    assert shipped == set(MODULES.values()), "Add a test profile for new usage scripts."


def leaves(node, prefix=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, (*prefix, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaves(value, (*prefix, index))
    else:
        yield prefix, node


@pytest.mark.parametrize("path", sorted((ROOT / "src/benchrep/examples/configs").glob("[0-9]*/*.yaml")),
                         ids=lambda path: f"{path.parent.name}/{path.name}")
def test_profiles_keep_features_and_original_yaml(path):
    original_bytes = path.read_bytes()
    raw = yaml.safe_load(original_bytes)
    before = deepcopy(raw)
    example = path.parent.name.split("_", 1)[0]
    if example == "08":
        example = "08A"
    edited = fast_config(raw, example)
    assert raw == before
    assert path.read_bytes() == original_bytes
    edited_leaves = dict(leaves(edited))
    for field, value in leaves(before):
        # Preserve feature switches, architectures, losses, preprocessing,
        # selections and formats. Only explicit budget/execution edits differ.
        if "enabled" in field or field[0] in {
            "encoder", "decoder", "model", "losses", "transform_pipelines",
            "composite_model_components", "composite_model_declarations", "composite_model_assembly",
        } or field[-1:] == ("selected",) or "formats" in field:
            assert edited_leaves[field] == value, field


def test_python_config_preserves_omission():
    raw = {"stage": "training", "trainer": {"max_epochs": 10}}
    edited = fast_config(raw, "03")
    assert "checkpointing" not in edited
    assert "logger" not in edited
    assert "seed" not in edited


def test_shared_yaml_aliases_are_not_mutated():
    raw = yaml.safe_load("stage: training\ntransform_pipelines:\n- steps: &steps\n  - name: normalize\n- steps: *steps\n")
    edited = fast_config(raw, "07")
    edited["transform_pipelines"][0]["steps"][0]["name"] = "changed"
    assert raw["transform_pipelines"][0]["steps"][0]["name"] == "normalize"


def test_schedule_budget_and_cosine_are_consistent():
    scheduler = {"stage": "training", "optimizer": {"lr_scheduler": {
        "name": "cosine_annealing", "params": {"T_max": 20}}}}
    edited = fast_config(scheduler, "07")
    assert edited["optimizer"]["lr_scheduler"]["params"]["T_max"] == edited["trainer"]["max_epochs"]
    raw = yaml.safe_load((ROOT / "src/benchrep/examples/configs/06_schedule_and_monitor_training/training.yaml").read_text())
    edited = fast_config(raw, "06")
    assert edited["trainer"]["max_epochs"] > max(raw["optimizer"]["lr_scheduler"]["params"]["milestones"])
    assert edited["losses"] == raw["losses"]


def test_external_prediction_datamodule_keeps_ownership():
    raw = {"stage": "prediction", "overrides": {"datamodule": {}}}
    assert fast_config(raw, "10")["data"] == {"max_batches": 4}


@pytest.mark.parametrize("status", ["failed", "skipped", "partially_completed", "pending", "running",
                                    "completed_after_interruption"])
def test_nested_bad_outcome_cannot_hide_behind_success(status):
    report = {"status": "completed", "anndata": {"outcomes": [{"name": "metric", "status": status}]}}
    with pytest.raises(AssertionError):
        check_status(report, [], "evaluation")


def test_warnings_are_recorded_without_suppressing_them():
    warnings = []
    check_status({"status": "completed_with_warnings", "issues": ["upstream warning"]}, warnings, "training")
    assert warnings == ["training: upstream warning"]


def test_change_report_distinguishes_omitted_and_null():
    assert changes({"a": None}, {"a": 1, "b": None}) == [
        {"field": "a", "before": None, "after": 1},
        {"field": "b", "before": "<omitted>", "after": None},
    ]


def test_balanced_sampling_is_repeatable_and_keeps_rare_classes():
    np = pytest.importorskip("numpy")
    from .runner import balanced_indices
    labels = np.array([0] * 100 + [1] * 100 + [2] * 3)
    first = balanced_indices(labels, 23)
    assert len(set(first)) == 23
    assert (first == balanced_indices(labels, 23)).all()
    assert (labels[first] == 2).sum() == 3
    assert abs((labels[first] == 0).sum() - (labels[first] == 1).sum()) <= 1


def test_dataset_cap_happens_before_enriched_preprocessing(tmp_path):
    torch = pytest.importorskip("torch")
    from contextlib import ExitStack
    from unittest.mock import patch
    from torchvision import datasets
    from .profiles import DATASET_LIMIT
    from .runner import patch_datasets

    calls = []

    def load(self, root, train=True, **kwargs):
        calls.append(root)
        self.data = torch.arange(2000).reshape(2000, 1, 1)
        self.targets = torch.arange(2000) % 10

    with patch.object(datasets.MNIST, "__init__", load), ExitStack() as stack:
        report = {"datasets": []}
        patch_datasets(stack, tmp_path, report)
        dataset = datasets.MNIST("data/mnist", train=True)
        assert len(dataset.data) == len(dataset.targets) == DATASET_LIMIT
        assert torch.equal(dataset.data[:, 0, 0] % 10, dataset.targets)
        assert calls == [str(tmp_path / "mnist")]
        assert report["datasets"][0]["before"] == 2000


def test_python_config_adjustment_retains_typed_dataset_and_authorship():
    pytest.importorskip("benchrep")
    from importlib import import_module
    from .runner import fast_object
    module = import_module("benchrep.examples.usage.03_config_object_pipeline")
    config = module.build_training_config()
    edited = fast_object(config, "03", "training", "cpu", {"config_changes": []})
    assert edited.dataset.name == "mnist"
    assert edited.dataset.model_fields_set == config.dataset.model_fields_set
    assert edited.checkpointing.model_fields_set == config.checkpointing.model_fields_set
    assert edited.trainer.max_epochs == 2
    assert edited.trainer.limit_train_batches == 4
    assert config.trainer.max_epochs == 5
    assert edited.datamodule.num_workers == 0
    assert not edited.datamodule.persistent_workers


@pytest.fixture
def all_noise_evaluation():
    from types import SimpleNamespace
    import pandas as pd
    outcomes = [
        {"name": "hdbscan", "category": "clustering", "status": "completed_with_warnings",
         "issues": ("Warning (RuntimeWarning): HDBSCAN produced no non-noise clusters: "
                    "all 3 observations were classified as noise. Dependent clustering metrics may be unavailable.",)},
        {"name": "internal_clustering_metrics_hdbscan", "category": "metrics", "status": "failed",
         "issues": ("Error (RecoverableEvaluationStepError): Internal clustering metrics require at least 2 clusters, got 0.",)},
        {"name": "external_clustering_metrics_hdbscan", "category": "metrics", "status": "failed",
         "issues": ("Error (RecoverableEvaluationStepError): External clustering metrics require at least one non-noise HDBSCAN observation.",)},
    ]
    result = SimpleNamespace(
        adata=SimpleNamespace(n_obs=3, obs=pd.DataFrame({"clusters": ["-1"] * 3})),
        config=SimpleNamespace(clustering=SimpleNamespace(hdbscan=SimpleNamespace(
            enabled=True, params=SimpleNamespace(key_added="clusters"))))
    )
    status = {"status": "partially_completed", "fatal_issue": None,
              "anndata": {"status": "partially_completed", "outcomes": outcomes},
              "exports": {"status": "completed", "outcomes": []}}
    return result, status, {"example": "09", "warnings": []}


def test_exact_all_noise_exception_preserves_original_status(all_noise_evaluation):
    from .runner import check_evaluation_status
    result, status, report = all_noise_evaluation
    original = deepcopy(status)
    check_evaluation_status(result, status, report)
    assert status == original
    assert len(report["expected_issues"]) == 3
    assert report["warnings"] == []


@pytest.mark.parametrize("problem", ["other_example", "real_cluster", "missing_labels", "disabled",
                                    "different_error", "extra_issue", "missing_outcome", "duplicate",
                                    "other_failure", "other_warning", "unmarked_issue", "fatal", "export_failure"])
def test_all_noise_exception_rejects_other_problems(all_noise_evaluation, problem):
    from .runner import check_evaluation_status
    result, status, report = all_noise_evaluation
    outcomes = status["anndata"]["outcomes"]
    if problem == "other_example":
        report["example"] = "08A"
    elif problem == "real_cluster":
        result.adata.obs.loc[0, "clusters"] = "0"
    elif problem == "missing_labels":
        result.adata.obs.drop(columns="clusters", inplace=True)
    elif problem == "disabled":
        result.config.clustering.hdbscan.enabled = False
    elif problem == "different_error":
        outcomes[1]["issues"] = ("some other error",)
    elif problem == "extra_issue":
        outcomes[0]["issues"] += ("another warning",)
    elif problem == "missing_outcome":
        outcomes.pop()
    elif problem == "duplicate":
        outcomes.append(deepcopy(outcomes[0]))
    elif problem in {"other_failure", "other_warning"}:
        outcomes.append({"name": "unrelated", "status": "failed" if problem == "other_failure"
                         else "completed_with_warnings", "issues": ("unrelated issue",)})
    elif problem == "unmarked_issue":
        status["exports"]["issues"] = ("unexpected issue",)
    elif problem == "fatal":
        status["fatal_issue"] = "fatal error"
    elif problem == "export_failure":
        status["exports"]["status"] = "failed"
    with pytest.raises(AssertionError):
        check_evaluation_status(result, status, report)
    assert not report.get("expected_issues")


@pytest.mark.parametrize("scenario,cleanup_flag,removed", [
    ("pass", None, True), ("pass", "--no-usage-clean-on-success", False),
    ("pass", "--usage-clean-on-success", True),
    ("fail", None, False), ("teardown", None, False), ("interrupt", None, False),
    ("collection", None, False),
])
def test_cleanup_only_after_successful_session(tmp_path, scenario, cleanup_flag, removed):
    import os
    import shutil
    import subprocess
    import sys
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "__init__.py").touch()
    for name in ("conftest.py", "profiles.py"):
        shutil.copy(Path(__file__).with_name(name), suite / name)
    (suite / "test_sample.py").write_text('''
import pytest
@pytest.fixture
def teardown_check():
    yield
    if SCENARIO == "teardown":
        raise RuntimeError("teardown failed")

def test_example(usage_root, teardown_check, request):
    assert request.config.getoption("--usage-device") == "auto"
    (usage_root / "run.log").write_text("diagnostics")
    if SCENARIO == "fail":
        assert False
    if SCENARIO == "interrupt":
        raise KeyboardInterrupt
'''.replace("SCENARIO", repr(scenario)))
    if scenario == "collection":
        (suite / "test_broken.py").write_text("raise RuntimeError('collection failed')")
    output = tmp_path / "outputs"
    command = [sys.executable, "-m", "pytest", str(suite), "-q", "--run-usage-examples",
               "--usage-output", str(output)]
    if scenario == "collection":
        command.append("--continue-on-collection-errors")
    if cleanup_flag:
        command.append(cleanup_flag)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="")
    completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert (completed.returncode == 0) == (scenario == "pass"), completed.stdout + completed.stderr
    assert (not output.exists()) == removed, completed.stdout + completed.stderr
    if not removed:
        assert (output / "run.log").read_text() == "diagnostics"
