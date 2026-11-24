"""Child process for one example. All patches live only in this process."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
from unittest.mock import patch

if __package__:
    from .profiles import (
        MODULES, DATASET_LIMIT, STANDALONE_SAMPLES, OPTUNA_TRIALS,
        changes, fast_config,
    )
else:
    from profiles import (
        MODULES, DATASET_LIMIT, STANDALONE_SAMPLES, OPTUNA_TRIALS,
        changes, fast_config,
    )


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_exists(path):
    require(path is not None and Path(path).is_file() and Path(path).stat().st_size > 0,
            f"Missing or empty artifact: {path}")


def balanced_indices(labels, limit):
    """Deterministic round-robin sample across labels (or joint strata)."""
    import numpy as np
    from itertools import zip_longest

    rng = np.random.default_rng(137)
    labels = np.asarray(labels)
    if len(labels) <= limit:
        return np.arange(len(labels))
    groups = [rng.permutation(np.flatnonzero(labels == label)) for label in np.unique(labels)]
    selected = []
    for row in zip_longest(*groups):
        for index in row:
            if index is not None:
                selected.append(index)
                if len(selected) == limit:
                    return rng.permutation(selected)
    raise AssertionError("Could not select requested dataset size")


def patch_datasets(stack, data_dir, report):
    """Keep real readers and samples; cap before enriched MNIST preprocessing."""
    import numpy as np
    import torch
    from torchvision import datasets

    classes = (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10, datasets.STL10)
    originals = [(cls, cls.__init__) for cls in classes]
    for cls, original in originals:
        signature = inspect.signature(original)

        def limited(self, *args, _original=original, _signature=signature, **kwargs):
            bound = _signature.bind(self, *args, **kwargs)
            root = Path(bound.arguments["root"])
            bound.arguments["root"] = str(data_dir / root.name)
            _original(*bound.args, **bound.kwargs)
            label_field = "labels" if hasattr(self, "labels") else "targets"
            labels = getattr(self, label_field)
            indices = balanced_indices(labels, DATASET_LIMIT)
            before = len(self.data)
            for field in ("data", label_field):
                values = getattr(self, field)
                if isinstance(values, torch.Tensor):
                    values = values[torch.as_tensor(indices)].clone()
                elif isinstance(values, np.ndarray):
                    values = values[indices].copy()
                else:
                    values = [values[int(i)] for i in indices]
                setattr(self, field, values)
            report["datasets"].append({"class": type(self).__name__, "before": before,
                                       "after": len(self.data), "root": str(bound.arguments["root"])})
        stack.enter_context(patch.object(cls, "__init__", limited))


def patch_configs(stack, example, device, report):
    import yaml
    import benchrep.examples as resources

    original_path = resources.example_config_path
    generated = {}

    def config_path(name, *, stage=None):
        source = original_path(name, stage=stage)
        if source not in generated:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
            edited = fast_config(raw, example, device=device)
            destination = Path("fast-configs") / source.relative_to(resources._ROOT / "configs")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")
            generated[source] = destination.resolve()
            report["config_changes"].append({"source": str(source), "changes": changes(raw, edited)})
        return generated[source]

    stack.enter_context(patch.object(resources, "example_config_path", config_path))
    # load_example_config calls this helper through its module globals, so its
    # original schema validation remains intact, including import-time loads.


def fast_object(config, example, stage, device, report):
    """Use BenchRep's updater to retain typed sections and field authorship."""
    from benchrep import update_config
    from benchrep.assembly.schemas.parsing import (
        TrainingConfigContext, PredictionConfigContext, EvaluationConfigContext,
    )
    from benchrep.interfaces.model_families import VAE_FAMILY

    raw = config.model_dump(mode="python")
    edited = fast_config(raw, example, device=device)
    edits = changes(raw, edited)
    report["config_changes"].append({"source": f"Python {stage} config", "changes": edits})
    updates = {}
    trainer_edits = {}
    for edit in edits:
        field = edit["field"]
        if field.startswith("trainer."):
            # Trainer passthrough settings may not exist yet; replace this typed
            # section instead of asking update_config to create a missing key.
            trainer_edits[field.removeprefix("trainer.")] = edit["after"]
        else:
            updates[field] = edit["after"]
    if trainer_edits:
        updates["trainer"] = config.trainer.model_copy(update=trainer_edits, deep=True)
    if stage == "training":
        context = TrainingConfigContext(external_model=False, external_datamodule=False)
    elif stage == "prediction":
        context = PredictionConfigContext(model_family=VAE_FAMILY,
                                          training_manifest_path_overridden=False)
    else:
        context = EvaluationConfigContext(prediction_manifest_path_overridden=False)
    return update_config(config, updates, context=context)


def check_status(node, warnings, prefix, *, allow_warnings=True):
    if isinstance(node, dict):
        if not allow_warnings:
            require(node.get("status") != "completed_with_warnings", f"{prefix}: unexpected warning status")
            require(not node.get("issues"), f"{prefix}: unexpected issues: {node.get('issues')}")
        state = node.get("status")
        require(state not in {"failed", "skipped", "partially_completed", "pending", "running",
                              "completed_after_interruption"}, f"{prefix}: {node}")
        if state == "completed_with_warnings" and node.get("issues"):
            warnings.extend(f"{prefix}: {issue}" for issue in node["issues"])
        require(not node.get("fatal_issue"), f"{prefix}: {node.get('fatal_issue')}")
        for key, value in node.items():
            check_status(value, warnings, f"{prefix}.{key}", allow_warnings=allow_warnings)
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            check_status(item, warnings, f"{prefix}[{index}]", allow_warnings=allow_warnings)


def check_evaluation_status(result, status, report):
    """Allow only example 09's verified all-noise HDBSCAN outcome."""
    if report.get("example") != "09" or status.get("status") != "partially_completed":
        check_status(status, report["warnings"], "evaluation")
        return

    adata = result.adata
    section = result.config.clustering.hdbscan
    require(section.enabled and adata is not None and adata.n_obs > 0,
            "Partial evaluation without enabled HDBSCAN and observations")
    key = section.params.key_added
    require(key in adata.obs and (adata.obs[key].astype(str) == "-1").all(),
            "Partial evaluation is not the expected all-noise HDBSCAN case")
    expected = {
        "hdbscan": ("clustering", "completed_with_warnings",
                    "Warning (RuntimeWarning): HDBSCAN produced no non-noise clusters: "
                    f"all {adata.n_obs} observations were classified as noise. "
                    "Dependent clustering metrics may be unavailable."),
        "internal_clustering_metrics_hdbscan": (
            "metrics", "failed", "Error (RecoverableEvaluationStepError): "
            "Internal clustering metrics require at least 2 clusters, got 0."),
        "external_clustering_metrics_hdbscan": (
            "metrics", "failed", "Error (RecoverableEvaluationStepError): "
            "External clustering metrics require at least one non-noise HDBSCAN observation."),
    }
    checked = deepcopy(status)
    anndata = checked["anndata"]
    require(anndata["status"] == "partially_completed", "Unexpected AnnData status")
    seen = []
    accepted = []
    for outcome in anndata["outcomes"]:
        name = outcome["name"]
        if name in expected:
            category, state, issue = expected[name]
            require(outcome["category"] == category and outcome["status"] == state
                    and tuple(outcome["issues"]) == (issue,),
                    f"Unexpected HDBSCAN outcome: {outcome}")
            seen.append(name)
            accepted.append(f"evaluation.{name}: {issue}")
            outcome["status"] = "completed"
            outcome["issues"] = ()
    require(Counter(seen) == Counter(expected.keys()), "Missing or duplicate HDBSCAN outcomes")
    checked["status"] = anndata["status"] = "completed"
    other_warnings = []
    check_status(checked, other_warnings, "evaluation", allow_warnings=False)
    # The stored status and manifest remain unchanged, including both failures.
    report.setdefault("expected_issues", []).extend(accepted)


def check_paths(node):
    if isinstance(node, Path):
        file_exists(node)
    elif is_dataclass(node):
        check_paths(asdict(node))
    elif isinstance(node, dict):
        for value in node.values():
            check_paths(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            check_paths(value)


def check_result(result, stage, report):
    import numpy as np

    file_exists(result.manifest_path)
    status = asdict(result.status_report)
    report["workflows"].append({"stage": stage, "manifest": str(result.manifest_path), "status": status})
    if stage == "evaluation":
        check_evaluation_status(result, status, report)
    else:
        check_status(status, report["warnings"], stage)
    if stage == "training":
        file_exists(result.checkpoint_callback.last_model_path)
        file_exists(result.checkpoint_callback.best_model_path)
        require(result.trainer.global_step > 0, "Training performed no optimizer steps")
        if result.config.inspection.torchview.enabled:
            file_exists(result.torchview_graph_path)
        if result.config.model.name == "composite":
            file_exists(result.composite_model_spec_graph_path)
    elif stage == "prediction":
        require(bool(result.predictions), "Prediction returned no batches")
        if result.config.exports.anndata.enabled:
            file_exists(result.export_result.anndata.path)
        if result.config.exports.reconstructions.enabled:
            pairs = result.export_result.reconstructions.pairs
            require(bool(pairs), "No reconstruction pairs exported")
            for pair in pairs:
                require((pair.paths.n_examples_exported or 0) > 0, "Empty reconstruction bundle")
                for path in (pair.paths.input_path, pair.paths.reconstruction_path,
                             pair.paths.obs_path, pair.paths.metadata_path):
                    if path is not None:
                        file_exists(path)
    else:
        require(bool(result.get_metric()), "Evaluation returned no metrics")
        file_exists(result.export_paths.metrics_json_path)
        check_paths(result.export_paths)
        adata = result.adata
        if adata is not None:
            require(adata.n_obs > 1, "Evaluation has no usable observations")
            for name in ("pca", "umap", "tsne"):
                section = getattr(result.config.reductions, name)
                if section.enabled:
                    values = adata.obsm[section.params.key_added]
                    require(np.isfinite(values).all(), f"Non-finite {name} coordinates")
                    if result.config.plots.enabled:
                        paths = result.export_paths.reduction_plot_paths or {}
                        require(bool(paths.get(f"{section.params.key_added}:uncolored")),
                                f"No {name} figures exported: {paths}")
            for name in ("kmeans", "leiden", "hdbscan"):
                section = getattr(result.config.clustering, name)
                if section.enabled:
                    require(section.params.key_added in adata.obs, f"Missing {name} assignments")
            probes = result.config.metrics.predictability
            if probes.enabled:
                for target, settings in probes.targets.items():
                    for probe in settings.selected:
                        score = result.get_metric("predictability", target, "probes", probe, "mean_score")
                        require(np.isfinite(score), f"Non-finite probe result: {target}/{probe}")
        if result.config.metrics.reconstruction.enabled:
            require(bool(result.reconstruction_outputs), "Missing reconstruction metrics")
            if result.config.plots.enabled:
                require(bool(result.export_paths.reconstruction_grid_paths), "Missing reconstruction grids")


def patch_workflows(stack, example, device, report):
    import benchrep

    for name, stage in (("train_ae", "training"), ("train_vae", "training"),
                        ("train_composite", "training"), ("predict_ae", "prediction"),
                        ("predict_vae", "prediction"), ("predict_composite", "prediction"),
                        ("evaluate", "evaluation")):
        original = getattr(benchrep, name)

        @wraps(original)
        def checked(*args, _original=original, _stage=stage, **kwargs):
            if example == "03":
                kwargs["config"] = fast_object(kwargs["config"], example, _stage, device, report)
            result = _original(*args, **kwargs)
            # Optuna deliberately catches Exception; retain failed checks so they
            # cannot disappear inside its failed-trial handling.
            try:
                check_result(result, _stage, report)
            except Exception as exc:
                report["check_failures"].append(f"{_stage}: {exc}")
                raise
            return result
        stack.enter_context(patch.object(benchrep, name, checked))


def prepare_standalone(stack, report):
    import anndata as ad
    import benchrep.examples as resources

    original = resources.example_asset_path
    source = original("09_standalone_evaluation_workflow")
    data = ad.read_h5ad(source / "anndata.h5ad")
    labels = data.obs["digit_label"].astype(str) + ":" + data.obs["acquisition_batch"].astype(str)
    indices = balanced_indices(labels.to_numpy(), STANDALONE_SAMPLES)
    destination = Path("standalone-inputs").resolve()
    destination.mkdir()
    data[indices].copy().write_h5ad(destination / "anndata.h5ad")
    shutil.copytree(source / "reconstruction_bundle", destination / "reconstruction_bundle")
    report["datasets"].append({"class": "Standalone AnnData", "before": data.n_obs, "after": len(indices)})

    def asset_path(relative):
        if relative == "09_standalone_evaluation_workflow":
            return destination
        return original(relative)
    stack.enter_context(patch.object(resources, "example_asset_path", asset_path))


def patch_optuna(stack, report):
    import optuna
    import optuna_dashboard

    original_optimize = optuna.Study.optimize
    original_pruner = optuna.pruners.MedianPruner
    original_sampler = optuna.samplers.TPESampler
    studies = []

    def optimize(study, *args, **kwargs):
        kwargs["n_trials"] = OPTUNA_TRIALS
        studies.append(study)
        return original_optimize(study, *args, **kwargs)

    def pruner(*args, **kwargs):
        kwargs.update(n_startup_trials=1, n_warmup_steps=0)
        return original_pruner(*args, **kwargs)

    def sampler(*args, **kwargs):
        kwargs["n_startup_trials"] = 1
        return original_sampler(*args, **kwargs)

    stack.enter_context(patch.object(optuna.Study, "optimize", optimize))
    stack.enter_context(patch.object(optuna.pruners, "MedianPruner", pruner))
    stack.enter_context(patch.object(optuna.samplers, "TPESampler", sampler))
    # The dashboard is an optional human-facing server, not the study itself.
    stack.enter_context(patch.object(optuna_dashboard, "run_server", lambda *a, **kw: None))
    report["optuna_settings"] = {"n_trials": OPTUNA_TRIALS, "n_startup_trials": 1,
                                 "n_warmup_steps": 0, "dashboard_server": False}
    return studies


def patch_wizard(stack, report):
    import webbrowser
    import benchrep
    from playwright.sync_api import sync_playwright, expect

    original = benchrep.configure_workflow

    def browser(url):
        with sync_playwright() as manager:
            browser = manager.chromium.launch(
                headless=True,
                executable_path=os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None,
            )
            try:
                page = browser.new_page(viewport={"width": 1600, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url, timeout=60000)
                page.get_by_role("tab", name="Review & finish", exact=False).click()
                page.get_by_role("button", name="Check configuration", exact=True).click()
                save = page.get_by_role("button", name="Save & finish", exact=True)
                expect(save).to_be_enabled(timeout=120000)
                save.click()
                expect(page.locator("#wizard-save-message")).to_contain_text("Authored configuration:", timeout=30000)
                require(not errors, f"Wizard JavaScript errors: {errors}")
            finally:
                browser.close()
        return True

    def configure(**kwargs):
        kwargs.update(port=0, open_browser=True)
        result = original(**kwargs)
        require(result is not None, "Wizard cancelled unexpectedly")
        file_exists(result.config_path)
        report["wizards"].append(kwargs["stage"])
        return result

    stack.enter_context(patch.object(webbrowser, "open", browser))
    stack.enter_context(patch.object(benchrep, "configure_workflow", configure))


def check_example_outputs(example, report, studies):
    import numpy as np
    import pandas as pd

    stages = Counter(item["stage"] for item in report["workflows"])
    if example == "06":
        require(stages == {"training": 1}, f"Unexpected workflow calls: {stages}")
        logs = list(Path("outputs").rglob("metrics.csv"))
        require(len(logs) == 1, f"Expected one training CSV: {logs}")
        data = pd.read_csv(logs[0])
        for column in ("lr/group_0", "loss_weights/train/regularization/gaussian_kl"):
            require(data[column].dropna().nunique() > 1, f"Schedule never changed: {column}")
        columns = [name for name in data if name.startswith("grad_norms_wrt_encoder/")]
        require(len(columns) >= 3, "Missing per-loss or optimizer gradient norms")
        require(all(np.isfinite(data[name].dropna()).all() and data[name].notna().any()
                    for name in columns), "Invalid gradient-norm logs")
    elif example == "09":
        require(stages == {"evaluation": 1}, f"Unexpected workflow calls: {stages}")
    elif example == "10":
        require(len(studies) == 1, "Expected one Optuna study")
        trials = studies[0].trials
        require(len(trials) == OPTUNA_TRIALS, "Incorrect trial count")
        require(trials[0].state.name == "FAIL" and "error_message" in trials[0].user_attrs,
                "Missing deliberately failed trial/error record")
        require("persistent_workers" in trials[0].user_attrs["error_message"],
                f"Trial 0 failed for an unexpected reason: {trials[0].user_attrs}")
        require(any(t.state.name == "COMPLETE" and np.isfinite(t.value) for t in trials),
                "No completed Optuna trial")
        require(all(t.state.name in {"COMPLETE", "PRUNED"} for t in trials[1:]),
                f"Unexpected failed trials: {[(t.number, t.state.name, t.user_attrs) for t in trials]}")
        require(stages["prediction"] >= 1 and stages["evaluation"] >= 1, "No scored Optuna trial")
        require(bool(list(Path("outputs").rglob("optuna.db"))), "Missing Optuna database")
        require(bool(list(Path("outputs").rglob("trials.csv"))), "Missing Optuna CSV")
        report["trials"] = [{"number": t.number, "state": t.state.name, "value": t.value} for t in trials]
    else:
        chains = 2 if example == "07" else 1
        require(stages == {"training": chains, "prediction": chains, "evaluation": chains},
                f"Unexpected workflow calls: {stages}")
        if example == "07":
            files = list(Path("outputs").rglob("comparison_*.csv"))
            require(len(files) == 1, f"Expected comparison CSV: {files}")
            rows = pd.read_csv(files[0])
            require(len(rows) == 8 and set(rows.initialization) == {"From scratch", "Pretrained"},
                    "Incomplete comparison CSV")
            require(np.isfinite(rows.mean_balanced_accuracy).all(), "Non-finite comparison scores")
        if example == "08B":
            require(report["wizards"] == ["training", "prediction", "evaluation"], "Missing wizard stages")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("example", choices=MODULES)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict-warnings", action="store_true")
    args = parser.parse_args()
    if diagnostic_root := os.environ.get("BENCHREP_CI_DIAGNOSTICS"):
        if __package__:
            from .ci_diagnostics import enable
        else:
            from ci_diagnostics import enable
        enable(Path(diagnostic_root) / args.example)
    report = {"example": args.example, "passed": False, "config_changes": [],
              "datasets": [], "workflows": [], "warnings": [], "check_failures": [], "wizards": []}
    started = time.monotonic()
    try:
        import torch
        import benchrep
        torch.set_num_threads(2)
        print(f"Testing installed BenchRep: {benchrep.__file__}", flush=True)
        with ExitStack() as stack:
            patch_configs(stack, args.example, args.device, report)
            patch_datasets(stack, args.data_dir, report)
            patch_workflows(stack, args.example, args.device, report)
            studies = []
            if args.example == "09":
                prepare_standalone(stack, report)
            if args.example == "10":
                studies = patch_optuna(stack, report)
            if args.example == "08B":
                patch_wizard(stack, report)
            sys.argv = [MODULES[args.example]]
            runpy.run_module(f"benchrep.examples.usage.{MODULES[args.example]}", run_name="__main__")
            require(not report["check_failures"], str(report["check_failures"]))
            check_example_outputs(args.example, report, studies)
            if args.strict_warnings:
                require(not report["warnings"], str(report["warnings"]))
            report["passed"] = True
    finally:
        report["seconds"] = round(time.monotonic() - started, 2)
        Path("report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
