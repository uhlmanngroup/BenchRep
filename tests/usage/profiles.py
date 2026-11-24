"""Small test-only edits to the real example configurations."""

from copy import deepcopy

MODULES = {
    "01": "01_yaml_pipeline",
    "02": "02_yaml_override_pipeline",
    "03": "03_config_object_pipeline",
    "04": "04_custom_dataset_pipeline",
    "05": "05_custom_loss_pipeline",
    "06": "06_schedule_and_monitor_training",
    "07": "07_learn_from_multiple_views",
    "08A": "08A_dual_input_multitask_composite_vae_yaml",
    "08B": "08B_dual_input_multitask_composite_vae_wizard",
    "09": "09_standalone_evaluation_workflow",
    "10": "10_optuna_composite_vae",
    "11": "11_pretrained_encoder_pipeline",
}
DATASET_LIMIT = 1024
PREDICTION_SAMPLES = 512
STANDALONE_SAMPLES = 1024
OPTUNA_TRIALS = 3


def training_budget(example):
    # Six epochs reach the original KL ramp (0–3) and LR milestone (5).
    return {
        "max_epochs": 6 if example == "06" else 2,
        "limit_train_batches": 2 if example == "06" else 4,
        "limit_val_batches": 2,
    }


def fast_config(original, example, *, device="cpu", skip_xgboost=False):
    """Return a budget-limited copy, with an optional explicit XGBoost exclusion."""
    raw = deepcopy(original)
    stage = raw.get("stage")
    batch_size = 8 if example == "11" else 16
    if stage == "training":
        trainer = raw.setdefault("trainer", {})
        trainer.update(training_budget(example))
        trainer.update(accelerator=device, devices=1, log_every_n_steps=1)
        dm = raw.setdefault("datamodule", {})
        dm.update(batch_size=batch_size, num_workers=0, persistent_workers=False)
        if "prefetch_factor" in dm:
            dm["prefetch_factor"] = None
        scheduler = (raw.get("optimizer") or {}).get("lr_scheduler")
        if scheduler and scheduler["name"] == "cosine_annealing":
            scheduler.setdefault("params", {})["T_max"] = trainer["max_epochs"]
    elif stage == "prediction":
        data = raw.setdefault("data", {})
        # Example 10 supplies a prediction datamodule with its own batch size.
        if example == "10":
            data["max_batches"] = 4
        else:
            data.update(batch_size=batch_size, num_workers=0,
                        max_batches=PREDICTION_SAMPLES // batch_size)
    elif stage == "evaluation":
        targets = (raw.get("metrics", {}).get("predictability") or {}).get("targets", {})
        for target in targets.values():
            if skip_xgboost:
                selected = target.get("selected")
                if selected == "all":
                    raise ValueError("XGBoost exclusion requires an explicit predictability probe list.")
                if isinstance(selected, list):
                    kept = [name for name in selected if name not in {"xgboost", "xgb"}]
                    if selected and not kept:
                        raise ValueError("XGBoost exclusion would leave no predictability probes.")
                    target["selected"] = kept
            target.setdefault("cv", {})["n_splits"] = 2
            tuning = target.get("tuning") or {}
            if tuning.get("enabled"):
                tuning.setdefault("inner_cv", {})["n_splits"] = 2
            for name, params in (target.get("params") or {}).items():
                if name in {"random_forest", "xgboost"} and isinstance(params, dict):
                    params.update(n_estimators=10, n_jobs=1)
        # Keep every plot type, color choice, channel and export format enabled.
        plots = raw.get("plots") or {}
        if plots.get("enabled"):
            plots.setdefault("params", {})["dpi"] = 72
    return raw


def changes(before, after, prefix=""):
    """Record exactly which values changed, including newly authored fields."""
    result = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before:
                result.append({"field": path, "before": "<omitted>", "after": after[key]})
            elif key not in after:
                result.append({"field": path, "before": before[key], "after": "<omitted>"})
            else:
                result.extend(changes(before[key], after[key], path))
    elif before != after:
        result.append({"field": prefix, "before": before, "after": after})
    return result
