from __future__ import annotations

import lightning as L

from benchrep.records import get_run_logger
from benchrep.runtime.utils import CompatibilityPolicy
from benchrep.interfaces.model_families import ModelFamilySpec
from benchrep.interfaces.compatibility import validate_external_model, sanity_check_predict_step_return_annotation


def validate_train_preconditions(
        model_family: ModelFamilySpec,
        model: L.LightningModule | None = None,
        datamodule: L.LightningDataModule | None = None,
        model_is_external: bool = False,
        datamodule_is_external: bool = False,
        compatibility_policy: CompatibilityPolicy = "error",
):
    external_model_only = model_is_external and not  datamodule_is_external
    external_datamodule_only = datamodule_is_external and not model_is_external
    fully_external_run = model_is_external and datamodule_is_external
    fully_internal_run = not model_is_external and not datamodule_is_external

    run_log = get_run_logger()

    if model_is_external:
        validate_external_model(model, model_family)

        try:
            sanity_check_predict_step_return_annotation(
                model=model,
                model_family=model_family,
                check_field_types=True,
            )

            run_log.info(
                "Training-time prediction-output annotation sanity check passed; "
                "this only checks declared annotations and does not guarantee that "
                "prediction/export/evaluation will succeed. Runtime compatibility "
                "will be checked during prediction."
            )

        except TypeError as exc:
            if compatibility_policy == "error":
                raise

            run_log.warning(
                "Training-time prediction-output annotation sanity check failed; "
                "continuing because compatibility_policy='warn'. "
                "BenchRep prediction/export/evaluation may fail later. Reason: %s",
                exc,
            )