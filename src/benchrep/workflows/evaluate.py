from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping
import logging

import anndata as ad

from benchrep.assembly.config import (
    compose_effective_config,
    SupportedConfigComponent,
)
from benchrep.assembly.resolvers import resolve_evaluation_config
from benchrep.assembly.schemas import EvaluationConfig
from benchrep.evaluation.pipelines import (
    create_anndata_evaluation_pipeline,
    create_reconstruction_evaluation_pipeline,
)
from benchrep.records import (
    capture_console_streams,
    save_config_records,
    setup_run_logger,
    export_evaluation_outputs,
    write_evaluation_manifest,
    write_audit_report,
    get_runtime_environment_filename,
    collect_evaluation_environment_context,
    write_runtime_environment,
)
from benchrep.records.utils import now_isoformat
from benchrep.runtime import RunContext
from benchrep.runtime.evaluate_run_validation import (
    prepare_evaluate_source_inputs,
    audit_evaluate_outputs,
)
from benchrep.assembly.registries.builtins import register_builtins
from benchrep.evaluation.status import (
    EvaluationOutcome,
    EvaluationStatusReport,
    build_evaluation_status_report,
)

if TYPE_CHECKING:
    from benchrep.assembly.resolvers.evaluation_config_resolver import (
        EvaluationRunSpec,
    )
    from benchrep.records.evaluation_exports import EvaluationExportPaths


@dataclass
class EvaluationWorkflowResult:
    config: EvaluationConfig
    run_spec: EvaluationRunSpec
    run_context: RunContext
    adata: ad.AnnData
    reconstruction_outputs: dict[str, Any] | None
    export_paths: EvaluationExportPaths
    status_report: EvaluationStatusReport
    manifest_path: Path
    audit_report_path: Path


def evaluate(
        config_path: Path | str | None = None,
        full_config_object: EvaluationConfig | None = None,
        config_components: Mapping[str, SupportedConfigComponent] | None = None,
        prediction_manifest_path: Path | str | None = None,
) -> EvaluationWorkflowResult:
    register_builtins()

    # Prediction manifest override
    if prediction_manifest_path is not None:
        prediction_manifest_path = Path(prediction_manifest_path).resolve()
        if prediction_manifest_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                "prediction_manifest_path override must point to a YAML file."
            )

    # Compose, parse, and resolve config
    config_composition_result = compose_effective_config(
        schema=EvaluationConfig,
        config_path=config_path,
        full_config_object=full_config_object,
        config_components=config_components,
        prediction_manifest_path_overridden=prediction_manifest_path is not None,
    )

    eval_config = config_composition_result.effective_config

    run_spec = resolve_evaluation_config(
        evaluation_config=eval_config,
        prediction_manifest_path_override=prediction_manifest_path,
    )

    # Setup paths
    run_context = RunContext.create(
        output_root=run_spec.run_identity.output_root,
        stage=run_spec.stage,
        run_name_stem=run_spec.run_identity.run_name_stem,
        project_name=run_spec.run_identity.project_name,
        model_name=run_spec.run_identity.model_name,
    )
    created_at = now_isoformat()

    # Initiate local run logger
    run_log = setup_run_logger(log_out_dir=run_context.log_dir)

    # Log composition messages and warnings
    for msg in config_composition_result.composition_messages:
        run_log.info(msg)
    for warning in config_composition_result.composition_warnings:
        run_log.warning(warning)

    run_log.info("Evaluation run initialized.")
    run_log.info(
        "Evaluation effective config source: '%s'",
        config_composition_result.effective_source,
    )
    run_log.info("Evaluation outputs will be saved to: '%s'", run_context.output_dir)
    run_log.info("Resolved embeddings path: '%s'", run_spec.input_spec.embeddings_path)

    # Bookkeeping --- config
    save_config_records(
        original_config_path=config_composition_result.original_config_path,
        resolved_config=run_spec.evaluation_config,
        config_out_dir=run_context.config_dir,
    )

    evaluation_environment_context = (
        collect_evaluation_environment_context(
            run_spec=run_spec,
        )
    )

    runtime_environment_path = write_runtime_environment(
        output_path=(
            run_context.metadata_dir
            / get_runtime_environment_filename(run_spec.stage)
        ),
        stage=run_spec.stage,
        run_name=run_context.run_name,
        workflow_context=evaluation_environment_context,
    )

    run_log.info(
        "Exported runtime environment to: '%s'",
        runtime_environment_path,
    )

    # Load and validate evaluation inputs
    run_log.info("Loading and validating evaluation source inputs...")
    source_inputs = prepare_evaluate_source_inputs(run_spec)

    adata = source_inputs.adata_input
    reconstruction_input = source_inputs.reconstruction_input

    run_log.info(
        "Loaded AnnData with shape %s, obs columns=%s, obsm keys=%s",
        adata.shape,
        tuple(adata.obs.columns),
        tuple(adata.obsm.keys()),
    )

    if reconstruction_input is None:
        run_log.info("No reconstruction evaluation inputs were loaded.")
    else:
        run_log.info(
            "Loaded reconstruction evaluation inputs with input shape=%s, "
            "reconstruction shape=%s, n_examples=%s, metadata=%s",
            reconstruction_input.inputs.shape,
            reconstruction_input.reconstructions.shape,
            reconstruction_input.n_examples,
            reconstruction_input.metadata is not None,
        )

    # Create and run AnnData evaluation pipeline
    embeddings_pipeline = create_anndata_evaluation_pipeline(run_spec)

    run_log.info("Starting AnnData evaluation pipeline...")

    with capture_console_streams(
        log_out_dir=run_context.log_dir,
        capture_stdout=False,
    ):
        adata = embeddings_pipeline.run(adata)

    run_log.info("Finished AnnData evaluation pipeline.")
    run_log.info("Final obsm keys: %s", tuple(adata.obsm.keys()))
    run_log.info("Final obs columns: %s", tuple(adata.obs.columns))

    # Create and run reconstruction evaluation pipeline
    reconstruction_outputs = None
    reconstruction_outcomes: tuple[EvaluationOutcome, ...] = ()

    if reconstruction_input is not None:
        run_log.info("Starting reconstruction evaluation pipeline...")

        reconstruction_pipeline = create_reconstruction_evaluation_pipeline(
            run_spec
        )

        with capture_console_streams(
            log_out_dir=run_context.log_dir,
            capture_stdout=False,
        ):
            reconstruction_outputs = reconstruction_pipeline.run(
                reconstruction_input
            )

        if reconstruction_outputs:
            run_log.info(
                "Finished reconstruction evaluation pipeline with outputs: %s",
                tuple(reconstruction_outputs),
            )
        else:
            run_log.info(
                "Finished reconstruction evaluation pipeline with no outputs."
            )

        reconstruction_outcomes = reconstruction_pipeline.outcomes

    # Export evaluation artifacts
    run_log.info("Starting evaluation artifact export...")

    export_result = export_evaluation_outputs(
        adata=adata,
        reconstruction_input=reconstruction_input,
        reconstruction_outputs=reconstruction_outputs,
        step_spec=run_spec.step_spec,
        embeddings_dir=run_context.evaluation_embeddings_dir,
        embeddings_figures_dir=run_context.evaluation_embeddings_figures_dir,
        metrics_dir=run_context.evaluation_metrics_dir,
        reconstructions_dir=run_context.evaluation_reconstructions_dir,
        reconstruction_figures_dir=(
            run_context.evaluation_reconstructions_figures_dir
        ),
        overwrite=False,
    )

    export_paths = export_result.paths

    run_log.info("Finished evaluation artifact export.")

    status_report = build_evaluation_status_report(
        embedding_outcomes=embeddings_pipeline.outcomes,
        reconstruction_outcomes=reconstruction_outcomes,
        export_outcomes=export_result.outcomes,
    )

    _log_evaluation_status_report(
        run_log=run_log,
        status_report=status_report,
    )

    completed_at = now_isoformat()

    # Export evaluation manifest
    manifest_path = run_context.metadata_dir / "evaluation_manifest.yaml"
    write_evaluation_manifest(
        config_composition_result=config_composition_result,
        output_path=manifest_path,
        run_spec=run_spec,
        run_context=run_context,
        adata=adata,
        export_paths=export_paths,
        status_report=status_report,
        created_at=created_at,
        completed_at=completed_at,
    )

    run_log.info("Exported evaluation manifest to: '%s'", manifest_path)

    audit_items = audit_evaluate_outputs(
        run_context=run_context,
        run_spec=run_spec,
        adata=adata,
        reconstruction_outputs=reconstruction_outputs,
        export_paths=export_paths,
        config_composition_result=config_composition_result,
        resolved_config_path=run_context.config_dir / "resolved_config.yaml",
        evaluation_manifest_path=manifest_path,
        runtime_environment_path=runtime_environment_path,
    )

    audit_report_path = write_audit_report(
        stage="evaluation",
        audit_items=audit_items,
        output_path=(
                run_context.metadata_dir / "evaluation_audit_report.yaml"
        ),
        audited_at=now_isoformat(),
    )

    run_log.info(
        "Exported evaluation audit report to: '%s'",
        audit_report_path,
    )

    return EvaluationWorkflowResult(
        config=run_spec.evaluation_config,
        run_spec=run_spec,
        run_context=run_context,
        adata=adata,
        reconstruction_outputs=reconstruction_outputs,
        export_paths=export_paths,
        status_report=status_report,
        manifest_path=manifest_path,
        audit_report_path=audit_report_path,
    )


def _log_evaluation_status_report(
    *,
    run_log: logging.Logger,
    status_report: EvaluationStatusReport,
) -> None:
    """Log evaluation section statuses and their recorded issues."""

    sections = (
        ("embeddings", status_report.embeddings),
        ("reconstructions", status_report.reconstructions),
        ("exports", status_report.exports),
    )

    for section_name, section in sections:
        run_log.info(
            "Evaluation %s status: %s",
            section_name,
            section.status,
        )

        for outcome in section.outcomes:
            for issue in outcome.issues:
                run_log.warning(
                    "Evaluation %s outcome %r [%s]: %s",
                    section_name,
                    outcome.name,
                    outcome.status,
                    issue,
                )

    log_workflow_status = (
        run_log.info
        if status_report.status == "completed"
        else run_log.warning
    )
    log_workflow_status(
        "Evaluation workflow status: %s",
        status_report.status,
    )
