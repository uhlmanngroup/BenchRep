from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

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
from benchrep.evaluation.embeddings.plotting import plot_2d_projection
from benchrep.records import (
    capture_console_streams,
    save_config_records,
    setup_run_logger,
    save_evaluation_metrics_json,
    export_reconstruction_tiffs,
)
from benchrep.runtime import RunContext
from benchrep.runtime.evaluate_run_validation import (
    prepare_evaluate_source_inputs,
)
from benchrep.assembly.registries.builtins import register_builtins


if TYPE_CHECKING:
    from benchrep.assembly.resolvers.evaluation_config_resolver import (
        EvaluationRunSpec,
    )


@dataclass
class EvaluationWorkflowResult:
    config: EvaluationConfig
    run_spec: EvaluationRunSpec
    run_context: RunContext
    adata: ad.AnnData
    reconstruction_outputs: dict[str, Any] | None
    metrics_path: Path


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

    with capture_console_streams(log_out_dir=run_context.log_dir, capture_stdout=False):
        adata = embeddings_pipeline.run(adata)

    run_log.info("Finished AnnData evaluation pipeline")
    run_log.info("Final obsm keys: %s", tuple(adata.obsm.keys()))
    run_log.info("Final obs columns: %s", tuple(adata.obs.columns))

    #FIXME
    # Quick-and-dirty plots for smoke testing
    if run_spec.step_spec.plots_enabled:
        run_log.info("Generating smoke-test plots...")
        write_smoke_test_plots(
            adata=adata,
            run_spec=run_spec,
            plot_dir=run_context.evaluation_embeddings_figures_dir,
            overwrite=True,
        )
        run_log.info("Finished generating smoke-test plots")

    # Create and run reconstructions evaluation pipeline
    reconstruction_outputs = None
    if reconstruction_input is not None:
        run_log.info("Starting reconstruction evaluation pipeline...")
        reconstruction_pipeline = create_reconstruction_evaluation_pipeline(run_spec)

        with capture_console_streams(
                log_out_dir=run_context.log_dir,
                capture_stdout=False,
        ):
            reconstruction_outputs = reconstruction_pipeline.run(reconstruction_input)

        if reconstruction_outputs:
            run_log.info(
                "Finished reconstruction evaluation pipeline with outputs: %s",
                tuple(reconstruction_outputs),
            )

            reconstruction_tiff_paths = export_reconstruction_tiffs(
                output_dir=run_context.evaluation_reconstructions_dir,
                reconstruction_input=reconstruction_input,
                reconstruction_outputs=reconstruction_outputs,
                overwrite=True,
            )
            run_log.info(
                "Saved reconstruction TIFF outputs: %s",
                {key: len(paths) for key, paths in reconstruction_tiff_paths.items()},
            )
        else:
            run_log.info(
                "Finished reconstruction evaluation pipeline with no outputs."
            )

    # Collect and export metrics
    metrics_path = save_evaluation_metrics_json(
        output_dir=run_context.evaluation_metrics_dir,
        adata=adata,
        reconstruction_outputs=reconstruction_outputs,
    )
    run_log.info("Saved evaluation metrics JSON to: '%s'", metrics_path)

    completed_at = datetime.now().isoformat(timespec="seconds")  # TODO use in eval manifest
    run_log.info("Evaluation completed at: %s", completed_at)

    return EvaluationWorkflowResult(
        config=run_spec.evaluation_config,
        run_spec=run_spec,
        run_context=run_context,
        adata=adata,
        reconstruction_outputs=reconstruction_outputs,
        metrics_path=metrics_path,
    )


def write_smoke_test_plots(
    *,
    adata: ad.AnnData,
    run_spec: "EvaluationRunSpec",
    plot_dir: Path,
    overwrite: bool = False,
) -> None:
    step_spec = run_spec.step_spec

    bases: list[str] = []

    if step_spec.pca_enabled:
        bases.append(step_spec.pca_params.get("key_added", "X_pca"))
    if step_spec.umap_enabled:
        bases.append(step_spec.umap_params.get("key_added", "X_umap"))
    if step_spec.tsne_enabled:
        bases.append(step_spec.tsne_params.get("key_added", "X_tsne"))

    color_by = step_spec.plot_params.get("color_by", [])
    if color_by is None:
        color_by = []

    plot_dir.mkdir(parents=True, exist_ok=True)

    for basis in bases:
        if basis not in adata.obsm:
            continue

        for color in color_by:
            if color not in adata.obs.columns:
                continue

            output_path = plot_dir / f"{basis}_colored_by_{color}.png"

            plot_2d_projection(
                adata,
                basis=basis,
                color=color,
                output_path=output_path,
                overwrite=overwrite,
            )
