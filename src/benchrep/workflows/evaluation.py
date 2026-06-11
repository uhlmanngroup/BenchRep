from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad

from benchrep.assembly import load_yaml
from benchrep.assembly.resolvers import resolve_evaluation_config
from benchrep.assembly.schemas import parse_evaluation_config
from benchrep.evaluation.pipelines import create_anndata_evaluation_pipeline
from benchrep.evaluation.plotting import plot_2d_projection
from benchrep.records import (
    capture_console_streams,
    save_config_records,
    setup_run_logger,
)
from benchrep.runtime import RunContext


if TYPE_CHECKING:
    from benchrep.assembly.resolvers.evaluation_config_resolver import (
        EvaluationRunSpec,
    )


def main() -> None:
    args = parse_args()

    # Parse and resolve config
    raw_eval_config_path = Path(args.config).resolve()
    raw_eval_config = load_yaml(raw_eval_config_path)
    eval_config = parse_evaluation_config(raw_eval_config)
    run_spec = resolve_evaluation_config(eval_config)

    # Setup paths
    run_context = RunContext.create(
        output_root=run_spec.run_identity.output_root,
        stage=run_spec.stage,
        run_name_stem=run_spec.run_identity.run_name_stem,
        project_name=run_spec.run_identity.project_name,
        model_name=run_spec.run_identity.model_name,
    )

    run_log = setup_run_logger(log_out_dir=run_context.log_dir)

    run_log.info("Evaluation run initialized from config: '%s'", raw_eval_config_path)
    run_log.info("Evaluation outputs will be saved to: '%s'", run_context.output_dir)
    run_log.info("Resolved embeddings path: '%s'", run_spec.input_spec.embeddings_path)

    # Bookkeeping --- config
    save_config_records(
        original_config_path=raw_eval_config_path,
        resolved_config=eval_config,
        config_out_dir=run_context.config_dir,
    )

    # Load embeddings AnnData
    run_log.info("Loading embeddings AnnData...")
    adata = ad.read_h5ad(run_spec.input_spec.embeddings_path)
    run_log.info(
        "Loaded AnnData with shape %s, obs columns=%s, obsm keys=%s",
        adata.shape,
        tuple(adata.obs.columns),
        tuple(adata.obsm.keys()),
    )

    # Create and run AnnData evaluation pipeline
    pipeline = create_anndata_evaluation_pipeline(run_spec)

    run_log.info("Starting AnnData evaluation pipeline...")

    with capture_console_streams(log_out_dir=run_context.log_dir, capture_stdout=False):
        adata = pipeline.run(adata)

    run_log.info("Finished AnnData evaluation pipeline")
    run_log.info("Final obsm keys: %s", tuple(adata.obsm.keys()))
    run_log.info("Final obs columns: %s", tuple(adata.obs.columns))

    # Quick-and-dirty plots for smoke testing
    if run_spec.step_spec.plots_enabled:
        run_log.info("Generating smoke-test plots...")
        write_smoke_test_plots(
            adata=adata,
            run_spec=run_spec,
            plot_dir=run_context.embedding_plots_dir,
            overwrite=True,
        )
        run_log.info("Finished generating smoke-test plots")

    # Quick-and-dirty metric print/log
    benchrep_results = adata.uns.get("benchrep", {})
    benchrep_metrics = benchrep_results.get("metrics", {})

    if benchrep_metrics:
        run_log.info("BenchRep eval metrics: %s", benchrep_metrics)
        print(benchrep_metrics)

    completed_at = datetime.now().isoformat(timespec="seconds")  # TODO use in eval manifest
    run_log.info("Evaluation completed at: %s", completed_at)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BenchRep evaluation from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()