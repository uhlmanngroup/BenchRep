from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast
import warnings

import torch

from benchrep.interfaces.contracts import (
    CompositePredictionOutput,
)
from benchrep.assembly.resolvers.prediction_config_resolver import (
    PredictionAnnDataExportSpec,
    PredictionExportSpec,
    PredictionReconstructionObservationSpec,
    PredictionReconstructionPairSpec,
    PredictionReconstructionsExportSpec,
)
from benchrep.records.logs import get_run_logger
from benchrep.records.anndata_io import (
    package_matrix_as_anndata,
    write_h5ad,
)
from benchrep.runtime.status import PredictionOutcome, PredictionOutcomeStatus





@dataclass(frozen=True)
class PredictionAnnDataExportResult:
    """AnnData artifact location and export outcome.

    Attributes
    ----------
    path : Path | None
        Written AnnData path, or ``None`` if no file was exported.
    outcome : PredictionOutcome
        AnnData export outcome and issues.
    """
    path: Path | None
    outcome: PredictionOutcome


@dataclass(frozen=True)
class PredictionReconstructionBundlePaths:
    """Locations and sampling counts for a reconstruction bundle.

    Attributes
    ----------
    bundle_dir : Path
        Bundle directory; may be a planned location if export failed.
    input_path : Path | None
        Exported input array path, or ``None``.
    reconstruction_path : Path | None
        Exported reconstruction array path, or ``None``.
    obs_path : Path | None
        Observation annotations path, or ``None``.
    metadata_path : Path | None
        Bundle metadata path, or ``None``.
    n_examples_exported : int | None
        Number of exported examples when known.
    n_strata : int | None
        Number of sampling strata when stratified sampling was used.
    n_represented_strata : int | None
        Number of strata represented in the exported selection, when known.
    n_omitted_strata : int | None
        Number of strata omitted from the exported selection, when known.
    """
    bundle_dir: Path
    input_path: Path | None = None
    reconstruction_path: Path | None = None
    obs_path: Path | None = None
    metadata_path: Path | None = None
    n_examples_exported: int | None = None
    n_strata: int | None = None
    n_represented_strata: int | None = None
    n_omitted_strata: int | None = None


@dataclass(frozen=True)
class PredictionReconstructionPairExportResult:
    """Export result for one declared input–reconstruction pair.

    Attributes
    ----------
    pair : PredictionReconstructionPairSpec
        Resolved specification identifying the input and reconstruction being exported.
    paths : PredictionReconstructionBundlePaths
        Bundle paths and selection counts.
    outcome : PredictionOutcome
        Export outcome for this pair.
    """
    pair: PredictionReconstructionPairSpec
    paths: PredictionReconstructionBundlePaths
    outcome: PredictionOutcome


@dataclass(frozen=True)
class PredictionReconstructionsExportResult:
    """Per-pair reconstruction exports and their aggregate outcome.

    Attributes
    ----------
    pairs : tuple[PredictionReconstructionPairExportResult, ...]
        Results for the requested reconstruction pairs.
    outcome : PredictionOutcome
        Aggregate reconstruction export outcome.
    """
    pairs: tuple[PredictionReconstructionPairExportResult, ...]
    outcome: PredictionOutcome


@dataclass(frozen=True)
class PredictionExportResult:
    """Artifact export results attached to a prediction workflow result.

    Attributes
    ----------
    anndata : PredictionAnnDataExportResult
        AnnData export path and outcome.
    reconstructions : PredictionReconstructionsExportResult
        Reconstruction bundle results and aggregate outcome.
    """
    anndata: PredictionAnnDataExportResult
    reconstructions: PredictionReconstructionsExportResult

    @property
    def outcomes(self) -> tuple[PredictionOutcome, PredictionOutcome]:
        """Return the AnnData and reconstruction outcomes, in that order."""
        return (
            self.anndata.outcome,
            self.reconstructions.outcome,
        )


@dataclass(frozen=True)
class ReconstructionSelectionResult:
    indices: torch.Tensor
    n_strata: int | None = None
    n_represented_strata: int | None = None
    n_omitted_strata: int = 0



def export_prediction_outputs(
    *,
    predictions: Sequence[object],
    export_spec: PredictionExportSpec,
    anndata_dir: Path,
    reconstruction_dir: Path,
) -> PredictionExportResult:
    """Export all prediction artifacts requested by a resolved export spec.

    YOYO exports read selected tensor fields from mapping keys or attributes
    and check their shapes and per-batch alignment at runtime. Tensor fields
    selected by any enabled export take precedence over optional annotation
    names throughout the export plan, including across reconstruction pairs.

    AnnData export and reconstruction export are isolated branches: failure in
    one branch does not prevent the other from running. Reconstruction pairs
    are isolated from one another as well, so one failed pair does not prevent
    the remaining pairs from being exported.

    Warnings and exceptions raised by an export branch are converted into
    ``PredictionOutcome`` records rather than propagated. Structural problems
    with the resolved export plan itself, such as enabled reconstruction export
    containing no pairs, are treated as internal errors and may still raise.

    Parameters
    ----------
    predictions:
        Batch-level outputs returned by ``Trainer.predict()``.
    export_spec:
        Fully resolved export specification, including concrete AnnData keys,
        primary key, and reconstruction pairs.
    anndata_dir:
        Directory in which the AnnData artifact is written.
    reconstruction_dir:
        Parent directory containing one ordered subdirectory per reconstruction
        pair.

    Returns
    -------
    PredictionExportResult
        Nested AnnData and reconstruction results containing artifact paths,
        branch outcomes, and per-pair reconstruction outcomes.
    """

    tensor_keys = set(export_spec.anndata.keys) if export_spec.anndata.enabled else set()
    reconstruction_spec = export_spec.reconstructions
    if reconstruction_spec.enabled:
        for pair in reconstruction_spec.pairs:
            if reconstruction_spec.include_input:
                tensor_keys.add(pair.input)
            if reconstruction_spec.include_reconstruction:
                tensor_keys.add(pair.reconstruction)

    anndata_result = PredictionAnnDataExportResult(
        path=None,
        outcome=PredictionOutcome(
            name="anndata",
            status="disabled",
        ),
    )

    if export_spec.anndata.enabled:
        captured_warnings: list[warnings.WarningMessage] = []

        try:
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")

                anndata_dir.mkdir(parents=True, exist_ok=True)

                anndata_path = _export_anndata(
                    predictions=predictions,
                    anndata_spec=export_spec.anndata,
                    tensor_keys=tensor_keys,
                    output_dir=anndata_dir,
                )

        except Exception as exc:
            warning_issues = _format_captured_warnings(
                captured_warnings
            )
            error_issue = f"Error ({type(exc).__name__}): {exc}"
            issues = (*warning_issues, error_issue)

            run_log = get_run_logger()

            for issue in warning_issues:
                run_log.warning("AnnData export: %s", issue)

            run_log.error(
                "AnnData export failed: %s",
                error_issue,
                exc_info=True,
            )

            anndata_result = PredictionAnnDataExportResult(
                path=None,
                outcome=PredictionOutcome(
                    name="anndata",
                    status="failed",
                    issues=issues,
                ),
            )

        else:
            issues = _format_captured_warnings(captured_warnings)

            for issue in issues:
                get_run_logger().warning(
                    "AnnData export: %s",
                    issue,
                )

            status: PredictionOutcomeStatus = (
                "completed_with_warnings"
                if issues
                else "completed"
            )

            anndata_result = PredictionAnnDataExportResult(
                path=anndata_path,
                outcome=PredictionOutcome(
                    name="anndata",
                    status=status,
                    issues=issues,
                ),
            )

    pair_results: list[
        PredictionReconstructionPairExportResult
    ] = []

    if export_spec.reconstructions.enabled:
        for pair in export_spec.reconstructions.pairs:
            bundle_dir = reconstruction_dir / pair.id
            bundle_paths = PredictionReconstructionBundlePaths(
                bundle_dir=bundle_dir,
            )
            captured_warnings = []

            try:
                with warnings.catch_warnings(
                    record=True
                ) as captured_warnings:
                    warnings.simplefilter("always")

                    bundle_dir.mkdir(parents=True, exist_ok=True)

                    bundle_paths = _export_reconstruction_pair(
                        predictions=predictions,
                        reconstruction_spec=(
                            export_spec.reconstructions
                        ),
                        pair=pair,
                        tensor_keys=tensor_keys,
                        output_dir=bundle_dir,
                    )

            except Exception as exc:
                warning_issues = _format_captured_warnings(
                    captured_warnings
                )
                error_issue = f"Error ({type(exc).__name__}): {exc}"
                issues = (*warning_issues, error_issue)

                run_log = get_run_logger()

                for issue in warning_issues:
                    run_log.warning(
                        "Reconstruction pair %r: %s",
                        pair.id,
                        issue,
                    )

                run_log.error(
                    "Reconstruction pair %r export failed: %s",
                    pair.id,
                    error_issue,
                    exc_info=True,
                )

                pair_outcome = PredictionOutcome(
                    name=f"reconstructions.{pair.id}",
                    status="failed",
                    issues=issues,
                )

            else:
                issues = _format_captured_warnings(
                    captured_warnings
                )

                for issue in issues:
                    get_run_logger().warning(
                        "Reconstruction pair %r: %s",
                        pair.id,
                        issue,
                    )

                pair_status: PredictionOutcomeStatus = (
                    "completed_with_warnings"
                    if issues
                    else "completed"
                )

                pair_outcome = PredictionOutcome(
                    name=f"reconstructions.{pair.id}",
                    status=pair_status,
                    issues=issues,
                )

            pair_results.append(
                PredictionReconstructionPairExportResult(
                    pair=pair,
                    paths=bundle_paths,
                    outcome=pair_outcome,
                )
            )

        reconstruction_outcome = (
            _aggregate_reconstruction_pair_outcomes(pair_results)
        )

    else:
        reconstruction_outcome = PredictionOutcome(
            name="reconstructions",
            status="disabled",
        )

    return PredictionExportResult(
        anndata=anndata_result,
        reconstructions=PredictionReconstructionsExportResult(
            pairs=tuple(pair_results),
            outcome=reconstruction_outcome,
        ),
    )


def _format_captured_warnings(
    captured_warnings: Sequence[warnings.WarningMessage],
) -> tuple[str, ...]:
    return tuple(
        f"Warning ({type(warning.message).__name__}): "
        f"{warning.message}"
        for warning in captured_warnings
    )


def _aggregate_reconstruction_pair_outcomes(
    pair_results: Sequence[
        PredictionReconstructionPairExportResult
    ],
) -> PredictionOutcome:
    """Summarize all reconstruction-pair outcomes as one branch outcome.

    Complete success produces ``completed`` or ``completed_with_warnings``.
    Mixed successful and failed pairs produce ``partially_completed``. If every
    pair fails, the complete reconstruction branch is marked ``failed``. Issues
    from individual pairs are retained with their pair IDs for attribution.
    """

    if not pair_results:
        raise RuntimeError(
            "Enabled reconstruction export resolved no reconstruction pairs."
        )

    pair_outcomes = tuple(
        result.outcome
        for result in pair_results
    )
    n_failed = sum(
        outcome.status == "failed"
        for outcome in pair_outcomes
    )

    issues = tuple(
        f"Pair {result.pair.id!r}: {issue}"
        for result in pair_results
        for issue in result.outcome.issues
    )

    if n_failed == len(pair_outcomes):
        status: PredictionOutcomeStatus = "failed"
    elif n_failed:
        status = "partially_completed"
    elif any(
        outcome.status == "completed_with_warnings"
        for outcome in pair_outcomes
    ):
        status = "completed_with_warnings"
    else:
        status = "completed"

    status: PredictionOutcomeStatus

    return PredictionOutcome(
        name="reconstructions",
        status=status,
        issues=issues,
    )


def _export_anndata(
    *,
    predictions: Sequence[object],
    anndata_spec: PredictionAnnDataExportSpec,
    tensor_keys: set[str],
    output_dir: Path,
) -> Path:
    """Package resolved non-image prediction outputs as one AnnData artifact.

    The resolved primary vector is stored in ``adata.X``. Additional selected
    vector outputs are stored in ``adata.obsm`` under their declared names,
    while selected scalar outputs are stored in ``adata.obs``.

    For YOYO models, [samples] tensors are scalar and [samples, features]
    tensors are vector-valued. The primary output must be a vector. Selected
    fields must agree on per-batch sample counts.

    Canonical prediction annotations are collected from ``sample_id``, ``label``,
    and ``metadata``. Composite annotations follow the resolved observation
    specifications: declared scalar model inputs and group metadata become
    ``adata.obs`` columns, while an optional metadata declaration with role
    ``index`` supplies ``adata.obs_names``.

    Parameters
    ----------
    predictions:
        Prediction batches from one canonical, Composite, or YOYO model.
    anndata_spec:
        Resolved AnnData export plan containing selected output keys, their
        scalar/vector structures when declared, the primary key, and observation
        sources. YOYO output structures are inferred here.
    tensor_keys:
        Tensor fields selected across all enabled exports; these names are not
        interpreted as optional annotations.
    output_dir:
        Directory in which ``anndata.h5ad`` is written.

    Returns
    -------
    Path
        Path to the successfully written AnnData artifact.
    """

    if not predictions:
        raise ValueError(
            "AnnData export requires at least one prediction batch."
        )

    primary_key = anndata_spec.primary_key
    assert primary_key is not None

    primary_matrix, batch_sizes = _concat_prediction_tensor_batches(
        predictions=predictions,
        key=primary_key,
        source="model_output",
    )
    if primary_matrix.ndim != 2:
        raise ValueError(
            f"AnnData primary key {primary_key!r} must provide a "
            f"[samples, features] tensor; got {tuple(primary_matrix.shape)}."
        )
    output_structures = dict(anndata_spec.output_structures_by_key)
    output_structures.setdefault(primary_key, "vector")
    sample_ids, labels, metadata, index_name = _collect_anndata_annotations(
        predictions=predictions,
        anndata_spec=anndata_spec,
        batch_sizes=batch_sizes,
        tensor_keys=tensor_keys,
    )
    adata = package_matrix_as_anndata(
        primary_matrix,
        sample_ids=sample_ids,
        labels=labels,
        metadata=metadata,
    )

    if index_name is not None:
        adata.obs.index.name = index_name

    for key in anndata_spec.keys:
        if key == primary_key or output_structures.get(key) == "scalar":
            # Declared Composite scalar outputs are already annotations.
            continue
        values, value_batch_sizes = _concat_prediction_tensor_batches(
            predictions=predictions,
            key=key,
            source="model_output",
        )
        if value_batch_sizes != batch_sizes:
            raise ValueError(
                f"AnnData output {key!r} has per-batch sizes {value_batch_sizes}; "
                f"primary output {primary_key!r} has {batch_sizes}."
            )
        if values.ndim not in {1, 2}:
            raise ValueError(
                f"AnnData output {key!r} must provide [samples] or "
                f"[samples, features]; got {tuple(values.shape)}."
            )
        structure = output_structures.get(key)
        if structure is None:
            structure = "scalar" if values.ndim == 1 else "vector"
            output_structures[key] = structure
        if structure == "scalar":
            if key in adata.obs.columns:
                raise ValueError(
                    f"AnnData output {key!r} conflicts with an annotation "
                    "column. Rename the output or annotation."
                )
            if adata.obs.index.name == key:
                adata.obs.index.name = None
            adata.obs[key] = values.numpy()
        else:
            if values.ndim != 2:
                raise ValueError(
                    f"AnnData vector output {key!r} must provide "
                    f"[samples, features]; got {tuple(values.shape)}."
                )
            adata.obsm[key] = values.numpy()

    benchrep_uns = adata.uns.setdefault("benchrep", {})
    benchrep_uns["prediction_export"] = {
        "mode": anndata_spec.mode,
        "keys": list(anndata_spec.keys),
        "primary_key": primary_key,
        "output_structures_by_key": dict(output_structures),
        "observations": {
            observation.name: {
                "source": observation.source,
                "use_as_index": observation.use_as_index,
            }
            for observation in anndata_spec.observations
        },
    }

    output_path = output_dir / "anndata.h5ad"

    write_h5ad(
        adata,
        output_path,
        overwrite=True,
    )

    return output_path


def _collect_anndata_annotations(
    *,
    predictions: Sequence[object],
    anndata_spec: PredictionAnnDataExportSpec,
    batch_sizes: Sequence[int],
    tensor_keys: set[str],
) -> tuple[
    list[Any] | None,
    list[Any] | None,
    dict[str, list[Any]] | None,
    str | None,
]:
    """Collect observation annotations aligned with exported tensor batches.

    Canonical outputs retain their established ``sample_id``, ``label``, and
    free-form ``metadata`` behavior. Composite outputs use only observation
    sources explicitly resolved from their declarations. YOYO outputs may
    use the same optional annotation names through mapping keys or attributes.
    Tensor keys selected by any enabled export take precedence over annotation
    names, including keys selected only by a reconstruction pair.
    """

    first_prediction = predictions[0]

    if not isinstance(first_prediction, CompositePredictionOutput):
        return (
            (
                None
                if "sample_id" in tensor_keys
                else _concat_optional_batch_values(
                    predictions=predictions,
                    key="sample_id",
                    batch_sizes=batch_sizes,
                )
            ),
            (
                None
                if "label" in tensor_keys
                else _concat_optional_batch_values(
                    predictions=predictions,
                    key="label",
                    batch_sizes=batch_sizes,
                )
            ),
            (
                None
                if "metadata" in tensor_keys
                else _concat_optional_metadata(
                    predictions=predictions,
                    batch_sizes=batch_sizes,
                )
            ),
            None,
        )

    sample_ids: list[Any] | None = None
    index_name: str | None = None
    metadata: dict[str, list[Any]] = {}

    for observation in anndata_spec.observations:
        values = _concat_prediction_observation_values(
            predictions=predictions,
            source=observation.source,
            key=observation.name,
        )

        if observation.use_as_index:
            sample_ids = values
            index_name = observation.name
        else:
            metadata[observation.name] = values

    return (
        sample_ids,
        None,
        metadata or None,
        index_name,
    )


def _get_prediction_field(prediction: object, key: str) -> Any:
    """Read an optional field from a mapping or an object's attributes."""
    if isinstance(prediction, Mapping):
        return prediction.get(key)
    return getattr(prediction, key, None)


def _concat_prediction_tensor_batches(
    *,
    predictions: Sequence[object],
    key: str,
    source: Literal["model_input", "model_output"],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Collect a named tensor and retain batch sizes for alignment checks.

    Composite outputs retain their input/output namespaces. Other outputs
    provide mapping keys or attributes. Every tensor needs a batch dimension.
    """
    tensors: list[torch.Tensor] = []
    batch_sizes: list[int] = []
    for batch_idx, prediction in enumerate(predictions):
        if isinstance(prediction, CompositePredictionOutput):
            namespace = (
                prediction.model_inputs
                if source == "model_input"
                else prediction.model_outputs
            )
            value = namespace.get(key)
        else:
            value = _get_prediction_field(prediction, key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Prediction batch {batch_idx} {source} field {key!r} "
                f"must provide a tensor; got {type(value).__name__}."
            )
        if value.ndim == 0:
            raise ValueError(
                f"Prediction batch {batch_idx} field {key!r} "
                "must have a batch dimension."
            )
        tensors.append(value.detach().cpu())
        batch_sizes.append(int(value.shape[0]))
    try:
        combined = torch.cat(tensors, dim=0)
    except RuntimeError as exc:
        raise ValueError(
            f"Could not concatenate prediction field {key!r}: {exc}"
        ) from exc
    return combined, tuple(batch_sizes)


def _concat_prediction_observation_values(
    *,
    predictions: Sequence[object],
    source: str,
    key: str,
) -> list[Any]:
    """Concatenate one resolved Composite observation source."""
    values: list[Any] = []

    for prediction in predictions:
        composite_prediction = cast(
            CompositePredictionOutput,
            prediction,
        )

        if source == "model_output":
            value = composite_prediction.model_outputs[key]
        elif source == "model_input":
            value = composite_prediction.model_inputs[key]
        else:
            value = composite_prediction.batch_metadata[key]

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()

            if tensor.ndim == 2:
                tensor = tensor[:, 0]

            values.extend(tensor.tolist())
        else:
            values.extend(value)

    return values


def _concat_optional_batch_values(
    *,
    predictions: Sequence[object],
    key: str,
    batch_sizes: Sequence[int],
) -> list[Any] | None:
    """Collect optional annotations, requiring one value per sample in each batch."""
    batches = [_get_prediction_field(prediction, key) for prediction in predictions]
    if all(value is None for value in batches):
        return None

    values: list[Any] = []
    for batch_idx, (value, batch_size) in enumerate(
        zip(batches, batch_sizes, strict=True)
    ):
        if value is None or isinstance(value, (str, bytes, Mapping)):
            raise TypeError(
                f"Prediction batch {batch_idx} annotation {key!r} "
                "must provide a sequence of per-sample values."
            )
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        try:
            batch_values = list(value)
        except TypeError as exc:
            raise TypeError(
                f"Prediction batch {batch_idx} annotation {key!r} "
                "must provide a sequence of per-sample values."
            ) from exc
        if len(batch_values) != batch_size:
            raise ValueError(
                f"Prediction batch {batch_idx} annotation {key!r} has "
                f"{len(batch_values)} values; expected {batch_size}."
            )
        values.extend(batch_values)
    return values


def _concat_optional_metadata(
    *,
    predictions: Sequence[object],
    batch_sizes: Sequence[int],
) -> dict[str, list[Any]] | None:
    """Collect optional metadata with consistent columns and aligned batch rows."""
    batches = [
        _get_prediction_field(prediction, "metadata") for prediction in predictions
    ]
    if all(metadata is None for metadata in batches):
        return None
    keys: tuple[str, ...] = ()
    for batch_idx, metadata in enumerate(batches):
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Prediction batch {batch_idx} metadata must be a mapping.")
        if batch_idx == 0:
            keys = tuple(metadata)
        elif set(metadata) != set(keys):
            raise ValueError(
                f"Prediction batch {batch_idx} metadata columns differ from the first batch."
            )
    merged_metadata: dict[str, list[Any]] = {}
    for key in keys:
        values = _concat_optional_batch_values(
            predictions=batches,
            key=key,
            batch_sizes=batch_sizes,
        )
        if values is None:
            raise ValueError(
                f"Prediction metadata field {key!r} must provide per-sample values."
            )
        merged_metadata[key] = values
    return merged_metadata or None


def _select_reconstruction_indices(
    *,
    n_samples: int,
    n_examples: int | Literal["all"],
    selection: Literal["first", "random"],
    seed: int | None,
    stratify_by: str | None = None,
    stratify_values: Sequence[Any] | None = None,
) -> ReconstructionSelectionResult:
    """Select row indices for reconstruction export.

    ``selection="first"`` returns the first requested rows.
    ``selection="random"`` samples without replacement using a local torch
    generator seeded by ``seed``. When ``stratify_by`` is set, random sampling
    is distributed approximately evenly across strata.

    If ``n_examples="all"``, all rows are selected. If an integer larger than
    ``n_samples`` is requested, selection is capped at ``n_samples``.
    """
    if n_examples == "all":
        return ReconstructionSelectionResult(
            indices=torch.arange(n_samples),
        )

    n_selected = min(n_examples, n_samples)

    if selection == "first":
        if stratify_by is not None:
            raise ValueError(
                "Stratified reconstruction selection requires "
                "selection='random'."
            )

        return ReconstructionSelectionResult(
            indices=torch.arange(n_selected),
        )

    if selection == "random":
        if seed is None:
            raise ValueError(
                "Random reconstruction selection requires a seed."
            )

        if stratify_by is not None:
            if stratify_values is None:
                raise ValueError(
                    "stratify_values must be provided when stratify_by is set."
                )

            if len(stratify_values) != n_samples:
                raise ValueError(
                    f"Reconstruction stratification field {stratify_by!r} "
                    f"contains {len(stratify_values)} values, expected "
                    f"{n_samples}."
                )

            return _select_stratified_reconstruction_indices(
                stratify_values=stratify_values,
                stratify_by=stratify_by,
                n_selected=n_selected,
                seed=seed,
            )

        generator = torch.Generator().manual_seed(seed)

        return ReconstructionSelectionResult(
            indices=torch.randperm(
                n_samples,
                generator=generator,
            )[:n_selected],
        )

    raise ValueError(
        f"Unsupported reconstruction selection {selection!r}. "
        "Available options: 'first', 'random'."
    )


def _export_reconstruction_pair(
    *,
    predictions: Sequence[object],
    reconstruction_spec: PredictionReconstructionsExportSpec,
    pair: PredictionReconstructionPairSpec,
    tensor_keys: set[str],
    output_dir: Path,
) -> PredictionReconstructionBundlePaths:
    """Export one resolved input/reconstruction pair as an artifact bundle.

    Input and reconstruction tensors are collected from the pair's resolved
    prediction namespaces, after which one shared selection is applied to every
    exported tensor and observation. The bundle contains optional input and
    reconstruction tensors, selected observations, and metadata describing the
    pair and selection result. YOYO tensor fields are read from mapping
    keys or attributes, and their per-batch sample counts must agree.

    Parameters
    ----------
    predictions:
        Batch-level prediction outputs; requested fields are checked here.
    reconstruction_spec:
        Resolved settings shared by every reconstruction pair.
    pair:
        Resolved pair identifying the model input and reconstruction output.
    tensor_keys:
        Tensor fields selected across all enabled exports; these names are not
        interpreted as optional annotations.
    output_dir:
        Dedicated output directory for this pair.

    Returns
    -------
    PredictionReconstructionBundlePaths
        Paths and selection statistics for the written bundle.
    """
    if not predictions:
        raise ValueError(
            "Reconstruction export requires at least one prediction batch."
        )

    input_tensor = None
    reconstruction_tensor = None
    batch_sizes: tuple[int, ...] | None = None
    if reconstruction_spec.include_input:
        input_tensor, batch_sizes = _concat_prediction_tensor_batches(
            predictions=predictions,
            key=pair.input,
            source="model_input",
        )
    if reconstruction_spec.include_reconstruction:
        reconstruction_tensor, reconstruction_batch_sizes = (
            _concat_prediction_tensor_batches(
                predictions=predictions,
                key=pair.reconstruction,
                source="model_output",
            )
        )
        if batch_sizes is not None and reconstruction_batch_sizes != batch_sizes:
            raise ValueError(
                f"Reconstruction pair {pair.id!r} has different per-batch "
                f"sizes: input={batch_sizes}, reconstruction={reconstruction_batch_sizes}."
            )
        batch_sizes = reconstruction_batch_sizes
    assert batch_sizes is not None
    n_samples = sum(batch_sizes)
    observations = _collect_reconstruction_observations(
        predictions=predictions,
        observation_specs=reconstruction_spec.observations,
        batch_sizes=batch_sizes,
        tensor_keys=tensor_keys,
    )

    stratify_values = None

    if reconstruction_spec.stratify_by is not None:
        if reconstruction_spec.stratify_by not in observations:
            raise KeyError(
                "Reconstruction export requested stratification by "
                f"{reconstruction_spec.stratify_by!r}, but that observation "
                "is unavailable. Available observations: "
                f"{sorted(observations)}."
            )

        stratify_values = observations[
            reconstruction_spec.stratify_by
        ]

    selection_result = _select_reconstruction_indices(
        n_samples=n_samples,
        n_examples=reconstruction_spec.n_examples,
        selection=reconstruction_spec.selection,
        seed=reconstruction_spec.seed,
        stratify_by=reconstruction_spec.stratify_by,
        stratify_values=stratify_values,
    )

    selected_indices = selection_result.indices
    selected_indices_list = selected_indices.tolist()

    if selection_result.n_omitted_strata > 0:
        warnings.warn(
            "Reconstruction export was stratified by "
            f"{reconstruction_spec.stratify_by!r}, but only "
            f"{selection_result.n_represented_strata} of "
            f"{selection_result.n_strata} strata could be represented "
            f"within the requested {len(selected_indices)} examples. "
            f"{selection_result.n_omitted_strata} strata were omitted.",
            RuntimeWarning,
            stacklevel=2,
        )

    input_path = None

    if input_tensor is not None:
        input_path = output_dir / "input.pt"
        torch.save(
            input_tensor[selected_indices],
            input_path,
        )

    reconstruction_path = None

    if reconstruction_tensor is not None:
        reconstruction_path = output_dir / "reconstruction.pt"
        torch.save(
            reconstruction_tensor[selected_indices],
            reconstruction_path,
        )

    selected_observations: dict[str, list[Any]] = {
        "source_index": selected_indices_list,
    }

    selected_observations.update(
        {
            key: [
                values[index]
                for index in selected_indices_list
            ]
            for key, values in observations.items()
        }
    )

    obs_path = output_dir / "obs.pt"
    torch.save(selected_observations, obs_path)

    metadata_path = output_dir / "reconstruction_export_metadata.pt"
    torch.save(
        {
            "pair": {
                "id": pair.id,
                "input": pair.input,
                "reconstruction": pair.reconstruction,
            },
            "n_samples_total": n_samples,
            "n_examples_exported": len(selected_indices),
            "selection": reconstruction_spec.selection,
            "stratify_by": reconstruction_spec.stratify_by,
            "seed": reconstruction_spec.seed,
            "n_strata": selection_result.n_strata,
            "n_represented_strata": (
                selection_result.n_represented_strata
            ),
            "n_omitted_strata": (
                selection_result.n_omitted_strata
            ),
            "include_input": reconstruction_spec.include_input,
            "include_reconstruction": (
                reconstruction_spec.include_reconstruction
            ),
            "exported_keys": [
                key
                for key, tensor in (
                    ("input", input_tensor),
                    ("reconstruction", reconstruction_tensor),
                )
                if tensor is not None
            ],
        },
        metadata_path,
    )

    return PredictionReconstructionBundlePaths(
        bundle_dir=output_dir,
        input_path=input_path,
        reconstruction_path=reconstruction_path,
        obs_path=obs_path,
        metadata_path=metadata_path,
        n_examples_exported=len(selected_indices),
        n_strata=selection_result.n_strata,
        n_represented_strata=(
            selection_result.n_represented_strata
        ),
        n_omitted_strata=(
            selection_result.n_omitted_strata
            if selection_result.n_strata is not None
            else None
        ),
    )


def _collect_reconstruction_observations(
    *,
    predictions: Sequence[object],
    observation_specs: Sequence[PredictionReconstructionObservationSpec],
    batch_sizes: Sequence[int],
    tensor_keys: set[str],
) -> dict[str, list[Any]]:
    """Collect observations attached to reconstruction examples.

    Canonical outputs retain their optional ``sample_id``, ``label``, and
    free-form metadata fields. Composite outputs use the scalar model-input and
    batch-metadata sources resolved specifically for reconstruction export.
    YOYO outputs may use the same optional annotation names through mapping
    keys or attributes. Tensor keys selected by any enabled export take
    precedence over annotations, including keys selected only by AnnData or
    another reconstruction pair.
    """
    first_prediction = predictions[0]

    if isinstance(first_prediction, CompositePredictionOutput):
        return {
            observation.name: _concat_prediction_observation_values(
                predictions=predictions,
                source=observation.source,
                key=observation.name,
            )
            for observation in observation_specs
        }

    observations: dict[str, list[Any]] = {}

    sample_ids = (
        None
        if "sample_id" in tensor_keys
        else _concat_optional_batch_values(
            predictions=predictions,
            key="sample_id",
            batch_sizes=batch_sizes,
        )
    )
    labels = (
        None
        if "label" in tensor_keys
        else _concat_optional_batch_values(
            predictions=predictions,
            key="label",
            batch_sizes=batch_sizes,
        )
    )
    metadata = (
        None
        if "metadata" in tensor_keys
        else _concat_optional_metadata(
            predictions=predictions,
            batch_sizes=batch_sizes,
        )
    )

    if sample_ids is not None:
        observations["sample_id"] = sample_ids

    if labels is not None:
        observations["label"] = labels

    if metadata is not None:
        conflicting_keys = set(metadata).intersection(
            {*observations, "source_index"}
        )

        if conflicting_keys:
            raise ValueError(
                "Prediction metadata contains keys that conflict with "
                "dedicated reconstruction observation fields: "
                f"{sorted(conflicting_keys)}."
            )

        observations.update(metadata)

    return observations


def _select_stratified_reconstruction_indices(
    *,
    stratify_values: Sequence[Any],
    stratify_by: str,
    n_selected: int,
    seed: int,
) -> ReconstructionSelectionResult:
    strata: dict[Any, list[int]] = {}

    for index, value in enumerate(stratify_values):
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(
                f"Reconstruction stratification field {stratify_by!r} must "
                "contain scalar, hashable values."
            ) from exc

        strata.setdefault(value, []).append(index)

    generator = torch.Generator().manual_seed(seed)

    groups = list(strata.values())
    group_order = torch.randperm(
        len(groups),
        generator=generator,
    ).tolist()

    shuffled_groups: list[list[int]] = []

    for group_index in group_order:
        group = groups[group_index]
        sample_order = torch.randperm(
            len(group),
            generator=generator,
        ).tolist()

        shuffled_groups.append([group[index] for index in sample_order])

    selected: list[int] = []
    position = 0

    while len(selected) < n_selected:
        added_example = False

        for group in shuffled_groups:
            if position >= len(group):
                continue

            selected.append(group[position])
            added_example = True

            if len(selected) == n_selected:
                break

        if not added_example:
            break

        position += 1

    represented_strata = sum(
        any(index in selected for index in group)
        for group in shuffled_groups
    )
    n_strata = len(shuffled_groups)

    return ReconstructionSelectionResult(
        indices=torch.tensor(selected, dtype=torch.long),
        n_strata=n_strata,
        n_represented_strata=represented_strata,
        n_omitted_strata=n_strata - represented_strata,
    )
