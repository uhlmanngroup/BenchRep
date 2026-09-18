from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias, cast
import warnings

import torch

from benchrep.interfaces.contracts import (
    AutoencoderPredictionOutput,
    VAEPredictionOutput,
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


PredictionOutputLike: TypeAlias = (
    AutoencoderPredictionOutput
    | VAEPredictionOutput
    | CompositePredictionOutput
)


@dataclass(frozen=True)
class PredictionAnnDataExportResult:
    path: Path | None
    outcome: PredictionOutcome


@dataclass(frozen=True)
class PredictionReconstructionBundlePaths:
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
    pair: PredictionReconstructionPairSpec
    paths: PredictionReconstructionBundlePaths
    outcome: PredictionOutcome


@dataclass(frozen=True)
class PredictionReconstructionsExportResult:
    pairs: tuple[PredictionReconstructionPairExportResult, ...]
    outcome: PredictionOutcome


@dataclass(frozen=True)
class PredictionExportResult:
    anndata: PredictionAnnDataExportResult
    reconstructions: PredictionReconstructionsExportResult

    @property
    def outcomes(self) -> tuple[PredictionOutcome, PredictionOutcome]:
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
    predictions: Sequence[PredictionOutputLike],
    export_spec: PredictionExportSpec,
    anndata_dir: Path,
    reconstruction_dir: Path,
) -> PredictionExportResult:
    """Export all prediction artifacts requested by a resolved export spec.

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
        Validated batch-level outputs returned by ``Trainer.predict()``.
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
        for pair_index, pair in enumerate(
            export_spec.reconstructions.pairs,
            start=1,
        ):
            bundle_dir = (
                reconstruction_dir
                / f"{pair_index:02d}_{pair.id}"
            )
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
    predictions: Sequence[PredictionOutputLike],
    anndata_spec: PredictionAnnDataExportSpec,
    output_dir: Path,
) -> Path:
    """Package resolved non-image prediction outputs as one AnnData artifact.

    The resolved primary vector is stored in ``adata.X``. Additional selected
    vector outputs are stored in ``adata.obsm`` under their declared names,
    while selected scalar outputs are stored in ``adata.obs``.

    Canonical prediction annotations are collected from ``sample_id``, ``label``,
    and ``metadata``. Composite annotations follow the resolved observation
    specifications: declared scalar model inputs and group metadata become
    ``adata.obs`` columns, while an optional metadata declaration with role
    ``index`` supplies ``adata.obs_names``.

    Parameters
    ----------
    predictions:
        Validated prediction batches from one canonical or Composite model.
    anndata_spec:
        Resolved AnnData export plan containing selected output keys, their
        scalar/vector structures, the primary key, and observation sources.
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

    primary_matrix = _concat_prediction_model_output_batches(
        predictions=predictions,
        key=primary_key,
    )

    (
        sample_ids,
        labels,
        metadata,
        index_name,
    ) = _collect_anndata_annotations(
        predictions=predictions,
        anndata_spec=anndata_spec,
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
        if (
                key == primary_key
                or anndata_spec.output_structures_by_key[key] == "scalar"
        ):
            continue

        matrix = _concat_prediction_model_output_batches(
            predictions=predictions,
            key=key,
        )

        adata.obsm[key] = matrix.numpy()

    benchrep_uns = adata.uns.setdefault("benchrep", {})
    benchrep_uns["prediction_export"] = {
        "mode": anndata_spec.mode,
        "keys": list(anndata_spec.keys),
        "primary_key": primary_key,
        "output_structures_by_key": dict(
            anndata_spec.output_structures_by_key
        ),
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
    predictions: Sequence[PredictionOutputLike],
    anndata_spec: PredictionAnnDataExportSpec,
) -> tuple[
    list[Any] | None,
    list[Any] | None,
    dict[str, list[Any]] | None,
    str | None,
]:
    """Collect canonical or Composite annotations for AnnData observations.

    Canonical outputs retain their established ``sample_id``, ``label``, and
    free-form ``metadata`` behavior. Composite outputs use only observation
    sources explicitly resolved from their declarations.
    """

    first_prediction = predictions[0]

    if not isinstance(first_prediction, CompositePredictionOutput):
        return (
            _concat_optional_batch_values(
                predictions=predictions,
                key="sample_id",
            ),
            _concat_optional_batch_values(
                predictions=predictions,
                key="label",
            ),
            _concat_optional_metadata(
                predictions=predictions,
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


def _concat_prediction_model_input_batches(
    *,
    predictions: Sequence[PredictionOutputLike],
    key: str,
) -> torch.Tensor:
    """Concatenate one resolved model input across prediction batches."""
    tensors = [
        (
            prediction.model_inputs[key]
            if isinstance(prediction, CompositePredictionOutput)
            else getattr(prediction, key)
        ).detach().cpu()
        for prediction in predictions
    ]

    return torch.cat(tensors, dim=0)


def _concat_prediction_model_output_batches(
    *,
    predictions: Sequence[PredictionOutputLike],
    key: str,
) -> torch.Tensor:
    """Concatenate one resolved model output across prediction batches."""
    tensors = [
        (
            prediction.model_outputs[key]
            if isinstance(prediction, CompositePredictionOutput)
            else getattr(prediction, key)
        ).detach().cpu()
        for prediction in predictions
    ]

    return torch.cat(tensors, dim=0)


def _concat_prediction_observation_values(
    *,
    predictions: Sequence[PredictionOutputLike],
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
    predictions: Sequence[PredictionOutputLike],
    key: str,
) -> list[Any] | None:
    if getattr(predictions[0], key) is None:
        return None

    values: list[Any] = []

    for prediction in predictions:
        value = getattr(prediction, key)

        if isinstance(value, torch.Tensor):
            values.extend(value.detach().cpu().tolist())
        else:
            values.extend(value)

    return values


def _concat_optional_metadata(
    *,
    predictions: Sequence[PredictionOutputLike],
) -> dict[str, list[Any]] | None:
    if getattr(predictions[0], "metadata") is None:
        return None

    merged_metadata: dict[str, list[Any]] = {}

    for prediction in predictions:
        metadata = getattr(prediction, "metadata")

        for key, value in metadata.items():
            values = (
                value.detach().cpu().tolist()
                if isinstance(value, torch.Tensor)
                else value
            )
            merged_metadata.setdefault(key, []).extend(values)

    return merged_metadata


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
    predictions: Sequence[PredictionOutputLike],
    reconstruction_spec: PredictionReconstructionsExportSpec,
    pair: PredictionReconstructionPairSpec,
    output_dir: Path,
) -> PredictionReconstructionBundlePaths:
    """Export one resolved input/reconstruction pair as an artifact bundle.

    Input and reconstruction tensors are collected from the pair's resolved
    prediction namespaces, after which one shared selection is applied to every
    exported tensor and observation. The bundle contains optional input and
    reconstruction tensors, selected observations, and metadata describing the
    pair and selection result.

    Parameters
    ----------
    predictions:
        Validated batch-level prediction outputs.
    reconstruction_spec:
        Resolved settings shared by every reconstruction pair.
    pair:
        Resolved pair identifying the model input and reconstruction output.
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

    input_tensor = (
        _concat_prediction_model_input_batches(
            predictions=predictions,
            key=pair.input,
        )
        if reconstruction_spec.include_input
        else None
    )

    reconstruction_tensor = (
        _concat_prediction_model_output_batches(
            predictions=predictions,
            key=pair.reconstruction,
        )
        if reconstruction_spec.include_reconstruction
        else None
    )

    reference_tensor = (
        input_tensor
        if input_tensor is not None
        else reconstruction_tensor
    )
    assert reference_tensor is not None

    n_samples = reference_tensor.shape[0]

    observations = _collect_reconstruction_observations(
        predictions=predictions,
        observation_specs=reconstruction_spec.observations,
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
    predictions: Sequence[PredictionOutputLike],
    observation_specs: Sequence[
        PredictionReconstructionObservationSpec
    ],
) -> dict[str, list[Any]]:
    """Collect observations attached to reconstruction examples.

    Canonical outputs retain their optional ``sample_id``, ``label``, and
    free-form metadata fields. Composite outputs use the scalar model-input and
    batch-metadata sources resolved specifically for reconstruction export.
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

    sample_ids = _concat_optional_batch_values(
        predictions=predictions,
        key="sample_id",
    )
    labels = _concat_optional_batch_values(
        predictions=predictions,
        key="label",
    )
    metadata = _concat_optional_metadata(
        predictions=predictions,
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
