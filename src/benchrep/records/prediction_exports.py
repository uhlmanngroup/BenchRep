from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import torch

from benchrep.interfaces.contracts import (
    AutoencoderPredictionOutput,
    VAEPredictionOutput,
)
from benchrep.assembly.resolvers.prediction_config_resolver import PredictionExportSpec
from benchrep.records.anndata_io import (
    package_matrix_as_anndata,
    write_h5ad,
)
from benchrep.interfaces.model_families import ModelFamilySpec
from benchrep.interfaces.compatibility import validate_prediction_output_structure


PredictionOutput = AutoencoderPredictionOutput | VAEPredictionOutput
PredictionOutputLike = object

@dataclass(frozen=True)
class PredictionExportPaths:
    embedding_export: EmbeddingExportPathAndKeys | None = None
    reconstruction_paths: ReconstructionExportPaths | None = None


@dataclass(frozen=True)
class EmbeddingExportPathAndKeys:
    embeddings_h5ad_path: Path | None = None
    resolved_keys: list[str] | None = None
    resolved_primary_key: str | None = None


@dataclass(frozen=True)
class ReconstructionExportPaths:
    input_path: Path | None = None
    reconstruction_path: Path | None = None
    obs_path: Path | None = None
    metadata_path: Path | None = None
    n_examples_exported: int | None = None


def export_prediction_outputs(
    *,
    model_family: ModelFamilySpec,
    predictions: Sequence[PredictionOutputLike],
    export_spec: PredictionExportSpec,
    embedding_dir: Path,
    reconstruction_dir: Path,
) -> PredictionExportPaths:
    """Export prediction outputs returned by ``Trainer.predict``.

    This function consumes the batch-level data objects returned by model
    ``predict_step`` methods and writes the requested prediction artifacts to the
    export directories provided by the caller.

    Embedding outputs are packaged as a BenchRep AnnData artifact. The primary
    embedding is stored in ``adata.X`` and additional exported embeddings are stored
    in ``adata.obsm``. Reconstruction outputs are exported separately as tensors,
    together with selected sample annotations and reconstruction export metadata.

    The exporter does not own the run directory layout. In BenchRep workflows, the
    provided directories should normally come from ``RunContext`` (for example,
    ``run_context.embedding_dir`` and ``run_context.reconstruction_dir``). The
    function returns the concrete file paths it wrote so callers can log them,
    test them, or include them in manifests.
    """
    _validate_prediction_outputs(
        model_family=model_family,
        predictions=predictions,
    )

    exported_embeddings_path_and_keys = None
    exported_reconstruction_paths = None

    if export_spec.embeddings.enabled:
        embedding_dir.mkdir(parents=True, exist_ok=True)

        exported_embeddings_path_and_keys = _export_embeddings(
            predictions=predictions,
            export_spec=export_spec,
            output_dir=embedding_dir,
        )

    if export_spec.reconstructions.enabled:
        reconstruction_dir.mkdir(parents=True, exist_ok=True)

        exported_reconstruction_paths = _export_reconstructions(
            predictions=predictions,
            export_spec=export_spec,
            output_dir=reconstruction_dir,
        )

    return PredictionExportPaths(
        embedding_export=exported_embeddings_path_and_keys,
        reconstruction_paths=exported_reconstruction_paths,
    )


def _validate_prediction_outputs(
        *,
        model_family: ModelFamilySpec,
        predictions: Sequence[PredictionOutputLike],
) -> None:
    if not predictions:
        raise ValueError("Cannot export prediction outputs because no predictions were returned.")

    for batch_idx, batch in enumerate(predictions):
        validate_prediction_output_structure(
            prediction=batch,
            model_family=model_family,
            batch_idx=batch_idx,
            check_value_types=True,
        )


def _export_embeddings(
    *,
    predictions: Sequence[PredictionOutputLike],
    export_spec: PredictionExportSpec,
    output_dir: Path,
) -> EmbeddingExportPathAndKeys:
    """Export embedding-like prediction outputs as a single AnnData artifact.

    The resolved primary embedding is stored in ``adata.X``. Any additional
    resolved embedding keys are stored in ``adata.obsm`` under their output key
    names. Optional ``sample_id``, ``label``, and ``metadata`` fields from
    prediction batches are preserved in ``adata.obs`` when present.

    The export spec controls which embedding keys are exported:
    ``"auto"`` resolves to ``["embedding"]``, ``"all"`` resolves to the currently
    recognized BenchRep representation keys, and explicit key lists are exported
    as requested.

    Returns
    -------
    EmbeddingExportPathAndKeys
        Paths and resolved embedding-key metadata for the written embedding export.
    """
    embedding_keys = _resolve_embedding_keys(
        predictions=predictions,
        export_spec=export_spec,
    )

    primary_key = _resolve_primary_embedding_key(
        embedding_keys=embedding_keys,
        primary_key=export_spec.embeddings.primary_key,
    )

    primary_embedding = _concat_tensor_batches(
        predictions=predictions,
        key=primary_key,
    )

    if primary_embedding.ndim != 2:
        raise ValueError(
            f"Primary embedding key {primary_key!r} must be a 2D tensor with shape "
            f"(n_samples, n_features), got shape {tuple(primary_embedding.shape)}."
        )

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

    adata = package_matrix_as_anndata(
        primary_embedding,
        sample_ids=sample_ids,
        labels=labels,
        metadata=metadata,
    )

    for key in embedding_keys:
        if key == primary_key:
            continue

        embedding = _concat_tensor_batches(
            predictions=predictions,
            key=key,
        )

        if embedding.ndim != 2:
            raise ValueError(
                f"Additional embedding key {key!r} must be a 2D tensor with shape "
                f"(n_samples, n_features), got shape {tuple(embedding.shape)}."
            )

        if embedding.shape[0] != adata.n_obs:
            raise ValueError(
                f"Additional embedding key {key!r} has {embedding.shape[0]} rows, "
                f"but primary embedding has {adata.n_obs} rows."
            )

        adata.obsm[key] = embedding.numpy()

    adata.uns["benchrep_prediction_export"] = {
        "mode": export_spec.mode,
        "embedding_keys": embedding_keys,
        "primary_key": primary_key,
    }

    embeddings_h5ad_path = output_dir / "embeddings.h5ad"

    write_h5ad(
        adata,
        embeddings_h5ad_path,
        overwrite=True,
    )

    return EmbeddingExportPathAndKeys(
        embeddings_h5ad_path=embeddings_h5ad_path,
        resolved_keys=embedding_keys,
        resolved_primary_key=primary_key,
    )


def _resolve_embedding_keys(
    *,
    predictions: Sequence[PredictionOutputLike],
    export_spec: PredictionExportSpec,
) -> list[str]:
    requested_keys = export_spec.embeddings.keys
    first_batch = predictions[0]

    if requested_keys == "auto":
        requested_keys = ["embedding"]

    elif requested_keys == "all":
        requested_keys = _find_embedding_like_keys(first_batch)

    missing_keys = [
        key for key in requested_keys
        if not _has_prediction_value(first_batch, key)
    ]

    if missing_keys:
        raise KeyError(
            "Embedding export requested keys that are not present in prediction "
            f"outputs: {missing_keys}. Available keys: {_available_prediction_keys(first_batch)}."
        )

    return list(requested_keys)


def _resolve_primary_embedding_key(
    *,
    embedding_keys: list[str],
    primary_key: str,
) -> str:
    if not embedding_keys:
        raise ValueError("No embedding keys were resolved for export.")

    if primary_key == "auto":
        if "embedding" in embedding_keys:
            return "embedding"
        return embedding_keys[0]

    if primary_key not in embedding_keys:
        raise ValueError(
            f"Primary embedding key {primary_key!r} is not among the exported "
            f"embedding keys: {embedding_keys}."
        )

    return primary_key


def _find_embedding_like_keys(first_batch: PredictionOutputLike) -> list[str]:
    """Find recognized embedding-like outputs in one prediction batch.

    This helper is used for ``exports.mode='all'`` / ``embeddings.keys='all'``.
    In this context, "all" means all currently recognized BenchRep representation
    outputs, not every 2D tensor returned by ``predict_step``.

    This intentionally uses a small whitelist rather than exporting all 2D
    tensors. Future model families may return other 2D outputs such as
    projections, logits, anchors, positives, or task-specific heads. Those should
    only become part of automatic embedding export once BenchRep gives them an
    explicit export contract.
    """
    recognized_embedding_keys = (
        "embedding",
        "z_mu",
        "z_logvar",
        "z_sample",
    )

    embedding_keys = []

    for key in recognized_embedding_keys:
        if not _has_prediction_value(first_batch, key):
            continue

        value = getattr(first_batch, key)

        if isinstance(value, torch.Tensor) and value.ndim == 2:
            embedding_keys.append(key)

    if not embedding_keys:
        raise ValueError(
            "Could not find any recognized embedding outputs for export. Expected "
            "at least one 2D tensor under one of the recognized keys "
            f"{recognized_embedding_keys}. Available keys: "
            f"{_available_prediction_keys(first_batch)}."
        )

    return embedding_keys


def _concat_optional_batch_values(
    *,
    predictions: Sequence[PredictionOutputLike],
    key: str,
) -> list[Any] | None:
    if not _has_prediction_value(predictions[0], key):
        return None

    values: list[Any] = []

    for batch_idx, batch in enumerate(predictions):
        if not _has_prediction_value(batch, key):
            raise KeyError(
                f"Prediction key {key!r} is present in the first batch but missing "
                f"from batch {batch_idx}."
            )

        value = getattr(batch, key)

        if isinstance(value, torch.Tensor):
            values.extend(value.detach().cpu().tolist())
        elif isinstance(value, list):
            values.extend(value)
        else:
            raise TypeError(
                f"Prediction key {key!r} must contain torch.Tensor or list values. "
                f"Batch {batch_idx} has value type {type(value).__name__}."
            )

    return values


def _concat_optional_metadata(
    *,
    predictions: Sequence[PredictionOutputLike],
) -> dict[str, list[Any]] | None:
    if not _has_prediction_value(predictions[0], "metadata"):
        return None

    merged_metadata: dict[str, list[Any]] = {}

    for batch_idx, batch in enumerate(predictions):
        if not _has_prediction_value(batch, "metadata"):
            raise KeyError(
                "'metadata' is present in the first prediction batch but missing "
                f"from batch {batch_idx}."
            )

        metadata = getattr(batch, "metadata")

        if not isinstance(metadata, dict):
            raise TypeError(
                "Prediction metadata must be a dictionary mapping metadata keys "
                f"to per-sample values. Batch {batch_idx} has metadata type "
                f"{type(metadata).__name__}."
            )

        for metadata_key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                value_list = value.detach().cpu().tolist()
            elif isinstance(value, list):
                value_list = value
            else:
                raise TypeError(
                    f"Metadata field {metadata_key!r} must contain torch.Tensor "
                    f"or list values. Batch {batch_idx} has value type "
                    f"{type(value).__name__}."
                )

            merged_metadata.setdefault(metadata_key, []).extend(value_list)

    return merged_metadata


def _select_reconstruction_indices(
    *,
    n_samples: int,
    n_examples: int | str,
    selection: str,
    seed: int | None,
) -> torch.Tensor:
    """Select row indices for reconstruction export.

    ``selection="first"`` returns the first ``n_examples`` rows. ``selection="random"``
    samples rows without replacement using a local torch generator seeded by
    ``seed``. The caller is responsible for passing the resolved reconstruction
    seed; this function only enforces that random selection cannot proceed with
    ``seed=None``.

    If ``n_examples="all"``, all rows are selected. If an integer larger than
    ``n_samples`` is requested, selection is capped at ``n_samples``.
    """
    if n_examples == "all":
        n_selected = n_samples
    else:
        n_selected = min(n_examples, n_samples)

    if selection == "first":
        return torch.arange(n_selected)

    if selection == "random":
        if seed is None:
            raise ValueError("Random reconstruction selection requires a seed.")

        generator = torch.Generator().manual_seed(seed)
        return torch.randperm(n_samples, generator=generator)[:n_selected]

    raise ValueError(
        f"Unsupported reconstruction selection {selection!r}. "
        "Available options: 'first', 'random'."
    )


def _export_reconstructions(
    *,
    predictions: Sequence[PredictionOutputLike],
    export_spec: PredictionExportSpec,
    output_dir: Path,
) -> ReconstructionExportPaths:
    """Export selected reconstruction examples as tensor artifacts.

    Depending on the reconstruction export spec, this function exports selected
    rows from ``input`` and/or ``reconstruction`` prediction outputs. Selection is
    controlled by ``n_examples`` and ``selection``; random selection uses the
    already-resolved reconstruction seed from the export spec.

    The function also writes an ``obs.pt`` file containing the selected source
    indices and any available sample-level annotations, plus a metadata file
    describing the reconstruction export settings and exported tensor keys.

    Returns
    -------
    ReconstructionExportPaths
        Paths to the written reconstruction tensor, annotation, and metadata
        artifacts. Paths for disabled tensor outputs are ``None``.
    """
    recon_spec = export_spec.reconstructions

    if not recon_spec.include_input and not recon_spec.include_prediction:
        raise ValueError(
            "Reconstruction export was enabled, but both include_input and "
            "include_prediction are False. Nothing would be exported."
        )

    tensors_to_export: dict[str, torch.Tensor] = {}

    if recon_spec.include_input:
        if not _has_prediction_value(predictions[0], "input"):
            raise KeyError(
                "Reconstruction export requested include_input=True, but prediction "
                f"outputs do not contain key 'input'. Available keys: "
                f"{_available_prediction_keys(predictions[0])}."
            )

        tensors_to_export["input"] = _concat_tensor_batches(
            predictions=predictions,
            key="input",
        )

    if recon_spec.include_prediction:
        if not _has_prediction_value(predictions[0], "reconstruction"):
            raise KeyError(
                "Reconstruction export requested include_prediction=True, but prediction "
                "outputs do not contain key 'reconstruction'. Available keys: "
                f"{_available_prediction_keys(predictions[0])}."
            )

        tensors_to_export["reconstruction"] = _concat_tensor_batches(
            predictions=predictions,
            key="reconstruction",
        )

    n_samples = next(iter(tensors_to_export.values())).shape[0]

    for key, tensor in tensors_to_export.items():
        if tensor.shape[0] != n_samples:
            raise ValueError(
                f"Reconstruction tensor {key!r} has {tensor.shape[0]} samples, "
                f"expected {n_samples}."
            )

    selected_indices = _select_reconstruction_indices(
        n_samples=n_samples,
        n_examples=recon_spec.n_examples,
        selection=recon_spec.selection,
        seed=recon_spec.seed,
    )
    selected_indices_list = selected_indices.tolist()

    input_path = None
    reconstruction_path = None

    for key, tensor in tensors_to_export.items():
        export_path = output_dir / f"{key}.pt"
        torch.save(tensor[selected_indices], export_path)

        if key == "input":
            input_path = export_path
        elif key == "reconstruction":
            reconstruction_path = export_path

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

    reconstruction_obs = {
        "source_index": selected_indices_list,
    }

    if sample_ids is not None:
        reconstruction_obs["sample_id"] = [
            sample_ids[i] for i in selected_indices_list
        ]

    if labels is not None:
        reconstruction_obs["label"] = [
            labels[i] for i in selected_indices_list
        ]

    if metadata is not None:
        for key, values in metadata.items():
            reconstruction_obs[key] = [
                values[i] for i in selected_indices_list
            ]

    obs_path = output_dir / "obs.pt"
    torch.save(reconstruction_obs, obs_path)

    metadata_path = output_dir / "reconstruction_export_metadata.pt"
    torch.save(
        {
            "n_samples_total": n_samples,
            "n_examples_exported": len(selected_indices),
            "selection": recon_spec.selection,
            "seed": recon_spec.seed,
            "include_input": recon_spec.include_input,
            "include_prediction": recon_spec.include_prediction,
            "exported_keys": list(tensors_to_export.keys()),
        },
        metadata_path,
    )

    return ReconstructionExportPaths(
        input_path=input_path,
        reconstruction_path=reconstruction_path,
        obs_path=obs_path,
        metadata_path=metadata_path,
        n_examples_exported=len(selected_indices),
    )


def _concat_tensor_batches(
    *,
    predictions: Sequence[PredictionOutputLike],
    key: str,
) -> torch.Tensor:
    tensors = []

    for batch_idx, batch in enumerate(predictions):
        if not _has_prediction_value(batch, key):
            raise KeyError(
                f"Prediction batch {batch_idx} does not contain key {key!r}. "
                f"Available keys: {_available_prediction_keys(batch)}."
            )

        value = getattr(batch, key)

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Prediction key {key!r} must contain torch.Tensor values for tensor export. "
                f"Batch {batch_idx} has value type {type(value).__name__}."
            )

        tensors.append(value.detach().cpu())

    return torch.cat(tensors, dim=0)


def _prediction_field_names(prediction: PredictionOutputLike) -> tuple[str, ...]:
    if not is_dataclass(prediction) or isinstance(prediction, type):
        raise TypeError(
            "Prediction output must be a dataclass instance, "
            f"got `{type(prediction).__name__}`."
        )

    return tuple(
        field.name
        for field in fields(cast(Any, prediction))
    )


def _has_prediction_value(prediction: PredictionOutputLike, key: str) -> bool:
    if key not in _prediction_field_names(prediction):
        return False

    return getattr(prediction, key) is not None


def _available_prediction_keys(prediction: PredictionOutputLike) -> tuple[str, ...]:
    return tuple(
        key
        for key in _prediction_field_names(prediction)
        if _has_prediction_value(prediction, key)
    )