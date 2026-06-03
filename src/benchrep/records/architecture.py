from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import lightning as L


def export_torchview_graph(
    *,
    model: L.LightningModule,
    input_size: Sequence[int],
    output_path: Path,
) -> Path | None:
    try:
        import torchview
    except ImportError:
        warnings.warn(
            "Skipping torchview graph export because `torchview` is not installed. "
            "Install the optional architecture dependencies to enable this.",
            UserWarning,
            stacklevel=2,
        )
        return None

    try:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_graph = torchview.draw_graph(
            model,
            input_size=tuple(input_size),
        )

        graph_base_path = output_path.with_suffix("")
        rendered_path = model_graph.visual_graph.render(
            str(graph_base_path),
            format=output_path.suffix.lstrip(".") or "png",
            cleanup=True,
        )

        return Path(rendered_path)

    except Exception as exc:
        warnings.warn(
            f"Could not export torchview graph: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None


def infer_dummy_input_size(
    datamodule: L.LightningDataModule,
) -> tuple[int, ...]:
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    try:
        x = batch["x"]
    except KeyError as exc:
        raise KeyError(
            "Could not infer dummy input size because the training batch "
            "does not contain key 'x'. BenchRep datamodules must return batches "
            "with batch['x'] as the model input tensor."
        ) from exc

    return (1, *x.shape[1:])