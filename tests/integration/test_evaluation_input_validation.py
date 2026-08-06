from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from benchrep import evaluate
from benchrep.assembly.schemas import (
    EvaluationConfig,
    EvaluationRunConfig,
    EvaluationSourceConfig,
    EvaluationReductionsConfig,
    UMAPConfig,
)


def test_evaluate_rejects_nonfinite_embeddings(
    tmp_path: Path,
) -> None:
    embeddings = np.arange(24, dtype=np.float32).reshape(6, 4)
    embeddings[0, 0] = np.nan
    embeddings[1, 1] = np.inf
    embeddings[2, 2] = -np.inf

    embeddings_path = tmp_path / "nonfinite_embeddings.h5ad"
    ad.AnnData(X=embeddings).write_h5ad(embeddings_path)

    config = EvaluationConfig(
        source=EvaluationSourceConfig(
            embeddings_path=embeddings_path,
        ),
        run=EvaluationRunConfig(
            output_root=tmp_path / "outputs",
            run_name="nonfinite_embeddings",
        ),
        reductions=EvaluationReductionsConfig(
            umap=UMAPConfig(enabled=False),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Found 1 NaN values and 2 infinite values across "
            r"3 observations"
        ),
    ):
        evaluate(full_config_object=config)