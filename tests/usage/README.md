# Usage-example tests

Run all 15 usage scripts with smaller workloads. Tests check workflow results and
saved outputs, not model performance. Temporary configs come from the real
examples; the original scripts and YAMLs are not edited.

## Run

From the repository root, in your BenchRep environment:

```bash
pytest tests/*/ --run-usage-examples -ra
```

For usage tests only, replace `tests/*/` with `tests/usage`. Without
`--run-usage-examples`, the scripts are skipped.

**Defaults:** automatic device selection, sequential execution, and a 30-minute
limit per example. Outputs are deleted only when the entire test session passes.
Failures and interruptions preserve them. Shared dataset and weight caches remain.

## Common options

| Option | Effect |
| --- | --- |
| `--usage-device cpu` | Force CPU execution |
| `--usage-skip-xgboost` | Omit only XGBoost probes from temporary configs (macOS CI) |
| `--no-usage-clean-on-success` | Keep outputs after success |
| `--usage-include 01 06 08A` | Run only these examples |
| `--usage-exclude 07 08B 11` | Skip these examples |
| `--usage-output PATH` | Use a new output directory; it must not already exist |
| `--durations=15` | Show the 15 slowest test phases |

`-ra` adds pytest's short summary of all outcomes except passes.
For all options, run `pytest tests/usage --help`.

## Setup

If dependencies are missing:

```bash
pip install -e '.[test,scverse,optuna,model_graph,artifact_generation,xgboost,gui-test]'
playwright install chromium
```

On macOS, omit `xgboost` from the install command and pass `--usage-skip-xgboost`;
see [Known issues](../../docs/known-issues.md#xgboost-on-macos).

Also install Graphviz (`dot`). Linux CI may need Playwright's system dependencies.
Example 09 needs its packaged data assets. The first run may download full datasets
and pretrained weights.

Examples 12–14 cover runtime overrides, YOYO training, and standalone pretrained
ViT prediction. The ViT test keeps the pretrained model and preprocessing, using
128 images; CI prepares its weights in the shared download cache.

## Results

Pytest prints output locations. Inspect `run.log` for diagnostics and `report.json`
for config changes, statuses, and warnings. Workload settings live in `profiles.py`.

Failed workflow outcomes fail the test. The sole accepted exception is example
09's all-noise HDBSCAN result and its two corresponding metric failures, checked
explicitly and recorded in the report.
