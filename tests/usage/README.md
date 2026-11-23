# Usage-example tests

These opt-in tests run all 12 shipped usage scripts in separate processes with
small workloads. They use the installed BenchRep package and its actual example
YAMLs. No production code, example script, or source YAML is changed.

## Run

From the repository root in your BenchRep environment:

```bash
python -m pytest tests/usage --run-usage-examples --durations=0 -ra
```

Required extras: `test`, `scverse`, `optuna`, `model_graph`, `artifact_generation`,
`xgboost`, and `gui-test`. If they are not installed:

```bash
python -m pip install -e '.[test,scverse,optuna,model_graph,artifact_generation,xgboost,gui-test]'
python -m playwright install chromium
```

Graph exports also require the Graphviz system executable (`dot`). Example 09
requires its usual packaged AnnData and reconstruction bundle. Missing dependencies
or inputs fail the selected example; they do not silently remove it from coverage.
On Linux CI, Playwright may need its system dependencies installed as well.

The default is CPU, one device, and two numerical-library threads. To use the
available GPU automatically:

```bash
python -m pytest tests/usage --run-usage-examples --usage-device auto --durations=0 -ra
```

Select or exclude examples by their exact IDs:

```bash
python -m pytest tests/usage --run-usage-examples --usage-include 01 06 08A --durations=0
python -m pytest tests/usage --run-usage-examples --usage-exclude 07 08B 11 --durations=0
```

Without `--run-usage-examples`, only the lightweight harness checks run; usage
scripts are skipped. Execution is sequential. Avoid parallel pytest workers for
this directory, especially on a shared GPU or during the first dataset download.

## What becomes faster

| Examples | Training | Prediction / evaluation |
| --- | --- | --- |
| 01–05 | 2 epochs; 4 training and 2 validation batches per epoch; batch size 16 | Up to 512 predictions |
| 06 | 6 epochs; 2 training and 2 validation batches per epoch; batch size 16 | Keeps the original KL ramp and epoch-5 LR change; checks both and gradient logs |
| 07 | Both initialization variants; 2 epochs each; 4 training and 2 validation batches; batch size 16 | Up to 512 predictions per variant; all four probes and comparison CSV retained |
| 08A–08B | 2 epochs; 4 training and 2 validation batches; batch size 16 | Up to 512 predictions; all targets, analyses, tuning, and exports retained |
| 09 | No training | 1,024 observations sampled across digit/acquisition-batch combinations; reconstruction bundle unchanged |
| 10 | 3 trials; 2 epochs and 4 training / 2 validation batches per trial; batch size 16 | Original held-out validation dataset; at most 4 prediction batches; first trial deliberately fails |
| 11 | Original pretrained ConvNeXt and decoder; 2 epochs; 4 training / 2 validation batches; batch size 8 | Up to 512 predictions |

Shared changes:

- Real Torchvision datasets are capped at 1,024 samples, balanced across labels
  where labels exist. Both data and labels are sliced together after the normal
  dataset constructor loads them. This also caps the input *before* example 08's
  expensive enrichment, without replacing its dataset implementation.
- Training/validation split fractions, seeds, model architectures, losses,
  transforms, and pretrained weights remain unchanged.
- Data loading uses zero workers and no persistent workers or prefetching.
  Multiprocessing behavior is not exercised by these fast example runs.
- Cosine `T_max` follows the reduced epoch count. Example 06 retains its original
  loss schedule and LR milestone instead of compressing them.
- Evaluation uses two outer and, when enabled, two inner CV folds; random forests
  and XGBoost use 10 estimators and one job. Other estimator parameters and all
  selected analyses are retained. Figures use 72 DPI, with all original formats,
  color choices, channels, and error-map types retained.
- Optuna keeps its real study, sampler, pruning callback, SQLite database, error
  capture, and CSV export. Sampler/pruner startup is one trial, pruning warmup is
  zero, and the human-facing dashboard server is disabled. A pruning outcome is
  allowed but not forced: tiny runs cannot guarantee that the real pruner will
  choose to prune.
- 08B uses real headless Chromium to check and save all three wizards. It does not
  replace the wizard with a fabricated configuration result.

These runs test execution and outputs, not learning quality. Small batches change
contrastive negatives, and short training may produce degenerate clusters or
numerical warnings. The tests report these; do not suppress them automatically.
Optional W&B logging is not exercised: examples 06/07 run their default CSV route.

## How the configs are changed

Before importing an example, the child process intercepts `example_config_path()`
and writes an edited YAML under that run's temporary `fast-configs/` directory.
`load_example_config()` continues calling the same helper and normal validator.
Example 03's Python-built configs receive equivalent changes immediately before
its workflow calls, with field omission preserved and normal schema validation.
Example 02's own section replacements still run normally.

The central budgets and edits live in `profiles.py`. Each `report.json` records
actual before/after config changes, dataset sizes, workflow statuses, and warnings.
Temporary YAMLs are generated from the current originals, not maintained as copies.

## Outputs and checks

Each selected example gets a separate working directory containing `run.log`,
`report.json`, temporary configs, and the script's normal outputs. Pytest prints
its path and runtime, including setup and downloads. Use `--usage-output PATH`
for a persistent **new** directory; an existing path is rejected to prevent stale
artifacts from passing checks. Otherwise pytest's temporary directory is used.
Add `--usage-clean-on-success` to delete this session's entire usage-output
directory only when the whole pytest session succeeds, including teardown. Any
failure, interruption, or collection error preserves all outputs for inspection.
Dataset and pretrained-weight caches outside that directory are retained. Keep
`--usage-data-dir` outside the output directory if using a custom cache.

`--usage-timeout SECONDS` changes the per-example limit (default 1,800 seconds).

The tests check workflow counts and statuses, optimizer steps, best/last
checkpoints, prediction artifacts, configured graph exports, evaluation metrics,
finite projections/probe scores, and reported export paths. Example-specific
checks cover schedules/gradient logs, comparison CSVs, wizard completion, and
Optuna's deliberate failure plus at least one successfully scored trial.
Failed, skipped, interrupted, or partially completed workflow outcomes fail the
test even if the script itself exited successfully, with one narrow exception:
example 09 may report all-noise HDBSCAN assignments and the two corresponding
clustering-metric failures. The harness verifies the actual assignments and exact
three outcomes; additional failures or warnings reject this exception. Original
statuses remain in the manifest and report; accepted issues are also listed under
`expected_issues`. These accepted issues do not fail `--usage-strict-warnings`.

Recorded workflow warnings are printed and retained in `report.json`; add
`--usage-strict-warnings` to fail on them too. Ordinary Python/library warnings
remain in `run.log` and are not silenced by this harness.

Dataset downloads reuse `repository/data` by default. Set `--usage-data-dir PATH`
to use a different cache whose child directories match the examples' roots
(`mnist`, `fashion_mnist`, `cifar10`, `stl10`). Pretrained weights use Torch's normal
cache. The first run still downloads full datasets and loads them before slicing;
this harness reduces computation, not download size or initial loading memory.
