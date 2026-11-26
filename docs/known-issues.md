# Known issues

## XGBoost on macOS

XGBoost predictability evaluation has stalled in BenchRep usage tests on macOS.
The diagnostics point to a conflict between OpenMP runtimes used by native
dependencies, including PyTorch and XGBoost. We have not established a reliable
combination of versions that avoids it. This affects XGBoost evaluation.

For now, install the core package and the optional extras you need, excluding
`xgboost`. The `full` and `dev` extras include XGBoost. For development from a
checkout, install the other development dependencies with:

```bash
python -m pip install -e ".[scverse,logging,model_graph,optuna,gui-test,artifact_generation,docs]"
```
In evaluation configs, use an explicit list under `metrics.predictability.targets.<target>.selected` instead of `selected: all`. Leave out `xgboost` and its alias `xgb`. If XGBoost was the only selected probe, choose another probe or disable predictability evaluation. Leaving XGBoost selected when it isn’t installed causes an error.


macOS CI omits XGBoost installation and uses `--usage-skip-xgboost` to remove
only that probe from temporary usage-test configs. The reports record the
exclusion; the shipped examples remain unchanged. Linux and Windows CI retain
XGBoost testing. A passing macOS CI run therefore does **not** establish XGBoost
compatibility.
