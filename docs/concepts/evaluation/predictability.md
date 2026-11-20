# Predictability probes

A predictability probe is a supervised estimator trained on a saved representation to predict an observation annotation. It measures whether that information is accessible from the embedding, without retraining the representation model.

For example, a linear probe asks whether a class can be separated by a linear decision boundary. A nonlinear probe may recover information a linear probe cannot. A dummy baseline helps distinguish useful prediction from performance achievable without reading the embedding.

## Define the target and probes

Features come from `adata.X`; each key in `metrics.predictability.targets` names a column of `adata.obs`. Targets are evaluated independently and can use different tasks and settings:

```yaml
metrics:
  predictability:
    enabled: true
    targets:
      label:
        task: classification
        selected: [dummy, linear]
      measured_value:
        task: regression
        selected: [dummy, linear]
```

This fragment requires both observation columns in the AnnData input. Classification predicts categories; regression predicts numeric values. Selecting one task does not infer the other from the column's dtype.

| Probe | Classification | Regression |
| --- | --- | --- |
| `dummy` | Dummy classifier; most-frequent class by default | Dummy regressor; target mean by default |
| `linear` | Logistic regression | Ridge regression |
| `knn` | Nearest-neighbor classifier | Nearest-neighbor regressor |
| `random_forest` | Random forest classifier | Random forest regressor |
| `svm_rbf` | RBF support-vector classifier | RBF support-vector regressor |
| `xgboost` | XGBoost classifier | XGBoost regressor |

Omitting `selected` uses all listed probes except optional XGBoost. `selected: all` uses every registered probe. Predictability itself remains disabled until explicitly enabled, avoiding an unexpectedly expensive set of fits.

Probe-specific settings live under the target's `params`. An explicit `linear` mapping must identify `model: logistic_regression` for classification or `model: ridge` for regression. Full parameters and dependency requirements appear in the [evaluation schema reference](../../reference/schemas/evaluation.md) and [component catalogue](../../reference/components.md).

## Estimate performance on held-out observations

Each probe is evaluated by cross-validation: fit it on some observations and score it on observations excluded from that fit. The defaults use five outer folds, stratified folds for classification and ordinary folds for regression, with shuffling and seed 137. Default scores are balanced accuracy and R², respectively.

These are partitions of the **evaluation artifact**, separate from the representation model's training/validation split. Every probe for a target uses its configured fold strategy. Linear, k-NN, and RBF-SVM probes standardize features by default, using a scikit-learn pipeline fitted within each training fold rather than scaling the full artifact before splitting.

For related observations that should stay together, choose a grouped strategy and provide an observation column identifying the groups:

```yaml
metrics:
  predictability:
    enabled: true
    targets:
      label:
        selected: [dummy, linear]
        cv:
          method: stratified_group_kfold
          group_key: subject_id
          n_splits: 5
```

The grouped splitter keeps each group entirely within one fold. Ordinary stratification aims to preserve class proportions but does not prevent related observations from entering both training and test partitions. The available observations, classes, and groups must support the requested number of folds.

!!! warning "What cross-validation protects"
    Probe cross-validation separates the probe's fitting data from its scoring data. It cannot undo information leakage introduced earlier, such as fitting preprocessing on held-out observations or using their labels to choose the representation. Choose the evaluation population and upstream training procedure according to the generalization question you want to answer.

## Tune inside the training folds

Hyperparameter tuning is disabled by default. Enable it and provide list-valued estimator parameters to search candidate values:

```yaml
metrics:
  predictability:
    enabled: true
    targets:
      label:
        selected: [linear]
        tuning:
          enabled: true
          inner_cv:
            n_splits: 3
        params:
          linear:
            model: logistic_regression
            C: [0.1, 1.0, 10.0]
```

For each outer fold, an inner grid search selects parameters using only that outer fold's training observations. The chosen estimator then scores the untouched outer test observations. The inner splitter follows the outer strategy, including group boundaries when applicable, and uses the same scoring rule. This is **nested cross-validation**: parameter selection and final scoring happen on different partitions.

Scalar parameters stay fixed. List-valued parameters require tuning, and enabling tuning requires a candidate list for at least one selected probe. The cost grows with the number of candidates and inner and outer folds, so start with a small probe set when checking a configuration.

## Results and implementation

Results under `metrics.predictability.<target>` record each probe's fold scores, mean, sample standard deviation across folds, and selected parameters per fold when tuned. Negated error scorers follow scikit-learn's higher-is-better convention. The workflow reports evaluation results; it does not export a final fitted probe for deployment.

The [probe builders](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/embeddings/predictability_probes.py) convert task-specific parameters into an estimator and optional parameter grid. They also put standardization inside the estimator pipeline and map tuning keys to the correct nested estimator.

The [predictability runner](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/embeddings/predictability.py) validates the feature, target, and group arrays, encodes classification labels, constructs splitters, and clones estimators for independent fits. With tuning, it creates a `GridSearchCV` inside each outer training fold and passes that fold's group labels to the inner splitter.

Expected fitting or split failures are reported as recoverable failures. Successful probes can remain available if another fails; if all selected probes fail recoverably, that target's step fails. Other evaluation analyses can still continue under the [pipeline's failure rules](inputs.md#implementation).
