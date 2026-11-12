# Tune with Optuna

Tune a composite VAE on MNIST to <strong class="highlight">improve clustering</strong> of its validation embeddings. Each Optuna trial runs a full workflow-chain with a different combination of training hyperparameters. <strong class="highlight">Training-time pruning</strong> uses validation reconstruction MSE to discourage poor reconstruction, while validation clustering AMI determines the best completed trial.

This example combines [Python config updates](configure-workflows/index.md#python-config-objects), a Lightning callback, and a metric returned by evaluation. Optuna manages the study, and BenchRep runs the workflows.

<strong class="highlight">Requirements.</strong> Install the Optuna extra which includes its dashboard:

```bash
python -m pip install "benchrep[optuna] @ git+https://github.com/uhlmanngroup/BenchRep.git"
```

## Run the example

From the directory where you want to keep your data and outputs, run:

```bash
python -m benchrep.examples.usage.10_optuna_composite_vae
```

The example attempts 10 trials and writes to `outputs/optuna_composite_vae/<timestamp>/`. Open <http://127.0.0.1:8080> to follow the study in the live dashboard.

!!! note "Trial 0 fails deliberately"
    The example gives trial 0 incompatible datamodule settings to demonstrate error capture. It records the failure and proceeds to the next trial. 

The [full script](https://github.com/uhlmanngroup/BenchRep/blob/main/examples/usage/10_optuna_composite_vae.py) contains the complete implementation. The snippets below explain the key logic.

## Define the search space

An Optuna **study** contains a set of **trials**. Each trial requests hyperparameter values, runs the workflows, and returns a scalar objective. The supplied model combines an MLP encoder, a Gaussian variational head, and an MLP decoder.

| Hyperparameter | Search space | What changes |
|---|---|---|
| Learning rate | `1e-5` to `1e-2`, logarithmic | Adam learning rate |
| Beta | `1e-4` to `1.0`, logarithmic | KL-divergence loss weight |
| Latent dimension | `8`, `16`, `32`, `64` | Variational head output and decoder input |
| Hidden width | `64`, `128`, `256` | Encoder and decoder hidden-layer widths |

Inside `run_trial()`, the `suggest_*` calls define these ranges and obtain values from the study’s sampler:

```python
learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
beta = trial.suggest_float("beta", 1e-4, 1.0, log=True)
latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 64])
hidden_width = trial.suggest_categorical("hidden_width", [64, 128, 256])
```

The script applies those values with `update_config()`, which returns a validated copy of the base config. See [Python config objects](configure-workflows/index.md#python-config-objects). For this composite VAE, update the variational head’s latent dimension and the decoder’s input dimension together so they match.

```python
components = "composite_model_components"
config = update_config(BASE_CONFIG, {
    "run.output_root": OUTPUT_DIR / f"trial_{trial.number:04d}",
    "optimizer.params.lr": learning_rate,
    "losses.regularization.0.weight": beta,
    f"{components}.image_encoder.params.hidden_dims": [hidden_width],
    f"{components}.image_decoder.params.hidden_dims": [hidden_width],
    f"{components}.variational_head.params.latent_dim": latent_dim,
    f"{components}.image_decoder.params.input_dim": latent_dim,
})
```

## Prune during training

Pass Optuna’s Lightning callback through the training entrypoint’s `runtime_callbacks` argument:

```python
training_result = train_composite(
    config=config,
    runtime_callbacks=[
        PyTorchLightningPruningCallback(
            trial, monitor="losses/val/reconstruction/mse"
        ),
    ],
)
```

!!! note "Runtime and configured callbacks"
    `runtime_callbacks` accepts callback instances created in Python, such as this pruning callback bound to the current Optuna trial. These are added alongside callbacks configured through `additional_callbacks`, `checkpointing`, and `early_stopping`. See [Callbacks and checkpoints](../understanding/training/callbacks.md) for details.

At each validation epoch, the callback reports reconstruction MSE to Optuna and raises `TrialPruned` if the study’s pruner decides to stop the trial. See the [Lightning callback reference](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.PyTorchLightningPruningCallback.html).

In this training YAML checkpoints are also selected based on validation reconstruction MSE. This is to keep checkpoint selection consistent while beta, affecting KL loss, varies across trials.

## Evaluate on the validation split

Since training-time pruning uses the validation split, we reuse that split for prediction so the final clustering objective assesses the same samples. We pass `training_result.datamodule.val_dataset` to a new datamodule supplied through `PredictionOverrides`, preserving the subset’s transforms and bypassing the prediction config’s dataset and transform settings. See [External models and datamodules](../understanding/extensibility/runtime-implementations.md) for the implementation contracts.

```python
validation_dataset = training_result.datamodule.val_dataset

prediction_result = predict_composite(
    config=example_config_path(EXAMPLE, stage="prediction"),
    overrides=PredictionOverrides(
        training_manifest_path=training_result.manifest_path,
        datamodule=BenchRepDataModule(
            predict_dataset=validation_dataset,
            val_fraction=0.0,
            batch_size=128,
            num_workers=0,
            seed=SEED,
        ),
    ),
)
```

The prediction YAML exports the latent means (`z_mu`). Evaluation runs k-means with 10 clusters on those embeddings and computes AMI against the digit labels. The test split remains available for evaluating the selected configuration afterward.

Following evaluation, the trial extracts its score using the path shown in `metrics.json`:

```python
ami = float(evaluation_result.get_metric(
    "clustering", "external", "kmeans", "metrics", "adjusted_mutual_info", "value"
))
if not math.isfinite(ami):
    raise ValueError(f"Evaluation returned non-finite AMI: {ami}")

return -ami
```

The study minimizes negative AMI, so the best completed trial has the highest AMI. Pruning compares reconstruction errors separately. This favors trials that retain reconstruction ability, but introduces a tradeoff: a trial pruned for poor reconstruction could have produced useful clusters.

## Configure the study

The script creates persistent storage, chooses a sampler and pruner, and then runs the objective repeatedly:

```python
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
    ),
    storage=storage,
)
```

These settings allow pruning after 5 completed trials, starting at epoch index 5 of each subsequent trial. See Optuna’s [median pruner reference](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html) for details.

!!! note "A longer study for adaptive sampling"
    TPE starts with random sampling, then uses trial results to guide suggestions. We use 10 trials to keep the example fast, but TPE’s default `n_startup_trials=10` keeps them all in the random phase. Increase `N_TRIALS` or lower the sampler’s `n_startup_trials` to enable adaptive sampling. Sampler and pruner startup settings are independent. See the [TPE reference](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).

## Handle failed trials

Trial 0 deliberately sets incompatible `num_workers` and `persistent_workers` values. The objective wrapper records the exception type and message through `trial.set_user_attr()`, then re-raises it. With `catch=(Exception,)`, Optuna marks the trial as `FAIL` and continues. 

The wrapper handles `TrialPruned` separately to avoid recording expected pruning as an error. For your own studies, you can narrow `catch` to the errors you expect (e.g. `ValidationError` would have sufficed in this example). Remove the deliberate failure when running an actual search.

## Inspect the results

The dashboard updates while the study runs. It includes objective values, sampled parameters, parameter importance, intermediate reconstruction errors, and trial states. 

| Output | Contents |
|---|---|
| `trials.csv` | Trial numbers, states, objective values, parameters, timings, and user attributes |
| `optuna.db` | Persistent study history, including intermediate values for the dashboard |
| `trial_XXXX/` | BenchRep workflow outputs and provenance for that trial |

The CSV updates after each trial and on exit, recording output paths and captured errors. Its `value` column contains negative AMI for completed trials and the last reported MSE for pruned trials. Failed trials may have no output directory.

The live dashboard stops when the script exits. To inspect the saved study later:

```bash
optuna-dashboard "sqlite:///outputs/optuna_composite_vae/<timestamp>/optuna.db"
```

See the [dashboard documentation](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html) for more information.

!!! note "Parallel trials"
    Keep `n_jobs=1`: Optuna uses threads for higher values, while BenchRep requires separate processes for concurrent trials.

    Create a named study shared by all workers, each running `study.optimize(..., n_jobs=1)`. Replace SQLite with `JournalStorage` and `JournalFileBackend` on one machine, or PostgreSQL/MySQL across machines. See Optuna’s [multiprocessing guide](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#multi-process-optimization) and [SQLite guidance](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-solve-the-error-that-occurs-when-performing-parallel-optimization-with-sqlite3).

    Use a shared output root, separate trial directories, and explicit CPU/GPU allocations. Keep dashboard and CSV export in one coordinating process. For GPU workers, use `spawn` as described in [PyTorch’s guidance](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html). Trial counts apply per worker, so divide the total accordingly.
