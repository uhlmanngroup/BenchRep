# Callbacks and checkpoints

Callbacks run at points in Lightning's training lifecycle, such as after validation or before an optimizer update. BenchRep uses them for checkpointing, early stopping, and diagnostics, and lets you add registered callbacks or live Python instances.

## Choose what to save

Training always has a BenchRep-managed checkpoint callback. The default policy monitors `losses/val/total_weighted_loss`, keeps the best checkpoint, and also saves the last checkpoint:

```yaml
checkpointing:
  monitor: losses/val/total_weighted_loss
  mode: min
  save_top_k: 1
  save_last: true
```

The monitor must be a metric the model actually logs. For example, if there is no validation loader, choose an appropriate training metric or use a last-only policy:

```yaml
checkpointing:
  monitor: null
  save_last: true
```

With no monitor, BenchRep resolves `save_top_k` to `0`. More generally, `save_top_k: 0` disables ranked saving, positive values retain that many ranked checkpoints, and `-1` retains all checkpoints produced by that policy. At least ranked saving or `save_last` must remain enabled.

The manifest records the checkpoint directory and the actual best and last paths. [Prediction](../prediction/loading-models.md#select-a-checkpoint) uses these records; `best` is not an alias for `last`, and a last-only run must be predicted with `checkpoint: last`.

## Stop when progress stalls

Early stopping is optional and independent of checkpoint retention:

```yaml
early_stopping:
  monitor: losses/val/reconstruction/mse
  mode: min
  patience: 5
  min_delta: 0.0001
```

Patience counts checks without sufficient improvement, not necessarily epochs. Lightning's validation/check frequency determines when those checks occur. Other supported options, including finite-value checks and stopping thresholds, appear in the [training schema](../../reference/schemas/training.md).

Early stopping requests an orderly end to fitting. BenchRep records whether it triggered, its reason, and its tracked values. It does not restore the best weights into the returned model; select the desired checkpoint when predicting.

## Add callbacks

| Route | Use it when |
| --- | --- |
| `checkpointing` and `early_stopping` | Configuring BenchRep's managed checkpoint and stopping behavior |
| `additional_callbacks` | A registered constructor can create the callback from config parameters |
| Entrypoint `runtime_callbacks` | The callback needs live Python state, such as an Optuna trial |

For example, a configured accumulation schedule is instantiated once for the run:

```yaml
additional_callbacks:
  - name: gradient_accumulation_scheduler
    params:
      scheduling: {0: 1, 5: 2}
```

Runtime callbacks are already-created Lightning callback instances. They are added alongside configured callbacks, and their classes are recorded in provenance. See [Tune with Optuna](../../guides/optuna.md) for a pruning callback passed through `runtime_callbacks` and [Python overrides](../workflows/overrides.md#runtime-callbacks-are-separate) for their relationship to other runtime inputs.

The [callback catalogue](../../reference/components.md#catalogue-callback) lists built-ins. Additional callbacks supplement BenchRep's managed checkpointing and early stopping. Callback-specific requirements still apply; for example, some monitoring callbacks require an experiment logger.

## Implementation: assembling the Trainer's callbacks

The [Trainer builder](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/trainer_builder.py) constructs the managed checkpoint callback and learning-rate logger, adds requested gradient inspection and early stopping, then appends configured and runtime callbacks. It rejects repeated references to the same callback instance. Unless a model-summary callback is already supplied, it adds an expanded summary so composite components and loss modules are visible.

Lightning invokes these callbacks around the model's hooks. A callback changes behavior by participating in that lifecycle; it is not part of the model's forward graph or loss specification. Training callbacks are not inherited by prediction.

## Interruptions and checkpoint validation

For recognized SIGINT and SIGTERM interruptions, BenchRep attempts to finish the run's records, including checkpoint validation and interruption status. Its checkpoint subclass permits exception checkpoint saving for these graceful interruptions, rather than for arbitrary training errors. Hard termination cannot perform this finalization.

After fitting, BenchRep checks the files actually produced. No usable checkpoint is a failure: it writes the failure manifest and raises. A missing requested last checkpoint or fewer ranked checkpoints than requested can instead become warnings when another usable checkpoint exists. Ordinary exceptions during fitting generally propagate. See [Statuses and failures](../results/statuses.md) for the full distinction between recorded outcomes and raised errors.
