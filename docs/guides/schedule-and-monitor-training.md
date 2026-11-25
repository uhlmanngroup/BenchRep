# Schedule and monitor training

<strong class="highlight">How can I diagnose training problems, balance objectives, and adjust the learning process over time?</strong> BenchRep supports inspecting gradient norms, weighting individual loss terms, and scheduling both loss weights and learning rates.

Here we train a small VAE on [MNIST](http://yann.lecun.com/exdb/mnist/), inspect how reconstruction and regularization influence the encoder, gradually increase the KL-divergence weight, and reduce the learning rate later in training. We track these changes through CSV logs or interactive plots in Weights & Biases.

The [:material-script-text-outline: complete script](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/examples/usage/06_schedule_and_monitor_training.py) uses a [training YAML and optional W&B logger YAML](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/examples/configs/06_schedule_and_monitor_training).

## Run the example

With BenchRep installed, choose either logging option and run from the directory where you want to keep the data and outputs:

=== "CSV (default)"

    Save training metrics locally as CSV files.

    ```bash
    python -m benchrep.examples.usage.06_schedule_and_monitor_training
    ```

=== "Weights & Biases"

    For live interactive plots, install the extra if needed, authenticate, and select W&B:

    ```bash
    python -m pip install "benchrep[wandb] @ git+https://github.com/uhlmanngroup/BenchRep.git"
    wandb login
    python -m benchrep.examples.usage.06_schedule_and_monitor_training --logger wandb
    ```

See [Choose a logger](#choose-a-logger) for logger configuration and other backends.

The example downloads [MNIST](http://yann.lecun.com/exdb/mnist/) if needed and trains for 8 epochs, using an MLP encoder and decoder with a 16-dimensional latent space. It holds out 10% of the training split for validation and limits each epoch to 40 training batches and 8 validation batches, at batch size 128. This keeps the experiment short enough to inspect without a long training run.

The script prints the training manifest, the CSV metric-history path, and a compact summary of the last logged training step in each epoch. Outputs are written under `outputs/06_schedule_and_monitor_training/`. This example focuses on training; for a full training → prediction → evaluation chain, see [Run your first workflow chain](first-workflow-chain.md).

## Schedule the objective and the optimizer

The VAE combines [reconstruction MSE][benchrep.architecture.losses.reconstruction.MSEReconstructionLoss] with [Gaussian KL divergence][benchrep.architecture.losses.regularization.GaussianKLDivergenceLoss]. The former rewards reconstruction accuracy; the latter regularizes the approximate posterior toward the standard-normal prior. We increase regularization gradually rather than giving it its final coefficient from the first batch.

| Control | What changes here | Purpose |
| --- | --- | --- |
| Loss-weight schedule | KL coefficient increases from `0.0001` to `0.001` over epoch indices 0–3 | Change the relative contribution of regularization |
| Learning-rate schedule | Adam's learning rate drops from `0.001` to `0.00025` at epoch index 5 | Change the optimizer's step-size setting |

### Increase the KL contribution

The loss term declares a base weight and a schedule that multiplies it:

```yaml
losses:
  reconstruction:
    - name: mse
      weight: 1.0
      params:
        reduction: mean
  regularization:
    - name: gaussian_kl
      weight: 0.001
      params:
        reduction: mean
      weight_schedule:
        name: linear
        interval: epoch
        params:
          start: 0
          end: 3
          start_factor: 0.1
          end_factor: 1.0
```

The effective coefficient is `weight × schedule factor`. With this [linear schedule][benchrep.architecture.losses.weight_schedules.LinearSchedule], the KL weights at epoch indices 0, 1, 2, and 3 are `0.0001`, `0.0004`, `0.0007`, and `0.001`; the last value is retained afterward. Reconstruction keeps weight `1.0`.

These weights are not normalized. Their useful scale depends on the loss definitions and reductions, so this is an illustrative recipe rather than a general recommendation for VAEs. See [Loss-weight schedules](../concepts/models/loss-weight-schedules.md) for other schedules, step-based timing, and custom implementations.

### Reduce the learning rate later

Once the KL coefficient has settled, a multi-step scheduler reduces Adam's learning rate:

```yaml
optimizer:
  name: adam
  params:
    lr: 0.001
  lr_scheduler:
    name: multi_step
    interval: epoch
    params:
      milestones: [5]
      gamma: 0.25
```

The registered scheduler wraps PyTorch's `MultiStepLR`. Lightning advances it at epoch boundaries; epoch index 5 is the sixth epoch. This changes the learning-rate setting, independently of the loss weights. See [Training and optimization](../concepts/training/optimization.md#schedule-the-learning-rate) and the [scheduler catalogue](../reference/components.md#catalogue-lr_scheduler).

<figure>
  <a href="../assets/guides/schedule-and-monitor-training/schedules.png">
    <img src="../assets/guides/schedule-and-monitor-training/schedules.png" alt="Logged KL coefficient and learning rate on aligned epoch axes, showing a KL ramp over epochs zero to three and a learning-rate drop at epoch five">
  </a>
  <figcaption>The KL coefficient changes the objective; the learning rate changes the optimizer's step-size setting. Plot their recorded values to check when each schedule takes effect.</figcaption>
</figure>

## Measure each objective's effect on the encoder

A loss value measures an objective; its gradient describes how that objective asks the parameters to change. To compare the two objectives on the **same parameters**, the YAML measures both with respect to the encoder:

```yaml
inspection:
  gradient_norms:
    - kind: per_loss
      target: encoder
      losses: [reconstruction.mse, regularization.gaussian_kl]
      norm_type: l2
      every_n_batches: 1
    - kind: optimizer_gradients
      target: encoder
      norm_type: l2
      every_n_optimizer_steps: 1
```

`per_loss` records the raw gradient norm of each term and its norm after applying the effective loss weight. `optimizer_gradients` measures the encoder's combined stored gradient before the optimizer update. These are measurements, not an automatic method for balancing losses.

| Measurement | How to read it |
| --- | --- |
| Raw per-loss norm | How sensitive that unweighted loss is to the encoder parameters on the current batch |
| Weighted per-loss norm | The size of that term's gradient contribution after its current weight is applied |
| Optimizer-gradient norm | The size of the combined gradient presented to the optimizer for the encoder |

The combined norm is generally **not the sum of the individual norms**: gradients can reinforce or oppose each other. Nor is it the size of Adam's eventual parameter update, which also depends on its learning rate and running statistics. Per-loss norms help identify a dominant contribution, but cannot establish whether objectives agree in direction.

<figure>
  <a href="../assets/guides/schedule-and-monitor-training/encoder-gradients.png">
    <img src="../assets/guides/schedule-and-monitor-training/encoder-gradients.png" alt="Reconstruction and KL encoder-gradient histories, with raw and weighted per-loss norms and the combined optimizer-gradient norm">
  </a>
  <figcaption>Compare weighted contributions on the same encoder parameters, and use the raw norms to distinguish changes in loss sensitivity from changes in the scheduled coefficient.</figcaption>
</figure>

The example samples every batch and logs every training step so these histories are easy to compare.

??? tip "Reduce inspection overhead"
    Per-loss inspection adds differentiation work and can increase runtime and memory use. For longer runs, increase `every_n_batches` and `every_n_optimizer_steps`, and coordinate them with `trainer.log_every_n_steps`. See [Monitoring and inspection](../concepts/training/inspection.md#inspect-gradient-contributions) for supported targets and execution conditions.

## Choose a logger

The training YAML uses CSV logging:

```yaml
logger:
  name: csv
  params:
    save_dir: outputs/06_schedule_and_monitor_training
    name: csv
    flush_logs_every_n_steps: 10
```

When launched with `--logger wandb`, the script reads `wandb_logger.yaml` into a [`TrainingLoggerConfig`][benchrep.assembly.schemas.training_config_schema.TrainingLoggerConfig], then replaces only the logger section through [`TrainingOverrides`][benchrep.TrainingOverrides]:

```python
overrides = TrainingOverrides(config_components={"logger": wandb_logger})

training_result = train_vae(
    config=example_config_path(EXAMPLE, stage="training"),
    overrides=overrides,
)
```

The same [`train_vae()`][benchrep.train_vae] call therefore uses the same model, data, and optimization settings with either backend. In W&B, add panels for the loss weights, learning rate, raw validation losses, and encoder-gradient norms. Use a shared step or epoch axis when comparing histories.

TensorBoard and MLflow are also [built-in logger choices](../reference/components.md#catalogue-logger), with their own optional dependencies and constructor parameters. Select them in the YAML when using the default command; this script's `--logger` flag only switches between its CSV/default path and the supplied W&B override.

For another backend, register a Lightning-compatible logger class or factory through [`LOGGERS.register()`][benchrep.assembly.registries.core.Registry.register] before launching training, then select its name in `logger`. [`inspect_registry("logger")`][benchrep.inspect_registry] lists the available choices. See [Register external components](register-components.md) for the registration pattern.

## Read the histories

The main metric names are:

| Series | Meaning |
| --- | --- |
| `lr/group_0` | Learning rate for the optimizer's first parameter group |
| `loss_weights/train/regularization/gaussian_kl` | Effective KL coefficient at the logged training step |
| `losses/val/reconstruction/mse` | Unweighted validation reconstruction error |
| `losses/val/regularization/gaussian_kl` | Unweighted validation KL loss |
| `losses/val/total_weighted_loss` | Validation loss using that epoch's effective coefficients |

Encoder-gradient series start with `grad_norms_wrt_encoder/`. The script compares `per_loss/reconstruction/mse/l2/weighted`, `per_loss/regularization/gaussian_kl/l2/weighted`, and `optimizer/l2`. The corresponding `/raw` per-loss series are also available.

For CSV logging, the script locates `metrics.csv` through `training_result.trainer.logger.log_dir`. Training-step metrics, validation metrics, and epoch summaries occupy different rows, so blank cells do not necessarily indicate missing measurements. Its `show_csv_summary()` helper selects rows containing the requested training metrics, then prints the last logged step of each epoch. **These are individual step measurements, not epoch averages.**

Read the histories together: verify the scheduled values first, then compare the weighted gradient contributions and raw validation losses. A growing KL coefficient need not produce a steadily growing KL gradient, because the model and sampled batches are also changing. The total weighted loss may rise simply because the coefficient increased.

## Select checkpoints with a consistent metric

The example monitors raw validation reconstruction MSE:

```yaml
checkpointing:
  monitor: losses/val/reconstruction/mse
  mode: min
  save_top_k: 1
  save_last: true
```

Unlike total weighted loss, this metric's definition does not change with the KL schedule. It selects the best reconstruction checkpoint and also saves the final one. That is a deliberate criterion for this demonstration, not a guarantee that the selected latent representation is best for clustering or prediction. See [Callbacks and checkpoints](../concepts/training/callbacks.md).

Experiment logging complements BenchRep's workflow records. The returned [`TrainingWorkflowResult`][benchrep.workflows.train.TrainingWorkflowResult] provides the manifest and status report, while the logger holds the metric history. Setting `logger: null` disables experiment tracking without disabling manifests or workflow logs.

To explore the tradeoff, change the KL schedule while keeping the learning-rate schedule fixed, or vice versa, and compare the same histories. For a downstream representation comparison using shared prediction and evaluation settings, continue with [Learn from multiple views](learn-from-multiple-views.md).
