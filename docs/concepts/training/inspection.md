# Monitoring and inspection

Training exposes several views of a run: workflow logs explain what BenchRep did, experiment loggers record metric histories, gradient inspection measures how objectives affect parameters, and graph exports show model structure. Each answers a different question.

## Logs and experiment tracking

BenchRep's run log is independent of the optional Lightning experiment logger. Setting `logger: null` disables experiment tracking but preserves workflow logs, manifests, and status records. `capture_stdout=True` on a workflow call additionally captures console streams during execution; see [Run directories and result objects](../results/runs.md).

To persist metric histories locally:

```yaml
logger:
  name: csv
  params:
    save_dir: experiment_logs
    name: my-experiment
```

Other registered backends include TensorBoard, Weights & Biases, and MLflow. Their dependencies, authentication, and constructor settings depend on the backend; [`inspect_registry("logger", "csv")`][benchrep.inspect_registry] exposes the selected constructor. Logger parameters control the backend's output location, independently of BenchRep's workflow-log directory.

The built-in models log raw losses, weighted losses, effective loss weights, and total loss. An attached callback also logs learning rates as `lr/group_0`, `lr/group_1`, and so on before optimizer updates when an experiment logger is present. See [Loss metrics](../models/losses.md#identify-and-inspect-contributions) for the loss naming convention.

## Inspect gradient contributions

Loss magnitudes alone do not show how strongly objectives affect shared parameters. Gradient-norm inspection measures either individual loss contributions or the combined gradients used by the optimizer:

```yaml
inspection:
  gradient_norms:
    - kind: per_loss
      target: encoder
      losses: [reconstruction.mse, regularization.gaussian_kl]
      norm_type: l2
      every_n_batches: 10
    - kind: optimizer_gradients
      target: whole_model
      norm_type: l2
      every_n_optimizer_steps: 10
```

This example assumes a canonical VAE with the named loss terms. Loss references use their resolved identities: aliases normalize to canonical names, and a term with `id: view_a` and `name: mse` is referenced as `reconstruction.view_a_mse`.

| Measurement | What it measures | Timing |
| --- | --- | --- |
| `per_loss` | Raw gradient norm and that norm multiplied by the effective loss weight | On selected training batches, before ordinary backward computation |
| `optimizer_gradients` | Combined stored gradients for the selected parameters | Before selected optimizer updates, after accumulation and precision unscaling, before clipping |

Canonical targets are `encoder`, `decoder`, and `whole_model`; VAE also supports `variational_head`. Composite targets are configured component IDs or `whole_model`. External model overrides support optimizer-gradient inspection for `whole_model`, but not per-loss inspection.

Measurements require single-process training, automatic optimization, and one optimizer. Choose `l1`, `l2`, or `inf`; the norm treats selected gradient elements as one vector, rather than averaging a norm for each parameter tensor. Frozen parameters are excluded.

??? note "Reading gradient measurements"
    Per-loss norms describe magnitudes, not agreement between objectives. Two large gradients can reinforce or oppose each other; their norms alone do not reveal the direction. A connected zero gradient is reported as zero. If no usable measurement exists, BenchRep warns and skips it, retrying at later sampling positions.

    Per-loss measurements add autograd work and can increase memory use. Their frequency and optimizer-gradient frequency control sampling; persistence also follows Lightning's logging cadence. Configure a logger to retain the history.

## Inspect model structure

Composite training automatically attempts to export the resolved specification graph, showing declared components, connections, and loss wiring. This describes the configuration-derived plan and does not trace tensor operations inside components. It can also be previewed in the [configuration wizard](../workflows/wizards.md).

Torchview provides a different view: it traces model execution to show internal operations and shapes. Enable it with:

```yaml
inspection:
  torchview:
    enabled: true
    expand_nested: true
    depth: 10
```

These exports use the optional `model_graph` dependencies and a Graphviz executable. They are diagnostic artifacts; they do not determine model execution. Torchview export happens during post-training finalization. A failed optional graph export is recorded without discarding an otherwise usable training run.

## Implementation: observing without changing optimization

The resolver turns gradient requests into concrete target/loss selections. At fit start, a callback attaches a gradient inspector to the model and checks the supported execution conditions.

For per-loss inspection, the built-in model passes each raw loss to the inspector before adding its weighted contribution. The inspector uses `torch.autograd.grad(..., retain_graph=True)` to measure the selected parameter gradients without populating their `.grad` buffers. The retained graph remains available for Lightning's normal backward pass.

For optimizer-gradient inspection, the callback reads existing `.grad` buffers in `on_before_optimizer_step()`. It includes only parameters shared by the selected target and active optimizer, and does not modify them. The [inspection implementation](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/inspection/gradient_norms.py) handles sampling, norm calculation, and deduplicated warnings; those warnings also enter the training status records.
