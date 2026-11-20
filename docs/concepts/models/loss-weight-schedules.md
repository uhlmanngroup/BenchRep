# Loss-weight schedules

A loss-weight schedule changes how strongly an objective contributes as training progresses. For example, you can gradually introduce a VAE's KL term while keeping its reconstruction weight constant. This changes the objective balance, independently of any [learning-rate scheduler](../training/optimization.md).

Every scheduled term uses:

```text
effective weight = configured weight × schedule factor
```

The configured `weight` stays unchanged. Its schedule supplies a factor based on the current training position, and the model multiplies the raw loss by the resulting effective weight.

## Introduce a loss gradually

This canonical VAE fragment increases the KL weight from `0` to `0.001` over epoch indices `0` to `10`:

```yaml
losses:
  reconstruction:
    - name: mse
  regularization:
    - name: gaussian_kl
      weight: 0.001
      weight_schedule:
        name: linear
        interval: epoch
        params:
          start: 0
          end: 10
          start_factor: 0.0
          end_factor: 1.0
```

At epoch index `5`, the factor is `0.5` and the effective KL weight is `0.0005`. At index `10` and thereafter, the weight is `0.001`. Epoch indices start at `0`, so the endpoint is reached in the 11th epoch. The same schedule configuration works on a composite loss term alongside its ordinary wiring.

Schedules can be applied independently to any loss role. Factors must be finite and nonnegative, but may exceed `1`. A factor of `0` removes the contribution while retaining the loss computation and logging.

## Choose the position

| `interval` | Position supplied to the schedule |
| --- | --- |
| `epoch` | Current zero-based epoch index; the default |
| `step` | Number of completed optimizer steps |

With gradient accumulation, multiple batches can use the same step position. Calling a schedule does not advance it: its factor is evaluated from the model's current position rather than from a separate call counter.

Training and validation both use the current scheduled weights. Consequently, a weighted validation total can change because the weights changed, even when the raw losses did not. When choosing a [checkpoint or early-stopping monitor](../training/callbacks.md), distinguish the weighted objective from the individual raw losses.

## Available schedules

| Registered name | Behavior |
| --- | --- |
| [`linear`][benchrep.architecture.losses.weight_schedules.LinearSchedule] | Interpolates linearly between start and end factors |
| [`cosine`][benchrep.architecture.losses.weight_schedules.CosineSchedule] | Uses a half-cosine transition, changing more slowly near the endpoints |
| [`sigmoid`][benchrep.architecture.losses.weight_schedules.SigmoidSchedule] | Uses an S-shaped transition, controlled by `steepness` |
| [`exponential`][benchrep.architecture.losses.weight_schedules.ExponentialSchedule] | Interpolates multiplicatively between strictly positive endpoint factors |
| [`piecewise_constant`][benchrep.architecture.losses.weight_schedules.PiecewiseConstantSchedule] | Holds each milestone's factor until the next milestone |
| [`piecewise_linear`][benchrep.architecture.losses.weight_schedules.PiecewiseLinearSchedule] | Interpolates linearly between consecutive milestones |

The first 4 use `start`, `end`, `start_factor`, and `end_factor`. Before and after the ramp, they hold the corresponding endpoint factor. Piecewise schedules instead take ordered `[position, factor]` pairs under `params.milestones`, with at least 2 milestones; they also hold the nearest endpoint factor outside that range.

For example, this schedule rises, holds, then falls:

```yaml
weight_schedule:
  name: piecewise_linear
  interval: epoch
  params:
    milestones: [[0, 0.0], [5, 1.0], [15, 1.0], [20, 0.2]]
```

See the [schedule reference](../../reference/components/schedules.md) for interpolation details and the [training schema](../../reference/schemas/training.md) for parameter constraints. [`inspect_registry("loss_weight_schedule")`][benchrep.inspect_registry] lists registered choices.

## Repeat a schedule

Add `cycle` to repeat a schedule instead of holding its final factor indefinitely:

```yaml
weight_schedule:
  name: linear
  interval: epoch
  params:
    start: 0
    end: 5
    start_factor: 0.0
    end_factor: 1.0
  cycle:
    length: 10
```

Each 10-epoch cycle rises over local positions `0`–`5`, holds at `1` through position `9`, then resets to position `0`. The cycle starts at training position `0`, and its length uses the same units as `interval`. For built-ins, the final ramp position or milestone must be less than the cycle length.

## Custom schedules

A registered schedule constructor returns an object satisfying [`LossWeightSchedule`][benchrep.architecture.losses.weight_schedules.LossWeightSchedule]: its `factor(position)` method supplies the multiplier. The interface receives a position, not the model, batch, or validation metrics; metric-adaptive weighting needs a different implementation approach.

Select a custom registration through the same `name`, `params`, and `interval` fields. See [Registering components](../extensibility/registration.md) for the registration mechanism.

## Implementation

Weight schedules are evaluated when a loss is computed. They do not install a callback that mutates the base weight or require Lightning to call a scheduler's `step()` method.

1. **Resolution:** the [loss resolver](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/loss_resolver.py) stores the canonical schedule name, constructor parameters, interval, and optional cycle length in a `LossWeightScheduleSpec` attached to the term's `LossSpec`.
2. **Construction:** the [loss builder](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/loss_builder.py) creates the schedule through its registry and checks that it provides a callable `factor(position)`. If cycling is configured, it wraps the instance in [`CyclicSchedule`][benchrep.architecture.losses.weight_schedules.CyclicSchedule]. The resulting object is stored on the `LossTerm`.
3. **Execution:** the model passes `current_epoch` and `global_step` to [`LossTerm.effective_weight()`][benchrep.architecture.losses.base.LossTerm.effective_weight]. The term selects the configured position, evaluates the factor, and multiplies it by the base weight.

The built-in ramps convert a position within their interval into a fraction between `0` and `1`, then apply their interpolation rule. Piecewise schedules locate the surrounding milestones and either hold the earlier factor or interpolate between the pair. The cyclic wrapper changes only the supplied position, using `position % length`; it reuses the underlying schedule unchanged.

These built-ins calculate directly from their parameters and the supplied position, without maintaining a separate progress counter. Repeated calls at the same position therefore return the same factor, including calls made during validation. Their implementation is in the [schedule module](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/architecture/losses/weight_schedules.py).

Before using a scheduled weight, `LossTerm` checks that the factor is a finite, nonnegative real scalar and that the multiplication does not overflow. The model logs the resulting coefficient alongside the raw and weighted loss; see [logged contributions](losses.md#identify-and-inspect-contributions).
