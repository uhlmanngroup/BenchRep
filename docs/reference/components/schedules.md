# Loss-weight schedules

A schedule's `factor(position)` multiplies a loss's base weight. Position means the zero-based epoch index or completed optimizer-step count selected by the loss term's interval. Calls do not advance the schedule: the same position produces the same factor.

These schedules control loss weights, not the optimizer's learning rate. See [`LossWeightSchedule`][benchrep.architecture.losses.weight_schedules.LossWeightSchedule] for the extension interface and the [catalogue](../components.md#catalogue-loss_weight_schedule) for registered names.

::: benchrep.architecture.losses.weight_schedules.LinearSchedule
    options:
      heading_level: 2
      members:
        - factor
      inherited_members: true

::: benchrep.architecture.losses.weight_schedules.CosineSchedule
    options:
      heading_level: 2
      members:
        - factor
      inherited_members: true

::: benchrep.architecture.losses.weight_schedules.SigmoidSchedule
    options:
      heading_level: 2
      members:
        - factor
      inherited_members: true

::: benchrep.architecture.losses.weight_schedules.ExponentialSchedule
    options:
      heading_level: 2
      members:
        - factor
      inherited_members: true

::: benchrep.architecture.losses.weight_schedules.PiecewiseConstantSchedule
    options:
      heading_level: 2
      members:
        - factor
      inherited_members: true

::: benchrep.architecture.losses.weight_schedules.PiecewiseLinearSchedule
    options:
      heading_level: 2
      members:
        - factor
      inherited_members: true

::: benchrep.architecture.losses.weight_schedules.CyclicSchedule
    options:
      heading_level: 2
      members:
        - factor
      inherited_members: true
