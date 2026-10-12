from __future__ import annotations

from typing import Protocol, runtime_checkable

from bisect import bisect_right
from collections.abc import Sequence
import math
from numbers import Real


@runtime_checkable
class LossWeightSchedule(Protocol):
    """Deterministic, position-based multiplier for a loss's base weight.

    Implementations may store configuration, but must not advance counters
    or depend on call history. Position is a nonnegative integer measured
    in epochs or completed optimizer steps, as configured by BenchRep.

    Constructors accept the configured schedule parameters as keyword
    arguments and validate them independently of the configuration schema.
    """

    def factor(self, position: int) -> float:
        """Return a finite, nonnegative multiplier; values above 1 are allowed."""
        ...


def _validate_position(value: int, name: str = "position") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < 0:
        raise ValueError(f"`{name}` must be nonnegative.")
    return value


def _validate_factor(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"`{name}` must be finite and nonnegative.")
    return value


def _interpolate(start_value: float, end_value: float, fraction: float) -> float:
    delta = end_value - start_value
    return start_value + fraction * delta


class LinearSchedule:
    """Linear ramp, holding the endpoint factors outside its interval."""

    def __init__(
        self,
        *,
        end: int,
        start: int = 0,
        start_factor: float = 0.0,
        end_factor: float = 1.0,
    ) -> None:
        self.start = _validate_position(start, "start")
        self.end = _validate_position(end, "end")
        if self.end <= self.start:
            raise ValueError("`end` must be greater than `start`.")

        self.start_factor = _validate_factor(start_factor, "start_factor")
        self.end_factor = _validate_factor(end_factor, "end_factor")

    def factor(self, position: int) -> float:
        position = _validate_position(position)
        if position <= self.start:
            return self.start_factor
        if position >= self.end:
            return self.end_factor

        fraction = (position - self.start) / (self.end - self.start)
        return self._ramp_factor(fraction)

    def _ramp_factor(self, fraction: float) -> float:
        return _interpolate(self.start_factor, self.end_factor, fraction)


class CosineSchedule(LinearSchedule):
    """Half-cosine ramp with the same endpoint behavior as a linear ramp."""

    def _ramp_factor(self, fraction: float) -> float:
        fraction = 0.5 * (1.0 - math.cos(math.pi * fraction))
        return _interpolate(self.start_factor, self.end_factor, fraction)


class SigmoidSchedule(LinearSchedule):
    """Endpoint-normalized logistic ramp with configurable steepness.

    Implementation relies on the identities:
        sigmoid(x) = (1 + tanh(x / 2)) / 2
        tanh(-x) = -tanh(x)
    Starts with a logistic sigmoid centered at 0.5:
        s(t) = sigmoid(k * (t - 0.5)), with t = fraction and k = steepness
    Then substitutes:
        centered_tanh = tanh(k * (t - 0.5) / 2)
        endpoint_tanh = tanh(k / 4)
    Into the normalization:
        f(t) = (s(t) - s(0)) / (s(1) - s(0)), so f(0) = 0 and f(1) = 1
    Yielding:
        f(t) = 0.5 * (1 + centered_tanh / endpoint_tanh)

    This form avoids exponential overflow and subtraction of nearly equal
    sigmoid values near 0.5. As steepness approaches zero, f(t) approaches t.
    For steepness below 1e-6, the implementation sets f(t) = t, producing
    linear interpolation between start_factor and end_factor.
    """

    def __init__(
        self,
        *,
        end: int,
        start: int = 0,
        start_factor: float = 0.0,
        end_factor: float = 1.0,
        steepness: float = 10.0,
    ) -> None:
        super().__init__(
            start=start,
            end=end,
            start_factor=start_factor,
            end_factor=end_factor,
        )
        self.steepness = _validate_factor(steepness, "steepness")
        if self.steepness == 0:
            raise ValueError("`steepness` must be strictly positive.")

    def _ramp_factor(self, fraction: float) -> float:
        if self.steepness >= 1e-6:
            fraction = 0.5 * (
                    1.0
                    + math.tanh(
                (self.steepness * 0.5) * (fraction - 0.5)
            )
                    / math.tanh(self.steepness * 0.25)
            )

        # Guard against floating-point rounding outside [0, 1].
        fraction = min(1.0, max(0.0, fraction))

        # Map the normalized fraction onto the configured endpoint factors.
        return _interpolate(self.start_factor, self.end_factor, fraction)


class ExponentialSchedule(LinearSchedule):
    """Exponential ramp implemented by linear interpolation in log space.

    Both endpoint factors must be strictly positive.
    """

    def __init__(
        self,
        *,
        end: int,
        start_factor: float,
        start: int = 0,
        end_factor: float = 1.0,
    ) -> None:
        super().__init__(
            start=start,
            end=end,
            start_factor=start_factor,
            end_factor=end_factor,
        )
        if self.start_factor == 0 or self.end_factor == 0:
            raise ValueError(
                "Exponential endpoint factors must be strictly positive."
            )

    def _ramp_factor(self, fraction: float) -> float:
        return math.exp(
            _interpolate(
                math.log(self.start_factor),
                math.log(self.end_factor),
                fraction,
            )
        )


class PiecewiseConstantSchedule:
    """Hold each milestone's factor until the next milestone.

    `milestones` is a sequence of (position, factor) pairs.

    Positions are zero-based epoch indices or completed optimizer-step counts,
    depending on the configured interval. Factors multiply the loss's base weight.

    Each new factor takes effect exactly at its milestone. Before the first milestone
    and after the last, the respective endpoint factors are held.

    At least two milestones are required, with strictly increasing nonnegative integer
    positions and finite, nonnegative factors.
    """

    def __init__(
        self,
        *,
        milestones: Sequence[tuple[int, float]],
    ) -> None:
        if len(milestones) < 2:
            raise ValueError("At least two milestones are required.")

        validated = []
        for milestone in milestones:
            if len(milestone) != 2:
                raise ValueError(
                    "Each milestone must contain a position and a factor."
                )
            position, factor = milestone
            validated.append((
                _validate_position(position, "milestone position"),
                _validate_factor(factor, "milestone factor"),
            ))

        self.milestones = tuple(validated)
        self._positions = tuple(position for position, _ in self.milestones)

        if any(
            right <= left
            for left, right in zip(self._positions, self._positions[1:])
        ):
            raise ValueError(
                "Milestone positions must be strictly increasing."
            )

    def factor(self, position: int) -> float:
        position = _validate_position(position)
        if position <= self._positions[0]:
            return self.milestones[0][1]
        if position >= self._positions[-1]:
            return self.milestones[-1][1]

        index = bisect_right(self._positions, position) - 1
        return self._segment_factor(index, position)

    def _segment_factor(self, index: int, position: int) -> float:
        return self.milestones[index][1]


class PiecewiseLinearSchedule(PiecewiseConstantSchedule):
    """Interpolate linearly between consecutive milestones.

    `milestones` is a sequence of (position, factor) pairs. Positions are
    zero-based epoch indices or completed optimizer-step counts, depending
    on the configured interval. Factors multiply the loss's base weight.

    Each segment interpolates between its endpoint factors; equal factors
    produce a plateau. Before the first milestone and after the last,
    the respective endpoint factors are held.

    At least two milestones are required, with strictly increasing nonnegative
    integer positions and finite, nonnegative factors.
    """

    def _segment_factor(self, index: int, position: int) -> float:
        left_position, left_factor = self.milestones[index]
        right_position, right_factor = self.milestones[index + 1]
        fraction = (
            (position - left_position) / (right_position - left_position)
        )
        return _interpolate(left_factor, right_factor, fraction)


class CyclicSchedule:
    """Repeat a schedule every `length` positions.

    The wrapped schedule receives positions from 0 through length - 1.
    At each cycle boundary, its position resets to zero.
    """

    def __init__(
        self,
        *,
        schedule: LossWeightSchedule,
        length: int,
    ) -> None:
        if not callable(getattr(schedule, "factor", None)):
            raise TypeError("`schedule` must provide a callable factor method.")

        self.length = _validate_position(length, "length")
        if self.length == 0:
            raise ValueError("`length` must be strictly positive.")

        self.schedule = schedule

    def factor(self, position: int) -> float:
        position = _validate_position(position)
        cycle_position = position % self.length
        return self.schedule.factor(cycle_position)
