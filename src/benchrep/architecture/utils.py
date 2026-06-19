from __future__ import annotations

from torch import nn


IntPair = int | tuple[int, int]

def validate_int_pair(
    value: IntPair,
    *,
    name: str,
    min_value: int,
    allow_equal_min: bool = True,
) -> None:
    values = (value, value) if isinstance(value, int) else value

    invalid = (
        any(v < min_value for v in values)
        if allow_equal_min
        else any(v <= min_value for v in values)
    )

    if invalid:
        comparator = ">=" if allow_equal_min else ">"
        raise ValueError(f"{name} values must be {comparator} {min_value}, got {value}.")


def resolve_activation(
    activation: str | type[nn.Module] | None,
) -> type[nn.Module]:
    """Resolve an activation name or nn.Module class to an nn.Module class.

    ``None`` defaults to ReLU for backward compatibility with existing configs.
    """
    if activation is None:
        return nn.ReLU

    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation

    if not isinstance(activation, str):
        raise TypeError(
            "activation must be None, a string, or an nn.Module class. "
            f"Got {type(activation).__name__}."
        )

    activation_key = activation.lower().replace("_", "").replace("-", "")

    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }

    if activation_key not in activations:
        raise ValueError(
            f"Unsupported activation {activation!r}. "
            f"Supported activations are: {sorted(activations)}."
        )

    return activations[activation_key]