from __future__ import annotations

from typing import Literal

from torch import nn


IntPair = int | tuple[int, int]

NormalizationLayout = Literal["vector", "spatial_2d"]

_NORMALIZATIONS_BY_LAYOUT = {
    "vector": ("batchnorm", "layernorm", "rmsnorm"),
    "spatial_2d": ("batchnorm", "instancenorm", "groupnorm"),
}


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


def validate_normalization(
    normalization: str | None,
    *,
    layout: NormalizationLayout,
    num_groups: int | None = None,
) -> str | None:
    """Validate and canonicalize a normalization configuration."""

    if normalization is None:
        if num_groups is not None:
            raise ValueError(
                "num_groups is only valid when normalization='groupnorm'."
            )
        return None

    if not isinstance(normalization, str):
        raise TypeError(
            "normalization must be None or a string, "
            f"got {type(normalization).__name__}."
        )

    normalization = normalization.lower()

    valid_normalizations = _NORMALIZATIONS_BY_LAYOUT[layout]

    if normalization not in valid_normalizations:
        raise ValueError(
            f"{layout!r} normalization must be one of "
            f"{valid_normalizations}, got {normalization!r}."
        )

    if normalization == "groupnorm":
        if num_groups is None:
            raise ValueError(
                "num_groups is required when normalization='groupnorm'."
            )
        if num_groups <= 0:
            raise ValueError(
                f"num_groups must be positive, got {num_groups}."
            )
    elif num_groups is not None:
        raise ValueError(
            "num_groups is only valid when normalization='groupnorm'."
        )

    return normalization


def resolve_normalization(
    normalization: str | None,
    *,
    num_features: int,
    layout: NormalizationLayout,
    num_groups: int | None = None,
) -> nn.Module | None:
    """Build a normalization module for the requested tensor layout."""

    normalization = validate_normalization(
        normalization,
        layout=layout,
        num_groups=num_groups,
    )

    if normalization is None:
        return None

    if num_features <= 0:
        raise ValueError(f"num_features must be positive, got {num_features}.")

    if normalization == "batchnorm":
        if layout == "vector":
            return nn.BatchNorm1d(num_features)
        return nn.BatchNorm2d(num_features)

    if normalization == "layernorm":
        return nn.LayerNorm(num_features)

    if normalization == "rmsnorm":
        return nn.RMSNorm(num_features)

    if normalization == "instancenorm":
        return nn.InstanceNorm2d(num_features)

    if normalization == "groupnorm":
        assert num_groups is not None

        if num_features % num_groups != 0:
            raise ValueError(
                "For GroupNorm, num_features must be divisible by num_groups. "
                f"Got num_features={num_features}, num_groups={num_groups}."
            )

        return nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_features,
        )

    raise RuntimeError(
        f"Unhandled normalization {normalization!r}."
    )
