from __future__ import annotations

import torch
from torch import nn
import torchvision

from benchrep.architecture.encoders.base import BaseEncoder


import torch
from torch import nn
import torchvision

from benchrep.architecture.encoders.base import BaseEncoder


class TorchvisionResNet(BaseEncoder):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_dim: int,
        *,
        variant: str = "resnet18",
        weights: str | None = None,
        small_stem: bool = False
    ) -> None:
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError(
                "TorchvisionResNet input_shape must be a 3-tuple "
                f"(channels, height, width), got {input_shape}."
            )
        if any(dim <= 0 for dim in input_shape):
            raise ValueError(
                f"All input_shape dimensions must be positive, got {input_shape}."
            )
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        variant = variant.lower()
        valid_variants = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
            "resnet152": torchvision.models.resnet152,
        }

        if variant not in valid_variants:
            raise ValueError(
                "variant must be one of "
                f"{sorted(valid_variants)}, got {variant!r}."
            )

        input_channels = input_shape[0]

        if weights is not None and input_channels != 3:
            raise ValueError(
                "Pretrained ResNet weights currently require input_shape[0]=3. "
                "Use weights=None or add an explicit conv1 adaptation policy."
            )

        resnet = valid_variants[variant](weights=weights)

        # Use less aggressive stem for tiny images like MNIST
        if small_stem:
            resnet.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=resnet.conv1.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            resnet.maxpool = nn.Identity()
        # Modify original ResNet's first conv (which hardcodes 3 channels)
        elif input_channels != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=old_conv.bias is not None,
                padding_mode=old_conv.padding_mode,
            )

        # Read input size before original ResNet's classifier head, then drop it as a precaution
        backbone_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self._input_shape = tuple(input_shape)
        self._output_dim = output_dim
        self.variant = variant
        self.weights = weights
        self.small_stem = small_stem

        # Mirror original ResNet only up to layer4 as feature extractor
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.pool = resnet.avgpool
        self.projection = nn.Linear(backbone_dim, output_dim)

        self._feature_shape = self._infer_feature_shape()

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self._input_shape

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return self._feature_shape

    def _infer_feature_shape(self) -> tuple[int, int, int]:
        dummy = torch.zeros(1, *self.input_shape)

        # Set model to eval mode, to guard against BatchNorm erroring if
        # dummy hits [1, C, 1, 1]
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                features = self.feature_extractor(dummy)
        except RuntimeError as error:
            raise ValueError(
                "TorchvisionResNet could not run a dummy input through the "
                "feature extractor. Check input_shape and ResNet variant."
            ) from error
        finally:
            self.train(was_training)

        if features.ndim != 4:
            raise ValueError(
                "TorchvisionResNet feature_extractor must return a 4D tensor "
                f"[batch, channels, height, width], got shape {tuple(features.shape)}."
            )

        feature_shape = tuple(int(dim) for dim in features.shape[1:])

        if any(dim <= 0 for dim in feature_shape):
            raise ValueError(
                f"Inferred feature_shape must be positive, got {feature_shape}."
            )

        return feature_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "TorchvisionResNet expects a 4D tensor [batch, channels, height, width], "
                f"got shape {tuple(x.shape)}."
            )

        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected input shape per sample {self.input_shape}, "
                f"got {tuple(x.shape[1:])}."
            )

        features = self.feature_extractor(x)
        pooled = self.pool(features)
        flat = torch.flatten(pooled, start_dim=1)

        return self.projection(flat)