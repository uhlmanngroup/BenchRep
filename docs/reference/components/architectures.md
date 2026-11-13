# Encoders, decoders, and heads

Constructor parameters become the component's `params` in configuration. Forward methods describe the tensors passed at runtime. Shapes exclude the batch dimension in constructor settings and include it in forward inputs and outputs.

See the [catalogue](../components.md) for registered names and aliases, and [Extension interfaces](../interfaces.md#architecture-components) for registration contracts. Only BenchRep's own methods and properties are listed here.

## Encoders

::: benchrep.architecture.encoders.mlp.MLPEncoder
    options:
      heading_level: 3
      members:
        - input_shape
        - output_dim
        - forward

::: benchrep.architecture.encoders.conv2d.Conv2DEncoder
    options:
      heading_level: 3
      members:
        - input_shape
        - output_dim
        - feature_shape
        - forward

::: benchrep.architecture.encoders.torchvision_resnets.TorchvisionResNet
    options:
      heading_level: 3
      members:
        - input_shape
        - output_dim
        - feature_shape
        - forward

## Decoders

::: benchrep.architecture.decoders.mlp.MLPDecoder
    options:
      heading_level: 3
      members:
        - input_dim
        - output_shape
        - forward

::: benchrep.architecture.decoders.upsample_conv2d.UpsampleConv2DDecoder
    options:
      heading_level: 3
      members:
        - input_dim
        - output_shape
        - initial_shape
        - forward

## Heads

::: benchrep.architecture.heads.mlp.MLPHead
    options:
      heading_level: 3
      members:
        - forward

::: benchrep.architecture.heads.variational.GaussianVariationalHead
    options:
      heading_level: 3
      members:
        - forward
        - reparameterize
