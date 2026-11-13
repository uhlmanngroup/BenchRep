# Built-in datasets

These adapters return BenchRep sample dictionaries containing `x`, `label`, and `sample_id`. Their constructors load a torchvision dataset and convert images to tensors; configured transform pipelines are applied separately.

For custom datasets and field-routed transforms, see [Extension interfaces](../interfaces.md#datasets-and-transforms). The [dataset schemas](../schemas/shared.md#datasets) describe their YAML configuration.

::: benchrep.architecture.data.datasets.MNISTDataset
    options:
      heading_level: 2
      members:
        - __len__
        - __getitem__
        - validate_sample
      inherited_members: true

::: benchrep.architecture.data.datasets.CIFAR10Dataset
    options:
      heading_level: 2
      members:
        - __len__
        - __getitem__
        - validate_sample
      inherited_members: true

::: benchrep.architecture.data.datasets.STL10Dataset
    options:
      heading_level: 2
      members:
        - __len__
        - __getitem__
        - validate_sample
      inherited_members: true
