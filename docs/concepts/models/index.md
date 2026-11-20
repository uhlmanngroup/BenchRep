# Models & objectives

The model determines which representations and predictions you can compute from a batch. Its objectives determine which of those computations contribute to training. You can compare encoders with the same reconstruction loss, for example, or train a shared encoder on several tasks.

## What is a model family?

A **model family** defines how BenchRep builds or accepts a model, what the workflow expects from it, and how prediction interprets its outputs. Two VAEs can have different encoders and losses and still belong to the same family.

BenchRep has three families. Autoencoder and VAE are the <strong class="highlight">canonical families</strong>. Composite is currently the only non-canonical family.

## Choose a model family

Choose a **canonical family** when you want BenchRep to connect the encoder and decoder, supply the tensors each loss needs, and produce standard prediction outputs. This is a good fit for comparing encoders or varying the balance between reconstruction and regularization in a VAE.

Choose **composite** when the connections are part of your experiment. For example, you can pass two views of an image through the same encoder, send one representation to a classification head, and train both views with a contrastive loss.

| | Autoencoder | VAE | Composite |
| --- | --- | --- | --- |
| Construction | Encoder → decoder | Encoder → Gaussian variational head → decoder | User-declared computation graph |
| Standard representation | Encoder output | Posterior mean; sampled latent values also available | Outputs you declare and select |
| Architecture configuration | `encoder` and `decoder` | `encoder`, `decoder`, and `model.params.latent_dim` | Declarations, components, and assembly sections |
| Standard loss roles | Reconstruction | Reconstruction and regularization | Reconstruction, regularization, contrastive, classification, and regression |
| Typical use | Compare image encoders and reconstruction objectives | Learn a Gaussian latent representation; vary reconstruction–KL balance | Multi-input, multitask, supervised, or self-supervised image models |
| Whole-model runtime override | Supported | Supported | Currently unsupported |
| Training / prediction entrypoints | [`train_ae()`][benchrep.train_ae] / [`predict_ae()`][benchrep.predict_ae] | [`train_vae()`][benchrep.train_vae] / [`predict_vae()`][benchrep.predict_vae] | [`train_composite()`][benchrep.train_composite] / [`predict_composite()`][benchrep.predict_composite] |

All three families accept a [datamodule override](../workflows/overrides.md#supply-an-implementation).

## Why separate implementations?

The canonical classes implement their forward passes, loss calls, and prediction outputs directly. Those decisions give users sensible defaults and give contributors a concrete class to inspect or subclass. For example, its VAE class shows where the decoder receives a sampled latent during training and where prediction returns the posterior mean as the embedding.

A composite model runs the connections you declare. Its executor can run an equivalent VAE, a classifier, or a model with several branches. The canonical VAE follows its own implementation and has no dependency on that executor.

## Define what the model learns

BenchRep groups loss terms by role, such as reconstruction or classification, then adds their weighted values into one scalar training loss. Each term can have its own parameters and an optional weight schedule.

For an ordinary reconstruction loss, a canonical model supplies the reconstruction and its target. A composite configuration names those two tensors explicitly. All three families also support custom objectives that receive the full batch and all model outputs. See [Losses and objectives](losses.md) for the supported combinations and calling conventions.

Training objectives are separate from [evaluation pipelines](../evaluation/inputs.md), which measure properties of exported or external artifacts after inference.

## From configuration to model execution

Three parts of BenchRep turn that choice into a running model.

1. **Resolution** checks the selected components and their compatibility. For a composite model, it also resolves the graph and its execution order.
2. **Construction** instantiates the components, losses, and model, and supplies factories for the optimizer and learning-rate scheduler.
3. **Execution** belongs to the model's Lightning hooks: the forward computation, loss calculation, optimization setup, and prediction output.

The concrete implementations are [`Autoencoder`][benchrep.architecture.models.autoencoder.Autoencoder], [`VAE`][benchrep.architecture.models.vae.VAE], and [`CompositeModel`][benchrep.architecture.models.composite.CompositeModel]. The pages below explain how each stage works, including the difference between a resolved computation plan and the autograd graph PyTorch builds during execution. Their [implementation reference](../../reference/components/models.md) exposes constructors and methods, with links to the source.

??? note "A model class and a model family are different extension points"
    A canonical model subclass can change the implementation while retaining its family's runtime contract. A family defines a broader agreement between configuration, construction, workflow entrypoints, and prediction outputs. [`ModelFamilySpec`][benchrep.interfaces.model_families.ModelFamilySpec] identifies a family; [`CanonicalModelFamilySpec`][benchrep.interfaces.model_families.CanonicalModelFamilySpec] additionally describes its model base and batch/prediction contracts.

    Creating one of these specifications does not register a new family. The supported families also have explicit dispatch and validation in the [model builder](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/model_builder.py), schemas, resolvers, and workflows. Component registration extends the existing construction paths; introducing a new family requires coordinating those paths and its prediction/export contract.

Explore each part in more detail:

- **[Canonical models](canonical.md):** AE/VAE computation, dimensions, customization, and the concrete model implementations.
- **[Composite models](composite.md):** declarations, semantic roles, and how a configured graph becomes an executable model.
- **[Architecture component contracts](component-contracts.md):** how component interfaces make connections checkable.
- **[Losses and objectives](losses.md):** loss roles, tensor wiring, custom objectives, and the lifecycle from registry entry to weighted loss.
- **[Loss-weight schedules](loss-weight-schedules.md):** how objective contributions change with training position and how schedules are evaluated.
