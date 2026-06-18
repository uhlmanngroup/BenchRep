_BUILTINS_REGISTERED = False
_BUILTINS_REGISTERING = False


def register_builtins() -> None:
    """Register BenchRep's built-in components exactly once.

    Registries are defined separately in ``benchrep.assembly.registry`` so that
    implementation modules can safely import registry objects for lookup without
    creating import cycles. This function is the single place where BenchRep's
    built-in datasets, transforms, architectures, losses, optimizers, loggers,
    evaluation methods, and metrics are attached to those registries.

    Imports are intentionally kept inside this function. This keeps importing
    the registration module cheap and avoids pulling in heavy optional/runtime
    dependencies, such as torch, torchvision, Lightning loggers, scikit-learn,
    model classes, and evaluation modules, before registration is actually
    needed.

    The function is idempotent: after built-ins have been registered once,
    later calls return immediately. The ``_BUILTINS_REGISTERING`` guard detects
    recursive registration attempts, which usually mean that a registry lookup
    happened while built-in registration was still in progress. In practice,
    that points to an import cycle or import-time registry access that should be
    fixed rather than silently ignored.
    """
    global _BUILTINS_REGISTERED, _BUILTINS_REGISTERING

    if _BUILTINS_REGISTERED:
        return

    if _BUILTINS_REGISTERING:
        raise RuntimeError(
            "BenchRep built-in registration is already in progress. "
            "A registry lookup occurred while builtins were still being "
            "registered, which usually indicates an import cycle or import-time "
            "registry access."
        )

    _BUILTINS_REGISTERING = True

    try:
        import torch
        from torchvision import transforms

        from lightning.pytorch.loggers import (
            CSVLogger,
            MLFlowLogger,
            TensorBoardLogger,
            WandbLogger,
        )

        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
            adjusted_mutual_info_score,
            adjusted_rand_score,
            homogeneity_score,
        )

        from benchrep.assembly.registry import (
            DATASETS,
            TRANSFORMS,
            ENCODERS,
            DECODERS,
            MODELS,
            RECONSTRUCTION_LOSSES,
            REGULARIZATION_LOSSES,
            OPTIMIZERS,
            LOGGERS,
            EVAL_REDUCTIONS,
            EVAL_CLUSTERING_METHODS,
            EVAL_INTERNAL_CLUSTERING_METRICS,
            EVAL_EXTERNAL_CLUSTERING_METRICS,
            EVAL_EMBEDDING_METRICS,
            EVAL_RECONSTRUCTION_METRICS,
        )

        from benchrep.architecture.data import MNISTDataset
        from benchrep.architecture.decoders import MLPDecoder
        from benchrep.architecture.encoders import MLPEncoder, Conv2DEncoder
        from benchrep.architecture.losses import (
            MSEReconstructionLoss,
            MAEReconstructionLoss,
            GaussianKLDivergenceLoss,
        )
        from benchrep.architecture.models import (
            Autoencoder,
            VAE,
        )
        from benchrep.evaluation.embeddings.clustering import run_kmeans, run_leiden
        from benchrep.evaluation.embeddings.reductions import run_pca, run_tsne, run_umap
        from benchrep.evaluation.reconstructions.reconstruction_metrics import (
            mean_absolute_error,
            mean_squared_error,
            root_mean_squared_error,
            max_absolute_error,
        )

        # --- Data ---
        DATASETS.register("mnist", MNISTDataset)

        TRANSFORMS.register("to_tensor", transforms.ToTensor)

        # --- Architecture and training ---
        ENCODERS.register("mlp", MLPEncoder, "dense", "fully_connected", "fc")
        ENCODERS.register("conv2d", Conv2DEncoder, "conv", "cnn", "convolutional")

        DECODERS.register("mlp", MLPDecoder, "dense", "fully_connected", "fc")

        MODELS.register("autoencoder", Autoencoder, "ae")
        MODELS.register(
            "vae",
            VAE,
            "variational_autoencoder",
            "variational_ae",
            "gaussian_vae",
        )

        RECONSTRUCTION_LOSSES.register("mse", MSEReconstructionLoss, "l2")
        RECONSTRUCTION_LOSSES.register("mae", MAEReconstructionLoss, "l1")

        REGULARIZATION_LOSSES.register(
            "gaussian_kl",
            GaussianKLDivergenceLoss,
            "kl",
            "kld",
            "kldiv",
            "kl_div",
            "gaussian_kld",
            "gaussian_kldiv",
            "gaussian_kl_div",
        )

        OPTIMIZERS.register("adam", torch.optim.Adam)
        OPTIMIZERS.register("adamw", torch.optim.AdamW)
        OPTIMIZERS.register("sgd", torch.optim.SGD)

        LOGGERS.register("csv", CSVLogger, "csvlogger")
        LOGGERS.register("wandb", WandbLogger, "wandblogger")
        LOGGERS.register(
            "tensorboard",
            TensorBoardLogger,
            "tensorboardlogger",
            "tb",
            "tblogger",
        )
        LOGGERS.register("mlflow", MLFlowLogger, "mlflowlogger")

        # --- Evaluation ---
        # Reductions
        EVAL_REDUCTIONS.register(
            "pca",
            run_pca,
            "principal_component_analysis",
            "principal_components",
        )
        EVAL_REDUCTIONS.register("umap", run_umap)
        EVAL_REDUCTIONS.register("tsne", run_tsne, "t_sne")

        # Clustering
        EVAL_CLUSTERING_METHODS.register("kmeans", run_kmeans, "k_means")
        EVAL_CLUSTERING_METHODS.register("leiden", run_leiden)

        # Internal clustering metrics
        EVAL_INTERNAL_CLUSTERING_METRICS.register(
            "silhouette",
            silhouette_score,
            "silhouette_score",
        )
        EVAL_INTERNAL_CLUSTERING_METRICS.register(
            "calinski_harabasz",
            calinski_harabasz_score,
            "calinski_harabasz_score",
            "ch",
            "ch_score",
        )
        EVAL_INTERNAL_CLUSTERING_METRICS.register(
            "davies_bouldin",
            davies_bouldin_score,
            "davies_bouldin_score",
            "db",
            "db_score",
        )

        # External clustering metrics
        EVAL_EXTERNAL_CLUSTERING_METRICS.register(
            "adjusted_mutual_info",
            adjusted_mutual_info_score,
            "adjusted_mutual_info_score",
            "adjusted_mutual_information",
            "adjusted_mutual_information_score",
            "adj_mutual_info",
            "adj_mutual_info_score",
            "ami",
            "ami_score",
        )
        EVAL_EXTERNAL_CLUSTERING_METRICS.register(
            "adjusted_rand_index",
            adjusted_rand_score,
            "adjusted_rand_score",
            "adjusted_rand",
            "adj_rand_index",
            "adj_rand_score",
            "ari",
            "ari_score",
        )
        EVAL_EXTERNAL_CLUSTERING_METRICS.register(
            "homogeneity",
            homogeneity_score,
            "homogeneity_score",
        )

        # Reconstruction metrics
        EVAL_RECONSTRUCTION_METRICS.register(
            "mae",
            mean_absolute_error,
            "mean_absolute_error",
            "mean_abs_error",
        )
        EVAL_RECONSTRUCTION_METRICS.register(
            "mse",
            mean_squared_error,
            "mean_squared_error",
            "mean_sq_error",
        )
        EVAL_RECONSTRUCTION_METRICS.register(
            "rmse",
            root_mean_squared_error,
            "root_mean_squared_error",
            "root_mean_sq_error",
        )
        EVAL_RECONSTRUCTION_METRICS.register(
            "max_absolute_error",
            max_absolute_error,
            "max_abs_error",
        )

        _BUILTINS_REGISTERED = True

    finally:
        _BUILTINS_REGISTERING = False