_BUILTINS_REGISTERED = False
_BUILTINS_REGISTERING = False

# noinspection PyProtectedMember
def register_builtins() -> None:
    """Register BenchRep's built-in components exactly once.

    Registries are defined separately in ``benchrep.assembly.registries.core`` so
    that implementation modules can safely import registry objects for lookup
    without creating import cycles. This function is the single place where
    BenchRep's built-in datasets, transforms, architectures, losses, optimizers,
    loggers, evaluation methods, and metrics are attached to those registries.

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
        from torchvision.transforms import v2

        from lightning.pytorch.loggers import (
            CSVLogger,
            MLFlowLogger,
            TensorBoardLogger,
            WandbLogger,
        )

        from lightning.pytorch.callbacks import (
            DeviceStatsMonitor,
            GradientAccumulationScheduler,
            LearningRateMonitor,
            ModelSummary,
            RichModelSummary,
            RichProgressBar,
            Timer,
            TQDMProgressBar,
            WeightAveraging,
        )

        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
            adjusted_mutual_info_score,
            adjusted_rand_score,
            homogeneity_score,
        )

        from benchrep.assembly.registries.core import (
            DATASETS,
            TRANSFORMS,
            ENCODERS,
            DECODERS,
            MODELS,
            RECONSTRUCTION_LOSSES,
            REGULARIZATION_LOSSES,
            OPTIMIZERS,
            LOGGERS,
            CALLBACKS,
            EVAL_REDUCTIONS,
            EVAL_CLUSTERING_METHODS,
            EVAL_INTERNAL_CLUSTERING_METRICS,
            EVAL_EXTERNAL_CLUSTERING_METRICS,
            EVAL_EMBEDDING_METRICS,
            EVAL_PREDICTABILITY_PROBES,
            EVAL_RECONSTRUCTION_METRICS,
        )

        from benchrep.architecture.data import (
            MNISTDataset,
            CIFAR10Dataset,
            STL10Dataset,
        )
        from benchrep.architecture.data.transforms import (
            create_to_dtype_transform,
        )
        from benchrep.architecture.decoders import MLPDecoder, UpsampleConv2DDecoder
        from benchrep.architecture.encoders import (
            MLPEncoder,
            Conv2DEncoder,
            TorchvisionResNet,
        )
        from benchrep.architecture.losses import (
            MSEReconstructionLoss,
            MAEReconstructionLoss,
            GaussianKLDivergenceLoss,
        )
        from benchrep.architecture.models import (
            Autoencoder,
            VAE,
        )
        from benchrep.evaluation.embeddings.clustering import run_kmeans, run_leiden, run_hdbscan
        from benchrep.evaluation.embeddings.reductions import run_pca, run_tsne, run_umap
        from benchrep.evaluation.embeddings.embedding_metrics import (
            dimensionwise_mean,
            dimensionwise_median,
            dimensionwise_standard_deviation,
            dimensionwise_minimum,
            dimensionwise_maximum,
            dimensionwise_quantiles,
        )
        from benchrep.evaluation.embeddings.predictability_probes import (
            build_dummy_predictability_probe,
            build_linear_predictability_probe,
            build_knn_predictability_probe,
            build_random_forest_predictability_probe,
            build_xgboost_predictability_probe,
            build_svm_rbf_predictability_probe,
        )
        from benchrep.evaluation.reconstructions.reconstruction_metrics import (
            mean_absolute_error,
            mean_squared_error,
            root_mean_squared_error,
            max_absolute_error,
        )


        # --- Data ---
        DATASETS._register_builtin("mnist", MNISTDataset)
        DATASETS._register_builtin("cifar10", CIFAR10Dataset, "cifar_10")
        DATASETS._register_builtin("stl10", STL10Dataset, "stl_10")

        TRANSFORMS._register_builtin(
            "to_dtype",
            create_to_dtype_transform,
            "todtype",
            "convert_dtype",
            "convert_image_dtype",
        )
        TRANSFORMS._register_builtin(
            "normalize",
            v2.Normalize,
            "normalise",
            "normalization",
            "normalisation",
        )
        TRANSFORMS._register_builtin(
            "resize",
            v2.Resize,
            "resizing",
        )
        TRANSFORMS._register_builtin(
            "random_horizontal_flip",
            v2.RandomHorizontalFlip,
            "randomhorizontalflip",
            "random_hflip",
            "rand_hflip",
        )
        TRANSFORMS._register_builtin(
            "random_vertical_flip",
            v2.RandomVerticalFlip,
            "randomverticalflip",
            "random_vflip",
            "rand_vflip",
        )
        TRANSFORMS._register_builtin(
            "random_rotation",
            v2.RandomRotation,
            "randomrotation",
            "random_rotate",
            "rand_rotation",
        )

        # --- Architecture and training ---
        ENCODERS._register_builtin("mlp", MLPEncoder, "dense", "fully_connected", "fc")
        ENCODERS._register_builtin("conv2d", Conv2DEncoder, "conv", "cnn", "convolutional")
        ENCODERS._register_builtin(
            "torchvision_resnet",
            TorchvisionResNet,
            "torchvision_resnets",
            "resnet",
            "resnets",
            "tv_resnet",
            "tv_resnets",
        )

        DECODERS._register_builtin("mlp", MLPDecoder, "dense", "fully_connected", "fc")
        DECODERS._register_builtin(
            "upsample_conv2d",
            UpsampleConv2DDecoder,
            "upsampleconv2d",
            "upsample_conv",
            "upconv",
            "resize_conv",
        )

        MODELS._register_builtin("autoencoder", Autoencoder, "ae")
        MODELS._register_builtin(
            "vae",
            VAE,
            "variational_autoencoder",
            "variational_ae",
            "gaussian_vae",
        )

        RECONSTRUCTION_LOSSES._register_builtin("mse", MSEReconstructionLoss, "l2")
        RECONSTRUCTION_LOSSES._register_builtin("mae", MAEReconstructionLoss, "l1")

        REGULARIZATION_LOSSES._register_builtin(
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

        OPTIMIZERS._register_builtin("adam", torch.optim.Adam)
        OPTIMIZERS._register_builtin("adamw", torch.optim.AdamW)
        OPTIMIZERS._register_builtin("sgd", torch.optim.SGD)

        LOGGERS._register_builtin("csv", CSVLogger, "csvlogger")
        LOGGERS._register_builtin("wandb", WandbLogger, "wandblogger")
        LOGGERS._register_builtin(
            "tensorboard",
            TensorBoardLogger,
            "tensorboardlogger",
            "tb",
            "tblogger",
        )
        LOGGERS._register_builtin("mlflow", MLFlowLogger, "mlflowlogger")

        CALLBACKS._register_builtin(
            "device_stats_monitor",
            DeviceStatsMonitor,
            "devicestatsmonitor",
        )
        CALLBACKS._register_builtin(
            "gradient_accumulation_scheduler",
            GradientAccumulationScheduler,
            "gradientaccumulationscheduler",
        )
        CALLBACKS._register_builtin(
            "learning_rate_monitor",
            LearningRateMonitor,
            "learningratemonitor",
            "lr_monitor",
        )
        CALLBACKS._register_builtin(
            "model_summary",
            ModelSummary,
            "modelsummary",
        )
        CALLBACKS._register_builtin(
            "rich_model_summary",
            RichModelSummary,
            "richmodelsummary",
        )
        CALLBACKS._register_builtin(
            "rich_progress_bar",
            RichProgressBar,
            "richprogressbar",
        )
        CALLBACKS._register_builtin(
            "timer",
            Timer,
        )
        CALLBACKS._register_builtin(
            "tqdm_progress_bar",
            TQDMProgressBar,
            "tqdmprogressbar",
        )
        CALLBACKS._register_builtin(
            "weight_averaging",
            WeightAveraging,
            "weightaveraging",
        )

        # --- Evaluation ---
        # Reductions
        EVAL_REDUCTIONS._register_builtin(
            "pca",
            run_pca,
            "principal_component_analysis",
            "principal_components",
        )
        EVAL_REDUCTIONS._register_builtin("umap", run_umap)
        EVAL_REDUCTIONS._register_builtin("tsne", run_tsne, "t_sne")

        # Clustering
        EVAL_CLUSTERING_METHODS._register_builtin("kmeans", run_kmeans, "k_means")
        EVAL_CLUSTERING_METHODS._register_builtin("leiden", run_leiden)
        EVAL_CLUSTERING_METHODS._register_builtin("hdbscan", run_hdbscan)

        # Internal clustering metrics
        EVAL_INTERNAL_CLUSTERING_METRICS._register_builtin(
            "silhouette",
            silhouette_score,
            "silhouette_score",
        )
        EVAL_INTERNAL_CLUSTERING_METRICS._register_builtin(
            "calinski_harabasz",
            calinski_harabasz_score,
            "calinski_harabasz_score",
            "ch",
            "ch_score",
        )
        EVAL_INTERNAL_CLUSTERING_METRICS._register_builtin(
            "davies_bouldin",
            davies_bouldin_score,
            "davies_bouldin_score",
            "db",
            "db_score",
        )

        # External clustering metrics
        EVAL_EXTERNAL_CLUSTERING_METRICS._register_builtin(
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
        EVAL_EXTERNAL_CLUSTERING_METRICS._register_builtin(
            "adjusted_rand_index",
            adjusted_rand_score,
            "adjusted_rand_score",
            "adjusted_rand",
            "adj_rand_index",
            "adj_rand_score",
            "ari",
            "ari_score",
        )
        EVAL_EXTERNAL_CLUSTERING_METRICS._register_builtin(
            "homogeneity",
            homogeneity_score,
            "homogeneity_score",
        )

        # Embedding metrics
        EVAL_EMBEDDING_METRICS._register_builtin(
            "mean",
            dimensionwise_mean,
            "dimensionwise_mean",
        )
        EVAL_EMBEDDING_METRICS._register_builtin(
            "median",
            dimensionwise_median,
            "dimensionwise_median",
        )
        EVAL_EMBEDDING_METRICS._register_builtin(
            "standard_deviation",
            dimensionwise_standard_deviation,
            "dimensionwise_standard_deviation",
            "std",
        )
        EVAL_EMBEDDING_METRICS._register_builtin(
            "minimum",
            dimensionwise_minimum,
            "dimensionwise_minimum",
            "min",
        )
        EVAL_EMBEDDING_METRICS._register_builtin(
            "maximum",
            dimensionwise_maximum,
            "dimensionwise_maximum",
            "max",
        )
        EVAL_EMBEDDING_METRICS._register_builtin(
            "quantiles",
            dimensionwise_quantiles,
            "dimensionwise_quantiles",
            "quantile",
        )

        # Predictability probes
        EVAL_PREDICTABILITY_PROBES._register_builtin(
            "dummy",
            build_dummy_predictability_probe,
            "baseline",
        )
        EVAL_PREDICTABILITY_PROBES._register_builtin(
            "linear",
            build_linear_predictability_probe,
            "linear_probe",
        )
        EVAL_PREDICTABILITY_PROBES._register_builtin(
            "knn",
            build_knn_predictability_probe,
            "k_nearest_neighbors",
        )
        EVAL_PREDICTABILITY_PROBES._register_builtin(
            "random_forest",
            build_random_forest_predictability_probe,
            "rf",
            "forest",
        )
        EVAL_PREDICTABILITY_PROBES._register_builtin(
            "xgboost",
            build_xgboost_predictability_probe,
            "xgb",
        )
        EVAL_PREDICTABILITY_PROBES._register_builtin(
            "svm_rbf",
            build_svm_rbf_predictability_probe,
            "support_vector_machine_rbf",
            "svm_radial_basis_function",
            "support_vector_machine_radial_basis_function",
            "rbf_svm",
        )

        # Reconstruction metrics
        EVAL_RECONSTRUCTION_METRICS._register_builtin(
            "mae",
            mean_absolute_error,
            "mean_absolute_error",
            "mean_abs_error",
        )
        EVAL_RECONSTRUCTION_METRICS._register_builtin(
            "mse",
            mean_squared_error,
            "mean_squared_error",
            "mean_sq_error",
        )
        EVAL_RECONSTRUCTION_METRICS._register_builtin(
            "rmse",
            root_mean_squared_error,
            "root_mean_squared_error",
            "root_mean_sq_error",
        )
        EVAL_RECONSTRUCTION_METRICS._register_builtin(
            "max_absolute_error",
            max_absolute_error,
            "max_abs_error",
        )


        _BUILTINS_REGISTERED = True

    finally:
        _BUILTINS_REGISTERING = False