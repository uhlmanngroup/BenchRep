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
            HEADS,
            MODELS,
            RECONSTRUCTION_LOSSES,
            REGULARIZATION_LOSSES,
            CLASSIFICATION_LOSSES,
            CONTRASTIVE_LOSSES,
            REGRESSION_LOSSES,
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

        from benchrep.architecture.composite_model_component_contracts import (
            ArchitectureComponent,
            ComponentPort,
            ComponentTensorResult,
            ComponentMappingResult,
        )

        from benchrep.architecture.losses.composite_model_contracts import (
            LossComponent,
            LossTensorPort,
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
        from benchrep.architecture.heads import (
            GaussianVariationalHead,
            MLPHead,
        )
        from benchrep.architecture.losses import (
            MSEReconstructionLoss,
            MAEReconstructionLoss,
            GaussianKLDivergenceLoss,
            TripletMarginContrastiveLoss,
            CrossEntropyClassificationLoss,
            MSERegressionLoss,
        )
        from benchrep.architecture.models import (
            Autoencoder,
            VAE,
            CompositeModel,
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
        from benchrep.evaluation.metrics import EvaluationMetric


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
        ENCODERS._register_builtin(
            "mlp",
            ArchitectureComponent(
                component=MLPEncoder,
                runtime_inputs=(
                    ComponentPort(
                        name="x",
                        supported_structures=("image",),
                    ),
                ),
                runtime_result=ComponentTensorResult(
                    supported_structures=("vector",),
                ),
            ),
            "dense",
            "fully_connected",
            "fc",
        )
        ENCODERS._register_builtin(
            "conv2d",
            ArchitectureComponent(
                component=Conv2DEncoder,
                runtime_inputs=(
                    ComponentPort(
                        name="x",
                        supported_structures=("image",),
                    ),
                ),
                runtime_result=ComponentTensorResult(
                    supported_structures=("vector",),
                ),
            ),
            "conv",
            "cnn",
            "convolutional",
        )
        ENCODERS._register_builtin(
            "torchvision_resnet",
            ArchitectureComponent(
                component=TorchvisionResNet,
                runtime_inputs=(
                    ComponentPort(
                        name="x",
                        supported_structures=("image",),
                    ),
                ),
                runtime_result=ComponentTensorResult(
                    supported_structures=("vector",),
                ),
            ),
            "torchvision_resnets",
            "resnet",
            "resnets",
            "tv_resnet",
            "tv_resnets",
        )

        DECODERS._register_builtin(
            "mlp",
            ArchitectureComponent(
                component=MLPDecoder,
                runtime_inputs=(
                    ComponentPort(
                        name="z",
                        supported_structures=("vector",),
                    ),
                ),
                runtime_result=ComponentTensorResult(
                    supported_structures=("image",),
                ),
            ),
            "dense",
            "fully_connected",
            "fc",
        )
        DECODERS._register_builtin(
            "upsample_conv2d",
            ArchitectureComponent(
                component=UpsampleConv2DDecoder,
                runtime_inputs=(
                    ComponentPort(
                        name="z",
                        supported_structures=("vector",),
                    ),
                ),
                runtime_result=ComponentTensorResult(
                    supported_structures=("image",),
                ),
            ),
            "upsampleconv2d",
            "upsample_conv",
            "upconv",
            "resize_conv",
        )

        HEADS._register_builtin(
            "mlp",
            ArchitectureComponent(
                component=MLPHead,
                runtime_inputs=(
                    ComponentPort(
                        name="x",
                        supported_structures=("vector",),
                    ),
                ),
                runtime_result=ComponentTensorResult(
                    supported_structures=("scalar", "vector"),
                ),
            ),
            "dense",
            "fully_connected",
            "fc",
        )

        HEADS._register_builtin(
            "gaussian_variational",
            ArchitectureComponent(
                component=GaussianVariationalHead,
                runtime_inputs=(
                    ComponentPort(
                        name="x",
                        supported_structures=("vector",),
                    ),
                ),
                runtime_result=ComponentMappingResult(
                    outputs=(
                        ComponentPort(
                            name="z_sample",
                            supported_structures=("vector",),
                        ),
                        ComponentPort(
                            name="z_mu",
                            supported_structures=("vector",),
                        ),
                        ComponentPort(
                            name="z_logvar",
                            supported_structures=("vector",),
                        ),
                    ),
                ),
            ),
            "variational",
            "gaussian",
        )

        MODELS._register_builtin("autoencoder", Autoencoder, "ae")
        MODELS._register_builtin(
            "vae",
            VAE,
            "variational_autoencoder",
            "variational_ae",
            "gaussian_vae",
        )
        MODELS._register_builtin(
            "composite",
            CompositeModel,
            "composite_model",
        )

        RECONSTRUCTION_LOSSES._register_builtin(
            "mse",
            LossComponent(
                component=MSEReconstructionLoss,
                runtime_inputs=(
                    LossTensorPort(
                        name="reconstruction",
                        supported_roles=("reconstruction_image",),
                    ),
                    LossTensorPort(
                        name="target",
                        supported_roles=(
                            "sample_image",
                            "positive_image",
                            "negative_image",
                        ),
                    ),
                ),
            ),
            "l2",
            "mean_squared_error",
        )
        RECONSTRUCTION_LOSSES._register_builtin(
            "mae",
            LossComponent(
                component=MAEReconstructionLoss,
                runtime_inputs=(
                    LossTensorPort(
                        name="reconstruction",
                        supported_roles=("reconstruction_image",),
                    ),
                    LossTensorPort(
                        name="target",
                        supported_roles=(
                            "sample_image",
                            "positive_image",
                            "negative_image",
                        ),
                    ),
                ),
            ),
            "l1",
        )

        REGULARIZATION_LOSSES._register_builtin(
            "gaussian_kl",
            LossComponent(
                component=GaussianKLDivergenceLoss,
                runtime_inputs=(
                    LossTensorPort(
                        name="z_mu",
                        supported_roles=(
                            "embedding_vector",
                            "continuous_auxiliary_vector",
                        ),
                    ),
                    LossTensorPort(
                        name="z_logvar",
                        supported_roles=("continuous_auxiliary_vector",),
                    ),
                ),
            ),
            "kl",
            "kld",
            "kldiv",
            "kl_div",
            "gaussian_kld",
            "gaussian_kldiv",
            "gaussian_kl_div",
        )

        CONTRASTIVE_LOSSES._register_builtin(
            "triplet_margin",
            LossComponent(
                component=TripletMarginContrastiveLoss,
                runtime_inputs=(
                    LossTensorPort(
                        name="anchor",
                        supported_roles=(
                            "embedding_vector",
                            "projection_vector",
                        ),
                    ),
                    LossTensorPort(
                        name="positive",
                        supported_roles=(
                            "embedding_vector",
                            "projection_vector",
                        ),
                    ),
                    LossTensorPort(
                        name="negative",
                        supported_roles=(
                            "embedding_vector",
                            "projection_vector",
                        ),
                    ),
                ),
            ),
            "triplet",
            "triplet_margin_loss",
        )

        CLASSIFICATION_LOSSES._register_builtin(
            "cross_entropy",
            LossComponent(
                component=CrossEntropyClassificationLoss,
                runtime_inputs=(
                    LossTensorPort(
                        name="prediction",
                        supported_roles=(
                            "categorical_prediction_vector",
                        ),
                    ),
                    LossTensorPort(
                        name="target",
                        supported_roles=(
                            "categorical_prediction_target_scalar",
                        ),
                    ),
                ),
            ),
            "categorical_cross_entropy",
            "ce",
        )

        REGRESSION_LOSSES._register_builtin(
            "mse",
            LossComponent(
                component=MSERegressionLoss,
                runtime_inputs=(
                    LossTensorPort(
                        name="prediction",
                        supported_roles=(
                            "continuous_prediction_scalar",
                            "continuous_prediction_vector",
                        ),
                    ),
                    LossTensorPort(
                        name="target",
                        supported_roles=(
                            "continuous_prediction_target_scalar",
                            "continuous_prediction_target_vector",
                        ),
                    ),
                ),
            ),
            "l2",
            "mean_squared_error",
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
            EvaluationMetric(
                fn=silhouette_score,
                result_kind="scalar",
            ),
            "silhouette_score",
        )
        EVAL_INTERNAL_CLUSTERING_METRICS._register_builtin(
            "calinski_harabasz",
            EvaluationMetric(
                fn=calinski_harabasz_score,
                result_kind="scalar",
            ),
            "calinski_harabasz_score",
            "ch",
            "ch_score",
        )
        EVAL_INTERNAL_CLUSTERING_METRICS._register_builtin(
            "davies_bouldin",
            EvaluationMetric(
                fn=davies_bouldin_score,
                result_kind="scalar",
            ),
            "davies_bouldin_score",
            "db",
            "db_score",
        )

        # External clustering metrics
        EVAL_EXTERNAL_CLUSTERING_METRICS._register_builtin(
            "adjusted_mutual_info",
            EvaluationMetric(
                fn=adjusted_mutual_info_score,
                result_kind="scalar",
            ),
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
            EvaluationMetric(
                fn=adjusted_rand_score,
                result_kind="scalar",
            ),
            "adjusted_rand_score",
            "adjusted_rand",
            "adj_rand_index",
            "adj_rand_score",
            "ari",
            "ari_score",
        )
        EVAL_EXTERNAL_CLUSTERING_METRICS._register_builtin(
            "homogeneity",
            EvaluationMetric(
                fn=homogeneity_score,
                result_kind="scalar",
            ),
            "homogeneity_score",
        )

        # Embedding metrics
        EVAL_EMBEDDING_METRICS._register_builtin(
            "mean",
            EvaluationMetric(
                fn=dimensionwise_mean,
                result_kind="vector",
                vector_axis="embedding_dimension",
            ),
            "dimensionwise_mean",
        )
        EVAL_EMBEDDING_METRICS._register_builtin(
            "median",
            EvaluationMetric(
                fn=dimensionwise_median,
                result_kind="vector",
                vector_axis="embedding_dimension",
            ),
            "dimensionwise_median",
        )
        EVAL_EMBEDDING_METRICS._register_builtin(
            "standard_deviation",
            EvaluationMetric(
                fn=dimensionwise_standard_deviation,
                result_kind="vector",
                vector_axis="embedding_dimension",
            ),
            "dimensionwise_standard_deviation",
            "std",
        )
        EVAL_EMBEDDING_METRICS._register_builtin(
            "minimum",
            EvaluationMetric(
                fn=dimensionwise_minimum,
                result_kind="vector",
                vector_axis="embedding_dimension",
            ),
            "dimensionwise_minimum",
            "min",
        )
        EVAL_EMBEDDING_METRICS._register_builtin(
            "maximum",
            EvaluationMetric(
                fn=dimensionwise_maximum,
                result_kind="vector",
                vector_axis="embedding_dimension",
            ),
            "dimensionwise_maximum",
            "max",
        )
        EVAL_EMBEDDING_METRICS._register_builtin(
            "quantiles",
            EvaluationMetric(
                fn=dimensionwise_quantiles,
                result_kind="vector_mapping",
                vector_axis="embedding_dimension",
            ),
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
            EvaluationMetric(
                fn=mean_absolute_error,
                result_kind="scalar",
            ),
            "mean_absolute_error",
            "mean_abs_error",
        )
        EVAL_RECONSTRUCTION_METRICS._register_builtin(
            "mse",
            EvaluationMetric(
                fn=mean_squared_error,
                result_kind="scalar",
            ),
            "mean_squared_error",
            "mean_sq_error",
        )
        EVAL_RECONSTRUCTION_METRICS._register_builtin(
            "rmse",
            EvaluationMetric(
                fn=root_mean_squared_error,
                result_kind="scalar",
            ),
            "root_mean_squared_error",
            "root_mean_sq_error",
        )
        EVAL_RECONSTRUCTION_METRICS._register_builtin(
            "max_absolute_error",
            EvaluationMetric(
                fn=max_absolute_error,
                result_kind="scalar",
            ),
            "max_abs_error",
        )

        _BUILTINS_REGISTERED = True

    finally:
        _BUILTINS_REGISTERING = False