from pathlib import Path

from benchrep.assembly.schemas.training_config_schema import (
    CheckpointConfig,
    CustomDatasetConfig,
    DataModuleConfig,
    DecoderConfig,
    EncoderConfig,
    InspectionConfig,
    LoggerConfig,
    LossTermConfig,
    MNISTDatasetConfig,
    MNISTDatasetParams,
    ModelConfig,
    OptimizerConfig,
    ReproducibilityConfig,
    RunConfig,
    TorchviewConfig,
    TrainerConfig,
    TrainingConfig,
    TransformConfig,
)
from benchrep.assembly.schemas.prediction_config_schema import (
    PredictionConfig,
    PredictionDataConfig,
    PredictionEmbeddingsExportConfig,
    PredictionExportConfig,
    PredictionInferenceConfig,
    PredictionReconstructionsExportConfig,
    PredictionSourceConfig,
)

from benchrep.assembly.schemas.evaluation_config_schema import (
    DummyProbeConfig,
    ErrorMapParams,
    EvalMetricGroupConfig,
    EvaluationCVTuningConfig,
    EvaluationClusteringConfig,
    EvaluationClusteringMetricsConfig,
    EvaluationConfig,
    EvaluationCrossValidationConfig,
    EvaluationMetricsConfig,
    EvaluationPlotsConfig,
    EvaluationPredictabilityConfig,
    EvaluationPredictabilityParamsConfig,
    EvaluationReconstructionConfig,
    EvaluationReductionsConfig,
    EvaluationRunConfig,
    EvaluationSourceConfig,
    ExternalClusteringMetricConfig,
    InternalClusteringMetricConfig,
    KMeansConfig,
    KMeansParams,
    LeidenConfig,
    LeidenParams,
    PCAConfig,
    PCAParams,
    PlotParams,
    ReconstructionGridConfig,
    ReconstructionMetricConfig,
    RidgeProbeConfig,
    TSNEConfig,
    TSNEParams,
    UMAPConfig,
    UMAPParams,
)


# -------------------------
# Training config components
# -------------------------
def make_training_run_config() -> RunConfig:
    return RunConfig(
        output_root=Path("overridden_outputs"),
        project_name="overridden_project",
    )


def make_training_reproducibility_config() -> ReproducibilityConfig:
    return ReproducibilityConfig(
        seed=999,
        seed_workers=False,
        float32_matmul_precision="medium",
    )


def make_training_model_config() -> ModelConfig:
    return ModelConfig(
        name="ae",
        params={},
    )


def make_training_encoder_config() -> EncoderConfig:
    return EncoderConfig(
        name="mlp",
        params={
            "input_shape": [1, 28, 28],
            "output_dim": 12,
            "hidden_dims": [48, 24],
            "activation": None,
            "normalization": None,
            "dropout": 0.1,
        },
    )


def make_training_decoder_config() -> DecoderConfig:
    return DecoderConfig(
        name="mlp",
        params={
            "output_shape": [1, 28, 28],
            "hidden_dims": [24, 48],
            "activation": None,
            "normalization": None,
            "output_activation": None,
            "dropout": 0.1,
        },
    )


def make_training_losses_config() -> dict[
    str,
    dict[str, LossTermConfig],
]:
    return {
        "reconstruction": {
            "mse": LossTermConfig(
                weight=0.75,
                params={"reduction": "mean"},
            ),
            "mae": LossTermConfig(
                weight=0.25,
                params={"reduction": "mean"},
            ),
        },
    }


def make_training_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        name="adamw",
        params={
            "lr": 0.0005,
            "weight_decay": 0.01,
        },
    )


def make_training_custom_dataset_config() -> CustomDatasetConfig:
    return CustomDatasetConfig(
        name="tiny_synthetic",
        params={
            "n_samples": 24,
            "image_shape": [1, 28, 28],
            "n_classes": 3,
            "n_groups": 2,
            "signal_strength": 0.7,
            "noise_std": 0.1,
            "seed": 999,
        },
    )


def make_training_mnist_dataset_config() -> MNISTDatasetConfig:
    return MNISTDatasetConfig(
        params=MNISTDatasetParams(
            root=Path("overridden_data/mnist"),
            split="test",
            download=False,
            transform=TransformConfig(
                name="to_tensor",
                params={},
            ),
        ),
    )


def make_training_datamodule_config() -> DataModuleConfig:
    return DataModuleConfig(
        batch_size=16,
        val_fraction=0.2,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )


def make_training_trainer_config() -> TrainerConfig:
    return TrainerConfig(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=2,
        deterministic=True,
        benchmark=False,
        precision="32-true",
    )


def make_training_logger_config() -> LoggerConfig:
    return LoggerConfig(
        name="csv",
        params={
            "save_dir": "overridden_logs",
            "name": "overridden_csv",
        },
        wandb_api_key_path=None,
    )


def make_training_checkpoint_config() -> CheckpointConfig:
    return CheckpointConfig(
        monitor="val/loss",
        mode="min",
        save_top_k=2,
        save_last=True,
        filename="overridden-{epoch:02d}-{step}",
    )


def make_training_inspection_config() -> InspectionConfig:
    return InspectionConfig(
        torchview=TorchviewConfig(
            enabled=True,
            expand_nested=False,
            depth=4,
        ),
    )


# -------------------------
# Complete training config
# -------------------------
def make_training_config() -> TrainingConfig:
    return TrainingConfig(
        run=make_training_run_config(),
        reproducibility=make_training_reproducibility_config(),
        model=make_training_model_config(),
        encoder=make_training_encoder_config(),
        decoder=make_training_decoder_config(),
        losses=make_training_losses_config(),
        optimizer=make_training_optimizer_config(),
        dataset=make_training_custom_dataset_config(),
        datamodule=make_training_datamodule_config(),
        trainer=make_training_trainer_config(),
        logger=make_training_logger_config(),
        checkpointing=make_training_checkpoint_config(),
        inspection=make_training_inspection_config(),
    )


# -------------------------
# Prediction config components
# -------------------------
def make_prediction_source_config() -> PredictionSourceConfig:
    return PredictionSourceConfig(
        training_manifest_path=Path(
            "overridden_training_manifest.yaml"
        ),
        checkpoint="last",
    )


def make_prediction_dataset_config() -> CustomDatasetConfig:
    return CustomDatasetConfig(
        name="tiny_synthetic",
        params={
            "n_samples": 24,
            "image_shape": [1, 28, 28],
            "n_classes": 3,
            "n_groups": 2,
            "signal_strength": 0.7,
            "noise_std": 0.1,
            "seed": 999,
        },
    )


def make_prediction_data_config() -> PredictionDataConfig:
    return PredictionDataConfig(
        batch_size=6,
        num_workers=2,
        max_batches=3,
    )


def make_prediction_inference_config() -> PredictionInferenceConfig:
    return PredictionInferenceConfig(
        seed=999,
        seed_workers=False,
        deterministic="warn",
        float32_matmul_precision="medium",
    )


def make_prediction_exports_config() -> PredictionExportConfig:
    return PredictionExportConfig(
        mode="custom",
        embeddings=PredictionEmbeddingsExportConfig(
            enabled=True,
            keys=["embedding"],
            primary_key="embedding",
        ),
        reconstructions=PredictionReconstructionsExportConfig(
            enabled=True,
            n_examples=6,
            selection="first",
            stratify_by=None,
            seed=999,
            include_input=False,
            include_prediction=True,
        ),
    )


# -------------------------
# Complete prediction config
# -------------------------
def make_prediction_config() -> PredictionConfig:
    return PredictionConfig(
        source=make_prediction_source_config(),
        dataset=make_prediction_dataset_config(),
        data=make_prediction_data_config(),
        inference=make_prediction_inference_config(),
        exports=make_prediction_exports_config(),
    )


# -------------------------
# Evaluation config components
# -------------------------
def make_evaluation_source_config() -> EvaluationSourceConfig:
    return EvaluationSourceConfig(
        prediction_manifest_path=Path(
            "overridden_prediction_manifest.yaml"
        ),
        embeddings_path=None,
        reconstructions_path=None,
    )


def make_evaluation_run_config() -> EvaluationRunConfig:
    return EvaluationRunConfig(
        output_root=Path("overridden_evaluation_outputs"),
        run_name="overridden_evaluation",
    )


def make_evaluation_reductions_config() -> EvaluationReductionsConfig:
    return EvaluationReductionsConfig(
        pca=PCAConfig(
            enabled=True,
            params=PCAParams(
                n_components=3,
                key_added="X_pca_overridden",
                random_state=999,
                overwrite=True,
            ),
        ),
        umap=UMAPConfig(
            enabled=False,
            params=UMAPParams(
                n_neighbors=7,
                n_pcs=3,
                min_dist=0.2,
                metric="cosine",
                key_added="X_umap_overridden",
                neighbors_key="umap_neighbors_overridden",
                random_state=999,
                overwrite=True,
            ),
        ),
        tsne=TSNEConfig(
            enabled=True,
            params=TSNEParams(
                n_pcs=3,
                perplexity=5.0,
                key_added="X_tsne_overridden",
                random_state=999,
                overwrite=True,
            ),
        ),
    )


def make_evaluation_clustering_config() -> EvaluationClusteringConfig:
    return EvaluationClusteringConfig(
        kmeans=KMeansConfig(
            enabled=True,
            params=KMeansParams(
                n_clusters=3,
                key_added="kmeans_overridden",
                random_state=999,
                n_init=5,
                overwrite=True,
            ),
        ),
        leiden=LeidenConfig(
            enabled=True,
            params=LeidenParams(
                resolution=0.5,
                n_neighbors=7,
                n_pcs=3,
                metric="cosine",
                key_added="leiden_overridden",
                neighbors_key="leiden_neighbors_overridden",
                random_state=999,
                overwrite=True,
                neighbors_kwargs={},
                leiden_kwargs={"directed": False},
            ),
        ),
    )


def make_evaluation_metrics_config() -> EvaluationMetricsConfig:
    return EvaluationMetricsConfig(
        clustering=EvaluationClusteringMetricsConfig(
            internal=InternalClusteringMetricConfig(
                enabled=True,
                selected=["davies_bouldin"],
                params=None,
            ),
            external=ExternalClusteringMetricConfig(
                enabled=True,
                label_key="label_str",
                selected=["homogeneity"],
                params=None,
            ),
        ),
        embedding=EvalMetricGroupConfig(
            enabled=False,
            selected=None,
            params=None,
        ),
        predictability=EvaluationPredictabilityConfig(
            enabled=True,
            selected=["dummy", "linear"],
            target_key="continuous_target",
            task="regression",
            cv=EvaluationCrossValidationConfig(
                method="kfold",
                n_splits=2,
                group_key=None,
                shuffle=True,
                random_state=999,
                scoring="r2",
            ),
            tuning=EvaluationCVTuningConfig(
                enabled=False,
                inner_cv=None,
            ),
            params=EvaluationPredictabilityParamsConfig(
                dummy=DummyProbeConfig(
                    strategy="mean",
                ),
                linear=RidgeProbeConfig(
                    model="ridge",
                    standardize=True,
                    alpha=0.5,
                ),
            ),
        ),
        reconstruction=ReconstructionMetricConfig(
            enabled=True,
            selected=["rmse"],
            params=None,
            reduction="both",
        ),
    )


def make_evaluation_reconstruction_config(
) -> EvaluationReconstructionConfig:
    return EvaluationReconstructionConfig(
        export_tiffs=False,
        n_examples=2,
        error_maps=ErrorMapParams(
            kinds=["squared"],
            denominator_floor=1e-6,
        ),
    )


def make_evaluation_plots_config() -> EvaluationPlotsConfig:
    return EvaluationPlotsConfig(
        enabled=True,
        params=PlotParams(
            accent_color="#336699",
            color_by=["group"],
            dpi=96,
            formats=["svg"],
            reconstruction_grid=ReconstructionGridConfig(
                include_error_maps=False,
                random_state=999,
                stratify_by="group",
                channel_selection="all",
            ),
        ),
    )


# -------------------------
# Complete evaluation config
# -------------------------
def make_evaluation_config() -> EvaluationConfig:
    return EvaluationConfig(
        source=make_evaluation_source_config(),
        run=make_evaluation_run_config(),
        reductions=make_evaluation_reductions_config(),
        clustering=make_evaluation_clustering_config(),
        metrics=make_evaluation_metrics_config(),
        reconstruction=make_evaluation_reconstruction_config(),
        plots=make_evaluation_plots_config(),
    )