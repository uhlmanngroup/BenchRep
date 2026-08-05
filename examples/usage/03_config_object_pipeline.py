"""Run a BenchRep pipeline using Pydantic configuration objects.

Run from the repository root:

    python examples/usage/03_config_object_pipeline.py

This example constructs the same general MNIST VAE pipeline as
01_yaml_pipeline.py without loading configuration from YAML.
"""

from pathlib import Path

from benchrep import (
    train_vae,
    predict_vae,
    evaluate,
)
from benchrep.assembly.schemas import (
# Training schema
    CheckpointConfig,
    DataModuleConfig,
    DecoderConfig,
    EncoderConfig,
    InspectionConfig,
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
# Prediction schema
    PredictionConfig,
    PredictionDataConfig,
    PredictionEmbeddingsExportConfig,
    PredictionExportConfig,
    PredictionInferenceConfig,
    PredictionReconstructionsExportConfig,
    PredictionSourceConfig,
# Evaluation schema
    DummyProbeConfig,
    EmbeddingMetricConfig,
    ErrorMapParams,
    EvaluationClusteringConfig,
    EvaluationClusteringMetricsConfig,
    EvaluationConfig,
    EvaluationCrossValidationConfig,
    EvaluationCVTuningConfig,
    EvaluationMetricsConfig,
    EvaluationPlotsConfig,
    EvaluationPredictabilityConfig,
    EvaluationPredictabilityParamsConfig,
    EvaluationReconstructionConfig,
    EvaluationReductionsConfig,
    EvaluationSourceConfig,
    ExternalClusteringMetricConfig,
    InternalClusteringMetricConfig,
    KMeansConfig,
    KMeansParams,
    LeidenConfig,
    LogisticRegressionProbeConfig,
    PCAConfig,
    PCAParams,
    PlotParams,
    ReconstructionGridConfig,
    ReconstructionMetricConfig,
    TSNEConfig,
    UMAPConfig,
    UMAPParams,
)


def build_training_config() -> TrainingConfig:
    return TrainingConfig(
        run=RunConfig(
            output_root=Path("outputs"),
            project_name="02_config_object_pipeline",
        ),
        reproducibility=ReproducibilityConfig(
            seed=137,
            seed_workers=True,
            float32_matmul_precision="highest",
        ),
        dataset=MNISTDatasetConfig(
            params=MNISTDatasetParams(
                root=Path("examples/data/mnist"),
                split="train",
                download=True,
            ),
        ),
        transforms=[
            TransformConfig(
                name="to_dtype",
                apply_to=["training", "validation"],
                params={
                    "dtype": "float32",
                    "scale": True,
                },
            ),
        ],
        datamodule=DataModuleConfig(
            batch_size=128,
            val_fraction=0.1,
            num_workers=4,
            pin_memory="auto",
            persistent_workers=True,
            drop_last=False,
        ),
        model=ModelConfig(
            name="vae",
            params={
                "latent_dim": 32,
            },
        ),
        encoder=EncoderConfig(
            name="conv2d",
            params={
                "input_shape": [1, 28, 28],
                "output_dim": 32,
                "channels": [32, 64],
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "normalization": "batchnorm",
            },
        ),
        decoder=DecoderConfig(
            name="upsample_conv2d",
            params={
                "output_shape": [1, 28, 28],
                "channels": [32, 16],
                "normalization": "batchnorm",
                "output_activation": "sigmoid",
            },
        ),
        losses={
            "reconstruction": {
                "mse": LossTermConfig(
                    weight=0.8,
                    params={"reduction": "mean"},
                ),
                "mae": LossTermConfig(
                    weight=0.2,
                    params={"reduction": "mean"},
                ),
            },
            "regularization": {
                "gaussian_kld": LossTermConfig(
                    weight=0.0001,
                    params={"reduction": "mean"},
                ),
            },
        },
        optimizer=OptimizerConfig(
            name="adam",
            params={"lr": 0.001},
        ),
        trainer=TrainerConfig(
            max_epochs=5,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=20,
            deterministic=True,
            benchmark=False,
            precision="32-true",
        ),
        checkpointing=CheckpointConfig(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        inspection=InspectionConfig(
            torchview=TorchviewConfig(enabled=False),
        ),
    )


def build_prediction_config(
    training_manifest_path: Path,
) -> PredictionConfig:
    return PredictionConfig(
        source=PredictionSourceConfig(
            training_manifest_path=training_manifest_path,
            checkpoint="best",
        ),
        dataset=MNISTDatasetConfig(
            params=MNISTDatasetParams(
                root=Path("examples/data/mnist"),
                split="test",
                download=True,
            ),
        ),
        transforms=None,
        data=PredictionDataConfig(),
        inference=PredictionInferenceConfig(
            reconstruction_latent_source="mean",
        ),
        exports=PredictionExportConfig(
            mode="standard",
            embeddings=PredictionEmbeddingsExportConfig(
                enabled=True,
            ),
            reconstructions=PredictionReconstructionsExportConfig(
                enabled=True,
                n_examples=32,
                selection="random",
                stratify_by="label",
                include_input=True,
                include_prediction=True,
            ),
        ),
    )


def build_evaluation_config(
    prediction_manifest_path: Path,
) -> EvaluationConfig:
    return EvaluationConfig(
        source=EvaluationSourceConfig(
            prediction_manifest_path=prediction_manifest_path,
        ),
        reductions=EvaluationReductionsConfig(
            pca=PCAConfig(
                enabled=True,
                params=PCAParams(
                    n_components=30,
                    random_state=137,
                ),
            ),
            umap=UMAPConfig(
                enabled=True,
                params=UMAPParams(
                    n_neighbors=15,
                    n_pcs=30,
                    min_dist=0.1,
                    metric="euclidean",
                    random_state=137,
                ),
            ),
            tsne=TSNEConfig(
                enabled=False,
            ),
        ),
        clustering=EvaluationClusteringConfig(
            kmeans=KMeansConfig(
                enabled=True,
                params=KMeansParams(
                    n_clusters=10,
                    random_state=137,
                ),
            ),
            leiden=LeidenConfig(
                enabled=False,
            ),
        ),
        metrics=EvaluationMetricsConfig(
            clustering=EvaluationClusteringMetricsConfig(
                internal=InternalClusteringMetricConfig(
                    enabled=True,
                    selected=[
                        "calinski_harabasz",
                        "davies_bouldin",
                    ],
                ),
                external=ExternalClusteringMetricConfig(
                    enabled=True,
                    label_key="label",
                    selected=[
                        "adjusted_mutual_info",
                        "adjusted_rand_index",
                        "homogeneity",
                    ],
                ),
            ),
            embedding=EmbeddingMetricConfig(
                enabled=True,
                selected=[
                    "mean",
                    "median",
                    "standard_deviation",
                    "minimum",
                    "maximum",
                    "quantiles",
                ],
            ),
            predictability=EvaluationPredictabilityConfig(
                enabled=True,
                selected=[
                    "dummy",
                    "linear",
                ],
                target_key="label",
                task="classification",
                cv=EvaluationCrossValidationConfig(
                    method="stratified_kfold",
                    n_splits=5,
                    shuffle=True,
                    random_state=137,
                    scoring="balanced_accuracy",
                ),
                tuning=EvaluationCVTuningConfig(
                    enabled=False,
                ),
                params=EvaluationPredictabilityParamsConfig(
                    dummy=DummyProbeConfig(
                        strategy="most_frequent",
                        random_state=137,
                    ),
                    linear=LogisticRegressionProbeConfig(
                        standardize=True,
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                    ),
                ),
            ),
            reconstruction=ReconstructionMetricConfig(
                enabled=True,
                selected=[
                    "mae",
                    "mse",
                    "rmse",
                    "max_absolute_error",
                ],
                reduction="global",
            ),
        ),
        reconstruction=EvaluationReconstructionConfig(
            export_tiffs=False,
            n_examples=None,
            error_maps=ErrorMapParams(
                kinds=["absolute"],
                denominator_floor=None,
            ),
        ),
        plots=EvaluationPlotsConfig(
            enabled=True,
            params=PlotParams(
                color_by=[
                    "label",
                    "kmeans",
                ],
                dpi=150,
                formats=["png"],
                reconstruction_grid=ReconstructionGridConfig(
                    include_error_maps=True,
                    random_state=137,
                    stratify_by="label",
                    channel_selection=0,
                ),
            ),
        ),
    )


def main() -> None:
    print("\n=== Training ===")
    training_config = build_training_config()
    training_result = train_vae(
        full_config_object=training_config,
    )
    print(f"Training manifest: {training_result.manifest_path}")

    print("\n=== Prediction ===")
    # Here the generated training manifest is stored directly in the prediction
    # config. A manifest path can also be passed to predict_vae() through its
    # training_manifest_path argument to override the path stored in the config.
    prediction_config = build_prediction_config(
        training_manifest_path=training_result.manifest_path,
    )
    prediction_result = predict_vae(
        full_config_object=prediction_config,
    )
    print(f"Prediction manifest: {prediction_result.manifest_path}")

    print("\n=== Evaluation ===")
    # Likewise, this stores the generated prediction manifest directly in the
    # evaluation config. evaluate(prediction_manifest_path=...) can override it.
    evaluation_config = build_evaluation_config(
        prediction_manifest_path=prediction_result.manifest_path,
    )
    evaluation_result = evaluate(
        full_config_object=evaluation_config,
    )
    print(f"Evaluation manifest: {evaluation_result.manifest_path}")

    print("\n=== Pipeline complete ===")
    print(f"Training outputs:   {training_result.run_context.output_dir}")
    print(f"Prediction outputs: {prediction_result.run_context.output_dir}")
    print(f"Evaluation outputs: {evaluation_result.run_context.output_dir}")


if __name__ == "__main__":
    main()
