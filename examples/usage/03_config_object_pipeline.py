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
    TrainingCheckpointConfig,
    TrainingDataModuleConfig,
    TrainingDecoderConfig,
    TrainingEncoderConfig,
    TrainingInspectionConfig,
    TrainingLossTermConfig,
    MNISTDatasetConfig,
    MNISTDatasetParams,
    TrainingModelConfig,
    TrainingOptimizerConfig,
    TrainingReproducibilityConfig,
    TrainingRunConfig,
    TrainingTorchviewConfig,
    TrainingTrainerConfig,
    TrainingConfig,
    TrainingTransformConfig,
# Prediction schema
    PredictionConfig,
    PredictionDataConfig,
    PredictionEmbeddingsExportConfig,
    PredictionExportConfig,
    PredictionInferenceConfig,
    PredictionReconstructionsExportConfig,
    PredictionSourceConfig,
# Evaluation schema
    EvaluationDummyProbeConfig,
    EvaluationEmbeddingMetricConfig,
    EvaluationErrorMapParams,
    EvaluationClusteringConfig,
    EvaluationClusteringMetricsConfig,
    EvaluationConfig,
    EvaluationCrossValidationConfig,
    EvaluationCVTuningConfig,
    EvaluationMetricsConfig,
    EvaluationPlotsConfig,
    EvaluationPredictabilityConfig,
    EvaluationPredictabilityParamsConfig,
    EvaluationPredictabilityTargetConfig,
    EvaluationReconstructionConfig,
    EvaluationReductionsConfig,
    EvaluationSourceConfig,
    EvaluationExternalClusteringMetricConfig,
    EvaluationInternalClusteringMetricConfig,
    EvaluationKMeansConfig,
    EvaluationKMeansParams,
    EvaluationLeidenConfig,
    EvaluationLogisticRegressionProbeConfig,
    EvaluationPCAConfig,
    EvaluationPCAParams,
    EvaluationPlotParams,
    EvaluationReconstructionGridConfig,
    EvaluationReconstructionMetricConfig,
    EvaluationTSNEConfig,
    EvaluationUMAPConfig,
    EvaluationUMAPParams,
)


def build_training_config() -> TrainingConfig:
    return TrainingConfig(
        run=TrainingRunConfig(
            output_root=Path("outputs"),
            project_name="02_config_object_pipeline",
        ),
        reproducibility=TrainingReproducibilityConfig(
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
            TrainingTransformConfig(
                name="to_dtype",
                apply_to=["training", "validation"],
                params={
                    "dtype": "float32",
                    "scale": True,
                },
            ),
        ],
        datamodule=TrainingDataModuleConfig(
            batch_size=128,
            val_fraction=0.1,
            num_workers=4,
            pin_memory="auto",
            persistent_workers=True,
            drop_last=False,
        ),
        model=TrainingModelConfig(
            name="vae",
            params={
                "latent_dim": 32,
            },
        ),
        encoder=TrainingEncoderConfig(
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
        decoder=TrainingDecoderConfig(
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
                "mse": TrainingLossTermConfig(
                    weight=0.8,
                    params={"reduction": "mean"},
                ),
                "mae": TrainingLossTermConfig(
                    weight=0.2,
                    params={"reduction": "mean"},
                ),
            },
            "regularization": {
                "gaussian_kld": TrainingLossTermConfig(
                    weight=0.0001,
                    params={"reduction": "mean"},
                ),
            },
        },
        optimizer=TrainingOptimizerConfig(
            name="adam",
            params={"lr": 0.001},
        ),
        trainer=TrainingTrainerConfig(
            max_epochs=5,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=20,
            deterministic=True,
            benchmark=False,
            precision="32-true",
        ),
        checkpointing=TrainingCheckpointConfig(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        inspection=TrainingInspectionConfig(
            torchview=TrainingTorchviewConfig(enabled=False),
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
            pca=EvaluationPCAConfig(
                enabled=True,
                params=EvaluationPCAParams(
                    n_components=30,
                    random_state=137,
                ),
            ),
            umap=EvaluationUMAPConfig(
                enabled=True,
                params=EvaluationUMAPParams(
                    n_neighbors=15,
                    n_pcs=30,
                    min_dist=0.1,
                    metric="euclidean",
                    random_state=137,
                ),
            ),
            tsne=EvaluationTSNEConfig(
                enabled=False,
            ),
        ),
        clustering=EvaluationClusteringConfig(
            kmeans=EvaluationKMeansConfig(
                enabled=True,
                params=EvaluationKMeansParams(
                    n_clusters=10,
                    random_state=137,
                ),
            ),
            leiden=EvaluationLeidenConfig(
                enabled=False,
            ),
        ),
        metrics=EvaluationMetricsConfig(
            clustering=EvaluationClusteringMetricsConfig(
                internal=EvaluationInternalClusteringMetricConfig(
                    enabled=True,
                    selected=[
                        "calinski_harabasz",
                        "davies_bouldin",
                    ],
                ),
                external=EvaluationExternalClusteringMetricConfig(
                    enabled=True,
                    label_key="label",
                    selected=[
                        "adjusted_mutual_info",
                        "adjusted_rand_index",
                        "homogeneity",
                    ],
                ),
            ),
            embedding=EvaluationEmbeddingMetricConfig(
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
                targets={
                    "label": EvaluationPredictabilityTargetConfig(
                        selected=[
                            "dummy",
                            "linear",
                        ],
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
                            dummy=EvaluationDummyProbeConfig(
                                strategy="most_frequent",
                                random_state=137,
                            ),
                            linear=EvaluationLogisticRegressionProbeConfig(
                                standardize=True,
                                C=1.0,
                                class_weight="balanced",
                                max_iter=5000,
                            ),
                        ),
                    ),
                },
            ),
            reconstruction=EvaluationReconstructionMetricConfig(
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
            error_maps=EvaluationErrorMapParams(
                kinds=["absolute"],
                denominator_floor=None,
            ),
        ),
        plots=EvaluationPlotsConfig(
            enabled=True,
            params=EvaluationPlotParams(
                color_by=[
                    "label",
                    "kmeans",
                ],
                dpi=150,
                formats=["png"],
                reconstruction_grid=EvaluationReconstructionGridConfig(
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
