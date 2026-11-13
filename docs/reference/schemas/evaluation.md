# Evaluation schemas

Fields for `EvaluationConfig`, its evaluation steps, metrics, probes, and exports. The shared enablement and metric-selection contracts are documented under [Evaluation base schemas](shared.md#evaluation-base-schemas).

Expand a field to see its type, description, constraints, and omission/null behavior. Defaults shown here are schema declarations; workflow resolution can derive additional settings. See [Reading this reference](index.md#reading-this-reference).

## Workflow configuration

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Source and run

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationSourceConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationRunConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Dimensionality reduction

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationReductionsConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationPCAConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationPCAParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationUMAPConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationUMAPParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationTSNEConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationTSNEParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Clustering

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationClusteringConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationKMeansConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationKMeansParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationLeidenConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationLeidenParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationHDBSCANConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationHDBSCANParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Metrics

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationMetricsConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationClusteringMetricsConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationInternalClusteringMetricConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationExternalClusteringMetricConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationEmbeddingMetricConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationReconstructionMetricConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Predictability and cross-validation

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationPredictabilityConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationPredictabilityTargetConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationCrossValidationConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationCVTuningConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationTuningInnerCVConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationPredictabilityParamsConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Predictability probes

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationDummyProbeConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationLogisticRegressionProbeConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationRidgeProbeConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationKNNProbeConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationRandomForestProbeConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationXGBoostProbeConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationSVMRBFProbeConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Reconstructions and plots

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationReconstructionConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationErrorMapParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationPlotsConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationPlotParams
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.ColorByConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationReconstructionGridConfig
    options:
      heading_level: 3
      show_bases: false
      merge_init_into_class: false

## Field types

These aliases define allowed values, constrained types, or alternative schemas. They are types, not additional YAML sections. Validator names in `Annotated` declarations describe validation hooks, not configuration arguments.

::: benchrep.assembly.schemas.evaluation_config_schema.RegistrySelection
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.PredictabilityTargetKey
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationReconstructionPairId
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.NSplits
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.PositiveFloatOrList
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.PositiveIntOrList
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.KNNWeights
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.KNNWeightsOrList
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.KNNMetric
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.KNNMetricOrList
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.SVMRBFGammaValue
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.SVMRBFGammaValueOrList
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.MaxIterWithNoLimitSentinel
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.MaxDepthValue
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.MaxDepthParam
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.ErrorMapKind
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.PlotFormat
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.HexColor
    options:
      heading_level: 3

::: benchrep.assembly.schemas.evaluation_config_schema.EvaluationLinearProbeConfig
    options:
      heading_level: 3
