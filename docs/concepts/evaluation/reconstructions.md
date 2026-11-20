# Reconstruction evaluation

Reconstruction evaluation compares saved model outputs with their corresponding inputs. It measures numerical differences, visualizes where those differences occur, and can export images for inspection. It does not reload the model or regenerate reconstructions.

## Supply aligned images

A reconstruction bundle contains:

| File | Contents |
| --- | --- |
| `input.pt` | Input images as a tensor or NumPy array |
| `reconstruction.pt` | Corresponding reconstructed images |
| `obs.pt` | Annotations with one entry per image |
| `reconstruction_export_metadata.pt` | Optional metadata, including channel names |

Image arrays must have identical, nonempty shapes: `(B, H, W)` or `(B, C, H, W)`. Values must be finite and real. Each input must correspond to the reconstruction at the same position, with matching channel order and pixel coordinates. <strong class="highlight">BenchRep checks shape compatibility, but does not infer or repair image correspondence.</strong>

Prediction produces this bundle for each selected input–reconstruction pair. Evaluation selects one pair from its manifest, or accepts an external bundle through `source.reconstructions_path`; see [Choose the source](inputs.md#choose-the-source). Only saved examples are available for analysis. If prediction exported a subset, evaluation cannot score the omitted examples.

Observation mappings or table-like annotations provide identifiers for filenames, grid labels, and stratified grid selection. They do not realign the image arrays. Channel names from the optional metadata label per-channel metric results; otherwise BenchRep generates names such as `channel_0`.

## Measure reconstruction error

Reconstruction metrics are enabled automatically when a bundle is available. The default metrics are MAE, MSE, RMSE, and maximum absolute error. They operate on the saved numerical values, with no automatic intensity normalization.

```yaml
metrics:
  reconstruction:
    selected: [mae, mse]
    reduction: both
```

`global` passes the complete arrays to each metric. `per_channel` evaluates each channel separately across all examples and pixels; `both` retains both views. For built-in MAE and MSE, global results aggregate all elements. An external metric follows its own computation and [result contract](metrics.md#result-contracts).

All available image pairs contribute to metrics. `reconstruction.n_examples` limits TIFF export, not metric computation, and reconstruction grids have their own sampling behavior. Comparing metric values between runs also requires comparable image scales and preprocessing: an MSE on normalized inputs is not directly comparable to an MSE on raw intensities.

## Locate errors within images

Error maps preserve the spatial layout of the comparison. Configure their kinds under `reconstruction.error_maps`:

| Kind | Pixel value |
| --- | --- |
| `signed` | Reconstruction minus input |
| `absolute` | Absolute difference |
| `squared` | Squared difference |
| `relative` | Absolute difference divided by absolute input intensity |
| `normalized_absolute_global` | Absolute difference divided by one input intensity range |
| `normalized_absolute_per_channel` | Absolute difference divided by the corresponding channel's input intensity range |

Relative errors and inferred ranges use `denominator_floor` to avoid division by zero or very small denominators. A global range can be supplied with `data_range`; otherwise it is inferred from the images used for that map computation. Per-channel ranges are inferred separately. Relative error can therefore be large near zero-valued inputs even when the absolute difference is small.

Error-map settings are shared by TIFF and grid output; setting them alone does not enable an export:

```yaml
reconstruction:
  export_tiffs: true
  n_examples: 12
  error_maps:
    kinds: [absolute, signed]
```

TIFF export uses the first available examples up to its limit. Grids independently select examples and can stratify them by annotation; see [Figures and visualization](figures.md#reconstruction-grids).

## Implementation

The [reconstruction loader](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/reconstructions/data.py) loads tensors on CPU, converts them into the validated array interface, and packages images, observations, metadata, and the TIFF example limit in `ReconstructionEvaluationInput`.

The [metric runner](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/reconstructions/reconstruction_metrics.py) dispatches registered metrics over the full arrays or individual channels. [Error-map computation](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/reconstructions/error_maps.py) produces arrays separately from file writing. Successful metric or map results can survive recoverable failures of other requested calculations.

The records layer then writes metric JSON, TIFFs, and comparison grids. Keeping computation separate from export means a figure-writing failure need not invalidate an already computed score. The returned reconstruction outputs and export paths show what was computed and what reached disk; the [status report](../results/statuses.md) records the distinction.
