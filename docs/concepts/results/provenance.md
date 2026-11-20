# Manifests and provenance

A manifest is the saved record of one workflow run. It identifies the run's configuration sources, runtime implementations, status, and artifact locations. Downstream workflows read these records instead of relying on filenames inferred by the user.

## Follow the source of a result

The 3 stages record different links:

| Manifest | Connection to upstream work |
| --- | --- |
| `training_manifest.yaml` | Model/data construction, configuration, and checkpoints produced by training |
| `prediction_manifest.yaml` | Training manifest and the exact selected checkpoint, plus exported artifact paths |
| `evaluation_manifest.yaml` | Prediction manifest or direct artifact paths used for each evaluation input |

Each lives in its run's `records/metadata/` directory. A chain can span separate Python processes: the connection is the manifest path. Evaluation can also use a mixture of manifest-linked and directly supplied artifacts, which its source records distinguish.

For an unexpected evaluation result, start at the evaluation manifest, identify the artifact source, then follow the prediction manifest to the selected training checkpoint. The records describe the actual run rather than assuming that prediction used the last model state or every requested export succeeded.

## Original, resolved, and embedded configuration

The original configuration preserves the supplied base YAML or config object. Section replacements, inherited values, runtime override decisions, and resolver-derived settings are reflected in the resolved configuration instead.

The manifest embeds that resolved mapping under `appendix.resolved_config` and also points to `records/configs/resolved_config.yaml`. <strong class="highlight">Prediction reads the training configuration embedded in the training manifest</strong>; the separate resolved YAML path is retained as an informational record.

Prediction and evaluation also list inherited fields under `config_inheritance`. This helps answer whether a setting was supplied explicitly or obtained from an upstream run. It is field-specific inheritance, not an automatic merge of every upstream setting. See [Configuration semantics](../workflows/configuration.md).

The resolved snapshot explains a past run. For a new run, an authored configuration usually better preserves intentions such as “use the best checkpoint” or “inherit these settings,” allowing resolution to happen in the new context. See [Configuration wizards](../workflows/wizards.md) for authored versus resolved exports.

## What can be reconstructed from configuration?

The manifest field `construction.config.run_reconstructable_from_resolved_config` records whether BenchRep's construction path depends on supplied runtime implementations. For training, it is `false` when the run uses a model override, datamodule override, or `runtime_callbacks`.

This is not a promise of complete reproducibility when `true`. Configurations still depend on registered implementations, installed dependencies, available datasets, and referenced artifacts. A registered component's name does not embed its source code or freeze its version.

| Runtime input used in training | Consequence for later prediction |
| --- | --- |
| Model class or instance | Supply a compatible model override again so checkpoint weights have an implementation to load into |
| Datamodule class or instance | Supply prediction data explicitly or provide a prediction datamodule override; the external training data construction cannot be inherited |
| Runtime callback | The callback is needed to reproduce that training setup, but is not required merely to predict from its checkpoint |

Prediction records constructability for its own model and datamodule inputs. Evaluation records configuration-based construction because it consumes artifact paths rather than live model implementations. In every case, referenced files and required registrations must remain available.

??? note "Recorded Python objects are descriptions, not serialized implementations"
    Manifests record whether model and datamodule construction came from configuration, an external class, or an external instance. Runtime callbacks are recorded by class name. Unsupported Python values encountered while serializing configuration are represented by `__benchrep_unserializable__` descriptions.

    These records help explain the run, but do not recreate those objects. Import registrations and supply necessary runtime implementations again in another process. See [External models and datamodules](../extensibility/runtime-implementations.md).

## Paths and portability

Ordinary configuration paths follow the working-directory rules described in [Configuration semantics](../workflows/configuration.md#paths-follow-their-source). Relative artifact paths read from a manifest are resolved against that manifest's directory. Many paths written by a completed workflow are absolute.

Moving a run directory therefore does not automatically make every recorded path valid at the new location. Preserve the linked files and update references deliberately when relocating an experiment. A manifest is a provenance record and input locator, not a self-contained archive of data, software, and weights.

## How manifests are assembled

Each workflow passes its resolved run specification, actual runtime class names, outcomes, and artifact paths to a stage-specific writer in [the manifest module](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/records/manifests.py). The writer combines those with shared config, log, and environment record locations and serializes YAML.

This happens during workflow finalization. Earlier setup or execution failures can leave logs and partial outputs without a final manifest. Conversely, a manifest may describe a failed or partially completed run. Read its status alongside its artifact entries; see [Statuses and failures](statuses.md).
