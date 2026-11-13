# Configuration schemas

Look up the fields accepted by each workflow, including nested parameters, inherited fields, defaults, and omission/null behavior. This reference is generated from BenchRep's schema declarations and their documentation metadata whenever the docs are built.

!!! tip "Prefer exploring interactively?"
     This is a detailed lookup reference, not a guide to read end to end. For a more approachable introduction, the [configuration wizard](../../guides/configure-workflows/index.md#browser-wizard) lets you explore the configuration hierarchy graphically, edit fields interactively, and read their documentation alongside them.

| Configuration | Reference |
| --- | --- |
| Training | [TrainingConfig and its sections](training.md) |
| Prediction | [PredictionConfig and its sections](prediction.md) |
| Evaluation | [EvaluationConfig and its sections](evaluation.md) |
| Shared contracts | [Datasets, runtime override requirements, composite schemas, and reusable bases](shared.md) |

## Reading this reference

Each schema shows its defining Python module and a source link. Expand a field for its description, type, constraints, and notes. Links in types lead to nested schemas or type aliases; inherited fields are included in concrete schemas.

- **Required** means the field has no schema default. The omission notes also describe conditional requirements enforced by validators or workflow resolution.
- **Default** is the value used when you omit a field. **Default factory** creates that value when needed—for example, `TrainingRunConfig()` creates a run configuration with its default settings.
- **If omitted** and **If null** describe separate cases. An empty mapping or list is also a supplied value; it is not automatically equivalent to either.
- **Additional fields** describes the schema's handling of unknown keys. Acceptance by a schema does not guarantee acceptance by a downstream constructor or resolver.
- **Field types** lists aliases used by these fields. Constraints inside `Annotated[...]` and named Pydantic types such as `PositiveInt` apply as well as the field-level settings shown in each entry.

The reference displays declarations and authored guidance; it does not execute validators or enumerate every cross-field rule. A workflow can reject a combination of individually valid fields during validation or resolution. See [Configuration validation and resolution](../../understanding/workflows/configuration.md) for the broader model.

## Inspect from Python

Most schemas are available from `benchrep.assembly.schemas`. The defining module is shown for every entry, including reusable bases that are not re-exported there.

```python
from benchrep import inspect_config
from benchrep.assembly.schemas import TrainingConfig

inspect_config(TrainingConfig)
```

`TrainingConfig.model_json_schema()` provides a machine-readable JSON Schema. The same inspection methods work for prediction, evaluation, and nested schema classes.

For configuration examples, see [Configure workflows](../../guides/configure-workflows/index.md).
