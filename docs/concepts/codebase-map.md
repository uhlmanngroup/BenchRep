# Codebase map

Expand a directory to see what it owns. File names and directory links open the corresponding source in the repository. For a path through the code, jump to [A training run across packages](#a-training-run-across-packages) or [Common changes](#common-changes).

<div class="br-code-map">
  <details open>
    <summary><code>src/benchrep/</code> — The installed package. Expand a package to follow the part of the workflow it owns.</summary>
    <div class="br-branch">
      <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep" aria-label="Browse src/benchrep on GitHub">Browse directory</a>
      <details>
        <summary><code>workflows/</code> — Public workflow calls and the order of operations within each run.</summary>
        <div class="br-branch">
          <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/workflows" aria-label="Browse src/benchrep/workflows on GitHub">Browse directory</a>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/workflows/train.py"><code>train.py</code></a> — The family-specific training calls converge on one fit procedure.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/workflows/predict.py"><code>predict.py</code></a> — Restores the model, runs inference, and dispatches output validation and exports.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/workflows/evaluate.py"><code>evaluate.py</code></a> — Loads input artifacts, executes analysis pipelines, and assembles the evaluation result.</p>
        </div>
      </details>
      <details>
        <summary><code>assembly/</code> — Turns experiment settings into instructions and live Python objects.</summary>
        <div class="br-branch">
          <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/assembly" aria-label="Browse src/benchrep/assembly on GitHub">Browse directory</a>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/config.py"><code>config.py</code></a> — Composes a workflow input with section replacements and implements validated field updates.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/assembly/schemas"><code>schemas/</code></a> — Defines the training, prediction, evaluation, composite, and runtime-override config types in Pydantic.</p>
          <details>
            <summary><code>registries/</code> — Associates component names and aliases with implementations or factories.</summary>
            <div class="br-branch">
              <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/assembly/registries" aria-label="Browse src/benchrep/assembly/registries on GitHub">Browse directory</a>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/registries/core.py"><code>core.py</code></a> — Defines the registry objects, their public registration policies, and the mapping from loss roles to registries.</p>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/registries/builtins.py"><code>builtins.py</code></a> — Registers the shipped components and their contracts.</p>
            </div>
          </details>
          <details>
            <summary><code>resolvers/</code> — Uses the configuration, registered components, and any upstream records to decide what will run.</summary>
            <div class="br-branch">
              <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/assembly/resolvers" aria-label="Browse src/benchrep/assembly/resolvers on GitHub">Browse directory</a>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/training_config_resolver.py"><code>training_config_resolver.py</code></a> — Assembles the training run specification, including graph and loss specifications when applicable.</p>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/composite_model_resolver.py"><code>composite_model_resolver.py</code></a> — Resolves tensor connections, verifies producers and consumers, and orders component calls by dependency.</p>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/loss_resolver.py"><code>loss_resolver.py</code></a> — Resolves term identities, constructor arguments, weights, schedules, and composite loss inputs.</p>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/prediction_config_resolver.py"><code>prediction_config_resolver.py</code></a> — Selects the checkpoint and determines what to inherit from training and what to export.</p>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/evaluation_config_resolver.py"><code>evaluation_config_resolver.py</code></a> — Resolves artifact sources and which analyses and exports apply to them.</p>
            </div>
          </details>
          <details>
            <summary><code>builders/</code> — Instantiates the objects named in a resolved configuration.</summary>
            <div class="br-branch">
              <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/assembly/builders" aria-label="Browse src/benchrep/assembly/builders on GitHub">Browse directory</a>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/model_builder.py"><code>model_builder.py</code></a> — Constructs a canonical or composite Lightning model and gives it its component and loss modules.</p>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/architecture_builder.py"><code>architecture_builder.py</code></a> — Constructs canonical encoders and decoders and supplies dimensions where the model determines them.</p>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/loss_builder.py"><code>loss_builder.py</code></a> — Creates loss modules and any schedules, then wraps each in a weighted loss term.</p>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/data_builder.py"><code>data_builder.py</code></a> — Constructs the configured dataset, transform sequences, and datamodule.</p>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/builders/trainer_builder.py"><code>trainer_builder.py</code></a> — Builds the Lightning trainer and integrates logging, checkpointing, and callbacks.</p>
            </div>
          </details>
        </div>
      </details>
      <details>
        <summary><code>architecture/</code> — The PyTorch and Lightning objects that operate on real batches.</summary>
        <div class="br-branch">
          <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/architecture" aria-label="Browse src/benchrep/architecture on GitHub">Browse directory</a>
          <details>
            <summary><code>models/</code> — Concrete autoencoder, VAE, and composite implementations.</summary>
            <div class="br-branch">
              <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/architecture/models" aria-label="Browse src/benchrep/architecture/models on GitHub">Browse directory</a>
              <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/architecture/models/composite.py"><code>composite.py</code></a> — Executes the resolved graph, gathers the inputs for each loss, and forms the weighted training objective.</p>
            </div>
          </details>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/architecture/data"><code>data/</code></a> — Dataset adapters, sample transforms, and the datamodule that creates subsets and dataloaders.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/architecture/encoders"><code>encoders/</code></a> — Image-to-feature modules and the canonical encoder base interface.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/architecture/decoders"><code>decoders/</code></a> — Latent-to-image modules and the canonical decoder base interface.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/architecture/heads"><code>heads/</code></a> — Modules that turn features into task outputs or Gaussian latent variables.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/architecture/losses"><code>losses/</code></a> — Loss computations, loss-entry contracts, runtime terms, and position-based weight schedules.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/architecture/composite_model_component_contracts.py"><code>composite_model_component_contracts.py</code></a> — Declares named input ports and the tensor or mapping returned by a component.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/architecture/composite_model_roles.py"><code>composite_model_roles.py</code></a> — Relates semantic input and output roles to tensor structures.</p>
        </div>
      </details>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/interfaces"><code>interfaces/</code></a> — The model-family definitions, whole-model bases, batch and prediction types, and annotation compatibility checks.</p>
      <details>
        <summary><code>runtime/</code> — Run directories, stage-specific preflight checks, runtime validation, and status reporting.</summary>
        <div class="br-branch">
          <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/runtime" aria-label="Browse src/benchrep/runtime on GitHub">Browse directory</a>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/runtime/run_context.py"><code>run_context.py</code></a> — Creates each run’s identity and the paths used by its writers.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/runtime/status"><code>status/</code></a> — Defines training, prediction, and evaluation outcomes and their summaries.</p>
        </div>
      </details>
      <details>
        <summary><code>records/</code> — Persists a run and its artifacts.</summary>
        <div class="br-branch">
          <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/records" aria-label="Browse src/benchrep/records on GitHub">Browse directory</a>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/records/configs.py"><code>configs.py</code></a> — Serializes authored and resolved configuration records.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/records/manifests.py"><code>manifests.py</code></a> — Writes the records downstream workflows use to locate and interpret a run’s outputs.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/records/prediction_exports.py"><code>prediction_exports.py</code></a> — Collects predictions by key and writes AnnData and reconstruction bundles.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/records/evaluation_exports.py"><code>evaluation_exports.py</code></a> — Writes evaluated data, metrics, and figure or TIFF outputs.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/records/runtime_environment.py"><code>runtime_environment.py</code></a> — Records library versions, hardware, and execution settings.</p>
        </div>
      </details>
      <details>
        <summary><code>evaluation/</code> — Analyses that consume an embedding matrix or reconstruction bundle.</summary>
        <div class="br-branch">
          <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/evaluation" aria-label="Browse src/benchrep/evaluation on GitHub">Browse directory</a>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/pipelines.py"><code>pipelines.py</code></a> — Builds and executes the ordered analysis steps for the available artifacts.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/evaluation/embeddings"><code>embeddings/</code></a> — Reductions, clustering, embedding metrics, predictability probes, and plots.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/evaluation/reconstructions"><code>reconstructions/</code></a> — Loading, error maps, metrics, and plotting for paired images.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/evaluation/metrics"><code>metrics/</code></a> — Contracts for registered metrics, computation, and normalization of returned values.</p>
        </div>
      </details>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/inspection"><code>inspection/</code></a> — Measures gradient norms for selected losses or parameter groups during training.</p>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/discovery"><code>discovery/</code></a> — Exposes schema and registry information for Python users and the wizard’s documentation panel.</p>
      <details>
        <summary><code>gui/</code> — Workflow configuration wizards.</summary>
        <div class="br-branch">
          <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/gui" aria-label="Browse src/benchrep/gui on GitHub">Browse directory</a>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/gui/rules"><code>rules/</code></a> — Stage-specific fields, compatible options, and review logic.</p>
          <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/gui/shared"><code>shared/</code></a> — Editor state, forms, validation, review, export, and the Python result.</p>
        </div>
      </details>
    </div>
  </details>
  <details>
    <summary><code>examples/</code> — Runnable scripts and their inputs. This directory is installed as benchrep.examples.</summary>
    <div class="br-branch">
      <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/examples" aria-label="Browse examples on GitHub">Browse directory</a>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/examples/usage"><code>usage/</code></a> — Workflow scripts combining the public API with example configurations.</p>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/examples/configs"><code>configs/</code></a> — YAML inputs for the shipped examples, organized by stage and task.</p>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/examples/artifact_generation"><code>artifact_generation/</code></a> — Recipes for preparing the standalone evaluation example’s inputs.</p>
    </div>
  </details>
  <details>
    <summary><code>tests/</code> — Test coverage organized by the behavior being exercised.</summary>
    <div class="br-branch">
      <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/tests" aria-label="Browse tests on GitHub">Browse directory</a>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/tests/unit"><code>unit/</code></a> — Focused tests for schemas and resolution rules, components, analyses, and runtime checks.</p>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/tests/integration"><code>integration/</code></a> — Checks behavior across stages, including checkpoint reuse, model equivalence, and config round trips.</p>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/tests/fixtures"><code>fixtures/</code></a> — Small datasets, models, configs, and registrations used across tests.</p>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/tests/gui"><code>gui/</code></a> — Tests of wizard editing, review, and stage-specific behavior.</p>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/tests/docs"><code>docs/</code></a> — Checks the metadata used to generate the Reference.</p>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/tests/packaging"><code>packaging/</code></a> — Checks example availability from an installed wheel.</p>
    </div>
  </details>
  <details>
    <summary><code>docs/</code> — Tutorials, Concepts &amp; design, and Reference pages.</summary>
    <div class="br-branch">
      <a class="br-browse" href="https://github.com/uhlmanngroup/BenchRep/tree/main/docs" aria-label="Browse docs on GitHub">Browse directory</a>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/docs/guides"><code>guides/</code></a> — Runnable tutorials and worked examples.</p>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/docs/concepts"><code>concepts/</code></a> — Design explanations and their connections to the implementation.</p>
      <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/docs/reference"><code>reference/</code></a> — Selects and organizes the documented public API and schema entries.</p>
    </div>
  </details>
  <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/docs_extensions"><code>docs_extensions/</code></a> — Extracts schema metadata and the component catalogue from Python source for the Reference.</p>
  <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/tree/main/docs_templates"><code>docs_templates/</code></a> — Templates controlling how API entries, signatures, and schema fields appear.</p>
  <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/mkdocs.yml"><code>mkdocs.yml</code></a> — Documentation navigation, Markdown extensions, and Reference rendering settings.</p>
  <p class="br-leaf"><a href="https://github.com/uhlmanngroup/BenchRep/blob/main/pyproject.toml"><code>pyproject.toml</code></a> — Package metadata, dependencies, optional extras, and package-data rules.</p>
</div>

## A training run across packages

Consider the Gaussian KL term in the [Optuna example’s training config](https://github.com/uhlmanngroup/BenchRep/blob/main/examples/configs/10_optuna_composite_vae/training.yaml). The configuration names the loss and connects its 2 arguments to values produced by a variational head. Following this one term shows how several packages share responsibility for the same setting.

| Follow the term | What happens |
| --- | --- |
| **Entrypoint → schema** | `train_composite()` composes the training config and hands it to the training resolver. The `regularization` section contains the term’s name, coefficient, and connections. |
| **Resolver → registry** | The loss resolver looks up `gaussian_kld` in the regularization registry. It resolves the alias to `gaussian_kl` and checks that the roles declared for the two outputs satisfy the registered loss contract. The resulting `LossSpec` identifies both outputs by name. |
| **Builder → model** | The builder constructs the Gaussian KL module inside a weighted `LossTerm`. The composite model receives both the live module and the instructions for supplying its tensors. |
| **Model → Lightning** | For each training batch, the variational head produces a mean and log-variance. The model retrieves them, calls the loss, and adds its weighted value to the scalar it returns to Lightning. |

The graph resolver determines which call produces each tensor. The loss resolver determines which of those tensors the loss receives. To trace a wrong connection, start with those two specifications. To investigate a wrong numerical value with the right connection, inspect the loss module and its reduction. See [Composite models](models/composite.md) and [Losses and objectives](models/losses.md).

## Common changes

| If you’re changing… | Start here | Check the behavior in… |
| --- | --- | --- |
| How a prediction setting inherits from training | The field in the [prediction schema](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/schemas/prediction_config_schema.py), then its use in the [prediction resolver](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/prediction_config_resolver.py) | [Prediction resolver tests](https://github.com/uhlmanngroup/BenchRep/blob/main/tests/unit/assembly/resolvers/test_prediction_config_resolver.py) and [resolved config round trips](https://github.com/uhlmanngroup/BenchRep/blob/main/tests/integration/test_resolved_config_roundtrip.py) |
| How a composite step connects to another | [Component contracts](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/architecture/composite_model_component_contracts.py), then the [composite resolver](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/composite_model_resolver.py) and [executor](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/architecture/models/composite.py) | [Resolver tests](https://github.com/uhlmanngroup/BenchRep/blob/main/tests/unit/assembly/resolvers/test_composite_model_resolver.py) for rejected graphs and [model equivalence tests](https://github.com/uhlmanngroup/BenchRep/blob/main/tests/integration/test_composite_and_canonical_model_equivalence.py) for numerical behavior |
| Which components users can select | [Built-in registrations](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/registries/builtins.py) for shipped components. For an external component, use the [public registration interface](extensibility/registration.md). | [Registry tests](https://github.com/uhlmanngroup/BenchRep/tree/main/tests/unit/assembly/registries) and [registry discovery](extensibility/discovery.md) |
| Which fields reach an exported artifact | The [prediction resolver](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/assembly/resolvers/prediction_config_resolver.py) selects the fields. The [exporter](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/records/prediction_exports.py) gathers and writes them. | [Prediction output validation](https://github.com/uhlmanngroup/BenchRep/blob/main/tests/unit/runtime/test_prediction_run_validation.py) and [end-to-end tests](https://github.com/uhlmanngroup/BenchRep/blob/main/tests/integration/test_end_to_end.py) |
| How an evaluation step fails or allows later steps to run | The step in [evaluation pipelines](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/evaluation/pipelines.py), then [evaluation status aggregation](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/runtime/status/evaluation.py) | [Pipeline](https://github.com/uhlmanngroup/BenchRep/blob/main/tests/unit/evaluation/test_pipelines.py) and [status tests](https://github.com/uhlmanngroup/BenchRep/blob/main/tests/unit/evaluation/test_status.py) |
| How the wizard edits a field | The corresponding [stage rule](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/gui/rules). Shared behaviors such as omission, collection editing, and saving belong in the [shared editor](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/gui/shared). | [Wizard tests](https://github.com/uhlmanngroup/BenchRep/tree/main/tests/gui) and [Wizard implementation](workflows/wizard-architecture.md) |
