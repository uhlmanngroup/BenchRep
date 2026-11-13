---
hide:
  - navigation
---

<div class="br-home-hero">
  <h1 class="br-home-title">
    <img class="br-image-light" src="assets/logo-light.svg" alt="">
    <img class="br-image-dark" src="assets/logo-dark.svg" alt="">
    <span class="br-wordmark">BenchRep</span>
  </h1>
  <p class="br-home-lead">A configuration-driven, Lightning-based, extensible Python <strong class="highlight">workbench for representation learning</strong>.</p>
</div>

<div class="br-intents">
  <div>
    <strong class="br-intent-title"><svg class="br-icon " aria-hidden="true"><use href="assets/interface-icons.svg#sliders"></use></svg> Experiment</strong>
    <p>Explore architectures and learning objectives using configurations that simplify repeated runs with targeted changes.</p>
  </div>
  <div>
    <strong class="br-intent-title"><svg class="br-icon " aria-hidden="true"><use href="assets/interface-icons.svg#chart"></use></svg> Benchmark</strong>
    <p>Compare models reproducibly through linked training, representation export, and standardized evaluation workflows.</p>
  </div>
  <div>
    <strong class="br-intent-title"><svg class="br-icon " aria-hidden="true"><use href="assets/interface-icons.svg#book"></use></svg> Teach</strong>
    <p>Explore representation learning with minimal coding overhead.</p>
  </div>
</div>

## Start here

<nav class="br-start" aria-label="Getting started">
  <a class="br-start-link" href="guides/installation.md">
    <svg class="br-icon " aria-hidden="true"><use href="assets/interface-icons.svg#download"></use></svg>
    <strong>Install BenchRep</strong>
    <span>Choose the optional dependencies you need.</span>
  </a>
  <a class="br-start-link br-start-link--primary" href="guides/first-workflow-chain.md">
    <svg class="br-icon " aria-hidden="true"><use href="assets/interface-icons.svg#play"></use></svg>
    <strong>Run your first workflow chain</strong>
    <span>Train a VAE, export embeddings and reconstructions, and evaluate them.</span>
  </a>
  <a class="br-start-link" href="guides/standalone-evaluation.md">
    <svg class="br-icon " aria-hidden="true"><use href="assets/interface-icons.svg#chart"></use></svg>
    <strong>Evaluate external outputs</strong>
    <span>Start with existing embeddings or reconstructions and go straight to evaluation.</span>
  </a>
</nav>

## Workflows and provenance

Training fits a model and saves checkpoints. Prediction uses a trained model to export embeddings, reconstructions, or both. Evaluation measures properties such as clustering quality, predictability, and reconstruction error.

<figure>
  <a class="br-image-light" href="assets/workflow-chain-light.svg">
    <img src="assets/workflow-chain-light.svg" alt="BenchRep workflow chain">
  </a>
  <a class="br-image-dark" href="assets/workflow-chain-dark.svg">
    <img src="assets/workflow-chain-dark.svg" alt="BenchRep workflow chain">
  </a>
  <figcaption>Dash-dot arrows mark optional connections. Wands mark configuration wizards.</figcaption>
</figure>

Each workflow takes a config, runs independently, and writes a manifest. To chain the three, pass an upstream manifest path to a downstream workflow. This lets you trace an evaluation result back to its training and prediction settings.

Each run also saves its resolved configuration, runtime environment, logs, and status report, showing which settings took effect, which steps completed, and where warnings or failures occurred. In Python, each workflow returns a result object. For example, you can read a clustering score directly from the evaluation result using the same keys shown in the exported `metrics.json`.

## Configuring an experiment

You can write a workflow config in <strong class="highlight">YAML</strong>, build it in <strong class="highlight">Python</strong>, or use the <strong class="highlight">browser wizard</strong> to explore the available fields and export a YAML. Configs refer to registered components by name. You can inspect registries and, where supported, register your own classes or factories. Once registered, custom components such as datasets and losses can be selected by name just like built-in ones.

For training, a [canonical autoencoder or VAE](understanding/models/canonical.md) gives you a predefined architecture with your choice of encoder and decoder. With a [composite model](understanding/models/composite.md), you specify which components to use, how they connect, and which values feed each loss. This lets you build multi-input and multitask models directly in config.


<strong class="highlight">Want to compare two encoders?</strong>
Change just the encoder section in your canonical training YAML and rerun the same workflow chain.

<strong class="highlight">Want to tune architectures or hyperparameters?</strong> Use Optuna to choose what to try. For each trial, use `update_config()` to change individual fields or pass whole config sections through a workflow’s overrides object. To prune during training, pass Optuna’s Lightning callback via `runtime_callbacks`.

<strong class="highlight">Want more control over the model or training objective?</strong>
Register your own components, wire them into a composite model, and define a custom loss with access to the full model context. You can also supply a model or datamodule instance or class through overrides where supported.


## Explore the docs

<nav class="br-doc-routes" aria-label="Documentation sections">
  <a class="br-doc-route" href="guides/index.md">
    <svg class="br-icon " aria-hidden="true"><use href="assets/interface-icons.svg#book"></use></svg>
    <span><strong>Tutorials</strong><small>Configuration routes, external components, composite models, and hyperparameter tuning through concrete examples.</small></span>
    <svg class="br-icon br-route-arrow" aria-hidden="true"><use href="assets/interface-icons.svg#arrow"></use></svg>
  </a>
  <a class="br-doc-route" href="understanding/index.md">
    <svg class="br-icon " aria-hidden="true"><use href="assets/interface-icons.svg#layers"></use></svg>
    <span><strong>Understanding BenchRep</strong><small>Configuration rules, model and data interfaces, workflow behavior, provenance, and execution internals.</small></span>
    <svg class="br-icon br-route-arrow" aria-hidden="true"><use href="assets/interface-icons.svg#arrow"></use></svg>
  </a>
  <a class="br-doc-route" href="reference/index.md">
    <svg class="br-icon " aria-hidden="true"><use href="assets/interface-icons.svg#code"></use></svg>
    <span><strong>Reference</strong><small>Lookup pages for public APIs, configuration fields, and registered components.</small></span>
    <svg class="br-icon br-route-arrow" aria-hidden="true"><use href="assets/interface-icons.svg#arrow"></use></svg>
  </a>
</nav>

<p class="br-development-note">BenchRep is under active development. Its public interfaces and configuration schemas may change before version 1.0.</p>
