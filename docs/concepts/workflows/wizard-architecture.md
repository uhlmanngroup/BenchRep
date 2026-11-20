# Wizard implementation

The wizard is a configuration editor connected to BenchRep's schemas, registries, and resolvers. Its central design choice is to keep an **unfinished editor state** separate from a **validated workflow config**. That lets it preserve an incomplete value, an unexpected field, or an omitted key while showing what needs attention.

The [configuration wizard guide](wizards.md) explains its controls and returned values. This page follows how those controls become a reviewed configuration that the calling Python script can use.

## One editor engine, workflow-specific rules

Training, prediction, and evaluation each provide their own sections and contextual rules to a shared Dash editor:

| Layer | Responsibility |
| --- | --- |
| Schema-derived editor specification | Describes field structure: objects, lists, mappings, choices, and alternative types |
| Shared editor state and forms | Retain entered values, render controls, collect the draft, and display errors |
| Workflow rules | Select appropriate sections, suggest compatible choices, adapt editors, and connect review to the matching resolver |
| Review session | Store checked results, control saving, and return the configuration to Python |

The shared editor derives structural information from Pydantic annotations and field metadata. Ordinary scalar fields become controls; nested models become grouped fields; lists and mappings become editable collections. More open-ended values, such as arbitrary parameter dictionaries, can use JSON editing.

Workflow rules add knowledge that a type annotation alone cannot provide. For example, the selected family determines which architecture sections are appropriate; registry contracts guide component and loss choices; a prediction manifest supplies information about the model being loaded.

These responsibilities are separated in [`gui/shared`](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/gui/shared) and [`gui/rules`](https://github.com/uhlmanngroup/BenchRep/tree/main/src/benchrep/gui/rules). The stage app factories connect them rather than maintaining separate form engines.

## Preserve editing state before validation

Each field has an editor node containing its value, a stable identifier, and its mode: `omit`, `null`, `empty`, or `value`. Nested nodes retain list entries, mapping keys, and alternative-type branches. Stable identifiers let the interface track a field even when a list is reordered or a mapping key changes.

This state is stored separately from the currently displayed controls. Changing sections therefore does not require reconstructing the draft from whatever happens to be visible in the browser.

When the editor collects a node:

- **Omit** leaves its key out of the authored mapping.
- **Null** includes the key with `None`.
- **Empty** includes an appropriate empty value.
- **Value** contributes the entered value, including incomplete text where necessary to show an error.

Collection preserves the authored choices; it does not replace the draft with Pydantic's validated or coerced output. For example, a malformed JSON field can remain visible with its error rather than disappearing. Unexpected fields are also retained so the user can identify and remove them.

The implementation is in [`state.py`](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/gui/shared/state.py). `EditorSpec` describes a field's structure; editor nodes hold its current state; `collect_node()` converts that state back into draft data and collection errors.

## Local checks and documentation

Each section is checked against its original schema type, then against applicable local and contextual rules. Validation errors retain their field paths so the form can attach messages to the relevant controls. The sidebar summarizes those section results; it does not stand in for validation of the whole workflow.

The Documentation panel uses schema and registry descriptions to show field behavior, aliases, constructors, and contracts. It follows the active section or selected field, while workflow-specific presentation rules supply compatible suggestions. The same registry and schema information underlies [discovery in Python](../extensibility/discovery.md).

Python-supplied `config_components` are incorporated into the draft but shown as read-only sections. Runtime overrides can remove the sections whose construction they bypass. These are launch-context decisions, distinct from an invalid or incompatible entry that the user must edit or remove.

## Review creates a checked snapshot

**Check configuration** passes the complete authored mapping into a server-owned `ReviewSession`. The stage adapter parses it with the appropriate override context, resolves a copy, checks known dependencies and export destinations, and optionally renders a composite specification graph.

This reuses the workflow's configuration logic. It does not build the model or run a batch through it. The graph preview comes from the resolved composite specification, not from tracing a live neural network.

The session retains:

| Retained value | Purpose |
| --- | --- |
| Authored mapping | Preserve the user's choices for YAML export |
| Parsed config | Return a valid config object for subsequent execution |
| Resolved specification | Show derived settings and prepare the resolved YAML preview |
| Check report | Distinguish failures, warnings, and checks that were skipped |
| Draft/export fingerprint | Establish which exact draft and save settings were checked |

Saving requires a successful review of the current draft and export settings. The server checks the fingerprint again rather than trusting the browser's enabled Save button. Prediction review additionally checks the source manifest's content identity, so a changed manifest requires another review.

This protects the relationship between **what was reviewed** and **what is saved**. It does not guarantee that external files or the execution environment will remain unchanged after the wizard finishes; the workflow resolves its inputs again when called.

## Save and return to the same process

The review session writes authored and resolved YAMLs without replacing existing files. Each file is prepared in a temporary file and published only if the destination is still unused. If publishing the resolved YAML fails after the authored YAML was saved, the authored file is removed so a failed save does not leave that pair half-finished.

The returned [`ConfiguredWorkflow`][benchrep.gui.shared.results.ConfiguredWorkflow] contains the parsed config and the remaining execution overrides. Section overrides are removed from that returned bundle because they have already been incorporated; runtime objects and manifest overrides remain available. The caller's original overrides object is not mutated.

[`configure_workflow()`][benchrep.configure_workflow] runs the local web server in a worker thread while the calling thread waits on a completion event. Saving or cancelling signals that event; the launcher shuts down the server and returns the result or `None`. Closing the browser tab alone does not signal completion.

The model and datamodule overrides remain Python objects in that process throughout the session. They are not serialized through the browser or recreated from the saved YAML. This is what allows a script to open a wizard and then continue directly into the workflow.

Review and save behavior lives in [`review.py`](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/gui/shared/review.py); server startup and completion handling live in [`gui/__init__.py`](https://github.com/uhlmanngroup/BenchRep/blob/main/src/benchrep/gui/__init__.py). For the wider configuration-to-execution path, see [Workflow implementation](workflow-anatomy.md).
