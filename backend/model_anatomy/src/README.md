# Running Model Anatomy

Run these commands from the project root, where `pyproject.toml` is located. Python 3.11 or later and `uv` are required.

## Install the dependencies you need

| Component                                    | Command                                       |
| -------------------------------------------- | --------------------------------------------- |
| CLI                                          | `uv sync --frozen --extra engine`             |
| API                                          | `uv sync --frozen --extra api`                |
| Worker, or API and worker in one environment | `uv sync --frozen --extra api --extra engine` |

Skip installation when the required dependencies are already available. `uv run --no-sync` uses the existing environment. A plain `uv sync` does not select extras; keep the extras you need when synchronizing again. API startup does not require Torch. The optional `dev` extra is for development tools.

## Inspect a model

```bash
uv run --no-sync model-anatomy openai-community/gpt2 --inspect
uv run --no-sync model-anatomy EleutherAI/gpt-neo-125m --inspect
```

Use a Hugging Face repository ID, rather than the configuration's `model_type`. Network access to Hugging Face is required. The CLI pins the repository revision, reads configuration and bounded Safetensors headers, constructs native modules on the meta device, and writes a static report. It never downloads checkpoint weight payloads or executes forward.

All models use the same native module pipeline and filename rules. The graph
follows actual registered modules down to basic modules such as Linear,
Embedding, LayerNorm, Activation, Conv and Dropout. Basic modules are leaves;
individual tensor expressions are outside the graph. Semantic role labels are
annotations, not replacements for observed native modules. Layer containers
retain every actual instance; registration alone does not establish execution
order, and matching layer names do not prove that layers can be merged.

The architecture support target is anonymously readable, publisher-provided
Safetensors checkpoints with a compatible installed native implementation.
Models without Safetensors and models requiring private/gated access are outside
that target. Missing metadata or an unavailable architecture produces an explicit
failure in the default strict CLI flow. There is no full-weight download,
checkpoint conversion, forward pass or execution of publisher remote code as a
fallback. Compatibility is model/revision-specific, not a universal support claim.

`--artifacts` explicitly permits a limited inspection result when whole-model
reconstruction is unavailable. Depending on available evidence, this may be a
graph-free artifact inventory or verified scoped components. Such a result does
not establish a complete whole-model architecture. The web worker enables this
inspection mode for its separate Artifacts inspected flow.

The CLI uses its arguments for output paths. API/worker settings in `backend/.env`
do not redirect CLI output or start the web services.

## Arguments

```bash
uv run --no-sync model-anatomy --help
```

| Argument            | Purpose                                                                                                                      | Default                                   |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `model`             | Required repository ID or supported Hugging Face Safetensors browser URL.                                                    | Required                                  |
| `--inspect`         | Print the analysis report, including architecture completion checks, checkpoint coverage, warnings and topology diagnostics. | Disabled                                  |
| `--artifacts`       | Permit artifact inventory or verified scoped components when whole-model reconstruction is unavailable.                      | Disabled                                  |
| `--revision SHA`    | Pin an immutable 40-character commit SHA.                                                                                    | Resolve the requested repository revision |
| `--output DIR`      | Exact destination for JSON and Mermaid files.                                                                                | `output/<slug>`                           |
| `--config-dir DIR`  | Exact destination for the configuration snapshot.                                                                            | `config/<slug>`                           |
| `--preview-dir DIR` | Directory for the English Markdown preview.                                                                                  | `preview`                                 |
| `-h`, `--help`      | Print usage.                                                                                                                 | —                                         |

`--no-sync` belongs to `uv run`, before `model-anatomy`. For example:

```bash
uv run --no-sync model-anatomy openai-community/gpt2 --inspect --output output/my-model --config-dir config/my-model --preview-dir preview
```

## Generated files

The slug is the repository ID lowercased with `/` replaced by `--`. For `openai-community/gpt2`, a successful run writes:

| Path                                                   | Contents                                                                                       |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `config/openai-community--gpt2/config.json`            | Source configuration snapshot.                                                                 |
| `output/openai-community--gpt2/model.json`             | Hierarchical IR, tensor metadata and provenance.                                               |
| `output/openai-community--gpt2/diagnostics.json`       | Analysis report, registered module tree, semantic annotations and module topology diagnostics. |
| `output/openai-community--gpt2/artifact_audit.json`    | Pinned artifact identity and metadata requests.                                                |
| `output/openai-community--gpt2/<root_graph_id>.mmd`    | Root Mermaid graph; use `root_graph.id` from `model.json` for its filename.                    |
| `output/openai-community--gpt2/*.mmd`                  | Nested Mermaid graphs.                                                                         |
| `preview/OPENAI_COMMUNITY__GPT2_<identity>_PREVIEW.md` | English hierarchy preview.                                                                     |

`<identity>` is the first 16 hexadecimal characters of SHA-256 over `model_id@revision`. The readable filename prefix is normalized from the model ID. Preview identity is independent of a custom output directory's basename; different models or revisions receive distinct names. The CLI prints the exact paths, resolved revision, parameter count and header bytes read. Paths are relative to the current working directory unless specified as absolute paths.

Files under the former `output/gpt2/`, `config/gpt2/` and old preview names are not automatically migrated or deleted.

## Reading completion and diagnostics

A successful CLI exit means a report was generated. Check
`metadata.analysis_report.completion` in `model.json`, or
`analysis_report.completion` in `diagnostics.json`; `--inspect` prints the same
analysis report. The native architecture contract uses these checks:

| Check                 | What it establishes                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------- |
| `module_inventory`    | Registered modules appear in the hierarchy or are covered by an explicit basic-module leaf. |
| `parameter_mapping`   | Native parameters have checkpoint bindings and no unexplained checkpoint tensors remain.    |
| `native_architecture` | Native architecture selection has a supported metadata/checkpoint compatibility basis.      |
| `module_topology`     | Bounded source analysis has no blocking gaps in module-call dependencies.                   |

`completion.status` is `complete` when all four checks pass, otherwise `partial`.
The web service separately validates this contract, checkpoint coverage, unknown
components and blocking diagnostics before assigning its API status `completed`
or `partial`. Neither status certifies inference or numerical equivalence to
publisher custom code.

Pure math details such as `log` and `exp`, intermediate activation shapes,
runtime branch selection and numerical equality of tied tensor payloads are
outside architecture completion. Missing activation shapes or unclassified
semantic roles alone do not make the module architecture partial. Checkpoint
shapes and storage dtypes remain direct Safetensors-header evidence.

Unresolved module selection, callbacks or control flow that can hide module
calls or dependencies remain blocking. Inspect `completion.topology_boundaries`,
`completion.missing_module_paths`, `completion.missing_parameter_paths`,
`completion.unmapped_checkpoint_tensors` and the report's `diagnostics` for the
recorded reason. Non-blocking details remain available in diagnostics without
being expanded into expression nodes. Warnings and limitations remain visible
on complete reports as well.

Graph edges describe source-backed module dependencies with intermediate
expressions omitted. Conditional edges retain their source conditions. They are
not runtime tensor traces, and registered-module presence does not prove execution
for every possible input. Optional modules remain represented.

CLI artifacts are separate from web storage: the API/worker defaults store jobs
and report objects under `.local/model-anatomy/`, not in these CLI directories.
Use the web Export menu for reports created through the application.

## Replacing an existing report

All serialization and Mermaid rendering finish before publication. New files are staged beside their destinations, then each file is replaced atomically. Caught publication failures roll back previously replaced files, preserving the previous report set. Obsolete managed Mermaid files are removed as part of the same rollback-protected operation. An existing valid output directory belonging to another model is rejected.

The config, output and preview directories do not form one filesystem transaction. This protects against ordinary caught write/rename failures, not process crashes, power loss or simultaneous writers. Use one writer per destination. If rollback itself fails, the error identifies retained recovery backups.
