# Running Model Anatomy

Run these commands from the project root, where `pyproject.toml` is located. Python 3.11 or later and `uv` are required.

## Install the dependencies you need

| Component | Command |
| --- | --- |
| CLI | `uv sync --frozen --extra engine` |
| API | `uv sync --frozen --extra api` |
| Worker, or API and worker in one environment | `uv sync --frozen --extra api --extra engine` |

Skip installation when the required dependencies are already available. `uv run --no-sync` uses the existing environment. A plain `uv sync` does not select extras; keep the extras you need when synchronizing again. API startup does not require Torch. The optional `dev` extra is for development tools.

## Inspect a model

```bash
uv run --no-sync model-anatomy openai-community/gpt2 --inspect
uv run --no-sync model-anatomy EleutherAI/gpt-neo-125m --inspect
```

Use a Hugging Face repository ID, rather than the configuration's `model_type`. Network access to Hugging Face is required. The CLI pins the repository revision, reads configuration and bounded Safetensors headers, constructs native modules on the meta device, and writes a static report. It never downloads checkpoint weight payloads or executes forward.

All models use the same pipeline, classification rules and filename rules. There is no GPT-2 adapter or shortened GPT-2 output directory. Models require a native Transformers architecture supported by the installed library and suitable Safetensors artifacts. Missing Safetensors and unsupported/custom architecture implementations fail explicitly. The CLI does not use the web worker's graph-free artifact-inspection fallback.

## Arguments

```bash
uv run --no-sync model-anatomy --help
```

| Argument | Purpose | Default |
| --- | --- | --- |
| `model` | Required repository ID or supported Hugging Face Safetensors browser URL. | Required |
| `--inspect` | Print the analysis report, including coverage, warnings and source completeness checks. | Disabled |
| `--revision SHA` | Pin an immutable 40-character commit SHA. | Resolve the requested repository revision |
| `--output DIR` | Exact destination for JSON and Mermaid files. | `output/<slug>` |
| `--config-dir DIR` | Exact destination for the configuration snapshot. | `config/<slug>` |
| `--preview-dir DIR` | Directory for the English Markdown preview. | `preview` |
| `-h`, `--help` | Print usage. | — |

`--no-sync` belongs to `uv run`, before `model-anatomy`. For example:

```bash
uv run --no-sync model-anatomy openai-community/gpt2 --inspect --output output/my-model --config-dir config/my-model --preview-dir preview
```

## Generated files

The slug is the repository ID lowercased with `/` replaced by `--`. For `openai-community/gpt2`, a successful run writes:

| Path | Contents |
| --- | --- |
| `config/openai-community--gpt2/config.json` | Source configuration snapshot. |
| `output/openai-community--gpt2/model.json` | Hierarchical IR, tensor metadata and provenance. |
| `output/openai-community--gpt2/diagnostics.json` | Classification, source flow and completeness diagnostics. |
| `output/openai-community--gpt2/artifact_audit.json` | Pinned artifact identity and metadata requests. |
| `output/openai-community--gpt2/overview.mmd` | Root Mermaid graph. |
| `output/openai-community--gpt2/*.mmd` | Nested Mermaid graphs. |
| `preview/OPENAI_COMMUNITY__GPT2_<identity>_PREVIEW.md` | English hierarchy preview. |

`<identity>` is the first 16 hexadecimal characters of SHA-256 over `model_id@revision`. The readable filename prefix is normalized from the model ID. Preview identity is independent of a custom output directory's basename; different models or revisions receive distinct names. The CLI prints the exact paths, resolved revision, parameter count and header bytes read. Paths are relative to the current working directory unless specified as absolute paths.

Files under the former `output/gpt2/`, `config/gpt2/` and old preview names are not automatically migrated or deleted.

## Reading diagnostics

A successful CLI exit means a report was generated; it does not prove every computation was reconstructed. Review `analysis_report.warnings` and `analysis_report.source_completeness` in `model.json` or `diagnostics.json`, or print them with `--inspect`.

Source completeness is separate from classification and tensor-shape coverage. It records module call sites found in inspected source, static invocations, unexplained operations, profile-excluded calls and structural nodes without invocation evidence. Its scope is the inspected source and representative repeated instances, not runtime coverage of every possible input or path. An empty denominator is unknown rather than 100 percent coverage.

Unsupported source statements and expressions retain explicit diagnostics and dependency boundaries. Unsupported exception flow covers the whole `try`, including handlers, `else` and `finally`; these sections are not assumed inactive. Helper arguments bind by AST parameter name, including keyword-only arguments and literal defaults. Unresolved argument expansion remains unsupported. Profile assumptions and branch conditions retain source evidence.

Static shapes are not operation-validity proofs. Repetition equivalence and attention-pattern configuration guards still have known limitations: computation-affecting attributes may be absent from equivalence signatures, and existing attention refinements are not fully proven for every configuration. These are separate pending fixes; high coverage must not be read as universal model support or verified execution.

## Replacing an existing report

All serialization and Mermaid rendering finish before publication. New files are staged beside their destinations, then each file is replaced atomically. Caught publication failures roll back previously replaced files, preserving the previous report set. Obsolete managed Mermaid files are removed as part of the same rollback-protected operation. An existing valid output directory belonging to another model is rejected.

The config, output and preview directories do not form one filesystem transaction. This protects against ordinary caught write/rename failures, not process crashes, power loss or simultaneous writers. Use one writer per destination. If rollback itself fails, the error identifies retained recovery backups.
