# Running Model Anatomy

Run all commands from the project root, where `pyproject.toml` is located. Python 3.11 or later and `uv` are required.

## Set up the environment

For a new checkout, install the project and its dependencies:

```bash
uv sync
```

If the environment is already installed and up to date, skip this step. The `--no-sync` option below uses the existing environment without updating dependencies. You can omit it to let `uv run` synchronize the environment automatically.

## Inspect GPT-2

```bash
uv run --no-sync model-anatomy openai-community/gpt2 --inspect
```

The CLI reads Hugging Face configuration and Safetensors metadata, constructs the model on the meta device, and generates a static architecture representation. It does not download checkpoint weight payloads or execute a forward pass. Network access to Hugging Face is required.

`--inspect` prints an additional coverage and diagnostics report in the terminal. It is optional; output files are generated with or without it.

## Generated files

For `openai-community/gpt2`, a successful run creates:

| Path | Contents |
| --- | --- |
| `config/gpt2/config.json` | Snapshot of the model configuration from Hugging Face. |
| `output/gpt2/model.json` | Hierarchical architecture IR, including tensor metadata and provenance. |
| `output/gpt2/diagnostics.json` | Coverage, module information, and static analysis diagnostics. |
| `output/gpt2/artifact_audit.json` | Resolved artifact metadata and Safetensors header inspection details. |
| `output/gpt2/overview.mmd` | Top-level Mermaid graph. |
| `output/gpt2/*.mmd` | Additional Mermaid graphs for nested components. |
| `preview/GPT2_PREVIEW.md` | English Markdown preview containing the Mermaid graphs. |

All paths are relative to the current working directory. Repeating the command updates the files for that model. The terminal also prints the model, revision, unique parameter count, header bytes read, and generated file paths.

## Inspect another model

Use a Hugging Face repository ID, rather than the configuration's `model_type`:

```bash
uv run --no-sync model-anatomy EleutherAI/gpt-neo-125m --inspect
```

By default, repository IDs are lowercased and `/` becomes `--` for the config and output directory names. This example writes to `config/eleutherai--gpt-neo-125m/`, `output/eleutherai--gpt-neo-125m/`, and `preview/ELEUTHERAI__GPT_NEO_125M_PREVIEW.md`. GPT-2 uses the shorter `gpt2` directory name.

Models must provide Safetensors artifacts and be supported by the installed native Transformers implementation. Models without Safetensors are outside the supported scope. Unsupported or custom model implementations can fail explicitly; static analysis may leave some shapes or components unknown.

## CLI arguments

```bash
uv run --no-sync model-anatomy --help
```

| Argument | Purpose |
| --- | --- |
| `model` | Required Hugging Face repository ID or browser URL to a `model.safetensors` file. |
| `--inspect` | Print the additional semantic coverage and diagnostics report. |
| `--output DIR` | Write JSON and Mermaid files to this exact directory. |
| `--config-dir DIR` | Write the configuration snapshot to this exact directory. |
| `-h`, `--help` | Show CLI help. |

For example, choose separate output directories:

```bash
uv run --no-sync model-anatomy openai-community/gpt2 --inspect --output output/gpt2-custom --config-dir config/gpt2-custom
```

The Markdown preview still goes to `preview/`. `--no-sync` is a `uv run` option, so it belongs before `model-anatomy`.
