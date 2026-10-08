# Engine acceptance

The working static inspection engine is imported from `feat/vv` commit
`8f341e93161428ae4c1b5ed9cd6cbdd79004f82b`. It now backs the public resolver,
reconstructor, and canonical report types. Python report imports remain light;
Torch is loaded only for inspection. The API and engine are separate dependency
extras, with CPU-only Torch wheels and a committed `uv.lock`.

## Verified models

On 2026-10-08, fresh CLI runs and committed network acceptance tests verified:

| Model | Pinned revision | Unique parameters | Header bytes | Weight payload bytes |
| --- | --- | ---: | ---: | ---: |
| GPT-2 | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 124,439,808 | 14,291 | 0 |
| SmolLM2-135M | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 134,515,008 | 30,536 | 0 |

GPT-2 matches the existing 160 checkpoint tensor records, 12 layers, 768 hidden
dimensions, and five nested graphs. SmolLM2 has 30 layers and 576 hidden dimensions.
Both match all native parameter identities to checkpoint metadata.

These are static reports, not runtime traces. GPT-2 retains an opaque tracing-call
warning. SmolLM2 has known shapes for approximately 94.8% of its dataflow edges.
Neither report's semantic coverage is a proof that every runtime path is known.
The web application must preserve these qualifications and show partial results.

## Reproduce

```powershell
uv sync --frozen --extra engine --extra api --extra dev
uv run --no-sync model-anatomy openai-community/gpt2 --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --output .local/verification/gpt2 --config-dir .local/verification/gpt2-config --preview-dir .local/verification/preview --inspect
uv run --no-sync model-anatomy HuggingFaceTB/SmolLM2-135M --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 --output .local/verification/smollm2 --config-dir .local/verification/smollm2-config --preview-dir .local/verification/preview --inspect
$env:RUN_HF_INTEGRATION = '1'
uv run --no-sync pytest -q backend/tests/test_engine.py backend/tests/test_engine_integration.py backend/tests/test_smoke.py
```

The deterministic tests cover malformed headers, exact range enforcement,
invalid configuration dimensions, remote-code rejection, tied parameters,
checkpoint mismatches, graph validation, unknown shapes, and report round trips.
Network tests are opt-in so the normal test suite does not depend on Hugging Face.

## Integration boundaries

`reconstruct` accepts a pinned revision and a progress callback. The service
worker can resolve a revision once, check its cache, and call the same pipeline
as the CLI. Supported stages are `resolving`, `reading_metadata`,
`building_structure`, `analyzing_components`, and `validating`.

Header inspection is bounded to 16 MiB per shard, 64 MiB overall, and 64 shards.
Native construction rejects models with more than 512 layers. These limits do
not replace the separate worker's process timeout and memory boundary.

The two-minute worker deadline applies to web jobs; direct CLI runs remain a
developer diagnostic tool. The new `--preview-dir` option prevents diagnostic
runs from overwriting the repository's recorded preview.
