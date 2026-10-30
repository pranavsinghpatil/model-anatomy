# Mistral-family static-report closure

Two pinned decoder-only reports exposed the same concrete gaps: a local alias for
the native causal-mask factory was recorded as an opaque call, and eight
hidden-state graph connections lacked a proven shape. The new source-hashed
Mistral contracts establish only the reviewed attention, MLP, norm and decoder
output ports. An exact callable, source digest and valid signature identify the
native causal or sliding-window mask factory without executing it. A shadowed
alias, altered source, invalid call or changed projection remains unresolved.

| Pinned repository | Revision | Static status | Graphs | Header tensor records | Unique native parameters |
| --- | --- | --- | ---: | ---: | ---: |
| `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | completed | 6 | 291 | 7,248,023,552 |
| `HuggingFaceH4/zephyr-7b-beta` | `892b3d7a7b1cf10c7a701c60881cd93df615734c` | completed | 6 | 291 | 7,241,732,096 |

Both fresh metadata-only runs had semantic, module-semantic, checkpoint-parameter
and graph-connection-shape coverage of 1; zero unknown nodes, unmapped tensors,
extraction warnings and weight-payload bytes read. The complete label applies to
the canonical static report. It does not establish mask values, every input or
cache path, numerical inference behavior, or runtime trace. Exact source/callable
evidence appears in static-flow diagnostics, and the unmodeled mask dataflow is
shown in report limitations. Saved reports stay immutable;
submitting a model again is required to use engine `2026.10.24.6`.

The regression suite includes altered-source and shadowed-alias negatives, a
changed-projection negative, a tiny native Mistral fixture, and opt-in real pinned
acceptance for both repositories. The generated IRs and request audits are kept
under ignored `.local/verification/`; they are inspection evidence, never an app
fallback. The other partial cohorts in the 250-model catalog still need their own
measured rechecks and source-backed fixes. DeepSeek's publisher-reference graph
has a separate partial status because native parameter ownership and tensor-flow
shapes are not established by its header/name audits.

Verification: 1,034 non-integration backend tests passed, two platform tests
skipped, and 22 opt-in integrations were deselected. Both new pinned Hugging Face
integrations passed separately. Ruff lint and scoped diff checks passed. The
existing Starlette/httpx deprecation warning remains. The two large pre-existing
source files still fail a repository-wide Ruff formatting check; formatting the
whole files would add unrelated changes, while the new test files pass it.
